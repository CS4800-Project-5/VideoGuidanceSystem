#!/usr/bin/env bash
# One-time setup for zed.launch.py (the real ZED camera on a laptop). Needs internet; asks for
# your password. Safe to re-run: it skips what is already done.
#
#   bash setup_zed.sh
#
# Afterwards `zed-demo` (in a new terminal) starts the camera + YOLO tracking, and needs no
# internet. Plug the ZED in (and connect it to the VM) before running this, so its factory
# calibration gets downloaded now too.
set -eo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV="$HOME/yolo_ros2_env"
PROJ1="$HOME/Proj_1"
PROJ1_URL="https://github.com/ArnavSingh121/Proj_1.git"

step() { echo; echo "=== $* ==="; }

step "ROS 2 Humble"
if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "ROS 2 Humble is not installed. Install ros-humble-desktop first (see README.md, Setup)."
    exit 1
fi
echo "ok"

step "System packages"
need=()
for p in python3-venv ros-humble-cv-bridge ros-humble-rosbag2; do
    dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if [ ${#need[@]} -gt 0 ]; then
    sudo apt-get update
    sudo apt-get install -y "${need[@]}"
else
    echo "ok"
fi

step "Keep the ZED from disconnecting (USB autosuspend off for Stereolabs devices)"
RULE=/etc/udev/rules.d/99-zed-no-autosuspend.rules
if [ ! -f "$RULE" ]; then
    echo 'ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="2b03", ATTR{power/control}="on"' \
        | sudo tee "$RULE" >/dev/null
    sudo udevadm control --reload
fi
echo "ok"

step "ZED USB publisher (from $PROJ1_URL, not part of this repo)"
if [ -f "$PROJ1/zed_usb/zed_usb_publisher.py" ]; then
    echo "found in $PROJ1/zed_usb"
elif [ -d "$PROJ1/.git" ]; then
    git -C "$PROJ1" pull --ff-only
elif [ -e "$PROJ1" ]; then
    echo "$PROJ1 exists but has no zed_usb folder. Move it away, or run zed-demo with"
    echo "zed_usb_dir:=/path/to/zed_usb, then run this script again."
    exit 1
else
    git clone "$PROJ1_URL" "$PROJ1"
fi

step "Python environment for YOLO ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
    # Same environment as README.md, Setup step 2.
    python3 -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install ultralytics opencv-python "numpy<2"
fi
# lap: needed by ByteTrack. Ultralytics would otherwise try to download it the first time the
# node runs, which fails without internet (e.g. in class).
"$VENV/bin/pip" install -q "lap>=0.5.12" "numpy<2"
source /opt/ros/humble/setup.bash
"$VENV/bin/python" -c "import rclpy, cv_bridge, ultralytics, lap, cv2; print('ok')"

step "YOLO models (saved in the repo folder, so no download is needed later)"
(cd "$HERE" && "$VENV/bin/python" -c "
from ultralytics import YOLO
import numpy as np
# yolov8n.pt: simulation default; yolo11s.pt: zed-demo default.
for weights in ('yolov8n.pt', 'yolo11s.pt'):
    for tracker in ('bytetrack_occlusion.yaml', 'botsort_zed.yaml'):
        YOLO(weights).track(np.zeros((376, 672, 3), np.uint8), persist=True,
                            tracker=tracker, verbose=False)
print('ok: yolov8n.pt, yolo11s.pt and the trackers work')
")

step "ZED factory calibration"
if PYTHONNOUSERSITE=1 python3 - "$PROJ1/zed_usb" <<'EOF'
import os, sys
sys.path.insert(0, sys.argv[1])
import zed_usb_publisher as z
_, serial = z.find_zed()
if not serial:
    sys.exit(1)
d = os.path.expanduser('~/.local/share/zed/settings')
os.makedirs(d, exist_ok=True)
p = os.path.join(d, f'SN{serial}.conf')
if not os.path.exists(p):
    z.download_calibration(serial, p)
print(f'ok: calibration for ZED SN {serial} in {p}')
EOF
then :
else
    echo "No ZED found. Plug it in (and connect it to the VM), then run this script again,"
    echo "so the calibration is saved now and no internet is needed later."
fi

step "zed-demo command"
LINE="alias zed-demo='source /opt/ros/humble/setup.bash && ros2 launch $HERE/zed.launch.py'"
touch ~/.bashrc
sed -i '/^alias zed-demo=/d' ~/.bashrc   # replaces any older zed-demo
echo "$LINE" >> ~/.bashrc
echo "Open a new terminal and type:  zed-demo"
echo "Options go after it, e.g.:     zed-demo flip:=true stale_timeout:=5.0"

step "sim-demo command (Gazebo simulation with the same tracking settings)"
# sim.launch.py runs ~/yolo_camera_node.py: link it to this repo's node (README, Setup step 4).
if [ -L "$HOME/yolo_camera_node.py" ] || [ ! -e "$HOME/yolo_camera_node.py" ]; then
    ln -sf "$HERE/yolo_camera_node.py" "$HOME/yolo_camera_node.py"
else
    echo "Note: ~/yolo_camera_node.py is a copy, not a link, so sim-demo runs that copy."
    echo "To run this repo's node instead: ln -sf $HERE/yolo_camera_node.py ~/yolo_camera_node.py"
fi
SIM_ARGS="tracker:=botsort"
# Gazebo's 3D rendering crashes on a VM's virtual graphics card (README, "Running in a VMware").
if [ "$(systemd-detect-virt 2>/dev/null || true)" != "none" ]; then
    SIM_ARGS="$SIM_ARGS software_render:=true"
fi
LINE="alias sim-demo='source /opt/ros/humble/setup.bash && ros2 launch $HERE/sim.launch.py $SIM_ARGS'"
sed -i '/^alias sim-demo=/d' ~/.bashrc
echo "$LINE" >> ~/.bashrc
echo "Open a new terminal and type:  sim-demo"
if [ ! -d "$HOME/PX4-Autopilot" ]; then
    echo "Note: ~/PX4-Autopilot not found; sim-demo needs PX4 + Gazebo (README, Setup)."
fi

echo
echo "Setup done."
