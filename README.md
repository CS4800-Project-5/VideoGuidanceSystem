# VideoGuidanceSystem: UAV Perception System

**CS 4800 Software Engineering, Fall 2026, Project Team 5**

A ROS 2 perception component for a simulated PX4 drone. It detects objects of interest in the drone's camera feed and keeps a consistent ID on each object as it moves, leaves the view, or is briefly hidden.

- **Repository:** https://github.com/CS4800-Project-5/VideoGuidanceSystem
- **Project board:** _TODO: add the GitHub Projects / Jira board link_
- **Assignment specs:** [Project #5: UAV Perception System](docs/CS_4800_proj_5.pdf) · [Assignment #3: Agile Development](docs/CS_4800_agile_development.pdf)

## Product goal

Develop a ROS 2 perception component that detects objects of interest in a simulated UAV camera feed and maintains their identities across successive observations.

## Intended users

- **UAV operators**, who need to tell individual objects apart and follow them over time.
- **Other teams on the autonomous drone project** (such as path planning), who will use the published detections and tracks.

## Team

| Role | Member |
| --- | --- |
| Product Owner | _TODO_ |
| Scrum Master | _TODO_ |
| Developers | _TODO_ |

The Product Owner and Scrum Master roles may rotate between sprints.

## Target hardware

Development and testing happen in the PX4 + Gazebo simulation. The component is meant to run eventually on this hardware, as specified by the instructor:

| Component | Details |
| --- | --- |
| Onboard computer | [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/): ARM64 with an NVIDIA GPU; JetPack 6 is based on Ubuntu 22.04, which matches ROS 2 Humble |
| Camera | Stereolabs ZED stereo camera (RGB + depth): [ZED SDK](https://www.stereolabs.com/developers/release) |
| ROS 2 camera driver | [zed-ros2-wrapper](https://github.com/stereolabs/zed-ros2-wrapper) |

The current code targets the simulation only. Before it can run on the hardware:

- [x] Make the camera topic a ROS parameter (`camera_topic`) so it can subscribe to the ZED wrapper's image topic. Tested with the real ZED on a laptop, see [Real ZED camera on a laptop](#real-zed-camera-on-a-laptop).
- [ ] Make the inference device a parameter (it is hardcoded to `cpu`) so YOLO can use the Jetson's GPU; consider a TensorRT export for speed.
- [x] Make the OpenCV display window optional (`show_window:=false`), because the drone runs without a monitor.
- [ ] Write and test Jetson setup steps. The x86 install in [Setup](#setup) does not apply: the Jetson needs NVIDIA's PyTorch builds, and the ZED SDK needs a real NVIDIA GPU, so it cannot be tested in a VM.

## How it works

```
PX4 SITL + Gazebo (x500_mono_cam)
        │  camera image
        ▼
ros_gz_image image_bridge  ──►  /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image
        │
        ▼
yolo_camera_node.py
  ├─ YOLOv8n detection (person class)
  ├─ ByteTrack persistent IDs (bytetrack_occlusion.yaml)
  ├─ Entry / exit / stale-track logic (stale_timeout param)
  └─ OpenCV window with boxes, labels, confidences, IDs
```

| File | Purpose |
| --- | --- |
| `sim.launch.py` | One command that starts PX4 + Gazebo, the image bridge, the YOLO node, and optionally MAVROS, QGroundControl, a raw camera viewer, and rosbag recording. |
| `yolo_camera_node.py` | ROS 2 node that detects and tracks objects and logs when objects enter or leave the view. |
| `bytetrack_occlusion.yaml` | ByteTrack settings. `track_buffer` controls how long a hidden object keeps its ID. |
| `scenario.py` | Repeatable Gazebo test scenarios (entry/exit and occlusion) driven by the `gz` CLI. |
| `zed.launch.py` | The real ZED camera on a laptop: camera + YOLO node, with each person's distance. |
| `setup_zed.sh` | One-time setup for `zed.launch.py`, and the `zed-demo` command. |

## Requirements status

| Core requirement | Status |
| --- | --- |
| Accept simulated camera images through ROS 2 | ✅ Done |
| Obtain UAV pose/state from PX4 through MAVROS | 🟡 MAVROS can be launched; the node does not use it yet |
| Detect objects of specified target classes | 🟡 Person class only (`classes=[0]`); not yet configurable |
| Bounding box, class label, and confidence per detection | 🟡 Shown in the visualization; not published |
| Assign persistent IDs | ✅ ByteTrack |
| Associate detections with existing tracks across frames | ✅ ByteTrack |
| Handle objects entering and leaving the view | ✅ Logged as entry/exit events |
| Handle temporary loss or occlusion | ✅ `track_buffer` keeps IDs through short occlusions |
| Remove stale tracks after a configurable period | ✅ `stale_timeout` ROS parameter (default 3.0 s) |
| Publish detections and tracks through documented ROS 2 interfaces | ❌ Not started |
| Visualize boxes, classes, confidences, and IDs | ✅ OpenCV window |
| Log detection and tracking results for analysis | 🟡 ROS logs + optional rosbag; no structured log yet |
| Measure performance (accuracy, track continuity, latency, FPS) | ❌ Not started |
| Automated tests with repeatable scenarios | 🟡 `scenario.py` drives scenarios; results are checked by hand |

## Setup

Runs on **Ubuntu 22.04** with **ROS 2 Humble** and **Gazebo Harmonic**. The launch file expects these in your home directory:

| Path | What it is |
| --- | --- |
| `~/PX4-Autopilot` | PX4 source, built for SITL with Gazebo |
| `~/yolo_ros2_env` | Python virtual environment with `ultralytics` and `opencv-python` |
| `~/yolo_camera_node.py` | Symlink to the node in this repo (step 4) |
| `~/Downloads/QGroundControl-4.4.3.AppImage` | Optional; run `chmod +x` on it once |

### 1. ROS 2 packages

Install ROS 2 Humble (`ros-humble-desktop`), Gazebo Harmonic, and PX4 (built with `make px4_sitl`) first by following their official guides. Then install:

```bash
sudo apt update
sudo apt install ros-humble-ros-gzharmonic ros-humble-image-tools ros-humble-rosbag2 \
                 ros-humble-mavros ros-humble-mavros-extras python3.10-venv
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh   # only needed for mavros:=true
```

Use `ros-humble-ros-gzharmonic`, **not** `ros-humble-ros-gz`. The plain package is built for Gazebo Fortress and conflicts with Harmonic. It comes from the OSRF apt repo that the Gazebo Harmonic install sets up.

### 2. Python environment for YOLO

```bash
python3 -m venv --system-site-packages ~/yolo_ros2_env
~/yolo_ros2_env/bin/pip install --upgrade pip
# CPU-only PyTorch: the node runs YOLO on the CPU, and this skips a multi-GB CUDA download
~/yolo_ros2_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/yolo_ros2_env/bin/pip install ultralytics opencv-python "numpy<2"
```

Keep the `numpy<2` pin. Humble's `cv_bridge` is built against NumPy 1.x, and ultralytics would otherwise install NumPy 2, which breaks it. The `yolov8n.pt` weights and the `lap` package download automatically the first time the node runs.

If `venv` fails with "ensurepip is not available", install `python3.10-venv`, run `rm -rf ~/yolo_ros2_env`, and create the environment again.

### 3. Person model for the test scenarios

```bash
gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/scrubs/2
ls ~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/scrubs/2/model.sdf   # must exist
```

### 4. Link the node into your home directory

`sim.launch.py` runs `~/yolo_camera_node.py`, not the copy in this repo. Symlink it once so your edits take effect:

```bash
ln -sf ~/VideoGuidanceSystem/yolo_camera_node.py ~/yolo_camera_node.py
```

The node follows the symlink and loads `bytetrack_occlusion.yaml` from the repo folder. If you copy the file instead of linking it, copy the YAML next to it too.

### 5. Check the install

```bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix ros_gz_image          # should print /opt/ros/humble
~/yolo_ros2_env/bin/python -c "import rclpy, cv_bridge, ultralytics, cv2; print('ok')"
```

### Running in a VMware (or other) virtual machine

Gazebo's 3D rendering crashes on VMware's virtual graphics adapter. The Gazebo window opens and closes within a couple of seconds, and Ubuntu shows a crash report for `ruby` or `ros2`. Always launch with software rendering:

```bash
ros2 launch ./sim.launch.py software_render:=true
```

Gazebo then renders on the CPU. It runs slower (a real-time factor around 35% is normal), but it is stable. If the window still closes on a Wayland session, also force X11 for Qt:

```bash
QT_QPA_PLATFORM=xcb ros2 launch ./sim.launch.py software_render:=true
```

Run `systemd-detect-virt` to check whether you are in a VM (it prints `vmware`, `oracle`, and so on, or `none`).

### Troubleshooting

- **`'./sim.launch.py' is not a valid package name`:** you are not in the repo folder. Run `cd ~/VideoGuidanceSystem` first, or pass the full path to the launch file.
- **`PX4 server already running for instance 0`:** a previous run is still alive. Stop it with Ctrl+C in its terminal, or run `pkill -x px4`, then launch again.
- **`Preflight Fail: No connection to the GCS`:** harmless. PX4 is reporting that QGroundControl isn't running. Add `qgc:=true` if you need it.
- **Ubuntu "application has stopped unexpectedly" popups** can be left over from an earlier failed run. Click **Don't send**.

## Usage

```bash
source /opt/ros/humble/setup.bash
cd ~/VideoGuidanceSystem   # ./sim.launch.py is a relative path

# Default: PX4 + Gazebo + image bridge + YOLO node (add software_render:=true in a VM)
ros2 launch ./sim.launch.py

# Optional extras
ros2 launch ./sim.launch.py software_render:=true mavros:=true qgc:=true

# Record the camera feed and logs to a rosbag
ros2 launch ./sim.launch.py record:=true bag_name:=object_occlusion_test

# Same tracking settings as the real ZED camera (BoT-SORT, see below)
ros2 launch ./sim.launch.py tracker:=botsort
```

Launch arguments: `px4`, `yolo`, `mavros`, `qgc`, `viewer`, `record`, `software_render`, `show_window` (all `true`/`false`), `bag_name`, `stale_timeout` (seconds, default 3.0) and `tracker`:

- `bytetrack` (default): the original settings, ByteTrack on every 3rd frame.
- `botsort`: the same tracking settings as `zed.launch.py`: BoT-SORT with camera motion
  compensation, every frame, an ID only after 3 frames in a row. One difference
  ([`botsort_sim.yaml`](botsort_sim.yaml)): tracking starts at 25% confidence instead of 40%,
  because YOLO is less confident about the rendered person and would otherwise never track them.

Every 5 seconds the node logs what each stage sees, for example
`Status: YOLO sees 1 people, best 37% -> tracker follows 1 -> 1 with an ID, 4.2 fps`. If a visible
person gets no box, this shows whether YOLO missed them or the tracker filtered them out
(`status_period:=0` turns it off).

`setup_zed.sh` also adds a `sim-demo` command: `ros2 launch ./sim.launch.py tracker:=botsort`, plus
`software_render:=true` when it detects a virtual machine. Options go after it, for example
`sim-demo record:=true bag_name:=sim_test`.

Ctrl+C stops everything and closes the rosbag cleanly.

## Real ZED camera on a laptop

`zed.launch.py` runs the node on the real Stereolabs ZED plugged into a laptop or VM, without an
NVIDIA GPU. The camera is read by `zed_usb_publisher.py` from another team's repo,
[ArnavSingh121/Proj_1](https://github.com/ArnavSingh121/Proj_1/tree/main/zed_usb) (credit to them;
it is not copied into this repo). It reads the ZED as a plain USB camera, computes depth on the CPU
and publishes the same topics as the ZED ROS 2 wrapper on the Jetson:

| Topic | What |
| --- | --- |
| `/zed/zed_node/rgb/color/rect/image` | Color image (the node's `camera_topic`) |
| `/zed/zed_node/depth/depth_registered` | Depth in meters, pixel-aligned with the color image (`depth_topic`) |

With a depth topic, the node shows each person's distance (median depth over their torso; `? m`
where stereo depth has a hole).

**Setup, once, with internet and the ZED plugged in:**

```bash
cd ~/VideoGuidanceSystem
bash setup_zed.sh
```

It clones Proj_1 to `~/Proj_1` (for the camera code), adds `lap` to `~/yolo_ros2_env` (ByteTrack
needs it; otherwise it is downloaded on first run), saves `yolov8n.pt` in the repo folder, saves the
camera's factory calibration, turns off USB autosuspend for the ZED, and adds the `zed-demo` command.
After that, no internet is needed.

**Run** (in a new terminal; you can plug the camera in after starting):

```bash
zed-demo                                  # same as: ros2 launch ./zed.launch.py
zed-demo flip:=true                       # camera mounted upside down
zed-demo stale_timeout:=5.0               # people keep their ID for up to 5 s out of view
zed-demo record:=true bag_name:=zed_test  # also record color, depth and /rosout to a rosbag
```

In a VM, connect the camera to the VM first (VMware: **VM → Removable Devices → Stereolabs ZED →
Connect**). `zed-demo` keeps ROS 2 on this computer (`ROS_LOCALHOST_ONLY=1`) so other groups' topics
on a shared network don't mix in; run `export ROS_LOCALHOST_ONLY=1` in other terminals to see the
topics, or launch with `localhost_only:=false`.

**IDs through short gaps:** ByteTrack keeps a hidden person's ID for `track_buffer` processed frames.
The node now sets that from `stale_timeout` and the measured frame rate, so a person who leaves the
view and comes back within `stale_timeout` seconds keeps their ID on any machine (sim, laptop or
Jetson). After longer they are logged as left and come back with a new ID.

**Fewer duplicate IDs with a moving camera:** with a handheld camera at a few frames per second,
ByteTrack loses people every time the camera moves and gives them new IDs (in a 30 s test, 65 IDs for
about 12 people). The ZED launch therefore uses BoT-SORT with camera motion compensation
([`botsort_zed.yaml`](botsort_zed.yaml)), passes weak detections to the tracker (`conf:=0.1`) so
partly hidden people stay tracked, and gives a person an ID only after 3 frames in a row
(`min_hits`), so one-frame false detections don't use up IDs. IDs are numbered in order
(1, 2, 3, ...); the tracker's own numbers skip every short-lived candidate. Same test: 23 IDs.

**Expect** a few frames per second: the CPU computes both the stereo depth and YOLO. Depth has holes
on plain walls and ceilings.

New node parameters (defaults keep the simulation behavior unchanged):

| Parameter | Default | ZED launch |
| --- | --- | --- |
| `camera_topic` | Gazebo camera topic | `/zed/zed_node/rgb/color/rect/image` |
| `depth_topic` | empty (no distances) | `/zed/zed_node/depth/depth_registered` |
| `camera_qos` | `sensor_data` | `reliable` (best effort drops the ZED's ~1 MB images) |
| `process_every` | `3` | `1` |
| `imgsz` | `320` | `640` |
| `show_window` | `true` | `true` (`false` on the drone) |
| `tracker` | `bytetrack_occlusion.yaml` | `botsort_zed.yaml` |
| `conf` | `0.35` | `0.1` |
| `min_hits` | `1` | `3` |

## Testing

With the simulation running, in a second terminal:

```bash
python3 scenario.py entry_exit                    # person enters, leaves, re-enters
python3 scenario.py occlusion --occlude-time 2.5  # a box hides the person briefly
python3 scenario.py all --hold 5.0
```

**Expected results** (check the YOLO node's log output):

- **entry_exit:** `Object entered` → `Object left` after the stale timeout → `Object entered` again on re-entry.
- **occlusion:** the person keeps the **same ID** after the box moves away.

`--hold` must be longer than the node's `stale_timeout` for exit events to appear. The person model must be downloaded from Gazebo Fuel to `~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/scrubs/2/model.sdf`.

## Agile process

The team follows Scrum with at least two sprints. The backlog, sprint goals, user stories, estimates, and assignments live on the project board. Scrum meeting notes, sprint reviews, and retrospectives are recorded each sprint. All work is done on feature branches and merged into `main` through peer-reviewed pull requests.
