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

- [ ] Make the camera topic a ROS parameter (it is hardcoded to the Gazebo topic) so it can subscribe to the ZED wrapper's image topic.
- [ ] Make the inference device a parameter (it is hardcoded to `cpu`) so YOLO can use the Jetson's GPU; consider a TensorRT export for speed.
- [ ] Make the OpenCV display window optional, because the drone runs without a monitor.
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
  ├─ OpenCV window with boxes, labels, confidences, IDs
  └─ Publishes /perception/{detections,tracks,track_events,image_annotated}
```

| File | Purpose |
| --- | --- |
| `sim.launch.py` | One command that starts PX4 + Gazebo, the image bridge, the YOLO node, and optionally MAVROS, QGroundControl, a raw camera viewer, and rosbag recording. |
| `yolo_camera_node.py` | ROS 2 node that detects and tracks objects and logs when objects enter or leave the view. |
| `bytetrack_occlusion.yaml` | ByteTrack settings. `track_buffer` controls how long a hidden object keeps its ID. |
| `scenario.py` | Repeatable Gazebo test scenarios (entry/exit and occlusion) driven by the `gz` CLI. |

## Requirements status

| Core requirement | Status |
| --- | --- |
| Accept simulated camera images through ROS 2 | ✅ Done |
| Obtain UAV pose/state from PX4 through MAVROS | 🟡 MAVROS can be launched; the node does not use it yet |
| Detect objects of specified target classes | 🟡 Person class only (`classes=[0]`); not yet configurable |
| Bounding box, class label, and confidence per detection | ✅ Published on `/perception/detections` |
| Assign persistent IDs | ✅ ByteTrack |
| Associate detections with existing tracks across frames | ✅ ByteTrack |
| Handle objects entering and leaving the view | ✅ Logged as entry/exit events |
| Handle temporary loss or occlusion | ✅ `track_buffer` keeps IDs through short occlusions |
| Remove stale tracks after a configurable period | ✅ `stale_timeout` ROS parameter (default 3.0 s) |
| Publish detections and tracks through documented ROS 2 interfaces | ✅ See [docs/ros_interfaces.md](docs/ros_interfaces.md) |
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
                 ros-humble-mavros ros-humble-mavros-extras ros-humble-vision-msgs python3.10-venv
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
~/yolo_ros2_env/bin/python -c "import rclpy, cv_bridge, ultralytics, cv2, vision_msgs; print('ok')"
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
```

Launch arguments: `px4`, `yolo`, `mavros`, `qgc`, `viewer`, `record`, `software_render` (all `true`/`false`), and `bag_name`.

Ctrl+C stops everything and closes the rosbag cleanly.

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
