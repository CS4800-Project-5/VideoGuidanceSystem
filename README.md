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
- [x] Make the inference device a parameter (`device:=cuda`) so YOLO can use the Jetson's GPU. Still to do: consider a TensorRT export for speed.
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
  ├─ YOLO detection (person class)
  ├─ BoT-SORT tracking with camera motion compensation (botsort_zed.yaml / botsort_sim.yaml)
  ├─ Appearance re-identification (reidentifier.py): same ID after leaving and returning
  ├─ Entry / exit / stale-track logic (stale_timeout param)
  ├─ UAV pose and state from PX4 through MAVROS (uav_state.py)
  ├─ ROS topics for other groups (track_publisher.py) + CSV log (tracking_logger.py)
  └─ OpenCV window with boxes, labels, confidences, IDs
```

The camera is always moving (handheld in tests, on the drone later), so tracking uses only
BoT-SORT: it estimates the camera's own motion from the background and corrects for it. ByteTrack
assumes a still camera and lost people every time the camera moved.

| File | Purpose |
| --- | --- |
| `sim.launch.py` | One command that starts PX4 + Gazebo, the image bridge, the YOLO node, and optionally MAVROS, QGroundControl, a raw camera viewer, and rosbag recording. |
| `yolo_camera_node.py` | ROS 2 node that detects and tracks objects and logs when objects enter or leave the view. |
| `botsort_zed.yaml`, `botsort_sim.yaml` | BoT-SORT tracker settings for the real camera and the simulation. |
| `track_publisher.py` | Publishes the tracked people and entered/left events as ROS topics for other groups ([ROS topics](#ros-topics-for-other-groups)). |
| `tracking_logger.py` | Writes the same detections and events to a CSV file for analysis. |
| `uav_state.py` | Reads the UAV's pose and state (armed, flight mode) from PX4 through MAVROS; optionally publishes the pose to TF. |
| `evaluate_tracking.py` | Performance report from the CSV log: frame rate, latency, detection rate, IDs per person, track continuity, re-identification ([Measuring performance](#measuring-performance)). |
| `scenario.py` | Repeatable Gazebo test scenarios (entry/exit and occlusion) driven by the `gz` CLI. |
| `zed.launch.py` | The real ZED camera on a laptop: camera + YOLO node, with each person's distance. |
| `setup_zed.sh` | One-time setup for `zed.launch.py`, and the `zed-demo` command. |
| `compare_models.py` | Compares YOLO models and input sizes on a recorded rosbag. |
| `reidentifier.py` | Re-identification: recognizes returning people by appearance (OSNet, or a color histogram fallback), with no time limit. The appearance measure is a swappable Strategy class. |
| `osnet.py` | OSNet person re-identification network, from [torchreid](https://github.com/KaiyangZhou/deep-person-reid) (MIT). |

## Requirements status

| Core requirement | Status |
| --- | --- |
| Accept simulated camera images through ROS 2 | ✅ Done |
| Obtain UAV pose/state from PX4 through MAVROS | ✅ `uav_state.py` reads pose and state (`mavros:=true` in the sim); shown in the status log, saved in the CSV, optionally published to TF. Missing or late messages only log a warning |
| Detect objects of specified target classes | 🟡 Person class only (`classes=[0]`); not yet configurable |
| Bounding box, class label, and confidence per detection | ✅ Shown in the visualization and published on `/perception/tracks` |
| Assign persistent IDs | ✅ BoT-SORT, plus appearance re-identification so a person keeps their ID after leaving and returning (no time limit by default) |
| Associate detections with existing tracks across frames | ✅ BoT-SORT |
| Handle objects entering and leaving the view | ✅ Logged as entry/exit events |
| Handle temporary loss or occlusion | ✅ `track_buffer` keeps IDs through short occlusions |
| Remove stale tracks after a configurable period | ✅ `stale_timeout` ROS parameter (default 3.0 s) |
| Publish detections and tracks through documented ROS 2 interfaces | ✅ `/perception/tracks`, `/perception/tracks_3d`, `/perception/track_events` ([ROS topics](#ros-topics-for-other-groups)) |
| Visualize boxes, classes, confidences, and IDs | ✅ OpenCV window |
| Log detection and tracking results for analysis | ✅ CSV file (`csv_log`, default `~/yolo_tracking_log.csv`), ROS logs, optional rosbag |
| Measure performance (accuracy, track continuity, latency, FPS) | 🟡 `evaluate_tracking.py` reports frame rate, latency, detection rate, IDs per real person and track continuity; no labeled ground truth for detection accuracy yet |
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
                 ros-humble-mavros ros-humble-mavros-extras ros-humble-vision-msgs \
                 ros-humble-tf2-ros-py python3.10-venv
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

The node follows the symlink and loads its other files (the BoT-SORT YAML files, `reidentifier.py`, `track_publisher.py`, `tracking_logger.py`, the model weights) from the repo folder, so link it rather than copying it.

### 5. Check the install

```bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix ros_gz_image          # should print /opt/ros/humble
~/yolo_ros2_env/bin/python -c "import rclpy, cv_bridge, ultralytics, cv2; print('ok')"
```

### 6. Lighter simulated camera (recommended, required in a VM)

PX4's simulated camera renders 1280×960 images at 30 fps. Without a real GPU (any VM), that rendering limits the whole simulation to about 45% of real time, and the camera feed stutters. YOLO shrinks every frame to 320 pixels anyway, so a smaller camera loses no detection quality. Back up PX4's camera model, then switch it to 640×480 at 10 fps:

```bash
f=~/PX4-Autopilot/Tools/simulation/gz/models/mono_cam/model.sdf
cp -n "$f" "$f.orig-1280x960-30hz"
sed -i 's|<width>1280</width>|<width>640</width>|; s|<height>960</height>|<height>480</height>|; s|<update_rate>30</update_rate>|<update_rate>10</update_rate>|' "$f"
grep -E "<width>|<height>|<update_rate>" "$f"   # should show 640, 480, 10
```

To undo it: `cp "$f.orig-1280x960-30hz" "$f"`. A PX4 update can reset this file, so re-check it after updating PX4. Measured on a 12-vCPU VMware VM: about 45% → **100%** of real time. (This edit lives outside the repo; moving the camera model into the repo is planned.)

### Running in a VMware (or other) virtual machine

Gazebo's 3D rendering crashes on VMware's virtual graphics adapter. The Gazebo window opens and closes within a couple of seconds, and Ubuntu shows a crash report for `ruby` or `ros2`. Always launch with software rendering:

```bash
ros2 launch ./sim.launch.py software_render:=true
```

Gazebo then renders on the CPU. It is stable, but it is only fast enough with the lighter camera from step 6 and enough CPU cores. Give the VM at least 8 cores (about half your host's threads) in *VM Settings → Processors*; 4 cores is too few. If the window still closes on a Wayland session, also force X11 for Qt:

```bash
QT_QPA_PLATFORM=xcb ros2 launch ./sim.launch.py software_render:=true
```

Run `systemd-detect-virt` to check whether you are in a VM (it prints `vmware`, `oracle`, and so on, or `none`).

### Troubleshooting

- **`'./sim.launch.py' is not a valid package name`:** you are not in the repo folder. Run `cd ~/VideoGuidanceSystem` first, or pass the full path to the launch file.
- **`PX4 server already running for instance 0`:** a previous run is still alive. Stop it with Ctrl+C in its terminal, or run `pkill -x px4` (and stop any leftover `gz sim` processes), then launch again.
- **`Preflight Fail: No connection to the GCS`:** harmless. PX4 is reporting that QGroundControl isn't running. Add `qgc:=true` if you need it.
- **The sim is slow, the camera looks frozen, or the person doesn't move:** check the camera settings (step 6) and the VM's core count. Run `uptime`: a high load right after boot is often Ubuntu's automatic updates (`unattended-upgrades`). Wait for them to finish, or stop them safely with `sudo systemctl stop unattended-upgrades`. Never kill `apt` or `dpkg`.
- **`Object left … (no camera frames)` repeating:** the camera stopped publishing (see the previous item).
- **Ubuntu "application has stopped unexpectedly" popups** can be left over from an earlier failed run. Click **Don't send**.

## Running the project

You need **two terminals**: one runs the whole stack, the other drives a test scenario. The default world is empty (just the drone, the ground, and the sky), so YOLO has nothing to detect until a scenario adds a person.

### Before you start

```bash
ps -eo pid,cmd | grep -E "bin/px4|gz sim|yolo_camera_node" | grep -v grep   # should print nothing
```

If anything is listed, a previous run is still alive. Stop it with Ctrl+C in its terminal, or `kill <pid>`.

### Terminal 1: start everything

```bash
cd ~/VideoGuidanceSystem          # required: ./sim.launch.py is a relative path
source /opt/ros/humble/setup.bash
ros2 launch ./sim.launch.py software_render:=true   # drop software_render on a machine with a real GPU
```

This uses the same tracking settings as the real ZED camera (BoT-SORT, re-identification on). After running `setup_zed.sh` once, the `sim-demo` command does the same (and adds `software_render:=true` in a VM). Run `setup_zed.sh` again after updating, so `sim-demo` stops passing the old `tracker:=botsort` option.

This starts PX4, the Gazebo server and window, the camera bridge, and the YOLO node. It's ready when, after about 15–30 seconds:

- the Gazebo window shows the drone on the ground,
- the log prints `YOLO object entry/exit tracker started on /world/default/... (stale timeout 3.0s)`,
- the **YOLO Tracking** window opens (grey sky and ground).

`Preflight Fail: No connection to the GCS` warnings are normal.

### Terminal 2: run a test scenario

```bash
cd ~/VideoGuidanceSystem
python3 scenario.py entry_exit --hold 25   # person enters, leaves, comes back
python3 scenario.py occlusion              # a box briefly hides the person
python3 scenario.py all --hold 25          # both
```

Watch the YOLO window and the node's log lines in Terminal 1.

**Expected results:**

| Scenario | Expected log in Terminal 1 |
| --- | --- |
| `entry_exit` | `Object entered camera view: ID 1` → `Object left camera view: ID 1 (unseen for 3.xs)` → `Object entered camera view: ID 1 (re-identified by appearance, similarity 0.xx)` |
| `occlusion` | The person keeps the **same ID** after the box moves away |

**Choosing `--hold`:** it is how long, in real seconds, the scenario waits at each step. It must be longer than the node's `stale_timeout` (3 s of *simulation* time) for "left" events to appear, so use more if your sim runs slower than real time. `--hold 25` also keeps the person away long enough that the tracker forgets them, so the return actually tests appearance re-identification.

Every 5 seconds the node also logs what each stage sees, for example
`Status: YOLO sees 1 people, best 37% -> tracker follows 1 -> 1 with an ID, 4.2 fps`. If a visible
person gets no box, this shows whether YOLO missed them or the tracker filtered them out
(`status_period:=0` turns it off).

The person model is a statue: it doesn't walk, and the scenario teleports it between positions.

### Stopping

Press Ctrl+C in Terminal 1. Then repeat the check from [Before you start](#before-you-start), because PX4 or the Gazebo server sometimes keeps running after the launch exits.

### Launch arguments

| Argument | Default | What it does |
| --- | --- | --- |
| `software_render` | `false` | Render on the CPU. Required in a VM. |
| `px4` | `true` | Start PX4 SITL and Gazebo |
| `yolo` | `true` | Start the YOLO node |
| `mavros` | `false` | Start the MAVROS bridge to PX4, and have the node read the UAV pose and state from it |
| `qgc` | `false` | Start QGroundControl |
| `viewer` | `false` | Show the raw camera feed (`image_tools showimage`) |
| `record` | `false` | Record the camera topic, `/rosout` and the `/perception` topics to a rosbag |
| `bag_name` | `object_tracking_test` | Output folder for the rosbag |
| `reid` | `true` | Recognize returning people by appearance |
| `stale_timeout` | `3.0` | Seconds before an unseen object counts as having left |
| `show_window` | `true` | Show the OpenCV tracking window |

Example: `ros2 launch ./sim.launch.py software_render:=true record:=true bag_name:=occlusion_run1`. Ctrl+C closes the rosbag cleanly.

The simulation uses the ZED's tracking settings except that tracking starts at 25% confidence instead of 40% ([`botsort_sim.yaml`](botsort_sim.yaml)), because YOLO is less confident about the rendered person.

### Node parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `stale_timeout` | `3.0` | Seconds (image time) without a detection before an object counts as having left |
| `tracker` | `botsort_zed.yaml` | BoT-SORT settings file (`sim.launch.py` passes `botsort_sim.yaml`) |
| `conf` | `0.1` | Lowest YOLO confidence passed to the tracker (weak detections only keep existing tracks alive) |
| `min_hits` | `3` | Frames in a row a person must be tracked before getting an ID |
| `process_every` | `1` | Run YOLO on every Nth frame; raise it to save CPU |
| `device` | `cpu` | Inference device, e.g. `cuda` on the Jetson (also used for re-ID) |
| `reid` | `true` | Re-identify returning people by appearance |
| `reid_model` | empty | OSNet weights file; empty = `osnet_x0_5` on a CPU, `osnet_ibn_x1_0` on a GPU |
| `reid_memory` | `0.0` | Seconds a departed person is remembered for re-identification; 0 = as long as the program runs |
| `reid_threshold` | `-1` | How similar a new person must look to a remembered one to get their ID; -1 = the model's default (OSNet 0.65, color histogram 0.7) |
| `reid_min_conf` | `0.3` | Lowest YOLO confidence at which a new appearance sample of a person is saved |
| `world_frame` | empty | Fixed TF frame (`map`, `odom`) for world positions; turns on position checks in re-ID |
| `uav` | `true` | Read the UAV pose and state from MAVROS (`sim.launch.py` sets it from `mavros`; `zed.launch.py` from its `uav` argument, default `false`) |
| `mavros_ns` | `/mavros` | Namespace of the MAVROS topics |
| `uav_timeout` | `1.0` | Seconds without a MAVROS message before it counts as missing (logged once; tracking continues) |
| `uav_publish_tf` | `false` | Publish the UAV pose to TF (`map` -> `base_link`); on in the sim with `mavros:=true`. Leave off when the ZED wrapper publishes that TF |
| `reid_max_speed` | `3.0` | With `world_frame`: fastest a person is assumed to walk (m/s) |
| `topic_prefix` | `/perception` | Prefix of the published topics |
| `csv_log` | `~/yolo_tracking_log.csv` | CSV file for detections and events; empty = no CSV |
| `camera_info_topic` | empty | Camera intrinsics; with depth, gives 3D positions (`tracks_3d`) |
| `status_period` | `5.0` | Seconds between status log lines; 0 turns them off |

The camera and model parameters used by the real ZED are listed in [Real ZED camera on a laptop](#real-zed-camera-on-a-laptop).

### Running the node on its own

Useful against a rosbag replay, or to restart only the node while Gazebo keeps running:

```bash
source /opt/ros/humble/setup.bash
~/yolo_ros2_env/bin/python ~/yolo_camera_node.py --ros-args -p stale_timeout:=3.0
# in another terminal, for a bag: ros2 bag play <bag_folder>
```

## ROS topics for other groups

The node publishes what it tracks so other groups (collision avoidance, path planning, mapping) can
use it. Message types are standard ROS 2 packages (`vision_msgs`, `std_msgs`), so nothing from this
repo has to be built to subscribe. Every message carries the camera image's header (`stamp` and
`frame_id`).

| Topic | Type | When | Contents |
| --- | --- | --- | --- |
| `/perception/tracks` | `vision_msgs/Detection2DArray` | every processed frame (empty when no one is in view) | one `Detection2D` per person: `id` = their ID (`"3"`), `bbox` in image pixels, `results[0].hypothesis` = `person` and the YOLO confidence |
| `/perception/tracks_3d` | `vision_msgs/Detection3DArray` | every processed frame, only with depth (the ZED) | one `Detection3D` per person whose distance is known: `id`, `bbox.center.position` = torso position in meters in the camera's optical frame (x right, y down, z forward), `bbox.size` = approximate width/height |
| `/perception/track_events` | `std_msgs/String` (JSON) | when someone enters or leaves | `{"event": "entered", "id": 3, "stamp": 1700000000.12, "how": "reidentified", "similarity": 0.81}` or `{"event": "left", "id": 3, "stamp": ..., "reason": "unseen for 3.2s"}` |

To get positions in the drone's or the map's frame, transform `tracks_3d` with tf2 using the
message's `frame_id`. Quick checks:

```bash
ros2 topic echo /perception/track_events
ros2 topic echo /perception/tracks_3d --field detections
ros2 topic hz /perception/tracks
```

The same detections and events also go to a CSV file (`~/yolo_tracking_log.csv`, set with
`csv_log`), one row per person per frame, with the image time, 3D position and the UAV's pose from
MAVROS. A CSV written with older columns is renamed (`.before-<time>.csv`) and a new one is started.

The UAV's own pose and state are not republished: subscribe to MAVROS directly
(`/mavros/local_position/pose`, `/mavros/state`).

## Measuring performance

`evaluate_tracking.py` turns the CSV log into a report. It needs only Python, so it also runs on a
log copied to another computer.

```bash
python3 evaluate_tracking.py                     # the last run in ~/yolo_tracking_log.csv
python3 evaluate_tracking.py --list              # the runs in the file (the node appends)
python3 evaluate_tracking.py --run 2 --people 3  # run 2, in which 3 different people were filmed
python3 evaluate_tracking.py --json              # machine-readable
```

```
Run 2 of 2: 100 frames, 9.9 s
  Frame rate         10.0 fps
  Latency            mean 41.6 ms, median 42.0, 95% under 49.0, max 49.0
  Detection          people in 70.0% of frames, 0.8 per frame, mean confidence 0.8
  IDs                2 different, 1.0 per real person (1.0 is ideal)
  Entries            3: 1 re-identified, 2 new
  Track continuity   1.5 entries per ID (1 = never lost), longest stretch 2.4 s on average, ...
  UAV pose           known in 50.0% of frames
```

**IDs per real person** is the main number for identity tracking: film a known number of people
(walking around them, from above, from behind), pass `--people`, and compare runs or settings. 1.0
means everyone kept one ID; 2.0 means each person was split into two IDs on average.

Limits make it usable in automated tests: `--min-fps`, `--max-latency-p95` (ms) and
`--max-ids-per-person` make it exit with code 1 when a limit is missed, for example
`python3 evaluate_tracking.py --people 3 --max-ids-per-person 1.5 --min-fps 5`.

A recorded rosbag can be re-run through the node with different settings: start the node on its
own ([Running the node on its own](#running-the-node-on-its-own)) and `ros2 bag play` the bag.

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
| `/zed/zed_node/rgb/color/rect/camera_info` | Intrinsics of the color image (`camera_info_topic`), for 3D positions |

With a depth topic, the node shows each person's distance (median depth over their torso; `? m`
where stereo depth has a hole).

**Setup, once, with internet and the ZED plugged in:**

```bash
cd ~/VideoGuidanceSystem
bash setup_zed.sh
```

It clones Proj_1 to `~/Proj_1` (for the camera code), adds `lap` to `~/yolo_ros2_env` (BoT-SORT
needs it; otherwise it is downloaded on first run), saves the YOLO and OSNet weights in the repo
folder, saves the camera's factory calibration, turns off USB autosuspend for the ZED, and adds the
`zed-demo` and `sim-demo` commands. After that, no internet is needed.

**Run** (in a new terminal; you can plug the camera in after starting):

```bash
zed-demo                                  # same as: ros2 launch ./zed.launch.py
zed-demo flip:=true                       # camera mounted upside down
zed-demo stale_timeout:=5.0               # people keep their ID for up to 5 s out of view
zed-demo record:=true bag_name:=zed_test  # also record color, depth, /rosout and /perception topics
```

In a VM, connect the camera to the VM first (VMware: **VM → Removable Devices → Stereolabs ZED →
Connect**). `zed-demo` keeps ROS 2 on this computer (`ROS_LOCALHOST_ONLY=1`) so other groups' topics
on a shared network don't mix in; run `export ROS_LOCALHOST_ONLY=1` in other terminals to see the
topics, or launch with `localhost_only:=false`.

**IDs through short gaps:** BoT-SORT keeps a hidden person's ID for `track_buffer` processed frames.
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

**Recognizing people who come back (re-identification):** the tracker alone remembers someone
only for `stale_timeout` seconds. With re-identification (on in `zed-demo` and `sim-demo`), the node
also remembers what every person looks like, with no time limit, so a person who leaves the view,
or the whole camera leaving the room and coming back, gets their old ID again:

```
Object left camera view: ID 3 (unseen for 3.2s)
...
Object entered camera view: ID 3 (re-identified by appearance, similarity 0.81)
Object entered camera view: ID 7 (new person; most similar remembered: ID 2 at 0.48, needs 0.65)
```

How it works ([`reidentifier.py`](reidentifier.py), based on Angel's re-ID in `angel-branch`):

1. While a new track is being confirmed (`min_hits` frames), each crop of the person goes through
   **OSNet**, a network trained to recognize the same person across cameras and angles (trained on
   MSMT17). Each crop becomes a 512-number appearance "fingerprint".
2. When the track is confirmed, the average fingerprint is compared with every remembered person
   who is **not in view right now** (so two people in view never share an ID). Each remembered
   person scores by their **single most similar sample**, so a view from above or behind matches
   the sample taken from that angle instead of being averaged with front views. The best match
   above `reid_threshold`, clearly better than the second best, gets that person's ID back;
   otherwise the person gets a new ID.
3. Every 5 frames, a new sample of each person in clear view (at least `reid_min_conf` = 30%
   confidence, so less certain views from above or behind are saved too, and not overlapping
   anyone) is added to their memory. When a person's 12 samples are full, the sample most like
   the others is dropped, so the memory keeps a spread of angles instead of only recent ones.
4. With `world_frame` set (a TF frame such as `map` that stays fixed while the camera moves, e.g.
   from the ZED wrapper's positional tracking on the Jetson), each person's last world position is
   remembered too. A remembered person is only a candidate if they could have walked to the new
   track's position since they were last seen (`reid_max_speed`, 3 m/s, plus 1 m for depth noise),
   so a look-alike across the room can't take their ID.

`setup_zed.sh` downloads two OSNet models from torchreid's model zoo on Google Drive:

| Model | Default on | Notes |
| --- | --- | --- |
| `osnet_x0_5_msmt17.pth` | CPU (laptop, VM) | small and fast: ~60 ms for 3 people on a laptop CPU |
| `osnet_ibn_x1_0_msmt17.pth` | GPU (`device:=cuda`, the Jetson) | full size, copes better with new viewpoints and lighting; ~10x the CPU time, so only practical on a GPU |

`reid_model:=osnet_ibn_x1_0_msmt17.pth` forces the larger model on a CPU. Without either file,
re-identification falls back to clothing colors (Angel's color histogram), which mixes up people
in similar clothes. Options:

| Option | Effect |
| --- | --- |
| `reid_threshold:=0.6` | similarity needed to count as the same person (default 0.65 with OSNet). Lower if returning people get new IDs; raise if two people get mixed up. The log lines show the scores |
| `reid:=false` | turn re-identification off |
| `world_frame:=map` | position check (needs TF from the camera to that frame) |

It recognizes people by their whole appearance (mostly clothes and build), not faces, so someone
who changes clothes gets a new ID, and people dressed alike can be confused. The memory is kept
only while the program runs; nothing is saved to disk.

Tested on photos (stand-in for the camera, with the clothing-color fallback, since the OSNet weights
can't be downloaded in that test environment): people hidden for 10 s and a camera "leaving the
room" for 20 s kept their IDs, and different people got new IDs. Re-ID added about 14% CPU time.

**Detecting more people:** the ZED launch uses `yolo11s.pt` instead of the sim's `yolov8n.pt`.
On a test crowd scene with near and far people (VGA, image size 640) it found 59% of them at 25%+
confidence instead of 20%, at about twice the CPU time. Options:

| Option | Effect |
| --- | --- |
| `model:=yolov8n.pt` | fastest, finds the fewest far-away people |
| `imgsz:=960` | about 2x slower, finds many more small / far people (80% with yolo11s in the same test) |
| `resolution:=HD720` | 4x the pixels per frame (try with `imgsz:=1280`); depth takes ~4x the CPU; needs USB 3 |

To choose on real footage and on your own computer, record a bag and compare:

```bash
zed-demo record:=true bag_name:=class1
~/yolo_ros2_env/bin/python compare_models.py class1    # after: source /opt/ros/humble/setup.bash
```

It prints people found per frame, average confidence and time per frame for each model and size.

**Expect** a few frames per second: the CPU computes both the stereo depth and YOLO. Depth has holes
on plain walls and ceilings.

Node parameters the ZED launch sets differently from the defaults:

| Parameter | Default | ZED launch |
| --- | --- | --- |
| `camera_topic` | Gazebo camera topic | `/zed/zed_node/rgb/color/rect/image` |
| `depth_topic` | empty (no distances) | `/zed/zed_node/depth/depth_registered` |
| `camera_info_topic` | empty (no 3D positions) | `/zed/zed_node/rgb/color/rect/camera_info` |
| `camera_qos` | `sensor_data` | `reliable` (best effort drops the ZED's ~1 MB images) |
| `model` | `yolov8n.pt` | `yolo11s.pt` |
| `imgsz` | `320` | `640` |

## Agile process

The team follows Scrum with at least two sprints. The backlog, sprint goals, user stories, estimates, and assignments live on the project board. Scrum meeting notes, sprint reviews, and retrospectives are recorded each sprint. All work is done on feature branches and merged into `main` through peer-reviewed pull requests.
