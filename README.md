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

Runs on Ubuntu with **ROS 2 Humble**. The launch file expects these in your home directory:

| Path | What it is |
| --- | --- |
| `~/PX4-Autopilot` | PX4 source, built for SITL with Gazebo |
| `~/yolo_ros2_env` | Python virtual environment with `ultralytics` and `opencv-python` |
| `~/yolo_camera_node.py` | The node the launch file runs (see note below) |
| `~/Downloads/QGroundControl-4.4.3.AppImage` | Optional; run `chmod +x` on it once |

ROS packages used: `ros_gz_image`, `image_tools`, `mavros` (optional), and `rosbag2`.

> **Note:** `sim.launch.py` runs `~/yolo_camera_node.py`, not the copy in this repo. After you edit the repo version, copy it to your home directory or symlink it once:
> ```bash
> ln -sf "$(pwd)/yolo_camera_node.py" ~/yolo_camera_node.py
> ```
> Keep `bytetrack_occlusion.yaml` next to whichever copy runs, because the node looks for it in its own folder.

## Usage

```bash
source /opt/ros/humble/setup.bash

# Default: PX4 + Gazebo + image bridge + YOLO node
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
