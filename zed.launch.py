"""
Real ZED camera on a laptop (no NVIDIA GPU needed) -> YOLO + BoT-SORT node, with the distance
to each person from the ZED's depth. The tracked people are published on /perception/tracks,
/perception/tracks_3d and /perception/track_events (see track_publisher.py).

The camera is read by zed_usb_publisher.py from another team's repo,
https://github.com/ArnavSingh121/Proj_1 (folder zed_usb/), which is not part of this repo:
setup_zed.sh clones it to ~/Proj_1. It publishes the same topics as the ZED ROS 2 wrapper,
so the YOLO node subscribes to the same topics here and on the Jetson.

Usage (after running ./setup_zed.sh once):
    zed-demo                                  # same as: ros2 launch ./zed.launch.py
    zed-demo flip:=true                       # camera mounted upside down
    zed-demo stale_timeout:=5.0               # people keep their ID for up to 5 s out of view
    zed-demo record:=true bag_name:=zed_test  # also record a rosbag
    zed-demo model:=yolov8n.pt                # fastest model (fewer detections)
    zed-demo imgsz:=960                       # more far-away people, ~2x slower
    zed-demo reid_threshold:=0.6              # re-identify returning people more readily
    zed-demo reid:=false                      # no re-identification (IDs only last stale_timeout)
    zed-demo world_frame:=map                 # with TF (Jetson + ZED wrapper): position-aware re-ID
    zed-demo classes:="[person, car]"        # also track cars (re-ID is for people only)
    zed-demo uav:=true                        # on the drone: read the UAV pose and state from MAVROS

Ctrl+C stops everything, and rosbag2 closes the bag cleanly.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.realpath(__file__))

VENV_PYTHON = os.path.join(HOME, "yolo_ros2_env", "bin", "python")
YOLO_NODE = os.path.join(HERE, "yolo_camera_node.py")
ZED_USB_DIR = os.path.join(HOME, "Proj_1", "zed_usb")

# Topics published by zed_usb_publisher.py (and by the ZED ROS 2 wrapper on the Jetson).
ZED = "/zed/zed_node/"
RGB_TOPIC = ZED + "rgb/color/rect/image"
DEPTH_TOPIC = ZED + "depth/depth_registered"
RGB_INFO_TOPIC = ZED + "rgb/color/rect/camera_info"
RECORD_TOPICS = [
    RGB_TOPIC, RGB_INFO_TOPIC,
    DEPTH_TOPIC, ZED + "depth/camera_info",
    "/rosout",
    "/perception/tracks", "/perception/tracks_3d", "/perception/track_events",
]


def launch_setup(context):
    """The camera, YOLO and rosbag processes, or a clear message instead of a stack trace if
    setup_zed.sh hasn't been run."""
    zed_usb_dir = LaunchConfiguration("zed_usb_dir").perform(context)
    missing = []
    if not os.path.exists(os.path.join(zed_usb_dir, "zed_usb_publisher.py")):
        missing.append(f"ZED USB publisher not found in {zed_usb_dir}")
    if not os.path.exists(VENV_PYTHON):
        missing.append(f"Python environment not found: {VENV_PYTHON}")
    if missing:
        return [LogInfo(msg=m) for m in missing] + [
            LogInfo(msg=f"Run the setup once (needs internet): bash {os.path.join(HERE, 'setup_zed.sh')}"),
            Shutdown(reason="setup missing"),
        ]

    # Passed only when set: an empty "-p name:=" would not be read as an empty string.
    world_frame = LaunchConfiguration("world_frame").perform(context)
    optional = ["-p", f"world_frame:={world_frame}"] if world_frame else []

    flip = LaunchConfiguration("flip")
    stale_timeout = LaunchConfiguration("stale_timeout")
    show_window = LaunchConfiguration("show_window")
    record = LaunchConfiguration("record")
    bag_name = LaunchConfiguration("bag_name")

    return [
        # ---- ZED camera over USB: color + depth (CPU stereo) ----
        ExecuteProcess(
            cmd=["python3", os.path.join(zed_usb_dir, "zed_usb_publisher.py"), "--ros-args",
                 "-r", "__node:=zed_node", "-r", "__ns:=/zed", "-p", ["flip:=", flip],
                 "-p", ["resolution:=", LaunchConfiguration("resolution")]],
            # The system OpenCV is built for NumPy 1; a NumPy 2 in ~/.local would break it.
            additional_env={"PYTHONNOUSERSITE": "1"},
            output="screen",
            emulate_tty=True,
        ),

        # ---- YOLO + BoT-SORT node on the ZED topics (runs with the venv's Python) ----
        ExecuteProcess(
            cmd=[VENV_PYTHON, YOLO_NODE, "--ros-args",
                 "-p", f"camera_topic:={RGB_TOPIC}",
                 "-p", f"depth_topic:={DEPTH_TOPIC}",
                 # Intrinsics: with depth, they give each person's 3D position (tracks_3d topic).
                 "-p", f"camera_info_topic:={RGB_INFO_TOPIC}",
                 "-p", "camera_qos:=reliable",
                 # The ZED gives far fewer frames per second than the sim on a CPU: use every one.
                 "-p", "process_every:=1",
                 # Detection: model and input size (see the launch arguments).
                 "-p", ["model:=", LaunchConfiguration("model")],
                 "-p", ["imgsz:=", LaunchConfiguration("imgsz")],
                 # Handheld camera: BoT-SORT compensates for the camera's own motion; weak
                 # detections keep partly hidden people tracked; an ID only after 3 frames in a row.
                 "-p", f"tracker:={os.path.join(HERE, 'botsort_zed.yaml')}",
                 "-p", "conf:=0.1",
                 "-p", ["min_hits:=", LaunchConfiguration("min_hits")],
                 # Re-identification: recognize returning people by appearance, no time limit.
                 "-p", ["classes:=", LaunchConfiguration("classes")],
                 "-p", ["reid:=", LaunchConfiguration("reid")],
                 "-p", ["reid_threshold:=", LaunchConfiguration("reid_threshold")],
                 "-p", ["stale_timeout:=", stale_timeout],
                 "-p", ["show_window:=", show_window],
                 # UAV pose/state from MAVROS: only with uav:=true (no drone on a laptop).
                 "-p", ["uav:=", LaunchConfiguration("uav")],
                 *optional],
            # yolov8n.pt is loaded from the repo folder (setup_zed.sh downloads it there), so no
            # internet is needed when this runs.
            cwd=HERE,
            output="screen",
            emulate_tty=True,
        ),

        # ---- Optional: rosbag2 recording ----
        ExecuteProcess(
            cmd=["ros2", "bag", "record", *RECORD_TOPICS, "-o", bag_name],
            output="screen",
            condition=IfCondition(record),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        # ---- Launch arguments ----
        DeclareLaunchArgument("flip", default_value="false",
                              description="'true' if the camera is mounted upside down"),
        DeclareLaunchArgument("stale_timeout", default_value="3.0",
                              description="Seconds a person can be out of view and keep their ID"),
        DeclareLaunchArgument("model", default_value="yolo11s.pt",
                              description="YOLO weights. yolo11s.pt finds far more small/far people "
                                          "than yolov8n.pt (the fastest) at about 2x the CPU time; "
                                          "yolov8m.pt is better still and ~5x slower"),
        DeclareLaunchArgument("imgsz", default_value="640",
                              description="YOLO input size. 960 finds more far-away people but takes "
                                          "about 2x as long; for resolution:=HD720 try 1280"),
        DeclareLaunchArgument("resolution", default_value="VGA",
                              description="ZED resolution: VGA (672x376) or HD720 (1280x720, more "
                                          "detail for far people, but depth takes ~4x the CPU; needs USB 3)"),
        DeclareLaunchArgument("classes", default_value="[person]",
                              description="Classes to detect and track (YOLO's COCO names), "
                                          "e.g. \"[person, car]\"; re-identification is for people"),
        DeclareLaunchArgument("reid", default_value="true",
                              description="Recognize people who come back (left the view, or the camera "
                                          "left the room) by appearance, and give them their old ID"),
        DeclareLaunchArgument("reid_threshold", default_value="-1",
                              description="Appearance similarity needed to count as the same person; "
                                          "-1 = the model's default (OSNet 0.65)"),
        DeclareLaunchArgument("world_frame", default_value="",
                              description="Fixed TF frame (e.g. map or odom) for people's world "
                                          "positions; re-ID then rules out people last seen too far "
                                          "away. Empty = off (needs a TF publisher, e.g. the ZED "
                                          "wrapper's positional tracking on the Jetson)"),
        DeclareLaunchArgument("uav", default_value="false",
                              description="Read the UAV pose and state from MAVROS (on the drone)"),
        DeclareLaunchArgument("min_hits", default_value="3",
                              description="Frames in a row a person must be tracked before getting an ID"),
        DeclareLaunchArgument("show_window", default_value="true",
                              description="Show the OpenCV window with boxes, IDs and distances"),
        DeclareLaunchArgument("record", default_value="false",
                              description="Record the ZED color + depth images and /rosout with rosbag2"),
        DeclareLaunchArgument("bag_name", default_value="zed_tracking_test",
                              description="Output folder name for the bag"),
        DeclareLaunchArgument("zed_usb_dir", default_value=ZED_USB_DIR,
                              description="Folder with zed_usb_publisher.py (from ArnavSingh121/Proj_1)"),
        DeclareLaunchArgument("localhost_only", default_value="true",
                              description="Keep ROS 2 on this computer, so other groups' topics on a "
                                          "shared network don't mix in. Other terminals then need "
                                          "'export ROS_LOCALHOST_ONLY=1' to see the topics."),

        # ---- Environment ----
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1",
                               condition=IfCondition(LaunchConfiguration("localhost_only"))),

        OpaqueFunction(function=launch_setup),
    ])
