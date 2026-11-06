"""
Real ZED camera on a laptop (no NVIDIA GPU needed) -> YOLO + ByteTrack node, with the distance
to each person from the ZED's depth.

The camera is read by zed_usb_publisher.py from another team's repo,
https://github.com/ArnavSingh121/Proj_1 (folder zed_usb/), which is not part of this repo:
setup_zed.sh clones it to ~/Proj_1. It publishes the same topics as the ZED ROS 2 wrapper,
so the YOLO node subscribes to the same topics here and on the Jetson.

Usage (after running ./setup_zed.sh once):
    zed-demo                                  # same as: ros2 launch ./zed.launch.py
    zed-demo flip:=true                       # camera mounted upside down
    zed-demo stale_timeout:=5.0               # people keep their ID for up to 5 s out of view
    zed-demo record:=true bag_name:=zed_test  # also record a rosbag

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
RECORD_TOPICS = [
    RGB_TOPIC, ZED + "rgb/color/rect/camera_info",
    DEPTH_TOPIC, ZED + "depth/camera_info",
    "/rosout",
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

    flip = LaunchConfiguration("flip")
    stale_timeout = LaunchConfiguration("stale_timeout")
    show_window = LaunchConfiguration("show_window")
    record = LaunchConfiguration("record")
    bag_name = LaunchConfiguration("bag_name")

    return [
        # ---- ZED camera over USB: color + depth (CPU stereo) ----
        ExecuteProcess(
            cmd=["python3", os.path.join(zed_usb_dir, "zed_usb_publisher.py"), "--ros-args",
                 "-r", "__node:=zed_node", "-r", "__ns:=/zed", "-p", ["flip:=", flip]],
            # The system OpenCV is built for NumPy 1; a NumPy 2 in ~/.local would break it.
            additional_env={"PYTHONNOUSERSITE": "1"},
            output="screen",
            emulate_tty=True,
        ),

        # ---- YOLO + ByteTrack node on the ZED topics (runs with the venv's Python) ----
        ExecuteProcess(
            cmd=[VENV_PYTHON, YOLO_NODE, "--ros-args",
                 "-p", f"camera_topic:={RGB_TOPIC}",
                 "-p", f"depth_topic:={DEPTH_TOPIC}",
                 "-p", "camera_qos:=reliable",
                 # The ZED gives far fewer frames per second than the sim on a CPU: use every one.
                 "-p", "process_every:=1",
                 # 672x376 images: 640 keeps people small in the image (far away) detectable.
                 "-p", "imgsz:=640",
                 "-p", ["stale_timeout:=", stale_timeout],
                 "-p", ["show_window:=", show_window]],
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
