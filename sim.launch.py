"""
Launches PX4 + Gazebo + YOLO stack. Goated w/ da sauce

Usage (source ROS once in the terminal, then launch):
    source /opt/ros/humble/setup.bash
    ros2 launch ./sim.launch.py
    ros2 launch ./sim.launch.py software_render:=true mavros:=true qgc:=true
    ros2 launch ./sim.launch.py record:=true bag_name:=object_occlusion_test

Ctrl+C stops everything, and rosbag2 closes the bag cleanly.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

HOME = os.path.expanduser("~")

# Paths are relative to whoever runs this, so it works on every teammate's machine.
PX4_DIR = os.path.join(HOME, "PX4-Autopilot")
VENV_PYTHON = os.path.join(HOME, "yolo_ros2_env", "bin", "python")
YOLO_NODE = os.path.join(HOME, "yolo_camera_node.py")
QGC_APPIMAGE = os.path.join(HOME, "Downloads", "QGroundControl-4.4.3.AppImage")

CAMERA_TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
FCU_URL = "udp://127.0.0.1:14540@127.0.0.1:14580"


def generate_launch_description():
    software_render = LaunchConfiguration("software_render")
    px4 = LaunchConfiguration("px4")
    mavros = LaunchConfiguration("mavros")
    qgc = LaunchConfiguration("qgc")
    yolo = LaunchConfiguration("yolo")
    viewer = LaunchConfiguration("viewer")
    record = LaunchConfiguration("record")
    bag_name = LaunchConfiguration("bag_name")

    return LaunchDescription([
        # ---- Launch arguments ----
        DeclareLaunchArgument("software_render", default_value="false",
                              description="Use Mesa software rendering (slower, avoids GPU issues)"),
        DeclareLaunchArgument("px4", default_value="true",
                              description="Start PX4 SITL + Gazebo"),
        DeclareLaunchArgument("mavros", default_value="false",
                              description="Start MAVROS bridge to PX4"),
        DeclareLaunchArgument("qgc", default_value="false",
                              description="Start QGroundControl"),
        DeclareLaunchArgument("yolo", default_value="true",
                              description="Start the YOLO + ByteTrack node"),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="Show the raw camera feed with image_tools"),
        DeclareLaunchArgument("record", default_value="false",
                              description="Record camera + /rosout with rosbag2"),
        DeclareLaunchArgument("bag_name", default_value="object_tracking_test",
                              description="Output folder name for the bag"),

        # ---- Environment ----
        UnsetEnvironmentVariable("RMW_IMPLEMENTATION"),
        GroupAction(
            scoped=False,  # let these variables apply to everything launched below
            condition=IfCondition(software_render),
            actions=[
                SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
                SetEnvironmentVariable("GALLIUM_DRIVER", "llvmpipe"),
                SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3"),
            ],
        ),

        # ---- Terminal 1: PX4 SITL + Gazebo ----
        ExecuteProcess(
            cmd=["make", "px4_sitl", "gz_x500_mono_cam"],
            cwd=PX4_DIR,
            output="screen",
            emulate_tty=True,
            condition=IfCondition(px4),
        ),

        # ---- Terminal 2: Gazebo -> ROS 2 image bridge ----
        Node(
            package="ros_gz_image",
            executable="image_bridge",
            arguments=[CAMERA_TOPIC],
            parameters=[{"qos": "sensor_data"}],
            output="screen",
        ),

        # ---- Terminal 3: YOLO + ByteTrack node (runs with the venv's Python) ----
        ExecuteProcess(
            cmd=[VENV_PYTHON, YOLO_NODE],
            output="screen",
            emulate_tty=True,
            condition=IfCondition(yolo),
        ),

        # ---- Optional: raw camera viewer ----
        Node(
            package="image_tools",
            executable="showimage",
            remappings=[("image", CAMERA_TOPIC)],
            output="screen",
            condition=IfCondition(viewer),
        ),

        # ---- Optional: MAVROS ----
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("mavros"), "launch", "px4.launch"])
            ),
            launch_arguments={
                "fcu_url": FCU_URL,
                "tgt_system": "1",
                "tgt_component": "1",
            }.items(),
            condition=IfCondition(mavros),
        ),

        # ---- Optional: QGroundControl (run `chmod +x` on the AppImage once first) ----
        ExecuteProcess(
            cmd=[QGC_APPIMAGE],
            output="log",
            condition=IfCondition(qgc),
        ),

        # ---- Optional: rosbag2 recording ----
        ExecuteProcess(
            cmd=["ros2", "bag", "record", CAMERA_TOPIC, "/rosout", "-o", bag_name],
            output="screen",
            condition=IfCondition(record),
        ),
    ])
