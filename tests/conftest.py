"""Test setup: make the repo importable, and stand in for ROS packages that aren't installed.

On the Ubuntu machine with ROS 2 Humble sourced, the real message types are used. Elsewhere (e.g. a
laptop without ROS), minimal stand-ins with the same fields let the logic be tested anyway. Only
packages that fail to import are replaced.

Run from the repo folder:
    ~/yolo_ros2_env/bin/python -m pytest tests          # after: source /opt/ros/humble/setup.bash
"""

import importlib
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ns = types.SimpleNamespace


def _missing(name):
    try:
        importlib.import_module(name)
        return False
    except ImportError:
        return True


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


class _Anything:
    """Accepts any constructor arguments, attribute access or call."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Anything()

    def __getattr__(self, name):
        return _Anything()


class _Msg:
    """Message stand-in: fields given as keyword arguments override the defaults."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _vec3():
    return ns(x=0.0, y=0.0, z=0.0)


class Header(_Msg):
    def __init__(self, **fields):
        self.frame_id = ""
        self.stamp = ns(sec=0, nanosec=0)
        super().__init__(**fields)


class Pose(_Msg):
    def __init__(self, **fields):
        self.position = _vec3()
        self.orientation = ns(x=0.0, y=0.0, z=0.0, w=1.0)
        super().__init__(**fields)


class PoseStamped(_Msg):
    def __init__(self, **fields):
        self.header = Header()
        self.pose = Pose()
        super().__init__(**fields)


class TransformStamped(_Msg):
    def __init__(self, **fields):
        self.header = Header()
        self.child_frame_id = ""
        self.transform = ns(translation=_vec3(), rotation=ns(x=0.0, y=0.0, z=0.0, w=1.0))
        super().__init__(**fields)


class String(_Msg):
    def __init__(self, data=""):
        self.data = data


class ObjectHypothesisWithPose(_Msg):
    def __init__(self, **fields):
        self.hypothesis = ns(class_id="", score=0.0)
        self.pose = ns(pose=Pose())
        super().__init__(**fields)


class Detection2D(_Msg):
    def __init__(self, **fields):
        self.header, self.id, self.results = Header(), "", []
        self.bbox = ns(center=ns(position=ns(x=0.0, y=0.0), theta=0.0), size_x=0.0, size_y=0.0)
        super().__init__(**fields)


class Detection3D(_Msg):
    def __init__(self, **fields):
        self.header, self.id, self.results = Header(), "", []
        self.bbox = ns(center=Pose(), size=_vec3())
        super().__init__(**fields)


class DetectionArray(_Msg):
    def __init__(self, **fields):
        self.header, self.detections = Header(), []
        super().__init__(**fields)


class State(_Msg):
    def __init__(self, **fields):
        self.connected, self.armed, self.mode = False, False, ""
        super().__init__(**fields)


class _Time:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def from_msg(message):
        return message


if _missing("rclpy"):
    _module("rclpy", init=lambda *a, **k: None, ok=lambda: False)
    _module("rclpy.node", Node=object)
    _module("rclpy.qos", HistoryPolicy=_Anything(), QoSProfile=_Anything,
            ReliabilityPolicy=_Anything(), qos_profile_sensor_data=None)
    _module("rclpy.duration", Duration=_Anything)
    _module("rclpy.time", Time=_Time)
if _missing("cv_bridge"):
    _module("cv_bridge", CvBridge=_Anything)
if _missing("rcl_interfaces.msg"):
    _module("rcl_interfaces")
    _module("rcl_interfaces.msg", ParameterDescriptor=_Anything)
if _missing("sensor_msgs.msg"):
    _module("sensor_msgs")
    _module("sensor_msgs.msg", Image=_Anything, CameraInfo=_Anything)
if _missing("std_msgs.msg"):
    _module("std_msgs")
    _module("std_msgs.msg", String=String, Header=Header)
if _missing("geometry_msgs.msg"):
    _module("geometry_msgs")
    _module("geometry_msgs.msg", PoseStamped=PoseStamped, TransformStamped=TransformStamped)
if _missing("vision_msgs.msg"):
    _module("vision_msgs")
    _module("vision_msgs.msg", Detection2D=Detection2D, Detection2DArray=DetectionArray,
            Detection3D=Detection3D, Detection3DArray=DetectionArray,
            ObjectHypothesisWithPose=ObjectHypothesisWithPose)
if _missing("mavros_msgs.msg"):
    _module("mavros_msgs")
    _module("mavros_msgs.msg", State=State)
if _missing("tf2_ros"):
    _module("tf2_ros", Buffer=_Anything, TransformListener=_Anything,
            TransformBroadcaster=_Anything)
if _missing("ultralytics"):
    _module("ultralytics", YOLO=_Anything)
