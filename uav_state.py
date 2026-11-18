"""UAV pose and state from PX4 through MAVROS.

Subscribes to MAVROS (namespace set by the node's mavros_ns parameter, default /mavros):

  <ns>/state                 mavros_msgs/State: connected, armed, flight mode
  <ns>/local_position/pose   geometry_msgs/PoseStamped: position (m) and orientation in the local
                             ENU frame ("map"), from PX4's own position estimate

The node uses it for the status log and the CSV log, and can publish the pose to TF
(publish_tf: map -> base_link), so world positions (the node's world_frame parameter) work with
only MAVROS running. Leave publish_tf off when something else already publishes that TF (e.g. the
ZED wrapper's positional tracking on the Jetson), because a frame can only have one parent.

Missing or late MAVROS messages never stop tracking: the node logs a warning once when the pose or
state goes stale, and again when it comes back.
"""

import math
import time

from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.qos import qos_profile_sensor_data

try:
    from mavros_msgs.msg import State
except ImportError:   # mavros_msgs not installed
    State = None

try:
    from tf2_ros import TransformBroadcaster
except ImportError:
    TransformBroadcaster = None


def yaw_of(q):
    """Heading in degrees (ENU: 0 = east, 90 = north) from a quaternion."""
    return math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class UavState:
    """Latest UAV pose and state from MAVROS, with how fresh they are."""

    def __init__(self, node, namespace="/mavros", stale_after=1.0, publish_tf=False,
                 base_frame="base_link"):
        self.logger = node.get_logger()
        self.stale_after = stale_after
        self.base_frame = base_frame
        self.pose = None            # geometry_msgs/PoseStamped
        self.state = None           # mavros_msgs/State
        self.pose_wall = self.state_wall = None
        self.reported_stale = {"pose": False, "state": False}

        ns = "/" + namespace.strip("/")
        # Best effort subscriptions receive from both best-effort and reliable publishers.
        node.create_subscription(PoseStamped, ns + "/local_position/pose", self.pose_callback,
                                 qos_profile_sensor_data)
        if State is not None:
            node.create_subscription(State, ns + "/state", self.state_callback,
                                     qos_profile_sensor_data)
        else:
            self.logger.warn("mavros_msgs is not installed, so the UAV state (armed, mode) is "
                             "unknown. Install it with: sudo apt install ros-humble-mavros-msgs")

        self.tf = None
        if publish_tf:
            if TransformBroadcaster is None:
                self.logger.warn("tf2_ros is not installed, so the UAV pose is not published to TF")
            else:
                self.tf = TransformBroadcaster(node)
        self.logger.info(f"Reading UAV pose and state from {ns}"
                         + (f", publishing it to TF ({base_frame})" if self.tf else ""))

    def pose_callback(self, message):
        self.pose = message
        self.pose_wall = time.monotonic()
        if self.tf is not None:
            t = TransformStamped()
            t.header = message.header                 # frame_id "map" (MAVROS local frame)
            t.child_frame_id = self.base_frame
            p = message.pose.position
            t.transform.translation.x, t.transform.translation.y = p.x, p.y
            t.transform.translation.z = p.z
            t.transform.rotation = message.pose.orientation
            self.tf.sendTransform(t)

    def state_callback(self, message):
        self.state = message
        self.state_wall = time.monotonic()

    def fresh(self, which):
        """Whether the latest 'pose' or 'state' arrived within stale_after seconds."""
        wall = self.pose_wall if which == "pose" else self.state_wall
        return wall is not None and time.monotonic() - wall <= self.stale_after

    def check(self):
        """Warn once when the pose or state goes stale (or never arrives), and once when it is
        back. Call it regularly (the node does, from a timer)."""
        for which in ("pose", "state"):
            if which == "state" and State is None:
                continue
            fresh = self.fresh(which)
            if not fresh and not self.reported_stale[which]:
                self.reported_stale[which] = True
                what = "never received" if (self.pose if which == "pose" else self.state) is None \
                    else f"none for over {self.stale_after:g}s"
                self.logger.warn(f"UAV {which} from MAVROS: {what}. Tracking continues without it.")
            elif fresh and self.reported_stale[which]:
                self.reported_stale[which] = False
                self.logger.info(f"UAV {which} from MAVROS is back")

    def current_pose(self):
        """(x, y, z, yaw in degrees) if the pose is fresh, else None."""
        if not self.fresh("pose"):
            return None
        p, q = self.pose.pose.position, self.pose.pose.orientation
        return (p.x, p.y, p.z, yaw_of(q))

    def summary(self):
        """Short text for the status log line."""
        parts = []
        if self.fresh("state"):
            s = self.state
            parts.append(("armed" if s.armed else "disarmed") + (f" {s.mode}" if s.mode else "")
                         + ("" if s.connected else " (PX4 not connected)"))
        pose = self.current_pose()
        if pose is not None:
            parts.append(f"at ({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}) m, heading {pose[3]:.0f}°")
        return "UAV " + ", ".join(parts) if parts else "no UAV data"
