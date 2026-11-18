"""ROS 2 topics with the people the node tracks, for the other groups (collision avoidance,
path planning, mapping) to subscribe to.

Topics (prefix set by the node's topic_prefix parameter, default /perception):

  <prefix>/tracks         vision_msgs/Detection2DArray, every processed frame (empty when no one
                          is in view). One Detection2D per confirmed person:
                            id                        their ID ("3"), the same one shown and logged
                            bbox                      box in image pixels (center + size)
                            results[0].hypothesis     class_id "person", score = YOLO confidence
                          header = the camera image's header (same stamp and frame_id).

  <prefix>/tracks_3d      vision_msgs/Detection3DArray, every processed frame, only when the node
                          has depth and camera_info (the ZED). One Detection3D per person whose
                          distance is known:
                            id                        their ID
                            bbox.center.position      position in meters in the camera's optical
                                                      frame (x right, y down, z forward), torso
                            bbox.size                 approximate width, height (and depth) in m
                            results[0].hypothesis     class_id "person", score
                          header.frame_id = the camera's frame, so tf2 can transform the positions
                          into the drone's or the map's frame.

  <prefix>/track_events   std_msgs/String holding JSON, one message per event, for example
                          {"event": "entered", "id": 3, "stamp": 1700000000.12,
                           "how": "reidentified", "similarity": 0.81}
                          {"event": "left", "id": 3, "stamp": 1700000004.5,
                           "reason": "unseen for 3.2s"}

vision_msgs is a standard ROS 2 package (sudo apt install ros-humble-vision-msgs). Without it the
node still publishes track_events and logs a warning.
"""

import json

from std_msgs.msg import String

try:
    from vision_msgs.msg import (
        Detection2D,
        Detection2DArray,
        Detection3D,
        Detection3DArray,
        ObjectHypothesisWithPose,
    )
except ImportError:   # vision_msgs not installed
    Detection2DArray = None


def _hypothesis(score):
    result = ObjectHypothesisWithPose()
    if hasattr(result, "hypothesis"):         # Humble and newer
        result.hypothesis.class_id = "person"
        result.hypothesis.score = float(score)
    else:                                     # Foxy / Galactic
        result.id = "person"
        result.score = float(score)
    return result


def _set_center_2d(bbox, x, y):
    center = bbox.center
    if hasattr(center, "position"):           # vision_msgs/Pose2D (Humble and newer)
        center.position.x, center.position.y = float(x), float(y)
    else:                                     # geometry_msgs/Pose2D (older)
        center.x, center.y = float(x), float(y)


class TrackPublisher:
    """Publishes the tracked people of each processed frame, and entered/left events."""

    def __init__(self, node, prefix="/perception"):
        prefix = "/" + prefix.strip("/")
        self.logger = node.get_logger()
        self.events = node.create_publisher(String, prefix + "/track_events", 50)
        self.tracks = self.tracks_3d = None
        if Detection2DArray is None:
            self.logger.warn("vision_msgs is not installed, so only track_events is published. "
                             "Install it with: sudo apt install ros-humble-vision-msgs")
            return
        self.tracks = node.create_publisher(Detection2DArray, prefix + "/tracks", 10)
        self.tracks_3d = node.create_publisher(Detection3DArray, prefix + "/tracks_3d", 10)
        self.logger.info(f"Publishing {prefix}/tracks, {prefix}/tracks_3d (with depth) "
                         f"and {prefix}/track_events")

    def publish_frame(self, header, people, with_3d):
        """people: (id, (x1, y1, x2, y2), confidence, position or None, size or None) per person,
        position = (x, y, z) and size = (width, height, depth) in meters in the camera frame.
        with_3d: whether 3D positions can be known this frame (depth and camera_info present)."""
        if self.tracks is None:
            return

        array = Detection2DArray(header=header)
        for person_id, (x1, y1, x2, y2), confidence, _, _ in people:
            detection = Detection2D(header=header, id=str(person_id))
            _set_center_2d(detection.bbox, (x1 + x2) / 2, (y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            detection.results.append(_hypothesis(confidence))
            array.detections.append(detection)
        self.tracks.publish(array)

        if not with_3d:
            return
        array = Detection3DArray(header=header)
        for person_id, _, confidence, position, size in people:
            if position is None:
                continue
            detection = Detection3D(header=header, id=str(person_id))
            center = detection.bbox.center
            center.position.x, center.position.y, center.position.z = (float(v) for v in position)
            center.orientation.w = 1.0
            detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = (
                float(v) for v in size)
            result = _hypothesis(confidence)
            result.pose.pose = center
            detection.results.append(result)
            array.detections.append(detection)
        self.tracks_3d.publish(array)

    def publish_event(self, event, person_id, stamp, **details):
        """event: "entered" or "left". details: extra JSON fields (None values are left out)."""
        message = {"event": event, "id": int(person_id), "stamp": round(float(stamp), 3)}
        message.update({k: v for k, v in details.items() if v is not None})
        self.events.publish(String(data=json.dumps(message)))
