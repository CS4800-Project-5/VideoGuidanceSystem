#!/usr/bin/env python3

import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO


# Default camera: the Gazebo simulation. zed.launch.py sets camera_topic and depth_topic
# to the ZED's topics instead (same names as the ZED ROS 2 wrapper on the Jetson).
CAMERA_TOPIC = (
    "/world/default/model/x500_mono_cam_0/"
    "link/camera_link/sensor/camera/image"
)

# The tracker config sits next to this file in the repo, so the path works on any machine.
TRACKER_CONFIG = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "bytetrack_occlusion.yaml",
)


def person_distance(depth, box):
    """Distance in meters to a person, from a depth image in meters that lines up with the color
    image. Median over the torso (middle half of the box's width, 20-60% of its height), which
    avoids the background around arms and legs. None if there is too little valid depth there
    (stereo depth has holes)."""
    rows, cols = depth.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    c1, c2 = int(max(0, x1 + 0.25 * w)), int(min(cols, x2 - 0.25 * w + 1))
    r1, r2 = int(max(0, y1 + 0.2 * h)), int(min(rows, y1 + 0.6 * h + 1))
    values = depth[r1:r2, c1:c2]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 10:
        return None
    return float(np.median(values))


class YoloCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_camera_node")

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        # ---- Camera input ----
        self.camera_topic = self.declare_parameter("camera_topic", CAMERA_TOPIC).value
        # Depth image in meters that lines up pixel for pixel with the camera image (the ZED's
        # depth/depth_registered). Empty = no depth, no distances (the simulation camera).
        self.depth_topic = self.declare_parameter("depth_topic", "").value
        # "sensor_data" (best effort) for Gazebo; "reliable" for the ZED, whose ~1 MB images
        # get dropped whole over best effort.
        camera_qos = self.declare_parameter("camera_qos", "sensor_data").value

        # Run YOLO on one out of every `process_every` frames to reduce CPU usage.
        self.process_every = max(1, int(self.declare_parameter("process_every", 3).value))
        self.imgsz = int(self.declare_parameter("imgsz", 320).value)
        # OpenCV window; false on the drone, which has no monitor.
        self.show_window = bool(self.declare_parameter("show_window", True).value)
        self.frame_count = 0

        # Entry/exit tracking state.
        # A track is stale once it has not been seen for stale_timeout seconds.
        # Time comes from the image timestamps, so it works the same in live
        # sim and in rosbag replay, regardless of frame rate or frame skipping.
        # dynamic_typing: accept "5" as well as "5.0" (ROS 2 would otherwise reject an integer).
        self.stale_timeout = float(
            self.declare_parameter(
                "stale_timeout", 3.0, ParameterDescriptor(dynamic_typing=True)
            ).value
        )
        self.last_seen = {}            # track_id -> last image time it was seen (seconds)
        self.last_stamp = None         # time of the most recent image (seconds)
        self.last_frame_wall = None    # wall clock time the most recent image arrived

        # Processed frames per second, measured from image timestamps. Used to keep ByteTrack's
        # memory equal to stale_timeout (see set_track_buffer).
        self.fps = 0.0
        self.last_processed_stamp = None

        # Depth images by timestamp, and a color image waiting for its depth image.
        self.depth_images = {}
        self.pending = None
        self.last_depth_wall = None

        if camera_qos == "reliable":
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        else:
            qos = qos_profile_sensor_data
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos,
        )
        if self.depth_topic:
            self.create_subscription(Image, self.depth_topic, self.depth_callback, qos)

        # If the camera stops sending images, image time stops moving and
        # tracks would never go stale. This timer catches that case.
        self.create_timer(0.5, self.check_for_stopped_camera)

        self.get_logger().info(
            f"YOLO object entry/exit tracker started on {self.camera_topic} "
            f"(stale timeout {self.stale_timeout:.1f}s"
            + (f", depth from {self.depth_topic}" if self.depth_topic else "") + ")"
        )

    @staticmethod
    def stamp_of(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def image_callback(self, message):
        stamp = self.stamp_of(message)

        # Time went backwards: the rosbag restarted or the sim was reset.
        if self.last_stamp is not None and stamp < self.last_stamp - 1.0:
            self.get_logger().warn("Image time went backwards, clearing all tracks")
            self.remove_all_tracks("stream restarted")
            self.reset_tracker()

        self.last_stamp = stamp
        self.last_frame_wall = time.monotonic()

        self.frame_count += 1
        if self.frame_count % self.process_every != 0:
            return

        if not self.depth_topic:
            self.process_image(message, None)
            return

        # The color and depth images of one frame carry the same timestamp. Process the pair
        # together; if the depth image hasn't arrived yet, wait for it (depth_callback).
        key = (message.header.stamp.sec, message.header.stamp.nanosec)
        if key in self.depth_images:
            self.process_image(message, self.depth_images[key])
        elif self.last_depth_wall is None or time.monotonic() - self.last_depth_wall > 1.0:
            self.process_image(message, None)   # no depth coming: still track, no distances
        else:
            self.pending = message

    def depth_callback(self, message):
        self.last_depth_wall = time.monotonic()
        key = (message.header.stamp.sec, message.header.stamp.nanosec)
        self.depth_images[key] = message
        while len(self.depth_images) > 5:
            self.depth_images.pop(next(iter(self.depth_images)))

        pending = self.pending
        if pending is not None and (pending.header.stamp.sec, pending.header.stamp.nanosec) == key:
            self.pending = None
            self.process_image(pending, message)

    def process_image(self, message, depth_message):
        stamp = self.stamp_of(message)

        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
            depth = None
            if depth_message is not None:
                depth = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")

            # YOLO detects people; ByteTrack assigns persistent IDs.
            results = self.model.track(
                frame,
                tracker=TRACKER_CONFIG,
                persist=True,
                imgsz=self.imgsz,
                conf=0.35,
                classes=[0],
                device="cpu",
                verbose=False,
            )
            self.set_track_buffer(stamp)

            boxes = results[0].boxes
            current_ids = set()
            distances = {}   # track_id -> meters (None where depth has a hole)

            if boxes is not None and boxes.id is not None:
                current_ids = set(boxes.id.int().cpu().tolist())
                if depth is not None:
                    for track_id, box in zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist()):
                        distances[track_id] = (box, person_distance(depth, box))

            # A newly observed ID means that an object entered the view.
            for track_id in current_ids:
                if track_id not in self.last_seen:
                    self.get_logger().info(
                        f"Object entered camera view: ID {track_id}"
                    )

                self.last_seen[track_id] = stamp

            # Remove tracks that have not been seen for stale_timeout seconds.
            for track_id, seen_at in list(self.last_seen.items()):
                if stamp - seen_at > self.stale_timeout:
                    self.get_logger().info(
                        f"Object left camera view: ID {track_id} "
                        f"(unseen for {stamp - seen_at:.1f}s)"
                    )
                    del self.last_seen[track_id]

            if self.show_window:
                detected_image = results[0].plot()
                self.draw_distances(detected_image, distances)
                cv2.imshow(
                    "YOLO Tracking (Ctrl+C in the terminal to stop)",
                    detected_image,
                )
                cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(
                f"Image processing error: {error}"
            )

    @staticmethod
    def draw_distances(image, distances):
        """Distance label under the top-left corner of each person's box."""
        for (x1, y1, _, _), meters in distances.values():
            label = f"{meters:.1f} m" if meters is not None else "? m"
            (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x, y = int(max(0, x1)), int(max(0, y1))
            cv2.rectangle(image, (x, y), (x + width + 6, y + height + 8), (0, 0, 0), -1)
            cv2.putText(image, label, (x + 3, y + height + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def set_track_buffer(self, stamp):
        """Keep ByteTrack's memory of a hidden object equal to stale_timeout seconds.

        ByteTrack counts how long an object has been gone in processed frames (track_buffer),
        but the frame rate differs a lot between the sim, a laptop and the Jetson. Converting
        stale_timeout to frames from the measured rate means an object that comes back within
        stale_timeout keeps its ID on any machine.
        """
        if self.last_processed_stamp is not None:
            dt = stamp - self.last_processed_stamp
            if 0 < dt < 5.0:
                self.fps = 1.0 / dt if self.fps == 0 else 0.8 * self.fps + 0.2 / dt
        self.last_processed_stamp = stamp

        trackers = getattr(getattr(self.model, "predictor", None), "trackers", None)
        if not trackers or self.fps <= 0:
            return
        frames = max(2, round(self.stale_timeout * self.fps))
        for tracker in trackers:
            # The attribute is called max_time_lost in older Ultralytics versions.
            for name in ("max_frames_lost", "max_time_lost"):
                if hasattr(tracker, name):
                    setattr(tracker, name, frames)

    def check_for_stopped_camera(self):
        if self.last_frame_wall is None or not self.last_seen:
            return

        if time.monotonic() - self.last_frame_wall > self.stale_timeout:
            self.remove_all_tracks("no camera frames")

    def remove_all_tracks(self, reason):
        for track_id in sorted(self.last_seen):
            self.get_logger().info(
                f"Object left camera view: ID {track_id} ({reason})"
            )
        self.last_seen.clear()

    def reset_tracker(self):
        # Clear ByteTrack's memory so old IDs cannot come back. IDs restart at 1.
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
        self.last_processed_stamp = None


def main(args=None):
    rclpy.init(args=args)
    node = YoloCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
