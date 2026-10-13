#!/usr/bin/env python3

import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO


CAMERA_TOPIC = (
    "/world/default/model/x500_mono_cam_0/"
    "link/camera_link/sensor/camera/image"
)

# The tracker config sits next to this file in the repo, so the path works on any machine.
TRACKER_CONFIG = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "bytetrack_occlusion.yaml",
)


class YoloCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_camera_node")

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        # Process one out of every three frames to reduce CPU usage.
        self.frame_count = 0

        # Entry/exit tracking state.
        # A track is stale once it has not been seen for stale_timeout seconds.
        # Time comes from the image timestamps, so it works the same in live
        # sim and in rosbag replay, regardless of frame rate or frame skipping.
        self.stale_timeout = float(
            self.declare_parameter("stale_timeout", 3.0).value
        )
        self.last_seen = {}            # track_id -> last image time it was seen (seconds)
        self.last_stamp = None         # time of the most recent image (seconds)
        self.last_frame_wall = None    # wall clock time the most recent image arrived

        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            qos_profile_sensor_data,
        )

        # If the camera stops sending images, image time stops moving and
        # tracks would never go stale. This timer catches that case.
        self.create_timer(0.5, self.check_for_stopped_camera)

        self.get_logger().info(
            f"YOLO object entry/exit tracker started "
            f"(stale timeout {self.stale_timeout:.1f}s)"
        )

    def image_callback(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

        # Time went backwards: the rosbag restarted or the sim was reset.
        if self.last_stamp is not None and stamp < self.last_stamp - 1.0:
            self.get_logger().warn("Image time went backwards, clearing all tracks")
            self.remove_all_tracks("stream restarted")
            self.reset_tracker()

        self.last_stamp = stamp
        self.last_frame_wall = time.monotonic()

        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )

            # YOLO detects people; ByteTrack assigns persistent IDs.
            results = self.model.track(
                frame,
                tracker=TRACKER_CONFIG,
                persist=True,
                imgsz=320,
                conf=0.35,
                classes=[0],
                device="cpu",
                verbose=False,
            )

            boxes = results[0].boxes
            current_ids = set()

            if boxes is not None and boxes.id is not None:
                current_ids = set(boxes.id.int().cpu().tolist())

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

            detected_image = results[0].plot()
            cv2.imshow(
                "PX4 Camera - YOLO Tracking",
                detected_image,
            )
            cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(
                f"Image processing error: {error}"
            )

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