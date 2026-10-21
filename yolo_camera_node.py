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

from reidentifier import ColorHistogramExtractor, ReIdentifier


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

        # "cpu" in the sim; "cuda" (or "0") on the Jetson.
        self.device = self.declare_parameter("device", "cpu").value

        # Run YOLO on one out of every `process_every` frames. 1 = every frame;
        # raise it to save CPU if the camera publishes faster than YOLO keeps up.
        self.process_every = max(1, int(self.declare_parameter("process_every", 1).value))
        self.frame_count = 0

        # Entry/exit tracking state.
        # A track is stale once it has not been seen for stale_timeout seconds.
        # Time comes from the image timestamps, so it works the same in live
        # sim and in rosbag replay, regardless of frame rate or frame skipping.
        self.stale_timeout = float(
            self.declare_parameter("stale_timeout", 3.0).value
        )
        self.last_seen = {}            # object_id -> last image time it was seen (seconds)
        self.last_stamp = None         # time of the most recent image (seconds)
        self.last_frame_wall = None    # wall clock time the most recent image arrived

        # Re-identification: an object that leaves and comes back within
        # reid_memory seconds gets its old ID, matched by appearance.
        self.reid = ReIdentifier(
            ColorHistogramExtractor(),
            memory_seconds=float(self.declare_parameter("reid_memory", 60.0).value),
            match_threshold=float(self.declare_parameter("reid_threshold", 0.7).value),
        )

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
            f"(stale timeout {self.stale_timeout:.1f}s, "
            f"re-ID memory {self.reid.memory_seconds:.0f}s, device {self.device})"
        )

    def image_callback(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

        # Time went backwards: the rosbag restarted or the sim was reset.
        if self.last_stamp is not None and stamp < self.last_stamp - 1.0:
            self.get_logger().warn("Image time went backwards, clearing all tracks")
            self.remove_all_tracks("stream restarted")
            self.reset_tracker()
            self.reid.reset()

        self.last_stamp = stamp
        self.last_frame_wall = time.monotonic()

        self.frame_count += 1
        if self.frame_count % self.process_every != 0:
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
                device=self.device,
                verbose=False,
            )

            boxes = results[0].boxes
            detections = []    # (object_id, xyxy, confidence)

            if boxes is not None and boxes.id is not None:
                tracker_ids = boxes.id.int().cpu().tolist()
                corners = boxes.xyxy.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for tracker_id, (x1, y1, x2, y2), confidence in zip(
                    tracker_ids, corners, confidences
                ):
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    object_id, how = self.reid.assign(tracker_id, crop, stamp)
                    detections.append((object_id, (x1, y1, x2, y2), confidence))

                    # A newly observed ID means that an object entered the view.
                    if object_id not in self.last_seen:
                        note = " (re-identified)" if how in ("reidentified", "returned") else ""
                        self.get_logger().info(
                            f"Object entered camera view: ID {object_id}{note}"
                        )

                    self.last_seen[object_id] = stamp

            # Remove tracks that have not been seen for stale_timeout seconds.
            for object_id, seen_at in list(self.last_seen.items()):
                if stamp - seen_at > self.stale_timeout:
                    self.get_logger().info(
                        f"Object left camera view: ID {object_id} "
                        f"(unseen for {stamp - seen_at:.1f}s)"
                    )
                    del self.last_seen[object_id]
                    self.reid.depart(object_id, seen_at)

            cv2.imshow(
                "PX4 Camera - YOLO Tracking",
                self.draw_detections(frame, detections),
            )
            cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(
                f"Image processing error: {error}"
            )

    def draw_detections(self, frame, detections):
        # Drawn by hand instead of results.plot(), which would show ByteTrack's
        # short-lived IDs rather than the re-identified object IDs.
        image = frame.copy()
        for object_id, (x1, y1, x2, y2), confidence in detections:
            label = f"id:{object_id} person {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - height - 8), (x1 + width + 4, y1), (255, 0, 0), -1)
            cv2.putText(image, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image

    def check_for_stopped_camera(self):
        if self.last_frame_wall is None or not self.last_seen:
            return

        if time.monotonic() - self.last_frame_wall > self.stale_timeout:
            self.remove_all_tracks("no camera frames")

    def remove_all_tracks(self, reason):
        for object_id in sorted(self.last_seen):
            self.get_logger().info(
                f"Object left camera view: ID {object_id} ({reason})"
            )
            self.reid.depart(object_id, self.last_seen[object_id])
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