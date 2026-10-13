#!/usr/bin/env python3

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


class YoloCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_camera_node")

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        # Process one out of every three frames to reduce CPU usage.
        self.frame_count = 0

        # Entry/exit tracking state.
        self.active_ids = set()
        self.missing_frames = {}
        self.max_missing_frames = 15

        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "YOLO object entry/exit tracker started"
        )

    def image_callback(self, message):
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
                tracker="/home/jinbo/bytetrack_occlusion.yaml",
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
                if track_id not in self.active_ids:
                    self.get_logger().info(
                        f"Object entered camera view: ID {track_id}"
                    )

                self.active_ids.add(track_id)
                self.missing_frames[track_id] = 0

            # Keep a track temporarily when a detection is missing.
            for track_id in list(self.active_ids):
                if track_id not in current_ids:
                    self.missing_frames[track_id] = (
                        self.missing_frames.get(track_id, 0) + 1
                    )

                    if (
                        self.missing_frames[track_id]
                        >= self.max_missing_frames
                    ):
                        self.get_logger().info(
                            f"Object left camera view: ID {track_id}"
                        )
                        self.active_ids.remove(track_id)
                        del self.missing_frames[track_id]

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
