#!/usr/bin/env python3
"""
Compare YOLO models and input sizes on a recorded rosbag, to pick the best trade-off between
detecting people and speed on this computer.

    source /opt/ros/humble/setup.bash
    ~/yolo_ros2_env/bin/python compare_models.py ~/bag_files/class1
    ~/yolo_ros2_env/bin/python compare_models.py ~/bag_files/class1 --models yolov8n.pt yolo11s.pt --imgsz 640 960

Record the bag with `zed-demo record:=true bag_name:=class1`. For each model and input size it runs
YOLO (people only) on frames spread over the whole bag and prints how many people it finds per
frame, at 25% and at 50% confidence, their average confidence, and the time per frame. Missing
weights are downloaded (needs internet) into the repo folder, where the launch files find them.
"""

import argparse
import os
import time

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from ultralytics import YOLO

HERE = os.path.dirname(os.path.realpath(__file__))
ZED_RGB = "/zed/zed_node/rgb/color/rect/image"


def to_array(msg):
    """sensor_msgs/Image (bgr8 or rgb8) -> BGR NumPy array."""
    channels = 3
    data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = data[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == "rgb8":
        image = image[:, :, ::-1]
    elif msg.encoding != "bgr8":
        raise SystemExit(f"Unsupported image encoding {msg.encoding}")
    return np.ascontiguousarray(image)


def read_frames(bag, topic, count):
    """`count` images spread evenly over the bag."""
    with open(os.path.join(bag, "metadata.yaml")) as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    total = next((t["message_count"] for t in info["topics_with_message_count"]
                  if t["topic_metadata"]["name"] == topic), 0)
    if total == 0:
        names = [t["topic_metadata"]["name"] for t in info["topics_with_message_count"]]
        raise SystemExit(f"No messages on {topic} in {bag}. Topics in the bag: {', '.join(names)}")
    wanted = set(np.linspace(0, total - 1, min(count, total)).astype(int).tolist())

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id=info.get("storage_identifier", "sqlite3")),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    frames, index = [], 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        if index in wanted:
            frames.append(to_array(deserialize_message(data, Image)))
        index += 1
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="rosbag folder, e.g. ~/bag_files/class1")
    parser.add_argument("--topic", default=ZED_RGB)
    parser.add_argument("--frames", type=int, default=40, help="frames to test (spread over the bag)")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolo11s.pt", "yolov8s.pt"])
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 960])
    args = parser.parse_args()

    frames = read_frames(os.path.abspath(os.path.expanduser(args.bag)), args.topic, args.frames)
    h, w = frames[0].shape[:2]
    print(f"{len(frames)} frames of {w}x{h} from {args.bag}\n")
    os.chdir(HERE)   # weights live in (and download to) the repo folder

    print(f"{'model':<12} {'imgsz':>5} {'people/frame':>13} {'>=50%':>6} {'avg conf':>9} {'ms/frame':>9} {'max fps':>8}")
    for model_name in args.models:
        model = YOLO(model_name)
        for imgsz in args.imgsz:
            model.predict(frames[0], classes=[0], imgsz=imgsz, device="cpu", verbose=False)  # warm-up
            counts, strong, confidences = [], [], []
            start = time.perf_counter()
            for frame in frames:
                conf = model.predict(frame, classes=[0], conf=0.25, imgsz=imgsz, device="cpu",
                                     verbose=False)[0].boxes.conf.cpu().numpy()
                counts.append(len(conf))
                strong.append(int((conf >= 0.5).sum()))
                confidences.extend(conf.tolist())
            ms = (time.perf_counter() - start) / len(frames) * 1000
            avg = np.mean(confidences) * 100 if confidences else 0
            print(f"{model_name:<12} {imgsz:>5} {np.mean(counts):>13.1f} {np.mean(strong):>6.1f} "
                  f"{avg:>8.0f}% {ms:>9.0f} {1000 / ms:>8.1f}")
    print("\nmax fps = YOLO alone; the depth computation takes CPU too, so zed-demo runs slower.")
    print("Pick with: zed-demo model:=<model> imgsz:=<imgsz>")


if __name__ == "__main__":
    main()
