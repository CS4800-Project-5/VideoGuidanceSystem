#!/usr/bin/env python3
"""Replay a recorded rosbag through the node and check the performance report.

Repeatable test on real footage: the same recording gives comparable numbers before and after a
change. Record once (zed-demo record:=true bag_name:=class1, or sim.launch.py record:=true), then:

    source /opt/ros/humble/setup.bash
    python3 replay_test.py class1 --people 3                         # report only
    python3 replay_test.py class1 --people 3 --max-ids-per-person 1.5 --min-fps 2
    python3 replay_test.py class1 --param reid_threshold:=0.6        # try a node setting
    python3 replay_test.py class1 --rate 0.5                         # play at half speed

The bag type (ZED or simulation) is detected from its topics, and the node gets the same settings
as zed.launch.py or sim.launch.py, without a window and with a temporary CSV log. Exits with 1 if a
limit is missed (limits as in evaluate_tracking.py).
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

import evaluate_tracking

HERE = os.path.dirname(os.path.realpath(__file__))
VENV_PYTHON = os.path.expanduser("~/yolo_ros2_env/bin/python")

ZED = "/zed/zed_node/"
ZED_TOPICS = [ZED + "rgb/color/rect/image", ZED + "depth/depth_registered",
              ZED + "rgb/color/rect/camera_info"]
SIM_TOPICS = ["/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"]

ZED_PARAMS = [f"camera_topic:={ZED_TOPICS[0]}", f"depth_topic:={ZED_TOPICS[1]}",
              f"camera_info_topic:={ZED_TOPICS[2]}", "camera_qos:=reliable",
              "model:=yolo11s.pt", "imgsz:=640",
              f"tracker:={os.path.join(HERE, 'botsort_zed.yaml')}"]
SIM_PARAMS = [f"tracker:={os.path.join(HERE, 'botsort_sim.yaml')}"]


def bag_topics(bag):
    info = subprocess.run(["ros2", "bag", "info", bag], capture_output=True, text=True)
    if info.returncode != 0:
        sys.exit(f"ros2 bag info failed (is ROS sourced? is {bag} a bag folder?):\n{info.stderr}")
    return info.stdout


def main():
    parser = argparse.ArgumentParser(description="Replay a rosbag through the node and evaluate")
    parser.add_argument("bag", help="bag folder")
    parser.add_argument("--rate", type=float, default=1.0, help="playback speed (default 1.0)")
    parser.add_argument("--param", action="append", default=[],
                        help="extra node parameter, e.g. reid_threshold:=0.6 (repeatable)")
    parser.add_argument("--people", type=int, help="number of different real people in the bag")
    parser.add_argument("--min-fps", type=float)
    parser.add_argument("--max-latency-p95", type=float)
    parser.add_argument("--max-ids-per-person", type=float)
    parser.add_argument("--keep-csv", help="also save the CSV log here")
    args = parser.parse_args()

    if not os.environ.get("ROS_DISTRO"):
        sys.exit("Source ROS first: source /opt/ros/humble/setup.bash")
    info = bag_topics(args.bag)
    if ZED_TOPICS[0] in info:
        topics, params, kind = [t for t in ZED_TOPICS if t in info], ZED_PARAMS, "ZED"
    elif SIM_TOPICS[0] in info:
        topics, params, kind = SIM_TOPICS, SIM_PARAMS, "simulation"
    else:
        sys.exit(f"No ZED or simulation camera topic in {args.bag}")

    csv_path = args.keep_csv or os.path.join(tempfile.mkdtemp(), "replay.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    params = params + ["show_window:=false", "uav:=false", f"csv_log:={csv_path}", *args.param]
    node_cmd = [VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable,
                os.path.join(HERE, "yolo_camera_node.py"), "--ros-args"]
    for p in params:
        node_cmd += ["-p", p]
    env = dict(os.environ, ROS_LOCALHOST_ONLY="1")   # keep other ROS traffic out

    print(f"Replaying {args.bag} ({kind} bag) at {args.rate}x")
    node = subprocess.Popen(node_cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    ready = threading.Event()

    def show_node_output():
        for line in node.stdout:
            if "tracker started" in line:
                ready.set()
            if any(word in line for word in ("WARN", "ERROR", "entered", "left", "Status")):
                print("  node: " + line.rstrip())

    threading.Thread(target=show_node_output, daemon=True).start()
    if not ready.wait(120):
        node.kill()
        sys.exit("The node did not start within 120 s (see its output above)")
    time.sleep(2)

    try:
        subprocess.run(["ros2", "bag", "play", args.bag, "--rate", str(args.rate),
                        "--topics", *topics], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)                                  # let the last frames finish
    finally:
        node.send_signal(signal.SIGINT)
        try:
            node.wait(20)
        except subprocess.TimeoutExpired:
            node.kill()

    if not os.path.exists(csv_path):
        sys.exit("The node wrote no CSV log")
    with open(csv_path, newline="", encoding="utf-8") as f:
        runs = evaluate_tracking.split_runs(list(csv.DictReader(f)))
    if not runs:
        sys.exit("The node logged nothing: did the camera topic in the bag reach it?")
    result = evaluate_tracking.evaluate(runs[-1], args.people)
    print()
    evaluate_tracking.print_report(result, 1, 1)

    failures = evaluate_tracking.check_limits(result, args.min_fps, args.max_latency_p95,
                                              args.max_ids_per_person)
    print("\n" + ("FAILED: " + "; ".join(failures) if failures else "PASSED"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
