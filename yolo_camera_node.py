#!/usr/bin/env python3

import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

try:
    from tf2_ros import Buffer, TransformListener
except ImportError:   # tf2_ros missing: no world positions, re-ID on appearance only
    Buffer = TransformListener = None

from reidentifier import ReIdentifier, make_extractor, mean_feature
from track_publisher import TrackPublisher
from tracking_logger import TrackingLogger


# Default camera: the Gazebo simulation. zed.launch.py sets camera_topic and depth_topic
# to the ZED's topics instead (same names as the ZED ROS 2 wrapper on the Jetson).
CAMERA_TOPIC = (
    "/world/default/model/x500_mono_cam_0/"
    "link/camera_link/sensor/camera/image"
)

HERE = os.path.dirname(os.path.realpath(__file__))

# BoT-SORT settings. The tracker config sits next to this file in the repo, so the path works on
# any machine. BoT-SORT compensates for the camera's own motion (handheld, or on the drone);
# sim.launch.py passes botsort_sim.yaml instead.
TRACKER_CONFIG = os.path.join(HERE, "botsort_zed.yaml")


def box_overlap(box, others):
    """Largest IoU between a box and a list of other boxes (0 if none)."""
    best = 0.0
    x1, y1, x2, y2 = box
    for ox1, oy1, ox2, oy2 in others:
        iw, ih = min(x2, ox2) - max(x1, ox1), min(y2, oy2) - max(y1, oy1)
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        best = max(best, inter / ((x2 - x1) * (y2 - y1) + (ox2 - ox1) * (oy2 - oy1) - inter))
    return best


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
        # YOLO weights: yolov8n.pt (smallest, fastest) by default. zed.launch.py uses yolo11s.pt,
        # which finds far more small / far-away people at about twice the CPU time. A file name
        # is looked up in the working directory (the repo folder) and downloaded there if missing.
        self.model = YOLO(self.declare_parameter("model", "yolov8n.pt").value)
        # "cpu" in the sim and on a laptop; "cuda" (or "0") on the Jetson.
        self.device = self.declare_parameter("device", "cpu").value

        # ---- Camera input ----
        self.camera_topic = self.declare_parameter("camera_topic", CAMERA_TOPIC).value
        # Depth image in meters that lines up pixel for pixel with the camera image (the ZED's
        # depth/depth_registered). Empty = no depth, no distances (the simulation camera).
        self.depth_topic = self.declare_parameter("depth_topic", "").value
        # "sensor_data" (best effort) for Gazebo; "reliable" for the ZED, whose ~1 MB images
        # get dropped whole over best effort.
        camera_qos = self.declare_parameter("camera_qos", "sensor_data").value
        # Camera intrinsics (sensor_msgs/CameraInfo) for the color image. With depth, they turn
        # each person's box and distance into a 3D position for the tracks_3d topic. Empty = no
        # 3D positions. zed.launch.py sets the ZED's rgb/color/rect/camera_info.
        self.camera_info_topic = self.declare_parameter("camera_info_topic", "").value
        self.camera_info = None
        # Fixed frame (e.g. "map" or "odom") that TF connects to the camera's frame. With it, each
        # person's position is also known in that frame, which lets re-identification rule out
        # people who were last seen too far away. Empty = off (camera-frame positions only).
        self.world_frame = self.declare_parameter("world_frame", "").value
        self.tf_buffer = None
        self.tf_warned = False
        if self.world_frame:
            if Buffer is None:
                self.get_logger().warn("tf2_ros is not installed, so world_frame is ignored")
            else:
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(self.tf_buffer, self)

        # Run YOLO on one out of every `process_every` frames to reduce CPU usage. BoT-SORT
        # follows a moving camera best when it sees every frame.
        self.process_every = max(1, int(self.declare_parameter("process_every", 1).value))
        self.imgsz = int(self.declare_parameter("imgsz", 320).value)

        # ---- Tracking (BoT-SORT) ----
        # Tracker settings file: botsort_zed.yaml by default, botsort_sim.yaml in the simulation.
        self.tracker = self.declare_parameter("tracker", TRACKER_CONFIG).value
        # Detections below this confidence are dropped before tracking. BoT-SORT is designed to
        # get weak detections too (down to track_low_thresh, 0.1): it uses them only to keep
        # following people it already tracks (e.g. partly hidden), and starts new IDs only above
        # new_track_thresh.
        self.conf = float(self.declare_parameter(
            "conf", 0.1, ParameterDescriptor(dynamic_typing=True)).value)
        # A person gets an ID only after being tracked in this many processed frames in a row,
        # so detections that flicker for a frame or two don't use up IDs. IDs are numbered
        # 1, 2, 3, ... in the order people are confirmed (the tracker's own numbers skip
        # every short-lived candidate).
        self.min_hits = max(1, int(self.declare_parameter("min_hits", 3).value))
        self.display_ids = {}          # tracker id -> our id
        self.hits = {}                 # tracker id -> processed frames tracked so far
        self.next_display_id = 1

        # ---- Re-identification (reidentifier.py) ----
        # Remember what each person looks like, with no time limit, so someone who leaves the view
        # (or the camera leaving the room) gets their old ID back when they are seen again.
        # reid:=false turns it off.
        self.reid = None
        if bool(self.declare_parameter("reid", True).value):
            # OSNet weights (setup_zed.sh downloads both; the architecture comes from the file
            # name). Empty = by device: the small, fast osnet_x0_5 on a CPU, and the full-size
            # osnet_ibn_x1_0 on a GPU (it copes better with new viewpoints and lighting, but
            # takes about 10x the CPU time). Either way the other one is the fallback.
            small = os.path.join(HERE, "osnet_x0_5_msmt17.pth")
            large = os.path.join(HERE, "osnet_ibn_x1_0_msmt17.pth")
            reid_model = self.declare_parameter("reid_model", "").value
            on_gpu = str(self.device).lower() not in ("cpu", "")
            candidates = [reid_model] if reid_model else []
            candidates += [large, small] if on_gpu else [small, large]
            extractor = make_extractor(candidates, self.get_logger(),
                                       device="cuda" if on_gpu else "cpu")
            self.reid = ReIdentifier(
                extractor,
                # Similarity needed to count as the same person; -1 = the model's default
                # (OSNet 0.65, clothing colors 0.7). Lower = fewer new IDs for returning people,
                # but more risk of mixing up two people.
                threshold=float(self.declare_parameter(
                    "reid_threshold", -1.0, ParameterDescriptor(dynamic_typing=True)).value),
                # Seconds to remember someone who is gone; 0 = for as long as the program runs.
                memory_seconds=float(self.declare_parameter(
                    "reid_memory", 0.0, ParameterDescriptor(dynamic_typing=True)).value),
                # With world_frame: fastest a person is assumed to move (m/s). A remembered person
                # last seen farther away than they could have walked is not a candidate.
                max_speed=float(self.declare_parameter(
                    "reid_max_speed", 3.0, ParameterDescriptor(dynamic_typing=True)).value),
            )
            # Add a new appearance sample of each person every this many processed frames.
            self.reid_every = max(1, int(self.declare_parameter("reid_every", 5).value))
            # Only samples with at least this YOLO confidence are added. Kept low so views from
            # above or behind (where YOLO is less sure) still get remembered.
            self.reid_min_conf = float(self.declare_parameter(
                "reid_min_conf", 0.3, ParameterDescriptor(dynamic_typing=True)).value)
        self.track_features = {}       # tracker id -> appearance samples while being confirmed
        self.since_sample = {}         # tracker id -> processed frames since its last sample
        self.reid_events = {}          # our id -> (how, similarity, closest id), this frame

        # OpenCV window; false on the drone, which has no monitor.
        self.show_window = bool(self.declare_parameter("show_window", True).value)
        # Every this many seconds, log what each stage sees (YOLO -> tracker -> IDs), which shows
        # where a person gets lost when they are visible but get no box. 0 turns it off.
        self.status_period = float(self.declare_parameter(
            "status_period", 5.0, ParameterDescriptor(dynamic_typing=True)).value)
        self.last_status_wall = time.monotonic()
        self.detections = (0, 0, 0.0)  # YOLO, last frame: people >= 25%, weaker ones, best confidence
        # Runs right after YOLO and before the tracker (the tracker adds its own callback on the
        # first track() call, after this one), so it sees YOLO's raw detections.
        self.model.add_callback("on_predict_postprocess_end", self.count_detections)
        self.frame_count = 0

        # ---- Output for other groups and for analysis ----
        # ROS topics with every tracked person and entered/left events (track_publisher.py).
        self.publisher = TrackPublisher(
            self, self.declare_parameter("topic_prefix", "/perception").value)
        # CSV file with the same detections and events (tracking_logger.py). Empty = no CSV.
        csv_log = self.declare_parameter("csv_log", "~/yolo_tracking_log.csv").value
        self.csv = TrackingLogger(csv_log) if csv_log else None
        if self.csv is not None:
            self.get_logger().info(f"Logging detections and events to {self.csv.log_path}")

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

        # Processed frames per second, measured from image timestamps. Used to keep BoT-SORT's
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
        if self.camera_info_topic:
            self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos)

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

    def camera_info_callback(self, message):
        self.camera_info = message

    def camera_point(self, box, meters, image_width):
        """3D position (x right, y down, z forward, in m) of a person's torso in the camera's
        optical frame, and their approximate (width, height, depth) in m, from the box, the
        distance and the camera intrinsics. (None, None) without a distance or intrinsics."""
        info = self.camera_info
        if meters is None or info is None or info.k[0] <= 0:
            return None, None
        # The intrinsics are for info.width pixels; scale them if the image has another size.
        scale = image_width / info.width if info.width else 1.0
        fx, fy = info.k[0] * scale, info.k[4] * scale
        cx, cy = info.k[2] * scale, info.k[5] * scale
        x1, y1, x2, y2 = box
        u, v = (x1 + x2) / 2, y1 + 0.4 * (y2 - y1)   # torso center, where the distance is measured
        width, height = (x2 - x1) * meters / fx, (y2 - y1) * meters / fy
        position = ((u - cx) * meters / fx, (v - cy) * meters / fy, meters)
        return position, (width, height, min(width, 0.5))

    def to_world(self, header, position):
        """A camera-frame position in world_frame (via TF), or None if unknown."""
        if self.tf_buffer is None or position is None or not header.frame_id:
            return None
        try:
            try:
                t = self.tf_buffer.lookup_transform(self.world_frame, header.frame_id,
                                                    Time.from_msg(header.stamp))
            except Exception:
                # The image may be newer than the last transform: use the latest one.
                t = self.tf_buffer.lookup_transform(self.world_frame, header.frame_id, Time(),
                                                    timeout=Duration(seconds=0))
        except Exception as error:
            if not self.tf_warned:
                self.tf_warned = True
                self.get_logger().warn(f"No transform {header.frame_id} -> {self.world_frame} "
                                       f"({error}); re-ID matches on appearance only until there is")
            return None
        q, tr = t.transform.rotation, t.transform.translation
        # Rotate the point by the quaternion (v' = v + 2w(q x v) + 2 q x (q x v)), then translate.
        v = np.array(position, float)
        u = np.array([q.x, q.y, q.z])
        c = np.cross(u, v)
        v = v + 2 * q.w * c + 2 * np.cross(u, c)
        return (v[0] + tr.x, v[1] + tr.y, v[2] + tr.z)

    def process_image(self, message, depth_message):
        stamp = self.stamp_of(message)
        started = time.perf_counter()

        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
            depth = None
            if depth_message is not None:
                depth = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")

            # YOLO detects people; BoT-SORT follows them from frame to frame.
            results = self.model.track(
                frame,
                tracker=self.tracker,
                persist=True,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=[0],
                device=self.device,
                verbose=False,
            )
            self.set_track_buffer(stamp)

            # (our id, box, confidence, distance in m, camera-frame position, size), the last
            # three None when unknown, for each confirmed person
            people = self.confirmed_people(results[0].boxes, depth, frame, stamp, message.header)
            current_ids = {person[0] for person in people}
            boxes = results[0].boxes
            tracked = 0 if boxes is None or boxes.id is None else len(boxes.id)
            self.log_status(tracked, len(people))

            # Publish every person (with a 3D position when depth and intrinsics are known),
            # and log them to the CSV file.
            with_3d = depth is not None and self.camera_info is not None
            published = [(person_id, box, confidence, position, size)
                         for person_id, box, confidence, _, position, size in people]
            self.publisher.publish_frame(message.header, published, with_3d)
            processing_ms = (time.perf_counter() - started) * 1000
            if self.csv is not None:
                for person_id, box, confidence, position, _ in published:
                    self.csv.log_detection(self.frame_count, person_id, 0, "person", confidence,
                                           box, processing_ms, stamp=stamp, position=position)
                if not published:
                    self.csv.log_frame_without_detections(self.frame_count, processing_ms,
                                                          stamp=stamp)

            # A newly observed ID means that an object entered the view.
            for track_id in current_ids:
                if track_id not in self.last_seen:
                    self.get_logger().info(
                        f"Object entered camera view: ID {track_id}{self.reid_note(track_id)}"
                    )
                    self.track_event("entered", track_id, stamp, processing_ms)

                self.last_seen[track_id] = stamp

            # Remove tracks that have not been seen for stale_timeout seconds.
            for track_id, seen_at in list(self.last_seen.items()):
                if stamp - seen_at > self.stale_timeout:
                    reason = f"unseen for {stamp - seen_at:.1f}s"
                    self.get_logger().info(f"Object left camera view: ID {track_id} ({reason})")
                    self.track_event("left", track_id, stamp, processing_ms, reason)
                    del self.last_seen[track_id]
                    self.forget(track_id)

            if self.show_window:
                cv2.imshow(
                    "YOLO Tracking (Ctrl+C in the terminal to stop)",
                    self.draw(frame, people, depth is not None),
                )
                cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(
                f"Image processing error: {error}"
            )
            if self.csv is not None:
                self.csv.log_processing_error(self.frame_count, error)

    def track_event(self, event, display_id, stamp, processing_ms="", reason=None):
        """An entered/left event on the track_events topic and in the CSV file."""
        how, score, _ = self.reid_events.get(display_id, (None, None, None))
        if event != "entered":
            how = score = None
        self.publisher.publish_event(
            event, display_id, stamp, reason=reason, how=how,
            similarity=None if score is None or how != "reidentified" else round(score, 3))
        if self.csv is not None:
            self.csv.log_tracking_event(self.frame_count, event, display_id, processing_ms,
                                        stamp=stamp)

    def reid_note(self, display_id):
        """How re-identification decided an ID, for the entered log line."""
        if self.reid is None or display_id not in self.reid_events:
            return ""
        how, score, closest = self.reid_events[display_id]
        if how == "reidentified":
            return f" (re-identified by appearance, similarity {score:.2f})"
        if closest is None:
            return " (new person)"
        return (f" (new person; most similar remembered: ID {closest} at {score:.2f}, "
                f"needs {self.reid.threshold:.2f})")

    def count_detections(self, predictor):
        boxes = predictor.results[0].boxes
        confidences = [] if boxes is None else boxes.conf.cpu().tolist()
        strong = sum(1 for c in confidences if c >= 0.25)
        self.detections = (strong, len(confidences) - strong, max(confidences, default=0.0))

    def log_status(self, tracked, with_id):
        now = time.monotonic()
        if self.status_period <= 0 or now - self.last_status_wall < self.status_period:
            return
        self.last_status_wall = now
        strong, weak, best = self.detections
        self.get_logger().info(
            f"Status: YOLO sees {strong} people"
            + (f" (+{weak} weak, under 25%)" if weak else "")
            + (f", best {best * 100:.0f}%" if strong or weak else "")
            + f" -> tracker follows {tracked} -> {with_id} with an ID, {self.fps:.1f} fps"
            + (f", remembering {len(self.reid)} people" if self.reid is not None else "")
        )

    def confirmed_people(self, boxes, depth, frame, stamp, header):
        """The tracker's results with our IDs: only people tracked for min_hits frames in a row.

        With re-identification, a newly confirmed track is first compared with every remembered
        person who is not in view right now, and gets their ID if the appearance matches.
        """
        self.reid_events = {}
        rows = []
        if boxes is not None and boxes.id is not None:
            rows = list(zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist(),
                            boxes.conf.cpu().tolist()))

        # Where each tracked person is: distance, camera-frame position and size, world position.
        where = {}
        for tracker_id, box, _ in rows:
            meters = person_distance(depth, box) if depth is not None else None
            position, size = self.camera_point(box, meters, frame.shape[1])
            where[tracker_id] = (meters, position, size, self.to_world(header, position))

        # 1. Which tracks already have an ID, and which reach min_hits now.
        visible, confirming = set(), []
        for tracker_id, _, _ in rows:
            self.hits[tracker_id] = self.hits.get(tracker_id, 0) + 1
            if tracker_id in self.display_ids:
                visible.add(self.display_ids[tracker_id])
            elif self.hits[tracker_id] >= self.min_hits:
                confirming.append(tracker_id)

        # 2. Appearance samples, in one batch: tracks not confirmed yet (for matching), and
        #    confirmed people due for a new sample when the view of them is clear.
        if self.reid is not None and rows:
            self.sample_appearance(rows, frame, stamp)

        # 3. Give newly confirmed tracks an ID: a remembered person's, or a new one.
        if confirming:
            matches = {}
            if self.reid is not None:
                queries = {t: mean_feature(self.track_features.get(t, [])) for t in confirming}
                matches = self.reid.match(queries, exclude=visible,
                                          positions={t: where[t][3] for t in confirming},
                                          stamp=stamp)
            for tracker_id in confirming:
                person_id, score, closest = matches.get(tracker_id, (None, None, None))
                if person_id is None:
                    person_id, how = self.next_display_id, "new"
                    self.next_display_id += 1
                else:
                    how = "reidentified"
                    # Their old track (lost earlier) no longer stands for them.
                    for t in [t for t, d in self.display_ids.items() if d == person_id]:
                        del self.display_ids[t]
                self.display_ids[tracker_id] = person_id
                visible.add(person_id)
                self.reid_events[person_id] = (how, score, closest)
                if self.reid is not None:
                    for feature in self.track_features.pop(tracker_id, []):
                        self.reid.add(person_id, feature, stamp)

        # 4. The confirmed people in this frame.
        people = []
        seen = set()
        for tracker_id, box, confidence in rows:
            seen.add(tracker_id)
            if tracker_id not in self.display_ids:
                continue
            person_id = self.display_ids[tracker_id]
            meters, position, size, world = where[tracker_id]
            if self.reid is not None:
                self.reid.seen(person_id, stamp, world)
            people.append((person_id, box, confidence, meters, position, size))

        # Candidates must be seen in consecutive frames to count.
        for tracker_id in list(self.hits):
            if tracker_id not in seen and tracker_id not in self.display_ids:
                del self.hits[tracker_id]
                self.track_features.pop(tracker_id, None)
        if self.reid is not None:
            self.reid.expire(stamp)
        return people

    def sample_appearance(self, rows, frame, stamp):
        img_h, img_w = frame.shape[:2]
        wanted, crops = [], []
        for i, (tracker_id, box, confidence) in enumerate(rows):
            others = [b for j, (_, b, _) in enumerate(rows) if j != i]
            if tracker_id in self.display_ids:
                self.since_sample[tracker_id] = self.since_sample.get(tracker_id, self.reid_every) + 1
                # A clear view only: confident, and not overlapping someone else (whose clothes
                # would end up in this person's sample).
                if (self.since_sample[tracker_id] < self.reid_every
                        or confidence < self.reid_min_conf or box_overlap(box, others) > 0.2):
                    continue
                self.since_sample[tracker_id] = 0
            elif box_overlap(box, others) > 0.5:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            if crop.size:
                wanted.append(tracker_id)
                crops.append(crop)
        for tracker_id, feature in zip(wanted, self.reid.extractor.extract_batch(crops)):
            if tracker_id in self.display_ids:
                self.reid.add(self.display_ids[tracker_id], feature, stamp)
            elif feature is not None:
                samples = self.track_features.setdefault(tracker_id, [])
                samples.append(feature)
                del samples[:-5]   # the last 5 samples are enough

    def forget(self, display_id):
        """Drop the tracker ids behind one of our ids once it has gone stale."""
        for tracker_id in [t for t, d in self.display_ids.items() if d == display_id]:
            del self.display_ids[tracker_id]
            self.hits.pop(tracker_id, None)
            self.since_sample.pop(tracker_id, None)
        # With re-identification, their appearance stays remembered (reidentifier.py).

    @staticmethod
    def draw(frame, people, with_depth):
        """Boxes with our IDs (one color per ID), confidence and, with depth, the distance."""
        image = frame.copy()
        for display_id, (x1, y1, x2, y2), confidence, meters, *_ in people:
            hue = (display_id * 47) % 180
            color = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])
            label = f"ID {display_id}  {confidence * 100:.0f}%"
            if with_depth:
                label += f"  {meters:.1f} m" if meters is not None else "  ? m"
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(height + 4, y1)
            tx = max(0, min(x1, image.shape[1] - width - 4))   # keep the label inside the image
            cv2.rectangle(image, (tx, ty - height - 4), (tx + width + 4, ty), color, -1)
            cv2.putText(image, label, (tx + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{len(people)} people", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def set_track_buffer(self, stamp):
        """Keep BoT-SORT's memory of a hidden object equal to stale_timeout seconds.

        BoT-SORT counts how long an object has been gone in processed frames (track_buffer),
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
            self.track_event("left", track_id, self.last_stamp or 0.0, reason=reason)
        self.last_seen.clear()

    def reset_tracker(self):
        # Clear BoT-SORT's memory so old IDs cannot come back. IDs restart at 1.
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
        self.last_processed_stamp = None
        self.display_ids.clear()
        self.hits.clear()
        self.next_display_id = 1
        self.track_features.clear()
        self.since_sample.clear()
        if self.reid is not None:
            self.reid.reset()


def main(args=None):
    rclpy.init(args=args)
    node = YoloCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if node.csv is not None:
            node.csv.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
