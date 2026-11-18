"""The ROS-facing parts: published topics (track_publisher.py), MAVROS input (uav_state.py) and the
node's per-frame logic (yolo_camera_node.py). Uses real ROS message types when ROS is sourced,
stand-ins otherwise (conftest.py). No ROS node is started."""

import json
import math
import time
import types

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from std_msgs.msg import Header

import track_publisher
import uav_state
import yolo_camera_node
from reidentifier import ReIdentifier

ns = types.SimpleNamespace


class FakeNode:
    """Just enough of rclpy's Node for the classes under test: publishers that record messages."""

    def __init__(self):
        self.published, self.logs = {}, []

    def get_logger(self):
        return ns(info=lambda m: self.logs.append(("info", m)),
                  warn=lambda m: self.logs.append(("warn", m)))

    def create_publisher(self, msg_type, topic, qos):
        return ns(publish=lambda m: self.published.setdefault(topic, []).append(m))

    def create_subscription(self, *args):
        return None

    def warnings(self):
        return [m for kind, m in self.logs if kind == "warn"]


# ---------------------------------------------------------------- track_publisher.py

def test_tracks_topics_carry_ids_boxes_and_3d_positions():
    node = FakeNode()
    publisher = track_publisher.TrackPublisher(node, "perception/")
    header = Header(frame_id="zed_left_camera_optical_frame")
    people = [(3, (100, 50, 200, 350), 0.9, (0.1, -0.2, 2.5), (0.6, 1.8, 0.5)),
              (4, (1, 2, 3, 4), 0.4, None, None)]
    publisher.publish_frame(header, people, with_3d=True)

    tracks = node.published["/perception/tracks"][0]
    first = tracks.detections[0]
    assert len(tracks.detections) == 2 and first.id == "3"
    assert (first.bbox.center.position.x, first.bbox.center.position.y) == (150.0, 200.0)
    assert (first.bbox.size_x, first.bbox.size_y) == (100.0, 300.0)
    assert first.results[0].hypothesis.class_id == "person"
    assert first.results[0].hypothesis.score == pytest.approx(0.9)

    tracks_3d = node.published["/perception/tracks_3d"][0]
    assert [d.id for d in tracks_3d.detections] == ["3"]          # only people with a distance
    center = tracks_3d.detections[0].bbox.center.position
    assert (center.x, center.y, center.z) == pytest.approx((0.1, -0.2, 2.5))
    assert tracks_3d.header.frame_id == "zed_left_camera_optical_frame"


def test_no_3d_topic_message_without_depth_and_empty_frames_still_published():
    node = FakeNode()
    publisher = track_publisher.TrackPublisher(node)
    publisher.publish_frame(Header(), [], with_3d=False)
    assert len(node.published["/perception/tracks"][0].detections) == 0
    assert "/perception/tracks_3d" not in node.published


def test_events_are_json():
    node = FakeNode()
    track_publisher.TrackPublisher(node).publish_event(
        "entered", 3, 1700.12345, how="reidentified", similarity=0.812, reason=None)
    event = json.loads(node.published["/perception/track_events"][0].data)
    assert event == {"event": "entered", "id": 3, "stamp": 1700.123, "how": "reidentified",
                     "similarity": 0.812}


# ---------------------------------------------------------------- uav_state.py

@pytest.fixture
def uav(monkeypatch):
    sent = []
    monkeypatch.setattr(uav_state, "TransformBroadcaster",
                        lambda node: ns(sendTransform=sent.append))
    node = FakeNode()
    state = uav_state.UavState(node, "/mavros", stale_after=0.2, publish_tf=True)
    return state, node, sent


def pose_message(x, y, z, yaw_degrees):
    message = PoseStamped()
    message.header.frame_id = "map"
    message.pose.position.x, message.pose.position.y, message.pose.position.z = x, y, z
    half = math.radians(yaw_degrees) / 2
    message.pose.orientation.z, message.pose.orientation.w = math.sin(half), math.cos(half)
    return message


def test_missing_mavros_is_reported_once_and_tracking_goes_on(uav):
    state, node, _ = uav
    assert state.current_pose() is None and state.summary() == "no UAV data"
    state.check()
    state.check()
    assert len(node.warnings()) == 2 and "never received" in node.warnings()[0]


def test_pose_and_state_are_read_and_sent_to_tf(uav):
    state, node, sent = uav
    state.pose_callback(pose_message(1.0, 2.0, 5.0, 90))
    state.state_callback(State(armed=True, mode="OFFBOARD", connected=True))
    x, y, z, yaw = state.current_pose()
    assert (x, y, z) == (1.0, 2.0, 5.0) and yaw == pytest.approx(90)
    assert "armed OFFBOARD" in state.summary()
    assert sent[0].child_frame_id == "base_link" and sent[0].transform.translation.z == 5.0


def test_stale_pose_is_not_used(uav):
    state, node, _ = uav
    state.pose_callback(pose_message(1.0, 2.0, 5.0, 0))
    time.sleep(0.3)
    state.check()
    assert state.current_pose() is None
    assert any("UAV pose" in w for w in node.warnings())


# ---------------------------------------------------------------- yolo_camera_node.py

class MeanColorExtractor:
    """Appearance = the crop's mean color, so two people in different colors are different."""
    default_threshold = 0.9

    def extract_batch(self, crops):
        features = []
        for crop in crops:
            v = crop.reshape(-1, 3).mean(0).astype(np.float32) + 1e-3
            features.append(v / np.linalg.norm(v))
        return features

    def similarity(self, a, b):
        return float(np.dot(a, b))


class Boxes:
    """The parts of an Ultralytics Boxes result the node reads."""

    def __init__(self, rows):
        self.id = Values([r[0] for r in rows]) if rows else None
        self.xyxy = Values([r[1] for r in rows])
        self.conf = Values([r[2] for r in rows])


class Values:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


RED, BLUE = (100, 100, 200, 400), (400, 100, 500, 400)


@pytest.fixture
def node():
    """A node object with its tracking state set up, without starting ROS or YOLO."""
    n = yolo_camera_node.YoloCameraNode.__new__(yolo_camera_node.YoloCameraNode)
    n.reid = ReIdentifier(MeanColorExtractor(), max_speed=2.0, position_slack=1.0)
    n.reid_every, n.reid_min_conf, n.min_hits = 1, 0.3, 2
    n.display_ids, n.hits, n.next_display_id = {}, {}, 1
    n.track_features, n.since_sample, n.reid_events = {}, {}, {}
    n.camera_info = ns(k=[500.0, 0, 320, 0, 500.0, 240, 0, 0, 1], width=640)
    # TF: the camera sits at (10, 0, 0) in the map, turned 90 degrees about the vertical axis.
    yaw = math.pi / 2
    rotation = ns(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    n.world_frame, n.tf_warned = "map", False
    n.tf_buffer = ns(lookup_transform=lambda *a, **k: ns(transform=ns(
        rotation=rotation, translation=ns(x=10.0, y=0.0, z=0.0))))
    return n


@pytest.fixture
def frame():
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:400, 100:200] = (0, 0, 255)      # red person on the left
    image[100:400, 400:500] = (255, 0, 0)      # blue person on the right
    return image


HEADER = ns(frame_id="camera_optical_frame", stamp=None)


def depth_at(meters):
    return np.full((480, 640), meters, np.float32)


def step(node, frame, rows, t, meters=4.0):
    return node.confirmed_people(Boxes(rows), depth_at(meters), frame, t, HEADER)


def test_camera_point_to_world(node):
    assert node.to_world(HEADER, (1.0, 0.0, 4.0)) == pytest.approx((10.0, 1.0, 4.0))


def test_ids_after_min_hits_with_3d_positions(node, frame):
    step(node, frame, [(11, RED, 0.9), (12, BLUE, 0.35)], 0.0)
    assert node.display_ids == {}                               # min_hits = 2: not yet
    people = step(node, frame, [(11, RED, 0.9), (12, BLUE, 0.35)], 0.1)
    assert sorted(p[0] for p in people) == [1, 2]
    red = next(p for p in people if p[1] == RED)
    assert red[3] == pytest.approx(4.0)                         # distance
    assert red[4] == pytest.approx((-1.36, -0.16, 4.0))         # camera-frame position
    # A person at 35% confidence (from above or behind) still gets appearance samples.
    assert len(node.reid.banks[2]) >= 1


def test_returning_person_gets_their_old_id(node, frame):
    for t in (0.0, 0.1):
        step(node, frame, [(11, RED, 0.9), (12, BLUE, 0.9)], t)
    node.forget(1)                                              # red left the view
    for t in (5.0, 5.1):
        people = step(node, frame, [(21, RED, 0.8), (12, BLUE, 0.9)], t)
    assert {p[0] for p in people} == {1, 2}
    assert node.reid_events[1][0] == "reidentified"


def test_lookalike_too_far_away_gets_a_new_id(node, frame):
    for t in (0.0, 0.1):
        step(node, frame, [(11, RED, 0.9)], t)
    node.forget(1)
    for t in (1.1, 1.2):                                        # 30 m away a second later
        people = step(node, frame, [(31, RED, 0.8)], t, meters=34.0)
    assert {p[0] for p in people} == {2}


def test_two_people_in_view_never_share_an_id(node, frame):
    for t in (0.0, 0.1):
        step(node, frame, [(11, RED, 0.9)], t)
    # A second red-clothed person appears while the first is still in view.
    for t in (0.2, 0.3):
        people = step(node, frame, [(11, RED, 0.9), (41, RED, 0.9)], t)
    assert sorted(p[0] for p in people) == [1, 2]


def test_window_drawing_handles_unknown_distance(node, frame):
    people = [(1, RED, 0.9, None, None, None), (2, BLUE, 0.5, 3.2, (0, 0, 3.2), (0.5, 1.7, 0.5))]
    assert node.draw(frame, people, True).shape == frame.shape
