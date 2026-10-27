# ROS 2 Interfaces

`yolo_camera_node.py` publishes its results on these topics so other teams (path planning, logging, analysis) can use them without reading the node's code.

All messages reuse the `header` of the camera image they came from, so `header.stamp` is the image time (sim time in Gazebo, original time in rosbag replay) and `header.frame_id` is the camera frame. Results are published for every **processed** frame. The node processes one of every three camera frames.

## Published topics

| Topic | Type | Contents |
| --- | --- | --- |
| `/perception/detections` | `vision_msgs/msg/Detection2DArray` | Every YOLO detection in the frame, tracked or not |
| `/perception/tracks` | `vision_msgs/msg/Detection2DArray` | Only detections that have a ByteTrack ID |
| `/perception/track_events` | `std_msgs/msg/String` (JSON) | One message each time an object enters or leaves the view |
| `/perception/image_annotated` | `sensor_msgs/msg/Image` (`bgr8`) | Camera image with boxes, labels, confidences, and IDs drawn on. Only published while something subscribes |

### Subscribed topics

| Topic | Type |
| --- | --- |
| `/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image` | `sensor_msgs/msg/Image` |

## `/perception/detections` and `/perception/tracks`

Each `Detection2D` in the array is filled in like this:

| Field | Meaning |
| --- | --- |
| `id` | ByteTrack track ID as a string, such as `"2"`. Empty string if the detection has no track yet (only on `/perception/detections`) |
| `bbox.center.position.x`, `.y` | Box center in **pixels**, origin at the image's top-left corner |
| `bbox.size_x`, `bbox.size_y` | Box width and height in pixels |
| `results[0].hypothesis.class_id` | Class name from the YOLO model, such as `"person"` |
| `results[0].hypothesis.score` | Detection confidence, 0.0 to 1.0 |

An empty array means the frame was processed and nothing was detected. If no messages arrive at all, the node is not running or not receiving images.

## `/perception/track_events`

The `data` field is a JSON object:

```json
{"event": "entered", "track_id": 2, "stamp": 412.533}
{"event": "left", "track_id": 2, "stamp": 418.012, "reason": "unseen for 3.1s"}
```

| Key | Meaning |
| --- | --- |
| `event` | `"entered"` or `"left"` |
| `track_id` | Integer track ID |
| `stamp` | Image time of the event, in seconds |
| `reason` | Only on `left`. `"unseen for N s"` (stale timeout), `"no camera frames"` (camera stopped), or `"stream restarted"` (sim reset or rosbag looped) |

## Examples

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep perception
ros2 topic echo /perception/tracks
ros2 topic echo /perception/track_events
ros2 run rqt_image_view rqt_image_view /perception/image_annotated
```

Minimal Python subscriber:

```python
from vision_msgs.msg import Detection2DArray

def on_tracks(msg):
    for d in msg.detections:
        c = d.bbox.center.position
        h = d.results[0].hypothesis
        print(f"ID {d.id}: {h.class_id} {h.score:.2f} at ({c.x:.0f}, {c.y:.0f})")

node.create_subscription(Detection2DArray, "/perception/tracks", on_tracks, 10)
```
