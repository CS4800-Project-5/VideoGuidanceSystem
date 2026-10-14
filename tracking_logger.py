"""CSV logging tracking for the YOLOv8 + ByteTrack tracking project.

This module is intentionally separate from the ROS 2 camera node so it can be
reviewed and maintained as needed. It does not create ROS topics or perform
object detection; the camera node calls these methods with the results.
"""

import csv
import os
from datetime import datetime, timezone


CSV_FIELDS = [
    "timestamp_utc",
    "frame_count",
    "record_type",
    "event",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "processing_ms",
]


def utc_timestamp():
    """Return the current time in UTC as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class TrackingLogger:
    """Append detection, tracking-event, and frame records to a CSV file."""

    def __init__(self, log_path=None):
        if log_path is None:
            log_path = os.environ.get(
                "YOLO_LOG_CSV", "~/yolo_tracking_log.csv"
            )

        self.log_path = os.path.abspath(os.path.expanduser(log_path))
        parent_directory = os.path.dirname(self.log_path)
        if parent_directory:
            os.makedirs(parent_directory, exist_ok=True)

        file_is_empty = (
            not os.path.exists(self.log_path)
            or os.path.getsize(self.log_path) == 0
        )
        self._file = open(
            self.log_path, "a", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(
            self._file, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )

        if file_is_empty:
            self._writer.writeheader()
        self._file.flush()

    def write(self, **values):
        """Write one record, filling any unspecified CSV columns with blanks."""
        if self._file.closed:
            raise ValueError("Cannot write to a closed TrackingLogger")

        row = {field: "" for field in CSV_FIELDS}
        row.update(values)
        if not row.get("timestamp_utc"):
            row["timestamp_utc"] = utc_timestamp()
        self._writer.writerow(row)
        self._file.flush()

    def log_detection(
        self,
        frame_count,
        track_id,
        class_id,
        class_name,
        confidence,
        xyxy,
        processing_ms,
    ):
        """Record one YOLO detection and its optional ByteTrack ID."""
        self.write(
            frame_count=frame_count,
            record_type="detection",
            event="observed",
            track_id="" if track_id is None else track_id,
            class_id=class_id,
            class_name=class_name,
            confidence=round(float(confidence), 5),
            x1=round(float(xyxy[0]), 2),
            y1=round(float(xyxy[1]), 2),
            x2=round(float(xyxy[2]), 2),
            y2=round(float(xyxy[3]), 2),
            processing_ms=round(float(processing_ms), 3),
        )

    def log_frame_without_detections(self, frame_count, processing_ms):
        """Record a processed frame in which no detections were returned."""
        self.write(
            frame_count=frame_count,
            record_type="frame",
            event="no_detection",
            processing_ms=round(float(processing_ms), 3),
        )

    def log_tracking_event(self, frame_count, event, track_id, processing_ms):
        """Record a track entry or exit event."""
        self.write(
            frame_count=frame_count,
            record_type="tracking_event",
            event=event,
            track_id=track_id,
            processing_ms=round(float(processing_ms), 3),
        )

    def log_processing_error(self, frame_count, error):
        """Record an image-processing exception by its type and message."""
        self.write(
            frame_count=frame_count,
            record_type="processing_error",
            event=type(error).__name__,
            class_name=str(error),
        )

    def close(self):
        """Closes the CSV file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
