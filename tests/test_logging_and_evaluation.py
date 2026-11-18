"""CSV log (tracking_logger.py) and the performance report built from it (evaluate_tracking.py)."""

import csv
import json
import os
import subprocess
import sys

import pytest

import evaluate_tracking
from tracking_logger import CSV_FIELDS, TrackingLogger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_rows_have_image_time_position_and_uav_pose(tmp_path):
    path = tmp_path / "log.csv"
    with TrackingLogger(str(path)) as log:
        log.log_detection(5, 3, 0, "person", 0.87, (10, 20, 110, 220), 41.2, stamp=1700.1234,
                          position=(0.1, -0.2, 2.53), uav_pose=(1, 2, 5, 90))
        log.log_detection(5, 4, 0, "person", 0.5, (1, 2, 3, 4), 41.2)
        log.log_tracking_event(6, "entered", 3, "", stamp=1700.2, detail="reidentified 0.812")
    rows = read(path)
    assert list(rows[0]) == CSV_FIELDS
    assert rows[0]["image_stamp"] == "1700.123" and rows[0]["z_m"] == "2.53"
    assert rows[0]["uav_z"] == "5.0" and rows[0]["uav_yaw"] == "90.0"
    assert rows[1]["x_m"] == "" and rows[1]["uav_x"] == ""            # unknown stays blank
    assert rows[2]["detail"] == "reidentified 0.812"


def test_file_with_old_columns_is_kept_and_a_new_one_started(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("timestamp_utc,frame_count\nold,1\n")
    TrackingLogger(str(path)).close()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("log.before-")]
    assert open(path).readline().strip().split(",") == CSV_FIELDS


def write_scenario_log(path):
    """Two runs. Run 2: 10 fps for 10 s; person 1 in view 0-4 s, gone, back (re-identified) at
    7-10 s; person 2 in view 2-3 s."""
    log = TrackingLogger(str(path))
    for frame in range(1, 4):
        log.log_frame_without_detections(frame, 30, stamp=100 + frame * 0.1)
    for frame in range(1, 101):
        t = 1000 + (frame - 1) * 0.1
        seen = ([1] if t < 1004 or t >= 1007 else []) + ([2] if 1002 <= t < 1003 else [])
        for person in seen:
            log.log_detection(frame, person, 0, "person", 0.8, (0, 0, 10, 10), 40 + frame % 10,
                              stamp=t, uav_pose=(1, 2, 5, 90) if frame % 2 else None)
        if not seen:
            log.log_frame_without_detections(frame, 35, stamp=t)
        if frame == 1:
            log.log_tracking_event(frame, "entered", 1, 40, stamp=t, detail="new")
        if frame == 21:
            log.log_tracking_event(frame, "entered", 2, 40, stamp=t, detail="new")
        if frame == 71:
            log.log_tracking_event(frame, "entered", 1, 40, stamp=t, detail="reidentified 0.8")
    log.close()


@pytest.fixture
def scenario_log(tmp_path):
    path = tmp_path / "log.csv"
    write_scenario_log(path)
    return path


def test_runs_are_split_where_the_frame_count_restarts(scenario_log):
    runs = evaluate_tracking.split_runs(read(scenario_log))
    assert [len(r) for r in runs] == [3, 113]


def test_report_metrics(scenario_log):
    result = evaluate_tracking.evaluate(evaluate_tracking.split_runs(read(scenario_log))[-1], people=2)
    assert result["frames"] == 100 and result["fps"] == 10.0
    assert result["detection"]["frames_with_people_pct"] == 70.0
    assert result["ids"] == {"count": 2, "per_person": 1.0, "entries": 3, "reidentified": 1, "new": 2}
    assert result["tracks"]["1"]["entries"] == 2 and result["tracks"]["1"]["longest_s"] == 3.9
    assert result["uav_pose_pct"] == 35.0      # odd frames with a person
    assert result["latency_ms"]["max"] == 49.0


def run_cli(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "evaluate_tracking.py"), *args],
                          capture_output=True, text=True)


def test_command_line_limits_set_the_exit_code(scenario_log):
    assert run_cli(str(scenario_log), "--people", "2", "--max-ids-per-person", "1.0",
                   "--min-fps", "5").returncode == 0
    failed = run_cli(str(scenario_log), "--people", "1", "--max-ids-per-person", "1.5")
    assert failed.returncode == 1 and "IDs per person 2.0" in failed.stderr


def test_json_output(scenario_log):
    result = json.loads(run_cli(str(scenario_log), "--json", "--run", "2").stdout)
    assert result["ids"]["count"] == 2
