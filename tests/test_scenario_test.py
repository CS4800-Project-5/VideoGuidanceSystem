"""scenario_test.py itself, against a fake simulation: Gazebo calls are replaced, and "waiting"
writes the CSV rows the node would write. Checks that a good run passes and a bad one fails."""

import pytest

import scenario
import scenario_test
from tracking_logger import TrackingLogger


class FakeSim:
    """Person in or out of view, box blocking or not; wait() logs what a node would see."""

    def __init__(self, path, monkeypatch, returns_with_new_id=False):
        self.log = TrackingLogger(str(path))
        self.t, self.frame = 1000.0, 0
        self.person_in_view = self.blocked = False
        self.person_id, self.next_id = None, 1
        self.last_seen = None
        self.ever_seen = False
        self.returns_with_new_id = returns_with_new_id
        monkeypatch.setattr(scenario, "spawn_from_file", self.spawn)
        monkeypatch.setattr(scenario, "spawn_from_string", self.spawn)
        monkeypatch.setattr(scenario, "set_pose", self.set_pose)
        monkeypatch.setattr(scenario, "wait", self.wait)

    def spawn(self, name, sdf, pos):
        self.set_pose(name, pos)
        return True

    def set_pose(self, name, pos):
        if name == scenario.PERSON_NAME:
            self.person_in_view = pos == scenario.PERSON_IN_VIEW
        else:
            self.blocked = pos == scenario.OCCLUDER_BLOCKING

    def wait(self, seconds, reason=""):
        for _ in range(int(seconds * 10)):           # 10 frames per second
            self.t += 0.1
            self.frame += 1
            visible = self.person_in_view and not self.blocked
            if visible:
                if self.person_id is None:
                    returning = self.ever_seen and not self.returns_with_new_id
                    if returning:
                        self.person_id, detail = 1, "reidentified 0.85"
                    else:
                        self.person_id, detail = self.next_id, "new"
                        self.next_id += 1
                    self.ever_seen = True
                    self.log.log_tracking_event(self.frame, "entered", self.person_id, 40,
                                                stamp=self.t, detail=detail)
                self.last_seen = self.t
                self.log.log_detection(self.frame, self.person_id, 0, "person", 0.8,
                                       (0, 0, 10, 10), 40, stamp=self.t)
            else:
                self.log.log_frame_without_detections(self.frame, 30, stamp=self.t)
                if self.person_id is not None and self.t - self.last_seen > 3.0:
                    self.log.log_tracking_event(self.frame, "left", self.person_id, 30,
                                                stamp=self.t, detail="unseen for 3.1s")
                    self.person_id = None


def run(tmp_path, monkeypatch, capsys, **fake):
    path = tmp_path / "log.csv"
    sim = FakeSim(path, monkeypatch, **fake)
    monkeypatch.setattr(scenario, "PERSON_SDF", __file__)          # "model exists"
    monkeypatch.setattr("sys.argv", ["scenario_test.py", "--hold", "6", "--csv", str(path)])
    with pytest.raises(SystemExit) as exit_info:
        scenario_test.main()
    sim.log.close()
    return exit_info.value.code, capsys.readouterr().out


def test_good_run_passes(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys)
    assert code == 0, out
    assert "FAIL" not in out and "PASSED" in out


def test_person_returning_with_a_new_id_fails(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys, returns_with_new_id=True)
    assert code == 1
    assert "[FAIL] came back with the same ID" in out


def test_csv_follower_skips_half_written_lines(tmp_path):
    path = tmp_path / "log.csv"
    TrackingLogger(str(path)).close()
    follower = scenario_test.CsvFollower(str(path))
    with open(path, "a") as f:
        f.write("2026,1.0,1,frame,no_detection,,,,,,,,,,,,,,,,30,\n2026,1.1,2,fra")
    assert len(follower.new_rows()) == 1
    with open(path, "a") as f:
        f.write("me,no_detection,,,,,,,,,,,,,,,,30,\n")
    rows = follower.new_rows()
    assert len(rows) == 1 and rows[0]["frame_count"] == "2"
