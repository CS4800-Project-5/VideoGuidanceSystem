#!/usr/bin/env python3
"""Automated simulation test: runs the Gazebo scenarios and checks what the node logged.

Start the simulation first (Terminal 1: ros2 launch ./sim.launch.py, or sim-demo), then:

    python3 scenario_test.py                  # both scenarios
    python3 scenario_test.py entry_exit       # or just one
    python3 scenario_test.py --hold 30        # slower machines: wait longer at each step

Checks, from the node's CSV log (~/yolo_tracking_log.csv, the node's csv_log parameter):
  entry_exit   the person enters and gets an ID; leaves (after stale_timeout); comes back with
               the SAME ID, re-identified by appearance; only one ID is used.
  occlusion    a box hides the person for --occlude-time seconds (less than stale_timeout); the
               person keeps the SAME ID, with no "left" event.
Prints PASS/FAIL per check and the performance report; exits with 1 if a check failed.

Only needs Python and the gz CLI (like scenario.py), not ROS.
"""

import argparse
import csv
import io
import os
import sys
import time

import evaluate_tracking
import scenario
from tracking_logger import CSV_FIELDS


class CsvFollower:
    """Reads the rows the node appends to its CSV log after this test started."""

    def __init__(self, path):
        self.path = path
        self.offset = os.path.getsize(path) if os.path.exists(path) else 0
        self.partial = ""

    def new_rows(self):
        if not os.path.exists(self.path):
            return []
        if os.path.getsize(self.path) < self.offset:    # file replaced: start from its top
            self.offset, self.partial = 0, ""
        with open(self.path, encoding="utf-8", newline="") as f:
            f.seek(self.offset)
            text = self.partial + f.read()
            self.offset = f.tell()
        complete, _, self.partial = text.rpartition("\n")   # a line may still be being written
        lines = [line for line in complete.split("\n") if line and not line.startswith("timestamp_utc")]
        return list(csv.DictReader(io.StringIO("\n".join(lines)), fieldnames=CSV_FIELDS))


def events(rows, kind=None):
    return [r for r in rows if r["record_type"] == "tracking_event" and (kind is None or r["event"] == kind)]


def ids_seen(rows):
    return {r["track_id"] for r in rows if r["record_type"] == "detection"}


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, ok, text):
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
        self.failed += not ok
        return ok


def start_out_of_view(stale_timeout):
    """Person out of view and the box away, and long enough for the node to log them as left."""
    if not scenario.spawn_from_file(scenario.PERSON_NAME, scenario.PERSON_SDF, scenario.PERSON_OUT_OF_VIEW):
        scenario.set_pose(scenario.PERSON_NAME, scenario.PERSON_OUT_OF_VIEW)
    if not scenario.spawn_from_string(scenario.OCCLUDER_NAME, scenario.OCCLUDER_SDF, scenario.OCCLUDER_AWAY):
        scenario.set_pose(scenario.OCCLUDER_NAME, scenario.OCCLUDER_AWAY)
    scenario.wait(stale_timeout + 3, "start with no one in view")


def test_entry_exit(log, checks, hold):
    print("\n=== Entry / exit ===")
    log.new_rows()
    scenario.set_pose(scenario.PERSON_NAME, scenario.PERSON_IN_VIEW)
    scenario.wait(hold, "person in view")
    first = log.new_rows()
    if not checks.check(bool(first), "the node is logging (CSV rows are being written)"):
        print("    Is the node running, with the CSV on (csv_log)? Is this the right file?")
        return []
    entered = events(first, "entered")
    if not checks.check(bool(entered), "person entered and got an ID"):
        return first
    person = entered[0]["track_id"]

    scenario.set_pose(scenario.PERSON_NAME, scenario.PERSON_OUT_OF_VIEW)
    scenario.wait(hold, "person out of view")
    away = log.new_rows()
    checks.check(any(r["track_id"] == person for r in events(away, "left")),
                 f"ID {person} left after the stale timeout")

    scenario.set_pose(scenario.PERSON_NAME, scenario.PERSON_IN_VIEW)
    scenario.wait(hold, "person back in view")
    back = log.new_rows()
    returned = events(back, "entered")
    checks.check(bool(returned) and returned[0]["track_id"] == person,
                 f"came back with the same ID {person} (got {[r['track_id'] for r in returned]})")
    checks.check(bool(returned) and returned[0]["detail"].startswith("reidentified"),
                 f"re-identified by appearance ({returned[0]['detail'] if returned else 'no entry'})")
    all_rows = first + away + back
    checks.check(ids_seen(all_rows) == {person}, f"only one ID used (saw {sorted(ids_seen(all_rows))})")
    return all_rows


def test_occlusion(log, checks, hold, occlude_time):
    print("\n=== Temporary occlusion ===")
    log.new_rows()
    scenario.set_pose(scenario.OCCLUDER_NAME, scenario.OCCLUDER_AWAY)
    scenario.set_pose(scenario.PERSON_NAME, scenario.PERSON_IN_VIEW)
    scenario.wait(hold, "person in view")
    before = log.new_rows()
    seen = ids_seen(before[len(before) // 2:])          # the settled second half
    if not checks.check(len(seen) == 1, f"one person tracked before the occlusion (saw {sorted(seen)})"):
        return before
    person = next(iter(seen))

    scenario.set_pose(scenario.OCCLUDER_NAME, scenario.OCCLUDER_BLOCKING)
    scenario.wait(occlude_time, "person hidden")
    scenario.set_pose(scenario.OCCLUDER_NAME, scenario.OCCLUDER_AWAY)
    scenario.wait(hold, "box moved away")
    after = log.new_rows()
    checks.check(not any(r["track_id"] == person for r in events(after, "left")),
                 f"ID {person} was not logged as left while hidden")
    later = ids_seen(after[len(after) // 2:])
    checks.check(later == {person}, f"same ID {person} after the box moved away (saw {sorted(later)})")
    return before + after


def main():
    parser = argparse.ArgumentParser(description="Automated Gazebo tracking test")
    parser.add_argument("scenario", nargs="?", choices=["entry_exit", "occlusion", "all"], default="all")
    parser.add_argument("--hold", type=float, default=25.0,
                        help="real seconds at each step; longer than the stale timeout in sim time")
    parser.add_argument("--occlude-time", type=float, default=2.0,
                        help="seconds the box hides the person (less than the stale timeout)")
    parser.add_argument("--stale-timeout", type=float, default=3.0, help="the node's stale_timeout")
    parser.add_argument("--csv", default=os.environ.get("YOLO_LOG_CSV", "~/yolo_tracking_log.csv"))
    args = parser.parse_args()

    if not os.path.exists(scenario.PERSON_SDF):
        sys.exit(f"Person model not found at {scenario.PERSON_SDF} (README, Setup step 3)")
    log = CsvFollower(os.path.expanduser(args.csv))
    checks = Checks()
    started = time.monotonic()

    start_out_of_view(args.stale_timeout)
    rows = []
    if args.scenario in ("entry_exit", "all"):
        rows += test_entry_exit(log, checks, args.hold)
    if args.scenario in ("occlusion", "all"):
        rows += test_occlusion(log, checks, args.hold, args.occlude_time)

    if rows:
        print()
        evaluate_tracking.print_report(evaluate_tracking.evaluate(rows, people=1), 1, 1)
    print(f"\n{'FAILED' if checks.failed else 'PASSED'}: {checks.failed} check(s) failed "
          f"({time.monotonic() - started:.0f} s)")
    sys.exit(1 if checks.failed else 0)


if __name__ == "__main__":
    main()
