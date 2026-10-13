#!/usr/bin/env python3
"""
Run this while the simulation is up (see sim.launch.py):
    python3 scenario.py entry_exit
    python3 scenario.py occlusion --occlude-time 2.5
    python3 scenario.py all

Only needs the `gz` CLI, not ROS.
"""

import argparse
import os
import subprocess
import sys
import time

WORLD = "default"

PERSON_NAME = "yolo_person"
PERSON_SDF = os.path.expanduser(
    "~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/scrubs/2/model.sdf"
)

OCCLUDER_NAME = "occluder"
# A static box tall enough to hide the person. Single quotes inside the SDF
# keep it from clashing with the double quotes in the gz request string.
OCCLUDER_SDF = (
    "<?xml version='1.0'?><sdf version='1.9'>"
    f"<model name='{OCCLUDER_NAME}'><static>true</static><link name='link'>"
    "<visual name='visual'><geometry><box><size>0.3 1.5 2.5</size></box></geometry></visual>"
    "<collision name='collision'><geometry><box><size>0.3 1.5 2.5</size></box></geometry></collision>"
    "</link></model></sdf>"
)

# Positions (x, y, z) taken from the test report.
PERSON_IN_VIEW = (4.0, 0.0, 0.0)
PERSON_OUT_OF_VIEW = (4.0, 5.0, 0.0)
OCCLUDER_BLOCKING = (2.0, 0.0, 1.25)
OCCLUDER_AWAY = (2.0, 5.0, 1.25)


def gz_service(service, reqtype, req, timeout_ms=5000):
    """Call a Gazebo world service. Returns True if Gazebo replied `data: true`."""
    cmd = [
        "gz", "service",
        "-s", f"/world/{WORLD}/{service}",
        "--reqtype", reqtype,
        "--reptype", "gz.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0 and "data: true" in result.stdout
    if not ok:
        print(f"  [warn] {service} failed: {result.stdout.strip() or result.stderr.strip()}")
    return ok


def pose_str(x, y, z):
    return f"position: {{x: {x}, y: {y}, z: {z}}}"


def spawn_from_file(name, sdf_path, pos):
    print(f"Spawning {name} at {pos}")
    req = f'sdf_filename: "{sdf_path}", name: "{name}", pose: {{{pose_str(*pos)}}}'
    return gz_service("create", "gz.msgs.EntityFactory", req, timeout_ms=10000)


def spawn_from_string(name, sdf, pos):
    print(f"Spawning {name} at {pos}")
    req = f'sdf: "{sdf}", name: "{name}", pose: {{{pose_str(*pos)}}}'
    return gz_service("create", "gz.msgs.EntityFactory", req, timeout_ms=10000)


def set_pose(name, pos):
    print(f"Moving {name} to {pos}")
    req = f'name: "{name}", {pose_str(*pos)}'
    return gz_service("set_pose", "gz.msgs.Pose", req)


def wait(seconds, reason):
    print(f"  waiting {seconds:.1f}s ({reason})")
    time.sleep(seconds)


def ensure_person():
    """Spawn the person, or move it into view if it already exists."""
    if not spawn_from_file(PERSON_NAME, PERSON_SDF, PERSON_IN_VIEW):
        print("  person may already exist, moving it into view instead")
        set_pose(PERSON_NAME, PERSON_IN_VIEW)


def entry_exit(hold):
    print("\n=== Entry / exit test ===")
    ensure_person()
    wait(hold, "expect ENTRY, new ID")
    set_pose(PERSON_NAME, PERSON_OUT_OF_VIEW)
    wait(hold, "expect EXIT after the stale timeout")
    set_pose(PERSON_NAME, PERSON_IN_VIEW)
    wait(hold, "expect ENTRY on re-entry")


def occlusion(hold, occlude_time):
    print("\n=== Temporary occlusion test ===")
    ensure_person()
    if not spawn_from_string(OCCLUDER_NAME, OCCLUDER_SDF, OCCLUDER_AWAY):
        print("  occluder may already exist, moving it out of view instead")
        set_pose(OCCLUDER_NAME, OCCLUDER_AWAY)
    wait(hold, "note the person's ID")
    set_pose(OCCLUDER_NAME, OCCLUDER_BLOCKING)
    wait(occlude_time, "person hidden")
    set_pose(OCCLUDER_NAME, OCCLUDER_AWAY)
    wait(hold, "expect the SAME ID to return")


def main():
    parser = argparse.ArgumentParser(description="Gazebo tracking test scenarios")
    parser.add_argument("scenario", choices=["entry_exit", "occlusion", "all"])
    parser.add_argument("--hold", type=float, default=5.0,
                        help="seconds between steps; keep this longer than the node's stale timeout")
    parser.add_argument("--occlude-time", type=float, default=2.5,
                        help="seconds the occluder blocks the person")
    args = parser.parse_args()

    if not os.path.exists(PERSON_SDF):
        sys.exit(f"Person model not found at {PERSON_SDF}")

    if args.scenario in ("entry_exit", "all"):
        entry_exit(args.hold)
    if args.scenario in ("occlusion", "all"):
        occlusion(args.hold, args.occlude_time)
    print("\nDone.")


if __name__ == "__main__":
    main()
