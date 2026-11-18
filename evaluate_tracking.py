#!/usr/bin/env python3
"""Performance report for a tracking run, from the node's CSV log (tracking_logger.py).

    python3 evaluate_tracking.py                         # last run in ~/yolo_tracking_log.csv
    python3 evaluate_tracking.py my_log.csv --list       # list the runs in a file
    python3 evaluate_tracking.py --run 2 --people 3      # run 2; 3 different people were filmed
    python3 evaluate_tracking.py --json                  # machine-readable, for automated tests
    python3 evaluate_tracking.py --people 3 --max-ids-per-person 1.5 --min-fps 5
                                                         # exit code 1 if a limit is missed

Metrics (times are image times, so a replayed rosbag gives the same numbers as the live run):
  frame rate          processed frames per second
  latency             time to process one frame (YOLO + tracking + re-ID + publishing), ms
  detection           share of frames with at least one person, people per frame, confidence
  IDs                 different IDs given out; with --people, IDs per real person (1.0 is ideal:
                      higher means people were given new IDs instead of keeping theirs)
  track continuity    per ID: how often it entered (1 = never lost), time in view, and the
                      longest stretch it was tracked without a gap
  re-identification   entries that got an old ID back by appearance vs. new IDs

Detection accuracy against ground truth needs labeled data, which the CSV doesn't have; with
--people, IDs per real person is the main accuracy number for identity tracking.

A file can hold several runs (the node appends). A new run starts where the frame count or the
image time goes backwards.
"""

import argparse
import csv
import json
import os
import statistics
import sys


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def split_runs(rows):
    """Lists of rows, one per node run."""
    runs, current = [], []
    last_frame = last_stamp = None
    for row in rows:
        frame, stamp = number(row.get("frame_count")), number(row.get("image_stamp"))
        restarted = ((frame is not None and last_frame is not None and frame < last_frame)
                     or (stamp is not None and last_stamp is not None and stamp < last_stamp - 1.0))
        if restarted and current:
            runs.append(current)
            current = []
        current.append(row)
        last_frame = frame if frame is not None else last_frame
        last_stamp = stamp if stamp is not None else last_stamp
    if current:
        runs.append(current)
    return runs


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return values[index]


def evaluate(rows, people=None):
    """The metrics of one run, as a dict."""
    frames = {}        # frame_count -> {"stamp", "ms", "people": [...], "uav": bool}
    events = []        # (stamp, event, id, detail)
    errors = 0
    for row in rows:
        kind = row.get("record_type")
        frame, stamp = number(row.get("frame_count")), number(row.get("image_stamp"))
        if kind in ("detection", "frame"):
            f = frames.setdefault(frame, {"stamp": stamp, "ms": number(row.get("processing_ms")),
                                          "people": [], "uav": False})
            f["uav"] = f["uav"] or number(row.get("uav_x")) is not None
            if kind == "detection":
                f["people"].append((row.get("track_id"), number(row.get("confidence"))))
        elif kind == "tracking_event":
            events.append((stamp, row.get("event"), row.get("track_id"), row.get("detail") or ""))
        elif kind == "processing_error":
            errors += 1

    ordered = sorted((f for f in frames.values() if f["stamp"] is not None),
                     key=lambda f: f["stamp"])
    result = {"frames": len(frames), "errors": errors}
    if len(ordered) >= 2:
        duration = ordered[-1]["stamp"] - ordered[0]["stamp"]
        result["duration_s"] = round(duration, 2)
        result["fps"] = round((len(ordered) - 1) / duration, 2) if duration > 0 else None
    latencies = [f["ms"] for f in frames.values() if f["ms"] is not None]
    if latencies:
        result["latency_ms"] = {"mean": round(statistics.mean(latencies), 1),
                                "median": round(statistics.median(latencies), 1),
                                "p95": round(percentile(latencies, 0.95), 1),
                                "max": round(max(latencies), 1)}
    if frames:
        with_people = [f for f in frames.values() if f["people"]]
        confidences = [c for f in frames.values() for _, c in f["people"] if c is not None]
        result["detection"] = {
            "frames_with_people_pct": round(100 * len(with_people) / len(frames), 1),
            "people_per_frame": round(sum(len(f["people"]) for f in frames.values()) / len(frames), 2),
            "mean_confidence": round(statistics.mean(confidences), 3) if confidences else None,
        }
        result["uav_pose_pct"] = round(100 * sum(f["uav"] for f in frames.values()) / len(frames), 1)

    # Per ID: frames seen, entries, time in view, longest continuous stretch.
    gap = 1.5 / result["fps"] if result.get("fps") else 1.0   # a missed frame or so is no gap
    tracks = {}
    for f in ordered:
        for track_id, _ in f["people"]:
            t = tracks.setdefault(track_id, {"stamps": []})
            t["stamps"].append(f["stamp"])
    for track_id, t in tracks.items():
        stamps = t.pop("stamps")
        start = stamps[0]
        best = 0.0
        for previous, current in zip(stamps, stamps[1:]):
            if current - previous > gap:
                best = max(best, previous - start)
                start = current
        best = max(best, stamps[-1] - start)
        t.update(first_s=round(stamps[0] - ordered[0]["stamp"], 2), frames=len(stamps),
                 in_view_s=round(stamps[-1] - stamps[0], 2), longest_s=round(best, 2),
                 entries=sum(1 for _, e, i, _ in events if e == "entered" and i == track_id))
    entered = [d for _, e, _, d in events if e == "entered"]
    reidentified = sum(1 for d in entered if d.startswith("reidentified"))
    result["ids"] = {"count": len(tracks),
                     "per_person": round(len(tracks) / people, 2) if people else None,
                     "entries": len(entered),
                     "reidentified": reidentified,
                     "new": len(entered) - reidentified}
    if tracks:
        result["continuity"] = {
            "mean_entries_per_id": round(statistics.mean(t["entries"] or 1 for t in tracks.values()), 2),
            "mean_longest_s": round(statistics.mean(t["longest_s"] for t in tracks.values()), 2),
            "ids_seen_once_pct": round(100 * sum(t["frames"] == 1 for t in tracks.values())
                                       / len(tracks), 1),
        }
    result["tracks"] = {str(k): v for k, v in sorted(tracks.items(), key=lambda kv: kv[1]["first_s"])}
    return result


def check_limits(result, min_fps=None, max_latency_p95=None, max_ids_per_person=None):
    """The limits a run misses, as text (empty if it meets them all, or none were given)."""
    failures = []
    if min_fps is not None and (result.get("fps") or 0) < min_fps:
        failures.append(f"frame rate {result.get('fps')} < {min_fps}")
    p95 = result.get("latency_ms", {}).get("p95")
    if max_latency_p95 is not None and (p95 is None or p95 > max_latency_p95):
        failures.append(f"latency p95 {p95} ms > {max_latency_p95}")
    per_person = result["ids"]["per_person"]
    if max_ids_per_person is not None and (per_person is None or per_person > max_ids_per_person):
        failures.append(f"IDs per person {per_person} > {max_ids_per_person}")
    return failures


def print_report(result, run_number, run_count):
    print(f"Run {run_number} of {run_count}: {result['frames']} frames"
          + (f", {result['duration_s']} s" if "duration_s" in result else ""))
    if result.get("fps") is not None:
        print(f"  Frame rate         {result['fps']} fps")
    if "latency_ms" in result:
        l = result["latency_ms"]
        print(f"  Latency            mean {l['mean']} ms, median {l['median']}, 95% under {l['p95']}, "
              f"max {l['max']}")
    if "detection" in result:
        d = result["detection"]
        print(f"  Detection          people in {d['frames_with_people_pct']}% of frames, "
              f"{d['people_per_frame']} per frame, mean confidence {d['mean_confidence']}")
    ids = result["ids"]
    print(f"  IDs                {ids['count']} different"
          + (f", {ids['per_person']} per real person (1.0 is ideal)" if ids["per_person"] else ""))
    print(f"  Entries            {ids['entries']}: {ids['reidentified']} re-identified, {ids['new']} new")
    if "continuity" in result:
        c = result["continuity"]
        print(f"  Track continuity   {c['mean_entries_per_id']} entries per ID (1 = never lost), "
              f"longest stretch {c['mean_longest_s']} s on average, "
              f"{c['ids_seen_once_pct']}% of IDs seen in one frame only")
    if "uav_pose_pct" in result:
        print(f"  UAV pose           known in {result['uav_pose_pct']}% of frames")
    if result["errors"]:
        print(f"  Processing errors  {result['errors']}")
    if result["tracks"]:
        print("\n  ID   first seen   frames   in view   longest   entries")
        for track_id, t in result["tracks"].items():
            print(f"  {track_id:>3}  {t['first_s']:>8.1f} s  {t['frames']:>7}  {t['in_view_s']:>6.1f} s"
                  f"  {t['longest_s']:>6.1f} s  {t['entries']:>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", nargs="?", default=os.environ.get("YOLO_LOG_CSV", "~/yolo_tracking_log.csv"))
    parser.add_argument("--run", type=int, default=0, help="run number (1 = first; default: last)")
    parser.add_argument("--list", action="store_true", help="list the runs in the file")
    parser.add_argument("--people", type=int, help="number of different real people filmed")
    parser.add_argument("--json", action="store_true", help="print the metrics as JSON")
    parser.add_argument("--min-fps", type=float, help="fail (exit 1) below this frame rate")
    parser.add_argument("--max-latency-p95", type=float, help="fail above this 95th percentile, ms")
    parser.add_argument("--max-ids-per-person", type=float, help="fail above this (needs --people)")
    args = parser.parse_args()

    path = os.path.expanduser(args.csv)
    with open(path, newline="", encoding="utf-8") as f:
        runs = split_runs(list(csv.DictReader(f)))
    if not runs:
        sys.exit(f"No rows in {path}")
    if args.list:
        for i, run in enumerate(runs, 1):
            print(f"Run {i}: {len(run)} rows, starts {run[0].get('timestamp_utc', '?')}")
        return
    index = args.run - 1 if args.run else len(runs) - 1
    if not 0 <= index < len(runs):
        sys.exit(f"--run must be 1 to {len(runs)}")

    result = evaluate(runs[index], args.people)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result, index + 1, len(runs))

    failures = check_limits(result, args.min_fps, args.max_latency_p95, args.max_ids_per_person)
    if failures:
        print("\nFAILED: " + "; ".join(failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
