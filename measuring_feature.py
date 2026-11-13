"""
Frame-level performance recorder for the YOLO tracking node.
The camera node calls frame_arrived() for every
image and frame_processed() for every frame it actually runs YOLO on.
"""

import csv
import os
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

FIELDS = ["timestamp_utc", "frame_count", "stamp_sim", "n_det", "ids",
          "proc_ms", "pipeline_ms", "e2e_ms"]


class PerfMonitor:
    def __init__(self, log_path=None, window=90):
        if log_path is None:
            name = datetime.now().strftime("frames_%Y%m%d_%H%M%S.csv")
            log_path = os.path.join(os.path.expanduser("~/perf_logs"), name)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._arrivals = deque(maxlen=window)  # (wall_s, sim_s, frame_count)
        self._done = deque(maxlen=window)      # (wall_s, proc_ms)

    def frame_arrived(self, frame_count, stamp_sim):
        """Call for EVERY incoming image, including the ones the node skips."""
        self._arrivals.append((time.monotonic(), stamp_sim, frame_count))

    def frame_processed(self, frame_count, stamp_sim, ids, n_det,
                        proc_ms, pipeline_ms, e2e_ms=None):
        self._done.append((time.monotonic(), proc_ms))
        self._writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": frame_count,
            "stamp_sim": f"{stamp_sim:.6f}",
            "n_det": n_det,
            "ids": ";".join(str(i) for i in ids),
            "proc_ms": round(proc_ms, 3),
            "pipeline_ms": round(pipeline_ms, 3),
            "e2e_ms": "" if e2e_ms is None else round(e2e_ms, 3),
        })
        self._file.flush()

    def summary(self):
        """One-line rolling summary for the node's log."""
        if len(self._arrivals) < 2 or len(self._done) < 2:
            return "perf: waiting for frames"
        w0, s0, c0 = self._arrivals[0]
        w1, s1, c1 = self._arrivals[-1]
        dw, ds = w1 - w0, s1 - s0
        in_fps = (c1 - c0) / dw if dw > 0 else 0.0
        rtf = ds / dw if dw > 0 else 0.0
        dw_done = self._done[-1][0] - self._done[0][0]
        out_fps = (len(self._done) - 1) / dw_done if dw_done > 0 else 0.0
        proc = np.array([d[1] for d in self._done])
        return (f"perf: in {in_fps:.1f} fps | processed {out_fps:.1f} fps | "
                f"RTF {rtf:.2f} | infer mean {proc.mean():.0f} ms "
                f"p95 {np.percentile(proc, 95):.0f} ms")

    def close(self):
        if not self._file.closed:
            self._file.close()