#!/usr/bin/env python3
"""Raspberry Pi classifier/end-to-end benchmark with thermal observations."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from laia_inference.features_v2 import build_pose_features_v2
from laia_inference.runtime_onnx import ONNXRuntimeClassifier


ROOT = Path(__file__).resolve().parent


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def temperature_c() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not path.is_file():
        return None
    return float(path.read_text(encoding="utf-8").strip()) / 1000.0


def throttled() -> str | None:
    command = shutil.which("vcgencmd")
    if command is None:
        return None
    result = subprocess.run([command, "get_throttled"], check=False, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "laia_lightstgcnv2_fp32.onnx")
    parser.add_argument("--example", type=Path, default=ROOT / "examples" / "pose_windows" / "smoke_window.npz")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "raspberry_pi_classifier_runtime.json")
    args = parser.parse_args()
    with np.load(args.example, allow_pickle=False) as source:
        features = build_pose_features_v2(
            source["keypoints_xy_normalized"],
            source["hand_present"],
            width=int(source["width"]),
            height=int(source["height"]),
            fps=float(source["fps"]),
        )
    classifier = ONNXRuntimeClassifier(args.model, intra_op_threads=args.threads)
    for _ in range(int(args.warmup)):
        classifier.logits(features)
    start_temperature = temperature_c()
    start_throttled = throttled()
    latencies: list[float] = []
    started = time.perf_counter()
    for _ in range(int(args.iterations)):
        before = time.perf_counter()
        classifier.logits(features)
        latencies.append((time.perf_counter() - before) * 1000.0)
    elapsed = time.perf_counter() - started
    report = {
        "status": "PASS",
        "scope": "classifier_only_pose_features_precomputed",
        "iterations": int(args.iterations),
        "threads": int(args.threads),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "maximum": max(latencies),
        },
        "inferences_per_second": float(args.iterations) / elapsed,
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        "temperature_c": {"start": start_temperature, "end": temperature_c()},
        "throttled": {"start": start_throttled, "end": throttled()},
        "warning": "This excludes video decode and MediaPipe; measure them end-to-end on the target camera.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

