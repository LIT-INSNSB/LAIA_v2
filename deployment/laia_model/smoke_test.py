#!/usr/bin/env python3
"""Portable ONNX smoke test intended to run unchanged on Raspberry Pi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from laia_inference.api import predict_pose_window
from laia_inference.runtime_onnx import ONNXRuntimeClassifier


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "laia_lightstgcnv2_fp32.onnx")
    parser.add_argument("--example", type=Path, default=ROOT / "examples" / "pose_windows" / "smoke_window.npz")
    parser.add_argument("--expected", type=Path, default=ROOT / "examples" / "expected_outputs.json")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))["smoke_window"]
    with np.load(args.example, allow_pickle=False) as source:
        prediction = predict_pose_window(
            ONNXRuntimeClassifier(args.model, intra_op_threads=args.threads),
            source["keypoints_xy_normalized"],
            source["hand_present"],
            width=int(source["width"]),
            height=int(source["height"]),
            fps=float(source["fps"]),
        )
    if prediction["class_id"] != expected["class_id"]:
        raise AssertionError(f"clase {prediction['class_id']} != {expected['class_id']}")
    maximum_difference = float(
        np.max(np.abs(np.asarray(prediction["probabilities"]) - np.asarray(expected["probabilities"])))
    )
    if maximum_difference > 1e-5:
        raise AssertionError(f"diferencia de probabilidad {maximum_difference} > 1e-5")
    print(
        json.dumps(
            {
                "status": "PASS",
                "class_id": prediction["class_id"],
                "class_name": prediction["class_name"],
                "max_probability_abs_difference": maximum_difference,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

