#!/usr/bin/env python3
"""Compare source PyTorch and exported ONNX outputs on real pose windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from laia_inference.features_v2 import build_pose_features_v2
from laia_inference.runtime_onnx import ONNXRuntimeClassifier, softmax
from laia_inference.runtime_torch import TorchClassifier


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "laia_lightstgcnv2_fp32.onnx")
    parser.add_argument("--examples", type=Path, default=ROOT / "examples" / "pose_windows")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "parity.json")
    args = parser.parse_args()

    source = TorchClassifier(args.checkpoint)
    exported = ONNXRuntimeClassifier(args.model)
    results: list[dict[str, object]] = []
    max_logit = 0.0
    max_probability = 0.0
    argmax_agreement = 0
    files = sorted(path for path in args.examples.glob("*.npz") if path.name != "smoke_window.npz")
    if not files:
        raise RuntimeError("no hay ventanas de paridad")
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            features = build_pose_features_v2(
                data["keypoints_xy_normalized"],
                data["hand_present"],
                width=int(data["width"]),
                height=int(data["height"]),
                fps=float(data["fps"]),
            )
        torch_logits = source.logits(features)
        onnx_logits = exported.logits(features)
        torch_probabilities = softmax(torch_logits)
        onnx_probabilities = softmax(onnx_logits)
        logit_difference = float(np.max(np.abs(torch_logits - onnx_logits)))
        probability_difference = float(np.max(np.abs(torch_probabilities - onnx_probabilities)))
        same_argmax = int(np.argmax(torch_logits)) == int(np.argmax(onnx_logits))
        max_logit = max(max_logit, logit_difference)
        max_probability = max(max_probability, probability_difference)
        argmax_agreement += int(same_argmax)
        results.append(
            {
                "example": path.name,
                "class_id": int(np.argmax(torch_logits)),
                "argmax_equal": same_argmax,
                "max_logit_abs_difference": logit_difference,
                "max_probability_abs_difference": probability_difference,
            }
        )
    report = {
        "status": "PASS" if argmax_agreement == len(files) and max_probability <= 1e-5 else "FAIL",
        "examples": len(files),
        "argmax_agreement": argmax_agreement / len(files),
        "max_logit_abs_difference": max_logit,
        "max_probability_abs_difference": max_probability,
        "thresholds": {"argmax_agreement": 1.0, "max_probability_abs_difference": 1e-5},
        "details": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

