#!/usr/bin/env python3
"""Create inference-only PyTorch and ONNX artifacts from the selected checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from laia_inference.model import load_torch_model


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = (
    "other_hand_washing_movement",
    "palm_to_palm",
    "palm_over_dorsum_fingers_interlaced",
    "palm_to_palm_fingers_interlaced",
    "backs_of_fingers_to_opposing_palm",
    "rotational_rubbing_of_thumb",
    "fingertips_to_palm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, source = load_torch_model(args.checkpoint)
    inference_checkpoint = args.output_dir / "laia_lightstgcnv2_best_state.pt"
    inference_payload = {
        "format_version": "LAIA-inference-state-v1",
        "architecture": "LightSTGCNV2",
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_checkpoint_epoch": int(source.get("epoch", -1)),
        "source_checkpoint_selection_metric": str(source.get("selection_metric", "macro_f1")),
        "provenance": source.get("provenance", {}),
    }
    temporary_checkpoint = inference_checkpoint.with_suffix(".pt.partial")
    torch.save(inference_payload, temporary_checkpoint)
    temporary_checkpoint.replace(inference_checkpoint)

    joint = torch.zeros((1, 3, 32, 42), dtype=torch.float32)
    motion = torch.zeros_like(joint)
    bone = torch.zeros_like(joint)
    node_valid = torch.zeros((1, 32, 42), dtype=torch.bool)
    onnx_path = args.output_dir / "laia_lightstgcnv2_fp32.onnx"
    temporary_onnx = onnx_path.with_suffix(".onnx.partial")
    torch.onnx.export(
        model,
        (joint, motion, bone, node_valid),
        str(temporary_onnx),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["joint_features", "motion_features", "bone_features", "node_valid"],
        output_names=["logits"],
        dynamic_axes={
            "joint_features": {0: "batch"},
            "motion_features": {0: "batch"},
            "bone_features": {0: "batch"},
            "node_valid": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    exported = onnx.load(str(temporary_onnx))
    onnx.checker.check_model(exported)
    temporary_onnx.replace(onnx_path)
    print(
        json.dumps(
            {
                "source": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
                "inference_state": {
                    "path": str(inference_checkpoint),
                    "sha256": sha256(inference_checkpoint),
                    "bytes": inference_checkpoint.stat().st_size,
                },
                "onnx": {
                    "path": str(onnx_path),
                    "sha256": sha256(onnx_path),
                    "bytes": onnx_path.stat().st_size,
                    "opset": int(args.opset),
                },
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

