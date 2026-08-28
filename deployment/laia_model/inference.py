#!/usr/bin/env python3
"""Minimal independent CLI for pose-window, image, and video inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from laia_inference.api import StreamingHandwashingRecognizer, predict_pose_window
from laia_inference.runtime_onnx import ONNXRuntimeClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "laia_lightstgcnv2_fp32.onnx"
DEFAULT_TASK_MODEL = ROOT / "model" / "hand_landmarker.task"


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def pose_window(args: argparse.Namespace) -> int:
    with np.load(args.input, allow_pickle=False) as source:
        points = source["keypoints_xy_normalized"]
        hands = source["hand_present"]
        width = int(source["width"])
        height = int(source["height"])
        fps = float(source["fps"])
    classifier = ONNXRuntimeClassifier(args.model, intra_op_threads=args.threads)
    _print(
        predict_pose_window(
            classifier,
            points,
            hands,
            width=width,
            height=height,
            fps=fps,
        )
    )
    return 0


def _recognizer(args: argparse.Namespace, *, fps: float, width: int, height: int) -> StreamingHandwashingRecognizer:
    task_model = args.task_model if args.pose_api == "tasks" else None
    return StreamingHandwashingRecognizer(
        args.model,
        fps=fps,
        width=width,
        height=height,
        pose_api=args.pose_api,
        task_model_path=task_model,
        intra_op_threads=args.threads,
    )


def image(args: argparse.Namespace) -> int:
    import cv2

    frame = cv2.imread(str(args.input))
    if frame is None:
        raise FileNotFoundError(args.input)
    height, width = frame.shape[:2]
    with _recognizer(args, fps=float(args.fps), width=width, height=height) as recognizer:
        # A single image only updates the temporal state. Returning warming_up
        # is intentional: an isolated frame cannot identify a motion class.
        _print(recognizer.update_frame(frame, timestamp_s=0.0))
    return 0


def video(args: argparse.Namespace) -> int:
    import cv2

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(args.input)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("video sin FPS/resolución válidos")
    frame_index = 0
    try:
        with _recognizer(args, fps=fps, width=width, height=height) as recognizer:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = recognizer.update_frame(frame, timestamp_s=frame_index / fps)
                if result["status"] in {"prediction", "insufficient_pose"} or args.include_state:
                    result["source_frame_index"] = frame_index
                    _print(result)
                frame_index += 1
    finally:
        capture.release()
    if frame_index == 0:
        raise RuntimeError("video sin frames decodificables")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    pose = subcommands.add_parser("pose-window", help="clasificar un .npz de pose explícito")
    pose.add_argument("--input", type=Path, required=True)
    pose.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pose.add_argument("--threads", type=int)
    pose.set_defaults(function=pose_window)
    for name, function in (("image", image), ("video", video)):
        item = subcommands.add_parser(name)
        item.add_argument("--input", type=Path, required=True)
        item.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        item.add_argument("--pose-api", choices=("legacy", "tasks"), default="tasks")
        item.add_argument("--task-model", type=Path, default=DEFAULT_TASK_MODEL)
        item.add_argument("--threads", type=int)
        if name == "image":
            item.add_argument("--fps", type=float, default=30.0)
        else:
            item.add_argument("--include-state", action="store_true")
        item.set_defaults(function=function)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
