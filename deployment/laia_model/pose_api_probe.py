#!/usr/bin/env python3
"""Dump tracked pose from one MediaPipe API for controlled migration checks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from laia_inference.pose_mediapipe import create_pose_backend
from laia_inference.tracking import StatefulHandTracker, detections_to_slots


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-api", choices=("legacy", "tasks"), required=True)
    parser.add_argument("--task-model", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("videos", type=Path, nargs="+")
    args = parser.parse_args()
    output_root = args.output_root / args.pose_api
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for video in args.videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(video)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        backend = create_pose_backend(args.pose_api, task_model_path=args.task_model)
        tracker = StatefulHandTracker()
        points_frames: list[np.ndarray] = []
        hands_frames: list[np.ndarray] = []
        tracks_frames: list[np.ndarray] = []
        started = time.perf_counter()
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = backend.process(frame, int(round(frame_index / fps * 1000.0)))
                tracked = tracker.update(detections)
                points, hands, track_ids = detections_to_slots(tracked)
                points_frames.append(points)
                hands_frames.append(hands)
                tracks_frames.append(track_ids)
                frame_index += 1
        finally:
            backend.close()
            capture.release()
        elapsed = time.perf_counter() - started
        output = output_root / f"{video.stem}.npz"
        np.savez_compressed(
            output,
            keypoints_xy_normalized=np.stack(points_frames),
            hand_present=np.stack(hands_frames),
            track_ids=np.stack(tracks_frames),
            fps=np.float32(fps),
            width=np.int32(width),
            height=np.int32(height),
        )
        summary.append(
            {
                "video": str(video),
                "output": str(output),
                "frames": frame_index,
                "seconds": elapsed,
                "fps_end_to_end": frame_index / elapsed,
                "coverage_ge1": float(np.mean(np.stack(hands_frames).sum(axis=1) >= 1)),
                "coverage_2": float(np.mean(np.stack(hands_frames).sum(axis=1) == 2)),
            }
        )
    report = {"pose_api": args.pose_api, "videos": summary}
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

