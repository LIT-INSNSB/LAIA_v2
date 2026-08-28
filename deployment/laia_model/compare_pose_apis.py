#!/usr/bin/env python3
"""Compare legacy and Tasks pose distributions without treating either as truth."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from laia_inference.api import predict_pose_window
from laia_inference.runtime_onnx import ONNXRuntimeClassifier


def matched_distance(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> list[float]:
    indices_a = np.flatnonzero(mask_a)
    indices_b = np.flatnonzero(mask_b)
    count = min(len(indices_a), len(indices_b))
    if count == 0:
        return []
    best: tuple[float, list[float]] | None = None
    for chosen_a in combinations(indices_a.tolist(), count):
        for chosen_b_base in combinations(indices_b.tolist(), count):
            for chosen_b in permutations(chosen_b_base):
                values = [
                    float(np.mean(np.linalg.norm(a[index_a] - b[index_b], axis=-1)))
                    for index_a, index_b in zip(chosen_a, chosen_b)
                ]
                total = sum(values)
                if best is None or total < best[0]:
                    best = (total, values)
    return best[1] if best else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    classifier = ONNXRuntimeClassifier(args.model)
    details = []
    all_distances: list[float] = []
    total_frames = 0
    equal_counts = 0
    prediction_pairs: list[tuple[int, int]] = []
    legacy_files = sorted((args.input_root / "legacy").glob("*.npz"))
    for legacy_path in legacy_files:
        tasks_path = args.input_root / "tasks" / legacy_path.name
        if not tasks_path.is_file():
            raise FileNotFoundError(tasks_path)
        with np.load(legacy_path, allow_pickle=False) as old, np.load(tasks_path, allow_pickle=False) as new:
            old_points, old_hands = old["keypoints_xy_normalized"], old["hand_present"].astype(bool)
            new_points, new_hands = new["keypoints_xy_normalized"], new["hand_present"].astype(bool)
            if old_points.shape != new_points.shape:
                raise RuntimeError(f"shape mismatch: {legacy_path.name}")
            fps, width, height = float(old["fps"]), int(old["width"]), int(old["height"])
        count_old = old_hands.sum(axis=1)
        count_new = new_hands.sum(axis=1)
        total_frames += len(count_old)
        equal_counts += int(np.sum(count_old == count_new))
        distances: list[float] = []
        for frame_index in range(len(count_old)):
            distances.extend(
                matched_distance(
                    old_points[frame_index],
                    new_points[frame_index],
                    old_hands[frame_index],
                    new_hands[frame_index],
                )
            )
        all_distances.extend(distances)
        size = max(2, int(round(1.5 * fps)))
        stride = max(1, int(round(0.375 * fps)))
        local_pairs = []
        for start in range(0, len(count_old) - size + 1, stride):
            stop = start + size
            old_prediction = predict_pose_window(
                classifier, old_points[start:stop], old_hands[start:stop], width=width, height=height, fps=fps
            )
            new_prediction = predict_pose_window(
                classifier, new_points[start:stop], new_hands[start:stop], width=width, height=height, fps=fps
            )
            pair = (
                int(old_prediction["class_id"]) if old_prediction["class_id"] is not None else -1,
                int(new_prediction["class_id"]) if new_prediction["class_id"] is not None else -1,
            )
            local_pairs.append(pair)
            prediction_pairs.append(pair)
        details.append(
            {
                "video": legacy_path.stem,
                "frames": len(count_old),
                "legacy_coverage_ge1": float(np.mean(count_old >= 1)),
                "tasks_coverage_ge1": float(np.mean(count_new >= 1)),
                "legacy_coverage_2": float(np.mean(count_old == 2)),
                "tasks_coverage_2": float(np.mean(count_new == 2)),
                "hand_count_agreement": float(np.mean(count_old == count_new)),
                "matched_landmark_distance_p50_image_normalized": float(np.percentile(distances, 50)) if distances else None,
                "matched_landmark_distance_p95_image_normalized": float(np.percentile(distances, 95)) if distances else None,
                "classifier_window_agreement": float(np.mean([a == b for a, b in local_pairs])) if local_pairs else None,
                "classifier_windows": len(local_pairs),
            }
        )
    report = {
        "interpretation": "agreement proxy only; neither MediaPipe API is ground truth",
        "frames": total_frames,
        "videos": len(details),
        "hand_count_agreement": equal_counts / total_frames if total_frames else None,
        "matched_landmark_distance_p50_image_normalized": float(np.percentile(all_distances, 50)) if all_distances else None,
        "matched_landmark_distance_p95_image_normalized": float(np.percentile(all_distances, 95)) if all_distances else None,
        "classifier_window_agreement": float(np.mean([a == b for a, b in prediction_pairs])) if prediction_pairs else None,
        "classifier_windows": len(prediction_pairs),
        "details": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
