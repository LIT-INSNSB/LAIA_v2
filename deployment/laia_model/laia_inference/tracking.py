"""Stateful two-hand association matching the productive PSKUS slot policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations

import numpy as np


@dataclass
class HandDetection:
    landmarks_xy_normalized: np.ndarray
    bounding_box_pixel: np.ndarray
    landmarks_z: np.ndarray | None = None
    raw_handedness_label: str = ""
    raw_handedness_score: float = float("nan")
    metadata: dict[str, object] = field(default_factory=dict)
    track_id: int = -1
    slot: int = -1
    assignment_status: str = "unassigned"


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box: np.ndarray) -> np.ndarray:
    return np.asarray(
        [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0],
        dtype=np.float64,
    )


def association_cost(
    previous: HandDetection,
    current: HandDetection,
    previous_previous: HandDetection | None,
) -> float:
    iou_cost = 1.0 - bbox_iou(previous.bounding_box_pixel, current.bounding_box_pixel)
    centre_distance = float(
        np.linalg.norm(_center(previous.bounding_box_pixel) - _center(current.bounding_box_pixel))
    )
    scale = max(
        float(
            np.linalg.norm(
                [
                    previous.bounding_box_pixel[2] - previous.bounding_box_pixel[0],
                    previous.bounding_box_pixel[3] - previous.bounding_box_pixel[1],
                ]
            )
        ),
        1.0,
    )
    centre_cost = min(1.0, centre_distance / scale)
    previous_points = np.asarray(previous.landmarks_xy_normalized, dtype=np.float64)
    current_points = np.asarray(current.landmarks_xy_normalized, dtype=np.float64)
    if previous_points.shape == current_points.shape and np.isfinite(previous_points).all() and np.isfinite(current_points).all():
        landmark_cost = min(1.0, float(np.mean(np.linalg.norm(previous_points - current_points, axis=-1))) * 4.0)
    else:
        landmark_cost = 1.0
    continuity_cost = centre_cost
    if previous_previous is not None:
        predicted = _center(previous.bounding_box_pixel) + (
            _center(previous.bounding_box_pixel) - _center(previous_previous.bounding_box_pixel)
        )
        continuity_cost = min(
            1.0,
            float(np.linalg.norm(predicted - _center(current.bounding_box_pixel))) / scale,
        )
    return 0.35 * iou_cost + 0.20 * centre_cost + 0.25 * landmark_cost + 0.20 * continuity_cost


def _minimum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Exact assignment for at most two hands; avoids a SciPy runtime dependency."""
    if cost.size == 0:
        return []
    rows, cols = cost.shape
    count = min(rows, cols)
    best: tuple[float, list[tuple[int, int]]] | None = None
    for selected_rows in combinations(range(rows), count):
        for selected_cols in combinations(range(cols), count):
            for permuted_cols in permutations(selected_cols):
                pairs = list(zip(selected_rows, permuted_cols))
                value = sum(float(cost[row, col]) for row, col in pairs)
                if best is None or value < best[0]:
                    best = (value, pairs)
    return best[1] if best is not None else []


class StatefulHandTracker:
    """Track IDs are temporal identities; slots are not anatomical Left/Right."""

    def __init__(self, *, max_cost: float = 0.82, max_gap: int = 1) -> None:
        self.max_cost = float(max_cost)
        self.max_gap = int(max_gap)
        self.reset()

    def reset(self) -> None:
        self._active: dict[int, tuple[HandDetection, int, HandDetection | None]] = {}
        self._next_track_id = 0
        self.tracks_created = 0
        self.track_fragmentation = 0

    def update(self, detections: list[HandDetection]) -> list[HandDetection]:
        current = list(detections)
        if len(current) > 2:
            raise ValueError("el contrato LAIA admite como máximo dos detecciones por frame")
        track_ids = [
            track_id
            for track_id, (_, gap, _) in self._active.items()
            if gap <= self.max_gap
        ]
        previous = [self._active[track_id][0] for track_id in track_ids]
        previous_previous = [self._active[track_id][2] for track_id in track_ids]
        costs = np.asarray(
            [
                [association_cost(old, new, prior) for new in current]
                for old, prior in zip(previous, previous_previous)
            ],
            dtype=np.float64,
        )
        if not previous or not current:
            costs = np.empty((len(previous), len(current)), dtype=np.float64)
        matched: dict[int, int] = {}
        for row, col in _minimum_assignment(costs):
            if float(costs[row, col]) <= self.max_cost:
                matched[col] = track_ids[row]

        assigned_ids: set[int] = set()
        for detection_index, detection in enumerate(current):
            if detection_index in matched:
                track_id = matched[detection_index]
                status = "matched_track_slot"
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                self.tracks_created += 1
                if previous:
                    self.track_fragmentation += 1
                status = "new_track_slot"
            old_detection = self._active[track_id][0] if track_id in self._active else None
            detection.track_id = int(track_id)
            detection.assignment_status = status
            self._active[track_id] = (detection, 0, old_detection)
            assigned_ids.add(track_id)

        for track_id, (old, gap, old_previous) in list(self._active.items()):
            if track_id not in assigned_ids:
                next_gap = gap + 1
                if next_gap > self.max_gap:
                    del self._active[track_id]
                else:
                    self._active[track_id] = (old, next_gap, old_previous)

        # Exact productive policy: compact currently observed track IDs into
        # slot0/slot1. This is stable tracking order, not anatomical side.
        for slot, detection in enumerate(sorted(current, key=lambda item: item.track_id)):
            detection.slot = int(slot)
        return current


def detections_to_slots(
    detections: list[HandDetection],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.full((2, 21, 2), np.nan, dtype=np.float32)
    hand_present = np.zeros((2,), dtype=bool)
    track_ids = np.full((2,), -1, dtype=np.int32)
    for detection in detections:
        if detection.slot not in (0, 1):
            continue
        landmarks = np.asarray(detection.landmarks_xy_normalized, dtype=np.float32)
        if landmarks.shape != (21, 2) or not np.isfinite(landmarks).all():
            raise ValueError("detección con landmarks inválidos")
        points[detection.slot] = landmarks
        hand_present[detection.slot] = True
        track_ids[detection.slot] = int(detection.track_id)
    return points, hand_present, track_ids

