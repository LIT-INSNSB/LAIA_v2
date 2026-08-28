"""Frozen NumPy implementation of PSKUS-pose-features-v2 inference.

The equations mirror ``tools.pskus_training.pose_features`` but omit all
training augmentation. Missing observations remain explicit until streams are
packed with a finite zero and a separate validity channel.
"""

from __future__ import annotations

from typing import Any

import numpy as np


POSE_FEATURE_SCHEMA = "PSKUS-pose-features-v2"
HAND_PARENTS = np.asarray(
    [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int64,
)
PALM_ANCHORS = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
PALM_SCALE_EDGES = ((0, 9), (5, 17))


def mask_aware_resample(
    points: np.ndarray,
    node_valid: np.ndarray,
    hand_present: np.ndarray,
    *,
    output_steps: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    node_valid = np.asarray(node_valid, dtype=bool)
    hand_present = np.asarray(hand_present, dtype=bool)
    if points.ndim != 4 or points.shape[1:] != (2, 21, 2):
        raise ValueError(f"points debe ser (F,2,21,2), recibido {points.shape}")
    if node_valid.shape != points.shape[:-1] or hand_present.shape != points.shape[:2]:
        raise ValueError("máscaras incompatibles con points")
    if points.shape[0] < 2 or int(output_steps) < 2:
        raise ValueError("resampling requiere al menos dos frames de entrada y salida")

    positions = np.linspace(0.0, points.shape[0] - 1, int(output_steps), dtype=np.float64)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    alpha = (positions - left).astype(np.float32)
    output_valid = node_valid[left] & node_valid[right]
    output_hands = hand_present[left] & hand_present[right]
    safe = np.where(node_valid[..., None], points, 0.0)
    weights = alpha.reshape(-1, 1, 1, 1)
    output = safe[left] * (1.0 - weights) + safe[right] * weights
    output = np.where(output_valid[..., None], output, np.nan).astype(np.float32, copy=False)
    return output, output_valid, output_hands, positions.astype(np.float32)


def aspect_correct_to_pixels(
    points_normalized: np.ndarray,
    node_valid: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width/height deben ser positivos")
    result = np.asarray(points_normalized, dtype=np.float32).copy()
    result[..., 0] *= float(width)
    result[..., 1] *= float(height)
    return np.where(np.asarray(node_valid, dtype=bool)[..., None], result, np.nan)


def normalize_common_two_hand_geometry(
    points_pixel: np.ndarray,
    node_valid: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    points = np.asarray(points_pixel, dtype=np.float32)
    valid = np.asarray(node_valid, dtype=bool)
    if points.shape[:-1] != valid.shape or points.shape[-3:] != (2, 21, 2):
        raise ValueError("shape inválido para normalización geométrica")

    palm_centres: list[np.ndarray] = []
    palm_scales: list[float] = []
    for time_index in range(points.shape[0]):
        for hand_index in range(2):
            hand_points = points[time_index, hand_index]
            hand_valid = valid[time_index, hand_index]
            anchor_valid = hand_valid[PALM_ANCHORS]
            if int(anchor_valid.sum()) >= 3:
                palm_centres.append(np.mean(hand_points[PALM_ANCHORS][anchor_valid], axis=0))
            for source, target in PALM_SCALE_EDGES:
                if hand_valid[source] and hand_valid[target]:
                    distance = float(np.linalg.norm(hand_points[source] - hand_points[target]))
                    if np.isfinite(distance) and distance > 1e-6:
                        palm_scales.append(distance)

    finite_points = points[valid]
    if palm_centres:
        centre = np.median(np.stack(palm_centres), axis=0).astype(np.float32)
    elif finite_points.size:
        centre = np.median(finite_points, axis=0).astype(np.float32)
    else:
        centre = np.asarray([float(width) / 2.0, float(height) / 2.0], dtype=np.float32)

    fallback = "none"
    if palm_scales:
        scale = float(np.median(np.asarray(palm_scales, dtype=np.float64)))
    elif finite_points.shape[0] >= 2:
        scale = float(np.linalg.norm(np.ptp(finite_points, axis=0)))
        fallback = "valid_point_extent"
    else:
        scale = float(max(width, height))
        fallback = "image_extent_no_pose"
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("escala geométrica degenerada")

    normalized = (points - centre.reshape(1, 1, 1, 2)) / scale
    normalized = np.where(valid[..., None], normalized, np.nan).astype(np.float32, copy=False)
    return normalized, centre, scale, fallback


def derive_motion(
    points: np.ndarray,
    node_valid: np.ndarray,
    *,
    seconds_per_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    if float(seconds_per_step) <= 0:
        raise ValueError("seconds_per_step debe ser positivo")
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(node_valid, dtype=bool)
    motion = np.full_like(points, np.nan, dtype=np.float32)
    motion_valid = np.zeros_like(valid, dtype=bool)
    motion_valid[1:] = valid[1:] & valid[:-1]
    differences = (points[1:] - points[:-1]) / float(seconds_per_step)
    motion[1:] = np.where(motion_valid[1:, ..., None], differences, np.nan)
    return motion, motion_valid


def derive_bones(points: np.ndarray, node_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(node_valid, dtype=bool)
    parents = HAND_PARENTS.reshape(1, 1, 21)
    parent_points = np.take_along_axis(points, parents[..., None], axis=2)
    parent_valid = np.take_along_axis(valid, parents, axis=2)
    bone_valid = valid & parent_valid
    bones = points - parent_points
    bones = np.where(bone_valid[..., None], bones, np.nan).astype(np.float32, copy=False)
    return bones, bone_valid


def pack_stream(points: np.ndarray, valid: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if points.shape[:-1] != valid.shape or points.shape[-3:] != (2, 21, 2):
        raise ValueError("shape inválido para stream de pose")
    safe = np.where(valid[..., None], points, float(fill_value)).reshape(points.shape[0], 42, 2)
    flat_valid = valid.reshape(points.shape[0], 42)
    stream = np.concatenate([safe, flat_valid[..., None].astype(np.float32)], axis=-1)
    stream = np.transpose(stream, (2, 0, 1)).copy().astype(np.float32, copy=False)
    if not np.isfinite(stream).all():
        raise ValueError("stream de pose contiene NaN/Inf después del empaquetado")
    return stream


def build_pose_features_v2(
    points_normalized: np.ndarray,
    hand_present: np.ndarray,
    *,
    width: int,
    height: int,
    fps: float,
    output_steps: int = 32,
    fill_value: float = 0.0,
) -> dict[str, Any]:
    points = np.asarray(points_normalized, dtype=np.float32)
    hands = np.asarray(hand_present, dtype=bool)
    if float(fps) <= 0:
        raise ValueError("fps debe ser positivo")
    node_finite = np.isfinite(points).all(axis=-1)
    backend_present = np.broadcast_to(hands[:, :, None], node_finite.shape)
    if np.any(backend_present & ~node_finite):
        raise ValueError("hand_present=1 con nodos no finitos")
    if np.any(~backend_present & node_finite):
        raise ValueError("hand_present=0 con nodos finitos")

    resampled, node_valid, resampled_hands, positions = mask_aware_resample(
        points,
        backend_present & node_finite,
        hands,
        output_steps=int(output_steps),
    )
    pixels = aspect_correct_to_pixels(resampled, node_valid, width=int(width), height=int(height))
    joints, centre, palm_scale, scale_fallback = normalize_common_two_hand_geometry(
        pixels,
        node_valid,
        width=int(width),
        height=int(height),
    )
    source_span_seconds = float((points.shape[0] - 1) / float(fps))
    seconds_per_step = source_span_seconds / float(int(output_steps) - 1)
    motion, motion_valid = derive_motion(joints, node_valid, seconds_per_step=seconds_per_step)
    bones, bone_valid = derive_bones(joints, node_valid)
    flat_node_valid = node_valid.reshape(int(output_steps), 42)
    return {
        "joint_features": pack_stream(joints, node_valid, fill_value=fill_value),
        "motion_features": pack_stream(motion, motion_valid, fill_value=fill_value),
        "bone_features": pack_stream(bones, bone_valid, fill_value=fill_value),
        "node_valid": flat_node_valid,
        "motion_valid": motion_valid.reshape(int(output_steps), 42),
        "bone_valid": bone_valid.reshape(int(output_steps), 42),
        "hand_present": resampled_hands,
        "source_frame_position": positions,
        "seconds_per_step": float(seconds_per_step),
        "geometry_center_pixel": centre,
        "geometry_scale_pixel": float(palm_scale),
        "geometry_scale_fallback": scale_fallback,
    }

