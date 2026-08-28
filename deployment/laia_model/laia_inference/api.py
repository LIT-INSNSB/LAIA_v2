"""Public low-level and streaming APIs for LAIA inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .features_v2 import build_pose_features_v2
from .pose_mediapipe import create_pose_backend
from .runtime_onnx import ONNXRuntimeClassifier
from .temporal_buffer import TemporalPoseBuffer
from .tracking import StatefulHandTracker, detections_to_slots


CLASS_NAMES = (
    "other_hand_washing_movement",
    "palm_to_palm",
    "palm_over_dorsum_fingers_interlaced",
    "palm_to_palm_fingers_interlaced",
    "backs_of_fingers_to_opposing_palm",
    "rotational_rubbing_of_thumb",
    "fingertips_to_palm",
)


def _prediction(
    logits: np.ndarray,
    probabilities: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1, 7)[0]
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1, 7)[0]
    class_id = int(np.argmax(probabilities))
    return {
        "status": "prediction",
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence_uncalibrated": float(probabilities[class_id]),
        "probabilities": [float(value) for value in probabilities],
        "logits": [float(value) for value in logits],
        "model_version": "laia-lightstgcnv2-pose-features-v2-v1",
        **metadata,
    }


def predict_pose_window(
    classifier: ONNXRuntimeClassifier,
    points_normalized: np.ndarray,
    hand_present: np.ndarray,
    *,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    """Classify one explicit pose window; no RGB/pose step is hidden."""
    features = build_pose_features_v2(
        points_normalized,
        hand_present,
        width=int(width),
        height=int(height),
        fps=float(fps),
        output_steps=32,
        fill_value=0.0,
    )
    hands = np.asarray(hand_present, dtype=bool)
    observed_counts = hands.sum(axis=1)
    metadata = {
        "pose_coverage_ge1": float(np.mean(observed_counts >= 1)),
        "pose_coverage_2": float(np.mean(observed_counts == 2)),
        "source_frame_count": int(hands.shape[0]),
        "source_fps": float(fps),
        "window_duration_seconds_requested": 1.5,
        "geometry_scale_fallback": str(features["geometry_scale_fallback"]),
    }
    if not np.any(observed_counts >= 1):
        # This is an integrity gate, not a tuned confidence threshold. A
        # movement prediction from an entirely empty pose window is not
        # operationally meaningful even though the network returns logits.
        return {
            "status": "insufficient_pose",
            "class_id": None,
            "class_name": None,
            "confidence_uncalibrated": None,
            "probabilities": None,
            "logits": None,
            "model_version": "laia-lightstgcnv2-pose-features-v2-v1",
            **metadata,
        }
    logits, probabilities = classifier.predict(features)
    return _prediction(logits, probabilities, metadata=metadata)


class StreamingHandwashingRecognizer:
    """MediaPipe + tracking + 1.5-second temporal model.

    The recognizer assumes that an upstream component or workflow has already
    established a hand-washing episode. Class 0 is *other washing movement*,
    not a negative/not-washing class.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        fps: float,
        width: int,
        height: int,
        pose_api: str,
        task_model_path: str | Path | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.classifier = ONNXRuntimeClassifier(model_path, intra_op_threads=intra_op_threads)
        self.pose_backend = create_pose_backend(pose_api, task_model_path=task_model_path)
        self.pose_api = str(pose_api)
        self.tracker = StatefulHandTracker(max_cost=0.82, max_gap=1)
        self.buffer = TemporalPoseBuffer(fps=self.fps, duration_seconds=1.5, stride_seconds=0.375)
        self._frame_index = 0

    def reset_temporal_state(self) -> None:
        """Clear pose, tracking and temporal-buffer state without reopening the camera."""

        self.pose_backend.reset()
        self.tracker.reset()
        self.buffer.reset()
        self._frame_index = 0

    def reset(self) -> None:
        """Reset the recognizer between independent sessions."""

        self.reset_temporal_state()

    def close(self) -> None:
        self.pose_backend.close()

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_s: float | None = None,
    ) -> dict[str, Any]:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame debe ser BGR uint8 HxWx3")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                f"resolución cambió durante la sesión: {frame.shape[1]}x{frame.shape[0]} "
                f"!= {self.width}x{self.height}"
            )
        timestamp = float(timestamp_s) if timestamp_s is not None else self._frame_index / self.fps
        detections = self.pose_backend.process(frame, int(round(timestamp * 1000.0)))
        tracked = self.tracker.update(detections)
        points, hand_present, track_ids = detections_to_slots(tracked)
        self.buffer.append(
            points,
            hand_present,
            timestamp_s=timestamp,
            track_ids=track_ids,
        )
        self._frame_index += 1

        if not self.buffer.ready:
            return {
                "status": "warming_up",
                "frames_collected": self.buffer.collected_frames,
                "frames_required": self.buffer.required_frames,
                "pose_api": self.pose_api,
            }
        if not self.buffer.should_emit:
            return {
                "status": "buffering",
                "next_prediction_in_frames": self.buffer.stride_frames
                - ((self._frame_index - self.buffer.required_frames) % self.buffer.stride_frames),
                "pose_api": self.pose_api,
            }

        window_points, window_hands, timestamps, _ = self.buffer.arrays()
        result = predict_pose_window(
            self.classifier,
            window_points,
            window_hands,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        result.update(
            {
                "window_start_s": float(timestamps[0]),
                "window_end_s": float(timestamps[-1]),
                "pose_api": self.pose_api,
                "tracks_created": int(self.tracker.tracks_created),
                "track_fragmentation": int(self.tracker.track_fragmentation),
            }
        )
        return result

    def __enter__(self) -> "StreamingHandwashingRecognizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
