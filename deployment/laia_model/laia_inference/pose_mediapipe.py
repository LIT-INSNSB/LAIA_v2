"""MediaPipe pose adapters for the x86 reference and Raspberry Pi target.

``legacy`` reproduces the PSKUS extractor exactly with MediaPipe 0.10.21.
``tasks`` uses the officially supported Hand Landmarker API available for
Raspberry Pi OS ARM64. It is deliberately named separately because changing
the MediaPipe API/model can change the pose distribution.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from .tracking import HandDetection


def _bbox(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [
            np.min(points[:, 0]) * width,
            np.min(points[:, 1]) * height,
            np.max(points[:, 0]) * width,
            np.max(points[:, 1]) * height,
        ],
        dtype=np.float32,
    )


class LegacyMediaPipeHands:
    variant = "legacy_solutions_0.10.21"

    def __init__(self) -> None:
        version = importlib.metadata.version("mediapipe")
        if version != "0.10.21":
            raise RuntimeError(f"legacy requiere mediapipe==0.10.21; encontrado {version}")
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError("este paquete MediaPipe no incluye mp.solutions")
        self._mp = mp
        self._create()

    def _create(self) -> None:
        self._hands = self._mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
        del timestamp_ms
        height, width = frame_bgr.shape[:2]
        result = self._hands.process(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        landmarks_list = result.multi_hand_landmarks or []
        handedness_list = result.multi_handedness or []
        detections: list[HandDetection] = []
        for index, hand in enumerate(landmarks_list):
            points = np.asarray([[item.x, item.y] for item in hand.landmark], dtype=np.float32)
            z = np.asarray([item.z for item in hand.landmark], dtype=np.float32)
            label = ""
            score = float("nan")
            if index < len(handedness_list) and handedness_list[index].classification:
                classification = handedness_list[index].classification[0]
                label = str(classification.label)
                score = float(classification.score)
            detections.append(
                HandDetection(
                    landmarks_xy_normalized=points,
                    landmarks_z=z,
                    bounding_box_pixel=_bbox(points, width, height),
                    raw_handedness_label=label,
                    raw_handedness_score=score,
                    metadata={"detection_index": index, "api": self.variant},
                )
            )
        return detections

    def reset(self) -> None:
        self._hands.close()
        self._create()

    def close(self) -> None:
        self._hands.close()


class TasksMediaPipeHandLandmarker:
    variant = "tasks_hand_landmarker_1.0.1"

    def __init__(self, model_path: str | Path) -> None:
        version = importlib.metadata.version("mediapipe")
        if version != "1.0.1":
            raise RuntimeError(f"Raspberry Tasks requiere mediapipe==1.0.1; encontrado {version}")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._python = python
        self._vision = vision
        self._last_timestamp_ms = -1
        self._create()

    def _create(self) -> None:
        options = self._vision.HandLandmarkerOptions(
            base_options=self._python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=self._vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = self._vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        height, width = frame_bgr.shape[:2]
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_bgr[:, :, ::-1]),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        detections: list[HandDetection] = []
        for index, hand in enumerate(result.hand_landmarks):
            points = np.asarray([[item.x, item.y] for item in hand], dtype=np.float32)
            z = np.asarray([item.z for item in hand], dtype=np.float32)
            label = ""
            score = float("nan")
            if index < len(result.handedness) and result.handedness[index]:
                classification = result.handedness[index][0]
                label = str(classification.category_name or "")
                score = float(classification.score)
            detections.append(
                HandDetection(
                    landmarks_xy_normalized=points,
                    landmarks_z=z,
                    bounding_box_pixel=_bbox(points, width, height),
                    raw_handedness_label=label,
                    raw_handedness_score=score,
                    metadata={"detection_index": index, "api": self.variant},
                )
            )
        return detections

    def reset(self) -> None:
        self._landmarker.close()
        self._create()

    def close(self) -> None:
        self._landmarker.close()


def create_pose_backend(
    variant: str,
    *,
    task_model_path: str | Path | None = None,
) -> LegacyMediaPipeHands | TasksMediaPipeHandLandmarker:
    if variant == "legacy":
        return LegacyMediaPipeHands()
    if variant == "tasks":
        if task_model_path is None:
            raise ValueError("--task-model es obligatorio para pose-api=tasks")
        return TasksMediaPipeHandLandmarker(task_model_path)
    raise ValueError(f"pose API desconocida: {variant}")

