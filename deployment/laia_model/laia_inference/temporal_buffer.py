"""Fixed-duration streaming buffer matching the training window convention."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseFrame:
    points: np.ndarray
    hand_present: np.ndarray
    timestamp_s: float
    track_ids: np.ndarray


class TemporalPoseBuffer:
    def __init__(
        self,
        *,
        fps: float,
        duration_seconds: float = 1.5,
        stride_seconds: float = 0.375,
    ) -> None:
        if float(fps) <= 0:
            raise ValueError("fps debe ser positivo")
        self.fps = float(fps)
        self.duration_seconds = float(duration_seconds)
        self.stride_seconds = float(stride_seconds)
        self.required_frames = max(2, int(round(self.duration_seconds * self.fps)))
        self.stride_frames = max(1, int(round(self.stride_seconds * self.fps)))
        self._frames: deque[PoseFrame] = deque(maxlen=self.required_frames)
        self._total_frames = 0

    def reset(self) -> None:
        self._frames.clear()
        self._total_frames = 0

    def append(
        self,
        points: np.ndarray,
        hand_present: np.ndarray,
        *,
        timestamp_s: float,
        track_ids: np.ndarray,
    ) -> None:
        points = np.asarray(points, dtype=np.float32)
        hand_present = np.asarray(hand_present, dtype=bool)
        track_ids = np.asarray(track_ids, dtype=np.int32)
        if points.shape != (2, 21, 2) or hand_present.shape != (2,) or track_ids.shape != (2,):
            raise ValueError("frame de pose incompatible")
        self._frames.append(
            PoseFrame(points.copy(), hand_present.copy(), float(timestamp_s), track_ids.copy())
        )
        self._total_frames += 1

    @property
    def collected_frames(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        return len(self._frames) == self.required_frames

    @property
    def should_emit(self) -> bool:
        return self.ready and (self._total_frames - self.required_frames) % self.stride_frames == 0

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError("buffer temporal incompleto")
        frames = list(self._frames)
        return (
            np.stack([item.points for item in frames]),
            np.stack([item.hand_present for item in frames]),
            np.asarray([item.timestamp_s for item in frames], dtype=np.float64),
            np.stack([item.track_ids for item in frames]),
        )

