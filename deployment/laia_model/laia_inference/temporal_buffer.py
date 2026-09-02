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
    tracks_created_delta: int = 0
    track_fragmentation_delta: int = 0


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


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
        tracks_created_delta: int = 0,
        track_fragmentation_delta: int = 0,
    ) -> None:
        points = np.asarray(points, dtype=np.float32)
        hand_present = np.asarray(hand_present, dtype=bool)
        track_ids = np.asarray(track_ids, dtype=np.int32)
        if points.shape != (2, 21, 2) or hand_present.shape != (2,) or track_ids.shape != (2,):
            raise ValueError("frame de pose incompatible")
        self._frames.append(
            PoseFrame(
                points.copy(),
                hand_present.copy(),
                float(timestamp_s),
                track_ids.copy(),
                tracks_created_delta=_nonnegative_int(tracks_created_delta),
                track_fragmentation_delta=_nonnegative_int(track_fragmentation_delta),
            )
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

    def window_diagnostics(self) -> dict[str, object]:
        """Return O(window) timing and local pose/tracking measurements.

        This method only reads the current complete window.  It deliberately
        leaves ``arrays()`` unchanged because inference callers depend on its
        four-array public contract.
        """

        if not self.ready:
            raise RuntimeError("buffer temporal incompleto")

        frames = tuple(self._frames)
        source_frame_count = len(frames)
        timestamps = np.asarray(
            [frame.timestamp_s for frame in frames],
            dtype=np.float64,
        )
        timing_valid = bool(
            np.isfinite(timestamps).all()
            and np.all(np.diff(timestamps) >= 0.0)
        )
        window_start_s: float | None = None
        window_end_s: float | None = None
        window_span_s: float | None = None
        effective_window_fps: float | None = None
        mean_frame_dt_ms: float | None = None
        min_frame_dt_ms: float | None = None
        max_frame_dt_ms: float | None = None
        if timing_valid:
            window_start_s = float(timestamps[0])
            window_end_s = float(timestamps[-1])
            window_span_s = window_end_s - window_start_s
            frame_dts = np.diff(timestamps)
            if frame_dts.size:
                mean_frame_dt_ms = float(np.mean(frame_dts) * 1000.0)
                min_frame_dt_ms = float(np.min(frame_dts) * 1000.0)
                max_frame_dt_ms = float(np.max(frame_dts) * 1000.0)
            if window_span_s > 0.0 and source_frame_count > 1:
                effective_window_fps = (source_frame_count - 1) / window_span_s

        hand_counts = np.asarray(
            [int(np.asarray(frame.hand_present, dtype=bool).sum()) for frame in frames],
            dtype=np.int32,
        )
        frames_0_hands = int(np.count_nonzero(hand_counts == 0))
        frames_1_hand = int(np.count_nonzero(hand_counts == 1))
        frames_2_hands = int(np.count_nonzero(hand_counts == 2))

        longest_two_hand_segment_frames = 0
        current_two_hand_segment = 0
        for hand_count in hand_counts:
            if int(hand_count) == 2:
                current_two_hand_segment += 1
                longest_two_hand_segment_frames = max(
                    longest_two_hand_segment_frames,
                    current_two_hand_segment,
                )
            else:
                current_two_hand_segment = 0

        hand_dropout_count = int(
            sum(
                previous == 2 and current < 2
                for previous, current in zip(hand_counts[:-1], hand_counts[1:])
            )
        )

        track_rows = [
            np.asarray(frame.track_ids, dtype=np.int32).reshape(2)
            for frame in frames
        ]
        visible_track_sets = [
            {int(track_id) for track_id in row if int(track_id) >= 0}
            for row in track_rows
        ]
        track_changes_in_window = int(
            sum(
                previous != current
                for previous, current in zip(
                    visible_track_sets[:-1],
                    visible_track_sets[1:],
                )
            )
        )
        slot_changes_in_window = int(
            sum(
                previous_slot >= 0
                and current_slot >= 0
                and previous_slot != current_slot
                for previous, current in zip(track_rows[:-1], track_rows[1:])
                for previous_slot, current_slot in zip(previous, current)
            )
        )
        unique_tracks_in_window = len(
            {track_id for row in visible_track_sets for track_id in row}
        )
        new_tracks_in_window = int(
            sum(frame.tracks_created_delta for frame in frames)
        )
        track_fragmentations_in_window = int(
            sum(frame.track_fragmentation_delta for frame in frames)
        )

        return {
            "timing_valid": timing_valid,
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
            "window_span_s": window_span_s,
            "effective_window_fps": effective_window_fps,
            "mean_frame_dt_ms": mean_frame_dt_ms,
            "min_frame_dt_ms": min_frame_dt_ms,
            "max_frame_dt_ms": max_frame_dt_ms,
            "source_frame_count": source_frame_count,
            "frames_0_hands": frames_0_hands,
            "frames_1_hand": frames_1_hand,
            "frames_2_hands": frames_2_hands,
            "fraction_2_hands": frames_2_hands / source_frame_count,
            "track_changes_in_window": track_changes_in_window,
            "new_tracks_in_window": new_tracks_in_window,
            "track_fragmentations_in_window": track_fragmentations_in_window,
            "slot_changes_in_window": slot_changes_in_window,
            "unique_tracks_in_window": unique_tracks_in_window,
            "hand_dropout_count": hand_dropout_count,
            "longest_two_hand_segment_frames": longest_two_hand_segment_frames,
        }
