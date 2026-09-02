from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from laia_inference.api import PoseSnapshot, StreamingHandwashingRecognizer, predict_pose_window
from laia_inference.features_v2 import build_pose_features_v2
from laia_inference.runtime_onnx import ONNXRuntimeClassifier
from laia_inference.temporal_buffer import TemporalPoseBuffer
from laia_inference.tracking import HandDetection, StatefulHandTracker, detections_to_slots


ROOT = Path(__file__).resolve().parents[1]


def detection(x: float) -> HandDetection:
    points = np.stack(
        [np.linspace(x, x + 0.1, 21), np.linspace(0.2, 0.4, 21)], axis=-1
    ).astype(np.float32)
    return HandDetection(
        landmarks_xy_normalized=points,
        bounding_box_pixel=np.asarray([x * 100, 20, (x + 0.1) * 100, 40], dtype=np.float32),
    )


class DeploymentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = ONNXRuntimeClassifier(ROOT / "model" / "laia_lightstgcnv2_fp32.onnx")

    def test_manifest_and_artifacts_exist(self) -> None:
        manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["architecture"]["name"], "LightSTGCNV2")
        self.assertEqual(len(manifest["classes"]), 7)
        for artifact in manifest["artifacts"].values():
            self.assertTrue((ROOT / artifact["path"]).is_file())

    def test_empty_pose_is_not_class_zero(self) -> None:
        points = np.full((45, 2, 21, 2), np.nan, dtype=np.float32)
        hands = np.zeros((45, 2), dtype=bool)
        result = predict_pose_window(
            self.classifier, points, hands, width=640, height=480, fps=30.0
        )
        self.assertEqual(result["status"], "insufficient_pose")
        self.assertIsNone(result["class_id"])
        self.assertIsNone(result["probabilities"])

    def test_slot_swap_is_invariant_for_known_window(self) -> None:
        with np.load(ROOT / "examples" / "pose_windows" / "smoke_window.npz") as source:
            points = source["keypoints_xy_normalized"]
            hands = source["hand_present"]
            width, height, fps = int(source["width"]), int(source["height"]), float(source["fps"])
        original = build_pose_features_v2(points, hands, width=width, height=height, fps=fps)
        swapped = build_pose_features_v2(
            points[:, ::-1], hands[:, ::-1], width=width, height=height, fps=fps
        )
        original_logits = self.classifier.logits(original)
        swapped_logits = self.classifier.logits(swapped)
        self.assertLess(float(np.max(np.abs(original_logits - swapped_logits))), 1e-5)

    def test_temporal_buffer_warmup_and_stride(self) -> None:
        buffer = TemporalPoseBuffer(fps=20.0, duration_seconds=1.5, stride_seconds=0.375)
        self.assertEqual(buffer.required_frames, 30)
        self.assertEqual(buffer.stride_frames, 8)
        points = np.full((2, 21, 2), np.nan, dtype=np.float32)
        hands = np.zeros(2, dtype=bool)
        tracks = np.full(2, -1, dtype=np.int32)
        for frame in range(buffer.required_frames - 1):
            buffer.append(points, hands, timestamp_s=frame / 20.0, track_ids=tracks)
        self.assertFalse(buffer.ready)
        buffer.append(points, hands, timestamp_s=(buffer.required_frames - 1) / 20.0, track_ids=tracks)
        self.assertTrue(buffer.ready)
        self.assertTrue(buffer.should_emit)
        for frame in range(1, buffer.stride_frames):
            buffer.append(points, hands, timestamp_s=(buffer.required_frames + frame) / 20.0, track_ids=tracks)
            self.assertFalse(buffer.should_emit)
        buffer.append(
            points,
            hands,
            timestamp_s=(buffer.required_frames + buffer.stride_frames) / 20.0,
            track_ids=tracks,
        )
        self.assertTrue(buffer.should_emit)

    @staticmethod
    def append_diagnostic_frame(
        buffer: TemporalPoseBuffer,
        timestamp_s: float,
        hand_present: tuple[bool, bool] = (False, False),
        track_ids: tuple[int, int] = (-1, -1),
        *,
        tracks_created_delta: int = 0,
        track_fragmentation_delta: int = 0,
    ) -> None:
        buffer.append(
            np.zeros((2, 21, 2), dtype=np.float32),
            np.asarray(hand_present, dtype=bool),
            timestamp_s=timestamp_s,
            track_ids=np.asarray(track_ids, dtype=np.int32),
            tracks_created_delta=tracks_created_delta,
            track_fragmentation_delta=track_fragmentation_delta,
        )

    def test_window_diagnostics_uses_real_regular_and_irregular_timestamps(self) -> None:
        regular = TemporalPoseBuffer(fps=4.0, duration_seconds=1.5, stride_seconds=0.375)
        for index in range(6):
            self.append_diagnostic_frame(regular, index * 0.1)

        regular_metrics = regular.window_diagnostics()
        self.assertTrue(regular_metrics["timing_valid"])
        self.assertEqual(regular_metrics["window_start_s"], 0.0)
        self.assertEqual(regular_metrics["window_end_s"], 0.5)
        self.assertEqual(regular_metrics["window_span_s"], 0.5)
        self.assertEqual(regular_metrics["effective_window_fps"], 10.0)
        self.assertAlmostEqual(regular_metrics["mean_frame_dt_ms"], 100.0)
        self.assertAlmostEqual(regular_metrics["min_frame_dt_ms"], 100.0)
        self.assertAlmostEqual(regular_metrics["max_frame_dt_ms"], 100.0)

        irregular = TemporalPoseBuffer(fps=4.0, duration_seconds=1.5, stride_seconds=0.375)
        for timestamp in (0.0, 0.1, 0.3, 0.35, 0.6, 1.0):
            self.append_diagnostic_frame(irregular, timestamp)

        irregular_metrics = irregular.window_diagnostics()
        self.assertTrue(irregular_metrics["timing_valid"])
        self.assertEqual(irregular_metrics["window_span_s"], 1.0)
        self.assertEqual(irregular_metrics["effective_window_fps"], 5.0)
        self.assertAlmostEqual(irregular_metrics["mean_frame_dt_ms"], 200.0)
        self.assertAlmostEqual(irregular_metrics["min_frame_dt_ms"], 50.0)
        self.assertAlmostEqual(irregular_metrics["max_frame_dt_ms"], 400.0)

    def test_window_diagnostics_invalid_timestamps_are_safe(self) -> None:
        for timestamps in (
            (0.0, 0.1, 0.2, 0.15, 0.3, 0.4),
            (0.0, 0.1, float("nan"), 0.3, 0.4, 0.5),
        ):
            buffer = TemporalPoseBuffer(fps=4.0, duration_seconds=1.5, stride_seconds=0.375)
            for timestamp in timestamps:
                self.append_diagnostic_frame(buffer, timestamp)

            metrics = buffer.window_diagnostics()
            self.assertFalse(metrics["timing_valid"])
            for key in (
                "window_start_s",
                "window_end_s",
                "window_span_s",
                "effective_window_fps",
                "mean_frame_dt_ms",
                "min_frame_dt_ms",
                "max_frame_dt_ms",
            ):
                self.assertIsNone(metrics[key])

    def test_window_diagnostics_counts_local_pose_and_tracking_only(self) -> None:
        buffer = TemporalPoseBuffer(fps=8.0, duration_seconds=1.0, stride_seconds=0.375)
        frames = (
            ((False, False), (-1, -1), 0, 0),
            ((True, False), (10, -1), 1, 0),
            ((True, True), (10, 11), 0, 0),
            ((True, True), (10, 11), 0, 1),
            ((True, False), (10, -1), 0, 0),
            ((True, True), (12, -1), 1, 0),
            ((True, True), (12, 13), 0, 2),
            ((False, False), (-1, -1), 0, 0),
        )
        for index, (hands, tracks, created, fragmented) in enumerate(frames):
            self.append_diagnostic_frame(
                buffer,
                index * 0.1,
                hands,
                tracks,
                tracks_created_delta=created,
                track_fragmentation_delta=fragmented,
            )

        metrics = buffer.window_diagnostics()
        self.assertEqual(metrics["frames_0_hands"], 2)
        self.assertEqual(metrics["frames_1_hand"], 2)
        self.assertEqual(metrics["frames_2_hands"], 4)
        self.assertEqual(metrics["fraction_2_hands"], 0.5)
        self.assertEqual(metrics["longest_two_hand_segment_frames"], 2)
        self.assertEqual(metrics["hand_dropout_count"], 2)
        self.assertEqual(metrics["track_changes_in_window"], 6)
        self.assertEqual(metrics["slot_changes_in_window"], 1)
        self.assertEqual(metrics["unique_tracks_in_window"], 4)
        self.assertEqual(metrics["new_tracks_in_window"], 2)
        self.assertEqual(metrics["track_fragmentations_in_window"], 3)

        arrays = buffer.arrays()
        self.assertEqual(len(arrays), 4)
        self.assertEqual(arrays[1].shape, (8, 2))
        self.assertEqual(arrays[3].shape, (8, 2))

    def test_streaming_emission_interval_resets_and_uses_window_end(self) -> None:
        class PoseBackend:
            def __init__(self):
                self.reset_calls = 0

            def process(self, frame, timestamp_ms):
                return []

            def reset(self):
                self.reset_calls += 1

        recognizer = StreamingHandwashingRecognizer.__new__(StreamingHandwashingRecognizer)
        recognizer.fps = 4.0
        recognizer.width = 8
        recognizer.height = 8
        recognizer.pose_api = "fake"
        recognizer.pose_backend = PoseBackend()
        recognizer.tracker = StatefulHandTracker()
        recognizer.buffer = TemporalPoseBuffer(fps=4.0, duration_seconds=1.5, stride_seconds=0.375)
        recognizer.classifier = None
        recognizer._frame_index = 0
        recognizer._latest_pose_snapshot = None
        recognizer._last_emission_window_end_s = None
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        result = None
        with patch(
            "laia_inference.api.predict_pose_window",
            return_value={"status": "prediction", "class_id": 4},
        ):
            for index in range(6):
                result = recognizer.update_frame(frame, timestamp_s=index * 0.1)
            self.assertEqual(result["status"], "prediction")
            self.assertEqual(result["window_start_s"], 0.0)
            self.assertEqual(result["window_end_s"], 0.5)
            self.assertEqual(result["window_span_s"], 0.5)
            self.assertEqual(result["effective_window_fps"], 10.0)
            self.assertIsNone(result["prediction_interval_ms"])

            recognizer.update_frame(frame, timestamp_s=0.6)
            second = recognizer.update_frame(frame, timestamp_s=0.7)
            self.assertEqual(second["window_end_s"], 0.7)
            self.assertAlmostEqual(second["prediction_interval_ms"], 200.0)

            recognizer.reset_temporal_state()
            for index in range(6):
                result = recognizer.update_frame(frame, timestamp_s=1.0 + index * 0.1)
            self.assertIsNone(result["prediction_interval_ms"])
            self.assertEqual(recognizer.pose_backend.reset_calls, 1)

    def test_streaming_invalid_timestamp_does_not_crash_or_invent_timing(self) -> None:
        class PoseBackend:
            def __init__(self):
                self.timestamps = []

            def process(self, frame, timestamp_ms):
                self.timestamps.append(timestamp_ms)
                return []

            def reset(self):
                pass

        recognizer = StreamingHandwashingRecognizer.__new__(StreamingHandwashingRecognizer)
        recognizer.fps = 20.0
        recognizer.width = 8
        recognizer.height = 8
        recognizer.pose_api = "fake"
        recognizer.pose_backend = PoseBackend()
        recognizer.tracker = StatefulHandTracker()
        recognizer.buffer = TemporalPoseBuffer(fps=20.0, duration_seconds=1.5, stride_seconds=0.375)
        recognizer.classifier = None
        recognizer._frame_index = 0
        recognizer._latest_pose_snapshot = None
        recognizer._last_emission_window_end_s = None
        recognizer._last_pose_timestamp_s = None
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        recognizer.update_frame(frame, timestamp_s=1.0)
        result = recognizer.update_frame(frame, timestamp_s=float("nan"))

        self.assertEqual(result["status"], "warming_up")
        self.assertEqual(recognizer.pose_backend.timestamps, [1000])
        self.assertTrue(np.isnan(recognizer.latest_pose_snapshot.timestamp_s))

    def test_tracker_reset_and_non_anatomical_slots(self) -> None:
        tracker = StatefulHandTracker()
        first = tracker.update([detection(0.2), detection(0.7)])
        self.assertEqual([item.track_id for item in first], [0, 1])
        points, present, tracks = detections_to_slots(tracker.update([detection(0.71)]))
        self.assertEqual(present.tolist(), [True, False])
        self.assertEqual(tracks.tolist(), [1, -1])
        self.assertTrue(np.isfinite(points[0]).all())
        tracker.reset()
        restarted = tracker.update([detection(0.3)])
        self.assertEqual(restarted[0].track_id, 0)

    def test_tracker_counter_deltas_are_published_per_pose_frame(self) -> None:
        class PoseBackend:
            def process(self, frame, timestamp_ms):
                return []

            def reset(self):
                pass

        class CounterTracker:
            tracks_created = 4
            track_fragmentation = 2

            def update(self, detections):
                self.tracks_created += 2
                self.track_fragmentation += 1
                return []

            def reset(self):
                self.tracks_created = 0
                self.track_fragmentation = 0

        recognizer = StreamingHandwashingRecognizer.__new__(StreamingHandwashingRecognizer)
        recognizer.fps = 20.0
        recognizer.width = 8
        recognizer.height = 8
        recognizer.pose_api = "fake"
        recognizer.pose_backend = PoseBackend()
        recognizer.tracker = CounterTracker()
        recognizer.buffer = TemporalPoseBuffer(fps=20.0, duration_seconds=1.5, stride_seconds=0.375)
        recognizer.classifier = None
        recognizer._frame_index = 0
        recognizer._latest_pose_snapshot = None
        recognizer._last_emission_window_end_s = None

        recognizer.update_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            timestamp_s=0.0,
        )
        snapshot = recognizer.latest_pose_snapshot
        self.assertEqual(snapshot.tracks_created, 6)
        self.assertEqual(snapshot.track_fragmentation, 3)
        self.assertEqual(snapshot.tracks_created_delta, 2)
        self.assertEqual(snapshot.track_fragmentation_delta, 1)

    def test_known_pose_window_outputs_remain_unchanged(self) -> None:
        with np.load(ROOT / "examples" / "pose_windows" / "smoke_window.npz") as source:
            points = source["keypoints_xy_normalized"]
            hands = source["hand_present"]
            width, height, fps = int(source["width"]), int(source["height"]), float(source["fps"])

        result = predict_pose_window(
            self.classifier,
            points,
            hands,
            width=width,
            height=height,
            fps=fps,
        )
        self.assertEqual(result["class_id"], 6)
        expected_logits = np.asarray(
            [-5.809308052062988, -9.937564849853516, -19.53554344177246,
             -26.177461624145508, -23.83920669555664, -29.968473434448242,
             22.995426177978516],
            dtype=np.float32,
        )
        expected_probabilities = np.asarray(
            [3.0921665510226515e-13, 4.981770492684017e-15,
             3.380917481606032e-19, 4.410483203331737e-22,
             4.5706438280050076e-21, 9.955673903935219e-24, 1.0],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result["logits"], expected_logits, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            result["probabilities"],
            expected_probabilities,
            rtol=0.0,
            atol=1e-12,
        )

    def test_streaming_reset_temporal_state_does_not_recreate_camera(self) -> None:
        class Resettable:
            def __init__(self):
                self.calls = 0

            def reset(self):
                self.calls += 1

        recognizer = StreamingHandwashingRecognizer.__new__(StreamingHandwashingRecognizer)
        recognizer.pose_backend = Resettable()
        recognizer.tracker = Resettable()
        recognizer.buffer = Resettable()
        recognizer._frame_index = 42

        recognizer.reset_temporal_state()

        self.assertEqual(recognizer.pose_backend.calls, 1)
        self.assertEqual(recognizer.tracker.calls, 1)
        self.assertEqual(recognizer.buffer.calls, 1)
        self.assertEqual(recognizer._frame_index, 0)

    def test_pose_snapshot_is_published_without_a_second_inference(self) -> None:
        class PoseBackend:
            def __init__(self):
                self.calls = 0

            def process(self, frame, timestamp_ms):
                self.calls += 1
                return [detection(0.25)]

        class Buffer:
            ready = False
            collected_frames = 1
            required_frames = 30

            def append(self, points, hands, *, timestamp_s, track_ids):
                self.last = (points, hands, timestamp_s, track_ids)

        recognizer = StreamingHandwashingRecognizer.__new__(StreamingHandwashingRecognizer)
        recognizer.fps = 20.0
        recognizer.width = 640
        recognizer.height = 480
        recognizer.pose_api = "fake"
        recognizer.pose_backend = PoseBackend()
        recognizer.tracker = StatefulHandTracker()
        recognizer.buffer = Buffer()
        recognizer.classifier = None
        recognizer._frame_index = 0
        recognizer._latest_pose_snapshot = None

        result = recognizer.update_frame(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=1.25)

        self.assertEqual(result["status"], "warming_up")
        self.assertEqual(recognizer.pose_backend.calls, 1)
        snapshot = recognizer.latest_pose_snapshot
        self.assertIsInstance(snapshot, PoseSnapshot)
        self.assertEqual(snapshot.timestamp_s, 1.25)
        self.assertEqual(snapshot.source_frame_count, 1)
        self.assertEqual(snapshot.points_normalized.shape, (2, 21, 2))
        self.assertEqual(snapshot.hand_present.tolist(), [True, False])
        self.assertEqual(snapshot.track_ids.tolist(), [0, -1])
        self.assertFalse(snapshot.points_normalized.flags.writeable)
        self.assertFalse(snapshot.hand_present.flags.writeable)
        self.assertFalse(snapshot.track_ids.flags.writeable)
        self.assertIs(recognizer.latest_pose_snapshot, snapshot)
        self.assertEqual(recognizer.pose_backend.calls, 1)


if __name__ == "__main__":
    unittest.main()
