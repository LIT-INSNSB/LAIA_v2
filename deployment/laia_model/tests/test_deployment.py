from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from laia_inference.api import predict_pose_window
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


if __name__ == "__main__":
    unittest.main()
