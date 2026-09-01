from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.config import DiagnosticsConfig
from app.diagnostics import DiagnosticsCoordinator, SessionIdAllocator
from app.state_machine import AppEvent, AppState
from app.telemetry import ThermalSample


class FakeRecorder:
    instances = []

    def __init__(self, recordings_dir, **_kwargs):
        self.recordings_dir = Path(recordings_dir)
        self.status = "idle"
        self.session_id = None
        self.offered = []
        self.reason = None
        FakeRecorder.instances.append(self)

    def start(self, session_id, **_kwargs):
        self.session_id = session_id
        self.status = "recording"
        return True

    def offer_frame(self, frame_packet):
        self.offered.append(frame_packet)
        return True

    def stop(self, reason, **_kwargs):
        self.reason = reason
        self.status = "completed"

    def stats(self):
        return {
            "session_id": self.session_id or "none",
            "recording_status": self.status,
            "stop_reason": self.reason or "none",
            "frames_seen": len(self.offered),
            "frames_sampling_selected": len(self.offered),
            "frames_enqueued": len(self.offered),
            "frames_dropped_queue": 0,
            "raw_frames_written": len(self.offered),
            "annotated_frames_written": len(self.offered),
            "queue_high_watermark": 1,
            "raw_effective_fps": 10.0,
            "annotated_effective_fps": 10.0,
            "session_duration_s": 1.0,
        }


class FakeThermalMonitor:
    def __init__(self, *, on_sample, **_kwargs):
        self.on_sample = on_sample
        self.latest_sample = None
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class DiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRecorder.instances.clear()

    def test_environment_overrides_are_separate_from_runtime_behavior(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LAIA_DIAGNOSTICS_ENABLED": "1",
                "LAIA_RECORDING_ENABLED": "0",
                "LAIA_THERMAL_ENABLED": "1",
            },
            clear=False,
        ):
            config = DiagnosticsConfig.from_environment()

        self.assertTrue(config.diagnostics_enabled)
        self.assertFalse(config.recording_enabled)
        self.assertTrue(config.thermal_enabled)
        self.assertEqual(config.recording_fps, 10.0)
        simulated = config.for_non_camera_mode()
        self.assertFalse(simulated.recording_enabled)
        self.assertFalse(simulated.thermal_enabled)

    def test_allocator_uses_suffix_without_overwriting_any_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings = root / "recordings"
            telemetry = root / "telemetry"
            recordings.mkdir()
            telemetry.mkdir()
            (recordings / "session_20260901_101500_raw.mp4").write_bytes(b"keep")
            (telemetry / "session_20260901_101500.csv.partial").write_bytes(b"keep")
            allocator = SessionIdAllocator(
                recordings,
                telemetry,
                now_factory=lambda: datetime(2026, 9, 1, 10, 15, 0),
            )

            self.assertEqual(allocator.allocate(), "20260901_101500_02")
            self.assertEqual((recordings / "session_20260901_101500_raw.mp4").read_bytes(), b"keep")

    def test_triggering_prediction_and_terminal_artifacts_share_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = DiagnosticsConfig(
                recording_enabled=True,
                thermal_enabled=False,
                disk_min_bytes=0,
                disk_min_fraction=0.0,
            )
            coordinator = DiagnosticsCoordinator(
                config,
                logs_root=Path(directory),
                recorder_factory=FakeRecorder,
            )
            machine = SimpleNamespace(
                state=AppState.WASHING,
                expected_step=2,
                correct_streak=1,
                incorrect_streak=0,
                accepted_steps={1},
            )
            events = [AppEvent("session_started")]
            session_id = coordinator.prepare_events(events)
            update = SimpleNamespace(
                frame_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
                frame_timestamp_s=0.25,
                pose_snapshot=None,
                update_frame_ms=12.0,
            )
            prediction = {
                "status": "prediction",
                "class_id": 2,
                "class_name": "palm_to_palm",
                "confidence_uncalibrated": 0.8,
                "pose_coverage_ge1": 1.0,
                "pose_coverage_2": 1.0,
            }
            self.assertTrue(
                coordinator.offer_runtime_update(
                    update,
                    prediction=prediction,
                    events=events,
                    machine=machine,
                )
            )
            recorder = FakeRecorder.instances[0]
            self.assertEqual(session_id, coordinator.session_id)
            self.assertEqual(recorder.offered[0].session_id, session_id)
            self.assertTrue(recorder.offered[0].prediction_is_new)
            self.assertEqual(recorder.offered[0].prediction_snapshot.class_id, 2)

            coordinator.finish_events([AppEvent("success", 6)])
            coordinator.wait_for_idle()
            self.assertEqual(recorder.reason, "success")
            self.assertEqual(coordinator.session_id, "none")

    def test_thermal_callback_writes_only_inside_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = DiagnosticsConfig(recording_enabled=False, thermal_enabled=True)
            coordinator = DiagnosticsCoordinator(
                config,
                logs_root=Path(directory),
                thermal_monitor_factory=FakeThermalMonitor,
            )
            sample = ThermalSample(
                timestamp_iso="2026-09-01T10:15:01+00:00",
                monotonic_s=10.5,
                soc_temp_c=49.0,
                throttled_now=False,
                cpu_freq_mhz=1500.0,
            )
            monitor = coordinator.thermal_monitor
            monitor.latest_sample = sample
            monitor.on_sample(sample)
            self.assertFalse(list((Path(directory) / "telemetry").glob("*.csv")))

            coordinator.prepare_events([AppEvent("session_started")])
            monitor.on_sample(sample)
            csv_files = list((Path(directory) / "telemetry").glob("*.csv"))
            self.assertEqual(len(csv_files), 1)
            self.assertIn(coordinator.session_id, csv_files[0].name)
            coordinator.finish("test", wait=True)
            self.assertEqual(coordinator.session_id, "none")


if __name__ == "__main__":
    unittest.main()
