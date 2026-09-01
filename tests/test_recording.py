from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from app.recording import (
    FFmpegEncoder,
    FramePacket,
    PairedRecorder,
    PredictionSnapshot,
    StateSnapshot,
    annotate_frame,
)
from app.telemetry import ThermalSample


def packet(frame: np.ndarray, timestamp_s: float = 0.0, *, thermal: ThermalSample | None = None) -> FramePacket:
    return FramePacket(
        frame_bgr=frame,
        frame_timestamp_s=timestamp_s,
        session_id="20260901_101500",
        session_elapsed_s=timestamp_s,
        prediction_snapshot=PredictionSnapshot(
            status="prediction",
            class_id=2,
            class_name="palm_to_palm",
            confidence=0.83,
            pose_coverage_ge1=0.95,
            pose_coverage_2=0.75,
            timestamp_s=timestamp_s,
        ),
        prediction_is_new=True,
        state_snapshot=StateSnapshot(
            state="WASHING",
            expected_step=2,
            correct_streak=1,
            incorrect_streak=0,
            accepted_steps=(1,),
        ),
        thermal_snapshot=thermal,
        thermal_age_s=0.1 if thermal is not None else None,
    )


class FakeEncoder:
    def __init__(self, path, **_kwargs):
        self.final_path = Path(path)
        self.partial_path = Path(str(path) + ".partial")
        self.codec = "fake"
        self.failed = False
        self.failure_reason = None
        self.frames = 0

    def start(self):
        return True

    def write(self, frame):
        self.frames += 1
        return True

    def close(self):
        return True


class RecordingTests(unittest.TestCase):
    def test_annotation_is_a_copy_and_handles_nan_absent_and_stale_pose(self) -> None:
        frame = np.full((120, 160, 3), 80, dtype=np.uint8)
        original = frame.copy()
        pose = SimpleNamespace(
            timestamp_s=1.0,
            points_normalized=np.full((2, 21, 2), np.nan, dtype=np.float32),
            hand_present=np.asarray([True, False]),
            track_ids=np.asarray([7, -1]),
            track_fragmentation=2,
        )
        pose.points_normalized[0, 0] = (0.1, 0.1)
        output = annotate_frame(
            packet(frame, timestamp_s=1.0, thermal=ThermalSample(soc_temp_c=48.0, throttled_now=False, cpu_freq_mhz=1500.0)),
        )
        stale_output = annotate_frame(packet(frame, timestamp_s=2.0))
        pose_packet = FramePacket(
            frame_bgr=frame,
            frame_timestamp_s=1.0,
            session_id="20260901_101500",
            session_elapsed_s=1.0,
            pose_snapshot=pose,
            state_snapshot=StateSnapshot(state="WASHING", expected_step=1),
            thermal_age_s=16.0,
        )
        pose_output = annotate_frame(pose_packet)

        self.assertTrue(np.array_equal(frame, original))
        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(stale_output.shape, frame.shape)
        self.assertEqual(pose_output.shape, frame.shape)
        self.assertFalse(np.array_equal(output, frame))
        self.assertFalse(np.array_equal(pose_output, frame))

    def test_real_ffmpeg_writes_two_probeable_mp4s_from_same_sampled_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = PairedRecorder(
                root,
                recording_fps=10.0,
                queue_size=4,
                disk_min_bytes=0,
                disk_min_fraction=0.0,
                disk_emergency_bytes=0,
            )
            self.assertTrue(recorder.start("20260901_101500", started_monotonic_s=time.monotonic()))
            frames = []
            for index in range(5):
                frame = np.zeros((64, 96, 3), dtype=np.uint8)
                frame[:, :, 0] = index * 20
                frame[:, :, 1] = 40
                frames.append(frame.copy())
                recorder.offer_frame(packet(frame, index * 0.1))
            recorder.stop(wait=True)
            stats = recorder.stats()

            raw = root / "session_20260901_101500_raw.mp4"
            annotated = root / "session_20260901_101500_annotated.mp4"
            self.assertTrue(raw.is_file())
            self.assertTrue(annotated.is_file())
            self.assertEqual(stats["frames_sampling_selected"], 5)
            self.assertEqual(stats["frames_enqueued"], 5)
            self.assertEqual(stats["raw_frames_written"], stats["annotated_frames_written"])
            for path in (raw, annotated):
                probe = subprocess.run(
                    [
                        "/usr/bin/ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,codec_name,avg_frame_rate",
                        "-of",
                        "json",
                        str(path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn('"width": 96', probe.stdout)
                self.assertIn('"height": 64', probe.stdout)
                self.assertIn('"codec_name"', probe.stdout)
            self.assertFalse((root / "session_20260901_101500_raw.mp4.partial").exists())
            self.assertFalse((root / "session_20260901_101500_annotated.mp4.partial").exists())

    def test_offer_is_nonblocking_and_queue_drop_is_distinguished_from_sampling_skip(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingEncoder(FakeEncoder):
            def start(self):
                started.set()
                release.wait(1.0)
                return True

        with tempfile.TemporaryDirectory() as directory:
            recorder = PairedRecorder(
                Path(directory),
                recording_fps=10.0,
                queue_size=1,
                disk_min_bytes=0,
                disk_min_fraction=0.0,
                disk_emergency_bytes=0,
                encoder_factory=BlockingEncoder,
            )
            self.assertTrue(recorder.start("20260901_101501", started_monotonic_s=0.0))
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            self.assertTrue(recorder.offer_frame(packet(frame, 0.0)))
            self.assertTrue(started.wait(1.0))
            self.assertTrue(recorder.offer_frame(packet(frame, 0.1)))
            started_at = time.perf_counter()
            self.assertFalse(recorder.offer_frame(packet(frame, 0.2)))
            elapsed = time.perf_counter() - started_at
            self.assertLess(elapsed, 0.05)
            self.assertEqual(recorder.counters.frames_dropped_queue, 1)
            self.assertEqual(recorder.counters.frames_sampling_skipped, 0)
            recorder.stop(wait=False)
            release.set()
            recorder.join(2.0)
            self.assertFalse(recorder._thread.is_alive())

    def test_ffmpeg_uses_safe_argv_and_falls_back_to_mpeg4(self) -> None:
        calls = []

        class Process:
            def __init__(self):
                self.stdin = SimpleNamespace(write=lambda _data: None, flush=lambda: None, close=lambda: None)
                self.returncode = 0

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            if "libx264" in command:
                raise OSError("encoder unavailable")
            return Process()

        with tempfile.TemporaryDirectory() as directory:
            encoder = FFmpegEncoder(
                Path(directory) / "output.mp4",
                width=32,
                height=24,
                fps=10,
                popen_factory=popen,
            )
            self.assertTrue(encoder.start())
            self.assertEqual(encoder.codec, "mpeg4")
            self.assertTrue(encoder.write(np.zeros((24, 32, 3), dtype=np.uint8)))
            encoder.close()

        self.assertEqual(len(calls), 2)
        command, kwargs = calls[1]
        self.assertFalse(kwargs["shell"])
        self.assertIn("pipe:0", command)
        self.assertIn("frag_keyframe+empty_moov+default_base_moof", command)
        self.assertIn("-n", command)

    def test_low_disk_skips_recording_without_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            usage = lambda _path: SimpleNamespace(total=1000, used=950, free=50)
            recorder = PairedRecorder(
                Path(directory),
                disk_usage=usage,
                disk_min_bytes=100,
                disk_min_fraction=0.05,
            )

            self.assertFalse(recorder.start("20260901_101502"))
            self.assertEqual(recorder.status, "skipped_low_disk")
            self.assertFalse(list(Path(directory).glob("*.mp4*")))


if __name__ == "__main__":
    unittest.main()
