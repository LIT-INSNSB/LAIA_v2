"""Paired raw/annotated diagnostic recording outside the inference/UI threads."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping

import cv2
import numpy as np

from .telemetry import ThermalSample


LOGGER = logging.getLogger("laia.diagnostics.recording")

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_text(value: object, default: str = "none") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _format_float(value: object, digits: int = 3) -> str:
    number = _finite_float(value)
    return "none" if number is None else f"{number:.{digits}f}"


@dataclass(frozen=True)
class PredictionSnapshot:
    status: str = "none"
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None
    pose_coverage_ge1: float | None = None
    pose_coverage_2: float | None = None
    timestamp_s: float | None = None

    @classmethod
    def from_prediction(
        cls,
        prediction: Mapping[str, object] | None,
        *,
        timestamp_s: float | None,
    ) -> "PredictionSnapshot | None":
        if prediction is None:
            return None
        status = _safe_text(prediction.get("status"))
        class_id = _safe_int(prediction.get("class_id")) if status == "prediction" else None
        class_name = (
            _safe_text(prediction.get("class_name")) if status == "prediction" else None
        )
        return cls(
            status=status,
            class_id=class_id,
            class_name=class_name,
            confidence=_finite_float(prediction.get("confidence_uncalibrated")),
            pose_coverage_ge1=_finite_float(prediction.get("pose_coverage_ge1")),
            pose_coverage_2=_finite_float(prediction.get("pose_coverage_2")),
            timestamp_s=_finite_float(timestamp_s),
        )


@dataclass(frozen=True)
class StateSnapshot:
    state: str = "none"
    expected_step: int | None = None
    correct_streak: int = 0
    incorrect_streak: int = 0
    accepted_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class FramePacket:
    """An immutable diagnostic boundary crossing for one sampled camera frame."""

    frame_bgr: np.ndarray
    frame_timestamp_s: float
    session_id: str
    session_elapsed_s: float
    pose_snapshot: object | None = None
    prediction_snapshot: PredictionSnapshot | None = None
    prediction_is_new: bool = False
    state_snapshot: StateSnapshot = field(default_factory=StateSnapshot)
    thermal_snapshot: ThermalSample | None = None
    thermal_age_s: float | None = None
    event_names: tuple[str, ...] = ()


@dataclass
class RecordingCounters:
    frames_seen: int = 0
    frames_sampling_selected: int = 0
    frames_enqueued: int = 0
    frames_dropped_queue: int = 0
    frames_sampling_skipped: int = 0
    raw_frames_written: int = 0
    annotated_frames_written: int = 0
    queue_high_watermark: int = 0


def _freeze_frame(frame: np.ndarray) -> np.ndarray:
    copied = np.array(frame, dtype=np.uint8, copy=True, order="C")
    if copied.ndim != 3 or copied.shape[2] != 3:
        raise ValueError("diagnostic frame must be uint8 HxWx3")
    copied.setflags(write=False)
    return copied


class FFmpegEncoder:
    """One safe-argv FFmpeg process writing a single MP4 output."""

    def __init__(
        self,
        final_path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        bitrate: str = "2M",
        threads: int = 2,
        ffmpeg_path: str = "ffmpeg",
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.final_path = Path(final_path)
        self.partial_path = Path(str(self.final_path) + ".partial")
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.bitrate = str(bitrate)
        self.threads = max(1, int(threads))
        self.ffmpeg_path = ffmpeg_path
        self.popen_factory = popen_factory
        self.process: subprocess.Popen | None = None
        self.codec: str | None = None
        self.failed = False
        self.failure_reason: str | None = None
        self.frames_written = 0

    def _command(self, codec: str) -> list[str]:
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec,
        ]
        if codec == "libx264":
            command.extend(("-preset", "ultrafast"))
        command.extend(
            (
                "-threads",
                str(self.threads),
                "-b:v",
                self.bitrate,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "frag_keyframe+empty_moov+default_base_moof",
                "-f",
                "mp4",
                "-n",
                str(self.partial_path),
            )
        )
        return command

    def start(self) -> bool:
        if self.process is not None:
            return not self.failed
        if self.final_path.exists() or self.partial_path.exists():
            self.failed = True
            self.failure_reason = "output_exists"
            return False
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        for codec in ("libx264", "mpeg4"):
            try:
                process = self.popen_factory(
                    self._command(codec),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.failure_reason = str(exc)
                continue
            if process.poll() not in (None, 0):
                self.failure_reason = f"{codec}_exited_{process.returncode}"
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except OSError:
                    pass
                continue
            self.process = process
            self.codec = codec
            return True
        self.failed = True
        if self.failure_reason is None:
            self.failure_reason = "ffmpeg_unavailable"
        return False

    def write(self, frame: np.ndarray) -> bool:
        if self.process is None or self.failed or self.process.stdin is None:
            return False
        if self.process.poll() is not None:
            self.failed = True
            self.failure_reason = f"process_exited_{self.process.returncode}"
            return False
        try:
            contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
            self.process.stdin.write(contiguous.tobytes(order="C"))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.failed = True
            self.failure_reason = str(exc)
            return False
        self.frames_written += 1
        return True

    def close(self) -> bool:
        process = self.process
        if process is None:
            return False
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            return_code = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
                return_code = process.wait(timeout=2.0)
            except (OSError, subprocess.SubprocessError):
                return_code = -1
        if return_code != 0:
            self.failed = True
            self.failure_reason = self.failure_reason or f"ffmpeg_exit_{return_code}"
        if not self.failed and return_code == 0 and self.partial_path.exists():
            try:
                if self.final_path.exists():
                    raise FileExistsError(str(self.final_path))
                self.partial_path.rename(self.final_path)
            except OSError as exc:
                self.failed = True
                self.failure_reason = str(exc)
        return not self.failed and self.final_path.exists()


def _prediction_age_ms(packet: FramePacket) -> float | None:
    if packet.prediction_snapshot is None or packet.prediction_snapshot.timestamp_s is None:
        return None
    age = (packet.frame_timestamp_s - packet.prediction_snapshot.timestamp_s) * 1000.0
    return max(0.0, age) if math.isfinite(age) else None


def _pose_is_stale(packet: FramePacket, max_age_s: float) -> bool:
    pose = packet.pose_snapshot
    timestamp = _finite_float(getattr(pose, "timestamp_s", None)) if pose is not None else None
    frame_timestamp = _finite_float(packet.frame_timestamp_s)
    if timestamp is None or frame_timestamp is None:
        return True
    return abs(frame_timestamp - timestamp) > max(0.0, max_age_s)


def _slot_color(slot: int, track_id: int | None) -> tuple[int, int, int]:
    palette = ((255, 170, 30), (50, 210, 120), (210, 90, 240), (60, 190, 240))
    index = (track_id if track_id is not None and track_id >= 0 else slot) % len(palette)
    return palette[index]


def _overlay_line(lines: list[str], key: str, value: object) -> None:
    lines.append(f"{key}={_safe_text(value)}")


def annotate_frame(
    packet: FramePacket,
    *,
    thermal_stale_seconds: float = 15.0,
    pose_stale_seconds: float = 0.25,
) -> np.ndarray:
    """Draw a safe compact diagnostic overlay onto a copy of one packet frame."""

    output = np.array(packet.frame_bgr, dtype=np.uint8, copy=True, order="C")
    height, width = output.shape[:2]
    pose_stale = _pose_is_stale(packet, pose_stale_seconds)
    pose = packet.pose_snapshot
    if pose is not None and not pose_stale:
        points = np.asarray(getattr(pose, "points_normalized", ()), dtype=np.float32)
        hands = np.asarray(getattr(pose, "hand_present", ()), dtype=bool).reshape(-1)
        track_ids = np.asarray(getattr(pose, "track_ids", ()), dtype=np.int32).reshape(-1)
        if points.ndim == 3 and points.shape[1] >= 21 and points.shape[2] >= 2:
            for slot in range(min(2, points.shape[0])):
                if slot >= hands.size or not bool(hands[slot]):
                    continue
                track_id = int(track_ids[slot]) if slot < track_ids.size else None
                color = _slot_color(slot, track_id)
                pixels: dict[int, tuple[int, int]] = {}
                for index in range(21):
                    x = _finite_float(points[slot, index, 0])
                    y = _finite_float(points[slot, index, 1])
                    if x is None or y is None:
                        continue
                    x_pixel = int(round(min(1.0, max(0.0, x)) * (width - 1)))
                    y_pixel = int(round(min(1.0, max(0.0, y)) * (height - 1)))
                    pixels[index] = (x_pixel, y_pixel)
                for start, end in HAND_CONNECTIONS:
                    if start in pixels and end in pixels:
                        cv2.line(output, pixels[start], pixels[end], color, 1, cv2.LINE_AA)
                for point in pixels.values():
                    cv2.circle(output, point, 2, color, -1, cv2.LINE_AA)

    state = packet.state_snapshot
    prediction = packet.prediction_snapshot
    thermal = packet.thermal_snapshot
    thermal_age = _finite_float(packet.thermal_age_s)
    thermal_stale = thermal is None or thermal_age is None or thermal_age > thermal_stale_seconds
    if thermal_stale:
        soc = "none"
        throttle = "none"
        cpu = "none"
    else:
        soc = _format_float(getattr(thermal, "soc_temp_c", None), 1)
        throttle_value = getattr(thermal, "throttled_now", None)
        throttle = "YES" if throttle_value is True else "NO" if throttle_value is False else "none"
        frequency = _finite_float(getattr(thermal, "cpu_freq_mhz", None))
        cpu = "none" if frequency is None else f"{frequency / 1000.0:.2f}"

    if prediction is None:
        class_value = "none"
        class_name = "none"
        confidence = "none"
        prediction_kind = "none"
    else:
        class_value = _safe_text(prediction.class_id)
        class_name = _safe_text(prediction.class_name)
        confidence = _format_float(prediction.confidence)
        prediction_kind = "NEW PRED" if packet.prediction_is_new else "LAST PRED"

    pose_age = None
    if pose is not None:
        pose_timestamp = _finite_float(getattr(pose, "timestamp_s", None))
        if pose_timestamp is not None:
            pose_age = max(0.0, (packet.frame_timestamp_s - pose_timestamp) * 1000.0)

    lines = [
        f"session_id={_safe_text(packet.session_id)} t={_format_float(packet.session_elapsed_s, 2)}",
        f"state={_safe_text(state.state)} expected={_safe_text(state.expected_step)}",
        f"pred={class_value} {class_name}",
        f"conf={confidence} {prediction_kind}",
        f"pose1={_format_float(prediction.pose_coverage_ge1 if prediction else None)} "
        f"pose2={_format_float(prediction.pose_coverage_2 if prediction else None)}",
        f"streak={_safe_text(state.correct_streak)}/{_safe_text(state.incorrect_streak)}",
        f"accepted={','.join(map(str, state.accepted_steps)) or 'none'}",
        f"track_frag={_safe_text(getattr(pose, 'track_fragmentation', None) if pose else None)}",
        f"SoC={soc} C throttle={throttle} CPU={cpu}GHz",
    ]
    if pose_stale:
        lines.append("pose_stale")
    elif pose_age is not None:
        lines.append(f"pose_age_ms={pose_age:.1f}")
    if thermal_stale:
        lines.append("thermal stale")
    elif thermal_age is not None:
        lines.append(f"thermal_age_s={thermal_age:.1f}")

    line_height = 13
    box_height = line_height * len(lines) + 8
    overlay = output.copy()
    cv2.rectangle(overlay, (4, 4), (min(width - 5, 470), min(height - 5, box_height)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, output, 0.38, 0.0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (8, 16 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (235, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return output


class PairedRecorder:
    """Bounded worker that writes the same sampled packets to two outputs."""

    def __init__(
        self,
        recordings_dir: Path,
        *,
        recording_fps: float = 10.0,
        queue_size: int = 4,
        bitrate: str = "2M",
        ffmpeg_threads: int = 2,
        ffmpeg_path: str = "ffmpeg",
        thermal_stale_seconds: float = 15.0,
        disk_check_interval_seconds: float = 60.0,
        disk_min_bytes: int = 2 * 1024**3,
        disk_min_fraction: float = 0.05,
        disk_emergency_bytes: int = 1 * 1024**3,
        disk_usage: Callable[[str | bytes | Path], shutil._ntuple_diskusage] = shutil.disk_usage,
        encoder_factory: Callable[..., FFmpegEncoder] = FFmpegEncoder,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.recordings_dir = Path(recordings_dir)
        self.recording_fps = max(0.1, float(recording_fps))
        self.queue_size = max(1, int(queue_size))
        self.bitrate = str(bitrate)
        self.ffmpeg_threads = max(1, int(ffmpeg_threads))
        self.ffmpeg_path = ffmpeg_path
        self.thermal_stale_seconds = max(0.0, float(thermal_stale_seconds))
        self.disk_check_interval_seconds = max(0.1, float(disk_check_interval_seconds))
        self.disk_min_bytes = max(0, int(disk_min_bytes))
        self.disk_min_fraction = max(0.0, float(disk_min_fraction))
        self.disk_emergency_bytes = max(0, int(disk_emergency_bytes))
        self.disk_usage = disk_usage
        self.encoder_factory = encoder_factory
        self.clock = clock
        self.data_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=self.queue_size)
        self.control_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.counters = RecordingCounters()
        self.session_id: str | None = None
        self.raw_path: Path | None = None
        self.annotated_path: Path | None = None
        self.status = "idle"
        self.stop_reason: str | None = None
        self.failure_warnings: list[str] = []
        self.session_started_s: float | None = None
        self.session_duration_s: float | None = None
        self._next_sample_s: float | None = None
        self._last_disk_check_s: float | None = None
        self._accepting = False
        self._lock = threading.Lock()
        self._stop_requested = False
        self._drain_on_stop = True
        self._thread: threading.Thread | None = None
        self._raw_encoder: FFmpegEncoder | None = None
        self._annotated_encoder: FFmpegEncoder | None = None

    def _free_bytes(self) -> int | None:
        try:
            usage = self.disk_usage(self.recordings_dir)
            return int(usage.free)
        except (OSError, AttributeError, TypeError, ValueError):
            return None

    def _required_free_bytes(self) -> int | None:
        try:
            usage = self.disk_usage(self.recordings_dir)
            total = int(usage.total)
        except (OSError, AttributeError, TypeError, ValueError):
            return None
        return max(self.disk_min_bytes, int(total * self.disk_min_fraction))

    def _paths_available(self, session_id: str) -> bool:
        raw = self.recordings_dir / f"session_{session_id}_raw.mp4"
        annotated = self.recordings_dir / f"session_{session_id}_annotated.mp4"
        return not any(
            path.exists() or Path(str(path) + ".partial").exists() for path in (raw, annotated)
        )

    def start(self, session_id: str, *, started_monotonic_s: float | None = None) -> bool:
        with self._lock:
            if self._accepting or (self._thread is not None and self._thread.is_alive()):
                return False
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            session_id = str(session_id)
            if not self._paths_available(session_id):
                self.status = "skipped_collision"
                self.stop_reason = "output_exists"
                return False
            required = self._required_free_bytes()
            free = self._free_bytes()
            if required is not None and free is not None and free < required:
                self.status = "skipped_low_disk"
                self.stop_reason = "low_disk_before_start"
                return False
            self.session_id = session_id
            self.raw_path = self.recordings_dir / f"session_{session_id}_raw.mp4"
            self.annotated_path = self.recordings_dir / f"session_{session_id}_annotated.mp4"
            self.session_started_s = self.clock() if started_monotonic_s is None else float(started_monotonic_s)
            self.session_duration_s = None
            self._next_sample_s = None
            self._last_disk_check_s = self.session_started_s
            self.counters = RecordingCounters()
            self.stop_reason = None
            self.failure_warnings = []
            self._stop_requested = False
            self._drain_on_stop = True
            self._accepting = True
            self.status = "recording"
            self.data_queue = queue.Queue(maxsize=self.queue_size)
            self.control_queue = queue.Queue()
            self._thread = threading.Thread(target=self._worker, name="laia-recorder", daemon=True)
            self._thread.start()
            # START is kept separate from the bounded data queue by design.
            self.control_queue.put(("start", None))
            return True

    def _make_packet(self, packet: FramePacket) -> FramePacket:
        return FramePacket(
            frame_bgr=_freeze_frame(packet.frame_bgr),
            frame_timestamp_s=float(packet.frame_timestamp_s),
            session_id=str(packet.session_id),
            session_elapsed_s=max(0.0, float(packet.session_elapsed_s)),
            pose_snapshot=packet.pose_snapshot,
            prediction_snapshot=packet.prediction_snapshot,
            prediction_is_new=bool(packet.prediction_is_new),
            state_snapshot=packet.state_snapshot,
            thermal_snapshot=packet.thermal_snapshot,
            thermal_age_s=packet.thermal_age_s,
            event_names=tuple(packet.event_names),
        )

    def offer_frame(
        self,
        packet: FramePacket,
        **_ignored: object,
    ) -> bool:
        """Offer one frame without ever waiting on the worker queue."""

        with self._lock:
            if not self._accepting:
                return False
            self.counters.frames_seen += 1
            timestamp = _finite_float(packet.frame_timestamp_s)
            if timestamp is None:
                timestamp = self.clock()
            period = 1.0 / self.recording_fps
            if self._next_sample_s is not None and timestamp < self._next_sample_s:
                self.counters.frames_sampling_skipped += 1
                return False
            self.counters.frames_sampling_selected += 1
            self._next_sample_s = timestamp + period
            copied_packet = self._make_packet(packet)
            try:
                self.data_queue.put_nowait(copied_packet)
            except queue.Full:
                self.counters.frames_dropped_queue += 1
                return False
            self.counters.frames_enqueued += 1
            self.counters.queue_high_watermark = max(
                self.counters.queue_high_watermark, self.data_queue.qsize()
            )
            return True

    def _request_stop(self, reason: str, *, drain: bool) -> None:
        with self._lock:
            if not self._stop_requested:
                self._stop_requested = True
                self._drain_on_stop = bool(drain)
                self._accepting = False
                self.stop_reason = reason
                self.control_queue.put(("stop", reason))

    def stop(self, reason: str = "terminal", *, wait: bool = False, timeout: float = 10.0) -> None:
        self._request_stop(reason, drain=True)
        if wait:
            self.join(timeout)

    def _check_disk(self) -> None:
        now = self.clock()
        if self._last_disk_check_s is not None and now - self._last_disk_check_s < self.disk_check_interval_seconds:
            return
        self._last_disk_check_s = now
        free = self._free_bytes()
        if free is not None and free < self.disk_emergency_bytes:
            self._request_stop("emergency_low_disk", drain=False)

    def _ensure_encoders(self, packet: FramePacket) -> None:
        if self.raw_path is None or self.annotated_path is None:
            return
        height, width = packet.frame_bgr.shape[:2]
        if self._raw_encoder is None:
            self._raw_encoder = self.encoder_factory(
                self.raw_path,
                width=width,
                height=height,
                fps=self.recording_fps,
                bitrate=self.bitrate,
                threads=self.ffmpeg_threads,
                ffmpeg_path=self.ffmpeg_path,
            )
            if not self._raw_encoder.start():
                self.failure_warnings.append(f"raw:{self._raw_encoder.failure_reason or 'start_failed'}")
        if self._annotated_encoder is None:
            self._annotated_encoder = self.encoder_factory(
                self.annotated_path,
                width=width,
                height=height,
                fps=self.recording_fps,
                bitrate=self.bitrate,
                threads=self.ffmpeg_threads,
                ffmpeg_path=self.ffmpeg_path,
            )
            if not self._annotated_encoder.start():
                self.failure_warnings.append(
                    f"annotated:{self._annotated_encoder.failure_reason or 'start_failed'}"
                )

    def _write_packet(self, packet: FramePacket) -> None:
        self._ensure_encoders(packet)
        if self._raw_encoder is not None and self._raw_encoder.write(packet.frame_bgr):
            self.counters.raw_frames_written += 1
        if self._annotated_encoder is not None:
            annotated = annotate_frame(
                packet,
                thermal_stale_seconds=self.thermal_stale_seconds,
            )
            if self._annotated_encoder.write(annotated):
                self.counters.annotated_frames_written += 1

    def _close_encoders(self) -> None:
        for label, encoder in (("raw", self._raw_encoder), ("annotated", self._annotated_encoder)):
            if encoder is None:
                continue
            try:
                encoder.close()
            except Exception as exc:
                self.failure_warnings.append(f"{label}:{exc}")
            if encoder.failed and encoder.failure_reason:
                warning = f"{label}:{encoder.failure_reason}"
                if warning not in self.failure_warnings:
                    self.failure_warnings.append(warning)
        if self.session_started_s is not None:
            self.session_duration_s = max(0.0, self.clock() - self.session_started_s)
        if self.status == "recording":
            self.status = "completed" if self.counters.frames_enqueued else "stopped_no_frames"

    def _worker(self) -> None:
        try:
            while True:
                try:
                    command, reason = self.control_queue.get_nowait()
                    if command == "stop":
                        self._stop_requested = True
                        self._drain_on_stop = reason != "emergency_low_disk"
                except queue.Empty:
                    pass
                self._check_disk()
                if self._stop_requested and not self._drain_on_stop:
                    while True:
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            break
                    break
                try:
                    packet = self.data_queue.get(timeout=0.05)
                except queue.Empty:
                    if self._stop_requested:
                        break
                    continue
                self._write_packet(packet)
                self.data_queue.task_done()
            self._close_encoders()
        except Exception as exc:
            LOGGER.exception("Error en worker de grabación diagnóstica")
            self.failure_warnings.append(f"worker:{exc}")
            self.status = "worker_error"
            self._close_encoders()

    def join(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))

    def close(self, timeout: float = 10.0) -> None:
        self.stop("application_close", wait=True, timeout=timeout)

    def stats(self) -> dict[str, object]:
        with self._lock:
            result = {
                "session_id": self.session_id or "none",
                "recording_status": self.status,
                "stop_reason": self.stop_reason or "none",
                "frames_seen": self.counters.frames_seen,
                "frames_sampling_selected": self.counters.frames_sampling_selected,
                "frames_sampling_skipped": self.counters.frames_sampling_skipped,
                "frames_enqueued": self.counters.frames_enqueued,
                "frames_dropped_queue": self.counters.frames_dropped_queue,
                "raw_frames_written": self.counters.raw_frames_written,
                "annotated_frames_written": self.counters.annotated_frames_written,
                "queue_high_watermark": self.counters.queue_high_watermark,
                "recording_fps": self.recording_fps,
                "session_duration_s": self.session_duration_s,
                "raw_path": str(self.raw_path) if self.raw_path else "none",
                "annotated_path": str(self.annotated_path) if self.annotated_path else "none",
                "failure_warnings": tuple(self.failure_warnings),
            }
        duration = _finite_float(result["session_duration_s"])
        result["raw_effective_fps"] = (
            result["raw_frames_written"] / duration if duration and duration > 0 else None
        )
        result["annotated_effective_fps"] = (
            result["annotated_frames_written"] / duration if duration and duration > 0 else None
        )
        result["raw_size_bytes"] = (
            self.raw_path.stat().st_size if self.raw_path is not None and self.raw_path.exists() else 0
        )
        result["annotated_size_bytes"] = (
            self.annotated_path.stat().st_size
            if self.annotated_path is not None and self.annotated_path.exists()
            else 0
        )
        result["raw_partial_size_bytes"] = (
            Path(str(self.raw_path) + ".partial").stat().st_size
            if self.raw_path is not None and Path(str(self.raw_path) + ".partial").exists()
            else 0
        )
        result["annotated_partial_size_bytes"] = (
            Path(str(self.annotated_path) + ".partial").stat().st_size
            if self.annotated_path is not None and Path(str(self.annotated_path) + ".partial").exists()
            else 0
        )
        result["codec_raw"] = self._raw_encoder.codec if self._raw_encoder is not None else "none"
        result["codec_annotated"] = (
            self._annotated_encoder.codec if self._annotated_encoder is not None else "none"
        )
        return result


RecordingWorker = PairedRecorder
