"""Invisible session-correlated diagnostics orchestration for LAIA."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from pathlib import Path
import re
import threading
import time

from .config import DiagnosticsConfig, LOGS
from .recording import FramePacket, PairedRecorder, PredictionSnapshot, StateSnapshot
from .telemetry import TelemetrySession, ThermalMonitor, ThermalSample


LOGGER = logging.getLogger("laia.diagnostics")

_SESSION_ID = re.compile(r"^[0-9]{8}_[0-9]{6}(?:_[0-9]{2,})?$")
_TERMINAL_EVENTS = {"success", "attempt_incomplete", "reset"}


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _token(value: object) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    if not text:
        return "none"
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)


def _number(value: object, digits: int = 3) -> str:
    result = _finite_float(value)
    return "none" if result is None else f"{result:.{digits}f}"


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(1.0, max(0.0, percentile))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class SessionIdAllocator:
    """Generate local IDs and reserve no path by overwriting existing files."""

    def __init__(
        self,
        recordings_dir: Path,
        telemetry_dir: Path,
        *,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self.recordings_dir = Path(recordings_dir)
        self.telemetry_dir = Path(telemetry_dir)
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())

    def _candidate_exists(self, session_id: str) -> bool:
        paths = (
            self.recordings_dir / f"session_{session_id}_raw.mp4",
            self.recordings_dir / f"session_{session_id}_annotated.mp4",
            self.telemetry_dir / f"session_{session_id}.csv",
        )
        return any(
            path.exists() or Path(str(path) + ".partial").exists() for path in paths
        )

    def allocate(self) -> str:
        base = self.now_factory().astimezone().strftime("%Y%m%d_%H%M%S")
        candidate = base
        suffix = 2
        while self._candidate_exists(candidate):
            candidate = f"{base}_{suffix:02d}"
            suffix += 1
        if not _SESSION_ID.fullmatch(candidate):
            raise ValueError(f"invalid generated session id: {candidate}")
        return candidate


@dataclass
class SessionMetrics:
    started_monotonic_s: float
    runtime_frames: int = 0
    model_emissions: int = 0
    update_times_ms: deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    prediction_intervals_s: deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    _last_prediction_timestamp_s: float | None = None

    def observe_update(self, update: object, prediction: Mapping[str, object] | None) -> None:
        self.runtime_frames += 1
        update_time = _finite_float(getattr(update, "update_frame_ms", None))
        if update_time is not None and update_time >= 0:
            self.update_times_ms.append(update_time)
        if prediction is None or prediction.get("status") != "prediction":
            return
        self.model_emissions += 1
        timestamp = _finite_float(getattr(update, "frame_timestamp_s", None))
        if timestamp is not None:
            if self._last_prediction_timestamp_s is not None:
                interval = timestamp - self._last_prediction_timestamp_s
                if math.isfinite(interval) and interval >= 0:
                    self.prediction_intervals_s.append(interval)
            self._last_prediction_timestamp_s = timestamp

    def summary(self, now: float) -> dict[str, object]:
        duration = max(0.0, float(now) - self.started_monotonic_s)
        return {
            "runtime_frames": self.runtime_frames,
            "runtime_effective_fps": self.runtime_frames / duration if duration > 0 else None,
            "model_emissions": self.model_emissions,
            "update_ms_mean": (
                sum(self.update_times_ms) / len(self.update_times_ms)
                if self.update_times_ms
                else None
            ),
            "update_ms_p95": _percentile(self.update_times_ms, 0.95),
            "update_ms_max": max(self.update_times_ms) if self.update_times_ms else None,
            "prediction_interval_mean_s": (
                sum(self.prediction_intervals_s) / len(self.prediction_intervals_s)
                if self.prediction_intervals_s
                else None
            ),
            "prediction_interval_p95_s": _percentile(self.prediction_intervals_s, 0.95),
        }


@dataclass
class _Session:
    session_id: str
    started_monotonic_s: float
    telemetry: TelemetrySession | None
    recorder: PairedRecorder | None
    metrics: SessionMetrics
    latest_prediction: PredictionSnapshot | None = None
    finalizing: bool = False


class DiagnosticsCoordinator:
    """Own session boundaries and pass immutable packets to worker components."""

    def __init__(
        self,
        config: DiagnosticsConfig | None = None,
        *,
        logs_root: Path = LOGS,
        clock: Callable[[], float] = time.monotonic,
        allocator: SessionIdAllocator | None = None,
        recorder_factory: Callable[..., PairedRecorder] = PairedRecorder,
        thermal_monitor_factory: Callable[..., ThermalMonitor] = ThermalMonitor,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or DiagnosticsConfig.from_environment()
        self.logs_root = Path(logs_root)
        self.recordings_dir = self.logs_root / "recordings"
        self.telemetry_dir = self.logs_root / "telemetry"
        self.clock = clock
        self.allocator = allocator or SessionIdAllocator(self.recordings_dir, self.telemetry_dir)
        self.recorder_factory = recorder_factory
        self.thermal_monitor_factory = thermal_monitor_factory
        self.logger = logger or LOGGER
        self._lock = threading.RLock()
        self._session: _Session | None = None
        self._thermal_monitor: ThermalMonitor | None = None
        self._finalizer_threads: set[threading.Thread] = set()
        if self.config.diagnostics_enabled and self.config.thermal_enabled:
            self._thermal_monitor = self.thermal_monitor_factory(
                interval_seconds=self.config.thermal_interval_seconds,
                on_sample=self._on_thermal_sample,
            )
            self._thermal_monitor.start()

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session.session_id if self._session is not None else "none"

    @property
    def thermal_monitor(self) -> ThermalMonitor | None:
        return self._thermal_monitor

    def _new_session(self) -> _Session:
        session_id = self.allocator.allocate()
        started = self.clock()
        telemetry = None
        if self.config.thermal_enabled:
            telemetry = TelemetrySession(
                self.telemetry_dir / f"session_{session_id}.csv",
                session_id,
                started,
            )
            try:
                telemetry.open()
            except OSError as exc:
                self.logger.warning(
                    "Diagnostics telemetry unavailable: session_id=%s error=%s",
                    session_id,
                    exc,
                )
                telemetry = None
        recorder = None
        if self.config.recording_enabled:
            recorder = self.recorder_factory(
                self.recordings_dir,
                recording_fps=self.config.recording_fps,
                queue_size=self.config.recorder_queue_size,
                bitrate=self.config.ffmpeg_bitrate,
                ffmpeg_threads=self.config.ffmpeg_threads,
                thermal_stale_seconds=self.config.thermal_stale_seconds,
                disk_check_interval_seconds=self.config.disk_check_interval_seconds,
                disk_min_bytes=self.config.disk_min_bytes,
                disk_min_fraction=self.config.disk_min_fraction,
                disk_emergency_bytes=self.config.disk_emergency_bytes,
            )
            recorder.start(session_id, started_monotonic_s=started)
        return _Session(
            session_id=session_id,
            started_monotonic_s=started,
            telemetry=telemetry,
            recorder=recorder,
            metrics=SessionMetrics(started),
        )

    def prepare_events(self, events: Iterable[object] | None) -> str:
        """Start a session before its triggering Prediction/Event log is written."""

        if not self.config.diagnostics_enabled:
            return "none"
        names = {str(getattr(event, "name", "")) for event in (events or ())}
        if "session_started" not in names:
            return self.session_id
        with self._lock:
            if self._session is None:
                self._session = self._new_session()
                recorder_status = (
                    self._session.recorder.status if self._session.recorder is not None else "disabled"
                )
                self.logger.info(
                    "Diagnostics: session_started session_id=%s recording_status=%s thermal=%s",
                    self._session.session_id,
                    recorder_status,
                    "enabled" if self._session.telemetry is not None else "disabled",
                )
            return self._session.session_id

    def _latest_thermal(self) -> tuple[ThermalSample | None, float | None]:
        monitor = self._thermal_monitor
        if monitor is None:
            return None, None
        sample = monitor.latest_sample
        if sample is None:
            return None, None
        age = max(0.0, self.clock() - float(sample.monotonic_s))
        return sample, age

    def offer_runtime_update(
        self,
        update: object,
        *,
        prediction: Mapping[str, object] | None,
        events: Iterable[object] | None,
        machine: object,
    ) -> bool:
        """Snapshot all mutable state on the UI boundary and enqueue once."""

        if not self.config.diagnostics_enabled:
            return False
        with self._lock:
            session = self._session
            if session is None or session.finalizing:
                return False
            timestamp = _finite_float(getattr(update, "frame_timestamp_s", None))
            if timestamp is None:
                timestamp = self.clock() - session.started_monotonic_s
            session.metrics.observe_update(update, prediction)
            current_prediction = PredictionSnapshot.from_prediction(
                prediction,
                timestamp_s=timestamp,
            )
            prediction_is_new = current_prediction is not None and current_prediction.status == "prediction"
            if current_prediction is not None and current_prediction.status == "prediction":
                session.latest_prediction = current_prediction
            display_prediction = session.latest_prediction or current_prediction
            accepted_steps = []
            try:
                accepted_steps = sorted(int(step) for step in machine.accepted_steps)
            except (AttributeError, TypeError, ValueError):
                accepted_steps = []
            state_value = getattr(getattr(machine, "state", None), "value", getattr(machine, "state", None))
            state = StateSnapshot(
                state=_token(state_value),
                expected_step=getattr(machine, "expected_step", None),
                correct_streak=int(getattr(machine, "correct_streak", 0)),
                incorrect_streak=int(getattr(machine, "incorrect_streak", 0)),
                accepted_steps=tuple(accepted_steps),
            )
            thermal, thermal_age = self._latest_thermal()
            frame = getattr(update, "frame_bgr", None)
            if frame is None:
                return False
            recorder = session.recorder
            if recorder is None:
                return False
            event_names = tuple(
                _token(getattr(event, "name", None)) for event in (events or ())
            )
            frame_packet = FramePacket(
                frame_bgr=frame,
                frame_timestamp_s=timestamp,
                session_id=session.session_id,
                session_elapsed_s=max(0.0, self.clock() - session.started_monotonic_s),
                pose_snapshot=getattr(update, "pose_snapshot", None),
                prediction_snapshot=display_prediction,
                prediction_is_new=prediction_is_new,
                state_snapshot=state,
                thermal_snapshot=thermal,
                thermal_age_s=thermal_age,
                event_names=event_names,
            )
            return recorder.offer_frame(frame_packet)

    def _on_thermal_sample(self, sample: ThermalSample) -> None:
        with self._lock:
            session = self._session
            if session is None or session.finalizing:
                return
            telemetry = session.telemetry
            if telemetry is None or not telemetry.active:
                return
            if not telemetry.record(sample):
                return
            elapsed = max(0.0, float(sample.monotonic_s) - session.started_monotonic_s)
            self.logger.info(
                "Thermal: session_id=%s elapsed_s=%s soc_temp_c=%s throttled_raw=%s "
                "throttled_now=%s cpu_freq_mhz=%s load1=%s",
                session.session_id,
                _number(elapsed),
                _number(sample.soc_temp_c),
                _token(sample.throttled_raw),
                _token(sample.throttled_now),
                _number(sample.cpu_freq_mhz),
                _number(sample.load1),
            )

    def finish_events(self, events: Iterable[object] | None) -> None:
        names = {str(getattr(event, "name", "")) for event in (events or ())}
        if names & _TERMINAL_EVENTS:
            self.finish(next(iter(names & _TERMINAL_EVENTS)), wait=False)

    def _summary_line(self, session: _Session, summary: Mapping[str, object]) -> str:
        fields = [
            f"session_id={session.session_id}",
            f"duration_s={_number(summary.get('session_duration_s'))}",
            f"recording_status={_token(summary.get('recording_status'))}",
            f"stop_reason={_token(summary.get('stop_reason'))}",
            f"frames_seen={summary.get('frames_seen', 0)}",
            f"frames_sampling_selected={summary.get('frames_sampling_selected', 0)}",
            f"frames_enqueued={summary.get('frames_enqueued', 0)}",
            f"frames_dropped_queue={summary.get('frames_dropped_queue', 0)}",
            f"raw_frames_written={summary.get('raw_frames_written', 0)}",
            f"annotated_frames_written={summary.get('annotated_frames_written', 0)}",
            f"queue_high_watermark={summary.get('queue_high_watermark', 0)}",
            f"effective_fps_raw={_number(summary.get('raw_effective_fps'))}",
            f"effective_fps_annotated={_number(summary.get('annotated_effective_fps'))}",
            f"raw_size_bytes={summary.get('raw_size_bytes', 0)}",
            f"annotated_size_bytes={summary.get('annotated_size_bytes', 0)}",
        ]
        runtime = session.metrics.summary(self.clock())
        fields.extend(
            (
                f"runtime_frames={runtime['runtime_frames']}",
                f"runtime_effective_fps={_number(runtime['runtime_effective_fps'])}",
                f"model_emissions={runtime['model_emissions']}",
                f"update_ms_mean={_number(runtime['update_ms_mean'])}",
                f"update_ms_p95={_number(runtime['update_ms_p95'])}",
                f"update_ms_max={_number(runtime['update_ms_max'])}",
                f"prediction_interval_mean_s={_number(runtime['prediction_interval_mean_s'])}",
                f"prediction_interval_p95_s={_number(runtime['prediction_interval_p95_s'])}",
            )
        )
        fields.extend(
            (
                f"thermal_initial_temp_c={_number(summary.get('thermal_initial_temp_c'))}",
                f"thermal_max_temp_c={_number(summary.get('thermal_max_temp_c'))}",
                f"thermal_final_temp_c={_number(summary.get('thermal_final_temp_c'))}",
                f"cpu_freq_min_mhz={_number(summary.get('cpu_freq_min_mhz'))}",
                f"cpu_freq_final_mhz={_number(summary.get('cpu_freq_final_mhz'))}",
                f"throttle_initial={_token(summary.get('throttle_initial'))}",
                f"throttle_final={_token(summary.get('throttle_final'))}",
                f"throttle_observed={_token(summary.get('throttle_observed'))}",
                f"thermal_samples={summary.get('thermal_samples', 0)}",
            )
        )
        return "SessionSummary: " + " ".join(fields)

    def _finalize(self, session: _Session, reason: str) -> None:
        if session.recorder is not None:
            session.recorder.stop(reason, wait=True, timeout=10.0)
        telemetry_summary: Mapping[str, object] = {}
        if session.telemetry is not None:
            telemetry_summary = session.telemetry.close()
        recorder_summary = session.recorder.stats() if session.recorder is not None else {
            "recording_status": "disabled",
            "stop_reason": reason,
        }
        summary = {**recorder_summary, **telemetry_summary}
        summary["session_duration_s"] = max(0.0, self.clock() - session.started_monotonic_s)
        self.logger.info(self._summary_line(session, summary))
        with self._lock:
            if self._session is session:
                self._session = None

    def finish(self, reason: str = "terminal", *, wait: bool = False) -> None:
        with self._lock:
            session = self._session
            if session is None or session.finalizing:
                return
            session.finalizing = True
        if wait:
            self._finalize(session, reason)
            return
        thread = threading.Thread(
            target=self._finalize,
            args=(session, reason),
            name="laia-diagnostics-finalizer",
            daemon=True,
        )
        with self._lock:
            self._finalizer_threads.add(thread)
        thread.start()

    def wait_for_idle(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                threads = tuple(self._finalizer_threads)
                self._finalizer_threads = {thread for thread in threads if thread.is_alive()}
            if not threads:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            for thread in threads:
                thread.join(min(0.1, remaining))

    def close(self, reason: str = "application_close") -> None:
        if self._thermal_monitor is not None:
            self._thermal_monitor.stop()
        self.finish(reason, wait=True)
        self.wait_for_idle()
