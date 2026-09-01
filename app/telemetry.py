"""Low-overhead Raspberry Pi thermal telemetry for diagnostic sessions."""

from __future__ import annotations

import csv
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


LOGGER = logging.getLogger("laia.diagnostics.telemetry")

THERMAL_COLUMNS = (
    "timestamp_iso",
    "monotonic_s",
    "session_id",
    "session_elapsed_s",
    "soc_temp_c",
    "temp_source",
    "throttled_raw",
    "under_voltage_now",
    "frequency_capped_now",
    "throttled_now",
    "soft_temperature_limit_now",
    "under_voltage_occurred",
    "frequency_capped_occurred",
    "throttled_occurred",
    "soft_temperature_limit_occurred",
    "cpu_freq_mhz",
    "load1",
    "cpu_count",
)

_SYSFS_TEMP = Path("/sys/class/thermal/thermal_zone0/temp")
_CPU_FREQ = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")
_LOADAVG = Path("/proc/loadavg")


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _csv_value(value: object) -> str | int | float:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.3f}" if math.isfinite(value) else "none"
    return value


@dataclass(frozen=True)
class ThermalSample:
    """A single coherent read of the thermal/throttling counters."""

    timestamp_iso: str = ""
    monotonic_s: float = 0.0
    soc_temp_c: float | None = None
    temp_source: str = "none"
    throttled_raw: str | None = None
    under_voltage_now: bool | None = None
    frequency_capped_now: bool | None = None
    throttled_now: bool | None = None
    soft_temperature_limit_now: bool | None = None
    under_voltage_occurred: bool | None = None
    frequency_capped_occurred: bool | None = None
    throttled_occurred: bool | None = None
    soft_temperature_limit_occurred: bool | None = None
    cpu_freq_mhz: float | None = None
    load1: float | None = None
    cpu_count: int | None = None


def parse_throttled(raw: str | bytes | None) -> dict[str, bool | None]:
    """Decode current bits 0-3 and occurred bits 16-19 from vcgencmd."""

    if raw is None:
        value = None
    else:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        match = re.search(r"0x([0-9a-fA-F]+)", str(raw))
        if match:
            value = int(match.group(1), 16)
        else:
            decimal = re.search(r"\b(\d+)\b", str(raw))
            value = int(decimal.group(1), 10) if decimal else None

    names = (
        ("under_voltage", 0),
        ("frequency_capped", 1),
        ("throttled", 2),
        ("soft_temperature_limit", 3),
    )
    result: dict[str, bool | None] = {}
    for name, bit in names:
        result[f"{name}_now"] = None if value is None else bool(value & (1 << bit))
        result[f"{name}_occurred"] = None if value is None else bool(value & (1 << (bit + 16)))
    return result


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _run_vcgencmd(
    args: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: float,
) -> str | None:
    try:
        result = runner(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if getattr(result, "returncode", 1) == 0 else None


def read_thermal_sample(
    *,
    sysfs_temp_path: Path = _SYSFS_TEMP,
    cpu_freq_path: Path = _CPU_FREQ,
    loadavg_path: Path = _LOADAVG,
    vcgencmd: str = "vcgencmd",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    monotonic_clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] | None = None,
) -> ThermalSample:
    """Read thermal data without psutil and tolerate missing Pi-only files."""

    monotonic_s = float(monotonic_clock())
    now = (wall_clock or (lambda: datetime.now().astimezone()))()
    timestamp_iso = now.isoformat(timespec="milliseconds")

    soc_temp_c: float | None = None
    temp_source = "none"
    raw_temp = _read_text(sysfs_temp_path)
    if raw_temp is not None:
        raw_millidegrees = _finite_float(raw_temp)
        if raw_millidegrees is not None:
            soc_temp_c = raw_millidegrees / 1000.0
            temp_source = "sysfs_cpu_thermal"
    if soc_temp_c is None:
        output = _run_vcgencmd(
            [vcgencmd, "measure_temp"], runner=runner, timeout=1.0
        )
        if output:
            match = re.search(r"(-?\d+(?:\.\d+)?)", output)
            if match:
                soc_temp_c = _finite_float(match.group(1))
                if soc_temp_c is not None:
                    temp_source = "vcgencmd_cpu_thermal"

    throttled_output = _run_vcgencmd(
        [vcgencmd, "get_throttled"], runner=runner, timeout=0.75
    )
    throttle = parse_throttled(throttled_output)
    raw_match = re.search(r"0x[0-9a-fA-F]+", throttled_output or "")
    throttled_raw = raw_match.group(0) if raw_match else None

    cpu_freq_mhz: float | None = None
    raw_frequency = _read_text(cpu_freq_path)
    if raw_frequency is not None:
        frequency_khz = _finite_float(raw_frequency)
        if frequency_khz is not None:
            cpu_freq_mhz = frequency_khz / 1000.0

    load1: float | None = None
    raw_load = _read_text(loadavg_path)
    if raw_load:
        load1 = _finite_float(raw_load.split()[0])

    return ThermalSample(
        timestamp_iso=timestamp_iso,
        monotonic_s=monotonic_s,
        soc_temp_c=soc_temp_c,
        temp_source=temp_source,
        throttled_raw=throttled_raw,
        cpu_freq_mhz=cpu_freq_mhz,
        load1=load1,
        cpu_count=os.cpu_count() or 1,
        **throttle,
    )


class ThermalMonitor:
    """Dedicated sampler thread; the inference and Tk threads never read sensors."""

    def __init__(
        self,
        *,
        interval_seconds: float = 5.0,
        reader: Callable[[], ThermalSample] = read_thermal_sample,
        on_sample: Callable[[ThermalSample], None] | None = None,
    ) -> None:
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.reader = reader
        self.on_sample = on_sample
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: ThermalSample | None = None
        self._thread: threading.Thread | None = None

    @property
    def latest_sample(self) -> ThermalSample | None:
        with self._lock:
            return self._latest

    def sample_once(self) -> ThermalSample | None:
        try:
            sample = self.reader()
        except Exception:
            LOGGER.exception("No se pudo leer telemetría térmica")
            return None
        with self._lock:
            self._latest = sample
        if self.on_sample is not None:
            try:
                self.on_sample(sample)
            except Exception:
                LOGGER.exception("Error al publicar telemetría térmica")
        return sample

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="laia-thermal", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_seconds)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))


class TelemetrySession:
    """Flush-on-write session CSV with bounded scalar summary accumulators."""

    def __init__(self, path: Path, session_id: str, started_monotonic_s: float) -> None:
        self.path = Path(path)
        self.session_id = str(session_id)
        self.started_monotonic_s = float(started_monotonic_s)
        self._file = None
        self._writer = None
        self._closed = False
        self.sample_count = 0
        self.initial_temp_c: float | None = None
        self.max_temp_c: float | None = None
        self.final_temp_c: float | None = None
        self.min_cpu_freq_mhz: float | None = None
        self.final_cpu_freq_mhz: float | None = None
        self.initial_throttled: bool | None = None
        self.final_throttled: bool | None = None
        self.throttle_observed = False

    @property
    def active(self) -> bool:
        return self._writer is not None and not self._closed

    def open(self) -> None:
        if self.active:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=THERMAL_COLUMNS, lineterminator="\n")
        self._writer.writeheader()
        self._file.flush()

    def record(self, sample: ThermalSample) -> bool:
        if not self.active:
            return False
        elapsed = max(0.0, float(sample.monotonic_s) - self.started_monotonic_s)
        row = {
            "timestamp_iso": sample.timestamp_iso or "none",
            "monotonic_s": _csv_value(float(sample.monotonic_s)),
            "session_id": self.session_id,
            "session_elapsed_s": _csv_value(elapsed),
            "soc_temp_c": _csv_value(sample.soc_temp_c),
            "temp_source": sample.temp_source or "none",
            "throttled_raw": sample.throttled_raw or "none",
            "under_voltage_now": _csv_value(sample.under_voltage_now),
            "frequency_capped_now": _csv_value(sample.frequency_capped_now),
            "throttled_now": _csv_value(sample.throttled_now),
            "soft_temperature_limit_now": _csv_value(sample.soft_temperature_limit_now),
            "under_voltage_occurred": _csv_value(sample.under_voltage_occurred),
            "frequency_capped_occurred": _csv_value(sample.frequency_capped_occurred),
            "throttled_occurred": _csv_value(sample.throttled_occurred),
            "soft_temperature_limit_occurred": _csv_value(sample.soft_temperature_limit_occurred),
            "cpu_freq_mhz": _csv_value(sample.cpu_freq_mhz),
            "load1": _csv_value(sample.load1),
            "cpu_count": _csv_value(sample.cpu_count),
        }
        self._writer.writerow(row)
        self._file.flush()
        self.sample_count += 1

        temperature = _finite_float(sample.soc_temp_c)
        if temperature is not None:
            if self.initial_temp_c is None:
                self.initial_temp_c = temperature
            self.max_temp_c = temperature if self.max_temp_c is None else max(self.max_temp_c, temperature)
            self.final_temp_c = temperature
        frequency = _finite_float(sample.cpu_freq_mhz)
        if frequency is not None:
            self.min_cpu_freq_mhz = frequency if self.min_cpu_freq_mhz is None else min(self.min_cpu_freq_mhz, frequency)
            self.final_cpu_freq_mhz = frequency
        if self.initial_throttled is None and sample.throttled_now is not None:
            self.initial_throttled = sample.throttled_now
        if sample.throttled_now is not None:
            self.final_throttled = sample.throttled_now
            self.throttle_observed = self.throttle_observed or sample.throttled_now
        self.throttle_observed = self.throttle_observed or bool(sample.throttled_occurred)
        return True

    def summary(self) -> dict[str, object]:
        return {
            "thermal_samples": self.sample_count,
            "thermal_initial_temp_c": self.initial_temp_c,
            "thermal_max_temp_c": self.max_temp_c,
            "thermal_final_temp_c": self.final_temp_c,
            "cpu_freq_min_mhz": self.min_cpu_freq_mhz,
            "cpu_freq_final_mhz": self.final_cpu_freq_mhz,
            "throttle_initial": self.initial_throttled,
            "throttle_final": self.final_throttled,
            "throttle_observed": self.throttle_observed,
            "telemetry_size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def close(self) -> dict[str, object]:
        if self._closed:
            return self.summary()
        self._closed = True
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._writer = None
        return self.summary()
