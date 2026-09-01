from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from app.telemetry import (
    THERMAL_COLUMNS,
    TelemetrySession,
    ThermalMonitor,
    ThermalSample,
    parse_throttled,
    read_thermal_sample,
)


class TelemetryTests(unittest.TestCase):
    def test_throttled_bits_decode_current_and_occurred_flags(self) -> None:
        flags = parse_throttled("throttled=0x5000d")

        self.assertTrue(flags["under_voltage_now"])
        self.assertFalse(flags["frequency_capped_now"])
        self.assertTrue(flags["throttled_now"])
        self.assertTrue(flags["soft_temperature_limit_now"])
        self.assertTrue(flags["under_voltage_occurred"])
        self.assertFalse(flags["frequency_capped_occurred"])
        self.assertTrue(flags["throttled_occurred"])
        self.assertFalse(flags["soft_temperature_limit_occurred"])

    def test_sample_reads_sysfs_without_psutil_and_falls_back_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "temp").write_text("48250\n", encoding="utf-8")
            (root / "freq").write_text("1500000\n", encoding="utf-8")
            (root / "load").write_text("0.37 0.22 0.18 1/100 42\n", encoding="utf-8")

            def runner(args, **_kwargs):
                if args[-1] == "get_throttled":
                    return type("Result", (), {"returncode": 0, "stdout": "throttled=0x0\n"})()
                return type("Result", (), {"returncode": 0, "stdout": "temp=51.2'C\n"})()

            sample = read_thermal_sample(
                sysfs_temp_path=root / "temp",
                cpu_freq_path=root / "freq",
                loadavg_path=root / "load",
                runner=runner,
                monotonic_clock=lambda: 12.5,
                wall_clock=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc),
            )

            self.assertEqual(sample.soc_temp_c, 48.25)
            self.assertEqual(sample.temp_source, "sysfs_cpu_thermal")
            self.assertEqual(sample.cpu_freq_mhz, 1500.0)
            self.assertEqual(sample.load1, 0.37)
            self.assertEqual(sample.throttled_raw, "0x0")

            (root / "temp").unlink()
            fallback = read_thermal_sample(
                sysfs_temp_path=root / "temp",
                cpu_freq_path=root / "missing-freq",
                loadavg_path=root / "missing-load",
                runner=runner,
                monotonic_clock=lambda: 13.0,
            )
            self.assertEqual(fallback.soc_temp_c, 51.2)
            self.assertEqual(fallback.temp_source, "vcgencmd_cpu_thermal")
            self.assertIsNone(fallback.cpu_freq_mhz)
            self.assertIsNone(fallback.load1)

    def test_missing_vcgencmd_and_sensor_files_are_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def failing_runner(*_args, **_kwargs):
                raise FileNotFoundError("vcgencmd")

            sample = read_thermal_sample(
                sysfs_temp_path=Path(directory) / "missing-temp",
                cpu_freq_path=Path(directory) / "missing-freq",
                loadavg_path=Path(directory) / "missing-load",
                runner=failing_runner,
            )

            self.assertIsNone(sample.soc_temp_c)
            self.assertEqual(sample.temp_source, "none")
            self.assertIsNone(sample.throttled_now)
            self.assertIsNone(sample.cpu_freq_mhz)

    def test_session_csv_has_exact_columns_flushes_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.csv"
            session = TelemetrySession(path, "20260901_101500", started_monotonic_s=10.0)
            self.assertFalse(path.exists())
            session.open()
            self.assertTrue(path.exists())
            session.record(
                ThermalSample(
                    timestamp_iso="2026-09-01T10:15:01+00:00",
                    monotonic_s=12.0,
                    soc_temp_c=48.25,
                    temp_source="sysfs_cpu_thermal",
                    throttled_raw="0x0",
                    under_voltage_now=False,
                    frequency_capped_now=False,
                    throttled_now=False,
                    soft_temperature_limit_now=False,
                    under_voltage_occurred=False,
                    frequency_capped_occurred=False,
                    throttled_occurred=False,
                    soft_temperature_limit_occurred=False,
                    cpu_freq_mhz=1500.0,
                    load1=0.37,
                    cpu_count=4,
                )
            )
            summary = session.close()

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(handle.closed, True)
            self.assertEqual(list(rows[0].keys()), list(THERMAL_COLUMNS))
            self.assertEqual(rows[0]["session_id"], "20260901_101500")
            self.assertEqual(rows[0]["session_elapsed_s"], "2.000")
            self.assertEqual(rows[0]["throttled_now"], "0")
            self.assertEqual(summary["thermal_samples"], 1)
            self.assertEqual(summary["thermal_max_temp_c"], 48.25)
            self.assertEqual(summary["cpu_freq_min_mhz"], 1500.0)
            self.assertFalse(summary["throttle_observed"])

    def test_monitor_samples_on_its_own_thread(self) -> None:
        received = []
        ready = threading.Event()

        def reader() -> ThermalSample:
            return ThermalSample(monotonic_s=time.monotonic(), soc_temp_c=45.0)

        def on_sample(sample: ThermalSample) -> None:
            received.append(sample)
            ready.set()

        monitor = ThermalMonitor(interval_seconds=0.05, reader=reader, on_sample=on_sample)
        monitor.start()
        self.assertTrue(ready.wait(1.0))
        monitor.stop()

        self.assertGreaterEqual(len(received), 1)
        self.assertIs(monitor.latest_sample, received[-1])
        self.assertFalse(monitor._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
