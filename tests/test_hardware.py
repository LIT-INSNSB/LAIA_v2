import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from app.hardware import CameraSource, LedController
from app.state_machine import AppState


class FakeLed:
    instances = []

    def __init__(self, pin, **_kwargs):
        self.pin = pin
        self.events = []
        FakeLed.instances.append(self)

    def on(self):
        self.events.append("on")

    def off(self):
        self.events.append("off")

    def close(self):
        self.events.append("close")


class LedSignalTests(unittest.TestCase):
    def test_success_and_error_use_requested_pins_and_auto_off(self):
        FakeLed.instances.clear()
        fake_gpiozero = types.SimpleNamespace(LED=FakeLed)
        with patch.dict(sys.modules, {"gpiozero": fake_gpiozero}):
            leds = LedController(
                success_pin=22,
                error_pin=17,
                blink_count=1,
                blink_seconds=0.01,
                hold_seconds=0.02,
            )
            leds.apply(AppState.SUCCESS)
            time.sleep(0.08)
            success, error = FakeLed.instances
            self.assertEqual((success.pin, error.pin), (22, 17))
            self.assertIn("on", success.events)
            self.assertGreaterEqual(success.events.count("off"), 1)
            leds.apply(AppState.CORRECTION)
            time.sleep(0.08)
            self.assertIn("on", error.events)
            self.assertGreaterEqual(error.events.count("off"), 1)
            leds.apply(AppState.INCOMPLETE)
            time.sleep(0.02)
            self.assertGreaterEqual(success.events.count("off"), 1)
            self.assertGreaterEqual(error.events.count("off"), 1)
            leds.close()


class FakePicamera2:
    instances = []

    def __init__(self):
        self.preview_configuration = None
        self.configured = None
        self.started = False
        self.stopped = False
        self.closed = False
        self.frame = np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        FakePicamera2.instances.append(self)

    def create_preview_configuration(self, **kwargs):
        self.preview_configuration = kwargs
        return {"main": kwargs["main"]}

    def configure(self, config):
        self.configured = config

    def start(self):
        self.started = True

    def capture_array(self, stream):
        assert stream == "main"
        return self.frame

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class CameraSourceTests(unittest.TestCase):
    def test_picamera2_rgb888_buffer_is_passed_through_as_bgr(self):
        FakePicamera2.instances.clear()
        fake_picamera2 = types.SimpleNamespace(Picamera2=FakePicamera2)
        with patch.dict(sys.modules, {"picamera2": fake_picamera2}), patch("app.hardware.time.sleep"):
            camera = CameraSource(2, 1, 20.0, source="picamera2")
            try:
                frame = camera.read()
                instance = FakePicamera2.instances[0]
                self.assertIs(frame, instance.frame)
                self.assertEqual(frame.dtype, np.uint8)
                self.assertEqual(frame.shape, (1, 2, 3))
                self.assertEqual(instance.preview_configuration["main"]["format"], "RGB888")
                self.assertEqual(camera.backend, "Picamera2")
            finally:
                camera.close()
        self.assertTrue(instance.stopped)
        self.assertTrue(instance.closed)


if __name__ == "__main__":
    unittest.main()
