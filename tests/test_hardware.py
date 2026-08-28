import sys
import time
import types
import unittest
from unittest.mock import patch

from app.hardware import LedController
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
            leds.close()


if __name__ == "__main__":
    unittest.main()
