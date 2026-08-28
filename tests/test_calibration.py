from __future__ import annotations

import unittest

from app.calibration import CalibrationLedController
from app.calibration_config import CalibrationConfig, CalibrationState, CalibrationStateMachine, HandPlacementValidator


def hand(x1: float, y1: float, x2: float, y2: float):
    """Genera 21 landmarks deterministas dentro de una caja normalizada."""

    points = []
    for index in range(21):
        column = index % 7
        row = index // 7
        x = x1 + (x2 - x1) * column / 6
        y = y1 + (y2 - y1) * row / 2
        points.append((x, y))
    return points


def valid_hands():
    return (hand(0.22, 0.30, 0.43, 0.58), hand(0.57, 0.30, 0.78, 0.58))


class CalibrationLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CalibrationConfig(stable_seconds=1.0, countdown_seconds=3.0)
        self.validator = HandPlacementValidator(self.config)
        self.machine = CalibrationStateMachine(self.config)

    def test_zero_hands_is_red_positioning(self) -> None:
        result = self.validator.evaluate(())
        self.machine.update(result, 0.0)
        self.assertFalse(result.valid)
        self.assertEqual(result.message, "No veo tus manos")
        self.assertEqual(self.machine.state, CalibrationState.POSITIONING)
        self.assertTrue(self.machine.red_on)
        self.assertFalse(self.machine.green_on)

    def test_one_hand_is_red(self) -> None:
        result = self.validator.evaluate((valid_hands()[0],))
        self.machine.update(result, 0.0)
        self.assertFalse(result.valid)
        self.assertEqual(result.message, "Muéstrame las dos manos")
        self.assertTrue(self.machine.red_on)

    def test_two_hands_outside_position_are_red(self) -> None:
        result = self.validator.evaluate((hand(0.01, 0.30, 0.20, 0.58), hand(0.80, 0.30, 0.99, 0.58)))
        self.machine.update(result, 0.0)
        self.assertFalse(result.valid)
        self.assertEqual(self.machine.state, CalibrationState.POSITIONING)
        self.assertEqual(result.message, "Coloca tus manos aquí")

    def test_valid_hands_less_than_one_second_keep_waiting(self) -> None:
        result = self.validator.evaluate(valid_hands())
        self.machine.update(result, 0.0)
        self.machine.update(result, 0.99)
        self.assertEqual(self.machine.state, CalibrationState.READY)
        self.assertIsNone(self.machine.countdown_number(0.99))
        self.assertTrue(self.machine.red_on)

    def test_valid_hands_after_one_second_start_countdown_and_green(self) -> None:
        result = self.validator.evaluate(valid_hands())
        self.machine.update(result, 0.0)
        events = self.machine.update(result, 1.0)
        self.assertEqual(events, ("countdown_started",))
        self.assertEqual(self.machine.state, CalibrationState.COUNTDOWN)
        self.assertEqual(self.machine.countdown_number(1.0), 3)
        self.assertFalse(self.machine.red_on)
        self.assertTrue(self.machine.green_on)

    def test_losing_hand_during_countdown_returns_to_positioning(self) -> None:
        valid = self.validator.evaluate(valid_hands())
        invalid = self.validator.evaluate((valid_hands()[0],))
        self.machine.update(valid, 0.0)
        self.machine.update(valid, 1.0)
        events = self.machine.update(invalid, 1.2)
        self.assertEqual(events, ("countdown_cancelled",))
        self.assertEqual(self.machine.state, CalibrationState.POSITIONING)
        self.assertTrue(self.machine.red_on)
        self.assertFalse(self.machine.green_on)

    def test_countdown_completion_reaches_calibration_ok(self) -> None:
        valid = self.validator.evaluate(valid_hands())
        self.machine.update(valid, 0.0)
        self.machine.update(valid, 1.0)
        events = self.machine.update(valid, 4.0)
        self.assertEqual(events, ("calibration_ok",))
        self.assertEqual(self.machine.state, CalibrationState.CALIBRATION_OK)
        self.assertTrue(self.machine.green_on)

    def test_gpio27_stays_off(self) -> None:
        class FakeLed:
            instances = []

            def __init__(self, pin, **_kwargs):
                self.pin = pin
                self.on_calls = 0
                self.off_calls = 0
                FakeLed.instances.append(self)

            def on(self):
                self.on_calls += 1

            def off(self):
                self.off_calls += 1

            def close(self):
                pass

        FakeLed.instances.clear()
        leds = CalibrationLedController(self.config, led_factory=FakeLed)
        by_pin = {item.pin: item for item in FakeLed.instances}
        for state in CalibrationState:
            leds.apply(state)
            self.assertEqual(by_pin[27].on_calls, 0)
        self.assertGreater(by_pin[27].off_calls, 0)
        leds.close()

    def test_reset_returns_to_positioning(self) -> None:
        valid = self.validator.evaluate(valid_hands())
        self.machine.update(valid, 0.0)
        self.machine.update(valid, 1.0)
        self.machine.reset()
        self.assertEqual(self.machine.state, CalibrationState.POSITIONING)
        self.assertIsNone(self.machine.ready_since)
        self.assertIsNone(self.machine.countdown_started)


class CalibrationIsolationTests(unittest.TestCase):
    def test_calibration_module_does_not_reference_who_classifier(self) -> None:
        from pathlib import Path

        source = Path(__file__).parents[1].joinpath("app", "calibration.py").read_text(encoding="utf-8").lower()
        for forbidden in ("onnx", "lightstgcn", "streaminghandwashingrecognizer", "temporalposebuffer"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
