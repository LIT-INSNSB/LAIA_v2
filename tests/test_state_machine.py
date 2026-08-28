from __future__ import annotations

import unittest

from app.config import RuntimeConfig
from app.state_machine import AppState, SessionStateMachine


class StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimeConfig(
            minimum_washing_seconds=0.0,
            minimum_steps_for_success=6,
            minimum_step_coverage=1.0,
            correct_predictions_required=2,
            incorrect_predictions_required=2,
            hands_lost_seconds=3.0,
            success_hold_seconds=8.0,
        )
        self.machine = SessionStateMachine(self.config)

    def start(self) -> None:
        events = self.machine.observe(class_id=None, pose_present=True, now=0.0)
        self.assertEqual([event.name for event in events], ["session_started"])

    def accept_step(self, step: int, now: float) -> None:
        self.assertEqual(self.machine.expected_step, step)
        self.machine.observe(class_id=step, pose_present=True, now=now)
        self.machine.observe(class_id=step, pose_present=True, now=now + 0.4)

    def test_complete_six_steps_reaches_success(self) -> None:
        self.start()
        for step in range(1, 7):
            self.accept_step(step, now=step * 2.0)
        self.assertEqual(self.machine.state, AppState.SUCCESS)
        self.assertEqual(self.machine.asset.name, "manos_limpias.png")

    def test_correction_keeps_expected_tutorial(self) -> None:
        self.start()
        self.accept_step(1, 1.0)
        self.accept_step(2, 3.0)
        self.machine.observe(class_id=5, pose_present=True, now=5.0)
        self.machine.observe(class_id=5, pose_present=True, now=5.4)
        self.assertEqual(self.machine.state, AppState.CORRECTION)
        self.assertEqual(self.machine.expected_step, 3)
        self.assertEqual(self.machine.asset.name, "Paso3.png")

    def test_class_zero_is_neutral_washing_movement(self) -> None:
        self.start()
        for offset in range(6):
            events = self.machine.observe(class_id=0, pose_present=True, now=1.0 + offset)
            self.assertEqual(events, [])
        self.assertEqual(self.machine.state, AppState.WASHING)

    def test_hands_lost_is_announced_once(self) -> None:
        self.start()
        self.machine.observe(class_id=None, pose_present=False, now=2.0)
        events = self.machine.observe(class_id=None, pose_present=False, now=3.1)
        self.assertEqual([event.name for event in events], ["hands_lost"])
        self.assertEqual(self.machine.observe(class_id=None, pose_present=False, now=6.0), [])

    def test_success_waits_for_twenty_seconds_even_with_all_steps(self) -> None:
        config = RuntimeConfig(
            correct_predictions_required=1,
            minimum_washing_seconds=20.0,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        for step in range(1, 7):
            machine.observe(class_id=step, pose_present=True, now=float(step))
        self.assertNotEqual(machine.state, AppState.SUCCESS)
        self.assertEqual(
            machine.observe(class_id=None, pose_present=True, now=19.9),
            [],
        )
        events = machine.observe(class_id=None, pose_present=True, now=20.0)
        self.assertEqual([event.name for event in events], ["success"])
        self.assertEqual(machine.state, AppState.SUCCESS)

    def test_four_distinct_steps_are_enough_after_twenty_seconds(self) -> None:
        config = RuntimeConfig(
            correct_predictions_required=1,
            minimum_washing_seconds=20.0,
            minimum_steps_for_success=4,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        for step in range(1, 5):
            machine.observe(class_id=step, pose_present=True, now=float(step))
        machine.observe(class_id=None, pose_present=True, now=19.9)
        self.assertNotEqual(machine.state, AppState.SUCCESS)
        events = machine.observe(class_id=None, pose_present=True, now=20.0)
        self.assertEqual([event.name for event in events], ["success"])

    def test_reported_eighty_percent_coverage_recovers_correction(self) -> None:
        config = RuntimeConfig(
            correct_predictions_required=2,
            incorrect_predictions_required=2,
            minimum_washing_seconds=20.0,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        machine.observe(class_id=6, pose_present=True, now=1.0)
        machine.observe(class_id=6, pose_present=True, now=1.4)
        self.assertEqual(machine.state, AppState.CORRECTION)
        events = machine.observe(
            class_id=6,
            pose_present=True,
            recognized_steps=[1, 2],
            step_coverage=0.80,
            now=20.0,
        )
        self.assertEqual([event.name for event in events], ["success"])
        self.assertEqual(machine.state, AppState.SUCCESS)

    def test_three_steps_do_not_pass_without_coverage(self) -> None:
        config = RuntimeConfig(
            correct_predictions_required=1,
            minimum_washing_seconds=20.0,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        for step in range(1, 4):
            machine.observe(class_id=step, pose_present=True, now=float(step))
        events = machine.observe(class_id=None, pose_present=True, now=20.0)
        self.assertEqual(events, [])
        self.assertNotEqual(machine.state, AppState.SUCCESS)


if __name__ == "__main__":
    unittest.main()
