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
            incomplete_hold_seconds=0.0,
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
        self.assertEqual(self.machine.accepted_steps, set(range(1, 7)))

    def test_wrong_class_does_not_enter_accepted_steps(self) -> None:
        self.start()
        self.accept_step(1, 1.0)
        self.accept_step(2, 3.0)
        self.accept_step(3, 5.0)

        self.machine.observe(class_id=6, pose_present=True, now=7.0)
        events = self.machine.observe(class_id=6, pose_present=True, now=7.4)

        self.assertEqual(self.machine.expected_step, 4)
        self.assertEqual(self.machine.accepted_steps, {1, 2, 3})
        self.assertIn(6, self.machine.observed_classes)
        self.assertIn("correction", [event.name for event in events])

    def test_correct_step_four_is_accepted_after_correction(self) -> None:
        self.start()
        for step in range(1, 4):
            self.accept_step(step, now=step * 2.0)
        self.machine.observe(class_id=6, pose_present=True, now=7.0)
        self.machine.observe(class_id=6, pose_present=True, now=7.4)
        self.assertEqual(self.machine.state, AppState.CORRECTION)

        self.machine.observe(class_id=4, pose_present=True, now=8.0)
        events = self.machine.observe(class_id=4, pose_present=True, now=8.4)

        self.assertIn(4, self.machine.accepted_steps)
        self.assertNotIn(6, self.machine.accepted_steps)
        self.assertEqual(self.machine.expected_step, 5)
        self.assertIn("step_accepted", [event.name for event in events])
        self.assertIn("recovered", [event.name for event in events])

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
        self.assertIn(0, self.machine.observed_classes)
        self.assertEqual(self.machine.accepted_steps, set())

    def test_four_steps_are_not_enough_after_twenty_seconds(self) -> None:
        config = RuntimeConfig(
            correct_predictions_required=1,
            minimum_washing_seconds=20.0,
            minimum_steps_for_success=4,
            minimum_step_coverage=0.80,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        for step in range(1, 5):
            machine.observe(class_id=step, pose_present=True, now=float(step))

        machine.observe(class_id=None, pose_present=True, now=20.0)
        self.assertNotEqual(machine.state, AppState.SUCCESS)
        self.assertFalse(machine.success_diagnostics(20.0)["steps_ok"])

    def test_reported_coverage_cannot_bypass_six_accepted_steps(self) -> None:
        config = RuntimeConfig(
            correct_predictions_required=2,
            incorrect_predictions_required=2,
            minimum_washing_seconds=20.0,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        machine.observe(class_id=6, pose_present=True, now=1.0)
        machine.observe(
            class_id=6,
            pose_present=True,
            recognized_steps=[1, 2, 3, 4, 5, 6],
            step_coverage=1.0,
            now=1.4,
        )

        machine.observe(class_id=None, pose_present=True, now=20.0)
        self.assertNotEqual(machine.state, AppState.SUCCESS)
        self.assertEqual(machine.accepted_steps, set())
        self.assertFalse(machine.success_diagnostics(20.0)["coverage_ok"])
        self.assertEqual(machine.success_diagnostics(20.0)["observed_step_coverage"], 1.0)
        self.assertFalse(machine.success_diagnostics(20.0)["final_success"])

    def test_success_waits_for_minimum_active_washing_time(self) -> None:
        config = RuntimeConfig(
            correct_predictions_required=1,
            minimum_washing_seconds=20.0,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        for step in range(1, 7):
            machine.observe(class_id=step, pose_present=True, now=float(step))

        self.assertNotEqual(machine.state, AppState.SUCCESS)
        self.assertEqual(machine.observe(class_id=None, pose_present=True, now=19.9), [])
        events = machine.observe(class_id=None, pose_present=True, now=20.0)
        self.assertEqual([event.name for event in events], ["success"])
        self.assertEqual(machine.state, AppState.SUCCESS)

    def test_washing_time_pauses_during_grace_and_resumes(self) -> None:
        config = RuntimeConfig(
            minimum_washing_seconds=10.0,
            hands_lost_seconds=3.0,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        machine.observe(class_id=None, pose_present=True, now=3.0)
        machine.observe(class_id=None, pose_present=False, now=4.0)

        self.assertAlmostEqual(machine.washing_time(6.0), 4.0)
        self.assertFalse(machine.success_diagnostics(6.0)["time_ok"])

        events = machine.observe(class_id=None, pose_present=True, now=6.5)

        self.assertEqual([event.name for event in events], ["hands_returned"])
        self.assertAlmostEqual(machine.washing_paused_seconds, 2.5)
        self.assertAlmostEqual(machine.last_hands_lost_duration, 2.5)
        self.assertAlmostEqual(machine.washing_time(6.5), 4.0)
        self.assertAlmostEqual(machine.washing_time(12.5), 10.0)
        self.assertTrue(machine.success_diagnostics(12.5)["time_ok"])

    def test_hands_loss_timeout_enters_incomplete_after_grace(self) -> None:
        config = RuntimeConfig(
            minimum_washing_seconds=10.0,
            hands_lost_seconds=3.0,
        )
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        machine.observe(class_id=None, pose_present=False, now=4.0)

        events = machine.tick(now=7.0)

        self.assertEqual(
            [event.name for event in events],
            ["hands_lost", "attempt_incomplete"],
        )
        self.assertEqual(machine.state, AppState.INCOMPLETE)
        self.assertAlmostEqual(machine.last_hands_lost_duration, 3.0)
        self.assertAlmostEqual(machine.last_active_washing_time, 4.0)
        self.assertIsNone(machine.washing_time(7.0))

    def test_brief_hands_loss_returns_to_same_step_with_clean_streak(self) -> None:
        self.start()
        self.machine.observe(class_id=1, pose_present=True, now=1.0)
        self.assertEqual(self.machine.correct_streak, 1)

        self.machine.observe(class_id=None, pose_present=False, now=2.0)
        events = self.machine.observe(class_id=1, pose_present=True, now=2.5)

        self.assertEqual([event.name for event in events], ["hands_returned"])
        self.assertEqual(self.machine.state, AppState.WASHING)
        self.assertEqual(self.machine.expected_step, 1)
        self.assertEqual(self.machine.accepted_steps, set())
        self.assertEqual(self.machine.correct_streak, 0)
        self.assertEqual(self.machine.incorrect_streak, 0)

        self.machine.observe(class_id=1, pose_present=True, now=3.0)
        events = self.machine.observe(class_id=1, pose_present=True, now=3.1)
        self.assertIn("step_accepted", [event.name for event in events])
        self.assertEqual(self.machine.accepted_steps, {1})

    def test_hands_loss_timeout_is_enforced_by_tick(self) -> None:
        self.start()
        self.machine.observe(class_id=None, pose_present=False, now=2.0)

        events = self.machine.tick(now=5.0)

        self.assertEqual(
            [event.name for event in events],
            ["hands_lost", "attempt_incomplete"],
        )
        self.assertEqual(self.machine.state, AppState.INCOMPLETE)

    def test_incomplete_does_not_resume_until_reset(self) -> None:
        config = RuntimeConfig(hands_lost_seconds=3.0, incomplete_hold_seconds=0.0)
        machine = SessionStateMachine(config)
        machine.observe(class_id=None, pose_present=True, now=0.0)
        machine.observe(class_id=None, pose_present=False, now=4.0)
        machine.tick(now=7.0)

        self.assertEqual(machine.state, AppState.INCOMPLETE)
        self.assertEqual(machine.observe(class_id=None, pose_present=True, now=8.0), [])
        self.assertEqual(machine.state, AppState.INCOMPLETE)

        events = machine.tick(now=8.0)
        self.assertEqual([event.name for event in events], ["reset"])
        self.assertEqual(machine.state, AppState.WAITING_FOR_HANDS)
        self.assertEqual(machine.accepted_steps, set())
        self.assertEqual(machine.observed_classes, set())
        self.assertEqual(machine.washing_paused_seconds, 0.0)

    def test_reset_clears_attempt_state(self) -> None:
        self.start()
        self.accept_step(1, 1.0)
        self.machine.observe(class_id=None, pose_present=False, now=2.0)
        self.machine.observe(class_id=None, pose_present=True, now=2.5)
        self.assertGreater(self.machine.washing_paused_seconds, 0.0)

        self.machine.reset(now=3.0)

        self.assertEqual(self.machine.state, AppState.WAITING_FOR_HANDS)
        self.assertEqual(self.machine.expected_step, 1)
        self.assertEqual(self.machine.accepted_steps, set())
        self.assertEqual(self.machine.observed_classes, set())
        self.assertEqual(self.machine.correct_streak, 0)
        self.assertEqual(self.machine.incorrect_streak, 0)
        self.assertIsNone(self.machine.washing_since)
        self.assertEqual(self.machine.washing_paused_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
