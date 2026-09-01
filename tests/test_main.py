from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.config import RuntimeConfig, audio_path
from app.main import (
    LaiaApplication,
    audio_key_for_event,
    format_prediction_diagnostics,
)
from app.state_machine import AppEvent, AppState, SessionStateMachine


class FakeAudio:
    def __init__(self) -> None:
        self.played = []
        self.stop_if_active_calls = []

    def play(self, path) -> None:
        self.played.append(path)

    def stop_if_active(self, path) -> bool:
        self.stop_if_active_calls.append(path)
        return True


class AudioRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio = FakeAudio()
        self.application = LaiaApplication.__new__(LaiaApplication)
        self.application.audio = self.audio
        self.application.machine = SimpleNamespace(
            state=AppState.WASHING,
            expected_step=1,
            correct_streak=0,
            incorrect_streak=0,
            accepted_steps=set(),
        )
        self.application.runtime = None

    def test_correction_maps_to_oops_clip(self) -> None:
        self.assertEqual(audio_key_for_event("correction"), "retry")

        self.application._handle_events([AppEvent("correction", 4)])

        self.assertEqual(self.audio.played, [audio_path("retry")])

    def test_retry_maps_to_no_sound_and_does_not_stop(self) -> None:
        self.assertIsNone(audio_key_for_event("retry"))

        self.application._handle_events([AppEvent("retry", 4)])

        self.assertEqual(self.audio.played, [])
        self.assertEqual(self.audio.stop_if_active_calls, [])

    def test_recovery_then_later_correction_allows_new_oops(self) -> None:
        self.application._handle_events([AppEvent("correction", 4)])
        self.application._handle_events([AppEvent("recovered", 4)])
        self.application._handle_events([AppEvent("correction", 5)])

        self.assertEqual(
            self.audio.played,
            [audio_path("retry"), audio_path("recover"), audio_path("retry")],
        )

    def test_hands_lost_grace_start_plays_warning(self) -> None:
        self.assertEqual(audio_key_for_event("hands_lost_grace_started"), "hands_lost")

        self.application._handle_events([AppEvent("hands_lost_grace_started", 4)])

        self.assertEqual(self.audio.played, [audio_path("hands_lost")])

    def test_hands_returned_stops_only_hand_loss_clip(self) -> None:
        self.application._handle_events([AppEvent("hands_returned", 4)])

        self.assertEqual(self.audio.stop_if_active_calls, [audio_path("hands_lost")])
        self.assertEqual(self.audio.played, [])

    def test_terminal_hands_lost_does_not_restart_warning(self) -> None:
        self.assertIsNone(audio_key_for_event("hands_lost"))

        self.application._handle_events([AppEvent("hands_lost")])

        self.assertEqual(self.audio.played, [])

    def test_incomplete_batch_has_one_final_feedback_playback(self) -> None:
        self.application._handle_events(
            [AppEvent("hands_lost"), AppEvent("attempt_incomplete", 4)]
        )

        self.assertEqual(self.audio.played, [audio_path("almost")])


class PredictionLoggingTests(unittest.TestCase):
    @staticmethod
    def application_for(machine: SessionStateMachine) -> LaiaApplication:
        application = LaiaApplication.__new__(LaiaApplication)
        application.audio = FakeAudio()
        application.machine = machine
        application.runtime = None
        return application

    @staticmethod
    def prediction_line(logger_mock) -> str:
        for call in logger_mock.call_args_list:
            if call.args and isinstance(call.args[0], str) and call.args[0].startswith("Prediction:"):
                return call.args[0]
        raise AssertionError("no se registró una línea Prediction")

    def test_prediction_line_uses_pre_step_and_post_observation_snapshot(self) -> None:
        machine = SessionStateMachine(
            RuntimeConfig(correct_predictions_required=2, minimum_washing_seconds=0.0)
        )
        machine.state = AppState.WASHING
        machine.expected_step = 4
        machine.correct_streak = 1
        machine.incorrect_streak = 1
        machine.accepted_steps = {3, 1, 2}
        application = self.application_for(machine)
        prediction = {
            "status": "prediction",
            "class_id": 4,
            "class_name": "backs_of_fingers_to_opposing_palm",
            "confidence_uncalibrated": 0.87,
            "pose_coverage_ge1": 0.91,
            "pose_coverage_2": 0.84,
            "source_frame_count": 30,
            "tracks_created": 2,
            "track_fragmentation": 0,
        }

        with patch("app.main.LOGGER.info") as logger:
            events = application._process_prediction(prediction)

        line = self.prediction_line(logger)
        self.assertEqual([event.name for event in events], ["step_accepted", "step_advanced"])
        self.assertIn("status=prediction", line)
        self.assertIn("expected_step=4", line)
        self.assertIn("predicted_class=4", line)
        self.assertIn("class_name=backs_of_fingers_to_opposing_palm", line)
        self.assertIn("confidence=0.870", line)
        self.assertIn("pose_coverage=0.910", line)
        self.assertIn("pose_coverage_2=0.840", line)
        self.assertIn("correct_streak=0", line)
        self.assertIn("incorrect_streak=0", line)
        self.assertIn("runtime_state=WASHING", line)
        self.assertIn("accepted_steps=1,2,3,4", line)
        self.assertIn("events=step_accepted,step_advanced", line)
        self.assertIn("track_fragmentation=0", line)
        self.assertIn("tracks_created=2", line)
        self.assertIn("source_frame_count=30", line)
        self.assertEqual(machine.expected_step, 5)

    def test_insufficient_pose_uses_safe_class_values_and_logs_edge_event(self) -> None:
        machine = SessionStateMachine(RuntimeConfig())
        machine.state = AppState.WASHING
        machine.expected_step = 4
        application = self.application_for(machine)
        prediction = {
            "status": "insufficient_pose",
            "class_id": None,
            "class_name": None,
            "confidence_uncalibrated": None,
            "pose_coverage_ge1": 0.0,
            "pose_coverage_2": 0.0,
            "source_frame_count": 30,
            "track_fragmentation": 1,
        }

        with patch("app.main.LOGGER.info") as logger:
            events = application._process_prediction(prediction)

        line = self.prediction_line(logger)
        self.assertEqual([event.name for event in events], ["hands_lost_grace_started"])
        self.assertIn("status=insufficient_pose", line)
        self.assertIn("expected_step=4", line)
        self.assertIn("predicted_class=none", line)
        self.assertIn("class_name=none", line)
        self.assertIn("confidence=none", line)
        self.assertIn("pose_coverage=0.000", line)
        self.assertIn("events=hands_lost_grace_started", line)
        self.assertIn("runtime_state=WASHING", line)

    def test_warming_up_and_buffering_are_not_prediction_logs(self) -> None:
        application = self.application_for(SessionStateMachine(RuntimeConfig()))

        with patch("app.main.LOGGER.info") as logger:
            self.assertIsNone(application._process_prediction({"status": "warming_up"}))
            self.assertIsNone(application._process_prediction({"status": "buffering"}))

        self.assertFalse(logger.called)

    def test_missing_and_non_finite_metadata_use_sentinel_without_crashing(self) -> None:
        line = format_prediction_diagnostics(
            {
                "status": "prediction",
                "class_id": 4,
                "class_name": "step_4",
                "confidence_uncalibrated": math.nan,
                "pose_coverage_ge1": object(),
                "pose_coverage_2": math.inf,
                "track_fragmentation": "bad",
            },
            expected_step=4,
            runtime_state=AppState.CORRECTION,
            correct_streak=0,
            incorrect_streak=1,
            accepted_steps={3, 1, 2},
            events=[AppEvent("correction", 4), AppEvent("retry", 4)],
        )

        self.assertIn("confidence=none", line)
        self.assertIn("pose_coverage=none", line)
        self.assertIn("pose_coverage_2=none", line)
        self.assertIn("accepted_steps=1,2,3", line)
        self.assertIn("events=correction,retry", line)
        self.assertIn("track_fragmentation=none", line)

    def test_event_log_contains_post_observation_context(self) -> None:
        machine = SessionStateMachine(RuntimeConfig())
        machine.state = AppState.CORRECTION
        machine.expected_step = 4
        machine.correct_streak = 0
        machine.incorrect_streak = 1
        machine.accepted_steps = {3, 1, 2}
        application = self.application_for(machine)

        with patch("app.main.LOGGER.info") as logger:
            application._handle_events([AppEvent("retry", 4)])

        event_lines = [call.args[0] for call in logger.call_args_list]
        self.assertEqual(len(event_lines), 1)
        self.assertIn("Evento: retry value=4", event_lines[0])
        self.assertIn("runtime_state=CORRECTION", event_lines[0])
        self.assertIn("expected_step=4", event_lines[0])
        self.assertIn("correct_streak=0", event_lines[0])
        self.assertIn("incorrect_streak=1", event_lines[0])
        self.assertIn("accepted_steps=1,2,3", event_lines[0])


if __name__ == "__main__":
    unittest.main()
