from __future__ import annotations

import math
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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

    def test_prediction_line_logs_full_vectors_and_signed_score_diagnostics(self) -> None:
        prediction = {
            "status": "prediction",
            "class_id": 6,
            "class_name": "fingertips_to_palm",
            "confidence_uncalibrated": 0.52,
            "pose_coverage_ge1": 1.0,
            "pose_coverage_2": 1.0,
            "logits": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "probabilities": [1e-9, 0.01, 0.02, 0.03, 0.42, 0.10, 0.52],
            "source_frame_count": 30,
        }
        expected_entropy = -sum(
            probability * math.log(probability)
            for probability in prediction["probabilities"]
            if probability > 0
        )

        with patch("app.main.LOGGER.info") as logger:
            line = format_prediction_diagnostics(
                prediction,
                expected_step=4,
                runtime_state=AppState.WASHING,
                correct_streak=0,
                incorrect_streak=0,
                accepted_steps={1, 2, 3},
                events=(),
                session_id="session",
                reset_generation=9,
                window_diagnostics={
                    "timing_valid": True,
                    "window_start_s": 10.0,
                    "window_end_s": 11.45,
                    "window_span_s": 1.45,
                    "effective_window_fps": 20.0,
                    "mean_frame_dt_ms": 50.0,
                    "min_frame_dt_ms": 45.0,
                    "max_frame_dt_ms": 60.0,
                    "prediction_interval_ms": 400.0,
                    "frames_0_hands": 0,
                    "frames_1_hand": 2,
                    "frames_2_hands": 28,
                    "fraction_2_hands": 28 / 30,
                    "track_changes_in_window": 2,
                    "new_tracks_in_window": 3,
                    "track_fragmentations_in_window": 1,
                    "slot_changes_in_window": 1,
                    "unique_tracks_in_window": 3,
                    "hand_dropout_count": 1,
                    "longest_two_hand_segment_frames": 20,
                },
            )

        self.assertFalse(logger.called)
        self.assertIn("schema=prediction_v2", line)
        self.assertIn("reset_generation=9", line)
        self.assertIn("logit0=-1", line)
        self.assertIn("logit6=5", line)
        self.assertIn("p0=1e-09", line)
        self.assertIn("p6=0.52", line)
        self.assertIn("p_expected=0.42", line)
        self.assertIn("expected_rank=2", line)
        self.assertIn("top1=6", line)
        self.assertIn("top1_probability=0.52", line)
        self.assertIn("top2=4", line)
        self.assertIn("top2_probability=0.42", line)
        self.assertIn("probability_margin=-0.1", line)
        self.assertIn("logit_margin=-2", line)
        self.assertIn(f"entropy_nats={expected_entropy:.10g}", line)
        self.assertIn("argmax_check=true", line)
        self.assertIn("window_span_s=1.45", line)
        self.assertIn("effective_window_fps=20", line)
        self.assertIn("frames_2_hands=28", line)
        self.assertIn("track_fragmentations_in_window=1", line)

    def test_score_ordering_is_deterministic_and_expected_win_is_positive(self) -> None:
        line = format_prediction_diagnostics(
            {
                "status": "prediction",
                "class_id": 1,
                "class_name": "step_1",
                "confidence_uncalibrated": 0.2,
                "logits": [0.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
                "probabilities": [0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
            },
            expected_step=4,
            runtime_state=AppState.WASHING,
            correct_streak=0,
            incorrect_streak=0,
            accepted_steps=(),
            events=(),
        )

        self.assertIn("p_expected=0.1", line)
        self.assertIn("expected_rank=5", line)
        self.assertIn("top1=1", line)
        self.assertIn("top2=2", line)
        self.assertIn("top2_probability=0.2", line)
        self.assertIn("probability_margin=-0.1", line)
        self.assertIn("logit_margin=-3", line)
        self.assertIn("argmax_check=true", line)

        winning_line = format_prediction_diagnostics(
            {
                "status": "prediction",
                "class_id": 4,
                "class_name": "step_4",
                "confidence_uncalibrated": 0.8,
                "logits": [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0],
                "probabilities": [0.01, 0.02, 0.03, 0.04, 0.8, 0.05, 0.05],
            },
            expected_step=4,
            runtime_state=AppState.WASHING,
            correct_streak=0,
            incorrect_streak=0,
            accepted_steps=(),
            events=(),
        )
        self.assertIn("expected_rank=1", winning_line)
        self.assertIn("probability_margin=0.75", winning_line)
        self.assertIn("logit_margin=3", winning_line)

        mismatch_line = format_prediction_diagnostics(
            {
                "status": "prediction",
                "class_id": 6,
                "class_name": "step_6",
                "confidence_uncalibrated": 0.1,
                "logits": [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "probabilities": [0.9, 0.02, 0.01, 0.01, 0.01, 0.01, 0.04],
            },
            expected_step=4,
            runtime_state=AppState.WASHING,
            correct_streak=0,
            incorrect_streak=0,
            accepted_steps=(),
            events=(),
        )
        self.assertIn("top1=6", mismatch_line)
        self.assertIn("top1_probability=0.04", mismatch_line)
        self.assertIn("argmax_check=false", mismatch_line)

    def test_malformed_score_vectors_use_safe_sentinels_without_recalculation(self) -> None:
        prediction = {
            "status": "prediction",
            "class_id": 6,
            "class_name": "step_6",
            "confidence_uncalibrated": 0.4,
            "logits": [0.0, float("nan"), 2.0, 3.0, 4.0, 5.0, 6.0],
            "probabilities": [0.1, 0.2, float("inf"), 0.1, 0.1, 0.1, 0.4],
        }
        original = dict(prediction)

        line = format_prediction_diagnostics(
            prediction,
            expected_step=4,
            runtime_state=AppState.WASHING,
            correct_streak=0,
            incorrect_streak=0,
            accepted_steps=(),
            events=(),
        )

        self.assertIn("logit1=none", line)
        self.assertIn("p2=none", line)
        self.assertIn("p_expected=none", line)
        self.assertIn("expected_rank=none", line)
        self.assertIn("top2=none", line)
        self.assertIn("probability_margin=none", line)
        self.assertIn("logit_margin=none", line)
        self.assertIn("entropy_nats=none", line)
        self.assertIn("argmax_check=none", line)
        self.assertEqual(prediction, original)

    def test_spy_receives_original_class_id_and_reset_generation_reaches_log(self) -> None:
        class SpyMachine:
            state = AppState.WASHING
            expected_step = 4
            correct_streak = 1
            incorrect_streak = 0
            accepted_steps = set()

            def __init__(self):
                self.observed = None

            def observe(self, **kwargs):
                self.observed = kwargs
                return []

        machine = SpyMachine()
        application = self.application_for(machine)
        prediction = {
            "status": "prediction",
            "class_id": 6,
            "class_name": "fingertips_to_palm",
            "confidence_uncalibrated": 0.99,
            "pose_coverage_ge1": 1.0,
            "pose_coverage_2": 1.0,
            "logits": [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 6.0],
            "probabilities": [0.0, 0.0, 0.0, 0.0, 0.001, 0.009, 0.99],
        }

        with patch("app.main.LOGGER.info") as logger:
            application._process_prediction(prediction, reset_generation=12)

        self.assertEqual(machine.observed["class_id"], prediction["class_id"])
        self.assertNotEqual(machine.observed["class_id"], machine.expected_step)
        line = self.prediction_line(logger)
        self.assertIn("reset_generation=12", line)

    def test_poll_discards_stale_generation_and_passes_fresh_generation(self) -> None:
        class FakeRuntime:
            def __init__(self, updates):
                self.updates = list(updates)

            def latest(self):
                return self.updates.pop(0) if self.updates else None

        class FakeView:
            def set_frame(self, frame):
                pass

            def set_camera_detail(self, detail):
                pass

            def render(self):
                pass

        class FakeRoot:
            def after(self, delay, callback):
                pass

        machine = SessionStateMachine(RuntimeConfig())
        application = LaiaApplication.__new__(LaiaApplication)
        application._ui_actions = queue.SimpleQueue()
        application.machine = machine
        application.runtime = FakeRuntime(
            [
                SimpleNamespace(
                    frame_bgr=None,
                    frame_timestamp_s=1.0,
                    prediction={"status": "prediction", "class_id": 6},
                    error=None,
                    camera_backend=None,
                    inference_generation=1,
                ),
                SimpleNamespace(
                    frame_bgr=None,
                    frame_timestamp_s=2.0,
                    prediction={"status": "prediction", "class_id": 6},
                    error=None,
                    camera_backend=None,
                    inference_generation=3,
                ),
            ]
        )
        application._inference_generation = 2
        application.view = FakeView()
        application.args = SimpleNamespace(preview_state=None)
        application._last_state = machine.state
        application._last_frame = None
        application._camera_error = ""
        application._last_render_key = None
        application.root = FakeRoot()
        application.leds = SimpleNamespace(apply=lambda state: None)
        application._lift_controls = lambda: None
        application._process_prediction = Mock(return_value=[])

        application._poll()
        application._process_prediction.assert_not_called()
        self.assertEqual(application._inference_generation, 2)

        application._poll()
        application._process_prediction.assert_called_once_with(
            {"status": "prediction", "class_id": 6},
            reset_generation=3,
        )
        self.assertEqual(application._inference_generation, 3)

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
        self.assertNotIn("logit0=", line)
        self.assertNotIn("p0=", line)

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
