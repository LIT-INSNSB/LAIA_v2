from __future__ import annotations

from types import SimpleNamespace
import unittest

from app.config import audio_path
from app.main import LaiaApplication, audio_key_for_event
from app.state_machine import AppEvent, AppState


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


if __name__ == "__main__":
    unittest.main()
