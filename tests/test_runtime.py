from __future__ import annotations

import unittest

from app.config import RuntimeConfig
from app.runtime import InferenceRuntime


class InferenceRuntimeTests(unittest.TestCase):
    def test_temporal_reset_is_generation_tracked_without_reopening_camera(self) -> None:
        runtime = InferenceRuntime(RuntimeConfig(), camera_source="picamera2")

        first = runtime.reset_temporal_state()
        second = runtime.reset_temporal_state()

        self.assertEqual((first, second), (1, 2))
        self.assertTrue(runtime._temporal_reset_requested.is_set())
        self.assertEqual(runtime.camera_source, "picamera2")
        self.assertFalse(runtime._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
