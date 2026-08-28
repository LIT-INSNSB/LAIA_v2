import unittest

from app.cameras import v4l2_device


class CameraSelectionTests(unittest.TestCase):
    def test_selected_v4l2_device_is_preserved(self):
        self.assertEqual(v4l2_device("v4l2:/dev/video7"), "/dev/video7")

    def test_auto_uses_legacy_index_as_fallback(self):
        self.assertEqual(v4l2_device("auto", 3), "/dev/video3")

    def test_rejects_non_video_device(self):
        with self.assertRaises(ValueError):
            v4l2_device("v4l2:/dev/media0")


if __name__ == "__main__":
    unittest.main()
