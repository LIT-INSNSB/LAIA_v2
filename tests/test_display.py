import unittest

from app.display import DisplayRotationController, parse_wlr_randr


SAMPLE = (
    'DSI-2 "(null) (null) (DSI-2)"\n'
    '  Enabled: yes\n'
    '  Modes:\n'
    '    800x480 px, 60.028999 Hz (preferred, current)\n'
    '  Position: 0,0\n'
    '  Transform: 270\n'
)


class DisplayParsingTests(unittest.TestCase):
    def test_parses_active_output_and_transform(self):
        display = parse_wlr_randr(SAMPLE)
        self.assertEqual(display.name, "DSI-2")
        self.assertEqual(display.transform, "270")
        self.assertEqual((display.width, display.height), (800, 480))
        self.assertEqual(display.logical_size, (480, 800))


    def test_selects_named_output(self):
        sample = 'HDMI-A-1\n  Enabled: yes\n  Transform: normal\n' + SAMPLE
        display = parse_wlr_randr(sample, "DSI-2")
        self.assertEqual(display.name, "DSI-2")

    def test_rotation_cycle(self):
        expected = {"normal": "90", "90": "180", "180": "270", "270": "normal"}
        for current, following in expected.items():
            self.assertEqual(DisplayRotationController.next_transform(current), following)


if __name__ == "__main__":
    unittest.main()
