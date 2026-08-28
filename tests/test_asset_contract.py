from __future__ import annotations

import unittest

from app.config import IMAGE_FILES, expected_asset, required_paths


class AssetContractTests(unittest.TestCase):
    def test_all_required_files_exist(self) -> None:
        missing = [path for path in required_paths() if not path.is_file()]
        self.assertEqual(missing, [])

    def test_definitive_names_are_preserved(self) -> None:
        self.assertEqual(IMAGE_FILES["logo"], "LOGO_LAIA-1.png")
        self.assertEqual(IMAGE_FILES["waiting"], "silueta.png")
        self.assertEqual(IMAGE_FILES["success"], "manos_limpias.png")
        self.assertEqual([IMAGE_FILES[f"step_{step}"] for step in range(1, 7)],
                         [f"Paso{step}.png" for step in range(1, 7)])

    def test_correction_uses_expected_step_not_detected_step(self) -> None:
        self.assertEqual(expected_asset("CORRECTION", 3).name, "Paso3.png")
        self.assertNotEqual(expected_asset("CORRECTION", 3).name, "Paso5.png")

    def test_success_never_uses_step_six(self) -> None:
        self.assertEqual(expected_asset("SUCCESS", 6).name, "manos_limpias.png")

    def test_incomplete_uses_waiting_asset(self) -> None:
        self.assertEqual(expected_asset("INCOMPLETE", 1).name, "silueta.png")


if __name__ == "__main__":
    unittest.main()

