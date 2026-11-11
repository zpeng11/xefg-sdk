import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("quality", Path(__file__).parents[1] / "check_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class QualityTests(unittest.TestCase):
    def report(self, mae=0.01, maximum=8, fraction=0.0002):
        return {"status": "measured", "pairs": [{"current_frame": 4, "valid_region": {
            "rgb_mae": mae, "rgb_max": maximum, "over_tolerance_pixel_fraction": fraction,
            "tolerance": 2}}]}

    def test_limits_inclusive(self):
        self.assertEqual(quality.evaluate(self.report())["status"], "pass")

    def test_each_limit_enforced(self):
        for values in ((0.01001, 8, 0.0002), (0.01, 9, 0.0002), (0.01, 8, 0.00021)):
            self.assertEqual(quality.evaluate(self.report(*values))["status"], "fail")

    def test_empty_not_pass(self):
        with self.assertRaises(ValueError):
            quality.evaluate({"status": "measured", "pairs": []})


if __name__ == "__main__":
    unittest.main()
