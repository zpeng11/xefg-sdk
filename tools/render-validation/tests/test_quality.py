import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("quality", Path(__file__).parents[1] / "check_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class QualityTests(unittest.TestCase):
    def report(self, mae=0.01, maximum=32, fraction=0.0002):
        return {"status": "measured", "pairs": [{"current_frame": 4, "valid_region": {
            "rgb_mae": mae, "rgb_max": maximum, "over_tolerance_pixel_fraction": fraction,
            "tolerance": 2}}]}

    def test_limits_inclusive(self):
        self.assertEqual(quality.evaluate(self.report())["status"], "pass")

    def test_each_limit_enforced(self):
        for values in ((0.01001, 8, 0.0002), (0.01, 33, 0.0002), (0.01, 8, 0.00021)):
            self.assertEqual(quality.evaluate(self.report(*values))["status"], "fail")

    def test_nonfinite_metrics_cannot_pass(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                quality.evaluate(self.report(mae=value))

    def test_mode_recorded_without_relaxing_limits(self):
        for mode in (1, 3, 4):
            report = {**self.report(maximum=33), "ui_mode": mode}
            result = quality.evaluate(report)
            self.assertEqual(result["ui_mode"], mode)
            self.assertEqual(result["status"], "fail")

    def test_channel_warning_does_not_fail(self):
        for maximum in (9, 20, 28, 32):
            result = quality.evaluate(self.report(maximum=maximum))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["warnings"][0]["rgb_max"], maximum)
        self.assertEqual(quality.evaluate(self.report(maximum=8))["warnings"], [])

    def test_warning_does_not_waive_other_limits(self):
        result = quality.evaluate(self.report(maximum=32, fraction=0.0003))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failures"][0]["exceeded"], ["over_2_fraction"])
        self.assertEqual(len(result["warnings"]), 1)

    def test_explicit_old_limit_still_enforced(self):
        for maximum, limit in ((9, 8), (28, 20)):
            result = quality.evaluate(self.report(maximum=maximum), maximum=limit)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["warnings"], [])

    def test_empty_not_pass(self):
        with self.assertRaises(ValueError):
            quality.evaluate({"status": "measured", "pairs": []})


if __name__ == "__main__":
    unittest.main()
