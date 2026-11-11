import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("metrics", Path(__file__).parents[1] / "metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((2, 3, 4), dtype=np.uint8)

    def test_identical_json_is_finite(self):
        result = metrics.compare(self.image, self.image)
        self.assertTrue(result["rgb_identical"])
        self.assertIsNone(result["rgb_psnr_db"])
        json.dumps(result, allow_nan=False)

    def test_known_difference(self):
        other = self.image.copy()
        other[:, :, 0] = 3
        result = metrics.compare(self.image, other)
        self.assertEqual(result["rgb_mae"], 1)
        self.assertEqual(result["rgb_max"], 3)
        self.assertEqual(result["over_tolerance_pixel_fraction"], 1)

    def test_alpha_is_separate(self):
        other = self.image.copy()
        other[:, :, 3] = 255
        result = metrics.compare(self.image, other)
        self.assertTrue(result["rgb_identical"])
        self.assertEqual(result["alpha_max"], 255)

    def test_mask(self):
        other = self.image.copy()
        other[0] = 255
        mask = np.zeros((2, 3), dtype=bool)
        mask[1] = True
        self.assertTrue(metrics.compare(self.image, other, mask)["rgb_identical"])
        with self.assertRaisesRegex(ValueError, "empty"):
            metrics.compare(self.image, other, np.zeros_like(mask))

    def test_channel_order_and_gain(self):
        other = self.image.copy()
        other[:, :, 0] = 10
        self.assertEqual(metrics.rgb_image(other)[0, 0].tolist(), [0, 0, 10])
        self.assertEqual(metrics.difference_image(self.image, other)[0, 0].tolist(), [0, 0, 80])

    def test_shape_and_dtype(self):
        with self.assertRaises(ValueError):
            metrics.compare(self.image, self.image.astype(float))
        with self.assertRaises(ValueError):
            metrics.compare(self.image, self.image[:1])


if __name__ == "__main__":
    unittest.main()
