import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("layout", Path(__file__).parents[1] / "inspect_tag_layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class TagLayoutTests(unittest.TestCase):
    def test_exact_rectangles_reconstruct_observation(self):
        mask = np.zeros((12, 20), dtype=bool)
        mask[:3, :4] = True
        mask[:, 2:4] = True
        mask[9:, 16:] = True
        rects = layout.rectangles(mask)
        rebuilt = np.zeros_like(mask)
        for x0, y0, x1, y1 in rects:
            rebuilt[y0:y1, x0:x1] = True
        np.testing.assert_array_equal(mask, rebuilt)
        self.assertEqual(rects, [[0, 0, 4, 3], [2, 3, 4, 12], [16, 9, 20, 12]])

    def test_no_purple_yields_no_rectangles(self):
        self.assertEqual(layout.rectangles(np.zeros((2, 4), dtype=bool)), [])


if __name__ == "__main__":
    unittest.main()
