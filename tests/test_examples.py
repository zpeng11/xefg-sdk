"""Normalized application manifest loading, independent of capture formats."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_COMMON = EXAMPLES / "_common.py"
spec = importlib.util.spec_from_file_location("_xefg_example_common", _COMMON)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load example helper: %s" % _COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class ExampleTests(unittest.TestCase):
    def test_normalized_manifest_and_direct_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "frame.npz", frame_id=np.int64(7), reset_history=True,
                     backbuffer_color=np.zeros((2, 4, 4), np.uint8),
                     motion_vectors=np.ones((1, 2, 2), np.float16),
                     depth=np.full((1, 2), 100.0, np.float32))
            path = root / "sequence.json"
            path.write_text(json.dumps({
                "schema_version": 1, "package": "model",
                "config": {"ui_mode": "NONE"},
                "frames": [{"path": "frame.npz"},
                           {"path": "frame.npz", "reset_history": False}],
            }), encoding="utf-8")
            args = argparse.Namespace(manifest=str(path), ui_mode=None,
                                      profile=None, package=None)
            defaults, entries, package = common.open_sequence(args)
            first = common.load_frame(entries[0], defaults)
            second = common.load_frame(entries[1], defaults)
            self.assertEqual(first.frame_id, 7)
            self.assertTrue(first.reset_history)
            self.assertFalse(second.reset_history)
            self.assertEqual(package, root / "model")
            np.testing.assert_array_equal(first.depth.data, [[100.0, 100.0]])
            self.assertFalse(hasattr(first, "constants"))

    def test_old_engine_constants_get_migration_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.json"
            path.write_text(json.dumps({
                "schema_version": 1, "package": "model", "frames": ["frame.npz"],
                "constants": {"jitter_offset_x": 0.25},
            }), encoding="utf-8")
            args = argparse.Namespace(manifest=str(path), ui_mode=None,
                                      profile=None, package=None)
            with self.assertRaisesRegex(ValueError, "normalized inputs"):
                common.open_sequence(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
