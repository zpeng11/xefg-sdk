"""Positive SDK integration with a small injected executor, no capture files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from xefg_sdk import (Config, FrameInputs, ImageView, Rect,
                      UiMode, Xefg2xRuntime, Xefg2xSession)


class CopyExecutor:
    def __init__(self):
        self.loaded = []
        self.calls = []
        self.closed = []

    def load(self, path):
        self.loaded.append(path)
        return str(path)

    def run(self, handle, inputs, output_names):
        self.calls.append({k: v.copy() for k, v in inputs.items()})
        return {output_names[0]: inputs["current_color"].copy()}

    def close(self, handle):
        self.closed.append(handle)


def make_package(root):
    variants = {}
    for mode in (1, 3):
        variant_id = "mode%d" % mode
        model = root / (variant_id + ".onnx")
        model.write_bytes(b"executor-owned opaque model " + bytes([mode]))
        profile = {
            "schema_version": 1, "id": variant_id + "-4x2-mv2x1", "variant": variant_id,
            "output_size": [4, 2], "motion_size": [2, 1],
            "feeds": {
                "shape_feed": {"dtype": "int64", "shape": [3], "values": [2, 4, 4]},
                "ratio": {"dtype": "float32", "shape": [1], "values": [1.0]},
            },
        }
        profile_name = variant_id + ".json"
        (root / profile_name).write_text(json.dumps(profile), encoding="utf-8")
        variants[variant_id] = {
            "mode": mode, "model": model.name,
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "output": "generated_frame", "profiles": [profile_name],
            "inputs": ["previous_color", "current_color", "motion_vectors", "depth"]
                      + (["ui"] if mode == 3 else []),
        }
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "name": "xefg2x", "variants": variants}), encoding="utf-8")


def frame(number, ui=False, reset=False):
    color = np.full((2, 4, 4), number, dtype=np.uint8)
    return FrameInputs(
        number, backbuffer_color=ImageView(color, "BGRA8_UNORM"),
        hudless_color=ImageView(color, "BGRA8_UNORM") if ui else None,
        depth=ImageView(np.ones((1, 2), dtype=np.float32), "R32_FLOAT"),
        motion_vectors=ImageView(np.zeros((1, 2, 2), dtype=np.float16), "RG16_FLOAT"),
        ui=ImageView(np.zeros((2, 4, 4), dtype=np.uint8), "BGRA8_UNORM") if ui else None,
        reset_history=reset)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        make_package(Path(self.temp.name))
        self.executor = CopyExecutor()
        self.runtime = Xefg2xRuntime.load(self.temp.name, self.executor)

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    def test_pair_without_ui_uses_profile(self):
        result = self.runtime.interpolate_pair(frame(0), frame(1))
        self.assertEqual(result.status, "generated")
        self.assertEqual(result.ui_mode, UiMode.NONE)
        self.assertEqual(result.profile_id, "mode1-4x2-mv2x1")
        np.testing.assert_array_equal(result.generated_frame.data, frame(1).backbuffer_color.data)
        self.assertEqual(self.executor.calls[0]["shape_feed"].dtype, np.int64)
        self.assertEqual(self.executor.calls[0]["ratio"].dtype, np.float32)
        self.runtime.interpolate_pair(frame(1), frame(2))
        self.assertEqual(len(self.executor.loaded), 1)

    def test_pair_with_ui_selects_mode3(self):
        result = self.runtime.interpolate_pair(frame(0, ui=True), frame(1, ui=True))
        self.assertEqual(result.ui_mode, UiMode.HUDLESS_UITEXTURE)
        self.assertIn("ui", self.executor.calls[0])

    def test_session_owns_history_and_resets(self):
        session = Xefg2xSession(self.runtime)
        first = frame(0)
        self.assertEqual(session.submit(first).status, "history_ready")
        first.backbuffer_color.data.fill(99)
        result = session.submit(frame(1))
        self.assertEqual((result.previous_frame_id, result.current_frame_id), (0, 1))
        np.testing.assert_array_equal(self.executor.calls[0]["previous_color"], np.zeros((2, 4, 4), np.uint8))
        self.assertEqual(session.submit(frame(10, reset=True)).status, "history_ready")
        self.assertEqual(session.submit(frame(11)).previous_frame_id, 10)
        session.reset()
        self.assertEqual(session.submit(frame(20)).status, "history_ready")

    def test_rgba_and_float_motion_adapters(self):
        old, new = frame(0), frame(1)
        rgba = np.tile(np.array([10, 20, 30, 255], np.uint8), (2, 4, 1))
        new = FrameInputs(new.frame_id, backbuffer_color=ImageView(rgba, "RGBA8_UNORM"),
                          depth=new.depth, motion_vectors=ImageView(
                              np.full((1, 2, 2), 0.3333, np.float32), "RG32_FLOAT"))
        self.runtime.interpolate_pair(old, new)
        np.testing.assert_array_equal(self.executor.calls[0]["current_color"][0, 0], [30, 20, 10, 255])
        self.assertEqual(self.executor.calls[0]["motion_vectors"].dtype, np.float16)

    def test_prepare_does_not_load_engine(self):
        mode, plan, values = self.runtime.prepare(frame(0), frame(1))
        self.assertEqual(self.executor.loaded, [])
        self.assertEqual(mode, UiMode.NONE)
        self.assertEqual(plan.output_size, (4, 2))
        self.assertEqual(values["previous_color"].shape, (2, 4, 4))

    def test_core_exposes_only_model_inputs(self):
        from dataclasses import fields
        import xefg_sdk

        self.assertEqual({f.name for f in fields(Config)}, {"ui_mode", "profile_id"})
        self.assertEqual({f.name for f in fields(FrameInputs)}, {
            "frame_id", "backbuffer_color", "hudless_color", "depth",
            "motion_vectors", "ui", "reset_history"})
        self.assertFalse(hasattr(xefg_sdk, "FrameConstants"))

    def test_normalized_motion_and_depth_are_not_reinterpreted(self):
        old, current = frame(0), frame(1)
        # An adapter has already included camera/object motion and removed
        # engine jitter/scale. Core must not apply those operations again.
        motion = np.array([[[2.0, -1.0], [-4.0, 3.0]]], np.float32)
        depth = np.array([[1.0, 100.0]], np.float32)
        current = FrameInputs(
            1, backbuffer_color=current.backbuffer_color,
            motion_vectors=ImageView(motion, "RG32_FLOAT"),
            depth=ImageView(depth, "R32_FLOAT"))
        self.runtime.interpolate_pair(old, current)
        np.testing.assert_array_equal(self.executor.calls[0]["motion_vectors"], motion.astype(np.float16))
        np.testing.assert_array_equal(self.executor.calls[0]["depth"], depth)
        self.assertNotIn("view_matrix", self.executor.calls[0])

    def test_direct_pair_rejects_history_reset(self):
        with self.assertRaisesRegex(ValueError, "reset_history"):
            self.runtime.interpolate_pair(frame(0), frame(1, reset=True))

    def test_package_conventions_match_runtime(self):
        path = Path(self.temp.name) / "manifest.json"
        data = json.loads(path.read_text())
        data["conventions"] = {"motion_direction": "previous_to_current"}
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported model convention"):
            Xefg2xRuntime.load(self.temp.name, self.executor)

    def test_close_releases_each_loaded_variant(self):
        self.runtime.interpolate_pair(frame(0), frame(1))
        self.runtime.interpolate_pair(frame(0, ui=True), frame(1, ui=True))
        self.runtime.close()
        self.runtime.close()
        self.assertEqual(len(self.executor.closed), 2)

    def test_profile_cache_contains_only_metadata(self):
        _, plan, _ = self.runtime.prepare(frame(0), frame(1))
        self.assertEqual(set(plan.feeds), {"shape_feed", "ratio"})
        self.assertFalse(plan.feeds["shape_feed"].flags.writeable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
