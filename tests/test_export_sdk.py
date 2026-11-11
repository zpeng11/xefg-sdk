"""Tests for the portable XeFG 2x ONNX SDK package exporter.

The focused tests build a tiny synthetic assembled graph with the exact
alias/LUT input names and exercise the rewrite without a capture.  Capture
tests run the real `m9_dynamic.build_specs()`/`Assembler` path against the
640x360 mode-3 and 1080p mode-1 retention captures when they are explicitly
configured through the documented environment variables:

    XEFG_SDK_MODE3_RUN      640x360 mode-3 capture run
    XEFG_SDK_MODE1_RUN      1080p mode-1 capture run
    XEFG_SDK_MODE3_SIZE_RUN second mode-3 capture size

Each variable is opt-in.  When it is absent the capture class is skipped; when
it is set but the run has no ``trace/trace.jsonl`` the test fails instead of
silently skipping a broken configuration.

Install the package first (``python -m pip install -e ".[test]"``), then run:

    python -I -B -m unittest discover -s tests -v
"""
import json
import os
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from xefg_sdk.export import export_sdk, m9_dynamic


def _capture_run(env_name):
    """Resolve an explicitly configured capture run, or skip when unset."""
    value = os.environ.get(env_name)
    if not value:
        raise unittest.SkipTest(
            "%s is not set; capture integration is opt-in" % env_name)
    run_dir = os.path.abspath(value)
    trace = os.path.join(run_dir, "trace", "trace.jsonl")
    if not os.path.isfile(trace):
        raise AssertionError(
            "%s=%s is configured but %s is missing"
            % (env_name, run_dir, trace))
    return run_dir


def _vi(name, elem, shape):
    return helper.make_tensor_value_info(name, elem, shape)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _synthetic_graph(mode, height=2, width=3, motion_height=1,
                     motion_width=2):
    """A tiny graph carrying every verified alias and the gate LUT."""
    color_shape = [height, width, 4]
    flat_colors = height * width * 4
    motion_shape = [motion_height, motion_width, 2]
    flat_motion = motion_height * motion_width * 2
    depth_shape = [motion_height, motion_width]
    flat_depth = motion_height * motion_width

    inputs = [
        _vi("n0001_x0", TensorProto.UINT8, color_shape),
        _vi("n0001_x1", TensorProto.UINT8, color_shape),
        _vi("n0025_x1", TensorProto.UINT8, [flat_colors]),
        _vi("n0025_x2", TensorProto.UINT8, [flat_colors]),
        _vi("n0024_x0", TensorProto.FLOAT16, motion_shape),
        _vi("n0025_x3", TensorProto.FLOAT, [flat_motion]),
        _vi("n0024_x1", TensorProto.FLOAT, depth_shape),
        _vi("n0025_x4", TensorProto.FLOAT, [flat_depth]),
        _vi("n0025_x5", TensorProto.FLOAT, [export_sdk.LUT_SIZE]),
        _vi("n0038_in_t11", TensorProto.UINT8, color_shape),
        _vi("n0038_in_t12", TensorProto.UINT8, color_shape),
        _vi("n0001_fine_w", TensorProto.INT64, [1]),
        _vi("n0001_u0_to_0002_t0_shape", TensorProto.INT64, [4]),
    ]
    if mode == 3:
        inputs.append(_vi("n0038_in_t9", TensorProto.UINT8, color_shape))
    elif mode == 4:
        inputs.append(_vi("n0038_in_t8", TensorProto.UINT8, color_shape))
        inputs.append(_vi("n0038_in_t10", TensorProto.UINT8, color_shape))

    nodes = [
        helper.make_node("Identity", ["n0001_x0"], ["n0001_ds_u0_out"]),
        helper.make_node("Identity", ["n0025_x1"], ["n0025_prev_used"]),
        helper.make_node("Identity", ["n0025_x2"], ["n0025_curr_used"]),
        helper.make_node("Identity", ["n0025_x3"], ["n0025_mv_used"]),
        helper.make_node("Identity", ["n0025_x4"], ["n0025_depth_used"]),
        helper.make_node("Identity", ["n0024_x0"], ["n0024_mv_used"]),
        helper.make_node("Identity", ["n0024_x1"], ["n0024_depth_used"]),
        helper.make_node("Identity", ["n0025_x5"], ["n0025_lut_used"]),
        helper.make_node("Identity", ["n0038_in_t12"],
                         ["n0038_curr_used"]),
        helper.make_node("Identity", ["n0001_fine_w"],
                         ["n0001_fine_used"]),
        helper.make_node("Identity", ["n0001_u0_to_0002_t0_shape"],
                         ["n0001_shape_used"]),
        helper.make_node("Identity", ["n0038_in_t11"], ["n0038_u0"]),
    ]
    if mode == 3:
        nodes.append(helper.make_node("Identity", ["n0038_in_t9"],
                                      ["n0038_ui_used"]))
    elif mode == 4:
        nodes.append(helper.make_node("Identity", ["n0038_in_t8"],
                                      ["n0038_current_used"]))
        nodes.append(helper.make_node("Identity", ["n0038_in_t10"],
                                      ["n0038_backbuffer_used"]))
    outputs = [
        _vi("n0001_ds_u0_out", TensorProto.UINT8, color_shape),
        _vi("n0038_u0", TensorProto.UINT8, color_shape),
    ]
    graph = helper.make_graph(nodes, "synthetic_m9", inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def _synthetic_feeds(mode, height=2, width=3, motion_height=1,
                     motion_width=2):
    color_shape = (height, width, 4)
    flat_colors = height * width * 4
    previous = np.arange(flat_colors, dtype=np.uint8).reshape(color_shape)
    current = np.flipud(previous).copy()
    motion = np.arange(motion_height * motion_width * 2,
                       dtype=np.float16).reshape(
                           motion_height, motion_width, 2)
    depth = np.arange(motion_height * motion_width,
                      dtype=np.float32).reshape(
                          motion_height, motion_width)
    feeds = {
        "n0001_x0": previous,
        "n0001_x1": current,
        "n0025_x1": previous.reshape(-1),
        "n0025_x2": current.reshape(-1),
        "n0024_x0": motion,
        "n0025_x3": motion.astype(np.float32).reshape(-1),
        "n0024_x1": depth,
        "n0025_x4": depth.reshape(-1),
        "n0025_x5": np.linspace(0.0, 1.0, export_sdk.LUT_SIZE, dtype=np.float32),
        "n0038_in_t11": previous,
        "n0038_in_t12": current,
        "n0001_fine_w": np.asarray([7], dtype=np.int64),
        "n0001_u0_to_0002_t0_shape": np.asarray(
            [1, motion_height, motion_width, 4], dtype=np.int64),
    }
    if mode == 3:
        feeds["n0038_in_t9"] = np.full(color_shape, 9, dtype=np.uint8)
    elif mode == 4:
        # The mode 4 composite consumes the current hudless colour as t8 and
        # the current backbuffer as t10.  t9 aliases t10 but is not consumed,
        # so it is deliberately absent from this graph.
        feeds["n0038_in_t8"] = current
        feeds["n0038_in_t10"] = np.full(color_shape, 200, dtype=np.uint8)
    return feeds


class SyntheticRewriteTests(unittest.TestCase):
    """The rewrite is driven by verified alias names, not by the capture."""

    def rewrite(self, mode):
        model = _synthetic_graph(mode)
        feeds = _synthetic_feeds(mode)
        return model, feeds, export_sdk.rewrite_graph(
            model, feeds, mode, want_debug=True)

    def test_exporter_modules_import(self):
        self.assertTrue(callable(export_sdk.main))
        self.assertTrue(callable(m9_dynamic.build_specs))

    def test_public_inputs_and_lut_are_folded(self):
        model, feeds, rewrite = self.rewrite(3)
        names = [value.name for value in rewrite.public_model.graph.input]

        self.assertEqual(
            names[:5],
            ["previous_color", "current_color", "motion_vectors", "depth",
             "ui"])
        for alias in ("n0001_x0", "n0001_x1", "n0025_x1", "n0025_x2",
                      "n0024_x0", "n0025_x3", "n0024_x1", "n0025_x4",
                      "n0038_in_t11", "n0038_in_t12", "n0038_in_t9"):
            self.assertNotIn(alias, names)
        self.assertIn("n0001_fine_w", names)
        self.assertIn("n0001_u0_to_0002_t0_shape", names)
        self.assertNotIn("n0025_x5", names)

        lut = next(tensor for tensor in rewrite.public_model.graph.initializer
                   if tensor.name == "n0025_x5")
        self.assertTrue(np.array_equal(numpy_helper.to_array(lut),
                                       feeds["n0025_x5"]))
        self.assertEqual(rewrite.info["public_output"], "generated_frame")
        self.assertEqual([value.name for value
                          in rewrite.public_model.graph.output],
                         ["generated_frame"])
        self.assertEqual(len(rewrite.public_model.graph.output[0]
                             .type.tensor_type.shape.dim), 3)
        onnx.checker.check_model(rewrite.public_model)

        adapters = {node.output[0]: node.op_type
                    for node in rewrite.public_model.graph.node
                    if node.name.startswith("sdk_")
                    and node.op_type != "Identity"}
        self.assertEqual(adapters["n0025_x1"], "Reshape")
        self.assertEqual(adapters["n0025_x2"], "Reshape")
        self.assertEqual(adapters["n0025_x3"], "Reshape")
        self.assertEqual(adapters["n0025_x4"], "Reshape")

    def test_profile_feeds_exclude_images_and_keep_scalars(self):
        _model, feeds, rewrite = self.rewrite(3)
        profile = export_sdk.collect_profile_feeds(
            feeds, rewrite.info["removed_inputs"])
        self.assertEqual(set(profile), {"n0001_fine_w",
                                        "n0001_u0_to_0002_t0_shape"})
        for record in profile.values():
            self.assertLessEqual(len(record["values"]), 8)
        self.assertEqual(profile["n0001_fine_w"]["dtype"], "int64")
        self.assertEqual(profile["n0001_fine_w"]["values"], [7])
        self.assertEqual(profile["n0001_u0_to_0002_t0_shape"]["values"],
                         [1, 1, 2, 4])

    def test_debug_model_keeps_original_outputs(self):
        _model, _feeds, rewrite = self.rewrite(3)
        self.assertIsNotNone(rewrite.debug_model)
        self.assertEqual(
            [value.name for value in rewrite.debug_model.graph.output],
            ["n0001_ds_u0_out", "n0038_u0"])
        onnx.checker.check_model(rewrite.debug_model)

    def test_mode1_has_no_ui(self):
        _model, _feeds, rewrite = self.rewrite(1)
        names = [value.name for value in rewrite.public_model.graph.input]
        self.assertEqual(names[:4], ["previous_color", "current_color",
                                     "motion_vectors", "depth"])
        self.assertNotIn("ui", names)
        onnx.checker.check_model(rewrite.public_model)

    def test_mode4_folds_current_color_and_backbuffer(self):
        _model, feeds, rewrite = self.rewrite(4)
        names = [value.name for value in rewrite.public_model.graph.input]
        self.assertEqual(names[:5], ["previous_color", "current_color",
                                     "motion_vectors", "depth", "backbuffer"])
        for alias in ("n0001_x0", "n0001_x1", "n0025_x1", "n0025_x2",
                      "n0024_x0", "n0025_x3", "n0024_x1", "n0025_x4",
                      "n0038_in_t11", "n0038_in_t12", "n0038_in_t8",
                      "n0038_in_t10"):
            self.assertNotIn(alias, names)
        self.assertNotIn("ui", names)
        self.assertNotIn("n0038_in_t9", names)
        self.assertIn("n0001_fine_w", names)
        self.assertIn("n0001_u0_to_0002_t0_shape", names)
        self.assertNotIn("n0025_x5", names)
        self.assertEqual(rewrite.info["public_inputs"], [
            "previous_color", "current_color", "motion_vectors", "depth",
            "backbuffer"])
        self.assertIn("n0038_in_t8", rewrite.info["direct_aliases"])
        self.assertIn("n0038_in_t10", rewrite.info["direct_aliases"])
        self.assertNotIn("n0038_in_t9",
                         rewrite.info["removed_inputs"])
        self.assertEqual([value.name for value in
                          rewrite.public_model.graph.output],
                         ["generated_frame"])
        onnx.checker.check_model(rewrite.public_model)
        onnx.checker.check_model(rewrite.debug_model)

    def test_mode4_missing_backbuffer_is_rejected(self):
        model = _synthetic_graph(4)
        feeds = _synthetic_feeds(4)
        del feeds["n0038_in_t10"]
        with self.assertRaisesRegex(export_sdk.ExportError,
                                    "n0038_in_t10"):
            export_sdk.rewrite_graph(model, feeds, 4)

    def test_mode4_missing_current_alias_is_rejected(self):
        model = _synthetic_graph(4)
        feeds = _synthetic_feeds(4)
        del feeds["n0038_in_t8"]
        with self.assertRaisesRegex(export_sdk.ExportError,
                                    "n0038_in_t8"):
            export_sdk.rewrite_graph(model, feeds, 4)

    def test_unknown_external_tensor_is_rejected(self):
        model = _synthetic_graph(3)
        feeds = _synthetic_feeds(3)
        model.graph.input.append(
            _vi("n9999_external", TensorProto.UINT8, [64]))
        feeds["n9999_external"] = np.zeros(64, dtype=np.uint8)
        with self.assertRaises(export_sdk.ExportError):
            export_sdk.rewrite_graph(model, feeds, 3)

    def test_mode_conflicts_are_rejected(self):
        model3 = _synthetic_graph(3)
        feeds3 = _synthetic_feeds(3)
        with self.assertRaises(export_sdk.ExportError):
            export_sdk.rewrite_graph(model3, feeds3, 1)

        model1 = _synthetic_graph(1)
        feeds1 = _synthetic_feeds(1)
        with self.assertRaises(export_sdk.ExportError):
            export_sdk.rewrite_graph(model1, feeds1, 3)

    def test_mode5_is_still_rejected(self):
        model = _synthetic_graph(3)
        feeds = _synthetic_feeds(3)
        with self.assertRaisesRegex(export_sdk.ExportError,
                                    "exports 1, 3 and 4"):
            export_sdk.rewrite_graph(model, feeds, 5)

    def test_composite_plan_mode_selects_the_variant(self):
        for mode in (1, 3, 4):
            self.assertEqual(
                export_sdk._plan_mode({38: {"plan": {"mode": mode}}}, "x"),
                mode)
        for mode in (0, 2, 5):
            with self.assertRaises(export_sdk.ExportError):
                export_sdk._plan_mode({38: {"plan": {"mode": mode}}}, "x")


class SyntheticCaptureValidationTests(unittest.TestCase):
    """Export-time alias/shape checks without a capture."""

    def capture(self, mode=4):
        feeds = _synthetic_feeds(mode)
        height, width = 2, 3
        feeds.update({
            "n0025_wi_motion_scale": np.asarray([4.0], np.float32),
            "n0025_wi_ratio": np.asarray([1.0], np.float32),
            "n0025_wi_lo_ax": np.asarray([0.5001], np.float32),
            "n0025_wi_lo_ay": np.asarray([0.5001], np.float32),
            "n0025_wi_hi_ax": np.asarray([width - 0.5001], np.float32),
            "n0025_wi_hi_ay": np.asarray([height - 0.5001], np.float32),
        })
        model = _synthetic_graph(mode)
        for name in ("n0024_sp_scale_x", "n0024_sp_scale_y"):
            model.graph.initializer.append(numpy_helper.from_array(
                np.asarray([0.5], np.float32), name))
        return SimpleNamespace(
            run_dir="synthetic", frame=1, meta={}, mode=mode, specs={},
            dispatches=[], assembler=None, model=model, feeds=feeds,
            output_size=(height, width), motion_size=(1, 2), config={})

    def test_mode4_backbuffer_shape_and_dtype(self):
        capture = self.capture()
        export_sdk._validate_capture(capture)

        capture.feeds["n0038_in_t10"] = np.zeros((2, 2, 4), np.uint8)
        with self.assertRaisesRegex(export_sdk.ExportError,
                                    "backbuffer shape"):
            export_sdk._validate_capture(capture)

        capture = self.capture()
        capture.feeds["n0038_in_t10"] = np.zeros((2, 3, 4), np.float32)
        with self.assertRaisesRegex(export_sdk.ExportError,
                                    "backbuffer is not uint8"):
            export_sdk._validate_capture(capture)

    def test_mode4_current_color_aliases_must_match(self):
        capture = self.capture()
        capture.feeds["n0038_in_t8"] = np.zeros_like(
            capture.feeds["n0001_x1"])
        with self.assertRaisesRegex(export_sdk.ExportError,
                                    "bytes differ"):
            export_sdk._validate_capture(capture)


class ProfileRegistryTests(unittest.TestCase):
    def profile(self, value):
        return {
            "schema_version": 1,
            "id": "mode1-2x1-mv1x1",
            "variant": "mode1",
            "output_size": [2, 1],
            "motion_size": [1, 1],
            "feeds": {"n0001_fine_w": {
                "dtype": "int64", "shape": [1], "values": [value]}},
            "configuration": {"mode": 1},
        }

    def test_identical_profiles_are_deduplicated(self):
        registry = export_sdk.ProfileRegistry()
        first = registry.add(self.profile(1))
        second = registry.add(self.profile(1))
        self.assertEqual(first, second)
        self.assertEqual(len(registry.profiles()), 1)
        self.assertEqual(first["id"], "mode1-2x1-mv1x1")

    def test_conflicting_profile_gets_a_digest_suffix(self):
        registry = export_sdk.ProfileRegistry()
        first = registry.add(self.profile(1))
        second = registry.add(self.profile(2))
        self.assertEqual(first["id"], "mode1-2x1-mv1x1")
        self.assertTrue(second["id"].startswith("mode1-2x1-mv1x1-"))
        self.assertEqual(len(registry.profiles()), 2)

    def test_same_digest_with_other_metadata_is_rejected(self):
        registry = export_sdk.ProfileRegistry()
        first = registry.add(self.profile(1))
        canonical = registry._canonical(self.profile(2))
        digest = export_sdk.hashlib.sha256(
            canonical.encode("utf-8")).hexdigest()[:8]
        registry.entries[first["id"] + "-" + digest] = {
            "profile": {"id": first["id"] + "-" + digest},
            "canonical": "different",
        }
        with self.assertRaises(export_sdk.ExportError):
            registry.add(self.profile(2))


class ConfigurationTests(unittest.TestCase):
    """Profile configuration describes normalized inputs, not engine state."""

    def test_configuration_has_no_engine_conventions(self):
        for mode in (1, 3, 4):
            configuration = export_sdk._configuration(mode)
            for removed in ("view_matrix", "projection_matrix",
                            "motion_vector_scale", "jitter"):
                self.assertNotIn(removed, configuration)
            self.assertNotIn("identity", json.dumps(configuration))
            self.assertEqual(configuration["input_contract"], "normalized")
            self.assertEqual(configuration["motion_units"], "mv_pixels")
            self.assertEqual(configuration["motion_direction"],
                             "current_to_previous")
            self.assertFalse(configuration["jittered_motion"])
            self.assertFalse(configuration["depth_inverted"])
            self.assertTrue(configuration["ui_premultiplied"])
            self.assertEqual(configuration["motion_vector_resolution"], "half")
            self.assertTrue(configuration["full_rect"])
            self.assertEqual(configuration["baked"]["splat_scale"],
                             [0.5, 0.5])
            self.assertEqual(configuration["profile_fixed"],
                             {"warp_motion_scale": 4.0,
                              "temporal_interpolation_ratio": 1.0})
            self.assertEqual(configuration["ui_enabled"], mode in (3, 4))

    def test_mode4_configuration_only_enables_ui(self):
        base = export_sdk._configuration(1)
        for mode in (3, 4):
            configuration = export_sdk._configuration(mode)
            self.assertEqual(set(configuration), set(base))
            self.assertEqual(
                {key: value for key, value in configuration.items()
                 if key != "ui_enabled"},
                {key: value for key, value in base.items()
                 if key != "ui_enabled"})
            self.assertTrue(configuration["ui_enabled"])


class CaptureMode3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = _capture_run("XEFG_SDK_MODE3_RUN")
        cls.capture = export_sdk.build_capture(cls.run_dir, 1)
        cls.rewrite = export_sdk.rewrite_graph(
            cls.capture.model, cls.capture.feeds, cls.capture.mode,
            want_debug=True)

    def test_capture_and_alias_identity(self):
        self.assertEqual(self.capture.mode, 3)
        self.assertEqual(self.capture.output_size, (360, 640))
        self.assertEqual(self.capture.motion_size, (180, 320))
        self.assertIn("n0038_in_t9", self.capture.feeds)
        self.assertEqual(self.capture.feeds["n0025_x5"].size,
                         export_sdk.LUT_SIZE)

    def test_public_model_names_and_lut(self):
        model = self.rewrite.public_model
        names = [value.name for value in model.graph.input]
        self.assertEqual(names[:5], ["previous_color", "current_color",
                                     "motion_vectors", "depth", "ui"])
        self.assertNotIn(export_sdk.LUT_INPUT, names)
        self.assertEqual([value.name for value in model.graph.output],
                         ["generated_frame"])
        onnx.checker.check_model(model)
        lut = next(tensor for tensor in model.graph.initializer
                   if tensor.name == export_sdk.LUT_INPUT)
        self.assertTrue(np.array_equal(
            numpy_helper.to_array(lut), self.capture.feeds[export_sdk.LUT_INPUT]))
        self.assertEqual(len(model.graph.input), 312)
        self.assertEqual(len(model.graph.output), 1)

    def test_profile_feeds_are_small_and_exclude_capture_data(self):
        profile = export_sdk.collect_profile_feeds(
            self.capture.feeds, self.rewrite.info["removed_inputs"])
        self.assertEqual(len(profile), 307)
        self.assertLessEqual(max(len(record["values"])
                                 for record in profile.values()), 8)
        for name in ("previous_color", "current_color", "motion_vectors",
                     "depth", "ui", "n0001_x0", "n0025_x1", "n0025_x5"):
            self.assertNotIn(name, profile)
        self.assertEqual(
            [entry.name for entry in self.rewrite.debug_model.graph.output],
            [entry.name for entry in self.capture.model.graph.output])

    def test_graph_is_identical_between_frames(self):
        second = export_sdk.build_capture(self.run_dir, 2)
        second_rewrite = export_sdk.rewrite_graph(
            second.model, second.feeds, second.mode)
        self.assertEqual(
            self.rewrite.public_model.SerializeToString(),
            second_rewrite.public_model.SerializeToString())
        self.assertEqual(self.rewrite.info["initializer_digest"],
                         second_rewrite.info["initializer_digest"])

    def test_mode3_package(self):
        with tempfile.TemporaryDirectory() as out:
            manifest = export_sdk.export_package(
                [self.run_dir], out, debug=True)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["name"], "xefg2x")
            self.assertEqual(manifest["conventions"], {
                "color_format": "BGRA8_UNORM",
                "motion_format": "RG16_FLOAT",
                "motion_units": "mv_pixels",
                "motion_direction": "current_to_previous",
                "depth_inverted": False,
                "jittered_motion": False,
                "ui_premultiplied": True,
            })
            self.assertEqual(manifest["input_contract"]["scope"],
                             "normalized model inputs")
            reference = manifest["source"]["reference_capture_conventions"]
            self.assertEqual(reference["motion_vector_scale"], [1.0, 1.0])
            self.assertEqual(reference["jitter"], [0.0, 0.0])
            self.assertIn("not a universal requirement", reference["note"])
            self.assertEqual(set(manifest["variants"]), {"mode3"})
            variant = manifest["variants"]["mode3"]
            self.assertEqual(variant["mode"], 3)
            self.assertEqual(variant["output"], "generated_frame")
            self.assertEqual(variant["inputs"], [
                "previous_color", "current_color", "motion_vectors", "depth",
                "ui"])
            self.assertEqual(len(variant["profiles"]), 1)
            model_path = os.path.join(out, variant["model"])
            self.assertEqual(export_sdk._file_sha256(model_path),
                             variant["model_sha256"])
            model = onnx.load(model_path)
            self.assertEqual([value.name for value in model.graph.output],
                             ["generated_frame"])
            onnx.checker.check_model(model)

            profile_path = os.path.join(out, variant["profiles"][0])
            profile = _read_json(profile_path)
            self.assertEqual(profile["schema_version"], 1)
            self.assertEqual(profile["variant"], "mode3")
            self.assertEqual(profile["output_size"], [640, 360])
            self.assertEqual(profile["motion_size"], [320, 180])
            self.assertEqual(len(profile["feeds"]), 307)
            self.assertNotIn("n0001_x0", profile["feeds"])
            self.assertNotIn("n0025_x5", profile["feeds"])
            self.assertLessEqual(
                max(len(entry["values"])
                    for entry in profile["feeds"].values()), 8)

            configuration = profile["configuration"]
            for removed in ("view_matrix", "projection_matrix",
                            "motion_vector_scale", "jitter"):
                self.assertNotIn(removed, configuration)
            self.assertEqual(configuration["input_contract"], "normalized")
            self.assertEqual(configuration["color_format"], "BGRA8_UNORM")
            self.assertEqual(configuration["motion_format"], "RG16_FLOAT")
            self.assertEqual(configuration["motion_units"], "mv_pixels")
            self.assertEqual(configuration["motion_direction"],
                             "current_to_previous")
            self.assertEqual(configuration["motion_vector_resolution"], "half")
            self.assertFalse(configuration["jittered_motion"])
            self.assertEqual(configuration["depth_format"], "R32_FLOAT")
            self.assertFalse(configuration["depth_inverted"])
            self.assertTrue(configuration["full_rect"])
            self.assertEqual(configuration["interpolation_factor"], 2)
            self.assertEqual(configuration["interpolation_position"], 0.5)
            self.assertTrue(configuration["ui_premultiplied"])
            self.assertTrue(configuration["ui_enabled"])
            self.assertEqual(configuration["baked"],
                             {"splat_scale": [0.5, 0.5]})
            self.assertEqual(configuration["profile_fixed"], {
                "warp_motion_scale": 4.0,
                "temporal_interpolation_ratio": 1.0,
            })
            self.assertNotIn("identity", json.dumps(configuration))

            contract = _read_json(
                os.path.join(out, "io_contract.json"))
            self.assertEqual(contract["input_contract"]["scope"],
                             "normalized model inputs")
            mode = contract["variants"]["mode3"]
            self.assertEqual(mode["graph"]["original_inputs"], 319)
            self.assertEqual(mode["graph"]["original_outputs"], 42)
            self.assertEqual(mode["graph"]["original_initializers"], 2497)
            self.assertEqual(len(mode["inputs"]), 319)
            self.assertEqual(len(mode["outputs"]), 42)
            by_name = {entry["graph_name"]: entry
                       for entry in mode["inputs"]}
            for name, slot, submodel in (
                    ("n0001_x0", "t0", "x0"),
                    ("n0001_x1", "t1", "x1"),
                    ("n0025_x1", "t2", "x1"),
                    ("n0025_x2", "t3", "x2"),
                    ("n0025_x3", "t4", "x3"),
                    ("n0025_x4", "t5", "x4"),
                    ("n0025_x5", "t6", "x5"),
                    ("n0024_x0", "t0", "x0"),
                    ("n0024_x1", "t1", "x1"),
                    ("n0038_in_t9", "t9", "in_t9"),
                    ("n0038_in_t11", "t11", "in_t11"),
                    ("n0038_in_t12", "t12", "in_t12")):
                self.assertEqual(by_name[name]["slot"], slot, name)
                self.assertEqual(by_name[name]["submodel_input"], submodel,
                                 name)
            self.assertIsNone(by_name["n0001_fine_w"]["slot"])
            self.assertEqual(by_name["n0001_fine_w"]["field_name"], "fine_w")
            self.assertEqual(
                by_name["n0025_wi_motion_scale"]["semantic"],
                "configuration.warp_motion_scale")
            self.assertIn("c47",
                          by_name["n0025_wi_motion_scale"]["evidence"])
            self.assertEqual(
                by_name["n0011_pack_rows_alloc"]["field_name"], "rows_alloc")
            self.assertEqual(
                by_name["n0001_u0_to_0002_t0_shape"]["field_name"],
                "x0_shape")
            self.assertIn("model_asset/gate_lut", mode["initializers"]["by_category"])
            self.assertEqual(
                mode["initializers"]["by_category"]
                ["configuration/splat_motion_scale"], 2)
            seeds = [item for item in mode["initializers"]["items"]
                     if item["subcategory"] == "seed"]
            self.assertEqual(len(seeds), 2)
            for seed in seeds:
                self.assertNotIn("history", seed["status"])
                self.assertIn("not temporal history", seed["description"])
            for item in mode["initializers"]["items"]:
                if item["name"].startswith("n0024_sp_scale"):
                    self.assertEqual(item["category"], "configuration")

            debug = onnx.load(os.path.join(out, "debug", "mode3.onnx"))
            self.assertEqual(len(debug.graph.output), 42)
            self.assertTrue(os.path.isfile(
                os.path.join(out, "debug", "mode3.outputs.json")))

            files = []
            for root, _dirs, names in os.walk(out):
                for name in names:
                    files.append(os.path.join(root, name))
            self.assertFalse([name for name in files
                              if name.endswith((".bin", ".raw", ".npy"))])

    def test_cli_rejects_existing_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "keep.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write("keep")
            status = export_sdk.main(
                ["--runs", self.run_dir, "--out", out])
            self.assertEqual(status, 1)


class CaptureMode4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = _capture_run("XEFG_SDK_MODE4_RUN")
        cls.capture = export_sdk.build_capture(cls.run_dir, 1)
        cls.rewrite = export_sdk.rewrite_graph(
            cls.capture.model, cls.capture.feeds, cls.capture.mode,
            want_debug=True)

    def _application_input(self, name):
        height, width = self.capture.output_size
        path = os.path.join(self.run_dir, "application_inputs", name)
        return np.fromfile(path, dtype=np.uint8).reshape(height, width, 4)

    def test_capture_contract(self):
        self.assertEqual(self.capture.mode, 4)
        self.assertEqual(self.capture.output_size, (360, 640))
        self.assertEqual(self.capture.motion_size, (180, 320))
        self.assertTrue(self.capture.has_backbuffer)
        self.assertFalse(self.capture.has_ui)
        # Mode 4 consumes t8/t10/t11/t12.  t9 aliases the backbuffer but is
        # not consumed by the mode 4 graph, so it is not a cut input.
        self.assertIn("n0038_in_t8", self.capture.feeds)
        self.assertIn("n0038_in_t10", self.capture.feeds)
        self.assertNotIn("n0038_in_t9", self.capture.feeds)
        self.assertNotIn("n0038_in_t9", self.capture.assembler.cut_inputs)
        self.assertEqual(self.capture.feeds["n0025_x5"].size,
                         export_sdk.LUT_SIZE)

    def test_current_backbuffer_not_previous(self):
        current = self.capture.feeds["n0038_in_t10"]
        np.testing.assert_array_equal(
            current, self._application_input("backbuffer_00001.bgra"))
        np.testing.assert_array_equal(
            self.capture.feeds["n0038_in_t8"],
            self._application_input("color_00001.bgra"))

        second = export_sdk.build_capture(self.run_dir, 2)
        np.testing.assert_array_equal(
            second.feeds["n0038_in_t10"],
            self._application_input("backbuffer_00002.bgra"))
        np.testing.assert_array_equal(
            second.feeds["n0038_in_t8"],
            self._application_input("color_00002.bgra"))
        self.assertFalse(np.array_equal(current, second.feeds["n0038_in_t10"]))

    def test_public_model_names_and_contract(self):
        model = self.rewrite.public_model
        names = [value.name for value in model.graph.input]
        self.assertEqual(names[:5], ["previous_color", "current_color",
                                     "motion_vectors", "depth", "backbuffer"])
        self.assertNotIn("ui", names)
        for alias in ("n0038_in_t8", "n0038_in_t10", "n0038_in_t11",
                      "n0038_in_t12"):
            self.assertNotIn(alias, names)
        self.assertNotIn(export_sdk.LUT_INPUT, names)
        self.assertEqual([value.name for value in model.graph.output],
                         ["generated_frame"])
        self.assertEqual(len(model.graph.input), 312)
        onnx.checker.check_model(model)
        onnx.checker.check_model(self.rewrite.debug_model)
        lut = next(tensor for tensor in model.graph.initializer
                   if tensor.name == export_sdk.LUT_INPUT)
        self.assertTrue(np.array_equal(
            numpy_helper.to_array(lut),
            self.capture.feeds[export_sdk.LUT_INPUT]))

    def test_profile_feeds_are_metadata_only(self):
        profile = export_sdk.collect_profile_feeds(
            self.capture.feeds, self.rewrite.info["removed_inputs"])
        self.assertEqual(len(profile), 307)
        self.assertLessEqual(
            max(len(record["values"]) for record in profile.values()), 8)
        for name in ("previous_color", "current_color", "motion_vectors",
                     "depth", "backbuffer", "n0038_in_t8", "n0038_in_t10"):
            self.assertNotIn(name, profile)

    def test_io_contract_slots_and_frame_offsets(self):
        contract = export_sdk._io_contract(
            [self.capture], {self.capture.run_dir: self.rewrite},
            {4: [export_sdk._build_profile(self.capture, self.rewrite)]})
        mode = contract["variants"]["mode4"]
        self.assertEqual(mode["graph"]["original_inputs"], 320)
        self.assertEqual(mode["graph"]["packaged_inputs"], 312)
        self.assertEqual(mode["graph"]["original_outputs"], 42)
        self.assertEqual(len(mode["inputs"]), 320)
        by_name = {entry["graph_name"]: entry for entry in mode["inputs"]}
        expected = {
            "n0038_in_t8": ("t8", "current_color", 0),
            "n0038_in_t10": ("t10", "backbuffer", 0),
            "n0038_in_t11": ("t11", "previous_color", -1),
            "n0038_in_t12": ("t12", "current_color", 0),
        }
        for name, (slot, semantic, offset) in expected.items():
            self.assertEqual(by_name[name]["slot"], slot, name)
            self.assertEqual(by_name[name]["semantic"], semantic, name)
            self.assertEqual(by_name[name]["frame_offset"], offset, name)
        self.assertEqual(mode["binding"]["backbuffer"]["aliases"],
                         ["n0038_in_t10"])
        self.assertEqual(
            [entry["name"] for entry in mode["public_inputs"]],
            ["previous_color", "current_color", "motion_vectors", "depth",
             "backbuffer"])

    def test_mode4_package(self):
        with tempfile.TemporaryDirectory() as out:
            manifest = export_sdk.export_package([self.run_dir], out)
            self.assertEqual(set(manifest["variants"]), {"mode4"})
            variant = manifest["variants"]["mode4"]
            self.assertEqual(variant["mode"], 4)
            self.assertEqual(variant["output"], "generated_frame")
            self.assertEqual(variant["inputs"], [
                "previous_color", "current_color", "motion_vectors", "depth",
                "backbuffer"])
            self.assertEqual(len(variant["profiles"]), 1)
            model_path = os.path.join(out, variant["model"])
            self.assertEqual(export_sdk._file_sha256(model_path),
                             variant["model_sha256"])

            profile = _read_json(os.path.join(out, variant["profiles"][0]))
            self.assertEqual(profile["schema_version"], 1)
            self.assertEqual(profile["variant"], "mode4")
            self.assertEqual(profile["output_size"], [640, 360])
            self.assertEqual(profile["motion_size"], [320, 180])
            self.assertEqual(len(profile["feeds"]), 307)
            self.assertTrue(profile["configuration"]["ui_enabled"])
            for name in ("n0038_in_t8", "n0038_in_t10", "n0038_in_t11",
                         "n0038_in_t12"):
                self.assertNotIn(name, profile["feeds"])

            contract = _read_json(os.path.join(out, "io_contract.json"))
            mode = contract["variants"]["mode4"]
            self.assertEqual(mode["graph"]["original_inputs"], 320)
            self.assertEqual(mode["graph"]["packaged_inputs"], 312)
            self.assertNotIn("ui", [entry["name"]
                                    for entry in mode["public_inputs"]])

    def test_graph_is_identical_between_frames(self):
        second = export_sdk.build_capture(self.run_dir, 2)
        second_rewrite = export_sdk.rewrite_graph(
            second.model, second.feeds, second.mode)
        self.assertEqual(
            self.rewrite.public_model.SerializeToString(),
            second_rewrite.public_model.SerializeToString())
        self.assertEqual(self.rewrite.info["initializer_digest"],
                         second_rewrite.info["initializer_digest"])


class CaptureMode4AnimatedTests(unittest.TestCase):
    """The animated-UI capture proves current backbuffer/UI frame ownership."""

    @classmethod
    def setUpClass(cls):
        cls.run_dir = _capture_run("XEFG_SDK_MODE4_ANIMATED_RUN")
        cls.first = export_sdk.build_capture(cls.run_dir, 1)
        cls.second = export_sdk.build_capture(cls.run_dir, 2)

    def _application_input(self, name):
        height, width = self.first.output_size
        path = os.path.join(self.run_dir, "application_inputs", name)
        return np.fromfile(path, dtype=np.uint8).reshape(height, width, 4)

    def test_animated_ui_uses_current_frame_backbuffer(self):
        for capture, frame in ((self.first, 1), (self.second, 2)):
            self.assertEqual(capture.mode, 4)
            self.assertNotIn("n0038_in_t9", capture.feeds)
            np.testing.assert_array_equal(
                capture.feeds["n0038_in_t8"],
                self._application_input("color_%05d.bgra" % frame))
            np.testing.assert_array_equal(
                capture.feeds["n0038_in_t10"],
                self._application_input("backbuffer_%05d.bgra" % frame))
        self.assertFalse(np.array_equal(self.first.feeds["n0038_in_t10"],
                                        self.second.feeds["n0038_in_t10"]))


class CaptureSizeEqualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first_run = _capture_run("XEFG_SDK_MODE3_RUN")
        cls.second_run = _capture_run("XEFG_SDK_MODE3_SIZE_RUN")

    def test_graph_is_identical_across_sizes(self):
        first = export_sdk.build_capture(self.first_run, 1)
        second = export_sdk.build_capture(self.second_run, 1)
        self.assertEqual(first.mode, 3)
        self.assertEqual(second.mode, 3)
        left = export_sdk.rewrite_graph(
            first.model, first.feeds, first.mode)
        right = export_sdk.rewrite_graph(
            second.model, second.feeds, second.mode)
        self.assertEqual(left.public_model.SerializeToString(),
                         right.public_model.SerializeToString())
        self.assertEqual(left.info["initializer_digest"],
                         right.info["initializer_digest"])


class CaptureMode1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = _capture_run("XEFG_SDK_MODE1_RUN")
        configuration = _read_json(os.path.join(cls.run_dir, "trace", "meta.json"))["configuration"]
        cls.output_size = (configuration["height"], configuration["width"])
        cls.motion_size = (configuration["motion_height"], configuration["motion_width"])
        cls.capture = export_sdk.build_capture(cls.run_dir, 1)
        cls.rewrite = export_sdk.rewrite_graph(
            cls.capture.model, cls.capture.feeds, cls.capture.mode)

    def test_mode1_contract(self):
        self.assertEqual(self.capture.mode, 1)
        self.assertEqual(self.capture.output_size, self.output_size)
        self.assertEqual(self.capture.motion_size, self.motion_size)
        self.assertEqual(self.output_size, tuple(value * 2 for value in self.motion_size))
        names = [value.name for value in self.rewrite.public_model.graph.input]
        self.assertEqual(names[:4], ["previous_color", "current_color",
                                     "motion_vectors", "depth"])
        self.assertNotIn("ui", names)
        self.assertEqual(len(names), 311)
        onnx.checker.check_model(self.rewrite.public_model)

    def test_mode1_package(self):
        with tempfile.TemporaryDirectory() as out:
            manifest = export_sdk.export_package([self.run_dir], out)
            self.assertEqual(set(manifest["variants"]), {"mode1"})
            variant = manifest["variants"]["mode1"]
            self.assertEqual(variant["inputs"], [
                "previous_color", "current_color", "motion_vectors",
                "depth"])
            profile = _read_json(
                os.path.join(out, variant["profiles"][0]))
            self.assertEqual(profile["output_size"], list(reversed(self.output_size)))
            self.assertEqual(profile["motion_size"], list(reversed(self.motion_size)))
            self.assertEqual(len(profile["feeds"]), 307)
            configuration = profile["configuration"]
            for removed in ("view_matrix", "projection_matrix",
                            "motion_vector_scale", "jitter"):
                self.assertNotIn(removed, configuration)
            self.assertNotIn("identity", json.dumps(configuration))
            self.assertEqual(configuration["input_contract"], "normalized")
            self.assertFalse(configuration["ui_enabled"])
            self.assertEqual(configuration["motion_vector_resolution"], "half")
            contract = _read_json(
                os.path.join(out, "io_contract.json"))
            mode = contract["variants"]["mode1"]
            self.assertEqual(mode["graph"]["original_inputs"], 318)
            self.assertEqual(mode["graph"]["original_outputs"], 42)
            self.assertEqual(len(mode["inputs"]), 318)
            self.assertEqual(mode["graph"]["packaged_inputs"], 311)
            by_name = {entry["graph_name"]: entry
                       for entry in mode["inputs"]}
            self.assertEqual(by_name["n0025_x1"]["slot"], "t2")
            self.assertEqual(by_name["n0038_in_t11"]["slot"], "t11")
            self.assertNotIn("n0038_in_t9", by_name)


if __name__ == "__main__":
    unittest.main()
