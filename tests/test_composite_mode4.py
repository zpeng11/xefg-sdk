"""Regression tests for the composite distance accumulator (modes 4 and 5).

The mode 4/5 branch sums three squared channel differences before the
``sqrt``.  The first term is ``cp_diff2_0``; the original builder then emitted
an ``Add`` for channel 1 that read a nonexistent ``cp_dsum_0``, so
``onnx.checker`` rejected the graph.  These tests build tiny synthetic plans
instead of using a capture, verify the graph wiring, and run the static and
dynamic exporters through ONNX Runtime against an independent NumPy replay.
"""
import os
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper

from xefg_sdk.export import dynamic_tail, export_onnx


def _mode_plan(mode, height=4, width=6, seed=7):
    """A tiny plan with the shape and dtypes ``composite_plan`` produces."""
    rng = np.random.default_rng(seed)
    feature_h, feature_w = height // 2, width // 2
    feature = rng.integers(0, 256, size=(feature_h, feature_w, 16),
                           dtype=np.uint8)
    weights = rng.integers(-128, 128, size=(20, 16),
                           dtype=np.int16).astype(np.int8)
    scales = rng.uniform(-4e-5, 4e-5, size=20).astype(np.float32)
    biases = rng.uniform(-1.0, 1.0, size=20).astype(np.float32)

    ys, xs = np.mgrid[0:height, 0:width]
    parity = ((xs & 1) | ((ys & 1) << 1)).reshape(-1)
    select = np.stack([np.full(height * width, 4 * gate, np.int64) + parity
                       for gate in range(5)], axis=1)

    overlays = {
        slot: rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)
        for slot in ("t1", "t2", "t3", "t11", "t12")
    }
    # Keep t8/t10 close so the distance gate is not saturated and the
    # mode-specific blend actually contributes to the output.
    base = rng.integers(80, 176, size=(height, width, 4), dtype=np.uint8)
    overlays["t8"] = np.clip(
        base.astype(np.int16) + rng.integers(-3, 4, size=base.shape),
        0, 255).astype(np.uint8)
    overlays["t10"] = np.clip(
        base.astype(np.int16) + rng.integers(-3, 4, size=base.shape),
        0, 255).astype(np.uint8)
    overlays["t9"] = rng.integers(0, 256, size=(height, width, 4),
                                  dtype=np.uint8)

    needed = ["t1", "t2", "t3", "t11", "t12", "t8", "t10"]
    if mode == 5:
        needed.append("t9")
    return {
        "mode": mode,
        "width": width,
        "height": height,
        "feature_w": feature_w,
        "feature_h": feature_h,
        "feature": feature,
        "weights": weights,
        "scales": scales,
        "biases": biases,
        "select": select,
        "overlays": {slot: overlays[slot] for slot in needed},
        "dispatch": 38,
    }


def _fma(a, b, c):
    """One-rounding float32 FMA, matching the graph's float64 emulation."""
    return (a.astype(np.float64) * b.astype(np.float64)
            + c.astype(np.float64)).astype(np.float32)


def _unorm8(value):
    """The texture unit's float -> UNORM8 conversion, computed exactly."""
    clipped = np.clip(value, np.float32(0), np.float32(1))
    return np.floor(clipped.astype(np.float64) * 255.0 + 0.5).astype(np.uint8)


def _rgba(plan, slot):
    value = plan["overlays"][slot].astype(np.float32) / np.float32(255.0)
    return value[..., 2], value[..., 1], value[..., 0], value[..., 3]


def _reference(plan, mode, mode4_blend="m"):
    """Independent NumPy replay of the mode 4/5 composite math.

    The d38 mode-4 branch blends against the final full `m` (IR `%223`), not
    the earlier `m2` (IR `%205`).  The parameter lets the regression test prove
    those two intermediates produce different bytes.
    """
    height, width = plan["height"], plan["width"]

    # MatMulInteger reinterprets the uint8 feature bytes as signed int8.
    feature = plan["feature"].view(np.int8).astype(np.int32)
    feature = np.repeat(np.repeat(feature, 2, axis=0), 2, axis=1)
    acc = feature.reshape(-1, 16) @ plan["weights"].T.astype(np.int32)
    select = plan["select"]
    acc_sel = np.take_along_axis(acc, select, axis=1)
    scale = plan["scales"][select]
    bias = plan["biases"][select]
    v = acc_sel.astype(np.float32) * scale + bias
    with np.errstate(over="ignore"):
        gates = np.float32(1.0) / (np.float32(1.0) + np.exp(-v))
    gates = gates.reshape(height, width, 5)
    s0, s1, s2, s3, s4 = (gates[..., index] for index in range(5))

    (ar, ag, ab, aa), (br, bg, bb, ba) = _rgba(plan, "t1"), _rgba(plan, "t2")
    (cr, cg, cb, ca), (dr, dg, db, _) = _rgba(plan, "t3"), _rgba(plan, "t11")
    er, eg, eb, _ = _rgba(plan, "t12")

    m0 = [_fma(ac - bc, s2, bc)
          for ac, bc in ((ar, br), (ag, bg), (ab, bb))]
    m1 = [_fma(channel - alpha, s0, alpha - m0[index])
          for index, (channel, alpha) in enumerate(
              zip((cr, cg, cb), (aa, ba, ca)))]
    m2 = [_fma(m1[index], s1, m0[index]) for index in range(3)]
    m3 = [_fma(dc - ec, s3, ec)
          for dc, ec in ((dr, er), (dg, eg), (db, eb))]
    _m = [_fma(m2[index] - m3[index], s4, m3[index])
          for index in range(3)]

    t8 = _rgba(plan, "t8")
    t10 = _rgba(plan, "t10")
    if mode == 4:
        prior = list(t8[:3])
    else:
        q = _rgba(plan, "t9")
        prior = [q[index] + t8[index] * (np.float32(1.0) - q[3])
                 for index in range(3)]

    diff = [prior[index] - t10[index] for index in range(3)]
    distance = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1]
                       + diff[2] * diff[2])
    g = np.clip(distance * np.float32(20.0),
                np.float32(0.0), np.float32(1.0))
    w = g * g * (np.float32(3.0) - np.float32(2.0) * g)
    keep = np.float32(1.0) - w

    if mode == 4:
        blend = _m if mode4_blend == "m" else m2
        out = [t10[index] + (blend[index] - t8[index]) * keep
               for index in range(3)]
    else:
        q = _rgba(plan, "t9")
        q2 = [q[index] + m2[index] * (np.float32(1.0) - q[3])
              for index in range(3)]
        out = [t10[index] - keep * prior[index] + keep * q2[index]
               for index in range(3)]

    return np.stack([
        _unorm8(out[2]), _unorm8(out[1]), _unorm8(out[0]),
        np.full((height, width), 255, dtype=np.uint8),
    ], axis=-1)


def _node_by_output(model, name):
    for node in model.graph.node:
        if name in node.output:
            return node
    raise AssertionError("no node produces %s" % name)


class CompositeMode4Tests(unittest.TestCase):
    def _assert_resolved_and_summed(self, model):
        """Every node input resolves; the three squared diffs are summed."""
        available = {value.name for value in model.graph.input}
        available.update(tensor.name for tensor in model.graph.initializer)
        for node in model.graph.node:
            for name in node.input:
                self.assertIn(name, available,
                              "%s reads missing input %s"
                              % (node.op_type, name))
            available.update(node.output)

        self.assertNotIn("cp_dsum_0", available)
        first = _node_by_output(model, "cp_dsum_1")
        second = _node_by_output(model, "cp_dsum_2")
        self.assertEqual(first.op_type, "Add")
        self.assertEqual(list(first.input), ["cp_diff2_0", "cp_diff2_1"])
        self.assertEqual(list(second.input), ["cp_dsum_1", "cp_diff2_2"])

    def _run(self, model, plan):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "composite.onnx")
            onnx.save(model, path)
            feeds = {"x0": plan["feature"]}
            feeds.update({"in_" + slot: data
                          for slot, data in plan["overlays"].items()})
            session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"])
            return session.run(["u0"], feeds)[0]

    def test_mode4_builders_check_and_wire_the_existing_accumulator(self):
        plan = _mode_plan(4)
        for name, build in (
                ("static", lambda: export_onnx.build_composite(plan)),
                ("dynamic", lambda: dynamic_tail.build_dynamic_composite(plan))):
            with self.subTest(builder=name):
                model = build()
                onnx.checker.check_model(model)
                self._assert_resolved_and_summed(model)

    def test_mode4_models_match_the_reference_math(self):
        plan = _mode_plan(4)
        expected = _reference(plan, 4)
        self.assertFalse(
            np.array_equal(_reference(plan, 4, mode4_blend="m2"), expected),
            "synthetic plan must distinguish final m from m2")
        for name, build in (
                ("static", lambda: export_onnx.build_composite(plan)),
                ("dynamic", lambda: dynamic_tail.build_dynamic_composite(plan))):
            with self.subTest(builder=name):
                got = self._run(build(), plan)
                np.testing.assert_array_equal(got, expected)

    def test_mode5_models_check_and_match_the_reference_math(self):
        # Mode 5 shares the accumulator and the sqrt/gate chain with mode 4.
        plan = _mode_plan(5)
        expected = _reference(plan, 5)
        for name, build in (
                ("static", lambda: export_onnx.build_composite(plan)),
                ("dynamic", lambda: dynamic_tail.build_dynamic_composite(plan))):
            with self.subTest(builder=name):
                model = build()
                onnx.checker.check_model(model)
                self._assert_resolved_and_summed(model)
                got = self._run(model, plan)
                np.testing.assert_array_equal(got, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
