"""Rounding parity for the motion-vector splat.

The retained DXIL uses two distinct coordinate rules:

* ``src == dst``: ``tid + floor(disp + 0.5)``.  This is ``Round_ni`` with ties
  toward +infinity (``floor`` after the float32 add).
* ``src != dst``: ``Round_ne((tid + 0.5 + disp) * dst / src)`` with no
  half-texel subtraction and no extra ``tid`` add.

These tests build tiny synthetic motion/depth textures, run them through the
three production lowerings and compare against an independent NumPy replay:

* ``recon.splat_plan`` (plus ``reconstruct_splat``'s UMax scatter)
* ``export_onnx.build_splat`` executed with ONNX Runtime
* ``dynamic_index.build_dynamic_splat`` executed with ONNX Runtime

The graph coordinate intermediates ``sp_dst_x``/``sp_dst_y`` are exposed as
extra outputs on the models returned by the builders so deviations are visible
at the rounding step instead of only through the packed bytes.
"""
import os
import struct
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper

from xefg_sdk.export import dynamic_index, export_onnx, recon

# Every displacement exercised on each axis of the covering cases.
ALL_VALUES = [-1.5, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.5]

# Exactly one float32 ulp either side of 1.0.  Multiplying 0.5 by these keeps
# the product exactly representable (0.5 +- 2**-24), i.e. exactly one ulp
# either side of the 0.5 tie.
ABOVE_ONE = float(np.nextafter(np.float32(1.0), np.float32(np.inf)))
BELOW_ONE = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
# Three ulps below 1.0.  With smaller steps 0.5 * scale + 0.5 rounds back up
# to 1.0: the sum sits within half an ulp of 1.0.  At 1 - 3*2**-24 the product
# is 0.5 - 2**-23 and the sum is 1 - 2**-23, so floor still sees below 1.
BELOW_THREE = float(np.nextafter(
    np.nextafter(np.float32(BELOW_ONE), np.float32(0.0)), np.float32(0.0)))

# name, src_w, src_h, dst_w, dst_h, scale_x, scale_y, x values, y values
CASES = [
    ("same_8x8", 8, 8, 8, 8, 1.0, 1.0, ALL_VALUES, ALL_VALUES),
    ("same_8x8_ulp", 8, 8, 8, 8, ABOVE_ONE, BELOW_ONE,
     ALL_VALUES, ALL_VALUES),
    ("same_4x3", 4, 3, 4, 3, 1.0, 1.0,
     ALL_VALUES[:4], ALL_VALUES[:3]),
    ("scale_4x3_to_8x6", 4, 3, 8, 6, 1.0, 1.0,
     ALL_VALUES[:4], ALL_VALUES[:3]),
    # Non-integer ratio on both axes: 6/4 and 8/3.
    ("scale_4x3_to_6x8", 4, 3, 6, 8, 1.0, 1.0,
     ALL_VALUES[:4], ALL_VALUES[:3]),
    ("scale_8x8_to_12x12_ulp", 8, 8, 12, 12, ABOVE_ONE, BELOW_ONE,
     ALL_VALUES, ALL_VALUES),
    # (0.5 + 0.5) * 1.5 and (1.5 - 0.5) * 1.5 are exact .5 ties; Round_ne
    # takes them to 2, while the old subtract-0.5/ceil form produced 1.
    ("tie_4x2_to_6x3", 4, 2, 6, 3, 1.0, 1.0,
     [0.5, -0.5, 0.25, -0.25], [0.5, -0.5]),
]


def _scale_bits(value):
    """The root-constant bit pattern for a float32 scale."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _motion(x_values, y_values, src_w, src_h):
    xv = np.asarray(x_values, dtype=np.float32).reshape(1, -1)
    yv = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    mx = np.broadcast_to(xv, (src_h, src_w))
    my = np.broadcast_to(yv, (src_h, src_w))
    return np.stack([mx, my], axis=2).astype(np.float16)


def _materialize(case):
    (name, src_w, src_h, dst_w, dst_h, scale_x, scale_y,
     x_values, y_values) = case
    motion = _motion(x_values, y_values, src_w, src_h)
    depth = np.linspace(0.05, 0.95, src_w * src_h,
                        dtype=np.float32).reshape(src_h, src_w)
    disp_x = motion[..., 0].astype(np.float32) * np.float32(scale_x)
    disp_y = motion[..., 1].astype(np.float32) * np.float32(scale_y)
    return {
        "name": name,
        "src_w": src_w, "src_h": src_h, "dst_w": dst_w, "dst_h": dst_h,
        "scale_x": np.float32(scale_x), "scale_y": np.float32(scale_y),
        "motion": motion, "depth": depth,
        "disp_x": disp_x, "disp_y": disp_y,
    }


# --------------------------------------------------------------------------
# Independent NumPy rules and scatter
# --------------------------------------------------------------------------

def expected_coords(disp_x, disp_y, src_w, src_h, dst_w, dst_h):
    tid_x = np.arange(src_w, dtype=np.float32)[None, :]
    tid_y = np.arange(src_h, dtype=np.float32)[:, None]
    if src_w == dst_w and src_h == dst_h:
        dst_x = np.arange(src_w, dtype=np.int64)[None, :] + np.floor(
            disp_x + np.float32(0.5)).astype(np.int64)
        dst_y = np.arange(src_h, dtype=np.int64)[:, None] + np.floor(
            disp_y + np.float32(0.5)).astype(np.int64)
    else:
        dst_x = np.rint(
            (tid_x + np.float32(0.5) + disp_x)
            * np.float32(dst_w) / np.float32(src_w)).astype(np.int64)
        dst_y = np.rint(
            (tid_y + np.float32(0.5) + disp_y)
            * np.float32(dst_h) / np.float32(src_h)).astype(np.int64)
    return dst_x, dst_y


def _depth_key(depth):
    value = np.float32(1.0) - np.ascontiguousarray(depth, dtype=np.float32)
    value = np.trunc(np.float32(value * np.float32(32767.0)))
    value = np.clip(value, 0.0, 32767.0).astype(np.uint32)
    return value << np.uint32(16)


def _half_bits(value):
    cast = np.ascontiguousarray(value, dtype=np.float32).astype(np.float16)
    return cast.view(np.uint16)


def expected_keys(disp_x, disp_y, depth):
    depth_key = _depth_key(depth)
    high = np.uint32(0x80000000)
    key_x = (high | depth_key | _half_bits(disp_x)).astype(np.uint32)
    key_y = (high | depth_key | _half_bits(disp_y)).astype(np.uint32)
    return key_x, key_y


def expected_scatter(dst_x, dst_y, key_x, key_y, dst_w, dst_h,
                     allocation_words):
    plane = dst_w * dst_h
    got = np.zeros(allocation_words, dtype=np.uint32)
    for ox in (0, 1):
        for oy in (0, 1):
            xx = dst_x + ox
            yy = dst_y + oy
            ok = (xx >= 0) & (xx < dst_w) & (yy >= 0) & (yy < dst_h)
            np.maximum.at(got, yy[ok] * dst_w + xx[ok], key_x[ok])
            np.maximum.at(got, plane + yy[ok] * dst_w + xx[ok], key_y[ok])
    return got


# --------------------------------------------------------------------------
# Synthetic capture for recon.splat_plan
# --------------------------------------------------------------------------

_DISASM = "\n".join([
    "; synthetic motion-vector splat",
    "atomicrmw umax i32 addrspace(3)",
    "call i32 @dx.op.atomicBinOp.i32(...) ; i32 7",
    "; root constants <2 x i32>, <2 x i32>, <2 x float>, <2 x i32>, "
    "<2 x i32>",
]) + "\n"


def write_capture(root, case_data, allocation):
    """Write the disassembly, tensor dumps and captured allocation."""
    trace = os.path.join(root, "trace")
    os.makedirs(os.path.join(trace, "disasm"))
    os.makedirs(os.path.join(trace, "tensors"))
    sha = "a" * 64
    with open(os.path.join(trace, "disasm", "d24-%s.ll" % sha[:16]),
              "w", encoding="utf-8") as handle:
        handle.write(_DISASM)

    motion = case_data["motion"]
    depth = case_data["depth"]
    motion.tofile(os.path.join(trace, "tensors", "motion.bin"))
    depth.astype(np.float32).tofile(os.path.join(trace, "tensors", "depth.bin"))
    allocation.view(np.uint8).tofile(
        os.path.join(trace, "tensors", "u0.bin"))

    src_w, src_h = case_data["src_w"], case_data["src_h"]
    dispatch = {
        "dispatch": 24,
        "key": "d24",
        "sha256": sha,
        "constants": [{"values": [
            src_w, src_h, case_data["dst_w"], case_data["dst_h"],
            _scale_bits(case_data["scale_x"]),
            _scale_bits(case_data["scale_y"]),
            0, 0, 0, 0,
        ]}],
        "resources": [
            {"slot": "t0", "file": "motion.bin", "dimension": "texture2d",
             "format": 34, "width": src_w, "height": src_h, "array": 0},
            {"slot": "t1", "file": "depth.bin", "dimension": "texture2d",
             "format": 41, "width": src_w, "height": src_h, "array": 0},
            {"slot": "u0", "file": "u0.bin", "dimension": "buffer",
             "array": 0},
        ],
    }
    return dispatch


def _coords_and_scatter(case_data):
    dst_x, dst_y = expected_coords(
        case_data["disp_x"], case_data["disp_y"], case_data["src_w"],
        case_data["src_h"], case_data["dst_w"], case_data["dst_h"])
    key_x, key_y = expected_keys(
        case_data["disp_x"], case_data["disp_y"], case_data["depth"])
    plane = case_data["dst_w"] * case_data["dst_h"]
    got = expected_scatter(dst_x, dst_y, key_x, key_y, case_data["dst_w"],
                           case_data["dst_h"], 2 * plane)
    return dst_x, dst_y, key_x, key_y, got


def _expose_coords(model, shape):
    model.graph.output.extend([
        helper.make_tensor_value_info("sp_dst_x", TensorProto.INT64, shape),
        helper.make_tensor_value_info("sp_dst_y", TensorProto.INT64, shape),
    ])
    return model


def _run(model, feeds):
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


class SplatRoundingTest(unittest.TestCase):
    def test_recon_plan_matches_independent_numpy(self):
        for case in CASES:
            with self.subTest(case=case[0]):
                data = _materialize(case)
                dst_x, dst_y, key_x, key_y, got = _coords_and_scatter(data)
                with tempfile.TemporaryDirectory() as root:
                    dispatch = write_capture(root, data, got)
                    plan, error = recon.splat_plan(root, dispatch)
                    self.assertIsNone(error)
                    np.testing.assert_array_equal(plan["dst_x"], dst_x)
                    np.testing.assert_array_equal(plan["dst_y"], dst_y)
                    np.testing.assert_array_equal(plan["key_x"], key_x)
                    np.testing.assert_array_equal(plan["key_y"], key_y)
                    capture = {}
                    result, error = recon.reconstruct_splat(
                        root, dispatch, capture=capture)
                    self.assertIsNone(error)
                    self.assertEqual(result[0], 0)
                    np.testing.assert_array_equal(
                        capture["u0"].view(np.uint32), got)

    def test_manual_anchors(self):
        """Hand-computed offsets pin the tie rules independently."""
        # Same size, exact scale: tid + floor(disp + 0.5).
        data = _materialize(("same", 8, 1, 8, 1, 1.0, 1.0,
                             ALL_VALUES, [0.0]))
        offsets = [-1, -1, 0, 0, 0, 1, 1, 2]
        dst_x, _ = expected_coords(data["disp_x"], data["disp_y"], 8, 1, 8, 1)
        np.testing.assert_array_equal(dst_x[0], np.arange(8) + offsets)

        # Same size, one-ulp scales: the negative half-integer tie goes to
        # floor(-1e-7) == -1 above.  Below, 0.5 * (1 - 2**-24) + 0.5 is
        # 1 - 2**-25, which float32 rounds back up to 1.0, so the result is
        # still 6; three ulps below survives as floor(1 - 2**-23) == 0.
        data = _materialize(("ulp", 8, 1, 8, 1, ABOVE_ONE, BELOW_ONE,
                             ALL_VALUES, [0.0]))
        dst_x, _ = expected_coords(data["disp_x"], data["disp_y"], 8, 1, 8, 1)
        self.assertEqual(dst_x[0, 2], 1)   # -0.5 * above_one
        self.assertEqual(dst_x[0, 5], 6)   # +0.5 * above_one
        data = _materialize(("ulp_roundback", 8, 1, 8, 1,
                             BELOW_ONE, ABOVE_ONE, ALL_VALUES, [0.0]))
        dst_x, _ = expected_coords(data["disp_x"], data["disp_y"], 8, 1, 8, 1)
        self.assertEqual(dst_x[0, 5], 6)   # float32 add rounds back to 1.0
        data = _materialize(("ulp_down", 8, 1, 8, 1, BELOW_THREE, ABOVE_ONE,
                             ALL_VALUES, [0.0]))
        dst_x, _ = expected_coords(data["disp_x"], data["disp_y"], 8, 1, 8, 1)
        self.assertEqual(dst_x[0, 2], 2)   # -0.5 * below_three
        self.assertEqual(dst_x[0, 5], 5)   # +0.5 * below_three

        # Rescaled 4 -> 6 (ratio 1.5): exact .5 ties round to even and no
        # half-texel subtraction or extra tid add is applied.
        data = _materialize(("tie", 4, 2, 6, 3, 1.0, 1.0,
                             [0.5, -0.5, 0.25, -0.25], [0.5, -0.5]))
        dst_x, dst_y = expected_coords(
            data["disp_x"], data["disp_y"], 4, 2, 6, 3)
        np.testing.assert_array_equal(dst_x[0], [2, 2, 4, 5])
        np.testing.assert_array_equal(dst_y[:, 0], [2, 2])

    def test_static_builder_matches_recon(self):
        self._check_builder("static")

    def test_dynamic_builder_matches_recon(self):
        self._check_builder("dynamic")

    def _check_builder(self, kind):
        for case in CASES:
            with self.subTest(case=case[0], builder=kind):
                data = _materialize(case)
                dst_x, dst_y, _key_x, _key_y, got = _coords_and_scatter(data)
                with tempfile.TemporaryDirectory() as root:
                    dispatch = write_capture(root, data, got)
                    plan, error = recon.splat_plan(root, dispatch)
                    self.assertIsNone(error, error)
                    if kind == "static":
                        model = export_onnx.build_splat(plan)
                        coords_shape = [data["src_h"], data["src_w"]]
                        feeds = {"x0": plan["motion_tex"],
                                 "x1": plan["depth_tex"]}
                    else:
                        model = dynamic_index.build_dynamic_splat(plan)
                        coords_shape = ["TH", "TW"]
                        feeds = {
                            "x0": plan["motion_tex"],
                            "x1": plan["depth_tex"],
                            "src_w": np.asarray([plan["src_w"]], np.int64),
                            "src_h": np.asarray([plan["src_h"]], np.int64),
                            "dst_w": np.asarray([plan["dst_w"]], np.int64),
                            "dst_h": np.asarray([plan["dst_h"]], np.int64),
                            "alloc_words": np.asarray(
                                [plan["allocation_words"]], np.int64),
                            "sp_motion_off_x": np.asarray(
                                [plan["offset_x"]], np.int64),
                            "sp_motion_off_y": np.asarray(
                                [plan["offset_y"]], np.int64),
                            "sp_depth_off_x": np.asarray(
                                [plan["depth_offset_x"]], np.int64),
                            "sp_depth_off_y": np.asarray(
                                [plan["depth_offset_y"]], np.int64),
                        }
                    _expose_coords(model, coords_shape)
                    onnx.checker.check_model(model)
                    outputs = _run(model, feeds)
                    result = {info.name: value
                              for info, value in zip(model.graph.output,
                                                     outputs)}
                    np.testing.assert_array_equal(
                        result["sp_dst_x"], dst_x, err_msg=case[0])
                    np.testing.assert_array_equal(
                        result["sp_dst_y"], dst_y, err_msg=case[0])
                    self.assertEqual(result["u0"].tobytes(),
                                     got.view(np.uint8).tobytes(), case[0])


if __name__ == "__main__":
    unittest.main()
