"""Parity harness for the warp-init bilinear sampler.

The warp-init sampler is implemented three times: the reference kernel
``recon._warp_init_sample``, the static lowering
``export_onnx._warp_init_sample_graph`` and the dynamic lowering
``dynamic_warp_init._sample_graph``.  These tests do not carry hardware
samples or hard-coded filtered values.  Every case feeds the same BGRA8
texture and float32 positions to the reference kernel and to minimal isolated
ONNX graphs built by the two lowerings, then requires the float64 results to
be bit-identical.  A production change to any of the three implementations is
therefore picked up on the next run without editing this file.

The positions cover the whole clamp domain: constant coordinates, texel
centres, interior fractional offsets, half-weight-step subtexel ties, texel
edges and
the half-pixel clamp boundary at zero.  Negative coordinates (outside the
clamp domain) are deliberately not exercised.
"""
import unittest

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper

from xefg_sdk.export import dynamic_warp_init, export_onnx, recon

# Tiny supported resolutions, including a 5x5 texture and non-square cases.
# Every coordinate array is at most a few dozen entries, so no fixture or
# large texture is needed.
RESOLUTIONS = ((1, 1), (2, 3), (3, 5), (5, 5), (8, 7))

# Real, non-power-of-two capture resolutions used by the render validation
# path.  The largest BGRA8 texture is 3840*2160*4 = 33,177,600 bytes (<35 MB);
# the shape test allocates one at a time and drops it before the next.
REAL_RESOLUTIONS = ((640, 360), (1280, 720), (1920, 1080),
                    (2560, 1440), (3840, 2160))

# Inline hardware vectors measured on an AMD Radeon RX 6800 XT through the
# sampler probe. Only the request UV bits, the four focus-channel taps, the
# integer total and the measured hardware units/truncation are embedded, so no
# external file is read at run time. The cases cover near-integer feature
# boundaries, both half-to-even directions, and high-contrast taps.
_HW_TEXTURE = (640, 360)
_HW_VECTORS = (
    {
        "label": "near-integer 24495 ch0 (frame8/t2)",
        "source": "inline hardware vector: near-integer boundary",
        "u_bits": 1057790183,
        "v_bits": 1054308853,
        "x0": 350, "y0": 151, "x1": 351, "y1": 152,
        "channel": 0,
        "taps": (107, 107, 106, 107),
        "total": 7012348,
        "grid_units": 107.0,
        "hw_units": 107.0,
        "hw_trunc": 107,
        "old_units": 106.99993896484375,
        "old_trunc": 106,
    },
    {
        "label": "near-integer 49010 ch0 (frame8/t3)",
        "source": "inline hardware vector: near-integer boundary",
        "u_bits": 1042337654,
        "v_bits": 1062859705,
        "x0": 100, "y0": 305, "x1": 101, "y1": 306,
        "channel": 0,
        "taps": (76, 75, 76, 76),
        "total": 4980734,
        "grid_units": 76.0,
        "hw_units": 76.0,
        "hw_trunc": 76,
        "old_units": 75.99996948242188,
        "old_trunc": 75,
    },
    {
        "label": "tie-even up 5216 ch0 (frame8/t2)",
        "source": "inline hardware vector: half-to-even tie",
        "u_bits": 1050125210,
        "v_bits": 1034324833,
        "x0": 189, "y0": 28, "x1": 190, "y1": 29,
        "channel": 0,
        "taps": (40, 52, 52, 52),
        "total": 3241660,
        "grid_units": 49.4638671875,
        "hw_units": 49.4638671875,
        "hw_trunc": 49,
        "tie": "up",
    },
    {
        "label": "tie-even down 5216 ch2 (frame8/t2)",
        "source": "inline hardware vector: half-to-even tie",
        "u_bits": 1050125210,
        "v_bits": 1034324833,
        "x0": 189, "y0": 28, "x1": 190, "y1": 29,
        "channel": 2,
        "taps": (67, 87, 87, 87),
        "total": 5424612,
        "grid_units": 82.77294921875,
        "hw_units": 82.77294921875,
        "hw_trunc": 82,
        "tie": "down",
    },
    {
        "label": "high-contrast 9817 ch0 (ctl_frame2/t3)",
        "source": "inline hardware vector: high-contrast taps",
        "u_bits": 1060123443,
        "v_bits": 1043449629,
        "x0": 439, "y0": 62, "x1": 440, "y1": 63,
        "channel": 0,
        "taps": (245, 51, 245, 51),
        "total": 3342336,
        "grid_units": 51.0,
        "hw_units": 51.0,
        "hw_trunc": 51,
        "min_span": 128,
    },
)


# Half a step of the 8-bit subtexel weight.  A centre offset by this amount
# targets a frac * 256 tie in the reference texel coordinate.
TIE_STEP = np.float32(1.0 / 512.0)


def _value_info(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def _finish_model(builder, name, inputs, output_name):
    graph = helper.make_graph(
        builder.nodes,
        name,
        inputs,
        [_value_info(output_name, TensorProto.DOUBLE, [None, None, 4])],
        builder.initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _sample_inputs():
    return [
        _value_info("tex", TensorProto.UINT8, [None, 4]),
        _value_info("pos_x", TensorProto.FLOAT, [None, None]),
        _value_info("pos_y", TensorProto.FLOAT, [None, None]),
    ]


def _static_model(width, height):
    """A minimal static-sampler graph, wired exactly as build_warp_init does."""
    plan = {"ow": width, "oh": height}
    builder = export_onnx.Builder(plan)
    output = export_onnx._warp_init_sample_graph(
        builder, plan, "sample", "tex", "pos_x", "pos_y")
    return _finish_model(builder, "warp_init_sample_static",
                         _sample_inputs(), output)


def _dynamic_model(width, height):
    """A minimal dynamic-sampler graph with build_dynamic's scalar plumbing.

    Only the derived values the sampler consumes are built: the int64 extent
    feeds become the float32 reciprocals, float64 extents and int32 bounds the
    sampler references in the full dynamic model.
    """
    builder = dynamic_warp_init.Builder({"ow": width, "oh": height})
    builder.const("wi_one_f", "fp32", [1.0])
    builder.const("wi_i0", "i32", [0])
    builder.const("wi_i1", "i32", [1])
    builder.const("wi_i256", "i32", [256])
    builder.const("wi_one_i64", "i64", [1])

    builder.add("Cast", ["wi_ow"], ["wi_ow_f"], to=TensorProto.FLOAT)
    builder.add("Cast", ["wi_oh"], ["wi_oh_f"], to=TensorProto.FLOAT)
    builder.add("Div", ["wi_one_f", "wi_ow_f"], ["wi_uv_x"])
    builder.add("Div", ["wi_one_f", "wi_oh_f"], ["wi_uv_y"])
    builder.add("Cast", ["wi_ow"], ["wi_ow_d"], to=TensorProto.DOUBLE)
    builder.add("Cast", ["wi_oh"], ["wi_oh_d"], to=TensorProto.DOUBLE)
    builder.add("Cast", ["wi_ow"], ["wi_ow_i32"], to=TensorProto.INT32)
    builder.add("Cast", ["wi_oh"], ["wi_oh_i32"], to=TensorProto.INT32)
    builder.add("Sub", ["wi_ow", "wi_one_i64"], ["wi_owm1"])
    builder.add("Sub", ["wi_oh", "wi_one_i64"], ["wi_ohm1"])
    builder.add("Cast", ["wi_owm1"], ["wi_xmax_i32"], to=TensorProto.INT32)
    builder.add("Cast", ["wi_ohm1"], ["wi_ymax_i32"], to=TensorProto.INT32)

    output = dynamic_warp_init._sample_graph(
        builder, "sample", "tex", "pos_x", "pos_y")
    inputs = _sample_inputs() + [
        _value_info("wi_ow", TensorProto.INT64, [1]),
        _value_info("wi_oh", TensorProto.INT64, [1]),
    ]
    return _finish_model(builder, "warp_init_sample_dynamic", inputs, output)


def _session(model):
    """Single-threaded ORT session with graph optimisations disabled."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return onnxruntime.InferenceSession(
        model.SerializeToString(), options,
        providers=["CPUExecutionProvider"])


def _texture(width, height):
    """A deterministic BGRA8 test texture with distinct channel patterns."""
    values = (np.arange(width * height * 4, dtype=np.int64) * 37 + 11) % 256
    return values.astype(np.uint8).reshape(height, width, 4)


def _rgba_feed(texture):
    """The flat RGBA tensor the lowerings expect from BGRA8 texture memory."""
    return np.ascontiguousarray(texture[..., [2, 1, 0, 3]]).reshape(-1, 4)


def _run_model(session, lowering, width, height, texture, pos_x, pos_y):
    feeds = {
        "tex": _rgba_feed(texture),
        "pos_x": np.asarray(pos_x, dtype=np.float32),
        "pos_y": np.asarray(pos_y, dtype=np.float32),
    }
    if lowering == "dynamic":
        feeds["wi_ow"] = np.array([width], dtype=np.int64)
        feeds["wi_oh"] = np.array([height], dtype=np.int64)
    return session.run(None, feeds)[0]


def _float32_bits(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _uv_from_bits(bits):
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def _pos_from_uv_bits(bits, size):
    """A shader position whose float32 normalisation reproduces `bits`.

    The captured request carries float32 UV bits, not the original shader
    position.  Multiplying the UV by the texture size usually inverts the
    normalisation exactly; when the float32 product lands one ulp away, walk
    the adjacent float32 values until the normalisation matches.  At most a
    few steps are needed for the embedded vectors.
    """
    inverse = np.float32(1.0) / np.float32(size)
    uv = _uv_from_bits(bits)
    candidate = np.float32(uv * np.float32(size))
    candidate_bits = _float32_bits(candidate)
    for step in range(17):
        for delta in ((0,) if step == 0 else (step, -step)):
            value = np.array([(candidate_bits + delta) % (1 << 32)],
                             dtype=np.uint32).view(np.float32)[0]
            if _float32_bits(np.float32(value * inverse)) == bits:
                return np.float32(value)
    raise ValueError("no float32 position reproduces uv bits %d at size %d"
                     % (bits, size))


def _hardware_texture(vector):
    """A sparse 640x360 BGRA texture carrying one vector's four taps."""
    width, height = _HW_TEXTURE
    texture = np.zeros((height, width, 4), dtype=np.uint8)
    channel = vector["channel"]
    # Output channel is RGBA; texture memory is BGRA.  recon swizzles with
    # [2, 1, 0, 3], so store the focus channel at its memory index.
    memory_channel = (2, 1, 0, 3)[channel]
    x0, y0, x1, y1 = (vector["x0"], vector["y0"],
                      vector["x1"], vector["y1"])
    b00, b10, b01, b11 = vector["taps"]
    texture[y0, x0, memory_channel] = b00
    texture[y0, x1, memory_channel] = b10
    texture[y1, x0, memory_channel] = b01
    texture[y1, x1, memory_channel] = b11
    return texture


def _resolution_texture(width, height):
    """A deterministic BGRA8 texture for a real capture resolution."""
    columns = ((np.arange(width * 4, dtype=np.uint32) * 37 + 11)
               % 256).astype(np.uint8).reshape(1, width, 4)
    rows = ((np.arange(height, dtype=np.uint32) * 53 + 7)
            % 256).astype(np.uint8).reshape(height, 1, 1)
    return columns + rows


def _assert_same_fp64(case, expected, got, context):
    """Require exact float64 representation, not an approximate match."""
    case.assertEqual(expected.dtype, np.float64, context)
    case.assertEqual(got.dtype, np.float64, context)
    case.assertEqual(expected.shape, got.shape, context)
    expected_bits = np.ascontiguousarray(expected).view(np.uint64)
    got_bits = np.ascontiguousarray(got).view(np.uint64)
    if not np.array_equal(expected_bits, got_bits):
        delta = np.abs(expected - got)
        index = np.unravel_index(int(np.argmax(delta)), delta.shape)
        case.fail(
            "%s: float64 results differ (max |delta|=%r at %r; "
            "reference=%r, graph=%r)"
            % (context, float(delta[index]), index,
               float(expected[index]), float(got[index])))


def _coordinate_sets(width, height):
    """Named (pos_x, pos_y) float32 arrays spanning the clamp domain."""
    centre_x = (np.arange(width, dtype=np.float32) + np.float32(0.5))
    centre_y = (np.arange(height, dtype=np.float32) + np.float32(0.5))
    grid_x = np.tile(centre_x, (height, 1)).astype(np.float32)
    grid_y = np.tile(centre_y[:, None], (1, width)).astype(np.float32)

    yield ("constant",
           np.full((2, 3), centre_x[width // 2], dtype=np.float32),
           np.full((2, 3), centre_y[height // 2], dtype=np.float32))
    yield "centre", grid_x, grid_y
    yield ("interior-quarter",
           grid_x + np.float32(0.25), grid_y - np.float32(0.75))
    yield ("interior-eighth",
           grid_x - np.float32(0.125), grid_y + np.float32(0.375))
    yield "tie-low", grid_x - TIE_STEP, grid_y + TIE_STEP
    yield "tie-high", grid_x + TIE_STEP, grid_y - TIE_STEP

    if width > 1 and height > 1:
        edge_x = np.arange(1, width, dtype=np.float32)
        edge_y = np.arange(1, height, dtype=np.float32)
        yield ("texel-edge",
               np.tile(edge_x, (height - 1, 1)).astype(np.float32),
               np.tile(edge_y[:, None], (1, width - 1)).astype(np.float32))

    # Pos 0 normalises to a texel coordinate of -0.5: the low half-pixel clamp
    # boundary.  Pos width/height is the opposite boundary.  No negative
    # positions are used.
    yield ("clamp-boundary",
           np.array([[0.0, width], [0.0, width]], dtype=np.float32),
           np.array([[0.0, 0.0], [height, height]], dtype=np.float32))


class WarpInitSamplerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = {}
        for width, height in RESOLUTIONS:
            cls.sessions[("static", width, height)] = _session(
                _static_model(width, height))
            cls.sessions[("dynamic", width, height)] = _session(
                _dynamic_model(width, height))

    def _run(self, lowering, width, height, texture, pos_x, pos_y):
        return _run_model(self.sessions[(lowering, width, height)],
                          lowering, width, height, texture, pos_x, pos_y)

    def _assert_same_fp64(self, expected, got, context):
        _assert_same_fp64(self, expected, got, context)

    def test_static_lowering_matches_reference(self):
        for width, height in RESOLUTIONS:
            texture = _texture(width, height)
            for name, pos_x, pos_y in _coordinate_sets(width, height):
                with self.subTest(width=width, height=height,
                                  coordinates=name):
                    expected = recon._warp_init_sample(
                        texture, pos_x, pos_y)
                    got = self._run(
                        "static", width, height, texture, pos_x, pos_y)
                    self._assert_same_fp64(
                        expected, got,
                        "static %dx%d %s" % (width, height, name))

    def test_dynamic_lowering_matches_reference(self):
        for width, height in RESOLUTIONS:
            texture = _texture(width, height)
            for name, pos_x, pos_y in _coordinate_sets(width, height):
                with self.subTest(width=width, height=height,
                                  coordinates=name):
                    expected = recon._warp_init_sample(
                        texture, pos_x, pos_y)
                    got = self._run(
                        "dynamic", width, height, texture, pos_x, pos_y)
                    self._assert_same_fp64(
                        expected, got,
                        "dynamic %dx%d %s" % (width, height, name))

    def test_bgra_texture_memory_is_sampled_as_rgba(self):
        width, height = 2, 3
        texture = np.zeros((height, width, 4), dtype=np.uint8)
        texture[0, 0] = (10, 20, 30, 40)
        texture[0, 1] = (50, 60, 70, 80)
        pos_x = np.array([[0.5, 1.5]], dtype=np.float32)
        pos_y = np.array([[0.5, 0.5]], dtype=np.float32)

        # Both positions land exactly on a texel centre, so the reference
        # result is the BGRA bytes in RGBA order with no fractional mixing.
        expected = np.asarray(
            texture[0, [0, 1]][:, [2, 1, 0, 3]], dtype=np.float64)[None]
        self._assert_same_fp64(
            expected, recon._warp_init_sample(texture, pos_x, pos_y),
            "reference BGRA->RGBA")
        for lowering in ("static", "dynamic"):
            got = self._run(
                lowering, width, height, texture, pos_x, pos_y)
            self._assert_same_fp64(
                expected, got, "%s BGRA->RGBA" % lowering)


class HardwareGoldenSamplerTests(unittest.TestCase):
    """Hardware vectors: exact fp64 grid, <=1 hardware ulp, exact truncation."""

    @classmethod
    def setUpClass(cls):
        width, height = _HW_TEXTURE
        cls.sessions = {
            "static": _session(_static_model(width, height)),
            "dynamic": _session(_dynamic_model(width, height)),
        }

    def _assert_hardware(self, predicted, vector, context):
        self.assertEqual(np.asarray(predicted).dtype, np.float64, context)
        self.assertEqual(float(predicted), float(vector["grid_units"]),
                         context)
        ulp = abs(_float32_bits(predicted)
                  - _float32_bits(vector["hw_units"]))
        self.assertLessEqual(
            ulp, 1, "%s: %d hardware float32 ulp from measured units"
            % (context, ulp))
        self.assertEqual(
            int(np.trunc(float(predicted))), vector["hw_trunc"],
            "%s: truncation differs from measured hardware" % context)

    def test_hardware_vectors_match_grid_and_truncation(self):
        width, height = _HW_TEXTURE
        for vector in _HW_VECTORS:
            with self.subTest(vector=vector["label"]):
                channel = vector["channel"]
                self.assertEqual((vector["x1"], vector["y1"]),
                                 (vector["x0"] + 1, vector["y0"] + 1))
                texture = _hardware_texture(vector)
                pos_x = np.array(
                    [[_pos_from_uv_bits(vector["u_bits"], width)]],
                    dtype=np.float32)
                pos_y = np.array(
                    [[_pos_from_uv_bits(vector["v_bits"], height)]],
                    dtype=np.float32)
                # The reconstructed position must normalise back to the exact
                # request UV bits, so the golden x0/y0/weights are reproduced.
                recovered_u = np.float32(
                    pos_x[0, 0] * (np.float32(1.0) / np.float32(width)))
                recovered_v = np.float32(
                    pos_y[0, 0] * (np.float32(1.0) / np.float32(height)))
                self.assertEqual(_float32_bits(recovered_u),
                                 vector["u_bits"])
                self.assertEqual(_float32_bits(recovered_v),
                                 vector["v_bits"])

                grid_m = int(np.rint(np.float64(vector["total"]) / 8.0))
                self.assertEqual(grid_m / 8192.0, vector["grid_units"])
                if "tie" in vector:
                    quotient, remainder = divmod(vector["total"], 8)
                    self.assertEqual(remainder, 4)
                    if vector["tie"] == "up":
                        self.assertEqual(quotient % 2, 1)
                        self.assertEqual(grid_m, quotient + 1)
                    else:
                        self.assertEqual(quotient % 2, 0)
                        self.assertEqual(grid_m, quotient)
                if "min_span" in vector:
                    self.assertGreaterEqual(
                        max(vector["taps"]) - min(vector["taps"]),
                        vector["min_span"])
                if "old_trunc" in vector:
                    self.assertEqual(
                        int(np.trunc(vector["old_units"])),
                        vector["old_trunc"])
                    self.assertNotEqual(
                        vector["old_trunc"], vector["hw_trunc"])

                predicted = recon._warp_init_sample(
                    texture, pos_x, pos_y)[0, 0, channel]
                self._assert_hardware(
                    predicted, vector, "%s reference" % vector["label"])
                for lowering in ("static", "dynamic"):
                    got = _run_model(
                        self.sessions[lowering], lowering, width, height,
                        texture, pos_x, pos_y)[0, 0, channel]
                    self._assert_hardware(
                        got, vector,
                        "%s %s" % (vector["label"], lowering))


class RealResolutionSamplerTests(unittest.TestCase):
    def test_five_real_resolutions_few_coordinates(self):
        for width, height in REAL_RESOLUTIONS:
            with self.subTest(width=width, height=height):
                texture = _resolution_texture(width, height)
                pos_x = np.array(
                    [[0.0], [0.5], [width / 2.0],
                     [width - 0.5], [width]], dtype=np.float32)
                pos_y = np.array(
                    [[0.0], [0.5], [height / 2.0],
                     [height - 0.5], [height]], dtype=np.float32)
                expected = recon._warp_init_sample(texture, pos_x, pos_y)
                static_session = _session(_static_model(width, height))
                got_static = _run_model(
                    static_session, "static", width, height,
                    texture, pos_x, pos_y)
                _assert_same_fp64(
                    self, expected, got_static,
                    "real resolution %dx%d static" % (width, height))
                dynamic_session = _session(_dynamic_model(width, height))
                got_dynamic = _run_model(
                    dynamic_session, "dynamic", width, height,
                    texture, pos_x, pos_y)
                _assert_same_fp64(
                    self, expected, got_dynamic,
                    "real resolution %dx%d dynamic" % (width, height))
                # Drop the large allocations before the next resolution.
                del texture, expected, got_static, got_dynamic
                del static_session, dynamic_session


if __name__ == "__main__":
    unittest.main()
