"""Export captured XeFG operators to ONNX.

The first lowered family is the int8 convolution template (15 dispatches). The
packed texture layout is unpacked with reshape/transpose/slice ops inside the
graph, the convolution runs as `ConvInteger`, and the integer requantisation is
reproduced with Clamp/Div/Where. Every exported model is run with onnxruntime
and compared with the captured output over the whole allocation.

    python -I trace/export_onnx.py <run> --dispatch 4 --out onnx
    python -I trace/export_onnx.py <run> --all-conv --out onnx
    python -I trace/export_onnx.py <run> --all --out onnx
    python -I trace/export_onnx.py <run> --coverage --artifacts onnx --out onnx
"""
# Internal legacy research module; not a supported CLI entry point.
import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import d25_bound, recon


def _require_onnxruntime():
    """Import ONNX Runtime on first use so the exporter imports without it."""
    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        if name != "onnxruntime" and not name.startswith("onnxruntime."):
            raise
        raise ImportError(
            "ONNX Runtime is required to execute exported graphs; install the "
            "'ort' extra (pip install \"xefg-sdk[ort]\")"
        ) from exc
    return onnxruntime

# Properties of the calibrated fg78/ec5e0c65 two-times model. Pinning the
# dispatch set turns --all-conv from "whatever was found passed" into a
# regression gate: a classifier change that drops or adds a candidate fails.
EXPECTED_CONV_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21)
EXPECTED_FLOW_IDS = (26, 27, 28, 29, 30, 32, 34, 36)
EXPECTED_UPSAMPLE_IDS = (31, 33, 35, 37)
EXPECTED_GEMV_IDS = (14,)
EXPECTED_BUFFER_IDS = (10, 22)
EXPECTED_RESAMPLE_IDS = (11, 12, 23)
EXPECTED_COMPOSITE_IDS = (38,)
EXPECTED_DOWNSAMPLE_IDS = (1,)
EXPECTED_WARP_IDS = (13,)
EXPECTED_SPLAT_IDS = (24,)
EXPECTED_WARP_INIT_IDS = (25,)
EXPECTED_DISPATCHES = 38
EXPECTED_LOWERED = 38


def conv_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.conv_shaped(d))


def flow_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.flow_shaped(d))


def upsample_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.upsample_shaped(d))


def conv_ids_problem(ids):
    """None when the conv dispatch ids are exactly the pinned set."""
    ids = sorted(ids)
    if ids != list(EXPECTED_CONV_IDS):
        return "conv dispatches %s, expected %s" % (ids, list(EXPECTED_CONV_IDS))
    return None


def conv_set_problem(dispatches):
    """None when the conv candidate set is exactly the pinned one."""
    return conv_ids_problem(conv_dispatch_ids(dispatches))


def flow_ids_problem(ids):
    """None when the flow dispatch ids are exactly the pinned set."""
    ids = sorted(ids)
    if ids != list(EXPECTED_FLOW_IDS):
        return "flow dispatches %s, expected %s" % (ids, list(EXPECTED_FLOW_IDS))
    return None


def flow_set_problem(dispatches):
    """None when the flow candidate set is exactly the pinned one."""
    return flow_ids_problem(flow_dispatch_ids(dispatches))


def upsample_ids_problem(ids):
    """None when the upsample dispatch ids are exactly the pinned set."""
    ids = sorted(ids)
    if ids != list(EXPECTED_UPSAMPLE_IDS):
        return "upsample dispatches %s, expected %s" % (
            ids, list(EXPECTED_UPSAMPLE_IDS))
    return None


def upsample_set_problem(dispatches):
    """None when the upsample candidate set is exactly the pinned one."""
    return upsample_ids_problem(upsample_dispatch_ids(dispatches))


def gemv_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.gemv_shaped(d))


def buffer_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.buffer_output_shaped(d))


def resample_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.resample_shaped(d))


def _extra_ids_problem(ids, expected, name):
    ids = sorted(ids)
    if ids != list(expected):
        return "%s dispatches %s, expected %s" % (name, ids, list(expected))
    return None


def gemv_set_problem(dispatches):
    return _extra_ids_problem(gemv_dispatch_ids(dispatches),
                              EXPECTED_GEMV_IDS, "GEMV")


def buffer_set_problem(dispatches):
    return _extra_ids_problem(buffer_dispatch_ids(dispatches),
                              EXPECTED_BUFFER_IDS, "buffer-output")


def resample_set_problem(dispatches):
    return _extra_ids_problem(resample_dispatch_ids(dispatches),
                              EXPECTED_RESAMPLE_IDS, "resample")


def composite_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches
                  if recon.final_composite_shaped(d))


def downsample_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.downsample_shaped(d))


def warp_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.warp_shaped(d))


def splat_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches if recon.splat_shaped(d))


def splat_set_problem(dispatches):
    return _extra_ids_problem(splat_dispatch_ids(dispatches),
                              EXPECTED_SPLAT_IDS, "motion-vector splat")


def composite_set_problem(dispatches):
    return _extra_ids_problem(composite_dispatch_ids(dispatches),
                              EXPECTED_COMPOSITE_IDS, "final composite")


def downsample_set_problem(dispatches):
    return _extra_ids_problem(downsample_dispatch_ids(dispatches),
                              EXPECTED_DOWNSAMPLE_IDS, "downsample")


def warp_set_problem(dispatches):
    return _extra_ids_problem(warp_dispatch_ids(dispatches),
                              EXPECTED_WARP_IDS, "warp")


def warp_init_dispatch_ids(dispatches):
    return sorted(d["dispatch"] for d in dispatches
                  if recon.warp_init_shaped(d))


def warp_init_set_problem(dispatches):
    return _extra_ids_problem(warp_init_dispatch_ids(dispatches),
                              EXPECTED_WARP_INIT_IDS, "warp + flow init")


def result_passes(result):
    """A lowered dispatch passes its gate.

    Every family reproduces the captured allocation exactly except the warp +
    flow init, whose graph must equal the reference kernel exactly while the
    kernel itself is bounded by the documented texture-filter envelope.
    """
    if result.get("error") or result.get("mismatch") is None:
        return False
    if result.get("family") == "warp + flow init":
        replay = result.get("replay_mismatch")
        if replay is None:
            # A recorded result from before the lowering gate; the capture
            # comparison is then the only evidence available.
            return result["mismatch"] == 0
        if result.get("bytes") is None:
            # Older recorded results predate the width/height fields and only
            # carry the fraction-derived bound.
            return (replay == 0
                    and result.get("max_diff", 1 << 30)
                    <= d25_bound.WARP_INIT_MAX_DIFF
                    and result.get("mismatch", 1 << 30)
                    <= result.get("bound_bytes", 0))
        captured, _reason = d25_bound.evaluate(
            result.get("mismatch", 1 << 30), result.get("bytes", 0),
            result.get("max_diff", 1 << 30),
            width=result.get("width"), height=result.get("height"),
            content=result.get("content", d25_bound.STATIC))
        return replay == 0 and captured
    return result["mismatch"] == 0


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_values(value):
    """One digest for a tensor or an ordered list of tensors."""
    if isinstance(value, (list, tuple)):
        digest = hashlib.sha256()
        for item in value:
            digest.update(np.ascontiguousarray(item).tobytes())
        return digest.hexdigest()
    return _sha256_array(value)


def _finish_result(result, path, feeds, got, expected, model):
    """Attach the evidence a later --coverage --artifacts run can re-check."""
    result["file"] = os.path.basename(path)
    result["artifact_bytes"] = os.path.getsize(path)
    result["model_sha256"] = _sha256_file(path)
    result["inputs_sha256"] = {
        name: _sha256_array(value) for name, value in feeds.items()
    }
    result["expected_sha256"] = _sha256_values(expected)
    result["output_sha256"] = _sha256_values(got)
    result["nodes"] = len(model.graph.node)
    result["initializers"] = len(model.graph.initializer)
    if result.get("mismatch") == 0:
        result["status"] = "EXACT"
    else:
        result["status"] = "FAILED"
    return result


def artifact_name(entry):
    """The model file a lowered entry must have, or None if no rule exists."""
    if entry["family"] == "conv":
        return "d%04d_conv.onnx" % entry["dispatch"]
    if entry["family"] == "flow conv":
        return "d%04d_flow.onnx" % entry["dispatch"]
    if entry["family"] == "upsample":
        return "d%04d_upsample.onnx" % entry["dispatch"]
    if entry["family"] == "GEMV":
        return "d%04d_gemv.onnx" % entry["dispatch"]
    if entry["family"] == "buffer output":
        return "d%04d_bufout.onnx" % entry["dispatch"]
    if entry["family"] == "resample":
        return "d%04d_resample.onnx" % entry["dispatch"]
    if entry["family"] == "final composite":
        return "d%04d_composite.onnx" % entry["dispatch"]
    if entry["family"] == "downsample":
        return "d%04d_downsample.onnx" % entry["dispatch"]
    if entry["family"] == "warp":
        return "d%04d_warp.onnx" % entry["dispatch"]
    if entry["family"] == "splat":
        return "d%04d_splat.onnx" % entry["dispatch"]
    if entry["family"] == "warp + flow init":
        return "d%04d_warp_init.onnx" % entry["dispatch"]
    return None


def coverage_problems(entries, artifacts=None):
    """Structural checks for a coverage report; empty means consistent."""
    problems = []
    if len(entries) != EXPECTED_DISPATCHES:
        problems.append("%d dispatches, expected %d"
                        % (len(entries), EXPECTED_DISPATCHES))
    lowered = [e for e in entries if e["lowered"]]
    if len(lowered) != EXPECTED_LOWERED:
        problems.append("%d lowered, expected %d"
                        % (len(lowered), EXPECTED_LOWERED))
    if artifacts is not None:
        for entry in lowered:
            name = artifact_name(entry)
            if name is None:
                problems.append("d%04d has no artifact naming rule"
                                % entry["dispatch"])
                continue
            path = os.path.join(artifacts, name)
            if not os.path.exists(path):
                problems.append("missing %s" % name)
            elif os.path.getsize(path) == 0:
                problems.append("empty %s" % name)
    return problems


class Builder:
    def __init__(self, plan, prefix=""):
        self.plan = plan
        self.nodes = []
        self.initializers = []
        self.prefix = prefix

    def name(self, value):
        return self.prefix + value if value else value

    def add(self, op, inputs, outputs, **attrs):
        self.nodes.append(helper.make_node(
            op, [self.name(value) for value in inputs],
            [self.name(value) for value in outputs], **attrs))

    def tensor(self, name, array):
        self.initializers.append(numpy_helper.from_array(array, self.name(name)))

    def const(self, name, op, values):
        if op == "i32":
            self.tensor(name, np.asarray(values, dtype=np.int32))
        elif op == "i64":
            self.tensor(name, np.asarray(values, dtype=np.int64))
        elif op == "i8":
            self.tensor(name, np.asarray(values, dtype=np.int8))
        elif op == "fp32":
            self.tensor(name, np.asarray(values, dtype=np.float32))
        elif op == "u8":
            self.tensor(name, np.asarray(values, dtype=np.uint8))
        elif op == "u32":
            self.tensor(name, np.asarray(values, dtype=np.uint32))
        elif op == "fp64":
            self.tensor(name, np.asarray(values, dtype=np.float64))
        else:
            raise ValueError(op)


def _expand_input(builder, plan):
    """Unpack the packed texture bytes into [1, C, H, W] int32 channels."""
    p = plan
    H, W = p["src"].shape[1], p["src"].shape[2]
    valid_h, in_w = p["valid_h"], p["in_w"]
    mask = np.zeros((1, H, W, 1), dtype=np.int32)
    mask[0, :valid_h, :in_w, 0] = 1
    builder.tensor("valid", mask)
    builder.add("Cast", ["x0"], ["x32"], to=TensorProto.INT32)
    builder.add("Mul", ["x32", "valid"], ["masked"])
    need = 1 if p["template"] == "A" else (
        2 * p["blocks"] if p["src_bytes"] == 8 else p["blocks"])
    builder.const("arr_start", "i64", [0])
    builder.const("arr_end", "i64", [need])
    builder.const("arr_axis", "i64", [0])
    builder.const("arr_step", "i64", [1])
    builder.add("Slice", ["masked", "arr_start", "arr_end", "arr_axis", "arr_step"],
              ["used"])
    if p["template"] == "A":
        # One 2-byte texel per block: [c0, c1, folded zero point, 0].
        builder.const("ch0", "i64", [0])
        builder.const("ch1", "i64", [1])
        builder.const("ch_axis", "i64", [3])
        builder.const("ch_step", "i64", [1])
        builder.add("Slice", ["used", "ch0", "ch1", "ch_axis", "ch_step"], ["c0"])
        builder.add("Slice", ["used", "ch1", "che", "ch_axis", "ch_step"], ["c1"])
        builder.const("che", "i64", [2])
        builder.const("zero", "i32", [0])
        builder.const("zp", "i32", [p["zero_point"]])
        builder.add("Mul", ["valid", "zp"], ["zp_plane"])
        builder.add("Mul", ["valid", "zero"], ["zero_plane"])
        builder.add("Concat", ["c0", "c1", "zp_plane", "zero_plane"], ["lanes"], axis=3)
        builder.const("tile", "i64", [1, 1, 1, p["blocks"]])
        builder.add("Tile", ["lanes", "tile"], ["tiled"])
        builder.add("Transpose", ["tiled"], ["xch"], perm=[0, 3, 1, 2])
    else:
        blocks, src_bytes = p["blocks"], p["src_bytes"]
        if src_bytes == 8:
            builder.const("shape_b", "i64", [blocks, 2, H, W, 2, 4])
            builder.add("Reshape", ["used", "shape_b"], ["r"])
            builder.add("Transpose", ["r"], ["t"], perm=[0, 1, 4, 5, 2, 3])
            builder.const("shape_c", "i64", [1, blocks * 16, H, W])
        else:
            builder.const("shape_b", "i64", [blocks, H, W, 4, 4])
            builder.add("Reshape", ["used", "shape_b"], ["r"])
            builder.add("Transpose", ["r"], ["t"], perm=[0, 3, 4, 1, 2])
            builder.const("shape_c", "i64", [1, blocks * 16, H, W])
        builder.add("Reshape", ["t", "shape_c"], ["xch"])
    return "xch"


def build_conv(plan, prefix=""):
    """One int8 conv dispatch as an ONNX graph with input `x0`, output `u0`."""
    p = plan
    H, W = p["H"], p["W"]
    out_h, out_w = p["out_h"], p["out_w"]
    stride_y, stride_x = p["stride_y"], p["stride_x"]
    phase_x, phase_y = p["phase"]
    slices, groups = p["slices"], p["groups"]

    b = Builder(p, prefix)
    xch = _expand_input(b, p)
    # Recon pads by three; the conv window therefore starts at 3+phase.
    b.const("pad", "i64", [0, 0, 3, 3, 0, 0, 3, 3])
    b.const("pad_value", "i32", [0])
    b.add("Pad", [xch, "pad", "pad_value"], ["padded"])
    start_y, start_x = 3 + phase_y, 3 + phase_x
    crop_h = stride_y * (out_h - 1) + 3
    crop_w = stride_x * (out_w - 1) + 3
    b.const("crop_starts", "i64", [0, 0, start_y, start_x])
    b.const("crop_ends", "i64", [1, p["blocks"] * 16 if p["template"] == "BC" else p["blocks"] * 4,
                                 start_y + crop_h, start_x + crop_w])
    b.const("crop_axes", "i64", [0, 1, 2, 3])
    b.const("crop_steps", "i64", [1, 1, 1, 1])
    b.add("Slice", ["padded", "crop_starts", "crop_ends", "crop_axes", "crop_steps"], ["crop"])
    b.add("Cast", ["crop"], ["crop8"], to=TensorProto.INT8)

    # [z, block, group, tap, word, lane_out, lane_in] -> [M, C, 3, 3];
    # the shader's last weight axis is the dot lane, so it becomes the input
    # channel, and the second-last (4 output lanes) becomes the output channel.
    weights = p["weights"].transpose(0, 2, 5, 1, 4, 6, 3).reshape(
        slices * groups * 4, p["blocks"] * p["words_per_tap"] * 4, 3, 3)
    b.tensor("w", weights.astype(np.int8))
    b.add("ConvInteger", ["crop8", "w"], ["acc"], strides=[stride_y, stride_x])

    bias = p["bias"].reshape(slices * groups * 4, 1, 1).astype(np.int32)
    b.tensor("bias", bias)
    b.add("Add", ["acc", "bias"], ["biased"])
    b.const("shape_out", "i64", [slices, groups, 4, out_h, out_w])
    b.add("Reshape", ["biased", "shape_out"], ["acc5"])
    b.add("Transpose", ["acc5"], ["acc_zgyl"], perm=[0, 1, 3, 4, 2])

    activation = p["activation"].reshape(slices, groups, 1, 1, 4).astype(np.int32)
    b.tensor("activation", activation)
    b.const("lo", "i32", [p["lo"]])
    b.const("hi", "i32", [p["hi"]])
    b.const("zero5", "i32", [0])
    b.add("Clip", ["acc_zgyl", "lo", "hi"], ["clamped"])
    b.add("Max", ["clamped", "zero5"], ["pos"])
    b.add("Min", ["clamped", "zero5"], ["neg"])

    def shift(src, addend, shift_bits, tag):
        # Arithmetic right shift: truncating division after biasing negatives.
        denom = 1 << shift_bits
        b.const(tag + "_add", "i32", [1 << (shift_bits - 1)])
        b.add("Add", [src, tag + "_add"], [tag + "_rounded"])
        b.const(tag + "_denom", "i32", [denom])
        b.const(tag + "_bias", "i32", [denom - 1])
        b.add("Less", [tag + "_rounded", "zero5"], [tag + "_neg"])
        b.add("Where", [tag + "_neg", tag + "_bias", "zero5"], [tag + "_adjust"])
        b.add("Sub", [tag + "_rounded", tag + "_adjust"], [tag + "_nudged"])
        b.add("Div", [tag + "_nudged", tag + "_denom"], [tag + "_shifted"])

    b.const("scale_c", "i32", [p["scale"]])
    b.add("Mul", ["pos", "scale_c"], ["pos_scaled"])
    shift("pos_scaled", None, p["s1"], "pos")
    b.add("Mul", ["neg", "activation"], ["neg_scaled"])
    shift("neg_scaled", None, p["s2"], "neg")
    b.add("Add", ["pos_shifted", "neg_shifted"], ["requant"])
    b.const("qlo", "i32", [-128])
    b.const("qhi", "i32", [127])
    b.add("Clip", ["requant", "qlo", "qhi"], ["qclamped"])
    b.add("Transpose", ["qclamped"], ["qzyx"], perm=[0, 2, 3, 1, 4])
    b.const("shape_bytes", "i64", [slices, out_h, out_w, p["out_bytes"]])
    b.add("Reshape", ["qzyx", "shape_bytes"], ["bytes"])
    b.const("out_pad", "i64", [0, 1, 0, 0, 0, H - 1 - out_h, W - out_w, 0])
    b.const("out_zero", "i32", [0])
    b.add("Pad", ["bytes", "out_pad", "out_zero"], ["u0_i32"])
    b.add("Cast", ["u0_i32"], ["u0"], to=TensorProto.INT8)

    graph = helper.make_graph(
        b.nodes, b.name("xefg_conv_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(b.name("x0"), TensorProto.INT8,
                                       list(p["src"].shape))],
        [helper.make_tensor_value_info(b.name("u0"), TensorProto.INT8,
                                       list(p["expected"].shape))],
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_conv(run_dir, dispatch, params, out_dir, verify_path=None):
    plan, error = recon.conv_plan(run_dir, dispatch, params)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "conv", "error": error}
    array = plan["src"].shape[0]
    need = 1 if plan["template"] == "A" else (
        2 * plan["blocks"] if plan["src_bytes"] == 8 else plan["blocks"])
    if array < need:
        return {"dispatch": dispatch["dispatch"], "family": "conv",
                "error": "input has %d slices, needs %d" % (array, need)}
    if plan["H"] - 1 - plan["out_h"] < 0 or plan["W"] - plan["out_w"] < 0:
        return {"dispatch": dispatch["dispatch"], "family": "conv",
                "error": "output does not fit the allocation"}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_conv(plan)
        onnx.checker.check_model(model)
        name = "d%04d_conv" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["src"].astype(np.int8)}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"].astype(np.int8)
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "conv",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
        "input_shape": list(plan["src"].shape),
        "output_shape": list(expected.shape),
    }
    if mismatch:
        bad = np.argwhere(diff != 0)[:8]
        result["first_bad"] = [list(map(int, b)) for b in bad]
    return _finish_result(result, path, feeds, got, expected, model)


def _arithmetic_shift(builder, source, bits, tag):
    """Signed right shift of an int32 tensor by `bits`, truncating toward -inf."""
    if bits <= 0:
        return source
    builder.const(tag + "_add", "i32", [1 << (bits - 1)])
    builder.add("Add", [source, tag + "_add"], [tag + "_rounded"])
    denom = 1 << bits
    builder.const(tag + "_denom", "i32", [denom])
    builder.const(tag + "_bias", "i32", [denom - 1])
    builder.const(tag + "_zero", "i32", [0])
    builder.add("Less", [tag + "_rounded", tag + "_zero"], [tag + "_neg"])
    builder.add("Where", [tag + "_neg", tag + "_bias", tag + "_zero"],
                [tag + "_adjust"])
    builder.add("Sub", [tag + "_rounded", tag + "_adjust"], [tag + "_nudged"])
    builder.add("Div", [tag + "_nudged", tag + "_denom"], [tag + "_shifted"])
    return tag + "_shifted"


def _flow_input(builder, plan):
    """Return [1, slices*16, H, W] int8 with the shader's byte semantics.

    `d26` adds 0x80808080 to each packed 32-bit word before `dot4AddPacked`.
    That is a word add, not four independent byte adds, so the graph rebuilds
    the word from its four bytes, adds the constant modulo 2^32, and extracts
    signed bytes. Out-of-bounds texels read raw zero, so the constant is still
    added there; `build_flow` pads with the resulting bias byte.
    """
    p = plan
    if p["add"] == 0:
        builder.add("Cast", ["x0"], ["flow_u32"], to=TensorProto.INT32)
        builder.const("flow_128", "i32", [128])
        builder.add("GreaterOrEqual", ["flow_u32", "flow_128"], ["flow_ge"])
        builder.const("flow_256", "i32", [256])
        builder.add("Sub", ["flow_u32", "flow_256"], ["flow_minus"])
        builder.add("Where", ["flow_ge", "flow_minus", "flow_u32"], ["flow_signed32"])
        builder.add("Cast", ["flow_signed32"], ["flow_signed"], to=TensorProto.INT8)
    else:
        # [slices, H, W, 4 words, 4 bytes]
        builder.const("flow_words_shape", "i64", [p["slices"], p["in_h"], p["in_w"], 4, 4])
        builder.add("Reshape", ["x0", "flow_words_shape"], ["flow_words"])
        builder.const("flow_split_words", "i64", [1, 1, 1, 1])
        builder.add("Split", ["flow_words", "flow_split_words"],
                    ["flow_word0", "flow_word1", "flow_word2", "flow_word3"],
                    axis=3)
        builder.const("flow_split_bytes", "i64", [1, 1, 1, 1])
        builder.const("flow_add64", "i64", [p["add"] & 0xFFFFFFFF])
        builder.const("flow_mod32", "i64", [1 << 32])
        builder.const("flow_256_64", "i64", [256])
        builder.const("flow_128_64", "i64", [128])
        words = []
        for word in range(4):
            builder.add("Split", ["flow_word%d" % word, "flow_split_bytes"],
                        ["flow_w%d_b%d" % (word, byte) for byte in range(4)],
                        axis=-1)
            packed = []
            for byte in range(4):
                builder.add("Cast", ["flow_w%d_b%d" % (word, byte)],
                            ["flow_w%d_b%d_64" % (word, byte)], to=TensorProto.INT64)
                if byte:
                    builder.const("flow_w%d_shift_%d" % (word, byte), "i64",
                                  [1 << (8 * byte)])
                    builder.add("Mul", ["flow_w%d_b%d_64" % (word, byte),
                                        "flow_w%d_shift_%d" % (word, byte)],
                                ["flow_w%d_b%d_s" % (word, byte)])
                    packed.append("flow_w%d_b%d_s" % (word, byte))
                else:
                    packed.append("flow_w%d_b%d_64" % (word, byte))
            builder.add("Add", [packed[0], packed[1]], ["flow_w%d_01" % word])
            builder.add("Add", [packed[2], packed[3]], ["flow_w%d_23" % word])
            builder.add("Add", ["flow_w%d_01" % word, "flow_w%d_23" % word],
                        ["flow_w%d_sum" % word])
            builder.add("Add", ["flow_w%d_sum" % word, "flow_add64"],
                        ["flow_w%d_added" % word])
            builder.add("Mod", ["flow_w%d_added" % word, "flow_mod32"],
                        ["flow_w%d_mod" % word])
            bytes_out = []
            for byte in range(4):
                source = "flow_w%d_mod" % word
                if byte:
                    builder.const("flow_w%d_div_%d" % (word, byte), "i64",
                                  [1 << (8 * byte)])
                    builder.add("Div", [source, "flow_w%d_div_%d" % (word, byte)],
                                ["flow_w%d_d%d" % (word, byte)])
                    source = "flow_w%d_d%d" % (word, byte)
                builder.add("Mod", [source, "flow_256_64"],
                            ["flow_w%d_byte%d" % (word, byte)])
                builder.add("GreaterOrEqual", ["flow_w%d_byte%d" % (word, byte),
                                               "flow_128_64"],
                            ["flow_w%d_ge%d" % (word, byte)])
                builder.add("Sub", ["flow_w%d_byte%d" % (word, byte), "flow_256_64"],
                            ["flow_w%d_minus%d" % (word, byte)])
                builder.add("Where", ["flow_w%d_ge%d" % (word, byte),
                                      "flow_w%d_minus%d" % (word, byte),
                                      "flow_w%d_byte%d" % (word, byte)],
                            ["flow_w%d_signed%d" % (word, byte)])
                builder.add("Cast", ["flow_w%d_signed%d" % (word, byte)],
                            ["flow_w%d_i8_%d" % (word, byte)], to=TensorProto.INT8)
                bytes_out.append("flow_w%d_i8_%d" % (word, byte))
            builder.add("Concat", bytes_out, ["flow_word%d_i8" % word], axis=-1)
            words.append("flow_word%d_i8" % word)
        builder.add("Concat", words, ["flow_bytes"], axis=3)
        builder.const("flow_src_shape", "i64", [p["slices"], p["in_h"], p["in_w"], 16])
        builder.add("Reshape", ["flow_bytes", "flow_src_shape"], ["flow_signed"])
    builder.const("flow_nchw_shape", "i64", [1, p["slices"] * 16, p["in_h"], p["in_w"]])
    builder.add("Transpose", ["flow_signed"], ["flow_nhwc"], perm=[0, 3, 1, 2])
    builder.add("Reshape", ["flow_nhwc", "flow_nchw_shape"], ["flow_x"])
    return "flow_x"


def build_flow(plan, prefix=""):
    """One flow/refinement dispatch as a dense int8 3x3 convolution."""
    p = plan
    builder = Builder(p, prefix)
    source = _flow_input(builder, p)
    builder.tensor("flow_w", p["weights"])

    accumulators = []
    for index, (left, top, right, bottom) in enumerate(p["pads"]):
        builder.const("flow_pads_%d" % index, "i64",
                      [0, 0, top, left, 0, 0, bottom, right])
        builder.const("flow_pad_value_%d" % index, "i8", [p["pad_value"]])
        builder.add("Pad", [source, "flow_pads_%d" % index,
                            "flow_pad_value_%d" % index],
                    ["flow_padded_%d" % index])
        builder.add("ConvInteger", ["flow_padded_%d" % index, "flow_w"],
                    ["flow_acc_%d" % index],
                    strides=[p["stride_y"], p["stride_x"]])
        accumulators.append("flow_acc_%d" % index)
    if len(accumulators) == 1:
        acc = accumulators[0]
    else:
        builder.add("Max", accumulators, ["flow_acc"])
        acc = "flow_acc"

    # The DXIL's `fmul fast` + `fadd fast` is contracted into one FMA by the
    # backend, so the requant has a single float32 rounding. ONNX has no FMA
    # op; double is wide enough to hold the exact float32 product and sum, so
    # computing in double and casting once to float32 reproduces it. At
    # 2560x1440 the two-round float32 path differs by one byte in d29.
    builder.add("Cast", [acc], ["flow_accd"], to=TensorProto.DOUBLE)
    builder.tensor("flow_scale", p["scale"].reshape(1, -1, 1, 1))
    builder.tensor("flow_bias", p["bias"].reshape(1, -1, 1, 1))
    builder.add("Cast", ["flow_scale"], ["flow_scale_d"], to=TensorProto.DOUBLE)
    builder.add("Cast", ["flow_bias"], ["flow_bias_d"], to=TensorProto.DOUBLE)
    builder.add("Mul", ["flow_accd", "flow_scale_d"], ["flow_scaled_d"])
    builder.add("Add", ["flow_scaled_d", "flow_bias_d"], ["flow_biased_d"])
    builder.add("Cast", ["flow_biased_d"], ["flow_biased"], to=TensorProto.FLOAT)
    builder.add("Cast", ["flow_biased"], ["flow_requant"], to=TensorProto.INT32)
    builder.const("flow_lo", "i32", [p["lo"]])
    builder.const("flow_hi", "i32", [p["hi"]])
    builder.add("Clip", ["flow_requant", "flow_lo", "flow_hi"], ["flow_clipped"])
    builder.add("Cast", ["flow_clipped"], ["flow_q"], to=TensorProto.INT8)
    builder.const("flow_out_shape", "i64", [p["outs"], 16, p["height"], p["width"]])
    builder.add("Reshape", ["flow_q", "flow_out_shape"], ["flow_q5"])
    builder.add("Transpose", ["flow_q5"], ["u0"], perm=[0, 2, 3, 1])

    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_flow_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(
            builder.name("x0"), TensorProto.UINT8,
            [p["slices"], p["in_h"], p["in_w"], 16])],
        [helper.make_tensor_value_info(
            builder.name("u0"), TensorProto.INT8,
            [p["outs"], p["height"], p["width"], 16])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def build_upsample(plan, prefix=""):
    """2x fixed-point bilinear staging + residual + packed int8 3x3 conv."""
    p = plan
    builder = Builder(p, prefix)
    channels = p["slices"] * 16

    # Low-res input [slices, in_h, in_w, 16] -> [1, C, in_h, in_w] float32.
    builder.const("up_src_shape", "i64", [p["slices"], p["in_h"], p["in_w"], 16])
    builder.add("Reshape", [builder.name("x0"), "up_src_shape"], ["up_src"])
    builder.add("Transpose", ["up_src"], ["up_src_hwc"], perm=[0, 3, 1, 2])
    builder.const("up_nchw_shape", "i64", [1, channels, p["in_h"], p["in_w"]])
    builder.add("Reshape", ["up_src_hwc", "up_nchw_shape"], ["up_src_nchw"])
    builder.add("Cast", ["up_src_nchw"], ["up_src_f"], to=TensorProto.FLOAT)

    # Half-pixel 2x bilinear with the separable integer kernel [1,3,3,1]/16.
    kernel = np.asarray([1, 3, 3, 1], dtype=np.float32)
    weights = np.zeros((channels, 1, 4, 4), dtype=np.float32)
    weights[:, 0] = np.outer(kernel, kernel)
    builder.tensor("up_kernel", weights)
    builder.add("ConvTranspose", ["up_src_f", "up_kernel"], ["up_raw"],
                strides=[2, 2], pads=[1, 1, 1, 1], group=channels)

    # sum is an exact float32 integer; dividing by 16 only shifts the exponent,
    # so the cast back to int32 truncates toward zero exactly like _sdiv(.,16).
    builder.const("up_16", "fp32", 16.0)
    builder.add("Div", ["up_raw", "up_16"], ["up_scaled"])
    builder.add("Cast", ["up_scaled"], ["up_int"], to=TensorProto.INT32)
    builder.const("up_crop_starts", "i64", [0, 0, 0, 0])
    builder.const("up_crop_ends", "i64", [1, channels, p["height"], p["width"]])
    builder.const("up_crop_axes", "i64", [0, 1, 2, 3])
    builder.const("up_crop_steps", "i64", [1, 1, 1, 1])
    builder.add("Slice", ["up_int", "up_crop_starts", "up_crop_ends",
                          "up_crop_axes", "up_crop_steps"], ["up_cropped"])

    # Co-located high-res residual.
    builder.const("up_high_shape", "i64", [p["slices"], p["height"], p["width"], 16])
    builder.add("Reshape", [builder.name("x1"), "up_high_shape"], ["up_high"])
    builder.add("Transpose", ["up_high"], ["up_high_hwc"], perm=[0, 3, 1, 2])
    builder.const("up_high_nchw", "i64", [1, channels, p["height"], p["width"]])
    builder.add("Reshape", ["up_high_hwc", "up_high_nchw"], ["up_high_nchw_t"])
    builder.add("Cast", ["up_high_nchw_t"], ["up_high_i"], to=TensorProto.INT32)

    builder.add("Add", ["up_cropped", "up_high_i"], ["up_sum"])
    builder.const("up_lo", "i32", [0])
    builder.const("up_hi", "i32", [127])
    builder.add("Clip", ["up_sum", "up_lo", "up_hi"], ["up_clamped"])
    builder.add("Cast", ["up_clamped"], ["up_staged"], to=TensorProto.INT8)

    builder.tensor("up_conv_w", p["weights"])
    builder.add("ConvInteger", ["up_staged", "up_conv_w"], ["up_acc"],
                pads=[1, 1, 1, 1])
    builder.add("Cast", ["up_acc"], ["up_accf"], to=TensorProto.FLOAT)
    builder.tensor("up_scale", p["scale"].reshape(1, -1, 1, 1))
    builder.tensor("up_bias", p["bias"].reshape(1, -1, 1, 1))
    builder.add("Mul", ["up_accf", "up_scale"], ["up_scaled_acc"])
    builder.add("Add", ["up_scaled_acc", "up_bias"], ["up_biased"])
    builder.add("Cast", ["up_biased"], ["up_requant"], to=TensorProto.INT32)
    builder.const("up_qlo", "i32", [0])
    builder.const("up_qhi", "i32", [127])
    builder.add("Clip", ["up_requant", "up_qlo", "up_qhi"], ["up_qclamped"])
    builder.add("Cast", ["up_qclamped"], ["up_q"], to=TensorProto.INT8)
    builder.const("up_out_shape", "i64", [p["outs"], 16, p["height"], p["width"]])
    builder.add("Reshape", ["up_q", "up_out_shape"], ["up_q5"])
    builder.add("Transpose", ["up_q5"], ["u0"], perm=[0, 2, 3, 1])

    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_upsample_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(
            builder.name("x0"), TensorProto.INT8,
            [p["slices"], p["in_h"], p["in_w"], 16]),
         helper.make_tensor_value_info(
            builder.name("x1"), TensorProto.INT8,
            [p["slices"], p["height"], p["width"], 16])],
        [helper.make_tensor_value_info(
            builder.name("u0"), TensorProto.INT8,
            [p["outs"], p["height"], p["width"], 16])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_upsample(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.upsample_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "upsample",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_upsample(plan)
        onnx.checker.check_model(model)
        name = "d%04d_upsample" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["low"], "x1": plan["high"]}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"]
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "upsample",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
        "input_shape": list(plan["low"].shape),
        "output_shape": list(expected.shape),
    }
    if mismatch:
        bad = np.argwhere(diff != 0)[:8]
        result["first_bad"] = [list(map(int, b)) for b in bad]
    return _finish_result(result, path, feeds, got, expected, model)


def export_flow(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.flow_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "flow conv",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_flow(plan)
        onnx.checker.check_model(model)
        name = "d%04d_flow" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["src"]}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"]
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "flow conv",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
        "input_shape": list(plan["src"].shape),
        "output_shape": list(expected.shape),
    }
    if mismatch:
        bad = np.argwhere(diff != 0)[:8]
        result["first_bad"] = [list(map(int, b)) for b in bad]
    return _finish_result(result, path, feeds, got, expected, model)


def _unsigned_byte(builder, source, tag):
    """Map a signed int32 byte to 0..255."""
    builder.const(tag + "_256", "i32", [256])
    builder.const(tag + "_zero", "i32", [0])
    builder.add("Less", [source, tag + "_zero"], [tag + "_neg"])
    builder.add("Add", [source, tag + "_256"], [tag + "_plus"])
    builder.add("Where", [tag + "_neg", tag + "_plus", source], [tag + "_byte"])
    return tag + "_byte"


def _overlay_channels(builder, plan, slot):
    """BGRA8 overlay -> float32 (r, g, b, a), each [height, width, 1]."""
    builder.add("Cast", [builder.name("in_" + slot)], ["cp_" + slot + "_f"],
                to=TensorProto.FLOAT)
    builder.add("Div", ["cp_" + slot + "_f", "cp_255"],
                ["cp_" + slot + "_n"])
    channels = []
    for index, tag in ((2, "r"), (1, "g"), (0, "b"), (3, "a")):
        builder.const("cp_%s_%s_idx" % (slot, tag), "i64", [index])
        builder.add("Gather", ["cp_" + slot + "_n", "cp_%s_%s_idx" % (slot, tag)],
                    ["cp_%s_%s" % (slot, tag)], axis=2)
        channels.append("cp_%s_%s" % (slot, tag))
    return channels


def _gate(builder, name, index):
    builder.const(name + "_idx%d" % index, "i64", [index])
    builder.add("Gather", [name, name + "_idx%d" % index],
                [name + "_g%d" % index], axis=2)
    return name + "_g%d" % index


def _warp_sample(builder, plan, tag, flow_x, flow_y, channel):
    """Bilinear grid sample of one int8 channel, in the DXIL operation order."""
    p = plan
    builder.add("Floor", [flow_x], [tag + "_xfloor"])
    builder.add("Sub", [flow_x, tag + "_xfloor"], [tag + "_fx"])
    builder.add("Floor", [flow_y], [tag + "_yfloor"])
    builder.add("Sub", [flow_y, tag + "_yfloor"], [tag + "_fy"])
    builder.add("Cast", [tag + "_xfloor"], [tag + "_ix"], to=TensorProto.INT32)
    builder.add("Cast", [tag + "_yfloor"], [tag + "_iy"], to=TensorProto.INT32)

    builder.const(tag + "_xi", "i32", np.arange(p["src_w"], dtype=np.int32)[None, :])
    builder.const(tag + "_yi", "i32", np.arange(p["src_h"], dtype=np.int32)[:, None])
    builder.const(tag + "_zero", "i32", [0])
    # The replay slices the texture to the logical src_w x src_h region before
    # sampling, so the clamp is against that region, not the larger allocation.
    builder.const(tag + "_wmax", "i32", [p["src_w"] - 1])
    builder.const(tag + "_hmax", "i32", [p["src_h"] - 1])
    builder.const(tag + "_one", "i32", [1])
    builder.const(tag + "_texw", "i32", [p["tex_w"]])
    builder.const(tag + "_two", "i32", [2])
    builder.const(tag + "_chan", "i32", [channel])

    builder.add("Add", [tag + "_ix", tag + "_xi"], [tag + "_bx"])
    builder.add("Add", [tag + "_iy", tag + "_yi"], [tag + "_by"])
    builder.add("Add", [tag + "_bx", tag + "_one"], [tag + "_bx1"])
    builder.add("Add", [tag + "_by", tag + "_one"], [tag + "_by1"])
    builder.add("Clip", [tag + "_bx", tag + "_zero", tag + "_wmax"],
                [tag + "_cx0"])
    builder.add("Clip", [tag + "_bx1", tag + "_zero", tag + "_wmax"],
                [tag + "_cx1"])
    builder.add("Clip", [tag + "_by", tag + "_zero", tag + "_hmax"],
                [tag + "_cy0"])
    builder.add("Clip", [tag + "_by1", tag + "_zero", tag + "_hmax"],
                [tag + "_cy1"])

    taps = {}
    for cy, cx, name in ((tag + "_cy0", tag + "_cx0", "00"),
                         (tag + "_cy0", tag + "_cx1", "01"),
                         (tag + "_cy1", tag + "_cx0", "10"),
                         (tag + "_cy1", tag + "_cx1", "11")):
        builder.add("Mul", [cy, tag + "_texw"], [tag + "_row_" + name])
        builder.add("Add", [tag + "_row_" + name, cx], [tag + "_cell_" + name])
        builder.add("Mul", [tag + "_cell_" + name, tag + "_two"],
                    [tag + "_byte_" + name])
        builder.add("Add", [tag + "_byte_" + name, tag + "_chan"],
                    [tag + "_idx_" + name])
        builder.add("Gather", [builder.name("x0"), tag + "_idx_" + name],
                    [tag + "_v_" + name], axis=0)
        builder.add("Cast", [tag + "_v_" + name], [tag + "_vf_" + name],
                    to=TensorProto.FLOAT)
        taps[name] = tag + "_vf_" + name

    builder.const(tag + "_onef", "fp32", 1.0)
    builder.add("Sub", [tag + "_onef", tag + "_fx"], [tag + "_ifx"])
    builder.add("Mul", [taps["01"], tag + "_fx"], [tag + "_m01"])
    builder.add("Mul", [taps["00"], tag + "_ifx"], [tag + "_m00"])
    builder.add("Add", [tag + "_m01", tag + "_m00"], [tag + "_top"])
    builder.add("Mul", [taps["10"], tag + "_ifx"], [tag + "_m10"])
    builder.add("Sub", [tag + "_m10", tag + "_top"], [tag + "_a"])
    builder.add("Mul", [taps["11"], tag + "_fx"], [tag + "_m11"])
    builder.add("Add", [tag + "_a", tag + "_m11"], [tag + "_b"])
    builder.add("Mul", [tag + "_b", tag + "_fy"], [tag + "_bfy"])
    builder.add("Add", [tag + "_bfy", tag + "_top"], [tag + "_res"])
    return tag + "_res"


def build_warp(plan, prefix=""):
    p = plan
    builder = Builder(p, prefix)
    builder.const("wp_flow_index", "i64", p["flow_index"])
    builder.add("Gather", [builder.name("x1"), "wp_flow_index"], ["wp_flow"],
                axis=0)
    builder.add("Cast", ["wp_flow"], ["wp_flow_f"], to=TensorProto.FLOAT)
    flows = []
    for index in range(4):
        builder.const("wp_flow_idx%d" % index, "i64", [index])
        builder.add("Gather", ["wp_flow_f", "wp_flow_idx%d" % index],
                    ["wp_f%d" % index], axis=2)
        builder.const("wp_sq%d" % index, "i64", [2])
        builder.add("Squeeze", ["wp_f%d" % index, "wp_sq%d" % index],
                    ["wp_fd%d" % index])
        flows.append("wp_fd%d" % index)

    low = _warp_sample(builder, p, "wp_low", flows[0], flows[1], 0)
    high = _warp_sample(builder, p, "wp_high", flows[2], flows[3], 1)
    channels = []
    for name, tag in ((low, "wp_low"), (high, "wp_high")):
        builder.add("Cast", [name], [tag + "_int"], to=TensorProto.INT32)
        builder.const(tag + "_lo", "i32", [-128])
        builder.const(tag + "_hi", "i32", [127])
        builder.add("Clip", [tag + "_int", tag + "_lo", tag + "_hi"],
                    [tag + "_clip"])
        builder.const(tag + "_m256", "i32", [256])
        builder.add("Mod", [tag + "_clip", tag + "_m256"], [tag + "_byte"])
        builder.add("Cast", [tag + "_byte"], [tag + "_u8"], to=TensorProto.UINT8)
        builder.const(tag + "_axis", "i64", [2])
        builder.add("Unsqueeze", [tag + "_u8", tag + "_axis"],
                    [tag + "_u8_3d"])
        channels.append(tag + "_u8_3d")

    builder.const("wp_zero_shape", "i64", [p["src_h"], p["src_w"], 1])
    builder.add("ConstantOfShape", ["wp_zero_shape"], ["wp_zero"],
                value=numpy_helper.from_array(np.array([0], dtype=np.uint8)))
    builder.add("Concat", [channels[0], channels[1], "wp_zero", "wp_zero"],
                ["wp_packed"], axis=2)
    builder.const("wp_pads", "i64",
                  [1, 0, 0, p["out_rows"] - 1 - p["src_h"], 0, 0])
    builder.const("wp_pad_zero", "u8", [0])
    builder.add("Pad", ["wp_packed", "wp_pads", "wp_pad_zero"], ["u0"])

    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_warp_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(builder.name("x0"), TensorProto.INT8,
                                       [p["texture"].size]),
         helper.make_tensor_value_info(builder.name("x1"), TensorProto.FLOAT16,
                                       [p["flow"].size])],
        [helper.make_tensor_value_info(builder.name("u0"), TensorProto.UINT8,
                                       [p["out_rows"], p["src_w"], 4])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_warp(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.warp_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "warp",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_warp(plan)
        onnx.checker.check_model(model)
        name = "d%04d_warp" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["texture"], "x1": plan["flow"]}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"].reshape(got.shape)
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "warp",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
    }
    return _finish_result(result, path, feeds, got, expected, model)


def _half_bits_to_float(builder, tag, bits):
    """Rebuild a finite fp16 value from its 16-bit pattern, arithmetically.

    ONNX has no bitcast. `bits` is a uint32 tensor holding the low 16 bits; the
    exponent and mantissa are decoded and recombined as `num * 2^(exp-25)`,
    which is exact for every finite half. Exponent 31 (inf/NaN) maps to zero;
    `warp_init_plan` refuses a capture that contains one.
    """
    b = builder
    b.const(tag + "_i0", "i32", [0])
    b.const(tag + "_i1", "i32", [1])
    # BitShift/BitwiseAnd need an unsigned operand; only the decoded fields are
    # cast to int32 for the arithmetic.
    b.const(tag + "_shift10", "u32", [10])
    b.add("BitShift", [bits, tag + "_shift10"], [tag + "_e"],
          direction="RIGHT")
    b.const(tag + "_mask1f", "u32", [0x1F])
    b.add("BitwiseAnd", [tag + "_e", tag + "_mask1f"], [tag + "_e_masked"])
    b.add("Cast", [tag + "_e_masked"], [tag + "_exp"], to=TensorProto.INT32)
    b.const(tag + "_mask3ff", "u32", [0x3FF])
    b.add("BitwiseAnd", [bits, tag + "_mask3ff"], [tag + "_m_masked"])
    b.add("Cast", [tag + "_m_masked"], [tag + "_mant"], to=TensorProto.INT32)
    b.add("Greater", [tag + "_exp", tag + "_i0"], [tag + "_normal"])
    b.add("Cast", [tag + "_normal"], [tag + "_normal_i"], to=TensorProto.INT32)
    b.const(tag + "_1024", "i32", [1024])
    b.add("Mul", [tag + "_normal_i", tag + "_1024"], [tag + "_fold"])
    b.add("Add", [tag + "_mant", tag + "_fold"], [tag + "_num"])
    b.tensor(tag + "_pow2", np.array([2.0 ** (e - 25) for e in range(32)],
                                     dtype=np.float32))
    b.add("Gather", [tag + "_pow2", tag + "_exp"], [tag + "_scale"], axis=0)
    b.add("Cast", [tag + "_num"], [tag + "_num_f"], to=TensorProto.FLOAT)
    b.add("Mul", [tag + "_num_f", tag + "_scale"], [tag + "_normal_v"])
    b.const(tag + "_sub_scale", "fp32", [2.0 ** -24])
    b.add("Mul", [tag + "_num_f", tag + "_sub_scale"], [tag + "_sub_v"])
    b.add("Equal", [tag + "_exp", tag + "_i0"], [tag + "_is_sub"])
    b.add("Where", [tag + "_is_sub", tag + "_sub_v", tag + "_normal_v"],
          [tag + "_value"])
    b.const(tag + "_exp31", "i32", [31])
    b.add("Equal", [tag + "_exp", tag + "_exp31"], [tag + "_is_inf"])
    b.const(tag + "_zero_f", "fp32", [0.0])
    b.add("Where", [tag + "_is_inf", tag + "_zero_f", tag + "_value"],
          [tag + "_finite"])
    b.const(tag + "_shift15", "u32", [15])
    b.add("BitShift", [bits, tag + "_shift15"], [tag + "_s"],
          direction="RIGHT")
    b.const(tag + "_u1", "u32", [1])
    b.add("BitwiseAnd", [tag + "_s", tag + "_u1"], [tag + "_sign_u"])
    b.add("Cast", [tag + "_sign_u"], [tag + "_sign"], to=TensorProto.INT32)
    b.add("Equal", [tag + "_sign", tag + "_i1"], [tag + "_neg"])
    b.add("Neg", [tag + "_finite"], [tag + "_neg_v"])
    b.add("Where", [tag + "_neg", tag + "_neg_v", tag + "_finite"],
          [tag + "_out"])
    return tag + "_out"


def _warp_init_sample_graph(builder, plan, tag, tex, pos_x, pos_y):
    """Bilinear four-tap of a BGRA8 texture, exactly as the replay.

    `tex` is a [height*width, 4] RGBA uint8 tensor. This mirrors the sampler
    model `recon._warp_init_sample` uses (`uv32exact` in trace/fit_coords.py):
    a float32 uv (`pos * float32(1 / size)`), an exact float64 `uv * size - 0.5`
    texel coordinate, and 8-bit subtexel weights that are ONNX `Round`
    (half-to-even) on `frac * 256`, clipped to 0..256. The four products are
    summed in int32 and divided once by 65536, so the graph and the reference
    kernel produce identical float64 units.
    """
    b = builder
    height, width = plan["oh"], plan["ow"]
    b.const(tag + "_inv_w", "fp32",
            [np.float32(1.0) / np.float32(width)])
    b.const(tag + "_inv_h", "fp32",
            [np.float32(1.0) / np.float32(height)])
    b.add("Mul", [pos_x, tag + "_inv_w"], [tag + "_uv_x"])
    b.add("Mul", [pos_y, tag + "_inv_h"], [tag + "_uv_y"])
    b.add("Cast", [tag + "_uv_x"], [tag + "_uv_xd"], to=TensorProto.DOUBLE)
    b.add("Cast", [tag + "_uv_y"], [tag + "_uv_yd"], to=TensorProto.DOUBLE)
    b.const(tag + "_w_d", "fp64", [float(width)])
    b.const(tag + "_h_d", "fp64", [float(height)])
    b.add("Mul", [tag + "_uv_xd", tag + "_w_d"], [tag + "_tex_xu"])
    b.add("Mul", [tag + "_uv_yd", tag + "_h_d"], [tag + "_tex_yu"])
    b.const(tag + "_half_d", "fp64", [0.5])
    b.add("Sub", [tag + "_tex_xu", tag + "_half_d"], [tag + "_tex_x"])
    b.add("Sub", [tag + "_tex_yu", tag + "_half_d"], [tag + "_tex_y"])
    b.add("Floor", [tag + "_tex_x"], [tag + "_x0"])
    b.add("Floor", [tag + "_tex_y"], [tag + "_y0"])
    b.add("Sub", [tag + "_tex_x", tag + "_x0"], [tag + "_fxf"])
    b.add("Sub", [tag + "_tex_y", tag + "_y0"], [tag + "_fyf"])
    b.const(tag + "_256_d", "fp64", [256.0])
    b.add("Mul", [tag + "_fxf", tag + "_256_d"], [tag + "_wx"])
    b.add("Mul", [tag + "_fyf", tag + "_256_d"], [tag + "_wy"])
    b.add("Round", [tag + "_wx"], [tag + "_wxr"])
    b.add("Round", [tag + "_wy"], [tag + "_wyr"])
    b.const(tag + "_i0_d", "fp64", [0.0])
    b.const(tag + "_i256_d", "fp64", [256.0])
    b.add("Clip", [tag + "_wxr", tag + "_i0_d", tag + "_i256_d"],
          [tag + "_wxc"])
    b.add("Clip", [tag + "_wyr", tag + "_i0_d", tag + "_i256_d"],
          [tag + "_wyc"])
    b.add("Cast", [tag + "_wxc"], [tag + "_ix"], to=TensorProto.INT32)
    b.add("Cast", [tag + "_wyc"], [tag + "_iy"], to=TensorProto.INT32)
    b.const(tag + "_i0", "i32", [0])
    b.const(tag + "_i256", "i32", [256])
    b.const(tag + "_xmax", "i32", [width - 1])
    b.const(tag + "_ymax", "i32", [height - 1])
    b.add("Cast", [tag + "_x0"], [tag + "_x0i"], to=TensorProto.INT32)
    b.add("Cast", [tag + "_y0"], [tag + "_y0i"], to=TensorProto.INT32)
    b.add("Clip", [tag + "_x0i", tag + "_i0", tag + "_xmax"], [tag + "_cx0"])
    b.add("Clip", [tag + "_y0i", tag + "_i0", tag + "_ymax"], [tag + "_cy0"])
    b.const(tag + "_i1", "i32", [1])
    b.add("Add", [tag + "_cx0", tag + "_i1"], [tag + "_cx0p"])
    b.add("Add", [tag + "_cy0", tag + "_i1"], [tag + "_cy0p"])
    b.add("Clip", [tag + "_cx0p", tag + "_i0", tag + "_xmax"], [tag + "_cx1"])
    b.add("Clip", [tag + "_cy0p", tag + "_i0", tag + "_ymax"], [tag + "_cy1"])
    b.add("Sub", [tag + "_i256", tag + "_ix"], [tag + "_iwx"])
    b.add("Sub", [tag + "_i256", tag + "_iy"], [tag + "_iwy"])
    b.add("Mul", [tag + "_iwy", tag + "_iwx"], [tag + "_w00"])
    b.add("Mul", [tag + "_iwy", tag + "_ix"], [tag + "_w01"])
    b.add("Mul", [tag + "_iy", tag + "_iwx"], [tag + "_w10"])
    b.add("Mul", [tag + "_iy", tag + "_ix"], [tag + "_w11"])
    b.const(tag + "_ow", "i32", [width])
    b.const(tag + "_ax2", "i64", [2])
    taps = []
    for cy, cx, weight, name in ((tag + "_cy0", tag + "_cx0", tag + "_w00", "00"),
                                 (tag + "_cy0", tag + "_cx1", tag + "_w01", "01"),
                                 (tag + "_cy1", tag + "_cx0", tag + "_w10", "10"),
                                 (tag + "_cy1", tag + "_cx1", tag + "_w11", "11")):
        b.add("Mul", [cy, tag + "_ow"], [tag + "_row" + name])
        b.add("Add", [tag + "_row" + name, cx], [tag + "_cell" + name])
        b.add("Gather", [tex, tag + "_cell" + name], [tag + "_tap" + name],
              axis=0)
        b.add("Cast", [tag + "_tap" + name], [tag + "_tap_i" + name],
              to=TensorProto.INT32)
        b.add("Unsqueeze", [weight, tag + "_ax2"], [tag + "_wu" + name])
        b.add("Mul", [tag + "_tap_i" + name, tag + "_wu" + name],
              [tag + "_term" + name])
        taps.append(tag + "_term" + name)
    b.add("Add", [taps[0], taps[1]], [tag + "_s01"])
    b.add("Add", [tag + "_s01", taps[2]], [tag + "_s012"])
    b.add("Add", [tag + "_s012", taps[3]], [tag + "_total"])
    b.add("Cast", [tag + "_total"], [tag + "_total_d"], to=TensorProto.DOUBLE)
    b.const(tag + "_denom", "fp64", [65536.0])
    b.add("Div", [tag + "_total_d", tag + "_denom"], [tag + "_out"])
    return tag + "_out"


def _lane_of(builder, tag, source, index):
    """One float channel of a [th, tw, 4] tensor, squeezed to [th, tw]."""
    builder.const(tag + "_pick", "i64", [index])
    builder.add("Gather", [source, tag + "_pick"], [tag + "_slice"], axis=2)
    builder.const(tag + "_axis2", "i64", [2])
    builder.add("Squeeze", [tag + "_slice", tag + "_axis2"], [tag + "_out"])
    return tag + "_out"


def _slice_channels(builder, tag, source, start, end):
    """A [th, tw, start:end] channel span of a [th, tw, 4] tensor."""
    builder.const(tag + "_start", "i64", [start])
    builder.const(tag + "_end", "i64", [end])
    builder.const(tag + "_axis", "i64", [2])
    builder.const(tag + "_step", "i64", [1])
    builder.add("Slice", [source, tag + "_start", tag + "_end", tag + "_axis",
                          tag + "_step"], [tag + "_out"])
    return tag + "_out"


def _warp_init_unorm8(builder, tag, value):
    """clip(floor(x + 0.5), 0, 255) in float64, then uint8 (the replay's unorm8)."""
    b = builder
    b.const(tag + "_half", "fp64", [0.5])
    b.add("Add", [value, tag + "_half"], [tag + "_shifted"])
    b.add("Floor", [tag + "_shifted"], [tag + "_floored"])
    b.const(tag + "_lo", "fp64", [0.0])
    b.const(tag + "_hi", "fp64", [255.0])
    b.add("Clip", [tag + "_floored", tag + "_lo", tag + "_hi"], [tag + "_clipped"])
    b.add("Cast", [tag + "_clipped"], [tag + "_out"], to=TensorProto.UINT8)
    return tag + "_out"


def build_warp_init(plan, prefix=""):
    """The warp + flow-init stage as one standard-op graph.

    Inputs are the captured t0/t2/t3/t4/t5/t6/t8 data; weights, scale/bias and
    the root constants are initializers. Outputs are u0 (16 int8 features per
    cell) and the u1/u2/u3 colour planes at 2x the cell grid.
    """
    p = plan
    tw, th, ow, oh = p["tw"], p["th"], p["ow"], p["oh"]
    cells = tw * th
    b = Builder(p, prefix)
    k = p["constants"]
    for name in ("span_x", "span_y", "uv_x", "uv_y", "motion_scale", "ratio",
                 "lo_ax", "lo_ay", "hi_ax", "hi_ay",
                 "lo_bx", "lo_by", "hi_bx", "hi_by"):
        b.const("wi_" + name, "fp32", [np.float32(k[name])])
    b.const("wi_half_f", "fp32", [0.5])
    b.const("wi_two", "fp32", [2.0])
    b.const("wi_one_f", "fp32", [1.0])
    b.const("wi_zero_f", "fp32", [0.0])
    b.const("wi_255_d", "fp64", [255.0])
    b.const("wi_zero_d", "fp64", [0.0])

    # Textures are reshaped to [pixels, 4] and swizzled BGRA -> RGBA once.
    b.const("wi_tex_shape", "i64", [ow * oh, 4])
    b.const("wi_swizzle", "i64", [2, 1, 0, 3])
    for source, tag in (("x1", "wi_t2"), ("x2", "wi_t3")):
        b.add("Reshape", [source, "wi_tex_shape"], [tag + "_packed"])
        b.add("Gather", [tag + "_packed", "wi_swizzle"], [tag + "_rgba"],
              axis=1)
    b.const("wi_t0_shape", "i64", [p["plane_h"] * p["stride"], 4])
    b.add("Reshape", ["x0", "wi_t0_shape"], ["wi_t0"])
    b.const("wi_t4_shape", "i64", [cells, 2])
    b.add("Reshape", ["x3", "wi_t4_shape"], ["wi_t4"])

    tx = np.broadcast_to(np.arange(tw, dtype=np.int32)[None, :], (th, tw))
    ty = np.broadcast_to(np.arange(th, dtype=np.int32)[:, None], (th, tw))
    b.tensor("wi_tx", np.ascontiguousarray(tx))
    b.tensor("wi_ty", np.ascontiguousarray(ty))
    b.const("wi_i0", "i32", [0])
    b.const("wi_i1", "i32", [1])
    b.const("wi_i2", "i32", [2])
    b.const("wi_minus1", "i32", [-1])
    b.const("wi_twmax", "i32", [tw - 1])
    b.const("wi_thmax", "i32", [th - 1])
    b.const("wi_tw", "i32", [tw])

    # -- cell search over the packed depth/flow planes ---------------------
    b.const("wi_mask7fff", "u32", [0x7FFF])
    b.const("wi_maskffff", "u32", [0xFFFF])
    b.const("wi_shift16", "u32", [16])
    b.const("wi_u0", "u32", [0])
    b.const("wi_32768", "fp32", [32768.0])
    b.const("wi_1000", "fp32", [1000.0])

    def depth(tag, index):
        b.add("Gather", ["x6", index], [tag + "_word"], axis=0)
        b.add("BitShift", [tag + "_word", "wi_shift16"], [tag + "_hi"],
              direction="RIGHT")
        b.add("BitwiseAnd", [tag + "_hi", "wi_mask7fff"], [tag + "_packed"])
        b.add("Cast", [tag + "_packed"], [tag + "_packed_f"],
              to=TensorProto.FLOAT)
        b.add("Div", [tag + "_packed_f", "wi_32768"], [tag + "_ratio"])
        b.add("Sub", ["wi_one_f", tag + "_ratio"], [tag + "_value"])
        b.add("Gather", ["x4", index], [tag + "_t5"], axis=0)
        b.add("Sub", [tag + "_t5", "wi_1000"], [tag + "_sentinel"])
        b.add("Equal", [tag + "_word", "wi_u0"], [tag + "_empty"])
        b.add("Where", [tag + "_empty", tag + "_sentinel", tag + "_value"],
              [tag + "_out"])
        return tag + "_out"

    index = p["depth_index"]
    for name in ("A", "B", "C", "D", "centre"):
        b.tensor("wi_idx_" + name, index[name])
    dep_a = depth("wi_da", "wi_idx_A")
    dep_b = depth("wi_db", "wi_idx_B")
    dep_c = depth("wi_dc", "wi_idx_C")
    dep_d = depth("wi_dd", "wi_idx_D")
    dep_centre = depth("wi_dm", "wi_idx_centre")
    for name, value in (("a", dep_a), ("b", dep_b), ("c", dep_c), ("d", dep_d)):
        b.add("Neg", [value], ["wi_neg_" + name])
    b.add("Max", ["wi_neg_a", "wi_neg_b"], ["wi_top"])
    b.add("Max", ["wi_neg_c", "wi_neg_d"], ["wi_bottom"])
    b.add("Greater", ["wi_top", "wi_bottom"], ["wi_row_sel"])
    b.add("Less", [dep_a, dep_b], ["wi_ab"])
    b.add("Where", ["wi_ab", "wi_minus1", "wi_i1"], ["wi_dx_top"])
    b.add("Less", [dep_c, dep_d], ["wi_cd"])
    b.add("Where", ["wi_cd", "wi_minus1", "wi_i1"], ["wi_dx_bottom"])
    b.add("Where", ["wi_row_sel", "wi_dx_top", "wi_dx_bottom"], ["wi_dx"])
    b.add("Where", ["wi_row_sel", "wi_minus1", "wi_i1"], ["wi_dy"])
    b.add("Max", ["wi_top", "wi_bottom"], ["wi_best"])
    b.add("Neg", [dep_centre], ["wi_neg_centre"])
    b.add("Greater", ["wi_best", "wi_neg_centre"], ["wi_keep"])
    b.add("Where", ["wi_keep", "wi_dx", "wi_i0"], ["wi_step_x"])
    b.add("Where", ["wi_keep", "wi_dy", "wi_i0"], ["wi_step_y"])
    b.add("Add", ["wi_tx", "wi_step_x"], ["wi_xp_raw"])
    b.add("Add", ["wi_ty", "wi_step_y"], ["wi_yp_raw"])
    b.add("Clip", ["wi_xp_raw", "wi_i0", "wi_twmax"], ["wi_xp"])
    b.add("Clip", ["wi_yp_raw", "wi_i0", "wi_thmax"], ["wi_yp"])
    b.add("Mul", ["wi_yp", "wi_tw"], ["wi_yp_row"])
    b.add("Add", ["wi_yp_row", "wi_xp"], ["wi_pick_cell"])
    b.add("Gather", ["x6", "wi_pick_cell"], ["wi_word0"], axis=0)
    # An empty picked word is a full fallback: skip the flow decode and the
    # gate, use 0.5*t4 and set flag 0 (DXIL %406 -> %460 / %483).
    b.add("Equal", ["wi_word0", "wi_u0"], ["wi_word0_empty"])
    b.const("wi_plane_offset", "i32", [cells])
    b.add("Add", ["wi_pick_cell", "wi_plane_offset"], ["wi_pick_plane1"])
    b.add("Gather", ["x6", "wi_pick_plane1"], ["wi_word1"], axis=0)
    b.add("BitwiseAnd", ["wi_word0", "wi_maskffff"], ["wi_f0_bits"])
    f0 = _half_bits_to_float(b, "wi_f0", "wi_f0_bits")
    b.add("BitwiseAnd", ["wi_word1", "wi_maskffff"], ["wi_f1_bits"])
    f1_raw = _half_bits_to_float(b, "wi_f1", "wi_f1_bits")
    b.add("Equal", ["wi_word1", "wi_u0"], ["wi_f1_empty"])
    b.add("Where", ["wi_f1_empty", "wi_zero_f", f1_raw], ["wi_f1"])
    f1 = "wi_f1"
    b.add("Gather", ["wi_t4", "wi_pick_cell"], ["wi_t4_at"], axis=0)
    fb0 = _lane_of(b, "wi_fb0", "wi_t4_at", 0)
    fb1 = _lane_of(b, "wi_fb1", "wi_t4_at", 1)
    b.add("Mul", ["wi_half_f", fb0], ["wi_fb0_half"])
    b.add("Mul", ["wi_half_f", fb1], ["wi_fb1_half"])

    # -- per-subpixel lane -------------------------------------------------
    lanes_q = []
    lane_u1, lane_u2, lane_u3 = [], [], []
    for number, lane in enumerate(p["lanes"]):
        tag = "wi_L%d" % number
        ax, ay = lane["offset_x"], lane["offset_y"]
        b.const(tag + "_ax", "i32", [ax])
        b.const(tag + "_ay", "i32", [ay])
        b.add("Mul", ["wi_tx", "wi_i2"], [tag + "_2tx"])
        b.add("Mul", ["wi_ty", "wi_i2"], [tag + "_2ty"])
        b.add("Add", [tag + "_2tx", tag + "_ax"], [tag + "_ix"])
        b.add("Add", [tag + "_2ty", tag + "_ay"], [tag + "_iy"])
        b.add("Cast", [tag + "_ix"], [tag + "_ixf"], to=TensorProto.FLOAT)
        b.add("Cast", [tag + "_iy"], [tag + "_iyf"], to=TensorProto.FLOAT)
        b.add("Add", [tag + "_ixf", "wi_half_f"], [tag + "_px"])
        b.add("Add", [tag + "_iyf", "wi_half_f"], [tag + "_py"])

        # The shader's manual t0 bilinear (float32, DXIL operation order).
        b.add("Mul", [tag + "_ixf", "wi_span_x"], [tag + "_u"])
        b.add("Mul", [tag + "_iyf", "wi_span_y"], [tag + "_v"])
        b.add("Floor", [tag + "_u"], [tag + "_fu"])
        b.add("Floor", [tag + "_v"], [tag + "_fv"])
        b.add("Sub", [tag + "_u", tag + "_fu"], [tag + "_fx"])
        b.add("Sub", [tag + "_v", tag + "_fv"], [tag + "_fy"])
        b.add("Cast", [tag + "_fu"], [tag + "_fui"], to=TensorProto.INT32)
        b.add("Cast", [tag + "_fv"], [tag + "_fvi"], to=TensorProto.INT32)
        b.const(tag + "_pwmax", "i32", [p["plane_w"] - 1])
        b.const(tag + "_phmax", "i32", [p["plane_h"] - 1])
        b.add("Clip", [tag + "_fui", "wi_i0", tag + "_pwmax"], [tag + "_ex"])
        b.add("Clip", [tag + "_fvi", "wi_i0", tag + "_phmax"], [tag + "_ey"])
        b.add("Add", [tag + "_ex", "wi_i1"], [tag + "_exp"])
        b.add("Add", [tag + "_ey", "wi_i1"], [tag + "_eyp"])
        b.add("Clip", [tag + "_exp", "wi_i0", tag + "_pwmax"], [tag + "_ex1"])
        b.add("Clip", [tag + "_eyp", "wi_i0", tag + "_phmax"], [tag + "_ey1"])
        b.const(tag + "_stride", "i32", [p["stride"]])
        grad = []
        for ey, ex, name in ((tag + "_ey", tag + "_ex", "00"),
                             (tag + "_ey", tag + "_ex1", "01"),
                             (tag + "_ey1", tag + "_ex", "10"),
                             (tag + "_ey1", tag + "_ex1", "11")):
            b.add("Mul", [ey, tag + "_stride"], [tag + "_row" + name])
            b.add("Add", [tag + "_row" + name, ex], [tag + "_cell" + name])
            b.add("Gather", ["wi_t0", tag + "_cell" + name], [tag + "_v" + name],
                  axis=0)
            grad.append(tag + "_v" + name)
        b.add("Sub", ["wi_one_f", tag + "_fx"], [tag + "_ifx"])
        b.add("Sub", ["wi_one_f", tag + "_fy"], [tag + "_ify"])
        b.add("Mul", [tag + "_ify", tag + "_ifx"], [tag + "_w00"])
        b.add("Mul", [tag + "_ify", tag + "_fx"], [tag + "_w10"])
        b.add("Mul", [tag + "_fy", tag + "_ifx"], [tag + "_w01"])
        b.add("Mul", [tag + "_fy", tag + "_fx"], [tag + "_w11"])
        b.const(tag + "_ax2", "i64", [2])
        for name in ("00", "10", "01", "11"):
            b.add("Unsqueeze", [tag + "_w" + name, tag + "_ax2"],
                  [tag + "_wu" + name])
        b.add("Mul", [grad[0], tag + "_wu00"], [tag + "_m00"])
        b.add("Mul", [grad[1], tag + "_wu10"], [tag + "_m10"])
        b.add("Mul", [grad[2], tag + "_wu01"], [tag + "_m01"])
        b.add("Mul", [grad[3], tag + "_wu11"], [tag + "_m11"])
        b.add("Add", [tag + "_m00", tag + "_m10"], [tag + "_s0"])
        b.add("Add", [tag + "_s0", tag + "_m01"], [tag + "_s1"])
        b.add("Add", [tag + "_s1", tag + "_m11"], [tag + "_motion"])

        motion_x = _lane_of(b, tag + "_mx", tag + "_motion", 0)
        motion_y = _lane_of(b, tag + "_my", tag + "_motion", 1)
        motion_z = _lane_of(b, tag + "_mz", tag + "_motion", 2)
        motion_w = _lane_of(b, tag + "_mw", tag + "_motion", 3)
        b.add("Mul", [motion_x, "wi_motion_scale"], [tag + "_mx_s"])
        b.add("Mul", [motion_y, "wi_motion_scale"], [tag + "_my_s"])
        b.add("Mul", [motion_z, "wi_motion_scale"], [tag + "_mz_s"])
        b.add("Mul", [motion_w, "wi_motion_scale"], [tag + "_mw_s"])
        b.add("Add", [tag + "_px", tag + "_mx_s"], [tag + "_pax"])
        b.add("Add", [tag + "_py", tag + "_my_s"], [tag + "_pay"])
        b.add("Add", [tag + "_px", tag + "_mz_s"], [tag + "_pbx"])
        b.add("Add", [tag + "_py", tag + "_mw_s"], [tag + "_pby"])
        b.add("Clip", [tag + "_pax", "wi_lo_ax", "wi_hi_ax"], [tag + "_posa_x"])
        b.add("Clip", [tag + "_pay", "wi_lo_ay", "wi_hi_ay"], [tag + "_posa_y"])
        b.add("Clip", [tag + "_pbx", "wi_lo_bx", "wi_hi_bx"], [tag + "_posb_x"])
        b.add("Clip", [tag + "_pby", "wi_lo_by", "wi_hi_by"], [tag + "_posb_y"])

        b.add("Mul", ["wi_two", "wi_uv_x"], [tag + "_two_uvx"])
        b.add("Mul", ["wi_two", "wi_uv_y"], [tag + "_two_uvy"])
        b.add("Mul", [tag + "_px", tag + "_two_uvx"], [tag + "_rx_raw"])
        b.add("Mul", [tag + "_py", tag + "_two_uvy"], [tag + "_ry_raw"])
        b.add("Sub", [tag + "_rx_raw", "wi_one_f"], [tag + "_rx"])
        b.add("Sub", [tag + "_ry_raw", "wi_one_f"], [tag + "_ry"])
        b.add("Abs", [tag + "_rx"], [tag + "_arx"])
        b.add("Abs", [tag + "_ry"], [tag + "_ary"])
        b.add("Max", [tag + "_arx", tag + "_ary"], [tag + "_radius"])
        b.const(tag + "_p8", "fp32", [0.8])
        b.const(tag + "_five", "fp32", [5.0])
        b.const(tag + "_p25", "fp32", [0.25])
        b.add("Sub", [tag + "_radius", tag + "_p8"], [tag + "_rad0"])
        b.add("Mul", [tag + "_rad0", tag + "_five"], [tag + "_rad1"])
        b.add("Add", [tag + "_rad1", tag + "_p25"], [tag + "_rad2"])
        b.add("Clip", [tag + "_rad2", "wi_zero_f", "wi_one_f"], [tag + "_ramp"])
        b.const(tag + "_p099", "fp32", [0.099])
        b.add("Mul", [tag + "_ramp", tag + "_p099"], [tag + "_shrink"])

        b.add("Mul", ["wi_two", f0], [tag + "_2f0"])
        b.add("Sub", [tag + "_px", tag + "_2f0"], [tag + "_qx"])
        b.add("Mul", ["wi_two", f1], [tag + "_2f1"])
        b.add("Sub", [tag + "_py", tag + "_2f1"], [tag + "_qy"])
        b.add("Mul", [tag + "_qx", tag + "_two_uvx"], [tag + "_srx"])
        b.add("Mul", [tag + "_qy", tag + "_two_uvy"], [tag + "_sry"])
        b.add("Sub", [tag + "_srx", "wi_one_f"], [tag + "_srx1"])
        b.add("Sub", [tag + "_sry", "wi_one_f"], [tag + "_sry1"])
        b.add("Abs", [tag + "_srx1"], [tag + "_asrx"])
        b.add("Abs", [tag + "_sry1"], [tag + "_asry"])
        b.add("Max", [tag + "_asrx", tag + "_asry"], [tag + "_moved"])
        b.const(tag + "_p9", "fp32", [0.9])
        b.const(tag + "_p1", "fp32", [0.1])
        b.add("Sub", [tag + "_moved", tag + "_p9"], [tag + "_c0"])
        b.add("Sub", [tag + "_c0", tag + "_shrink"], [tag + "_c1"])
        b.add("Sub", [tag + "_p1", tag + "_shrink"], [tag + "_c2"])
        b.add("Div", [tag + "_c1", tag + "_c2"], [tag + "_c3"])
        b.add("Clip", [tag + "_c3", "wi_zero_f", "wi_one_f"], [tag + "_c4"])
        b.add("Sub", ["wi_one_f", tag + "_c4"], [tag + "_confidence"])
        b.tensor(tag + "_gate_index", lane["gate_index"])
        b.add("Gather", ["x5", tag + "_gate_index"], [tag + "_gate"], axis=0)
        b.add("Less", [tag + "_confidence", tag + "_gate"], [tag + "_fallback"])
        b.add("Or", [tag + "_fallback", "wi_word0_empty"],
              [tag + "_fallback_all"])
        b.add("Where", [tag + "_fallback_all", "wi_fb0_half", f0],
              [tag + "_fxs"])
        b.add("Where", [tag + "_fallback_all", "wi_fb1_half", f1],
              [tag + "_fys"])

        b.add("Mul", ["wi_two", "wi_ratio"], [tag + "_two_ratio"])
        b.add("Mul", [tag + "_two_ratio", tag + "_fxs"], [tag + "_fdx"])
        b.add("Add", [tag + "_px", tag + "_fdx"], [tag + "_pcx_raw"])
        b.add("Mul", [tag + "_two_ratio", tag + "_fys"], [tag + "_fdy"])
        b.add("Add", [tag + "_py", tag + "_fdy"], [tag + "_pcy_raw"])
        b.add("Clip", [tag + "_pcx_raw", "wi_lo_ax", "wi_hi_ax"],
              [tag + "_posc_x"])
        b.add("Clip", [tag + "_pcy_raw", "wi_lo_ay", "wi_hi_ay"],
              [tag + "_posc_y"])
        b.add("Mul", ["wi_two", tag + "_fxs"], [tag + "_2fx"])
        b.add("Sub", [tag + "_px", tag + "_2fx"], [tag + "_pdx_raw"])
        b.add("Mul", ["wi_two", tag + "_fys"], [tag + "_2fy"])
        b.add("Sub", [tag + "_py", tag + "_2fy"], [tag + "_pdy_raw"])
        b.add("Clip", [tag + "_pdx_raw", "wi_lo_bx", "wi_hi_bx"],
              [tag + "_posd_x"])
        b.add("Clip", [tag + "_pdy_raw", "wi_lo_by", "wi_hi_by"],
              [tag + "_posd_y"])

        centre2 = _warp_init_sample_graph(b, p, tag + "_c2", "wi_t2_rgba",
                                          tag + "_posc_x", tag + "_posc_y")
        centre3 = _warp_init_sample_graph(b, p, tag + "_c3", "wi_t3_rgba",
                                          tag + "_posd_x", tag + "_posd_y")
        warp2 = _warp_init_sample_graph(b, p, tag + "_w2", "wi_t2_rgba",
                                        tag + "_posa_x", tag + "_posa_y")
        warp3 = _warp_init_sample_graph(b, p, tag + "_w3", "wi_t3_rgba",
                                        tag + "_posb_x", tag + "_posb_y")
        b.add("Clip", [warp2, "wi_zero_d", "wi_255_d"], [tag + "_w2c"])
        b.add("Clip", [warp3, "wi_zero_d", "wi_255_d"], [tag + "_w3c"])

        b.tensor(tag + "_self_index", lane["self_index"])
        b.add("Gather", ["wi_t2_rgba", tag + "_self_index"], [tag + "_self2"],
              axis=0)
        b.add("Gather", ["wi_t3_rgba", tag + "_self_index"], [tag + "_self3"],
              axis=0)
        b.add("Cast", [tag + "_self2"], [tag + "_self2f"], to=TensorProto.FLOAT)
        b.add("Cast", [tag + "_self3"], [tag + "_self3f"], to=TensorProto.FLOAT)
        b.add("Where", [tag + "_fallback_all", "wi_zero_f", "wi_one_f"],
              [tag + "_flag"])
        b.add("Cast", [tag + "_flag"], [tag + "_flag_d"], to=TensorProto.DOUBLE)
        b.add("Mul", [tag + "_flag_d", "wi_255_d"], [tag + "_flag_scaled"])
        b.const(tag + "_flag_axis", "i64", [2])
        b.add("Unsqueeze", [tag + "_flag_scaled", tag + "_flag_axis"],
              [tag + "_flag255"])

        parts = [
            _slice_channels(b, tag + "_fc2", centre2, 0, 3),
            _slice_channels(b, tag + "_fc3", centre3, 0, 3),
            _slice_channels(b, tag + "_fw2", tag + "_w2c", 0, 3),
            _slice_channels(b, tag + "_fw3", tag + "_w3c", 0, 3),
            tag + "_flag255",
        ]
        for source, name in ((tag + "_self2f", "s2"), (tag + "_self3f", "s3")):
            b.add("Cast", [source], [tag + "_" + name + "d"], to=TensorProto.DOUBLE)
            parts.append(_slice_channels(b, tag + "_f" + name,
                                         tag + "_" + name + "d", 0, 3))
        b.add("Concat", parts, [tag + "_features"], axis=2)
        b.add("Floor", [tag + "_features"], [tag + "_trunc"])
        b.const(tag + "_128", "fp64", [128.0])
        b.add("Sub", [tag + "_trunc", tag + "_128"], [tag + "_centred"])
        b.const(tag + "_lo8", "fp64", [-128.0])
        b.const(tag + "_hi8", "fp64", [127.0])
        b.add("Clip", [tag + "_centred", tag + "_lo8", tag + "_hi8"],
              [tag + "_clipped"])
        b.add("Cast", [tag + "_clipped"], [tag + "_q"], to=TensorProto.INT8)
        lanes_q.append(tag + "_q")

        # u1/u2/u3: three warped channels plus one centre channel.
        one = [_slice_channels(b, tag + "_o1", tag + "_w2c", 0, 1),
               _slice_channels(b, tag + "_o2", tag + "_w2c", 1, 2),
               _slice_channels(b, tag + "_o3", tag + "_w2c", 2, 3),
               _slice_channels(b, tag + "_o4", centre3, 0, 1)]
        b.add("Concat", one, [tag + "_one"], axis=2)
        b.add("Gather", [tag + "_one", "wi_swizzle"],
              [tag + "_one_store"], axis=2)
        lane_u1.append(_warp_init_unorm8(b, tag + "_n1", tag + "_one_store"))
        two = [_slice_channels(b, tag + "_p1", tag + "_w3c", 0, 1),
               _slice_channels(b, tag + "_p2", tag + "_w3c", 1, 2),
               _slice_channels(b, tag + "_p3", tag + "_w3c", 2, 3),
               _slice_channels(b, tag + "_p4", centre3, 1, 2)]
        b.add("Concat", two, [tag + "_two_rgb"], axis=2)
        b.add("Gather", [tag + "_two_rgb", "wi_swizzle"],
              [tag + "_two_rgb_store"], axis=2)
        lane_u2.append(_warp_init_unorm8(b, tag + "_n2", tag + "_two_rgb_store"))
        three = [_slice_channels(b, tag + "_q1", centre2, 0, 1),
                 _slice_channels(b, tag + "_q2", centre2, 1, 2),
                 _slice_channels(b, tag + "_q3", centre2, 2, 3),
                 _slice_channels(b, tag + "_q4", centre3, 2, 3)]
        b.add("Concat", three, [tag + "_three_rgb"], axis=2)
        b.add("Gather", [tag + "_three_rgb", "wi_swizzle"],
              [tag + "_three_rgb_store"], axis=2)
        lane_u3.append(_warp_init_unorm8(b, tag + "_n3", tag + "_three_rgb_store"))

    # -- 76 -> 16 folded GEMV and the output layouts -----------------------
    q_parts = []
    for number, name in enumerate(lanes_q):
        b.const("wi_q_axis%d" % number, "i64", [3])
        b.add("Unsqueeze", [name, "wi_q_axis%d" % number],
              ["wi_q_u%d" % number])
        q_parts.append("wi_q_u%d" % number)
    b.add("Concat", q_parts, ["wi_q_packed"], axis=3)
    b.const("wi_q_shape", "i64", [cells, 76])
    b.add("Reshape", ["wi_q_packed", "wi_q_shape"], ["wi_a76"])
    b.const("wi_bias_pad", "i8", np.full((cells, 4), -128, dtype=np.int8))
    b.add("Concat", ["wi_a76", "wi_bias_pad"], ["wi_a80"], axis=1)
    b.tensor("wi_weights", np.ascontiguousarray(p["weights"], dtype=np.int8))
    b.add("MatMulInteger", ["wi_a80", "wi_weights"], ["wi_acc"])
    b.add("Cast", ["wi_acc"], ["wi_acc_f"], to=TensorProto.FLOAT)
    b.tensor("wi_scale", np.ascontiguousarray(p["scale"], dtype=np.float32))
    b.tensor("wi_bias", np.ascontiguousarray(p["bias"], dtype=np.float32))
    b.add("Mul", ["wi_acc_f", "wi_scale"], ["wi_scaled"])
    b.add("Add", ["wi_scaled", "wi_bias"], ["wi_biased"])
    b.add("Floor", ["wi_biased"], ["wi_floored"])
    b.const("wi_qlo", "fp32", [0.0])
    b.const("wi_qhi", "fp32", [127.0])
    b.add("Clip", ["wi_floored", "wi_qlo", "wi_qhi"], ["wi_qclipped"])
    b.add("Cast", ["wi_qclipped"], ["wi_u0_flat"], to=TensorProto.UINT8)
    b.const("wi_u0_shape", "i64", [th, tw, 16])
    b.add("Reshape", ["wi_u0_flat", "wi_u0_shape"], ["u0"])

    # Lanes are ordered (ax, ay) = (0,0), (1,0), (0,1), (1,1); reshaping the
    # lane axis to 2x2 and transposing yields row-major output pixels.
    b.const("wi_col_shape", "i64", [th, tw, 2, 2, 4])
    b.const("wi_col_out", "i64", [oh, ow, 4])
    for planes, out in ((lane_u1, "u1"), (lane_u2, "u2"), (lane_u3, "u3")):
        parts = []
        for number, value in enumerate(planes):
            axis = "wi_c_%s_axis%d" % (out, number)
            b.const(axis, "i64", [2])
            b.add("Unsqueeze", [value, axis], ["wi_c_%s_u%d" % (out, number)])
            parts.append("wi_c_%s_u%d" % (out, number))
        b.add("Concat", parts, ["wi_c_" + out], axis=2)
        b.add("Reshape", ["wi_c_" + out, "wi_col_shape"], ["wi_r_" + out])
        b.add("Transpose", ["wi_r_" + out], ["wi_t_" + out],
              perm=[0, 2, 1, 3, 4])
        b.add("Reshape", ["wi_t_" + out, "wi_col_out"], [out])

    graph = helper.make_graph(
        b.nodes, b.name("xefg_warp_init_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(b.name("x0"), TensorProto.FLOAT,
                                       [p["t0"].size]),
         helper.make_tensor_value_info(b.name("x1"), TensorProto.UINT8,
                                       [p["t2"].size]),
         helper.make_tensor_value_info(b.name("x2"), TensorProto.UINT8,
                                       [p["t3"].size]),
         helper.make_tensor_value_info(b.name("x3"), TensorProto.FLOAT,
                                       [p["t4"].size]),
         helper.make_tensor_value_info(b.name("x4"), TensorProto.FLOAT,
                                       [p["t5"].size]),
         helper.make_tensor_value_info(b.name("x5"), TensorProto.FLOAT,
                                       [p["t6"].size]),
         helper.make_tensor_value_info(b.name("x6"), TensorProto.UINT32,
                                       [p["t8"].size])],
        [helper.make_tensor_value_info(b.name("u0"), TensorProto.UINT8,
                                       [th, tw, 16]),
         helper.make_tensor_value_info(b.name("u1"), TensorProto.UINT8,
                                       [oh, ow, 4]),
         helper.make_tensor_value_info(b.name("u2"), TensorProto.UINT8,
                                       [oh, ow, 4]),
         helper.make_tensor_value_info(b.name("u3"), TensorProto.UINT8,
                                       [oh, ow, 4])],
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def export_warp_init(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.warp_init_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "warp + flow init",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    # D3: the exported d25 is the dynamic graph. A pre-D3 artifact can still be
    # verified by naming; it has no `wi_tw` feed.
    dynamic = True
    if verify_path is None:
        from . import dynamic_warp_init
        model = dynamic_warp_init.build_dynamic(plan)
        onnx.checker.check_model(model)
        name = "d%04d_warp_init" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)
        dynamic = any(value.name.endswith("wi_tw")
                      for value in model.graph.input)

    if dynamic:
        from . import dynamic_warp_init
        feed_values = dynamic_warp_init.feeds(plan)
    else:
        feed_values = {"x0": plan["t0"], "x1": plan["t2"], "x2": plan["t3"],
                       "x3": plan["t4"], "x4": plan["t5"], "x5": plan["t6"],
                       "x6": plan["t8"]}
    feeds = feed_values
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0", "u1", "u2", "u3"], feeds)
    expected = plan["expected"]
    mismatch = 0
    max_diff = 0
    for value, reference in zip(got, expected):
        diff = (value.astype(np.int16)
                - reference.reshape(value.shape).astype(np.int16))
        mismatch += int((diff != 0).sum())
        max_diff = max(max_diff, int(np.abs(diff).max()))
    total = int(sum(reference.size for reference in expected))

    # The lowering's own gate: the graph must reproduce the reference kernel
    # exactly. The remaining difference against the capture is the documented
    # texture-filter residual and is bounded, not exact.
    capture = {}
    _replay, error = recon.reconstruct_warp_init(run_dir, dispatch, capture)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "warp + flow init",
                "error": "reference replay failed: %s" % error}
    replay_mismatch = 0
    for slot, value in zip(("u0", "u1", "u2", "u3"), got):
        diff = value.astype(np.int16) - capture[slot].astype(np.int16)
        replay_mismatch += int((diff != 0).sum())

    content = d25_bound.run_content(run_dir)
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "warp + flow init",
        "template": plan["template"],
        "bytes": total,
        "mismatch": mismatch,
        "max_diff": max_diff,
        "replay_mismatch": replay_mismatch,
        "width": plan["ow"],
        "height": plan["oh"],
        "content": content,
        "bound_bytes": d25_bound.aggregate_limit(
            total, plan["ow"], plan["oh"], content=content),
        "dynamic": dynamic,
    }
    result = _finish_result(result, path, feeds, got, expected, model)
    result["status"] = "BOUND" if result_passes(result) else "FAILED"
    return result

def _splat_window(builder, tag, source, plan):
    """Apply the shader's integer SRV offsets with zero fill.

    `source` is the full texture in FLOAT. textureLoad returns zero outside the
    texture, which is Pad(0) followed by the offset Slice.
    """
    tex_w, tex_h = plan["tex_w"], plan["tex_h"]
    src_w, src_h = plan["src_w"], plan["src_h"]
    off_x, off_y = plan["offset_x"], plan["offset_y"]
    top, left = max(0, -off_y), max(0, -off_x)
    bottom = max(0, off_y + src_h - tex_h)
    right = max(0, off_x + src_w - tex_w)
    rank = len(plan["motion_tex"].shape) if tag.endswith("motion") else 2
    if any((top, left, bottom, right)):
        if rank == 3:
            pads = [top, 0, left, bottom, 0, right]
        else:
            pads = [top, left, bottom, right]
        builder.const(tag + "_pads", "i64", pads)
        builder.const(tag + "_pad_value", "fp32", [0.0])
        builder.add("Pad", [source, tag + "_pads", tag + "_pad_value"],
                    [tag + "_padded"])
        source = tag + "_padded"
        padded_h, padded_w = tex_h + top + bottom, tex_w + left + right
    else:
        padded_h, padded_w = tex_h, tex_w
    start_y, start_x = max(0, off_y), max(0, off_x)
    if (start_y, start_x) != (0, 0) or (src_h, src_w) != (padded_h, padded_w):
        builder.const(tag + "_starts", "i64", [start_y, start_x])
        builder.const(tag + "_ends", "i64", [start_y + src_h, start_x + src_w])
        builder.const(tag + "_axes", "i64", [0, 1])
        builder.const(tag + "_steps", "i64", [1, 1])
        builder.add("Slice", [source, tag + "_starts", tag + "_ends",
                              tag + "_axes", tag + "_steps"], [tag + "_window"])
        source = tag + "_window"
    return source


def _splat_half_bits(builder, tag, disp, sign_source, finite):
    """float32 -> fp16 bit pattern, in integer arithmetic.

    ONNX has no bitcast, so the shader's bit chain is reproduced arithmetically:
    exponent by floor(log2) with a one-step correction, a 10-bit mantissa, and
    the subnormal path. This was validated against `recon._f32_to_half_bits` for
    every finite fp16 value; `disp` is a motion value times 0.5, so it stays in
    that domain. The sign comes from the pre-mask motion value: ONNX Runtime's
    `Where` canonicalises -0.0 to +0.0, and the packed key needs the -0.0 sign
    bit that the static scene uses.
    """
    b = builder
    b.const(tag + "_f0", "fp32", 0.0)
    b.const(tag + "_f1", "fp32", 1.0)
    b.const(tag + "_f2", "fp32", 2.0)
    b.const(tag + "_fhalf", "fp32", 0.5)
    b.const(tag + "_invln2", "fp32", 1.4426950408889634)
    b.const(tag + "_f2m14", "fp32", 2.0 ** -14)
    b.const(tag + "_f2p24", "fp32", 2.0 ** 24)
    b.const(tag + "_f1024", "fp32", 1024.0)
    b.const(tag + "_i0", "i32", [0])
    b.const(tag + "_i1", "i32", [1])
    b.const(tag + "_i10", "i32", [10])
    b.const(tag + "_i15", "i32", [15])
    b.const(tag + "_i1024", "i32", [1024])
    b.const(tag + "_i_sign", "i32", [0x8000])

    b.add("Equal", [sign_source, tag + "_f0"], [tag + "_eq0s"])
    b.add("Less", [sign_source, tag + "_f0"], [tag + "_lt0"])
    b.add("Div", [tag + "_f1", sign_source], [tag + "_inv"])
    b.add("Less", [tag + "_inv", tag + "_f0"], [tag + "_invneg"])
    b.add("And", [tag + "_eq0s", tag + "_invneg"], [tag + "_negzero"])
    b.add("Or", [tag + "_lt0", tag + "_negzero"], [tag + "_neg0"])
    b.add("And", [finite, tag + "_neg0"], [tag + "_neg"])
    b.add("Where", [tag + "_neg", tag + "_i_sign", tag + "_i0"],
          [tag + "_sign"])

    b.add("Equal", [disp, tag + "_f0"], [tag + "_eq0"])
    b.add("Abs", [disp], [tag + "_abs"])
    b.add("Where", [tag + "_eq0", tag + "_f1", tag + "_abs"],
          [tag + "_safe"])
    b.add("Log", [tag + "_safe"], [tag + "_log"])
    b.add("Mul", [tag + "_log", tag + "_invln2"], [tag + "_log2"])
    b.add("Floor", [tag + "_log2"], [tag + "_log2f"])
    b.add("Cast", [tag + "_log2f"], [tag + "_e0"], to=TensorProto.INT32)
    b.add("Cast", [tag + "_e0"], [tag + "_e0f"], to=TensorProto.FLOAT)
    b.add("Pow", [tag + "_f2", tag + "_e0f"], [tag + "_p0"])
    b.add("Add", [tag + "_e0", tag + "_i1"], [tag + "_e0p1"])
    b.add("Cast", [tag + "_e0p1"], [tag + "_e0p1f"], to=TensorProto.FLOAT)
    b.add("Pow", [tag + "_f2", tag + "_e0p1f"], [tag + "_p1"])
    b.add("Less", [tag + "_abs", tag + "_p0"], [tag + "_low"])
    b.add("GreaterOrEqual", [tag + "_abs", tag + "_p1"], [tag + "_high"])
    b.add("Sub", [tag + "_e0", tag + "_i1"], [tag + "_e0m1"])
    b.add("Add", [tag + "_e0", tag + "_i1"], [tag + "_e0p1b"])
    b.add("Where", [tag + "_low", tag + "_e0m1",
                    tag + "_e0"], [tag + "_e_low"])
    b.add("Where", [tag + "_high", tag + "_e0p1b",
                    tag + "_e_low"], [tag + "_e"])

    b.add("GreaterOrEqual", [tag + "_abs", tag + "_f2m14"],
          [tag + "_normal"])
    b.add("Sub", [tag + "_i10", tag + "_e"], [tag + "_shift"])
    b.add("Cast", [tag + "_shift"], [tag + "_shiftf"], to=TensorProto.FLOAT)
    b.add("Pow", [tag + "_f2", tag + "_shiftf"], [tag + "_scale"])
    b.add("Mul", [tag + "_abs", tag + "_scale"], [tag + "_scaled"])
    b.add("Sub", [tag + "_scaled", tag + "_f1024"], [tag + "_m0"])
    b.add("Add", [tag + "_m0", tag + "_fhalf"], [tag + "_m1"])
    b.add("Floor", [tag + "_m1"], [tag + "_m2"])
    b.add("Cast", [tag + "_m2"], [tag + "_m"], to=TensorProto.INT32)
    b.add("GreaterOrEqual", [tag + "_m", tag + "_i1024"],
          [tag + "_carry"])
    b.add("Cast", [tag + "_carry"], [tag + "_carry_i"], to=TensorProto.INT32)
    b.add("Add", [tag + "_e", tag + "_carry_i"], [tag + "_e2"])
    b.add("Mul", [tag + "_carry_i", tag + "_i1024"], [tag + "_carry_m"])
    b.add("Sub", [tag + "_m", tag + "_carry_m"], [tag + "_m3"])
    b.add("Add", [tag + "_e2", tag + "_i15"], [tag + "_e15"])
    b.add("Mul", [tag + "_e15", tag + "_i1024"], [tag + "_e15s"])
    b.add("Add", [tag + "_e15s", tag + "_m3"], [tag + "_normal_bits"])

    b.add("Mul", [tag + "_abs", tag + "_f2p24"], [tag + "_sub0"])
    b.add("Add", [tag + "_sub0", tag + "_fhalf"], [tag + "_sub1"])
    b.add("Floor", [tag + "_sub1"], [tag + "_sub2"])
    b.add("Cast", [tag + "_sub2"], [tag + "_sub"], to=TensorProto.INT32)
    b.add("Where", [tag + "_normal", tag + "_normal_bits",
                    tag + "_sub"], [tag + "_bits0"])
    b.add("Where", [tag + "_eq0", tag + "_i0", tag + "_bits0"],
          [tag + "_bits1"])
    b.add("BitwiseOr", [tag + "_bits1", tag + "_sign"], [tag + "_bits"])
    return tag + "_bits"


def build_splat(plan, prefix=""):
    p = plan
    b = Builder(p, prefix)
    b.const("sp_scale_x", "fp32", float(p["scale_x"]))
    b.const("sp_scale_y", "fp32", float(p["scale_y"]))
    b.const("sp_f0", "fp32", 0.0)
    b.const("sp_f1", "fp32", 1.0)
    b.const("sp_f32767", "fp32", 32767.0)
    b.const("sp_i0_64", "i64", [0])
    b.const("sp_i65536", "i32", [65536])
    b.const("sp_i_plane", "i64", [p["dst_w"] * p["dst_h"]])
    b.const("sp_i_dstw", "i64", [p["dst_w"]])
    b.const("sp_i_dsth", "i64", [p["dst_h"]])

    # Motion and depth windows, with the shader's zero-filled SRV offsets.
    b.add("Cast", [b.name("x0")], ["sp_motion_f"], to=TensorProto.FLOAT)
    motion = _splat_window(b, "sp_motion", "sp_motion_f", p)
    b.add("Gather", [motion, "sp_mx_idx"], ["sp_mx3"], axis=2)
    b.add("Gather", [motion, "sp_my_idx"], ["sp_my3"], axis=2)
    b.const("sp_mx_idx", "i64", [0])
    b.const("sp_my_idx", "i64", [1])
    b.const("sp_axis2", "i64", [2])
    b.add("Squeeze", ["sp_mx3", "sp_axis2"], ["sp_mx"])
    b.add("Squeeze", ["sp_my3", "sp_axis2"], ["sp_my"])
    depth = _splat_window(b, "sp_depth", b.name("x1"), p)

    # finite = all four components finite.
    b.add("IsNaN", ["sp_mx"], ["sp_mx_nan"])
    b.add("IsNaN", ["sp_my"], ["sp_my_nan"])
    b.add("IsInf", ["sp_mx"], ["sp_mx_inf"])
    b.add("IsInf", ["sp_my"], ["sp_my_inf"])
    b.add("Or", ["sp_mx_nan", "sp_my_nan"], ["sp_nan"])
    b.add("Or", ["sp_mx_inf", "sp_my_inf"], ["sp_inf"])
    b.add("Or", ["sp_nan", "sp_inf"], ["sp_bad"])
    b.add("Not", ["sp_bad"], ["sp_finite"])

    b.add("Mul", ["sp_mx", "sp_scale_x"], ["sp_dx0"])
    b.add("Mul", ["sp_my", "sp_scale_y"], ["sp_dy0"])
    b.add("Where", ["sp_finite", "sp_dx0", "sp_f0"], ["sp_dx"])
    b.add("Where", ["sp_finite", "sp_dy0", "sp_f0"], ["sp_dy"])

    b.const("sp_tid_x", "fp32", np.arange(p["src_w"], dtype=np.float32)[None, :])
    b.const("sp_tid_y", "fp32", np.arange(p["src_h"], dtype=np.float32)[:, None])
    if p["src_w"] == p["dst_w"] and p["src_h"] == p["dst_h"]:
        b.add("Ceil", ["sp_dx"], ["sp_cx"])
        b.add("Ceil", ["sp_dy"], ["sp_cy"])
    else:
        b.const("sp_fhalf", "fp32", 0.5)
        b.const("sp_ratio_x", "fp32", p["dst_w"] / p["src_w"])
        b.const("sp_ratio_y", "fp32", p["dst_h"] / p["src_h"])
        b.add("Add", ["sp_tid_x", "sp_fhalf"], ["sp_tx0"])
        b.add("Add", ["sp_tx0", "sp_dx"], ["sp_tx1"])
        b.add("Mul", ["sp_tx1", "sp_ratio_x"], ["sp_tx2"])
        b.add("Sub", ["sp_tx2", "sp_fhalf"], ["sp_tx3"])
        b.add("Ceil", ["sp_tx3"], ["sp_cx"])
        b.add("Add", ["sp_tid_y", "sp_fhalf"], ["sp_ty0"])
        b.add("Add", ["sp_ty0", "sp_dy"], ["sp_ty1"])
        b.add("Mul", ["sp_ty1", "sp_ratio_y"], ["sp_ty2"])
        b.add("Sub", ["sp_ty2", "sp_fhalf"], ["sp_ty3"])
        b.add("Ceil", ["sp_ty3"], ["sp_cy"])
    b.add("Cast", ["sp_tid_x"], ["sp_tidxi"], to=TensorProto.INT64)
    b.add("Cast", ["sp_tid_y"], ["sp_tidyi"], to=TensorProto.INT64)
    b.add("Cast", ["sp_cx"], ["sp_cxi"], to=TensorProto.INT64)
    b.add("Cast", ["sp_cy"], ["sp_cyi"], to=TensorProto.INT64)
    b.add("Add", ["sp_tidxi", "sp_cxi"], ["sp_dst_x"])
    b.add("Add", ["sp_tidyi", "sp_cyi"], ["sp_dst_y"])

    # Depth key: trunc((1 - depth) * 32767), clamped, shifted by 16.
    b.add("Sub", ["sp_f1", depth], ["sp_dk0"])
    b.add("Mul", ["sp_dk0", "sp_f32767"], ["sp_dk1"])
    b.add("Clip", ["sp_dk1", "sp_f0", "sp_f32767"], ["sp_dk2"])
    b.add("IsNaN", [depth], ["sp_depth_nan"])
    b.add("Where", ["sp_depth_nan", "sp_f0", "sp_dk2"], ["sp_dk3"])
    b.add("Cast", ["sp_dk3"], ["sp_dk4"], to=TensorProto.INT32)
    b.add("Mul", ["sp_dk4", "sp_i65536"], ["sp_depth_key"])

    b.const("sp_i_high", "i32", [-2147483648])
    b.tensor("sp_u0", np.asarray([0], dtype=np.uint32))
    bits_x = _splat_half_bits(b, "sp_hx", "sp_dx", "sp_mx", "sp_finite")
    bits_y = _splat_half_bits(b, "sp_hy", "sp_dy", "sp_my", "sp_finite")
    for suffix, bits in (("x", bits_x), ("y", bits_y)):
        b.add("BitwiseOr", ["sp_depth_key", bits], ["sp_key_%s_i" % suffix])
        b.add("BitwiseOr", ["sp_key_%s_i" % suffix, "sp_i_high"],
              ["sp_key_%s_full" % suffix])
        b.add("Cast", ["sp_key_%s_full" % suffix], ["sp_key_%s" % suffix],
              to=TensorProto.UINT32)

    # Four-footprint, two-plane scatter with UMax. Invalid entries write 0 at
    # index 0; every valid key is >= 0, so max(data, 0) leaves the zeros alone.
    indices = []
    updates = []
    for ox in (0, 1):
        for oy in (0, 1):
            tag = "sp_f%d%d" % (ox, oy)
            b.const(tag + "_ox", "i64", [ox])
            b.const(tag + "_oy", "i64", [oy])
            b.add("Add", ["sp_dst_x", tag + "_ox"], [tag + "_x"])
            b.add("Add", ["sp_dst_y", tag + "_oy"], [tag + "_y"])
            b.add("GreaterOrEqual", [tag + "_x", "sp_i0_64"], [tag + "_x0"])
            b.add("Less", [tag + "_x", "sp_i_dstw"], [tag + "_x1"])
            b.add("GreaterOrEqual", [tag + "_y", "sp_i0_64"], [tag + "_y0"])
            b.add("Less", [tag + "_y", "sp_i_dsth"], [tag + "_y1"])
            b.add("And", [tag + "_x0", tag + "_x1"], [tag + "_xok"])
            b.add("And", [tag + "_y0", tag + "_y1"], [tag + "_yok"])
            b.add("And", [tag + "_xok", tag + "_yok"], [tag + "_in"])
            b.add("And", ["sp_finite", tag + "_in"], [tag + "_ok"])
            b.add("Mul", [tag + "_y", "sp_i_dstw"], [tag + "_row"])
            b.add("Add", [tag + "_row", tag + "_x"], [tag + "_flat"])
            for plane in (0, 1):
                ptag = tag + "_p%d" % plane
                b.add("Where", [tag + "_ok", tag + "_flat", "sp_i0_64"],
                      [ptag + "_idx0"])
                key = "sp_key_x" if plane == 0 else "sp_key_y"
                b.add("Where", [tag + "_ok", key, "sp_u0"],
                      [ptag + "_upd0"])
                if plane:
                    b.add("Add", [ptag + "_idx0", "sp_i_plane"],
                          [ptag + "_idx1"])
                else:
                    b.add("Identity", [ptag + "_idx0"], [ptag + "_idx1"])
                b.const(ptag + "_shape", "i64", [p["src_h"] * p["src_w"]])
                b.add("Reshape", [ptag + "_idx1", ptag + "_shape"],
                      [ptag + "_idx"])
                b.add("Reshape", [ptag + "_upd0", ptag + "_shape"],
                      [ptag + "_upd"])
                indices.append(ptag + "_idx")
                updates.append(ptag + "_upd")

    b.const("sp_scatter_shape", "i64", [p["allocation_words"]])
    b.add("ConstantOfShape", ["sp_scatter_shape"], ["sp_zeros"],
          value=numpy_helper.from_array(np.array([0], dtype=np.uint32)))
    b.add("Concat", indices, ["sp_indices_flat"], axis=0)
    b.add("Concat", updates, ["sp_updates"], axis=0)
    b.const("sp_index_shape", "i64", [-1, 1])
    b.add("Reshape", ["sp_indices_flat", "sp_index_shape"], ["sp_indices"])
    # uint32 UMax keeps the untouched tail at zero while every written key has
    # bit 31 set, so no post-scatter masking is needed.
    b.add("ScatterND", ["sp_zeros", "sp_indices", "sp_updates"], ["sp_max"],
          reduction="max")

    b.add("Cast", ["sp_max"], ["sp_key_u64"], to=TensorProto.INT64)
    b.const("sp_i256", "i64", [256])
    b.const("sp_i65536_64", "i64", [65536])
    b.const("sp_i16m", "i64", [16777216])
    b.add("Mod", ["sp_key_u64", "sp_i256"], ["sp_b0"])
    b.add("Div", ["sp_key_u64", "sp_i256"], ["sp_q8"])
    b.add("Mod", ["sp_q8", "sp_i256"], ["sp_b1"])
    b.add("Div", ["sp_key_u64", "sp_i65536_64"], ["sp_q16"])
    b.add("Mod", ["sp_q16", "sp_i256"], ["sp_b2"])
    b.add("Div", ["sp_key_u64", "sp_i16m"], ["sp_b3"])
    b.const("sp_axis1", "i64", [1])
    channels = []
    for name in ("sp_b0", "sp_b1", "sp_b2", "sp_b3"):
        b.add("Unsqueeze", [name, "sp_axis1"], [name + "_2d"])
        channels.append(name + "_2d")
    b.add("Concat", channels, ["sp_bytes2d"], axis=1)
    b.const("sp_byte_shape", "i64", [p["allocation_words"] * 4])
    b.add("Reshape", ["sp_bytes2d", "sp_byte_shape"], ["sp_bytes"])
    b.add("Cast", ["sp_bytes"], ["u0"], to=TensorProto.UINT8)

    graph = helper.make_graph(
        b.nodes, b.name("xefg_splat_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(b.name("x0"), TensorProto.FLOAT16,
                                       [p["tex_h"], p["tex_w"], 2]),
         helper.make_tensor_value_info(b.name("x1"), TensorProto.FLOAT,
                                       [p["tex_h"], p["tex_w"]])],
        [helper.make_tensor_value_info(b.name("u0"), TensorProto.UINT8,
                                       [p["allocation_words"] * 4])],
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def export_splat(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.splat_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "splat",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_splat(plan)
        onnx.checker.check_model(model)
        name = "d%04d_splat" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["motion_tex"], "x1": plan["depth_tex"]}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"].reshape(got.shape)
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "splat",
        "template": "ScatterND(max) + fp16 pack",
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
    }
    return _finish_result(result, path, feeds, got, expected, model)


def _downsample_fma(builder, out, value, factor, addend, tag):
    """One-rounding float32 FMA via a float64 intermediate.

    The downsample DXIL marks its luma multiplies/adds `fast`; the GPU
    contracts them into FMAs. Reproduce that rounding (the separate
    multiply-then-add order differs at a trunc boundary on the smooth scene).
    """
    builder.add("Cast", [value], [tag + "_a64"], to=TensorProto.DOUBLE)
    builder.add("Cast", [factor], [tag + "_f64"], to=TensorProto.DOUBLE)
    builder.add("Mul", [tag + "_a64", tag + "_f64"], [tag + "_p64"])
    builder.add("Cast", [addend], [tag + "_b64"], to=TensorProto.DOUBLE)
    builder.add("Add", [tag + "_p64", tag + "_b64"], [tag + "_s64"])
    builder.add("Cast", [tag + "_s64"], [out], to=TensorProto.FLOAT)


def _downsample_luma(builder, plan, tag, input_name):
    """BGRA8 -> Rec.709 luma in 0..255, in the DXIL operation order.

    The DXIL's `fmul fast` + `fadd fast` chain contracts to two FMAs:
    `fma(green, 0.7152, red*0.2126)` then `fma(blue, 0.0722, value)`.
    """
    builder.add("Cast", [input_name], [tag + "_f"], to=TensorProto.FLOAT)
    builder.add("Div", [tag + "_f", "ds_255"], [tag + "_n"])
    for index, name in ((0, "b"), (1, "g"), (2, "r")):
        builder.const("%s_%s_idx" % (tag, name), "i64", [index])
        builder.add("Gather", [tag + "_n", "%s_%s_idx" % (tag, name)],
                    ["%s_%s" % (tag, name)], axis=2)
    builder.add("Mul", [tag + "_r", "ds_2126"], [tag + "_r2"])
    _downsample_fma(builder, tag + "_rg", tag + "_g", "ds_7152",
                    tag + "_r2", tag + "_luma_g")
    _downsample_fma(builder, tag + "_rgb", tag + "_b", "ds_0722",
                    tag + "_rg", tag + "_luma_b")
    builder.add("Mul", [tag + "_rgb", "ds_255"], [tag + "_luma"])
    builder.const(tag + "_flat_shape", "i64",
                  [plan["src" + tag[-1]].shape[0] * plan["src" + tag[-1]].shape[1]])
    builder.add("Reshape", [tag + "_luma", tag + "_flat_shape"],
                [tag + "_luma_flat"])
    return tag + "_luma_flat"


def _downsample_sum(builder, plan, tag, luma, index_name, mask_name,
                    out_h, out_w, split_count):
    """Gather the taps and add them in the trace's order."""
    builder.const(tag + "_idx", "i64", plan[index_name])
    builder.tensor(tag + "_mask", plan[mask_name])
    builder.add("Gather", [luma, tag + "_idx"], [tag + "_taps"], axis=0)
    builder.add("Mul", [tag + "_taps", tag + "_mask"], [tag + "_masked"])
    builder.const(tag + "_split", "i64", [1] * split_count)
    names = ["%s_part%d" % (tag, i) for i in range(split_count)]
    builder.add("Split", [tag + "_masked", tag + "_split"], names, axis=2)
    total = names[0]
    for i in range(1, split_count):
        builder.add("Add", [total, names[i]], [tag + "_sum%d" % i])
        total = tag + "_sum%d" % i
    builder.const(tag + "_shape", "i64", [out_h, out_w])
    builder.add("Reshape", [total, tag + "_shape"], [tag + "_sum"])
    return tag + "_sum"


def _downsample_quantise(builder, tag, source, divisor, scale_name):
    builder.add("Mul", [source, scale_name], [tag + "_scaled"])
    builder.add("Add", [tag + "_scaled", "ds_neg128"], [tag + "_shifted"])
    builder.add("Cast", [tag + "_shifted"], [tag + "_int"], to=TensorProto.INT32)
    builder.const(tag + "_m256", "i32", [256])
    builder.add("Mod", [tag + "_int", tag + "_m256"], [tag + "_byte"])
    builder.add("Cast", [tag + "_byte"], [tag + "_u8"], to=TensorProto.UINT8)
    return tag + "_u8"


def build_downsample(plan, prefix=""):
    p = plan
    builder = Builder(p, prefix)
    for name, value in (("ds_255", 255.0), ("ds_7152", 0.7152),
                        ("ds_2126", 0.2126), ("ds_0722", 0.0722),
                        ("ds_neg128", -128.0), ("ds_1_16", 1.0 / 16.0),
                        ("ds_1_256", 1.0 / 256.0)):
        builder.const(name, "fp32", value)

    luma0 = _downsample_luma(builder, p, "ds0", builder.name("x0"))
    fine_sum0 = _downsample_sum(builder, p, "ds0", luma0, "index0", "mask0",
                                p["fine_h"], p["fine_w"], 16)
    fine0 = _downsample_quantise(builder, "ds0f", fine_sum0, 16, "ds_1_16")

    builder.const("ds_fold", "i64", p["fold"])
    builder.const("ds0_fold_shape", "i64", [p["coarse_h"], p["coarse_w"]])
    builder.const("ds0_fold_split", "i64", [1] * 16)
    # Flatten the fine sums first so the fold index is a flat cell index.
    builder.const("ds0_sum_flat_shape", "i64", [p["fine_h"] * p["fine_w"]])
    builder.add("Reshape", ["ds0_sum", "ds0_sum_flat_shape"], ["ds0_sum_flat"])
    builder.add("Gather", ["ds0_sum_flat", "ds_fold"], ["ds0_fold_taps"], axis=0)
    builder.add("Split", ["ds0_fold_taps", "ds0_fold_split"],
                ["ds0_c%d" % i for i in range(16)], axis=2)
    total = "ds0_c0"
    for i in range(1, 16):
        builder.add("Add", [total, "ds0_c%d" % i], ["ds0_csum%d" % i])
        total = "ds0_csum%d" % i
    builder.add("Reshape", [total, "ds0_fold_shape"], ["ds0_csum"])
    coarse0 = _downsample_quantise(builder, "ds0c", "ds0_csum", 256, "ds_1_256")

    luma1 = _downsample_luma(builder, p, "ds1", builder.name("x1"))
    fine_sum1 = _downsample_sum(builder, p, "ds1", luma1, "index1", "mask1",
                                p["fine_h"], p["fine_w"], 16)
    fine1 = _downsample_quantise(builder, "ds1f", fine_sum1, 16, "ds_1_16")
    builder.const("ds1_sum_flat_shape", "i64", [p["fine_h"] * p["fine_w"]])
    builder.add("Reshape", ["ds1_sum", "ds1_sum_flat_shape"], ["ds1_sum_flat"])
    builder.add("Gather", ["ds1_sum_flat", "ds_fold"], ["ds1_fold_taps"], axis=0)
    builder.add("Split", ["ds1_fold_taps", "ds0_fold_split"],
                ["ds1_c%d" % i for i in range(16)], axis=2)
    total = "ds1_c0"
    for i in range(1, 16):
        builder.add("Add", [total, "ds1_c%d" % i], ["ds1_csum%d" % i])
        total = "ds1_csum%d" % i
    builder.const("ds1_fold_shape", "i64", [p["coarse_h"], p["coarse_w"]])
    builder.add("Reshape", [total, "ds1_fold_shape"], ["ds1_csum"])
    coarse1 = _downsample_quantise(builder, "ds1c", "ds1_csum", 256, "ds_1_256")

    def pack(name, a, b, height, width, alloc_h, alloc_w):
        builder.const(name + "_a_axis", "i64", [2])
        builder.const(name + "_b_axis", "i64", [2])
        builder.add("Unsqueeze", [a, name + "_a_axis"], [name + "_a"])
        builder.add("Unsqueeze", [b, name + "_b_axis"], [name + "_b"])
        builder.add("Concat", [name + "_a", name + "_b"], [name + "_2ch"],
                    axis=2)
        builder.const(name + "_pads", "i64",
                      [0, 0, 0, alloc_h - height, alloc_w - width, 0])
        builder.const(name + "_zero", "u8", [0])
        builder.add("Pad", [name + "_2ch", name + "_pads", name + "_zero"],
                    [name + "_out"])
        return name + "_out"

    out0 = pack("ds_u0", coarse0, coarse1, p["coarse_h"], p["coarse_w"],
                p["alloc0"][0], p["alloc0"][1])
    out1 = pack("ds_u1", fine0, fine1, p["fine_h"], p["fine_w"],
                p["alloc1"][0], p["alloc1"][1])

    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_downsample_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(
            builder.name("x0"), TensorProto.UINT8,
            list(p["src0"].shape)),
         helper.make_tensor_value_info(
            builder.name("x1"), TensorProto.UINT8,
            list(p["src1"].shape))],
        [helper.make_tensor_value_info(builder.name(out0), TensorProto.UINT8,
                                       [p["alloc0"][0], p["alloc0"][1], 2]),
         helper.make_tensor_value_info(builder.name(out1), TensorProto.UINT8,
                                       [p["alloc1"][0], p["alloc1"][1], 2])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_downsample(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.downsample_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "downsample",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_downsample(plan)
        onnx.checker.check_model(model)
        name = "d%04d_downsample" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["src0"], "x1": plan["src1"]}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run([model.graph.output[0].name,
                       model.graph.output[1].name], feeds)
    expected = [plan["expected0"], plan["expected1"]]
    allocs = [plan["alloc0"], plan["alloc1"]]
    total = sum(int(e.size) for e in expected)
    mismatch = 0
    max_diff = 0
    for value, ref, alloc in zip(got, expected, allocs):
        ref = ref.reshape(alloc[0], alloc[1], 2)
        diff = value.astype(np.int16) - ref.astype(np.int16)
        mismatch += int((diff != 0).sum())
        if diff.size:
            max_diff = max(max_diff, int(np.abs(diff).max()))
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "downsample",
        "template": plan["template"],
        "bytes": total,
        "mismatch": mismatch,
        "max_diff": max_diff,
    }
    return _finish_result(result, path, feeds, got, expected, model)


def build_composite(plan, prefix=""):
    """The final composite as MatMulInteger plus the static pointwise blend."""
    p = plan
    builder = Builder(p, prefix)
    N = p["width"] * p["height"]
    builder.const("cp_zero_f", "fp32", 0.0)
    builder.const("cp_one_f", "fp32", 1.0)
    builder.const("cp_two_f", "fp32", 2.0)
    builder.const("cp_three_f", "fp32", 3.0)
    builder.const("cp_half", "fp32", 0.5)
    builder.const("cp_255", "fp32", 255.0)

    # dot4AddPacked is signed int8; reinterpret the uint8 feature bytes.
    builder.add("Cast", [builder.name("x0")], ["cp_feat_i8"],
                to=TensorProto.INT8)

    # Repeat the half-res 16-byte feature vector over the 2x2 block.
    builder.const("cp_feat5_shape", "i64",
                  [p["feature_h"], 1, p["feature_w"], 1, 16])
    builder.add("Reshape", ["cp_feat_i8", "cp_feat5_shape"],
                ["cp_feat5"])
    builder.const("cp_tile", "i64", [1, 2, 1, 2, 1])
    builder.add("Tile", ["cp_feat5", "cp_tile"], ["cp_feat_tiled"])
    builder.const("cp_feat_shape", "i64", [N, 16])
    builder.add("Reshape", ["cp_feat_tiled", "cp_feat_shape"], ["cp_feat"])

    weights = np.zeros((16, 20), dtype=np.int8)
    for gate in range(5):
        for parity in range(4):
            weights[:, gate * 4 + parity] = p["weights"][gate * 4 + parity]
    builder.tensor("cp_weights", weights)
    builder.add("MatMulInteger", ["cp_feat", "cp_weights"], ["cp_acc"])

    builder.const("cp_select", "i64", p["select"])
    builder.add("GatherElements", ["cp_acc", "cp_select"], ["cp_acc_sel"], axis=1)
    builder.const("cp_select_flat", "i64", p["select"].reshape(-1))
    builder.tensor("cp_scales", p["scales"])
    builder.tensor("cp_biases", p["biases"])
    builder.add("Gather", ["cp_scales", "cp_select_flat"], ["cp_scale_row"],
                axis=0)
    builder.add("Gather", ["cp_biases", "cp_select_flat"], ["cp_bias_row"],
                axis=0)
    builder.const("cp_n5_shape", "i64", [N, 5])
    builder.add("Reshape", ["cp_scale_row", "cp_n5_shape"], ["cp_scale"])
    builder.add("Reshape", ["cp_bias_row", "cp_n5_shape"], ["cp_bias"])
    builder.add("Cast", ["cp_acc_sel"], ["cp_acc_f"], to=TensorProto.FLOAT)
    builder.add("Mul", ["cp_acc_f", "cp_scale"], ["cp_v_scaled"])
    builder.add("Add", ["cp_v_scaled", "cp_bias"], ["cp_v"])
    # The numpy replay's float32 chain; ORT's fused Sigmoid rounds differently.
    builder.add("Neg", ["cp_v"], ["cp_neg_v"])
    builder.add("Exp", ["cp_neg_v"], ["cp_exp_v"])
    builder.add("Add", ["cp_exp_v", "cp_one_f"], ["cp_den_v"])
    builder.add("Div", ["cp_one_f", "cp_den_v"], ["cp_s_flat"])
    builder.const("cp_hw5", "i64", [p["height"], p["width"], 5])
    builder.add("Reshape", ["cp_s_flat", "cp_hw5"], ["cp_s"])

    a = _overlay_channels(builder, p, "t1")
    b = _overlay_channels(builder, p, "t2")
    c = _overlay_channels(builder, p, "t3")
    d = _overlay_channels(builder, p, "t11")
    e = _overlay_channels(builder, p, "t12")
    s0 = _gate(builder, "cp_s", 0)
    s1 = _gate(builder, "cp_s", 1)
    s2 = _gate(builder, "cp_s", 2)
    s3 = _gate(builder, "cp_s", 3)
    s4 = _gate(builder, "cp_s", 4)

    def fma(out, a, factor, addend, tag):
        """One-rounding float32 FMA via a float64 product/sum."""
        builder.add("Cast", [a], [tag + "_a64"], to=TensorProto.DOUBLE)
        builder.add("Cast", [factor], [tag + "_f64"], to=TensorProto.DOUBLE)
        builder.add("Mul", [tag + "_a64", tag + "_f64"], [tag + "_p64"])
        builder.add("Cast", [addend], [tag + "_b64"], to=TensorProto.DOUBLE)
        builder.add("Add", [tag + "_p64", tag + "_b64"], [tag + "_s64"])
        builder.add("Cast", [tag + "_s64"], [out], to=TensorProto.FLOAT)

    def blend(ac, aa, bc, cc, dc, ec, tag):
        # m0 = B + (A-B)*s2; m1 = (X.a-m0) + (C-X.a)*s0, where X is
        # t1/t2/t3 for the R/G/B channel; m2 = m0 + m1*s1;
        # m3 = E + (D-E)*s3; m = m3 + (m2-m3)*s4.
        # `fmul fast` + `fadd fast` contract to one FMA.
        def n(name):
            return tag + name

        builder.add("Sub", [ac, bc], [n("ab")])
        fma(n("m0"), n("ab"), s2, bc, n("fma0"))
        builder.add("Sub", [aa, n("m0")], [n("aa_m0")])
        builder.add("Sub", [cc, aa], [n("c_aa")])
        fma(n("m1"), n("c_aa"), s0, n("aa_m0"), n("fma1"))
        fma(n("m2"), n("m1"), s1, n("m0"), n("fma2"))
        builder.add("Sub", [dc, ec], [n("de")])
        fma(n("m3"), n("de"), s3, ec, n("fma3"))
        builder.add("Sub", [n("m2"), n("m3")], [n("m2_m3")])
        fma(n("m"), n("m2_m3"), s4, n("m3"), n("fma4"))
        return n("m"), n("m2")

    m, m2 = [], []
    for channel, (ac, aa, bc, cc, dc, ec) in enumerate((
            (a[0], a[3], b[0], c[0], d[0], e[0]),
            (a[1], b[3], b[1], c[1], d[1], e[1]),
            (a[2], c[3], b[2], c[2], d[2], e[2]))):
        m_name, m2_name = blend(ac, aa, bc, cc, dc, ec,
                                "cp_ch%d_" % channel)
        m.append(m_name)
        m2.append(m2_name)

    if p["mode"] == 2:
        q = _overlay_channels(builder, p, "t9")
        t10 = _overlay_channels(builder, p, "t10")
        builder.const("cp_ten", "fp32", 10.0)
        builder.add("Mul", [q[3], "cp_ten"], ["cp_q10"])
        builder.add("Clip", ["cp_q10", "cp_zero_f", "cp_one_f"], ["cp_f"])
        builder.add("Mul", ["cp_f", "cp_f"], ["cp_f2"])
        builder.add("Mul", ["cp_f", "cp_two_f"], ["cp_2f"])
        builder.add("Sub", ["cp_three_f", "cp_2f"], ["cp_3_2f"])
        builder.add("Mul", ["cp_f2", "cp_3_2f"], ["cp_w"])
        out = []
        for channel in range(3):
            builder.add("Sub", [t10[channel], m2[channel]],
                        ["cp_o_t10_%d" % channel])
            builder.add("Mul", ["cp_o_t10_%d" % channel, "cp_w"],
                        ["cp_o_w_%d" % channel])
            builder.add("Add", [m2[channel], "cp_o_w_%d" % channel],
                        ["cp_out_%d" % channel])
            out.append("cp_out_%d" % channel)
    elif p["mode"] == 3:
        q = _overlay_channels(builder, p, "t9")
        out = []
        for channel in range(3):
            builder.add("Sub", ["cp_one_f", q[3]], ["cp_1qa_%d" % channel])
            fma("cp_out_%d" % channel, m[channel], "cp_1qa_%d" % channel,
                q[channel], "cp_fmaq_%d" % channel)
            out.append("cp_out_%d" % channel)
    elif p["mode"] in (4, 5):
        t8 = _overlay_channels(builder, p, "t8")
        t10 = _overlay_channels(builder, p, "t10")
        q = _overlay_channels(builder, p, "t9") if p["mode"] == 5 else None
        pcol = []
        for channel in range(3):
            if p["mode"] == 4:
                pcol.append(t8[channel])
            else:
                builder.add("Sub", ["cp_one_f", q[3]],
                            ["cp_1q_%d" % channel])
                builder.add("Mul", [t8[channel], "cp_1q_%d" % channel],
                            ["cp_t8q_%d" % channel])
                builder.add("Add", [q[channel], "cp_t8q_%d" % channel],
                            ["cp_p_%d" % channel])
                pcol.append("cp_p_%d" % channel)
        # g = clip(sqrt(sum diff^2) * 20, 0, 1); w = g*g*(3-2g); keep = 1-w.
        accum = None
        for channel in range(3):
            builder.add("Sub", [pcol[channel], t10[channel]],
                        ["cp_diff_%d" % channel])
            builder.add("Mul", ["cp_diff_%d" % channel, "cp_diff_%d" % channel],
                        ["cp_diff2_%d" % channel])
            if channel == 0:
                accum = "cp_diff2_0"
            else:
                previous = ("cp_diff2_0" if channel == 1 else
                            "cp_dsum_%d" % (channel - 1))
                accum = "cp_dsum_%d" % channel
                builder.add("Add", [previous, "cp_diff2_%d" % channel],
                            [accum])
        builder.add("Sqrt", [accum], ["cp_dnorm"])
        builder.const("cp_20", "fp32", 20.0)
        builder.add("Mul", ["cp_dnorm", "cp_20"], ["cp_g20"])
        builder.add("Clip", ["cp_g20", "cp_zero_f", "cp_one_f"], ["cp_g"])
        builder.add("Mul", ["cp_g", "cp_g"], ["cp_g2"])
        builder.add("Mul", ["cp_g", "cp_two_f"], ["cp_g2f"])
        builder.add("Sub", ["cp_three_f", "cp_g2f"], ["cp_g3"])
        builder.add("Mul", ["cp_g2", "cp_g3"], ["cp_w"])
        builder.add("Sub", ["cp_one_f", "cp_w"], ["cp_keep"])
        out = []
        for channel in range(3):
            if p["mode"] == 4:
                # d38 uses the final full m, not m2.
                builder.add("Sub", [m[channel], t8[channel]],
                            ["cp_m2t8_%d" % channel])
                builder.add("Mul", ["cp_m2t8_%d" % channel, "cp_keep"],
                            ["cp_keep_m2_%d" % channel])
                builder.add("Add", [t10[channel], "cp_keep_m2_%d" % channel],
                            ["cp_out_%d" % channel])
            else:
                builder.add("Sub", ["cp_one_f", q[3]],
                            ["cp_1q2_%d" % channel])
                builder.add("Mul", [m2[channel], "cp_1q2_%d" % channel],
                            ["cp_q2m_%d" % channel])
                builder.add("Add", [q[channel], "cp_q2m_%d" % channel],
                            ["cp_q2_%d" % channel])
                builder.add("Mul", ["cp_keep", pcol[channel]],
                            ["cp_kp_%d" % channel])
                builder.add("Sub", [t10[channel], "cp_kp_%d" % channel],
                            ["cp_t10_kp_%d" % channel])
                builder.add("Mul", ["cp_keep", "cp_q2_%d" % channel],
                            ["cp_kq_%d" % channel])
                builder.add("Add", ["cp_t10_kp_%d" % channel,
                                    "cp_kq_%d" % channel],
                            ["cp_out_%d" % channel])
            out.append("cp_out_%d" % channel)
    else:
        out = m

    # BGRA8: B=out[2], G=out[1], R=out[0], A=255. The texture unit performs
    # the UNORM8 conversion, so the 8-bit rounding is computed exactly.
    builder.const("cp_255d", "fp64", [255.0])
    builder.const("cp_halfd", "fp64", [0.5])
    channels = []
    for source in (out[2], out[1], out[0]):
        builder.add("Clip", [source, "cp_zero_f", "cp_one_f"],
                    ["cp_clip_" + source])
        builder.add("Cast", ["cp_clip_" + source], ["cp_clipd_" + source],
                    to=TensorProto.DOUBLE)
        builder.add("Mul", ["cp_clipd_" + source, "cp_255d"],
                    ["cp_255d_" + source])
        builder.add("Add", ["cp_255d_" + source, "cp_halfd"],
                    ["cp_halfd_" + source])
        builder.add("Floor", ["cp_halfd_" + source], ["cp_floord_" + source])
        builder.add("Cast", ["cp_floord_" + source], ["cp_u8_" + source],
                    to=TensorProto.UINT8)
        channels.append("cp_u8_" + source)
    builder.const("cp_alpha_shape", "i64", [p["height"], p["width"], 1])
    builder.add("ConstantOfShape", ["cp_alpha_shape"], ["cp_alpha"],
                value=numpy_helper.from_array(np.array([255], dtype=np.uint8)))
    channels.append("cp_alpha")
    builder.add("Concat", channels, ["u0"], axis=2)

    inputs = [helper.make_tensor_value_info(builder.name("x0"), TensorProto.UINT8,
                                            [p["feature_h"], p["feature_w"], 16])]
    for slot in p["overlays"]:
        inputs.append(helper.make_tensor_value_info(
            builder.name("in_" + slot), TensorProto.UINT8,
            [p["height"], p["width"], 4]))
    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_composite_d%04d" % p["dispatch"]),
        inputs,
        [helper.make_tensor_value_info(builder.name("u0"), TensorProto.UINT8,
                                       [p["height"], p["width"], 4])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_composite(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.composite_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "final composite",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_composite(plan)
        onnx.checker.check_model(model)
        name = "d%04d_composite" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["feature"]}
    for slot, data in plan["overlays"].items():
        feeds["in_" + slot] = data
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"]
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    pixels = int((diff != 0).any(axis=-1).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "final composite",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
        "pixels": pixels,
    }
    return _finish_result(result, path, feeds, got, expected, model)


def build_resample(plan, prefix=""):
    """Half-pixel bilinear gather as standard ONNX; output is float32."""
    p = plan
    builder = Builder(p, prefix)
    builder.const("rs_gather", "i64", p["gather"])
    builder.add("Gather", [builder.name("x0"), "rs_gather"], ["rs_corners"],
                axis=0)
    builder.add("Cast", ["rs_corners"], ["rs_f"], to=TensorProto.FLOAT)

    corners = []
    for index in range(4):
        builder.const("rs_pick_%d" % index, "i64", [index])
        builder.add("Gather", ["rs_f", "rs_pick_%d" % index],
                    ["rs_c%d" % index], axis=2)
        corners.append("rs_c%d" % index)
    for index in range(4):
        builder.tensor("rs_w%d" % index, p["weights"][index])
        builder.add("Mul", [corners[index], "rs_w%d" % index],
                    ["rs_m%d" % index])
    builder.add("Add", ["rs_m1", "rs_m0"], ["rs_s01"])
    builder.add("Add", ["rs_m2", "rs_s01"], ["rs_s012"])
    builder.add("Add", ["rs_m3", "rs_s012"], ["rs_value"])

    if p["residual"]:
        # The replay rounds the product and sum once, in float64, then casts.
        builder.add("Cast", ["rs_value"], ["rs_value_d"], to=TensorProto.DOUBLE)
        builder.const("rs_gain_d", "fp64", float(p["gain"]))
        builder.add("Mul", ["rs_value_d", "rs_gain_d"], ["rs_scaled_d"])
        builder.const("rs_res_index", "i64", p["residual_index"])
        builder.add("Gather", [builder.name("x1"), "rs_res_index"], ["rs_res"],
                    axis=0)
        builder.add("Cast", ["rs_res"], ["rs_res_d"], to=TensorProto.DOUBLE)
        builder.const("rs_res_axis", "i64", [2])
        builder.add("Unsqueeze", ["rs_res_d", "rs_res_axis"], ["rs_res_u"])
        builder.add("Add", ["rs_scaled_d", "rs_res_u"], ["rs_sum_d"])
        builder.add("Cast", ["rs_sum_d"], ["rs_out4"], to=TensorProto.FLOAT)
    else:
        builder.tensor("rs_gain", p["gain_vector"])
        builder.add("Mul", ["rs_value", "rs_gain"], ["rs_out4"])
    builder.const("rs_squeeze", "i64", [2])
    builder.add("Squeeze", ["rs_out4", "rs_squeeze"], ["y0"])

    inputs = [helper.make_tensor_value_info(
        builder.name("x0"),
        TensorProto.FLOAT16 if p["source_f16"] else TensorProto.INT8,
        [p["src"].size])]
    if p["residual"]:
        inputs.append(helper.make_tensor_value_info(
            builder.name("x1"), TensorProto.FLOAT16, [p["res"].size]))
    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_resample_d%04d" % p["dispatch"]),
        inputs,
        [helper.make_tensor_value_info(
            builder.name("y0"), TensorProto.FLOAT,
            [p["dst_h"], p["dst_w"], p["channels"]])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_resample(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.resample_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "resample",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_resample(plan)
        onnx.checker.check_model(model)
        name = "d%04d_resample" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["src"]}
    if plan["residual"]:
        feeds["x1"] = plan["res"]
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["y0"], feeds)[0]

    expected = plan["expected"]
    packed = np.zeros(expected.size, dtype=np.uint8)
    texels = packed.view("<u2").reshape(
        plan["rows_alloc"], plan["dst_w"], plan["cell_bytes"] // 2)
    texels[plan["start_row"]:plan["start_row"] + plan["dst_h"], :,
           :plan["channels"]] = recon._f32_to_half_bits(got)
    diff = packed.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "resample",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
        "f16_pack": "harness",
    }
    if mismatch:
        bad = np.argwhere(diff != 0)[:8]
        result["first_bad"] = [list(map(int, b)) for b in bad]
    return _finish_result(result, path, feeds, packed, expected, model)


def build_gemv(plan, prefix=""):
    """One buffer GEMV as a gather/byte pack followed by MatMulInteger."""
    p = plan
    builder = Builder(p, prefix)
    builder.const("gm_index_halves", "i64", p["index_halves"])
    builder.const("gm_index_t4", "i64", p["index_t4"])
    builder.tensor("gm_in_range", p["in_range"])

    builder.add("Gather", [builder.name("x0"), "gm_index_halves"],
                ["gm_halves"], axis=0)
    builder.add("Cast", ["gm_halves"], ["gm_values_f"], to=TensorProto.FLOAT)
    builder.add("Cast", ["gm_values_f"], ["gm_values"], to=TensorProto.INT32)
    builder.const("gm_vlo", "i32", [-128])
    builder.const("gm_vhi", "i32", [127])
    builder.add("Clip", ["gm_values", "gm_vlo", "gm_vhi"], ["gm_values_clipped"])

    bytes_out = []
    names = ["gm_v%d" % k for k in range(6)]
    for k in range(6):
        source = "gm_values_clipped"
        builder.const("gm_vstart_%d" % k, "i64", [k])
        builder.const("gm_vend_%d" % k, "i64", [k + 1])
        builder.const("gm_vaxis_%d" % k, "i64", [1])
        builder.const("gm_vstep_%d" % k, "i64", [1])
        builder.add("Slice", [source, "gm_vstart_%d" % k, "gm_vend_%d" % k,
                              "gm_vaxis_%d" % k, "gm_vstep_%d" % k], [names[k]])
        unsigned = _unsigned_byte(builder, names[k], "gm_vu%d" % k)
        builder.add("Mul", [unsigned, "gm_in_range"], ["gm_vm%d" % k])
        bytes_out.append("gm_vm%d" % k)

    builder.add("Gather", [builder.name("x1"), "gm_index_t4"], ["gm_t4"], axis=0)
    builder.add("Cast", ["gm_t4"], ["gm_t4_i"], to=TensorProto.INT32)
    for k in range(2):
        builder.const("gm_qstart_%d" % k, "i64", [k])
        builder.const("gm_qend_%d" % k, "i64", [k + 1])
        builder.const("gm_qaxis_%d" % k, "i64", [1])
        builder.const("gm_qstep_%d" % k, "i64", [1])
        builder.add("Slice", ["gm_t4_i", "gm_qstart_%d" % k, "gm_qend_%d" % k,
                              "gm_qaxis_%d" % k, "gm_qstep_%d" % k],
                    ["gm_q%d" % k])
        unsigned = _unsigned_byte(builder, "gm_q%d" % k, "gm_qu%d" % k)
        builder.add("Mul", [unsigned, "gm_in_range"], ["gm_qm%d" % k])
    builder.const("gm_kfill", "i32", [p["kfill"]])
    builder.add("Mul", ["gm_kfill", "gm_in_range"], ["gm_kfill_masked"])

    # Word 0 is q0,q1,v0,v1; word 1 is v2,v3,v4,kfill. Nine taps make 72 bytes.
    packed = ["gm_qm0", "gm_qm1", "gm_vm0", "gm_vm1",
              "gm_vm2", "gm_vm3", "gm_vm4", "gm_kfill_masked"]
    builder.add("Concat", packed, ["gm_packed32"], axis=1)
    builder.add("Cast", ["gm_packed32"], ["gm_packed8"], to=TensorProto.INT8)
    builder.const("gm_packed_shape", "i64", [p["gx"] * p["gy"], 72])
    builder.add("Reshape", ["gm_packed8", "gm_packed_shape"], ["gm_packed"])

    builder.tensor("gm_weights", p["weights"])
    builder.add("MatMulInteger", ["gm_packed", "gm_weights"], ["gm_acc"])
    builder.tensor("gm_bias", p["bias"])
    builder.add("Add", ["gm_acc", "gm_bias"], ["gm_biased"])
    builder.const("gm_lo", "i32", [-131072])
    builder.const("gm_hi", "i32", [131071])
    builder.add("Clip", ["gm_biased", "gm_lo", "gm_hi"], ["gm_clamped"])
    builder.const("gm_zero", "i32", [0])
    builder.add("Max", ["gm_clamped", "gm_zero"], ["gm_pos"])
    builder.add("Min", ["gm_clamped", "gm_zero"], ["gm_neg"])
    builder.const("gm_cscale", "i32", [p["cscale"]])
    builder.add("Mul", ["gm_pos", "gm_cscale"], ["gm_pos_scaled"])
    pos = _arithmetic_shift(builder, "gm_pos_scaled", p["s1"], "gm_pos")
    builder.tensor("gm_activation", p["activation"])
    builder.add("Mul", ["gm_neg", "gm_activation"], ["gm_neg_scaled"])
    neg = _arithmetic_shift(builder, "gm_neg_scaled", p["s2"], "gm_neg")
    builder.add("Add", [pos, neg], ["gm_requant"])
    builder.const("gm_qlo", "i32", [-128])
    builder.const("gm_qhi", "i32", [127])
    builder.add("Clip", ["gm_requant", "gm_qlo", "gm_qhi"], ["gm_q"])
    builder.add("Cast", ["gm_q"], ["gm_q8"], to=TensorProto.UINT8)

    builder.const("gm_out5_shape", "i64", [p["gx"], p["gy"], p["gz"], 4, 4])
    builder.add("Reshape", ["gm_q8", "gm_out5_shape"], ["gm_out5"])
    builder.add("Transpose", ["gm_out5"], ["gm_tile"], perm=[2, 1, 0, 3, 4])
    builder.const("gm_tile_shape", "i64", [p["gz"], p["gy"], p["gx"], 16])
    builder.add("Reshape", ["gm_tile", "gm_tile_shape"], ["gm_tile_flat"])
    builder.const("gm_crop_starts", "i64", [0, 0, 0, 0])
    builder.const("gm_crop_ends", "i64", [p["gz"], p["oh"], p["ow"], 16])
    builder.const("gm_crop_axes", "i64", [0, 1, 2, 3])
    builder.const("gm_crop_steps", "i64", [1, 1, 1, 1])
    builder.add("Slice", ["gm_tile_flat", "gm_crop_starts", "gm_crop_ends",
                          "gm_crop_axes", "gm_crop_steps"], ["gm_cropped"])
    builder.const("gm_out_pads", "i64", [0, 1, 0, 0,
                                         0, p["height"] - 1 - p["oh"],
                                         p["width"] - p["ow"], 0])
    builder.const("gm_out_zero", "u8", [0])
    builder.add("Pad", ["gm_cropped", "gm_out_pads", "gm_out_zero"], ["u0"])

    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_gemv_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(builder.name("x0"), TensorProto.FLOAT16,
                                       [p["src0"].size]),
         helper.make_tensor_value_info(builder.name("x1"), TensorProto.INT8,
                                       [p["src1"].size])],
        [helper.make_tensor_value_info(
            builder.name("u0"), TensorProto.UINT8,
            [p["gz"], p["height"], p["width"], 16])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_gemv(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.gemv_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "GEMV",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_gemv(plan)
        onnx.checker.check_model(model)
        name = "d%04d_gemv" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["src0"], "x1": plan["src1"]}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"]
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "GEMV",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
    }
    if mismatch:
        bad = np.argwhere(diff != 0)[:8]
        result["first_bad"] = [list(map(int, b)) for b in bad]
    return _finish_result(result, path, feeds, got, expected, model)


def build_buffer_output(plan, prefix=""):
    """One buffer-output GEMV as MatMulInteger plus a static byte scatter."""
    p = plan
    builder = Builder(p, prefix)
    builder.const("bo_index0", "i64", p["index0"])
    builder.const("bo_index1", "i64", p["index1"])
    builder.add("Gather", [builder.name("x0"), "bo_index0"], ["bo_g0"], axis=0)
    builder.add("Gather", [builder.name("x1"), "bo_index1"], ["bo_g1"], axis=0)
    builder.add("Concat", ["bo_g0", "bo_g1"], ["bo_input"], axis=1)
    builder.tensor("bo_weights", p["weights"])
    builder.add("MatMulInteger", ["bo_input", "bo_weights"], ["bo_acc"])
    builder.tensor("bo_seed", p["seed"])
    builder.add("Add", ["bo_acc", "bo_seed"], ["bo_seeded"])
    builder.const("bo_lo", "i32", [-131072])
    builder.const("bo_hi", "i32", [131071])
    builder.add("Clip", ["bo_seeded", "bo_lo", "bo_hi"], ["bo_clamped"])
    builder.const("bo_scale", "i32", [p["scale"]])
    builder.add("Mul", ["bo_clamped", "bo_scale"], ["bo_scaled"])
    shifted = _arithmetic_shift(builder, "bo_scaled", p["shift"], "bo")
    builder.const("bo_qlo", "i32", [-128])
    builder.const("bo_qhi", "i32", [127])
    builder.add("Clip", [shifted, "bo_qlo", "bo_qhi"], ["bo_q"])
    builder.add("Cast", ["bo_q"], ["bo_q8"], to=TensorProto.UINT8)
    builder.const("bo_flat_shape", "i64", [p["gx"] * p["gy"] * p["nacc"]])
    builder.add("Reshape", ["bo_q8", "bo_flat_shape"], ["bo_flat"])
    builder.const("bo_update_index", "i64", p["update_index"])
    builder.add("Gather", ["bo_flat", "bo_update_index"], ["bo_updates"], axis=0)
    builder.const("bo_scatter_index", "i64", p["scatter_index"])
    builder.const("bo_output_zero", "u8", [0] * p["expected"].size)
    builder.add("ScatterND", ["bo_output_zero", "bo_scatter_index", "bo_updates"],
                ["u0"])

    graph = helper.make_graph(
        builder.nodes, builder.name("xefg_bufout_d%04d" % p["dispatch"]),
        [helper.make_tensor_value_info(builder.name("x0"), TensorProto.INT8,
                                       [p["src0"].size]),
         helper.make_tensor_value_info(builder.name("x1"), TensorProto.INT8,
                                       [p["src1"].size])],
        [helper.make_tensor_value_info(builder.name("u0"), TensorProto.UINT8,
                                       [p["expected"].size])],
        builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def export_buffer_output(run_dir, dispatch, out_dir, verify_path=None):
    plan, error = recon.buffer_output_plan(run_dir, dispatch)
    if error:
        return {"dispatch": dispatch["dispatch"], "family": "buffer output",
                "error": error}
    plan["dispatch"] = dispatch["dispatch"]
    if verify_path is None:
        model = build_buffer_output(plan)
        onnx.checker.check_model(model)
        name = "d%04d_bufout" % dispatch["dispatch"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, path)
    else:
        path = verify_path
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)

    feeds = {"x0": plan["src0"], "x1": plan["src1"]}
    session = _require_onnxruntime().InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(["u0"], feeds)[0]
    expected = plan["expected"]
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    result = {
        "dispatch": dispatch["dispatch"],
        "family": "buffer output",
        "template": plan["template"],
        "bytes": int(expected.size),
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
    }
    if mismatch:
        bad = np.argwhere(diff != 0)[:8]
        result["first_bad"] = [list(map(int, b)) for b in bad]
    return _finish_result(result, path, feeds, got, expected, model)


_FAMILY_SHAPES = (
    ("conv", recon.conv_shaped),
    ("flow conv", recon.flow_shaped),
    ("upsample", recon.upsample_shaped),
    ("GEMV", recon.gemv_shaped),
    ("buffer output", recon.buffer_output_shaped),
    ("resample", recon.resample_shaped),
    ("final composite", recon.final_composite_shaped),
    ("downsample", recon.downsample_shaped),
    ("warp", recon.warp_shaped),
    ("splat", recon.splat_shaped),
    ("warp + flow init", recon.warp_init_shaped),
)


def classify_family(dispatch):
    """The lowered family name for a dispatch, or None for a custom node."""
    for family, shaped in _FAMILY_SHAPES:
        if shaped(dispatch):
            return family
    return None


def export_dispatch(run_dir, dispatch, family, params, out_dir, verify_path=None):
    """Build (or re-run) one dispatch's model; verify_path rechecks an artifact."""
    if family == "conv":
        return export_conv(run_dir, dispatch, params.get(dispatch["key"], {}),
                           out_dir, verify_path)
    if family == "flow conv":
        return export_flow(run_dir, dispatch, out_dir, verify_path)
    if family == "upsample":
        return export_upsample(run_dir, dispatch, out_dir, verify_path)
    if family == "GEMV":
        return export_gemv(run_dir, dispatch, out_dir, verify_path)
    if family == "buffer output":
        return export_buffer_output(run_dir, dispatch, out_dir, verify_path)
    if family == "resample":
        return export_resample(run_dir, dispatch, out_dir, verify_path)
    if family == "final composite":
        return export_composite(run_dir, dispatch, out_dir, verify_path)
    if family == "downsample":
        return export_downsample(run_dir, dispatch, out_dir, verify_path)
    if family == "warp":
        return export_warp(run_dir, dispatch, out_dir, verify_path)
    if family == "splat":
        return export_splat(run_dir, dispatch, out_dir, verify_path)
    if family == "warp + flow init":
        return export_warp_init(run_dir, dispatch, out_dir, verify_path)
    return {"dispatch": dispatch["dispatch"], "family": family,
            "error": "no exporter for family %s" % family}


FAMILIES = (
    ("conv", recon.conv_shaped, "ConvInteger + integer requant", True),
    ("flow conv", recon.flow_shaped, "ConvInteger + 4-phase max + float requant", True),
    ("upsample", recon.upsample_shaped, "fixed-point bilinear + ConvInteger", True),
    ("GEMV", recon.gemv_shaped, "gather + byte pack + MatMulInteger", True),
    ("buffer output", recon.buffer_output_shaped, "MatMulInteger + static scatter", True),
    ("resample", recon.resample_shaped, "Gather + baked bilinear weights", True),
    ("splat", recon.splat_shaped, "ScatterND(max) + fp16 pack", True),
    ("downsample", recon.downsample_shaped, "ordered luma sums + quantise", True),
    ("warp", recon.warp_shaped, "dynamic gather + bilinear", True),
    ("warp + flow init", recon.warp_init_shaped,
     "integer texel filter + MatMulInteger", True),
    ("final composite", recon.final_composite_shaped, "MatMulInteger + pointwise blend", True),
)


def classify(dispatch):
    for name, predicate, target, lowered in FAMILIES:
        if predicate(dispatch):
            return name, target, lowered
    return "unclassified", "custom", False


def coverage_result_problems(entries, verification):
    """Every lowered dispatch needs a passing, hash-stamped verification."""
    problems = []
    by_id = {result["dispatch"]: result for result in verification or []}
    for entry in entries:
        if not entry["lowered"]:
            continue
        dispatch = entry["dispatch"]
        result = by_id.get(dispatch)
        if result is None:
            problems.append("d%04d has no verification result" % dispatch)
        elif not result_passes(result):
            detail = result.get("error")
            if detail is None:
                detail = "%s bytes differ, max %s" % (
                    result.get("mismatch"), result.get("max_diff"))
            problems.append("d%04d failed: %s" % (dispatch, detail))
        elif not result.get("model_sha256") or not result.get("output_sha256"):
            problems.append("d%04d result has no hashes" % dispatch)
    return problems


def coverage(run_dir, out_dir, artifacts=None):
    rows = recon.read_events(run_dir)
    dispatches = [r for r in rows if r.get("type") == "dispatch" and r["frame"] == 1]
    params = recon.layer_parameters(rows)
    entries = []
    for dispatch in dispatches:
        name, target, lowered = classify(dispatch)
        entries.append({
            "dispatch": dispatch["dispatch"],
            "program": dispatch["key"],
            "family": name,
            "onnx_target": target,
            "lowered": lowered,
            "outputs": [r["slot"] for r in dispatch["resources"] if r["kind"] == "uav"],
        })
    counts = {}
    for entry in entries:
        counts[entry["family"]] = counts.get(entry["family"], 0) + 1
    report = {"nodes": entries, "families": counts,
              "lowered": sum(1 for e in entries if e["lowered"]),
              "total": len(entries)}

    problems = coverage_problems(entries, artifacts)
    verification = []
    if artifacts is not None:
        by_id = {dispatch["dispatch"]: dispatch for dispatch in dispatches}
        missing, empty = [], []
        for entry in entries:
            if not entry["lowered"]:
                continue
            name = artifact_name(entry)
            if name is None:
                continue
            artifact = os.path.join(artifacts, name)
            if not os.path.exists(artifact):
                missing.append(name)
            elif os.path.getsize(artifact) == 0:
                empty.append(name)

        for entry in entries:
            if not entry["lowered"]:
                continue
            dispatch = by_id.get(entry["dispatch"])
            if dispatch is None:
                verification.append({"dispatch": entry["dispatch"],
                                     "family": entry["family"],
                                     "error": "dispatch is not in frame 1"})
                continue
            name = artifact_name(entry)
            artifact = os.path.join(artifacts, name) if name else ""
            try:
                result = export_dispatch(run_dir, dispatch, entry["family"],
                                         params, artifacts, verify_path=artifact)
            except Exception as exc:  # corrupt ONNX, ORT failure, ...
                result = {"dispatch": entry["dispatch"],
                          "family": entry["family"],
                          "error": "artifact verification failed: %s" % exc}
            verification.append(result)

        report["artifacts"] = {"dir": artifacts, "missing": missing,
                               "empty": empty,
                               "failed": [result["dispatch"]
                                          for result in verification
                                          if not result_passes(result)]}
        report["verification"] = verification
        report["artifact_verified"] = sum(
            1 for result in verification if result_passes(result))
        problems += coverage_result_problems(entries, verification)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "onnx_coverage.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    os.replace(tmp, path)

    for entry in entries:
        print("d%04d %-20s -> %-38s %s"
              % (entry["dispatch"], entry["family"], entry["onnx_target"],
                 "LOWERED" if entry["lowered"] else "custom"))
    if artifacts is not None:
        for result in verification:
            status = result.get("status") or result.get("error") or "FAILED"
            print("  d%04d %-16s %-8s %s"
                  % (result.get("dispatch", -1), result.get("family", "?"),
                     status, result.get("model_sha256", "")[:16]))
        print("%d/%d artifacts passed their family gate"
              % (report["artifact_verified"], len(verification)))
    print("%d/%d dispatches lowered to standard ONNX ops; %s"
          % (report["lowered"], report["total"], path))
    for problem in problems:
        print("  COVERAGE FAILED: %s" % problem)
    return 1 if problems else 0


_FAMILY_REQUESTS = (
    ("conv", "all_conv", conv_set_problem, recon.conv_shaped),
    ("flow conv", "all_flow", flow_set_problem, recon.flow_shaped),
    ("upsample", "all_upsample", upsample_set_problem, recon.upsample_shaped),
    ("GEMV", "all_gemv", gemv_set_problem, recon.gemv_shaped),
    ("buffer output", "all_buffer", buffer_set_problem, recon.buffer_output_shaped),
    ("resample", "all_resample", resample_set_problem, recon.resample_shaped),
    ("final composite", "all_composite", composite_set_problem,
     recon.final_composite_shaped),
    ("downsample", "all_downsample", downsample_set_problem,
     recon.downsample_shaped),
    ("warp", "all_warp", warp_set_problem, recon.warp_shaped),
    ("splat", "all_splat", splat_set_problem, recon.splat_shaped),
    ("warp + flow init", "all_warp_init", warp_init_set_problem,
     recon.warp_init_shaped),
)


def select_families(args, dispatches):
    """(families, error); families is [(name, candidates)]."""
    if args.dispatch is not None:
        selected = [d for d in dispatches if d["dispatch"] == args.dispatch]
        if not selected:
            return None, "dispatch %d is not in frame 1" % args.dispatch
        family = classify_family(selected[0])
        if family is None:
            return None, "dispatch %d is not a lowered program" % args.dispatch
        return [(family, selected)], None

    if args.all:
        enabled = list(_FAMILY_REQUESTS)
    else:
        enabled = [request for request in _FAMILY_REQUESTS
                   if getattr(args, request[1])]
    if not enabled:
        return None, "pass --dispatch N, --all, or an --all-<family> flag"

    families = []
    for family, _flag, problem_fn, shape_fn in enabled:
        problem = problem_fn(dispatches)
        if problem:
            return None, "%s gate failed: %s" % (family, problem)
        families.append((family, [d for d in dispatches if shape_fn(d)]))
    return families, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--dispatch", type=int)
    parser.add_argument("--all-conv", action="store_true")
    parser.add_argument("--all-flow", action="store_true")
    parser.add_argument("--all-upsample", action="store_true")
    parser.add_argument("--all-gemv", action="store_true")
    parser.add_argument("--all-buffer", action="store_true")
    parser.add_argument("--all-resample", action="store_true")
    parser.add_argument("--all-composite", action="store_true")
    parser.add_argument("--all-downsample", action="store_true")
    parser.add_argument("--all-warp", action="store_true")
    parser.add_argument("--all-splat", action="store_true")
    parser.add_argument("--all-warp-init", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="export every lowered family into one report")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--artifacts",
                        help="with --coverage: execute and compare each .onnx")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.coverage:
        return coverage(args.run_dir, args.out, args.artifacts)

    rows = recon.read_events(args.run_dir)
    dispatches = [r for r in rows if r.get("type") == "dispatch" and r["frame"] == 1]
    params = recon.layer_parameters(rows)
    families, error = select_families(args, dispatches)
    if error:
        print(error)
        return 1

    results = []
    for family, candidates in families:
        for dispatch in candidates:
            result = export_dispatch(args.run_dir, dispatch, family, params,
                                     args.out)
            results.append(result)
            status = result.get("status") or result.get("error") or "DIFF"
            print("d%04d %-4s %-2s %6d/%-6d %s"
                  % (result["dispatch"], result.get("family", "?"),
                     result.get("template", "?"), result.get("mismatch", -1),
                     result.get("bytes", 0), status))

    summary = {
        "exported": len(results),
        "exact": sum(1 for result in results
                     if result.get("mismatch") == 0),
        "failed": sum(1 for result in results if not result_passes(result)),
        "families": [family for family, _candidates in families],
    }
    report = os.path.join(args.out, "onnx_report.json")
    os.makedirs(args.out, exist_ok=True)
    tmp = report + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"run": os.path.abspath(args.run_dir), "frame": 1,
                   "dispatches": results, "summary": summary},
                  handle, indent=2)
    os.replace(tmp, report)
    print("%d/%d dispatches passed the gate; report %s"
          % (len(results) - summary["failed"], len(results), report))
    return 1 if summary["failed"] else 0
