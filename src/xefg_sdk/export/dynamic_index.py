#!/usr/bin/env python
"""D2: dynamic-shape models for the remaining index-table dispatches.

Each family is built once from the first run and then executed against every
supplied capture. The static plans carry fully enumerated gather/scatter
tables; this module rebuilds those tables in-graph from `Range`/`Mod`/`Clip`
and the runtime extents.
"""
# Internal legacy research module; not a supported CLI entry point.
import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import recon
from .dynamic_onnx import (Builder, arithmetic_shift, scalar_const,
                           shape_dim, squeeze_scalar)


FAMILIES = ("splat", "resample", "downsample", "gemv", "buffer")


# --------------------------------------------------------------------------
# frame dispatch numbering
# --------------------------------------------------------------------------

def frame_dispatches(run_dir, frame=1, rows=None):
    """The dispatch rows retained for `frame`, ordered by capture id."""
    if rows is None:
        rows = recon.read_events(run_dir)
    dispatches = [row for row in rows
                  if row.get("type") == "dispatch"
                  and row.get("frame") == frame]
    if not dispatches:
        raise SystemExit("frame %d has no dispatches" % frame)
    return sorted(dispatches, key=lambda row: row["dispatch"])


def normalized_number_map(dispatches):
    """Capture dispatch id -> frame-relative node id (1..N).

    A retained frame keeps the same 38 dispatches in capture order, but its
    absolute ids continue from the previous frame (frame 2 is 39..76). The
    plans/models are keyed by the frame-relative id; reads and tensor file
    names keep using the dispatch id in the row itself.
    """
    return {row["dispatch"]: index + 1
            for index, row in enumerate(sorted(
                dispatches, key=lambda row: row["dispatch"]))}


def numbered_dispatches(run_dir, frame=1, rows=None):
    """(frame-relative node id, capture dispatch row) pairs for one frame."""
    dispatches = frame_dispatches(run_dir, frame, rows)
    numbers = normalized_number_map(dispatches)
    return [(numbers[row["dispatch"]], row) for row in dispatches]


# --------------------------------------------------------------------------
# splat (d24)
# --------------------------------------------------------------------------

def _splat_window(builder, tag, source, rank):
    """Pad(0) + offset Slice, with every extent derived at runtime."""
    b = builder
    th, tw = "TH", "TW"
    src_h, src_w = "src_h", "src_w"
    off_y, off_x = tag + "_off_y", tag + "_off_x"

    b.add("Sub", ["zero64", off_y], [tag + "_noy"])
    b.add("Sub", ["zero64", off_x], [tag + "_nox"])
    b.add("Max", [tag + "_noy", "zero64"], [tag + "_top64"])
    b.add("Max", [tag + "_nox", "zero64"], [tag + "_left64"])
    b.add("Add", [off_y, src_h], [tag + "_ob_y"])
    b.add("Sub", [tag + "_ob_y", th], [tag + "_oy_th"])
    b.add("Max", [tag + "_oy_th", "zero64"], [tag + "_bottom64"])
    b.add("Add", [off_x, src_w], [tag + "_ob_x"])
    b.add("Sub", [tag + "_ob_x", tw], [tag + "_ox_tw"])
    b.add("Max", [tag + "_ox_tw", "zero64"], [tag + "_right64"])
    for name in ("top", "left", "bottom", "right"):
        b.add("Cast", [tag + "_" + name + "64"], [tag + "_" + name],
              to=TensorProto.INT64)

    if rank == 3:
        panes = [tag + "_top", tag + "_left", "zero64",
                 tag + "_bottom", tag + "_right", "zero64"]
    else:
        panes = [tag + "_top", tag + "_left", tag + "_bottom", tag + "_right"]
    b.add("Concat", panes, [tag + "_pads"], axis=0)
    b.const(tag + "_pad_value", "fp32", [0.0])
    b.add("Pad", [source, tag + "_pads", tag + "_pad_value"],
          [tag + "_padded"])

    b.add("Max", [off_y, "zero64"], [tag + "_sy"])
    b.add("Max", [off_x, "zero64"], [tag + "_sx"])
    b.add("Add", [tag + "_sy", src_h], [tag + "_ey"])
    b.add("Add", [tag + "_sx", src_w], [tag + "_ex"])
    b.add("Concat", [tag + "_sy", tag + "_sx"], [tag + "_starts"], axis=0)
    b.add("Concat", [tag + "_ey", tag + "_ex"], [tag + "_ends"], axis=0)
    b.const(tag + "_axes", "i64", [0, 1])
    b.const(tag + "_steps", "i64", [1, 1])
    b.add("Slice", [tag + "_padded", tag + "_starts", tag + "_ends",
                    tag + "_axes", tag + "_steps"], [tag + "_window"])
    return tag + "_window"


def build_dynamic_splat(plan):
    p = plan
    b = Builder()
    shape_dim(b, "x0", 0, "TH")
    shape_dim(b, "x0", 1, "TW")
    squeeze_scalar(b, "src_w")
    squeeze_scalar(b, "src_h")
    scalar_const(b, "zero64_scalar", 0)
    scalar_const(b, "one64_scalar", 1)
    b.const("zero64", "i64", [0])
    b.const("sp_i0_64", "i64", [0])
    b.const("sp_i65536", "i32", [65536])

    b.const("sp_scale_x", "fp32", float(p["scale_x"]))
    b.const("sp_scale_y", "fp32", float(p["scale_y"]))
    b.const("sp_f0", "fp32", 0.0)
    b.const("sp_f1", "fp32", 1.0)
    b.const("sp_fhalf", "fp32", 0.5)
    b.const("sp_f32767", "fp32", 32767.0)

    b.add("Cast", ["x0"], ["sp_motion_f"], to=TensorProto.FLOAT)
    motion = _splat_window(b, "sp_motion", "sp_motion_f", 3)
    b.const("sp_mx_idx", "i64", [0])
    b.const("sp_my_idx", "i64", [1])
    b.const("sp_axis2", "i64", [2])
    b.add("Gather", [motion, "sp_mx_idx"], ["sp_mx3"], axis=2)
    b.add("Gather", [motion, "sp_my_idx"], ["sp_my3"], axis=2)
    b.add("Squeeze", ["sp_mx3", "sp_axis2"], ["sp_mx"])
    b.add("Squeeze", ["sp_my3", "sp_axis2"], ["sp_my"])
    depth = _splat_window(b, "sp_depth", "x1", 2)

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

    b.add("Range", ["zero64_scalar", "src_w_scalar", "one64_scalar"],
          ["sp_w_range"])
    b.add("Range", ["zero64_scalar", "src_h_scalar", "one64_scalar"],
          ["sp_h_range"])
    b.add("Cast", ["sp_w_range"], ["sp_tid_x"], to=TensorProto.FLOAT)
    b.add("Cast", ["sp_h_range"], ["sp_tid_y"], to=TensorProto.FLOAT)
    b.const("sp_ax0", "i64", [0])
    b.const("sp_ax1", "i64", [1])
    b.add("Unsqueeze", ["sp_tid_x", "sp_ax0"], ["sp_tid_x2"])
    b.add("Unsqueeze", ["sp_tid_y", "sp_ax1"], ["sp_tid_y2"])

    # Always the half-texel-centre form; at src == dst it reduces exactly to
    # tid + ceil(disp).
    b.add("Cast", ["dst_w"], ["sp_dstw_f"], to=TensorProto.FLOAT)
    b.add("Cast", ["src_w"], ["sp_srcw_f"], to=TensorProto.FLOAT)
    b.add("Cast", ["dst_h"], ["sp_dsth_f"], to=TensorProto.FLOAT)
    b.add("Cast", ["src_h"], ["sp_srch_f"], to=TensorProto.FLOAT)
    b.add("Div", ["sp_dstw_f", "sp_srcw_f"], ["sp_ratio_x"])
    b.add("Div", ["sp_dsth_f", "sp_srch_f"], ["sp_ratio_y"])
    b.add("Add", ["sp_tid_x2", "sp_fhalf"], ["sp_tx0"])
    b.add("Add", ["sp_tx0", "sp_dx"], ["sp_tx1"])
    b.add("Mul", ["sp_tx1", "sp_ratio_x"], ["sp_tx2"])
    b.add("Sub", ["sp_tx2", "sp_fhalf"], ["sp_tx3"])
    b.add("Ceil", ["sp_tx3"], ["sp_cx"])
    b.add("Add", ["sp_tid_y2", "sp_fhalf"], ["sp_ty0"])
    b.add("Add", ["sp_ty0", "sp_dy"], ["sp_ty1"])
    b.add("Mul", ["sp_ty1", "sp_ratio_y"], ["sp_ty2"])
    b.add("Sub", ["sp_ty2", "sp_fhalf"], ["sp_ty3"])
    b.add("Ceil", ["sp_ty3"], ["sp_cy"])
    # Two forms: the same-size shader uses tid + ceil(disp); the rescale form
    # already yields the destination coordinate. They differ by one float32
    # rounding at a tie (one byte at 1920x1080), so select at runtime rather
    # than assuming the forms are bit-identical.
    b.add("Equal", ["src_w", "dst_w"], ["sp_same_w"])
    b.add("Equal", ["src_h", "dst_h"], ["sp_same_h"])
    b.add("And", ["sp_same_w", "sp_same_h"], ["sp_same"])
    b.add("Ceil", ["sp_dx"], ["sp_dx_ceil"])
    b.add("Ceil", ["sp_dy"], ["sp_dy_ceil"])
    b.add("Add", ["sp_tid_x2", "sp_dx_ceil"], ["sp_same_x"])
    b.add("Add", ["sp_tid_y2", "sp_dy_ceil"], ["sp_same_y"])
    b.add("Where", ["sp_same", "sp_same_x", "sp_cx"], ["sp_dst_x_f"])
    b.add("Where", ["sp_same", "sp_same_y", "sp_cy"], ["sp_dst_y_f"])
    b.add("Cast", ["sp_dst_x_f"], ["sp_dst_x"], to=TensorProto.INT64)
    b.add("Cast", ["sp_dst_y_f"], ["sp_dst_y"], to=TensorProto.INT64)

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

    b.add("Mul", ["dst_w", "dst_h"], ["sp_plane"])
    b.add("Mul", ["src_h", "src_w"], ["sp_pixels"])
    indices, updates = [], []
    for ox in (0, 1):
        for oy in (0, 1):
            tag = "sp_f%d%d" % (ox, oy)
            b.const(tag + "_ox", "i64", [ox])
            b.const(tag + "_oy", "i64", [oy])
            b.add("Add", ["sp_dst_x", tag + "_ox"], [tag + "_x"])
            b.add("Add", ["sp_dst_y", tag + "_oy"], [tag + "_y"])
            b.add("GreaterOrEqual", [tag + "_x", "sp_i0_64"], [tag + "_x0"])
            b.add("Less", [tag + "_x", "dst_w"], [tag + "_x1"])
            b.add("GreaterOrEqual", [tag + "_y", "sp_i0_64"], [tag + "_y0"])
            b.add("Less", [tag + "_y", "dst_h"], [tag + "_y1"])
            b.add("And", [tag + "_x0", tag + "_x1"], [tag + "_xok"])
            b.add("And", [tag + "_y0", tag + "_y1"], [tag + "_yok"])
            b.add("And", [tag + "_xok", tag + "_yok"], [tag + "_in"])
            b.add("And", ["sp_finite", tag + "_in"], [tag + "_ok"])
            b.add("Mul", [tag + "_y", "dst_w"], [tag + "_row"])
            b.add("Add", [tag + "_row", tag + "_x"], [tag + "_flat"])
            for plane in (0, 1):
                ptag = tag + "_p%d" % plane
                b.add("Where", [tag + "_ok", tag + "_flat", "sp_i0_64"],
                      [ptag + "_idx0"])
                key = "sp_key_x" if plane == 0 else "sp_key_y"
                b.add("Where", [tag + "_ok", key, "sp_u0"],
                      [ptag + "_upd0"])
                if plane:
                    b.add("Add", [ptag + "_idx0", "sp_plane"],
                          [ptag + "_idx1"])
                else:
                    b.add("Identity", [ptag + "_idx0"], [ptag + "_idx1"])
                b.add("Reshape", [ptag + "_idx1", "sp_pixels"], [ptag + "_idx"])
                b.add("Reshape", [ptag + "_upd0", "sp_pixels"], [ptag + "_upd"])
                indices.append(ptag + "_idx")
                updates.append(ptag + "_upd")

    b.add("ConstantOfShape", ["alloc_words"], ["sp_zeros"],
          value=numpy_helper.from_array(np.array([0], dtype=np.uint32)))
    b.add("Concat", indices, ["sp_indices_flat"], axis=0)
    b.add("Concat", updates, ["sp_updates"], axis=0)
    b.const("sp_index_shape", "i64", [-1, 1])
    b.add("Reshape", ["sp_indices_flat", "sp_index_shape"], ["sp_indices"])
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
    b.const("sp_four", "i64", [4])
    b.add("Mul", ["alloc_words", "sp_four"], ["sp_byte_count"])
    b.add("Reshape", ["sp_bytes2d", "sp_byte_count"], ["sp_bytes"])
    b.add("Cast", ["sp_bytes"], ["u0"], to=TensorProto.UINT8)

    inputs = [
        helper.make_tensor_value_info("x0", TensorProto.FLOAT16,
                                      ["TH", "TW", 2]),
        helper.make_tensor_value_info("x1", TensorProto.FLOAT, ["TH", "TW"]),
    ]
    for name in ("src_w", "src_h", "dst_w", "dst_h", "alloc_words",
                 "sp_motion_off_x", "sp_motion_off_y",
                 "sp_depth_off_x", "sp_depth_off_y"):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, [1]))
    outputs = [helper.make_tensor_value_info(
        "u0", TensorProto.UINT8, ["UB"])]
    graph = helper.make_graph(b.nodes, "xefg_splat_dynamic_d%04d"
                              % p["dispatch"], inputs, outputs,
                              b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def _splat_half_bits(builder, tag, disp, sign_source, finite):
    """float32 -> fp16 bit pattern, copied from export_onnx."""
    from . import export_onnx
    return export_onnx._splat_half_bits(builder, tag, disp, sign_source, finite)


def _splat_plan(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    dispatch = [row for node, row in numbered_dispatches(run_dir, frame, rows)
                if node == 24][0]
    plan, error = recon.splat_plan(run_dir, dispatch)
    if error:
        raise SystemExit("splat: %s" % error)
    plan["capture_dispatch"] = dispatch["dispatch"]
    plan["dispatch"] = 24
    return plan


def _splat_feeds(plan):
    feeds = {"x0": plan["motion_tex"], "x1": plan["depth_tex"]}
    for name, value in (
            ("src_w", plan["src_w"]), ("src_h", plan["src_h"]),
            ("dst_w", plan["dst_w"]), ("dst_h", plan["dst_h"]),
            ("alloc_words", plan["allocation_words"]),
            ("sp_motion_off_x", plan["offset_x"]),
            ("sp_motion_off_y", plan["offset_y"]),
            ("sp_depth_off_x", plan["depth_offset_x"]),
            ("sp_depth_off_y", plan["depth_offset_y"])):
        feeds[name] = np.asarray([value], dtype=np.int64)
    return feeds


def _splat_fingerprint(plan):
    return hashlib.sha256(json.dumps({
        "scale_x": float(plan["scale_x"]),
        "scale_y": float(plan["scale_y"]),
    }, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# resample (d11, d12, d23)
# --------------------------------------------------------------------------

def _resample_plans(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    plans = {}
    for number, dispatch in numbered_dispatches(run_dir, frame, rows):
        if number not in (11, 12, 23):
            continue
        plan, error = recon.resample_plan(run_dir, dispatch)
        if error:
            raise SystemExit("d%04d resample: %s" % (number, error))
        c = recon.constants_of(dispatch)
        nconst = len(dispatch["constants"][0]["values"])
        if nconst == 14:
            scale = (recon._splat_f32(c[10]), recon._splat_f32(c[11]))
            ratio = (recon._splat_f32(c[12]), recon._splat_f32(c[13]))
        else:
            scale = (recon._splat_f32(c[8]), recon._splat_f32(c[9]))
            ratio = (recon._splat_f32(c[10]), recon._splat_f32(c[11]))
        plan.update({
            "dispatch": number,
            "capture_dispatch": dispatch["dispatch"],
            "scale_x": np.float32(scale[0]), "scale_y": np.float32(scale[1]),
            "ratio_x": np.float32(ratio[0]), "ratio_y": np.float32(ratio[1]),
            "stride": c[0],
            "source_cell": 4 if (plan["residual"] and not plan["source_f16"])
                            else 8,
        })
        plans[number] = plan
    if sorted(plans) != [11, 12, 23]:
        raise SystemExit("found resample dispatches %s" % sorted(plans))
    return plans


def _resample_weights(builder, tag, src_w, src_h, dst_w, dst_h):
    """ix/iy and the four float32 bilinear weights, in the DXAH order."""
    b = builder
    scalar_const(b, "rs_zero_scalar", 0)
    scalar_const(b, "rs_one_scalar", 1)
    b.add("Range", ["rs_zero_scalar", "rs_dst_w_scalar", "rs_one_scalar"],
          ["rs_x_range"])
    b.add("Range", ["rs_zero_scalar", "rs_dst_h_scalar", "rs_one_scalar"],
          ["rs_y_range"])
    b.add("Cast", ["rs_x_range"], ["rs_x"], to=TensorProto.FLOAT)
    b.add("Cast", ["rs_y_range"], ["rs_y"], to=TensorProto.FLOAT)
    b.const("rs_ax0", "i64", [0])
    b.const("rs_ax1", "i64", [1])
    b.add("Unsqueeze", ["rs_x", "rs_ax0"], ["rs_x2"])
    b.add("Unsqueeze", ["rs_y", "rs_ax1"], ["rs_y2"])

    b.const("rs_f0", "fp32", 0.0)
    b.const("rs_f1", "fp32", 1.0)
    b.const("rs_fhalf", "fp32", 0.5)
    b.add("Sub", ["ratio_x", "rs_f1"], ["rs_rm1x"])
    b.add("Mul", ["rs_fhalf", "rs_rm1x"], ["rs_offx"])
    b.add("Sub", ["rs_x2", "rs_offx"], ["rs_xc"])
    b.add("Mul", ["scale_x", "rs_xc"], ["rs_ax0v"])
    b.add("Max", ["rs_ax0v", "rs_f0"], ["rs_ax"])
    b.add("Sub", ["ratio_y", "rs_f1"], ["rs_rm1y"])
    b.add("Mul", ["rs_fhalf", "rs_rm1y"], ["rs_offy"])
    b.add("Sub", ["rs_y2", "rs_offy"], ["rs_yc"])
    b.add("Mul", ["scale_y", "rs_yc"], ["rs_ay0"])
    b.add("Max", ["rs_ay0", "rs_f0"], ["rs_ay"])
    b.add("Floor", ["rs_ax"], ["rs_ixf"])
    b.add("Floor", ["rs_ay"], ["rs_iyf"])
    b.add("Cast", ["rs_ixf"], ["rs_ix"], to=TensorProto.INT32)
    b.add("Cast", ["rs_iyf"], ["rs_iy"], to=TensorProto.INT32)
    b.add("Add", ["rs_ixf", "rs_f1"], ["rs_ix1f"])
    b.add("Add", ["rs_iyf", "rs_f1"], ["rs_iy1f"])
    b.add("Sub", ["rs_ix1f", "rs_ax"], ["rs_dx1"])
    b.add("Sub", ["rs_ax", "rs_ixf"], ["rs_dx0"])
    b.add("Sub", ["rs_iy1f", "rs_ay"], ["rs_dy1"])
    b.add("Sub", ["rs_ay", "rs_iyf"], ["rs_dy0"])
    b.add("Mul", ["rs_dy1", "rs_dx1"], ["rs_w0"])
    b.add("Mul", ["rs_dy1", "rs_dx0"], ["rs_w1"])
    b.add("Mul", ["rs_dy0", "rs_dx1"], ["rs_w2"])
    b.add("Mul", ["rs_dy0", "rs_dx0"], ["rs_w3"])

    b.const("rs_i0", "i32", [0])
    b.const("rs_i1", "i32", [1])
    b.add("Sub", ["src_w", "rs_i1_64"], ["rs_wmax"])
    b.add("Sub", ["src_h", "rs_i1_64"], ["rs_hmax"])
    b.add("Cast", ["rs_wmax"], ["rs_wmax32"], to=TensorProto.INT32)
    b.add("Cast", ["rs_hmax"], ["rs_hmax32"], to=TensorProto.INT32)
    for name, source, other in (
            ("cx0", "rs_ix", "rs_i0"), ("cx1", "rs_ix1", "rs_wmax32"),
            ("cy0", "rs_iy", "rs_i0"), ("cy1", "rs_iy1", "rs_hmax32")):
        if name == "cx1":
            b.add("Add", ["rs_ix", "rs_i1"], ["rs_ix1"])
        elif name == "cy1":
            b.add("Add", ["rs_iy", "rs_i1"], ["rs_iy1"])
        max_name = "rs_wmax32" if name.startswith("cx") else "rs_hmax32"
        b.add("Clip", [source, "rs_i0", max_name], ["rs_" + name])
    return "rs_ix", "rs_iy"


def build_dynamic_resample(plan):
    p = plan
    b = Builder()
    squeeze_scalar(b, "src_w")
    squeeze_scalar(b, "src_h")
    squeeze_scalar(b, "dst_w", "rs_dst_w_scalar")
    squeeze_scalar(b, "dst_h", "rs_dst_h_scalar")
    b.const("rs_i1_64", "i64", [1])
    b.const("rs_ax22", "i64", [2, 3])
    _resample_weights(b, "rs", "src_w", "src_h", "dst_w", "dst_h")

    # Corner gather index: ((cy*stride + cx)*source_cell + channel).
    b.add("Cast", ["src_w"], ["rs_srcw_f"], to=TensorProto.FLOAT)
    b.add("Cast", ["src_h"], ["rs_srch_f"], to=TensorProto.FLOAT)
    b.const("rs_cell_i64", "i64", [int(p["source_cell"])])
    b.const("rs_zero64", "i64", [0])
    b.const("rs_one64", "i64", [1])
    scalar_const(b, "rs_ch_scalar", int(p["channels"]))
    b.add("Range", ["rs_zero_scalar", "rs_ch_scalar", "rs_one_scalar"],
          ["rs_ch_range"])
    b.const("rs_ax2", "i64", [2])
    b.const("rs_ax23", "i64", [2, 3])
    b.const("rs_ax01", "i64", [0, 1])
    b.const("rs_ax012", "i64", [0, 1, 2])
    b.add("Unsqueeze", ["rs_ch_range", "rs_ax01"], ["rs_ch_3d"])
    b.add("Unsqueeze", ["rs_ch_range", "rs_ax012"], ["rs_ch_4d"])

    corners = []
    for name, cy, cx in (("00", "rs_cy0", "rs_cx0"),
                         ("01", "rs_cy0", "rs_cx1"),
                         ("10", "rs_cy1", "rs_cx0"),
                         ("11", "rs_cy1", "rs_cx1")):
        b.add("Cast", [cy], [name + "_cy64"], to=TensorProto.INT64)
        b.add("Cast", [cx], [name + "_cx64"], to=TensorProto.INT64)
        b.add("Mul", [name + "_cy64", "stride"], [name + "_row"])
        b.add("Add", [name + "_row", name + "_cx64"], [name + "_cell"])
        b.add("Mul", [name + "_cell", "rs_cell_i64"], [name + "_base"])
        b.add("Unsqueeze", [name + "_base", "rs_ax23"], [name + "_base3"])
        b.add("Add", [name + "_base3", "rs_ch_4d"], [name + "_idx"])
        corners.append(name + "_idx")
    b.add("Concat", corners, ["rs_gather"], axis=2)

    b.add("Gather", ["x0", "rs_gather"], ["rs_corners"], axis=0)
    b.add("Cast", ["rs_corners"], ["rs_f"], to=TensorProto.FLOAT)

    values = []
    for index in range(4):
        b.const("rs_pick_%d" % index, "i64", [index])
        b.add("Gather", ["rs_f", "rs_pick_%d" % index],
              ["rs_c%d" % index], axis=2)
        b.add("Unsqueeze", ["rs_w%d" % index, "rs_ax22"], ["rs_w%d_u" % index])
        b.add("Mul", ["rs_c%d" % index, "rs_w%d_u" % index],
              ["rs_m%d" % index])
    b.add("Add", ["rs_m1", "rs_m0"], ["rs_s01"])
    b.add("Add", ["rs_m2", "rs_s01"], ["rs_s012"])
    b.add("Add", ["rs_m3", "rs_s012"], ["rs_value"])

    if p["residual"]:
        b.add("Cast", ["rs_value"], ["rs_value_d"], to=TensorProto.DOUBLE)
        b.const("rs_gain_d", "fp64", float(p["gain"]))
        b.add("Mul", ["rs_value_d", "rs_gain_d"], ["rs_scaled_d"])
        b.add("Range", ["rs_zero_scalar", "rs_dst_w_scalar", "rs_one_scalar"],
              ["rs_xg"])
        b.add("Range", ["rs_zero_scalar", "rs_dst_h_scalar", "rs_one_scalar"],
              ["rs_yg"])
        b.add("Add", ["rs_yg", "rs_one64"], ["rs_yg1"])
        b.const("rs_ax1g", "i64", [1])
        b.const("rs_ax0g", "i64", [0])
        b.add("Unsqueeze", ["rs_yg1", "rs_ax1g"], ["rs_yg2"])
        b.add("Unsqueeze", ["rs_xg", "rs_ax0g"], ["rs_xg2"])
        b.add("Mul", ["rs_yg2", "dst_w"], ["rs_rrow"])
        b.add("Add", ["rs_rrow", "rs_xg2"], ["rs_rbase"])
        b.const("rs_eight", "i64", [8])
        b.add("Mul", ["rs_rbase", "rs_eight"], ["rs_rbase8"])
        b.add("Unsqueeze", ["rs_rbase8", "rs_ax2"], ["rs_rbase3"])
        b.add("Add", ["rs_rbase3", "rs_ch_3d"], ["rs_res_index"])
        b.add("Gather", ["x1", "rs_res_index"], ["rs_res"], axis=0)
        b.add("Cast", ["rs_res"], ["rs_res_d"], to=TensorProto.DOUBLE)
        b.add("Unsqueeze", ["rs_res_d", "rs_ax2"], ["rs_res_u"])
        b.add("Add", ["rs_scaled_d", "rs_res_u"], ["rs_sum_d"])
        b.add("Cast", ["rs_sum_d"], ["rs_out"], to=TensorProto.FLOAT)
    else:
        b.tensor("rs_gain", p["gain_vector"])
        b.add("Mul", ["rs_value", "rs_gain"], ["rs_out"])
    b.const("rs_squeeze", "i64", [2])
    b.add("Squeeze", ["rs_out", "rs_squeeze"], ["y0"])

    inputs = [helper.make_tensor_value_info(
        "x0", TensorProto.FLOAT16 if p["source_f16"] else TensorProto.INT8,
        ["SRC"])]
    if p["residual"]:
        inputs.append(helper.make_tensor_value_info(
            "x1", TensorProto.FLOAT16, ["RES"]))
    for name in ("src_w", "src_h", "dst_w", "dst_h", "stride"):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, [1]))
    for name in ("scale_x", "scale_y", "ratio_x", "ratio_y"):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]))
    outputs = [helper.make_tensor_value_info(
        "y0", TensorProto.FLOAT, ["DH", "DW", int(p["channels"])])]
    graph = helper.make_graph(
        b.nodes, "xefg_resample_dynamic_d%04d" % p["dispatch"],
        inputs, outputs, b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _resample_feeds(plan):
    feeds = {"x0": plan["src"]}
    if plan["residual"]:
        feeds["x1"] = plan["res"]
    for name, value in (("src_w", plan["src_w"]), ("src_h", plan["src_h"]),
                        ("dst_w", plan["dst_w"]), ("dst_h", plan["dst_h"]),
                        ("stride", plan["stride"])):
        feeds[name] = np.asarray([value], dtype=np.int64)
    for name in ("scale_x", "scale_y", "ratio_x", "ratio_y"):
        feeds[name] = np.asarray([plan[name]], dtype=np.float32)
    return feeds


def _resample_fingerprint(plan):
    return hashlib.sha256(json.dumps({
        "channels": plan["channels"], "cell_bytes": plan["cell_bytes"],
        "start_row": plan["start_row"], "source_f16": plan["source_f16"],
        "residual": plan["residual"], "gain": float(plan["gain"]),
        "source_cell": plan["source_cell"],
    }, sort_keys=True).encode()).hexdigest()


def _resample_pack(plan, got):
    expected = plan["expected"]
    packed = np.zeros(expected.size, dtype=np.uint8)
    texels = packed.view("<u2").reshape(
        plan["rows_alloc"], plan["dst_w"], plan["cell_bytes"] // 2)
    texels[plan["start_row"]:plan["start_row"] + plan["dst_h"], :,
           :plan["channels"]] = recon._f32_to_half_bits(got)
    return packed


# --------------------------------------------------------------------------
# downsample (d01)
# --------------------------------------------------------------------------

def _downsample_plans(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    dispatch = [row for node, row in numbered_dispatches(run_dir, frame, rows)
                if node == 1][0]
    plan, error = recon.downsample_plan(run_dir, dispatch)
    if error:
        raise SystemExit("downsample: %s" % error)
    c = recon.constants_of(dispatch)
    group = dispatch.get("group")
    plan.update({
        "dispatch": 1,
        "capture_dispatch": dispatch["dispatch"],
        "rx0": c[0], "ry0": c[1], "rw0": c[2], "rh0": c[3],
        "rx1": c[4], "ry1": c[5], "rw1": c[6], "rh1": c[7],
        "off_x": c[8], "off_y": c[9],
        "group_w": group[0], "group_h": group[1],
        "tex_h": plan["src0"].shape[0], "tex_w": plan["src0"].shape[1],
    })
    return plan


def _ds_fma(builder, out, value, factor, addend, tag):
    """One-rounding float32 FMA via a float64 intermediate.

    The downsample DXIL marks its luma multiplies/adds `fast`; the GPU
    contracts them into FMAs. Reproduce that rounding (the separate
    multiply-then-add order differs at a trunc boundary on the smooth scene).
    """
    b = builder
    b.add("Cast", [value], [tag + "_a64"], to=TensorProto.DOUBLE)
    b.add("Cast", [factor], [tag + "_f64"], to=TensorProto.DOUBLE)
    b.add("Mul", [tag + "_a64", tag + "_f64"], [tag + "_p64"])
    b.add("Cast", [addend], [tag + "_b64"], to=TensorProto.DOUBLE)
    b.add("Add", [tag + "_p64", tag + "_b64"], [tag + "_s64"])
    b.add("Cast", [tag + "_s64"], [out], to=TensorProto.FLOAT)


def _ds_luma(builder, tag, input_name):
    b = builder
    b.add("Cast", [input_name], [tag + "_f"], to=TensorProto.FLOAT)
    b.add("Div", [tag + "_f", "ds_255"], [tag + "_n"])
    for index, name in ((0, "b"), (1, "g"), (2, "r")):
        b.const("%s_%s_idx" % (tag, name), "i64", [index])
        b.add("Gather", [tag + "_n", "%s_%s_idx" % (tag, name)],
              ["%s_%s" % (tag, name)], axis=2)
    b.add("Mul", [tag + "_r", "ds_2126"], [tag + "_r2"])
    _ds_fma(b, tag + "_rg", tag + "_g", "ds_7152", tag + "_r2",
            tag + "_luma_g")
    _ds_fma(b, tag + "_rgb", tag + "_b", "ds_0722", tag + "_rg",
            tag + "_luma_b")
    b.add("Mul", [tag + "_rgb", "ds_255"], [tag + "_luma"])
    b.add("Reshape", [tag + "_luma", "ds_pixels"], [tag + "_luma_flat"])
    return tag + "_luma_flat"


def _ds_region_index(builder, tag, rx, ry, rw, rh):
    """[fine_h, fine_w, 16] tap index and mask for one source region."""
    b = builder
    b.add("Range", ["ds_zero_scalar", "ds_fine_w_scalar", "ds_one_scalar"],
          [tag + "_fx"])
    b.add("Range", ["ds_zero_scalar", "ds_fine_h_scalar", "ds_one_scalar"],
          [tag + "_fy"])
    b.add("Range", ["ds_zero_scalar", "ds_four_scalar", "ds_one_scalar"],
          [tag + "_d"])
    b.add("Unsqueeze", [tag + "_fx", "ds_ax1"], [tag + "_fx2"])
    b.add("Unsqueeze", [tag + "_fy", "ds_ax1"], [tag + "_fy2"])
    b.add("Unsqueeze", [tag + "_d", "ds_ax0"], [tag + "_dx"])
    b.add("Unsqueeze", [tag + "_d", "ds_ax0"], [tag + "_dy"])

    b.add("Mul", [tag + "_fx2", "ds_four"], [tag + "_fx4"])
    b.add("Mul", [tag + "_fy2", "ds_four"], [tag + "_fy4"])
    b.add("Add", [tag + "_fx4", rx], [tag + "_fxr"])
    b.add("Sub", [tag + "_fxr", "off_x"], [tag + "_fxo"])
    b.add("Add", [tag + "_fxo", tag + "_dx"], [tag + "_sx"])   # [W,4]
    b.add("Add", [tag + "_fy4", ry], [tag + "_fyr"])
    b.add("Sub", [tag + "_fyr", "off_y"], [tag + "_fyo"])
    b.add("Add", [tag + "_fyo", tag + "_dy"], [tag + "_sy"])   # [H,4]

    b.add("Unsqueeze", [tag + "_sx", "ds_ax2"], [tag + "_sx4"])
    b.add("Unsqueeze", [tag + "_sx4", "ds_ax0"], [tag + "_sx5"])
    b.add("Unsqueeze", [tag + "_sy", "ds_ax1"], [tag + "_sy4"])
    b.add("Unsqueeze", [tag + "_sy4", "ds_ax2"], [tag + "_sy5"])

    b.add("GreaterOrEqual", [tag + "_sx5", rx], [tag + "_sx0"])
    b.add("Add", [rx, rw], [tag + "_rx1"])
    b.add("Less", [tag + "_sx5", tag + "_rx1"], [tag + "_sx1"])
    b.add("GreaterOrEqual", [tag + "_sy5", ry], [tag + "_sy0"])
    b.add("Add", [ry, rh], [tag + "_ry1"])
    b.add("Less", [tag + "_sy5", tag + "_ry1"], [tag + "_sy1"])
    b.add("And", [tag + "_sx0", tag + "_sx1"], [tag + "_okx"])
    b.add("And", [tag + "_sy0", tag + "_sy1"], [tag + "_oky"])
    b.add("And", [tag + "_okx", tag + "_oky"], [tag + "_ok"])

    b.add("Sub", ["TH", "ds_one64"], [tag + "_hmax"])
    b.add("Sub", ["TW", "ds_one64"], [tag + "_wmax"])
    b.add("Clip", [tag + "_sy5", "ds_zero64", tag + "_hmax"], [tag + "_cy"])
    b.add("Clip", [tag + "_sx5", "ds_zero64", tag + "_wmax"], [tag + "_cx"])
    b.add("Mul", [tag + "_cy", "TW"], [tag + "_row"])
    b.add("Add", [tag + "_row", tag + "_cx"], [tag + "_flat"])
    b.add("Reshape", [tag + "_flat", "ds_16_shape"], [tag + "_index"])
    b.add("Cast", [tag + "_ok"], [tag + "_maskf"], to=TensorProto.FLOAT)
    b.add("Reshape", [tag + "_maskf", "ds_16_shape"], [tag + "_mask"])
    return tag + "_index", tag + "_mask"


def _ds_sum(builder, tag, luma, index_name, mask_name):
    b = builder
    b.add("Gather", [luma, index_name], [tag + "_taps"], axis=0)
    b.add("Mul", [tag + "_taps", mask_name], [tag + "_masked"])
    b.const(tag + "_split", "i64", [1] * 16)
    names = ["%s_part%d" % (tag, i) for i in range(16)]
    b.add("Split", [tag + "_masked", tag + "_split"], names, axis=2)
    total = names[0]
    for i in range(1, 16):
        b.add("Add", [total, names[i]], [tag + "_sum%d" % i])
        total = tag + "_sum%d" % i
    b.add("Reshape", [total, "ds_fine_shape"], [tag + "_sum"])
    return tag + "_sum"


def _ds_fold_index(builder):
    b = builder
    b.add("Range", ["ds_zero_scalar", "ds_coarse_w_scalar", "ds_one_scalar"],
          ["ds_cx"])
    b.add("Range", ["ds_zero_scalar", "ds_coarse_h_scalar", "ds_one_scalar"],
          ["ds_cy"])
    b.add("Range", ["ds_zero_scalar", "ds_four_scalar", "ds_one_scalar"],
          ["ds_fd"])
    b.add("Div", ["ds_cy", "ds_two64"], ["ds_gy"])
    b.add("Mod", ["ds_cy", "ds_two64"], ["ds_gyq"])
    b.add("Div", ["ds_cx", "ds_two64"], ["ds_gx"])
    b.add("Mod", ["ds_cx", "ds_two64"], ["ds_gxq"])
    b.add("Mul", ["ds_gy", "ds_eight64"], ["ds_gy8"])
    b.add("Mul", ["ds_gyq", "ds_four"], ["ds_gyq4"])
    b.add("Add", ["ds_gy8", "ds_gyq4"], ["ds_gyb"])
    b.add("Mul", ["ds_gx", "ds_eight64"], ["ds_gx8"])
    b.add("Mul", ["ds_gxq", "ds_four"], ["ds_gxq4"])
    b.add("Add", ["ds_gx8", "ds_gxq4"], ["ds_gxb"])
    b.add("Unsqueeze", ["ds_gyb", "ds_ax1"], ["ds_gyb2"])
    b.add("Unsqueeze", ["ds_gxb", "ds_ax1"], ["ds_gxb2"])
    b.add("Unsqueeze", ["ds_fd", "ds_ax0"], ["ds_fdy"])
    b.add("Unsqueeze", ["ds_fd", "ds_ax0"], ["ds_fdx"])
    b.add("Add", ["ds_gyb2", "ds_fdy"], ["ds_ty"])   # [Hc,4]
    b.add("Add", ["ds_gxb2", "ds_fdx"], ["ds_tx"])   # [Wc,4]
    b.add("Unsqueeze", ["ds_ty", "ds_ax1"], ["ds_ty4"])
    b.add("Unsqueeze", ["ds_ty4", "ds_ax3"], ["ds_ty5"])
    b.add("Unsqueeze", ["ds_tx", "ds_ax0"], ["ds_tx4"])
    b.add("Unsqueeze", ["ds_tx4", "ds_ax2"], ["ds_tx5"])
    b.add("Mul", ["ds_ty5", "fine_w"], ["ds_frow"])
    b.add("Add", ["ds_frow", "ds_tx5"], ["ds_flat"])
    b.add("Reshape", ["ds_flat", "ds_fold_16_shape"], ["ds_fold_index"])
    return "ds_fold_index"


def _ds_quantise(builder, tag, source, scale_name):
    b = builder
    b.add("Mul", [source, scale_name], [tag + "_scaled"])
    b.add("Add", [tag + "_scaled", "ds_neg128"], [tag + "_shifted"])
    b.add("Cast", [tag + "_shifted"], [tag + "_int"], to=TensorProto.INT32)
    b.const(tag + "_m256", "i32", [256])
    b.add("Mod", [tag + "_int", tag + "_m256"], [tag + "_byte"])
    b.add("Cast", [tag + "_byte"], [tag + "_u8"], to=TensorProto.UINT8)
    return tag + "_u8"


def build_dynamic_downsample(plan):
    p = plan
    b = Builder()
    shape_dim(b, "x0", 0, "TH")
    shape_dim(b, "x0", 1, "TW")
    squeeze_scalar(b, "fine_w", "ds_fine_w_scalar")
    squeeze_scalar(b, "fine_h", "ds_fine_h_scalar")
    squeeze_scalar(b, "coarse_w", "ds_coarse_w_scalar")
    squeeze_scalar(b, "coarse_h", "ds_coarse_h_scalar")
    scalar_const(b, "ds_zero_scalar", 0)
    scalar_const(b, "ds_one_scalar", 1)
    scalar_const(b, "ds_four_scalar", 4)
    b.const("ds_zero64", "i64", [0])
    b.const("ds_one64", "i64", [1])
    b.const("ds_two64", "i64", [2])
    b.const("ds_four", "i64", [4])
    b.const("ds_eight64", "i64", [8])
    b.const("ds_16", "i64", [16])
    b.const("ds_ax0", "i64", [0])
    b.const("ds_ax1", "i64", [1])
    b.const("ds_ax2", "i64", [2])
    b.const("ds_ax3", "i64", [3])
    b.const("ds_255", "fp32", 255.0)
    b.const("ds_7152", "fp32", 0.7152)
    b.const("ds_2126", "fp32", 0.2126)
    b.const("ds_0722", "fp32", 0.0722)
    b.const("ds_neg128", "fp32", -128.0)
    b.const("ds_1_16", "fp32", 1.0 / 16.0)
    b.const("ds_1_256", "fp32", 1.0 / 256.0)
    b.add("Concat", ["fine_h", "fine_w"], ["ds_fine_shape"], axis=0)
    b.add("Concat", ["fine_h", "fine_w", "ds_16"], ["ds_16_shape"], axis=0)
    b.add("Concat", ["coarse_h", "coarse_w", "ds_16"],
          ["ds_fold_16_shape"], axis=0)
    b.add("Mul", ["TH", "TW"], ["ds_pixels"])

    luma0 = _ds_luma(b, "ds0", "x0")
    index0, mask0 = _ds_region_index(b, "ds0", "rx0", "ry0", "rw0", "rh0")
    fine_sum0 = _ds_sum(b, "ds0", luma0, index0, mask0)
    fine0 = _ds_quantise(b, "ds0f", fine_sum0, "ds_1_16")

    luma1 = _ds_luma(b, "ds1", "x1")
    index1, mask1 = _ds_region_index(b, "ds1", "rx1", "ry1", "rw1", "rh1")
    fine_sum1 = _ds_sum(b, "ds1", luma1, index1, mask1)
    fine1 = _ds_quantise(b, "ds1f", fine_sum1, "ds_1_16")

    fold = _ds_fold_index(b)
    b.add("Mul", ["fine_h", "fine_w"], ["ds_fine_pixels"])
    b.add("Concat", ["coarse_h", "coarse_w"], ["ds_coarse_shape"], axis=0)
    fold_sum = []
    for tag, source in (("ds0", "ds0_sum"), ("ds1", "ds1_sum")):
        b.add("Reshape", [source, "ds_fine_pixels"], [tag + "_sum_flat"])
        b.add("Gather", [tag + "_sum_flat", fold], [tag + "_fold_taps"], axis=0)
        b.const(tag + "_fold_split", "i64", [1] * 16)
        names = ["%s_c%d" % (tag, i) for i in range(16)]
        b.add("Split", [tag + "_fold_taps", tag + "_fold_split"], names, axis=2)
        total = names[0]
        for i in range(1, 16):
            b.add("Add", [total, names[i]], ["%s_csum%d" % (tag, i)])
            total = "%s_csum%d" % (tag, i)
        b.add("Reshape", [total, "ds_coarse_shape"], [tag + "_csum"])
        fold_sum.append((tag, tag + "_csum"))
    coarse0 = _ds_quantise(b, "ds0c", "ds0_csum", "ds_1_256")
    coarse1 = _ds_quantise(b, "ds1c", "ds1_csum", "ds_1_256")

    def pack(name, a, bname, height, width, alloc_h, alloc_w, tag):
        b.const(name + "_a_axis", "i64", [2])
        b.add("Unsqueeze", [a, name + "_a_axis"], [name + "_a"])
        b.add("Unsqueeze", [bname, name + "_a_axis"], [name + "_b"])
        b.add("Concat", [name + "_a", name + "_b"], [name + "_2ch"], axis=2)
        b.add("Sub", [alloc_h, height], [tag + "_ph"])
        b.add("Sub", [alloc_w, width], [tag + "_pw"])
        b.const(name + "_pad_pre", "i64", [0, 0, 0])
        b.const(name + "_pad_post", "i64", [0])
        b.add("Concat", [name + "_pad_pre", tag + "_ph", tag + "_pw",
                         name + "_pad_post"], [name + "_pads"], axis=0)
        b.const(name + "_zero", "u8", [0])
        b.add("Pad", [name + "_2ch", name + "_pads", name + "_zero"],
              [name + "_out"])
        return name + "_out"

    out0 = pack("ds_u0", coarse0, coarse1, "coarse_h", "coarse_w",
                "alloc0_h", "alloc0_w", "ds0p")
    out1 = pack("ds_u1", fine0, fine1, "fine_h", "fine_w",
                "alloc1_h", "alloc1_w", "ds1p")

    inputs = [
        helper.make_tensor_value_info("x0", TensorProto.UINT8, ["TH", "TW", 4]),
        helper.make_tensor_value_info("x1", TensorProto.UINT8, ["TH", "TW", 4]),
    ]
    for name in ("fine_w", "fine_h", "coarse_w", "coarse_h",
                 "rx0", "ry0", "rw0", "rh0", "rx1", "ry1", "rw1", "rh1",
                 "off_x", "off_y", "alloc0_h", "alloc0_w",
                 "alloc1_h", "alloc1_w"):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, [1]))
    outputs = [
        helper.make_tensor_value_info("ds_u0_out", TensorProto.UINT8,
                                      ["A0H", "A0W", 2]),
        helper.make_tensor_value_info("ds_u1_out", TensorProto.UINT8,
                                      ["A1H", "A1W", 2]),
    ]
    graph = helper.make_graph(b.nodes, "xefg_downsample_dynamic_d0001",
                              inputs, outputs, b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _downsample_feeds(plan):
    feeds = {"x0": plan["src0"], "x1": plan["src1"]}
    values = {
        "fine_w": plan["fine_w"], "fine_h": plan["fine_h"],
        "coarse_w": plan["coarse_w"], "coarse_h": plan["coarse_h"],
        "rx0": plan["rx0"], "ry0": plan["ry0"],
        "rw0": plan["rw0"], "rh0": plan["rh0"],
        "rx1": plan["rx1"], "ry1": plan["ry1"],
        "rw1": plan["rw1"], "rh1": plan["rh1"],
        "off_x": plan["off_x"], "off_y": plan["off_y"],
        "alloc0_h": plan["alloc0"][0], "alloc0_w": plan["alloc0"][1],
        "alloc1_h": plan["alloc1"][0], "alloc1_w": plan["alloc1"][1],
    }
    for name, value in values.items():
        feeds[name] = np.asarray([value], dtype=np.int64)
    return feeds


# --------------------------------------------------------------------------
# GEMV (d14)
# --------------------------------------------------------------------------

def _gemv_plans(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    dispatch = [row for node, row in numbered_dispatches(run_dir, frame, rows)
                if node == 14][0]
    plan, error = recon.gemv_plan(run_dir, dispatch)
    if error:
        raise SystemExit("GEMV: %s" % error)
    c = recon.constants_of(dispatch)
    plan.update({
        "dispatch": 14,
        "capture_dispatch": dispatch["dispatch"],
        "stride": c[0], "xmax": c[2], "ymax": c[3] + 1,
        "taps": list(recon.GEMV_TAPS),
    })
    return plan


def _unsigned_byte(builder, source, tag):
    b = builder
    b.const(tag + "_256", "i32", [256])
    b.const(tag + "_zero", "i32", [0])
    b.add("Less", [source, tag + "_zero"], [tag + "_neg"])
    b.add("Add", [source, tag + "_256"], [tag + "_plus"])
    b.add("Where", [tag + "_neg", tag + "_plus", source], [tag + "_byte"])
    return tag + "_byte"


def build_dynamic_gemv(plan):
    p = plan
    b = Builder()
    scalar_const(b, "gm_zero_scalar", 0)
    scalar_const(b, "gm_one_scalar", 1)
    scalar_const(b, "gm_two_scalar", 2)
    scalar_const(b, "gm_six_scalar", 6)
    b.const("gm_zero64", "i64", [0])
    b.const("gm_one64", "i64", [1])
    b.const("gm_ax0", "i64", [0])
    b.const("gm_ax1", "i64", [1])
    b.const("gm_ax2", "i64", [2])
    b.const("gm_ax12", "i64", [1, 2])
    b.const("gm_four64", "i64", [4])
    b.const("gm_six64", "i64", [6])
    b.const("gm_two64", "i64", [2])
    b.const("gm_eight64", "i64", [8])
    b.const("gm_taps64", "i64", [len(p["taps"])])

    squeeze_scalar(b, "gx", "gm_gx_scalar")
    squeeze_scalar(b, "gy", "gm_gy_scalar")
    b.add("Range", ["gm_zero_scalar", "gm_gx_scalar", "gm_one_scalar"],
          ["gm_tx"])
    b.add("Range", ["gm_zero_scalar", "gm_gy_scalar", "gm_one_scalar"],
          ["gm_ty"])
    b.add("Range", ["gm_zero_scalar", "gm_six_scalar", "gm_one_scalar"],
          ["gm_vrange"])
    b.add("Range", ["gm_zero_scalar", "gm_two_scalar", "gm_one_scalar"],
          ["gm_qrange"])
    b.const("gm_vax", "i64", [0, 1])
    b.add("Unsqueeze", ["gm_vrange", "gm_vax"], ["gm_v6"])
    b.add("Unsqueeze", ["gm_qrange", "gm_vax"], ["gm_q2"])

    halves, t4, ranges = [], [], []
    for index, (dx, dy) in enumerate(p["taps"]):
        tag = "gm_t%d" % index
        b.const(tag + "_dx", "i64", [dx])
        b.const(tag + "_dy", "i64", [dy])
        b.add("Mul", ["gm_tx", "gm_two64"], [tag + "_x2"])
        b.add("Add", [tag + "_x2", tag + "_dx"], [tag + "_x"])
        b.add("Mul", ["gm_ty", "gm_two64"], [tag + "_y2"])
        b.add("Add", [tag + "_y2", tag + "_dy"], [tag + "_y"])
        b.add("Unsqueeze", [tag + "_x", "gm_ax1"], [tag + "_xc"])
        b.add("Unsqueeze", [tag + "_y", "gm_ax0"], [tag + "_yc"])
        b.add("GreaterOrEqual", [tag + "_xc", "gm_zero64"], [tag + "_x0"])
        b.add("Less", [tag + "_xc", "xmax"], [tag + "_x1"])
        b.add("GreaterOrEqual", [tag + "_yc", "gm_one64"], [tag + "_y0"])
        b.add("Less", [tag + "_yc", "ymax"], [tag + "_y1"])
        b.add("And", [tag + "_x0", tag + "_x1"], [tag + "_xok"])
        b.add("And", [tag + "_y0", tag + "_y1"], [tag + "_yok"])
        b.add("And", [tag + "_xok", tag + "_yok"], [tag + "_ok"])
        b.add("Sub", ["xmax", "gm_one64"], [tag + "_xm1"])
        b.add("Sub", ["ymax", "gm_one64"], [tag + "_ym1"])
        b.add("Clip", [tag + "_xc", "gm_zero64", tag + "_xm1"], [tag + "_xc2"])
        b.add("Clip", [tag + "_yc", "gm_one64", tag + "_ym1"], [tag + "_yc2"])
        b.add("Mul", [tag + "_yc2", "stride"], [tag + "_row"])
        b.add("Add", [tag + "_row", tag + "_xc2"], [tag + "_lin"])
        b.add("Mul", [tag + "_lin", "gm_eight64"], [tag + "_base8"])
        b.add("Unsqueeze", [tag + "_base8", "gm_ax2"], [tag + "_base83"])
        b.add("Add", [tag + "_base83", "gm_v6"], [tag + "_h6"])
        b.add("Unsqueeze", [tag + "_h6", "gm_ax2"], [tag + "_h4"])
        halves.append(tag + "_h4")
        b.add("Mul", [tag + "_lin", "gm_four64"], [tag + "_base4"])
        b.add("Unsqueeze", [tag + "_base4", "gm_ax2"], [tag + "_base43"])
        b.add("Add", [tag + "_base43", "gm_q2"], [tag + "_q2"])
        b.add("Unsqueeze", [tag + "_q2", "gm_ax2"], [tag + "_q4"])
        t4.append(tag + "_q4")
        b.add("Cast", [tag + "_ok"], [tag + "_oki"], to=TensorProto.INT32)
        b.add("Unsqueeze", [tag + "_oki", "gm_ax2"], [tag + "_ok3"])
        ranges.append(tag + "_ok3")

    b.add("Concat", halves, ["gm_half4"], axis=2)   # [gx,gy,9,6]
    b.add("Concat", t4, ["gm_t44"], axis=2)         # [gx,gy,9,2]
    b.add("Concat", ranges, ["gm_range3"], axis=2)  # [gx,gy,9]
    b.const("gm_rows_pre", "i64", [1, 6])
    b.add("Mul", ["gx", "gy"], ["gm_cells"])
    b.add("Mul", ["gm_cells", "gm_taps64"], ["gm_index_rows"])
    b.add("Concat", ["gm_index_rows", "gm_six64"], ["gm_half_shape"], axis=0)
    b.add("Concat", ["gm_index_rows", "gm_two64"], ["gm_t4_shape"], axis=0)
    b.add("Concat", ["gm_index_rows", "gm_one64"], ["gm_range_shape"], axis=0)
    b.add("Reshape", ["gm_half4", "gm_half_shape"], ["gm_index_halves"])
    b.add("Reshape", ["gm_t44", "gm_t4_shape"], ["gm_index_t4"])
    b.add("Reshape", ["gm_range3", "gm_range_shape"], ["gm_in_range"])

    b.add("Gather", ["x0", "gm_index_halves"], ["gm_halves"], axis=0)
    b.add("Cast", ["gm_halves"], ["gm_values_f"], to=TensorProto.FLOAT)
    b.add("Cast", ["gm_values_f"], ["gm_values"], to=TensorProto.INT32)
    b.const("gm_vlo", "i32", [-128])
    b.const("gm_vhi", "i32", [127])
    b.add("Clip", ["gm_values", "gm_vlo", "gm_vhi"], ["gm_values_clipped"])

    bytes_out = []
    for k in range(6):
        b.const("gm_vstart_%d" % k, "i64", [k])
        b.const("gm_vend_%d" % k, "i64", [k + 1])
        b.const("gm_vax_%d" % k, "i64", [1])
        b.const("gm_vstep_%d" % k, "i64", [1])
        b.add("Slice", ["gm_values_clipped", "gm_vstart_%d" % k,
                        "gm_vend_%d" % k, "gm_vax_%d" % k,
                        "gm_vstep_%d" % k], ["gm_v%d" % k])
        unsigned = _unsigned_byte(b, "gm_v%d" % k, "gm_vu%d" % k)
        b.add("Mul", [unsigned, "gm_in_range"], ["gm_vm%d" % k])
        bytes_out.append("gm_vm%d" % k)

    b.add("Gather", ["x1", "gm_index_t4"], ["gm_t4g"], axis=0)
    b.add("Cast", ["gm_t4g"], ["gm_t4_i"], to=TensorProto.INT32)
    for k in range(2):
        b.const("gm_qstart_%d" % k, "i64", [k])
        b.const("gm_qend_%d" % k, "i64", [k + 1])
        b.const("gm_qaxis_%d" % k, "i64", [1])
        b.const("gm_qstep_%d" % k, "i64", [1])
        b.add("Slice", ["gm_t4_i", "gm_qstart_%d" % k, "gm_qend_%d" % k,
                        "gm_qaxis_%d" % k, "gm_qstep_%d" % k],
              ["gm_q%d" % k])
        unsigned = _unsigned_byte(b, "gm_q%d" % k, "gm_qu%d" % k)
        b.add("Mul", [unsigned, "gm_in_range"], ["gm_qm%d" % k])
    b.const("gm_kfill", "i32", [p["kfill"]])
    b.add("Mul", ["gm_kfill", "gm_in_range"], ["gm_kfill_masked"])

    packed = ["gm_qm0", "gm_qm1", "gm_vm0", "gm_vm1",
              "gm_vm2", "gm_vm3", "gm_vm4", "gm_kfill_masked"]
    b.add("Concat", packed, ["gm_packed32"], axis=1)
    b.add("Cast", ["gm_packed32"], ["gm_packed8"], to=TensorProto.INT8)
    b.const("gm_72", "i64", [72])
    b.add("Concat", ["gm_cells", "gm_72"], ["gm_packed_shape"], axis=0)
    b.add("Reshape", ["gm_packed8", "gm_packed_shape"], ["gm_packed"])

    b.tensor("gm_weights", p["weights"])
    b.add("MatMulInteger", ["gm_packed", "gm_weights"], ["gm_acc"])
    b.tensor("gm_bias", p["bias"])
    b.add("Add", ["gm_acc", "gm_bias"], ["gm_biased"])
    b.const("gm_lo", "i32", [-131072])
    b.const("gm_hi", "i32", [131071])
    b.add("Clip", ["gm_biased", "gm_lo", "gm_hi"], ["gm_clamped"])
    b.const("gm_zero", "i32", [0])
    b.add("Max", ["gm_clamped", "gm_zero"], ["gm_pos"])
    b.add("Min", ["gm_clamped", "gm_zero"], ["gm_neg"])
    b.const("gm_cscale", "i32", [p["cscale"]])
    b.add("Mul", ["gm_pos", "gm_cscale"], ["gm_pos_scaled"])
    pos = arithmetic_shift(b, "gm_pos_scaled", p["s1"], "gm_pos")
    b.tensor("gm_activation", p["activation"])
    b.add("Mul", ["gm_neg", "gm_activation"], ["gm_neg_scaled"])
    neg = arithmetic_shift(b, "gm_neg_scaled", p["s2"], "gm_neg")
    b.add("Add", [pos, neg], ["gm_requant"])
    b.const("gm_qlo", "i32", [-128])
    b.const("gm_qhi", "i32", [127])
    b.add("Clip", ["gm_requant", "gm_qlo", "gm_qhi"], ["gm_q"])
    b.add("Cast", ["gm_q"], ["gm_q8"], to=TensorProto.UINT8)

    b.const("gm_four4", "i64", [4])
    b.const("gm_16", "i64", [16])
    b.add("Concat", ["gx", "gy", "gz", "gm_four4", "gm_four4"],
          ["gm_out5_shape"], axis=0)
    b.add("Reshape", ["gm_q8", "gm_out5_shape"], ["gm_out5"])
    b.add("Transpose", ["gm_out5"], ["gm_tile"], perm=[2, 1, 0, 3, 4])
    b.add("Concat", ["gz", "gy", "gx", "gm_16"],
          ["gm_tile_shape"], axis=0)
    b.add("Reshape", ["gm_tile", "gm_tile_shape"], ["gm_tile_flat"])
    b.const("gm_crop_starts", "i64", [0, 0, 0, 0])
    b.add("Concat", ["gz", "oh", "ow", "gm_16"], ["gm_crop_ends"], axis=0)
    b.const("gm_crop_axes", "i64", [0, 1, 2, 3])
    b.const("gm_crop_steps", "i64", [1, 1, 1, 1])
    b.add("Slice", ["gm_tile_flat", "gm_crop_starts", "gm_crop_ends",
                    "gm_crop_axes", "gm_crop_steps"], ["gm_cropped"])

    b.add("Sub", ["height", "gm_one64"], ["gm_h_m1"])
    b.add("Sub", ["gm_h_m1", "oh"], ["gm_h_tail"])
    b.add("Sub", ["width", "ow"], ["gm_w_tail"])
    b.const("gm_pad_pre", "i64", [0, 1, 0, 0, 0])
    b.const("gm_pad_post", "i64", [0])
    b.add("Concat", ["gm_pad_pre", "gm_h_tail", "gm_w_tail", "gm_pad_post"],
          ["gm_out_pads"], axis=0)
    b.const("gm_out_zero", "u8", [0])
    b.add("Pad", ["gm_cropped", "gm_out_pads", "gm_out_zero"], ["u0"])

    inputs = [
        helper.make_tensor_value_info("x0", TensorProto.FLOAT16, ["GMX"]),
        helper.make_tensor_value_info("x1", TensorProto.INT8, ["GMY"]),
    ]
    for name in ("gx", "gy", "gz", "ow", "oh", "height", "width",
                 "stride", "xmax", "ymax"):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, [1]))
    outputs = [helper.make_tensor_value_info(
        "u0", TensorProto.UINT8, ["GZ", "GH", "GW", 16])]
    graph = helper.make_graph(b.nodes, "xefg_gemv_dynamic_d0014",
                              inputs, outputs, b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _gemv_feeds(plan):
    feeds = {"x0": plan["src0"], "x1": plan["src1"]}
    for name in ("gx", "gy", "gz", "ow", "oh", "height", "width",
                 "stride", "xmax", "ymax"):
        feeds[name] = np.asarray([plan[name]], dtype=np.int64)
    return feeds


def _gemv_fingerprint(plan):
    digest = hashlib.sha256()
    for array in (plan["weights"], plan["bias"], plan["activation"]):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps({
        "kfill": plan["kfill"], "cscale": plan["cscale"],
        "s1": plan["s1"], "s2": plan["s2"],
    }, sort_keys=True).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# buffer output (d10, d22)
# --------------------------------------------------------------------------

def _buffer_plans(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    plans = {}
    for number, dispatch in numbered_dispatches(run_dir, frame, rows):
        if number not in (10, 22):
            continue
        plan, error = recon.buffer_output_plan(run_dir, dispatch)
        if error:
            raise SystemExit("d%04d buffer output: %s" % (number, error))
        slots = {r["slot"]: r for r in dispatch["resources"]}
        c = recon.constants_of(dispatch)
        texel = recon.FORMAT_BYTES[slots["t0"]["format"]]
        plan.update({
            "dispatch": number,
            "capture_dispatch": dispatch["dispatch"],
            "h0": slots["t0"]["height"], "w0": slots["t0"]["width"],
            "h1": slots["t1"]["height"], "w1": slots["t1"]["width"],
            "row_stride": c[2], "row_guard": c[5],
            "alloc_bytes": plan["expected"].size,
            "texel": texel, "wpc": texel // 4,
        })
        plans[number] = plan
    if sorted(plans) != [10, 22]:
        raise SystemExit("found buffer-output dispatches %s" % sorted(plans))
    return plans


def _bo_index(builder, tag, height, width):
    """[gx*gy, trip*4] gather index for one input buffer."""
    b = builder
    b.add("Range", ["bo_zero_scalar", "bo_trip_scalar", "bo_one_scalar"],
          [tag + "_steps"])
    b.add("Div", [tag + "_steps", "wpc"], [tag + "_z"])
    b.add("Mod", [tag + "_steps", "wpc"], [tag + "_word"])
    b.add("Unsqueeze", [tag + "_z", "bo_ax0"], [tag + "_z2"])
    b.add("Unsqueeze", [tag + "_z2", "bo_ax0"], [tag + "_z3"])
    b.add("Mul", [tag + "_z3", height], [tag + "_zh"])
    b.add("Add", ["bo_ty", "bo_one64"], [tag + "_ty1"])
    b.add("Unsqueeze", [tag + "_ty1", "bo_ax1"], [tag + "_ty1a"])
    b.add("Unsqueeze", [tag + "_ty1a", "bo_ax2"], [tag + "_ty1c"])
    b.add("Add", [tag + "_zh", tag + "_ty1c"], [tag + "_row"])
    b.add("Mul", [tag + "_row", width], [tag + "_rowo"])
    b.add("Unsqueeze", ["bo_tx", "bo_ax02"], [tag + "_txc"])
    b.add("Add", [tag + "_rowo", tag + "_txc"], [tag + "_cell"])
    b.add("Mul", [tag + "_cell", "bo_texel_scalar"], [tag + "_cellt"])
    b.add("Mul", [tag + "_word", "bo_four64"], [tag + "_word4"])
    b.add("Unsqueeze", [tag + "_word4", "bo_ax0"], [tag + "_word42"])
    b.add("Unsqueeze", [tag + "_word42", "bo_ax0"], [tag + "_word43"])
    b.add("Add", [tag + "_cellt", tag + "_word43"], [tag + "_base"])
    b.add("Unsqueeze", [tag + "_base", "bo_ax3"], [tag + "_base3"])
    b.add("Add", [tag + "_base3", "bo_ch4"], [tag + "_idx4"])
    b.add("Concat", ["bo_cells", "bo_row4"], [tag + "_idx_shape"], axis=0)
    b.add("Reshape", [tag + "_idx4", tag + "_idx_shape"], [tag + "_idx"])
    return tag + "_idx"


def build_dynamic_buffer_output(plan):
    p = plan
    b = Builder()
    scalar_const(b, "bo_zero_scalar", 0)
    scalar_const(b, "bo_one_scalar", 1)
    squeeze_scalar(b, "gx", "bo_gx_scalar")
    squeeze_scalar(b, "gy", "bo_gy_scalar")
    squeeze_scalar(b, "trip", "bo_trip_scalar")
    squeeze_scalar(b, "texel", "bo_texel_scalar")
    b.const("bo_zero64", "i64", [0])
    b.const("bo_one64", "i64", [1])
    b.const("bo_four64", "i64", [4])
    b.const("bo_ax0", "i64", [0])
    b.const("bo_ax1", "i64", [1])
    b.const("bo_ax2", "i64", [2])
    b.const("bo_ax3", "i64", [3])
    b.const("bo_ax02", "i64", [0, 2])
    b.add("Range", ["bo_zero_scalar", "bo_gx_scalar", "bo_one_scalar"],
          ["bo_tx"])
    b.add("Range", ["bo_zero_scalar", "bo_gy_scalar", "bo_one_scalar"],
          ["bo_ty"])
    scalar_const(b, "bo_four_scalar", 4)
    b.add("Range", ["bo_zero_scalar", "bo_four_scalar", "bo_one_scalar"],
          ["bo_ch"])
    b.add("Unsqueeze", ["bo_ch", "bo_ax0"], ["bo_ch_u"])
    b.add("Unsqueeze", ["bo_ch_u", "bo_ax0"], ["bo_ch_u2"])
    b.add("Unsqueeze", ["bo_ch_u2", "bo_ax0"], ["bo_ch4"])
    b.add("Mul", ["gx", "gy"], ["bo_cells"])
    b.const("bo_four", "i64", [4])
    b.add("Mul", ["trip", "bo_four"], ["bo_row4"])

    idx0 = _bo_index(b, "bo0", "h0", "w0")
    idx1 = _bo_index(b, "bo1", "h1", "w1")
    b.add("Gather", ["x0", idx0], ["bo_g0"], axis=0)
    b.add("Gather", ["x1", idx1], ["bo_g1"], axis=0)
    b.add("Concat", ["bo_g0", "bo_g1"], ["bo_input"], axis=1)

    b.tensor("bo_weights", p["weights"])
    b.add("MatMulInteger", ["bo_input", "bo_weights"], ["bo_acc"])
    b.tensor("bo_seed", p["seed"])
    b.add("Add", ["bo_acc", "bo_seed"], ["bo_seeded"])
    b.const("bo_lo", "i32", [-131072])
    b.const("bo_hi", "i32", [131071])
    b.add("Clip", ["bo_seeded", "bo_lo", "bo_hi"], ["bo_clamped"])
    b.const("bo_scale", "i32", [p["scale"]])
    b.add("Mul", ["bo_clamped", "bo_scale"], ["bo_scaled"])
    shifted = arithmetic_shift(b, "bo_scaled", p["shift"], "bo")
    b.const("bo_qlo", "i32", [-128])
    b.const("bo_qhi", "i32", [127])
    b.add("Clip", [shifted, "bo_qlo", "bo_qhi"], ["bo_q"])
    b.add("Cast", ["bo_q"], ["bo_q8"], to=TensorProto.UINT8)

    b.add("Concat", ["gy", "gx"], ["bo_gyx"], axis=0)
    b.const("bo_nacc", "i64", [int(p["nacc"])])
    b.add("Concat", ["bo_gyx", "bo_nacc"], ["bo_flat_shape"], axis=0)
    b.add("Reshape", ["bo_q8", "bo_flat_shape"], ["bo_flat3"])
    b.add("Mul", ["bo_cells", "bo_nacc"], ["bo_flat_n"])
    squeeze_scalar(b, "bo_flat_n", "bo_flat_scalar")
    b.add("Range", ["bo_zero_scalar", "bo_flat_scalar", "bo_one_scalar"],
          ["bo_updates_range"])
    b.add("Reshape", ["bo_updates_range", "bo_flat_shape"],
          ["bo_update_index"])
    b.add("Reshape", ["bo_flat3", "bo_flat_n"], ["bo_flat"])
    b.add("Gather", ["bo_flat", "bo_update_index"], ["bo_updates"], axis=0)

    # scatter[k] = unit*base + offsets; cells with 2*ty >= row_guard-1 are
    # skipped. They scatter 0 at index 0 with reduction=max, which is a no-op
    # on a zero-initialised uint8 buffer.
    b.add("Mul", ["bo_ty", "row_stride"], ["bo_row_off"])
    b.add("Unsqueeze", ["bo_row_off", "bo_ax1"], ["bo_row_off2"])
    b.add("Unsqueeze", ["bo_tx", "bo_ax0"], ["bo_tx2"])
    b.add("Add", ["bo_row_off2", "bo_tx2"], ["bo_base2"])
    b.const("bo_unit", "i64", [int(p["unit"])])
    b.const("bo_half", "i64", [int(p["unit"]) // 2])
    b.const("bo_quarter", "i64", [int(p["unit"]) // 4])
    b.add("Mul", ["bo_base2", "bo_unit"], ["bo_base_unit"])
    b.add("Sub", ["row_guard", "bo_one64"], ["bo_guard_m1"])
    scatter = []
    for k in range(p["nacc"]):
        tag = "bo_s%d" % k
        if k < 16:
            coef = (int(p["unit"]) // 2) * ((k % 4) // 2)
            flat = (int(p["unit"]) // 2) * (k % 2) + k // 4
        else:
            coef = (int(p["unit"]) // 2) * ((k - 16) // 2)
            flat = (int(p["unit"]) // 4) + (int(p["unit"]) // 2) * ((k - 16) % 2)
        b.const(tag + "_coef", "i64", [coef])
        b.add("Mul", ["row_stride", tag + "_coef"], [tag + "_rh"])
        b.add("Add", ["bo_base_unit", tag + "_rh"], [tag + "_base0"])
        b.const(tag + "_flat", "i64", [flat])
        b.add("Add", [tag + "_base0", tag + "_flat"], [tag + "_v"])
        b.add("Unsqueeze", [tag + "_v", "bo_ax2"], [tag + "_v3"])
        scatter.append(tag + "_v3")
    b.add("Concat", scatter, ["bo_scatter3"], axis=2)
    b.add("Mul", ["bo_ty", "bo_one64"], ["bo_keep_ty"])
    b.const("bo_two64", "i64", [2])
    b.add("Mul", ["bo_ty", "bo_two64"], ["bo_keep2"])
    b.add("Less", ["bo_keep2", "bo_guard_m1"], ["bo_keep"])
    b.const("bo_keep_ax", "i64", [1, 2])
    b.add("Unsqueeze", ["bo_keep", "bo_keep_ax"], ["bo_keep3"])
    b.const("bo_zero_i32", "i32", [0])
    b.add("Where", ["bo_keep3", "bo_scatter3", "bo_zero64"],
          ["bo_scatter_keep"])
    b.add("Cast", ["bo_updates"], ["bo_updates_i32"], to=TensorProto.INT32)
    b.add("Where", ["bo_keep3", "bo_updates_i32", "bo_zero_i32"],
          ["bo_updates_keep"])
    b.add("Unsqueeze", ["bo_scatter_keep", "bo_ax3"], ["bo_scatter_index"])
    b.add("ConstantOfShape", ["alloc_bytes"], ["bo_zero_out"],
          value=numpy_helper.from_array(np.array([0], dtype=np.int32)))
    b.add("ScatterND", ["bo_zero_out", "bo_scatter_index",
                        "bo_updates_keep"], ["bo_scattered"], reduction="max")
    b.add("Cast", ["bo_scattered"], ["u0"], to=TensorProto.UINT8)

    inputs = [
        helper.make_tensor_value_info("x0", TensorProto.INT8, ["BOX"]),
        helper.make_tensor_value_info("x1", TensorProto.INT8, ["BOY"]),
    ]
    for name in ("gx", "gy", "trip", "texel", "wpc", "h0", "w0", "h1", "w1",
                 "row_stride", "row_guard", "alloc_bytes"):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, [1]))
    outputs = [helper.make_tensor_value_info("u0", TensorProto.UINT8, ["OB"])]
    graph = helper.make_graph(
        b.nodes, "xefg_bufout_dynamic_d%04d" % p["dispatch"],
        inputs, outputs, b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _buffer_feeds(plan):
    feeds = {"x0": plan["src0"], "x1": plan["src1"]}
    for name in ("gx", "gy", "trip", "texel", "wpc", "h0", "w0", "h1", "w1",
                 "row_stride", "row_guard", "alloc_bytes"):
        feeds[name] = np.asarray([plan[name]], dtype=np.int64)
    return feeds


def _buffer_fingerprint(plan):
    digest = hashlib.sha256()
    for array in (plan["weights"], plan["seed"]):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps({
        "nacc": plan["nacc"], "unit": plan["unit"],
        "scale": plan["scale"], "shift": plan["shift"],
        "texel": plan["texel"], "wpc": plan["wpc"],
    }, sort_keys=True).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _compare(got, expected):
    expected = expected.astype(np.uint8)
    if got.shape != expected.shape:
        if expected.size == got.size:
            expected = expected.reshape(got.shape)
        else:
            return {"error": "shape %s != %s" % (got.shape, expected.shape)}
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    return {"mismatch": mismatch,
            "max_diff": int(np.abs(diff).max()) if mismatch else 0,
            "shape": list(got.shape)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=",".join(FAMILIES),
                        help="comma-separated: " + ",".join(FAMILIES))
    parser.add_argument("--frame", type=int, default=1)
    args = parser.parse_args()

    import onnxruntime

    runs = [os.path.abspath(path) for path in args.runs.split(",") if path]
    if len(runs) < 2:
        raise SystemExit("pass at least two runs")
    only = [name for name in args.only.split(",") if name]
    unknown = set(only) - set(FAMILIES)
    if unknown:
        raise SystemExit("unknown families: %s" % sorted(unknown))
    os.makedirs(args.out, exist_ok=True)

    plans = {run: {} for run in runs}
    paths, fingerprints = {}, {}
    for run in runs:
        if "splat" in only:
            plans[run]["splat"] = _splat_plan(run, args.frame)
        if "resample" in only:
            plans[run]["resample"] = _resample_plans(run, args.frame)
        if "downsample" in only:
            plans[run]["downsample"] = _downsample_plans(run, args.frame)
        if "gemv" in only:
            plans[run]["gemv"] = _gemv_plans(run, args.frame)
        if "buffer" in only:
            plans[run]["buffer"] = _buffer_plans(run, args.frame)

    if "splat" in only:
        plan = plans[runs[0]]["splat"]
        model = build_dynamic_splat(plan)
        onnx.checker.check_model(model)
        path = os.path.join(args.out, "d0024_splat_dynamic.onnx")
        onnx.save(model, path)
        paths["splat"] = path
        fingerprints["splat"] = _splat_fingerprint(plan)
        print("built d0024_splat_dynamic.onnx")

    if "resample" in only:
        for number, plan in sorted(plans[runs[0]]["resample"].items()):
            model = build_dynamic_resample(plan)
            onnx.checker.check_model(model)
            name = "d%04d_resample_dynamic.onnx" % number
            path = os.path.join(args.out, name)
            onnx.save(model, path)
            paths[str(number)] = path
            fingerprints[str(number)] = _resample_fingerprint(plan)
            print("built %s" % name)

    if "downsample" in only:
        plan = plans[runs[0]]["downsample"]
        model = build_dynamic_downsample(plan)
        onnx.checker.check_model(model)
        path = os.path.join(args.out, "d0001_downsample_dynamic.onnx")
        onnx.save(model, path)
        paths["downsample"] = path
        fingerprints["downsample"] = "downsample"
        print("built d0001_downsample_dynamic.onnx")

    if "gemv" in only:
        plan = plans[runs[0]]["gemv"]
        model = build_dynamic_gemv(plan)
        onnx.checker.check_model(model)
        path = os.path.join(args.out, "d0014_gemv_dynamic.onnx")
        onnx.save(model, path)
        paths["gemv"] = path
        fingerprints["gemv"] = _gemv_fingerprint(plan)
        print("built d0014_gemv_dynamic.onnx")

    if "buffer" in only:
        for number, plan in sorted(plans[runs[0]]["buffer"].items()):
            model = build_dynamic_buffer_output(plan)
            onnx.checker.check_model(model)
            name = "d%04d_bufout_dynamic.onnx" % number
            path = os.path.join(args.out, name)
            onnx.save(model, path)
            paths["buf%d" % number] = path
            fingerprints["buf%d" % number] = _buffer_fingerprint(plan)
            print("built %s" % name)

    results, passed = {}, True
    for run in runs:
        per_run = {}
        if "splat" in only:
            plan = plans[run]["splat"]
            if _splat_fingerprint(plan) != fingerprints["splat"]:
                raise SystemExit("splat parameters differ in %s" % run)
            session = onnxruntime.InferenceSession(
                paths["splat"], providers=["CPUExecutionProvider"])
            got = session.run(["u0"], _splat_feeds(plan))[0]
            per_run["24"] = _compare(got, plan["expected"])
        if "resample" in only:
            for number, plan in sorted(plans[run]["resample"].items()):
                key = str(number)
                if _resample_fingerprint(plan) != fingerprints[key]:
                    raise SystemExit("d%04d resample parameters differ in %s"
                                     % (number, run))
                session = onnxruntime.InferenceSession(
                    paths[key], providers=["CPUExecutionProvider"])
                got = session.run(["y0"], _resample_feeds(plan))[0]
                per_run[key] = _compare(_resample_pack(plan, got),
                                        plan["expected"])
        if "downsample" in only:
            plan = plans[run]["downsample"]
            session = onnxruntime.InferenceSession(
                paths["downsample"], providers=["CPUExecutionProvider"])
            got0, got1 = session.run(["ds_u0_out", "ds_u1_out"],
                                     _downsample_feeds(plan))
            per_run["1_u0"] = _compare(got0, plan["expected0"])
            per_run["1_u1"] = _compare(got1, plan["expected1"])
        if "gemv" in only:
            plan = plans[run]["gemv"]
            if _gemv_fingerprint(plan) != fingerprints["gemv"]:
                raise SystemExit("GEMV parameters differ in %s" % run)
            session = onnxruntime.InferenceSession(
                paths["gemv"], providers=["CPUExecutionProvider"])
            got = session.run(["u0"], _gemv_feeds(plan))[0]
            per_run["14"] = _compare(got, plan["expected"])
        if "buffer" in only:
            for number, plan in sorted(plans[run]["buffer"].items()):
                key = "buf%d" % number
                if _buffer_fingerprint(plan) != fingerprints[key]:
                    raise SystemExit("d%04d buffer parameters differ in %s"
                                     % (number, run))
                session = onnxruntime.InferenceSession(
                    paths[key], providers=["CPUExecutionProvider"])
                got = session.run(["u0"], _buffer_feeds(plan))[0]
                per_run[str(number)] = _compare(got, plan["expected"])
        results[run] = per_run
        passed = passed and all(value.get("mismatch") == 0
                                for value in per_run.values())
        print("%-60s %s" % (os.path.basename(run),
                            "EXACT" if all(value.get("mismatch") == 0
                                           for value in per_run.values())
                            else "DIFF"))

    report = {
        "runs": runs,
        "families": only,
        "models": {name: {
            "path": paths[name],
            "sha256": hashlib.sha256(open(paths[name], "rb").read()).hexdigest(),
        } for name in paths},
        "results": {run: {number: value for number, value in per_run.items()}
                    for run, per_run in results.items()},
        "passed": passed,
    }
    path = os.path.join(args.out, "dynamic_index_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("%d models x %d runs; %s; report %s"
          % (len(paths), len(runs), "PASSED" if passed else "FAILED", path))
    return 0 if passed else 1
