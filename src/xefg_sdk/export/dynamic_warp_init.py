#!/usr/bin/env python
"""D3: one dynamic-shape model for the `d25` warp + flow init.

The static lowering (`export_onnx.build_warp_init`) bakes the cell grid, the
depth-search index tables, the four lane index tables and every output shape.
This module rebuilds all of that from runtime scalars:

  * feeds: `tw`, `th`, `plane_w`, `plane_h`, `stride` (int64[1]) and
    `motion_scale`, `ratio`, the four `lo_*` / four `hi_*` clip bounds
    (float32[1]);
  * derived in-graph: `ow = 2*tw`, `oh = 2*th`, `cells = tw*th`,
    `span_x = (plane_w-1)/(ow-1)`, `span_y = (plane_h-1)/(oh-1)`,
    `uv_x = 1/ow`, `uv_y = 1/oh`;
  * the depth-search and lane tables become `Range`/`Clip`/`Mul` arithmetic;
  * the `-128` bias pad becomes an `Expand` of one int8 row.

The weights, scale/bias tables, sampler constants and the int8 GEMV stay
initializers. One model is built from the first run and executed against every
supplied capture. Both gates are checked:

  * the graph must equal `recon.reconstruct_warp_init` byte-for-byte;
  * the graph must stay inside the shared `d25_bound.py` capture bound.
"""
# Internal legacy research module; not a supported CLI entry point.
import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper

from . import d25_bound, export_onnx, recon
from .export_onnx import (Builder, _half_bits_to_float, _lane_of,
                          _slice_channels, _warp_init_unorm8)

INT_FEEDS = ("wi_tw", "wi_th", "wi_plane_w", "wi_plane_h", "wi_stride")
FLOAT_FEEDS = ("wi_motion_scale", "wi_ratio",
               "wi_lo_ax", "wi_lo_ay", "wi_hi_ax", "wi_hi_ay",
               "wi_lo_bx", "wi_lo_by", "wi_hi_bx", "wi_hi_by")


def _shape(b, tag, parts):
    """A 1-D int64 shape tensor from ints and/or int64[1] scalar names."""
    names = []
    for number, part in enumerate(parts):
        if isinstance(part, int):
            name = "%s_c%d" % (tag, number)
            b.const(name, "i64", [part])
        else:
            name = part
        names.append(name)
    if len(names) == 1:
        return names[0]
    b.add("Concat", names, [tag], axis=0)
    return tag


def _scalar0(b, tag, source):
    """Squeeze an int64[1] feed to the scalar `Range` wants."""
    b.const(tag + "_axis", "i64", [0])
    b.add("Squeeze", [source, tag + "_axis"], [tag])
    return tag


def _cast(b, tag, source, to):
    b.add("Cast", [source], [tag], to=to)
    return tag


def _sample_graph(b, tag, tex, pos_x, pos_y):
    """Dynamic twin of `export_onnx._warp_init_sample_graph`.

    `wi_ow`/`wi_oh` are the t2/t3 extents, `wi_uv_x`/`wi_uv_y` their float32
    reciprocals, and `wi_ow_i32`/`wi_xmax_i32` the integer bounds.
    """
    b.add("Mul", [pos_x, "wi_uv_x"], [tag + "_uv_x"])
    b.add("Mul", [pos_y, "wi_uv_y"], [tag + "_uv_y"])
    b.add("Cast", [tag + "_uv_x"], [tag + "_uv_xd"], to=TensorProto.DOUBLE)
    b.add("Cast", [tag + "_uv_y"], [tag + "_uv_yd"], to=TensorProto.DOUBLE)
    b.add("Mul", [tag + "_uv_xd", "wi_ow_d"], [tag + "_tex_xu"])
    b.add("Mul", [tag + "_uv_yd", "wi_oh_d"], [tag + "_tex_yu"])
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
    b.add("Cast", [tag + "_x0"], [tag + "_x0i"], to=TensorProto.INT32)
    b.add("Cast", [tag + "_y0"], [tag + "_y0i"], to=TensorProto.INT32)
    b.add("Clip", [tag + "_x0i", "wi_i0", "wi_xmax_i32"], [tag + "_cx0"])
    b.add("Clip", [tag + "_y0i", "wi_i0", "wi_ymax_i32"], [tag + "_cy0"])
    b.add("Add", [tag + "_cx0", "wi_i1"], [tag + "_cx0p"])
    b.add("Add", [tag + "_cy0", "wi_i1"], [tag + "_cy0p"])
    b.add("Clip", [tag + "_cx0p", "wi_i0", "wi_xmax_i32"], [tag + "_cx1"])
    b.add("Clip", [tag + "_cy0p", "wi_i0", "wi_ymax_i32"], [tag + "_cy1"])
    b.add("Sub", ["wi_i256", tag + "_ix"], [tag + "_iwx"])
    b.add("Sub", ["wi_i256", tag + "_iy"], [tag + "_iwy"])
    b.add("Mul", [tag + "_iwy", tag + "_iwx"], [tag + "_w00"])
    b.add("Mul", [tag + "_iwy", tag + "_ix"], [tag + "_w01"])
    b.add("Mul", [tag + "_iy", tag + "_iwx"], [tag + "_w10"])
    b.add("Mul", [tag + "_iy", tag + "_ix"], [tag + "_w11"])
    b.const(tag + "_ax2", "i64", [2])
    taps = []
    for cy, cx, weight, name in ((tag + "_cy0", tag + "_cx0", tag + "_w00", "00"),
                                 (tag + "_cy0", tag + "_cx1", tag + "_w01", "01"),
                                 (tag + "_cy1", tag + "_cx0", tag + "_w10", "10"),
                                 (tag + "_cy1", tag + "_cx1", tag + "_w11", "11")):
        b.add("Mul", [cy, "wi_ow_i32"], [tag + "_row" + name])
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


def build_dynamic(plan, prefix=""):
    """Build one d25 graph whose geometry comes from scalar feeds."""
    p = plan
    b = Builder(p, prefix)

    # -- content constants (resolution independent) ------------------------
    b.const("wi_half_f", "fp32", [0.5])
    b.const("wi_two", "fp32", [2.0])
    b.const("wi_one_f", "fp32", [1.0])
    b.const("wi_zero_f", "fp32", [0.0])
    b.const("wi_255_d", "fp64", [255.0])
    b.const("wi_zero_d", "fp64", [0.0])
    b.const("wi_i0", "i32", [0])
    b.const("wi_i1", "i32", [1])
    b.const("wi_i2", "i32", [2])
    b.const("wi_i256", "i32", [256])
    b.const("wi_minus1", "i32", [-1])
    b.const("wi_mask7fff", "u32", [0x7FFF])
    b.const("wi_maskffff", "u32", [0xFFFF])
    b.const("wi_shift16", "u32", [16])
    b.const("wi_u0", "u32", [0])
    b.const("wi_32768", "fp32", [32768.0])
    b.const("wi_1000", "fp32", [1000.0])
    b.const("wi_one_i64", "i64", [1])
    b.const("wi_two_i64", "i64", [2])
    b.tensor("wi_range_zero", np.asarray(0, dtype=np.int64))
    b.tensor("wi_range_one", np.asarray(1, dtype=np.int64))
    b.const("wi_swizzle", "i64", [2, 1, 0, 3])

    # -- derived geometry --------------------------------------------------
    b.add("Mul", ["wi_tw", "wi_two_i64"], ["wi_ow"])
    b.add("Mul", ["wi_th", "wi_two_i64"], ["wi_oh"])
    b.add("Mul", ["wi_tw", "wi_th"], ["wi_cells"])
    b.add("Sub", ["wi_plane_w", "wi_one_i64"], ["wi_plane_wm1"])
    b.add("Sub", ["wi_plane_h", "wi_one_i64"], ["wi_plane_hm1"])
    b.add("Sub", ["wi_ow", "wi_one_i64"], ["wi_owm1"])
    b.add("Sub", ["wi_oh", "wi_one_i64"], ["wi_ohm1"])
    b.add("Sub", ["wi_tw", "wi_one_i64"], ["wi_twm1"])
    b.add("Sub", ["wi_th", "wi_one_i64"], ["wi_thm1"])

    for src, dst in (("wi_tw", "wi_tw_i32"), ("wi_th", "wi_th_i32"),
                     ("wi_ow", "wi_ow_i32"), ("wi_oh", "wi_oh_i32"),
                     ("wi_cells", "wi_cells_i32"),
                     ("wi_plane_w", "wi_plane_w_i32"),
                     ("wi_plane_h", "wi_plane_h_i32"),
                     ("wi_stride", "wi_stride_i32"),
                     ("wi_plane_wm1", "wi_pwmax"),
                     ("wi_plane_hm1", "wi_phmax"),
                     ("wi_twm1", "wi_twmax"),
                     ("wi_thm1", "wi_thmax")):
        _cast(b, dst, src, TensorProto.INT32)
    _cast(b, "wi_xmax_i32", "wi_owm1", TensorProto.INT32)
    _cast(b, "wi_ymax_i32", "wi_ohm1", TensorProto.INT32)

    # span = (plane-1)/(out-1), uv = 1/out, in float32 like the shader.
    _cast(b, "wi_plane_wm1_f", "wi_plane_wm1", TensorProto.FLOAT)
    _cast(b, "wi_plane_hm1_f", "wi_plane_hm1", TensorProto.FLOAT)
    _cast(b, "wi_owm1_f", "wi_owm1", TensorProto.FLOAT)
    _cast(b, "wi_ohm1_f", "wi_ohm1", TensorProto.FLOAT)
    _cast(b, "wi_ow_f", "wi_ow", TensorProto.FLOAT)
    _cast(b, "wi_oh_f", "wi_oh", TensorProto.FLOAT)
    b.add("Div", ["wi_plane_wm1_f", "wi_owm1_f"], ["wi_span_x"])
    b.add("Div", ["wi_plane_hm1_f", "wi_ohm1_f"], ["wi_span_y"])
    b.add("Div", ["wi_one_f", "wi_ow_f"], ["wi_uv_x"])
    b.add("Div", ["wi_one_f", "wi_oh_f"], ["wi_uv_y"])
    _cast(b, "wi_ow_d", "wi_ow", TensorProto.DOUBLE)
    _cast(b, "wi_oh_d", "wi_oh", TensorProto.DOUBLE)

    # -- texture layouts ---------------------------------------------------
    b.add("Mul", ["wi_ow", "wi_oh"], ["wi_tex_pixels"])
    _shape(b, "wi_tex_shape", ["wi_tex_pixels", 4])
    for source, tag in (("x1", "wi_t2"), ("x2", "wi_t3")):
        b.add("Reshape", [source, "wi_tex_shape"], [tag + "_packed"])
        b.add("Gather", [tag + "_packed", "wi_swizzle"], [tag + "_rgba"],
              axis=1)
    b.add("Mul", ["wi_plane_h", "wi_stride"], ["wi_t0_rows"])
    _shape(b, "wi_t0_shape", ["wi_t0_rows", 4])
    b.add("Reshape", ["x0", "wi_t0_shape"], ["wi_t0"])
    _shape(b, "wi_t4_shape", ["wi_cells", 2])
    b.add("Reshape", ["x3", "wi_t4_shape"], ["wi_t4"])

    # -- cell grid: Range broadcasts to [th, tw] ---------------------------
    tw0 = _scalar0(b, "wi_tw0", "wi_tw")
    th0 = _scalar0(b, "wi_th0", "wi_th")
    b.add("Range", ["wi_range_zero", tw0, "wi_range_one"], ["wi_tx_flat"])
    b.add("Range", ["wi_range_zero", th0, "wi_range_one"], ["wi_ty_flat"])
    _shape(b, "wi_tx_row_shape", [1, "wi_tw"])
    _shape(b, "wi_ty_col_shape", ["wi_th", 1])
    b.add("Reshape", ["wi_tx_flat", "wi_tx_row_shape"], ["wi_tx64"])
    b.add("Reshape", ["wi_ty_flat", "wi_ty_col_shape"], ["wi_ty64"])
    _cast(b, "wi_tx", "wi_tx64", TensorProto.INT32)
    _cast(b, "wi_ty", "wi_ty64", TensorProto.INT32)

    # -- depth search indices: clip(y)*tw + clip(x) ------------------------
    def cell_index(tag, y, x):
        b.add("Clip", [y, "wi_i0", "wi_thmax"], [tag + "_yc"])
        b.add("Clip", [x, "wi_i0", "wi_twmax"], [tag + "_xc"])
        b.add("Mul", [tag + "_yc", "wi_tw_i32"], [tag + "_row"])
        b.add("Add", [tag + "_row", tag + "_xc"], [tag + "_cell"])
        return _cast(b, tag + "_idx", tag + "_cell", TensorProto.INT64)

    def shifted(tag, axis, delta):
        op = "Add" if delta > 0 else "Sub"
        b.const(tag + "_d", "i32", [abs(delta)])
        b.add(op, [axis, tag + "_d"], [tag])
        return tag

    xm = shifted("wi_xm", "wi_tx", -1)
    xp = shifted("wi_xp", "wi_tx", 1)
    ym = shifted("wi_ym", "wi_ty", -1)
    yp = shifted("wi_yp", "wi_ty", 1)
    depth_index = {
        "A": cell_index("wi_idx_A", ym, xm),
        "B": cell_index("wi_idx_B", ym, xp),
        "C": cell_index("wi_idx_C", yp, xm),
        "D": cell_index("wi_idx_D", yp, xp),
        "centre": cell_index("wi_idx_centre", "wi_ty", "wi_tx"),
    }

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

    dep_a = depth("wi_da", depth_index["A"])
    dep_b = depth("wi_db", depth_index["B"])
    dep_c = depth("wi_dc", depth_index["C"])
    dep_d = depth("wi_dd", depth_index["D"])
    dep_centre = depth("wi_dm", depth_index["centre"])
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
    b.add("Clip", ["wi_xp_raw", "wi_i0", "wi_twmax"], ["wi_xp_c"])
    b.add("Clip", ["wi_yp_raw", "wi_i0", "wi_thmax"], ["wi_yp_c"])
    b.add("Mul", ["wi_yp_c", "wi_tw_i32"], ["wi_yp_row"])
    b.add("Add", ["wi_yp_row", "wi_xp_c"], ["wi_pick_cell32"])
    pick = _cast(b, "wi_pick_cell", "wi_pick_cell32", TensorProto.INT64)
    b.add("Gather", ["x6", pick], ["wi_word0"], axis=0)
    b.add("Add", [pick, "wi_cells"], ["wi_pick_plane1"])
    b.add("Gather", ["x6", "wi_pick_plane1"], ["wi_word1"], axis=0)
    # An empty picked word is a full fallback: skip the flow decode and the
    # gate, use 0.5*t4 and set flag 0 (DXIL %406 -> %460 / %483). On static
    # content no cell is empty; the branch exists for motion.
    b.add("Equal", ["wi_word0", "wi_u0"], ["wi_word0_empty"])
    b.add("BitwiseAnd", ["wi_word0", "wi_maskffff"], ["wi_f0_bits"])
    f0 = _half_bits_to_float(b, "wi_f0", "wi_f0_bits")
    b.add("BitwiseAnd", ["wi_word1", "wi_maskffff"], ["wi_f1_bits"])
    f1_raw = _half_bits_to_float(b, "wi_f1", "wi_f1_bits")
    b.add("Equal", ["wi_word1", "wi_u0"], ["wi_f1_empty"])
    b.add("Where", ["wi_f1_empty", "wi_zero_f", f1_raw], ["wi_f1"])
    f1 = "wi_f1"
    b.add("Gather", ["wi_t4", pick], ["wi_t4_at"], axis=0)
    fb0 = _lane_of(b, "wi_fb0", "wi_t4_at", 0)
    fb1 = _lane_of(b, "wi_fb1", "wi_t4_at", 1)
    b.add("Mul", ["wi_half_f", fb0], ["wi_fb0_half"])
    b.add("Mul", ["wi_half_f", fb1], ["wi_fb1_half"])

    # -- per-subpixel lane -------------------------------------------------
    lanes_q = []
    lane_u1, lane_u2, lane_u3 = [], [], []
    for number, (ax, ay) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        tag = "wi_L%d" % number
        b.const(tag + "_ax", "i32", [ax])
        b.const(tag + "_ay", "i32", [ay])
        b.add("Mul", ["wi_tx", "wi_i2"], [tag + "_2tx"])
        b.add("Mul", ["wi_ty", "wi_i2"], [tag + "_2ty"])
        b.add("Add", [tag + "_2tx", tag + "_ax"], [tag + "_ix"])
        b.add("Add", [tag + "_2ty", tag + "_ay"], [tag + "_iy"])
        _cast(b, tag + "_ixf", tag + "_ix", TensorProto.FLOAT)
        _cast(b, tag + "_iyf", tag + "_iy", TensorProto.FLOAT)
        b.add("Add", [tag + "_ixf", "wi_half_f"], [tag + "_px"])
        b.add("Add", [tag + "_iyf", "wi_half_f"], [tag + "_py"])

        # The shader's manual t0 bilinear (float32, DXIL operation order).
        b.add("Mul", [tag + "_ixf", "wi_span_x"], [tag + "_u"])
        b.add("Mul", [tag + "_iyf", "wi_span_y"], [tag + "_v"])
        b.add("Floor", [tag + "_u"], [tag + "_fu"])
        b.add("Floor", [tag + "_v"], [tag + "_fv"])
        b.add("Sub", [tag + "_u", tag + "_fu"], [tag + "_fx"])
        b.add("Sub", [tag + "_v", tag + "_fv"], [tag + "_fy"])
        _cast(b, tag + "_fui", tag + "_fu", TensorProto.INT32)
        _cast(b, tag + "_fvi", tag + "_fv", TensorProto.INT32)
        b.add("Clip", [tag + "_fui", "wi_i0", "wi_pwmax"], [tag + "_ex"])
        b.add("Clip", [tag + "_fvi", "wi_i0", "wi_phmax"], [tag + "_ey"])
        b.add("Add", [tag + "_ex", "wi_i1"], [tag + "_exp"])
        b.add("Add", [tag + "_ey", "wi_i1"], [tag + "_eyp"])
        b.add("Clip", [tag + "_exp", "wi_i0", "wi_pwmax"], [tag + "_ex1"])
        b.add("Clip", [tag + "_eyp", "wi_i0", "wi_phmax"], [tag + "_ey1"])
        grad = []
        for ey, ex, name in ((tag + "_ey", tag + "_ex", "00"),
                             (tag + "_ey", tag + "_ex1", "01"),
                             (tag + "_ey1", tag + "_ex", "10"),
                             (tag + "_ey1", tag + "_ex1", "11")):
            b.add("Mul", [ey, "wi_stride_i32"], [tag + "_row" + name])
            b.add("Add", [tag + "_row" + name, ex], [tag + "_cell" + name])
            b.add("Gather", ["wi_t0", tag + "_cell" + name],
                  [tag + "_v" + name], axis=0)
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
        b.add("Clip", [tag + "_pax", "wi_lo_ax", "wi_hi_ax"],
              [tag + "_posa_x"])
        b.add("Clip", [tag + "_pay", "wi_lo_ay", "wi_hi_ay"],
              [tag + "_posa_y"])
        b.add("Clip", [tag + "_pbx", "wi_lo_bx", "wi_hi_bx"],
              [tag + "_posb_x"])
        b.add("Clip", [tag + "_pby", "wi_lo_by", "wi_hi_by"],
              [tag + "_posb_y"])

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

        # gate LUT index: ((iy & 31) * 32 + (ix & 31))
        b.const(tag + "_m31", "i32", [31])
        b.const(tag + "_s32", "i32", [32])
        b.add("BitwiseAnd", [tag + "_iy", tag + "_m31"], [tag + "_iy31"])
        b.add("BitwiseAnd", [tag + "_ix", tag + "_m31"], [tag + "_ix31"])
        b.add("Mul", [tag + "_iy31", tag + "_s32"], [tag + "_iyrow"])
        b.add("Add", [tag + "_iyrow", tag + "_ix31"], [tag + "_gate32"])
        gate_index = _cast(b, tag + "_gate_index", tag + "_gate32",
                           TensorProto.INT64)
        b.add("Gather", ["x5", gate_index], [tag + "_gate"], axis=0)

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

        centre2 = _sample_graph(b, tag + "_c2", "wi_t2_rgba",
                                tag + "_posc_x", tag + "_posc_y")
        centre3 = _sample_graph(b, tag + "_c3", "wi_t3_rgba",
                                tag + "_posd_x", tag + "_posd_y")
        warp2 = _sample_graph(b, tag + "_w2", "wi_t2_rgba",
                              tag + "_posa_x", tag + "_posa_y")
        warp3 = _sample_graph(b, tag + "_w3", "wi_t3_rgba",
                              tag + "_posb_x", tag + "_posb_y")
        b.add("Clip", [warp2, "wi_zero_d", "wi_255_d"], [tag + "_w2c"])
        b.add("Clip", [warp3, "wi_zero_d", "wi_255_d"], [tag + "_w3c"])

        # self index: iy*ow + ix
        b.add("Mul", [tag + "_iy", "wi_ow_i32"], [tag + "_iyrow_out"])
        b.add("Add", [tag + "_iyrow_out", tag + "_ix"], [tag + "_self32"])
        self_index = _cast(b, tag + "_self_index", tag + "_self32",
                           TensorProto.INT64)
        b.add("Gather", ["wi_t2_rgba", self_index], [tag + "_self2"], axis=0)
        b.add("Gather", ["wi_t3_rgba", self_index], [tag + "_self3"], axis=0)
        _cast(b, tag + "_self2f", tag + "_self2", TensorProto.FLOAT)
        _cast(b, tag + "_self3f", tag + "_self3", TensorProto.FLOAT)
        b.add("Where", [tag + "_fallback_all", "wi_zero_f", "wi_one_f"],
              [tag + "_flag"])
        _cast(b, tag + "_flag_d", tag + "_flag", TensorProto.DOUBLE)
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
            _cast(b, tag + "_" + name + "d", source, TensorProto.DOUBLE)
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
        lane_u2.append(_warp_init_unorm8(b, tag + "_n2",
                                         tag + "_two_rgb_store"))
        three = [_slice_channels(b, tag + "_q1", centre2, 0, 1),
                 _slice_channels(b, tag + "_q2", centre2, 1, 2),
                 _slice_channels(b, tag + "_q3", centre2, 2, 3),
                 _slice_channels(b, tag + "_q4", centre3, 2, 3)]
        b.add("Concat", three, [tag + "_three_rgb"], axis=2)
        b.add("Gather", [tag + "_three_rgb", "wi_swizzle"],
              [tag + "_three_rgb_store"], axis=2)
        lane_u3.append(_warp_init_unorm8(b, tag + "_n3",
                                         tag + "_three_rgb_store"))

    # -- 76 -> 16 folded GEMV and the output layouts -----------------------
    q_parts = []
    for number, name in enumerate(lanes_q):
        b.const("wi_q_axis%d" % number, "i64", [3])
        b.add("Unsqueeze", [name, "wi_q_axis%d" % number],
              ["wi_q_u%d" % number])
        q_parts.append("wi_q_u%d" % number)
    b.add("Concat", q_parts, ["wi_q_packed"], axis=3)
    _shape(b, "wi_q_shape", ["wi_cells", 76])
    b.add("Reshape", ["wi_q_packed", "wi_q_shape"], ["wi_a76"])
    _shape(b, "wi_bias_shape", ["wi_cells", 4])
    b.const("wi_bias_row", "i8", np.full((1, 4), -128, dtype=np.int8))
    b.add("Expand", ["wi_bias_row", "wi_bias_shape"], ["wi_bias_pad"])
    b.add("Concat", ["wi_a76", "wi_bias_pad"], ["wi_a80"], axis=1)
    b.tensor("wi_weights", np.ascontiguousarray(p["weights"], dtype=np.int8))
    b.add("MatMulInteger", ["wi_a80", "wi_weights"], ["wi_acc"])
    _cast(b, "wi_acc_f", "wi_acc", TensorProto.FLOAT)
    b.tensor("wi_scale", np.ascontiguousarray(p["scale"], dtype=np.float32))
    b.tensor("wi_bias", np.ascontiguousarray(p["bias"], dtype=np.float32))
    b.add("Mul", ["wi_acc_f", "wi_scale"], ["wi_scaled"])
    b.add("Add", ["wi_scaled", "wi_bias"], ["wi_biased"])
    b.add("Floor", ["wi_biased"], ["wi_floored"])
    b.const("wi_qlo", "fp32", [0.0])
    b.const("wi_qhi", "fp32", [127.0])
    b.add("Clip", ["wi_floored", "wi_qlo", "wi_qhi"], ["wi_qclipped"])
    _cast(b, "wi_u0_flat", "wi_qclipped", TensorProto.UINT8)
    _shape(b, "wi_u0_shape", ["wi_th", "wi_tw", 16])
    b.add("Reshape", ["wi_u0_flat", "wi_u0_shape"], ["u0"])

    _shape(b, "wi_col_shape", ["wi_th", "wi_tw", 2, 2, 4])
    _shape(b, "wi_col_out", ["wi_oh", "wi_ow", 4])
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

    inputs = [
        helper.make_tensor_value_info(b.name("x0"), TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info(b.name("x1"), TensorProto.UINT8, [None]),
        helper.make_tensor_value_info(b.name("x2"), TensorProto.UINT8, [None]),
        helper.make_tensor_value_info(b.name("x3"), TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info(b.name("x4"), TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info(b.name("x5"), TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info(b.name("x6"), TensorProto.UINT32, [None]),
    ]
    inputs += [helper.make_tensor_value_info(b.name(name), TensorProto.INT64,
                                             [1]) for name in INT_FEEDS]
    inputs += [helper.make_tensor_value_info(b.name(name), TensorProto.FLOAT,
                                             [1]) for name in FLOAT_FEEDS]
    graph = helper.make_graph(
        b.nodes, b.name("xefg_warp_init_dynamic"),
        inputs,
        [helper.make_tensor_value_info(b.name("u0"), TensorProto.UINT8,
                                       [None, None, 16]),
         helper.make_tensor_value_info(b.name("u1"), TensorProto.UINT8,
                                       [None, None, 4]),
         helper.make_tensor_value_info(b.name("u2"), TensorProto.UINT8,
                                       [None, None, 4]),
         helper.make_tensor_value_info(b.name("u3"), TensorProto.UINT8,
                                       [None, None, 4])],
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def feeds(plan):
    """The runtime scalar/tensor feeds for one capture."""
    c = plan["constants"]
    values = {
        "wi_tw": np.array([plan["tw"]], dtype=np.int64),
        "wi_th": np.array([plan["th"]], dtype=np.int64),
        "wi_plane_w": np.array([plan["plane_w"]], dtype=np.int64),
        "wi_plane_h": np.array([plan["plane_h"]], dtype=np.int64),
        "wi_stride": np.array([plan["stride"]], dtype=np.int64),
        "wi_motion_scale": np.array([c["motion_scale"]], dtype=np.float32),
        "wi_ratio": np.array([c["ratio"]], dtype=np.float32),
        "wi_lo_ax": np.array([c["lo_ax"]], dtype=np.float32),
        "wi_lo_ay": np.array([c["lo_ay"]], dtype=np.float32),
        "wi_hi_ax": np.array([c["hi_ax"]], dtype=np.float32),
        "wi_hi_ay": np.array([c["hi_ay"]], dtype=np.float32),
        "wi_lo_bx": np.array([c["lo_bx"]], dtype=np.float32),
        "wi_lo_by": np.array([c["lo_by"]], dtype=np.float32),
        "wi_hi_bx": np.array([c["hi_bx"]], dtype=np.float32),
        "wi_hi_by": np.array([c["hi_by"]], dtype=np.float32),
        "x0": np.ascontiguousarray(plan["t0"], dtype=np.float32),
        "x1": np.ascontiguousarray(plan["t2"], dtype=np.uint8),
        "x2": np.ascontiguousarray(plan["t3"], dtype=np.uint8),
        "x3": np.ascontiguousarray(plan["t4"], dtype=np.float32),
        "x4": np.ascontiguousarray(plan["t5"], dtype=np.float32),
        "x5": np.ascontiguousarray(plan["t6"], dtype=np.float32),
        "x6": np.ascontiguousarray(plan["t8"], dtype=np.uint32),
    }
    return values


def run(run_dirs, out_dir, with_replay=True):
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.abspath(value) for value in run_dirs]
    plans, dispatches, errors = [], [], []
    for path in paths:
        rows = recon.read_events(path)
        dispatch = [r for r in rows if r.get("type") == "dispatch"
                    and r["frame"] == 1 and r["dispatch"] == 25][0]
        plan, error = recon.warp_init_plan(path, dispatch)
        if error:
            errors.append({"run": path, "error": error})
            continue
        plans.append(plan)
        dispatches.append(dispatch)
    if errors:
        return {"passed": False, "errors": errors}

    model = build_dynamic(plans[0])
    onnx.checker.check_model(model)
    path = os.path.join(out_dir, "d0025_warp_init_dynamic.onnx")
    onnx.save(model, path)
    sha = hashlib.sha256(model.SerializeToString()).hexdigest()
    session = export_onnx._require_onnxruntime().InferenceSession(
        path, providers=["CPUExecutionProvider"])

    results = []
    passed = True
    for run_dir, dispatch, plan in zip(paths, dispatches, plans):
        got = session.run(None, feeds(plan))
        expected = plan["expected"]
        shapes = [list(value.shape) for value in got]
        mismatch = int(sum((a.astype(np.int16) - b.astype(np.int16) != 0).sum()
                           for a, b in zip(got, expected)))
        total = int(sum(b.size for b in expected))
        max_diff = int(max(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()
                           for a, b in zip(got, expected)))
        content = d25_bound.run_content(run_dir)
        ok, reason = d25_bound.evaluate(
            mismatch, total, max_diff, plan["ow"], plan["oh"],
            planes=None, content=content)

        replay_mismatch = None
        replay_ok = True
        if with_replay:
            capture = {}
            _result, error = recon.reconstruct_warp_init(run_dir, dispatch,
                                                         capture)
            if error:
                results.append({"run": run_dir, "error": error})
                passed = False
                continue
            replay = [capture["u0"], capture["u1"], capture["u2"],
                      capture["u3"]]
            replay_mismatch = int(sum(
                (a.astype(np.int16) - b.astype(np.int16) != 0).sum()
                for a, b in zip(got, replay)))
            replay_ok = replay_mismatch == 0
            passed_reason = ("graph equals the numpy replay" if replay_ok
                             else "%d bytes differ from replay"
                             % replay_mismatch)
        else:
            passed_reason = "replay comparison disabled"

        passed = passed and ok and not reason and replay_ok
        results.append({
            "run": run_dir,
            "width": plan["ow"], "height": plan["oh"],
            "tw": plan["tw"], "th": plan["th"],
            "plane_w": plan["plane_w"], "plane_h": plan["plane_h"],
            "stride": plan["stride"],
            "shapes": shapes,
            "mismatch": mismatch, "bytes": total, "max_diff": max_diff,
            "content": content,
            "bound": d25_bound.aggregate_limit(
                total, plan["ow"], plan["oh"], content=content),
            "gate": d25_bound.gate(plan["ow"], plan["oh"], content=content),
            "capture_gate": "" if ok and not reason else reason,
            "replay_mismatch": replay_mismatch,
            "replay_gate": passed_reason,
        })

    report = {
        "model": path,
        "model_sha256": sha,
        "runs": paths,
        "feeds": list(feeds(plans[0])),
        "results": results,
        "passed": bool(passed),
    }
    report_path = os.path.join(out_dir, "dynamic_warp_init_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True,
                        help="comma-separated run directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    report = run(args.runs.split(","), args.out, not args.no_replay)
    for result in report.get("results", []):
        if "error" in result:
            print("%s ERROR %s" % (result["run"], result["error"]))
        else:
            print("%s %dx%d  capture %d/%d (max %d)  replay %s"
                  % (os.path.basename(result["run"]), result["width"],
                     result["height"], result["mismatch"], result["bytes"],
                     result["max_diff"], result["replay_mismatch"]))
    print("passed" if report.get("passed") else "FAILED")
    return 0 if report.get("passed") else 1
