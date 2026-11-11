#!/usr/bin/env python
"""D1: dynamic-shape models for the flow warp (d13) and final composite (d38).

The warp's texture sampling and the composite's packed dot are already in-graph;
only their extents change. The warp takes four `int64[1]` feeds
(`src_w`, `src_h`, `tex_w`, `out_rows`) and rebuilds its flow index, ranges and
output pad with `Range`/`Concat`. The composite derives everything from
`Shape(x0)` (feature) and `Shape(in_tN)` (full-res overlays), including the
per-pixel parity select and the alpha plane.
"""
# Internal legacy research module; not a supported CLI entry point.
import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import dynamic_index, recon
from .dynamic_onnx import Builder, scalar_const, shape_dim, squeeze_scalar


WARP_FEEDS = ("src_w", "src_h", "tex_w", "out_rows")


def _warp_sample(builder, tag, flow_x, flow_y, channel):
    """Bilinear grid sample of one int8 channel, in the DXIL operation order."""
    b = builder
    b.add("Floor", [flow_x], [tag + "_xfloor"])
    b.add("Sub", [flow_x, tag + "_xfloor"], [tag + "_fx"])
    b.add("Floor", [flow_y], [tag + "_yfloor"])
    b.add("Sub", [flow_y, tag + "_yfloor"], [tag + "_fy"])
    b.add("Cast", [tag + "_xfloor"], [tag + "_ix"], to=TensorProto.INT32)
    b.add("Cast", [tag + "_yfloor"], [tag + "_iy"], to=TensorProto.INT32)

    b.const(tag + "_ax0", "i64", [0])
    b.const(tag + "_ax1", "i64", [1])
    b.add("Cast", ["w_range"], [tag + "_xi64"], to=TensorProto.INT32)
    b.add("Cast", ["h_range"], [tag + "_yi64"], to=TensorProto.INT32)
    b.add("Unsqueeze", [tag + "_xi64", tag + "_ax0"], [tag + "_xi"])
    b.add("Unsqueeze", [tag + "_yi64", tag + "_ax1"], [tag + "_yi"])

    b.const(tag + "_zero", "i32", [0])
    b.const(tag + "_one", "i32", [1])
    b.const(tag + "_two", "i32", [2])
    b.const(tag + "_chan", "i32", [channel])
    b.add("Sub", ["src_w", "one64"], [tag + "_wmax64"])
    b.add("Sub", ["src_h", "one64"], [tag + "_hmax64"])
    b.add("Cast", [tag + "_wmax64"], [tag + "_wmax"], to=TensorProto.INT32)
    b.add("Cast", [tag + "_hmax64"], [tag + "_hmax"], to=TensorProto.INT32)
    b.add("Cast", ["tex_w"], [tag + "_texw"], to=TensorProto.INT32)

    b.add("Add", [tag + "_ix", tag + "_xi"], [tag + "_bx"])
    b.add("Add", [tag + "_iy", tag + "_yi"], [tag + "_by"])
    b.add("Add", [tag + "_bx", tag + "_one"], [tag + "_bx1"])
    b.add("Add", [tag + "_by", tag + "_one"], [tag + "_by1"])
    b.add("Clip", [tag + "_bx", tag + "_zero", tag + "_wmax"],
          [tag + "_cx0"])
    b.add("Clip", [tag + "_bx1", tag + "_zero", tag + "_wmax"],
          [tag + "_cx1"])
    b.add("Clip", [tag + "_by", tag + "_zero", tag + "_hmax"],
          [tag + "_cy0"])
    b.add("Clip", [tag + "_by1", tag + "_zero", tag + "_hmax"],
          [tag + "_cy1"])

    taps = {}
    for cy, cx, name in ((tag + "_cy0", tag + "_cx0", "00"),
                         (tag + "_cy0", tag + "_cx1", "01"),
                         (tag + "_cy1", tag + "_cx0", "10"),
                         (tag + "_cy1", tag + "_cx1", "11")):
        b.add("Mul", [cy, tag + "_texw"], [tag + "_row_" + name])
        b.add("Add", [tag + "_row_" + name, cx], [tag + "_cell_" + name])
        b.add("Mul", [tag + "_cell_" + name, tag + "_two"],
              [tag + "_byte_" + name])
        b.add("Add", [tag + "_byte_" + name, tag + "_chan"],
              [tag + "_idx_" + name])
        b.add("Gather", ["x0", tag + "_idx_" + name],
              [tag + "_v_" + name], axis=0)
        b.add("Cast", [tag + "_v_" + name], [tag + "_vf_" + name],
              to=TensorProto.FLOAT)
        taps[name] = tag + "_vf_" + name

    b.const(tag + "_onef", "fp32", 1.0)
    b.add("Sub", [tag + "_onef", tag + "_fx"], [tag + "_ifx"])
    b.add("Mul", [taps["01"], tag + "_fx"], [tag + "_m01"])
    b.add("Mul", [taps["00"], tag + "_ifx"], [tag + "_m00"])
    b.add("Add", [tag + "_m01", tag + "_m00"], [tag + "_top"])
    b.add("Mul", [taps["10"], tag + "_ifx"], [tag + "_m10"])
    b.add("Sub", [tag + "_m10", tag + "_top"], [tag + "_a"])
    b.add("Mul", [taps["11"], tag + "_fx"], [tag + "_m11"])
    b.add("Add", [tag + "_a", tag + "_m11"], [tag + "_b"])
    b.add("Mul", [tag + "_b", tag + "_fy"], [tag + "_bfy"])
    b.add("Add", [tag + "_bfy", tag + "_top"], [tag + "_res"])
    return tag + "_res"


def build_dynamic_warp(plan):
    p = plan
    b = Builder()
    for name in WARP_FEEDS:
        squeeze_scalar(b, name)
    b.const("one64", "i64", [1])
    scalar_const(b, "zero64_s", 0)
    scalar_const(b, "one64_s", 1)
    scalar_const(b, "four64_s", 4)
    b.add("Range", ["zero64_s", "src_w_scalar", "one64_s"], ["w_range"])
    b.add("Range", ["zero64_s", "src_h_scalar", "one64_s"], ["h_range"])
    b.add("Range", ["zero64_s", "four64_s", "one64_s"], ["ch_range"])

    # flow_index[h,w,c] = ((1+h)*src_w + w)*8 + c
    b.add("Add", ["h_range", "one64_s"], ["h_range1"])
    b.const("wp_ax0", "i64", [0])
    b.const("wp_ax1", "i64", [1])
    b.const("wp_ax2", "i64", [2])
    b.const("wp_ax01", "i64", [0, 1])
    b.add("Unsqueeze", ["h_range1", "wp_ax1"], ["h_rows"])
    b.add("Unsqueeze", ["w_range", "wp_ax0"], ["w_cols"])
    b.add("Mul", ["h_rows", "src_w"], ["h_base"])
    b.add("Add", ["h_base", "w_cols"], ["flow_base"])
    b.const("wp_eight", "i64", [8])
    b.add("Mul", ["flow_base", "wp_eight"], ["flow_base8"])
    b.add("Unsqueeze", ["flow_base8", "wp_ax2"], ["flow_base3"])
    b.add("Unsqueeze", ["ch_range", "wp_ax01"], ["ch_4d"])
    b.add("Add", ["flow_base3", "ch_4d"], ["flow_index"])
    b.add("Gather", ["x1", "flow_index"], ["wp_flow"], axis=0)
    b.add("Cast", ["wp_flow"], ["wp_flow_f"], to=TensorProto.FLOAT)

    flows = []
    for index in range(4):
        b.const("wp_flow_idx%d" % index, "i64", [index])
        b.add("Gather", ["wp_flow_f", "wp_flow_idx%d" % index],
              ["wp_f%d" % index], axis=2)
        b.const("wp_sq%d" % index, "i64", [2])
        b.add("Squeeze", ["wp_f%d" % index, "wp_sq%d" % index],
              ["wp_fd%d" % index])
        flows.append("wp_fd%d" % index)

    low = _warp_sample(b, "wp_low", flows[0], flows[1], 0)
    high = _warp_sample(b, "wp_high", flows[2], flows[3], 1)
    channels = []
    for name, tag in ((low, "wp_low"), (high, "wp_high")):
        b.add("Cast", [name], [tag + "_int"], to=TensorProto.INT32)
        b.const(tag + "_lo", "i32", [-128])
        b.const(tag + "_hi", "i32", [127])
        b.add("Clip", [tag + "_int", tag + "_lo", tag + "_hi"],
              [tag + "_clip"])
        b.const(tag + "_m256", "i32", [256])
        b.add("Mod", [tag + "_clip", tag + "_m256"], [tag + "_byte"])
        b.add("Cast", [tag + "_byte"], [tag + "_u8"], to=TensorProto.UINT8)
        b.const(tag + "_axis", "i64", [2])
        b.add("Unsqueeze", [tag + "_u8", tag + "_axis"],
              [tag + "_u8_3d"])
        channels.append(tag + "_u8_3d")

    b.add("Concat", ["src_h", "src_w"], ["wp_zero_hw"], axis=0)
    b.add("Concat", ["wp_zero_hw", "one64"], ["wp_zero_shape"], axis=0)
    b.add("ConstantOfShape", ["wp_zero_shape"], ["wp_zero"],
          value=numpy_helper.from_array(np.array([0], dtype=np.uint8)))
    b.add("Concat", [channels[0], channels[1], "wp_zero", "wp_zero"],
          ["wp_packed"], axis=2)

    b.add("Sub", ["out_rows", "one64"], ["wp_rows_m1"])
    b.add("Sub", ["wp_rows_m1", "src_h"], ["wp_tail"])
    b.const("wp_pad_pre", "i64", [1, 0, 0])
    b.const("wp_pad_post", "i64", [0, 0])
    b.add("Concat", ["wp_pad_pre", "wp_tail", "wp_pad_post"], ["wp_pads"],
          axis=0)
    b.const("wp_pad_zero", "u8", [0])
    b.add("Pad", ["wp_packed", "wp_pads", "wp_pad_zero"], ["u0"])

    inputs = [
        helper.make_tensor_value_info("x0", TensorProto.INT8, ["TEX"]),
        helper.make_tensor_value_info("x1", TensorProto.FLOAT16, ["FLOW"]),
    ]
    for name in WARP_FEEDS:
        inputs.append(helper.make_tensor_value_info(
            name, TensorProto.INT64, [1]))
    outputs = [
        helper.make_tensor_value_info(
            "u0", TensorProto.UINT8, ["OR", "SW", 4]),
    ]
    graph = helper.make_graph(
        b.nodes, "xefg_warp_dynamic_d%04d" % p["dispatch"], inputs, outputs,
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _overlay_channels(builder, slot):
    """BGRA8 overlay -> float32 (r, g, b, a), each [height, width, 1]."""
    b = builder
    b.add("Cast", ["in_" + slot], ["cp_" + slot + "_f"],
          to=TensorProto.FLOAT)
    b.add("Div", ["cp_" + slot + "_f", "cp_255"], ["cp_" + slot + "_n"])
    channels = []
    for index, tag in ((2, "r"), (1, "g"), (0, "b"), (3, "a")):
        b.const("cp_%s_%s_idx" % (slot, tag), "i64", [index])
        b.add("Gather", ["cp_" + slot + "_n", "cp_%s_%s_idx" % (slot, tag)],
              ["cp_%s_%s" % (slot, tag)], axis=2)
        channels.append("cp_%s_%s" % (slot, tag))
    return channels


def _gate(builder, name, index):
    b = builder
    b.const(name + "_idx%d" % index, "i64", [index])
    b.add("Gather", [name, name + "_idx%d" % index],
          [name + "_g%d" % index], axis=2)
    return name + "_g%d" % index


def build_dynamic_composite(plan):
    p = plan
    b = Builder()
    shape_dim(b, "x0", 0, "FH")
    shape_dim(b, "x0", 1, "FW")
    first_overlay = next(iter(p["overlays"]))
    shape_dim(b, "in_" + first_overlay, 0, "H")
    shape_dim(b, "in_" + first_overlay, 1, "W")
    b.add("Mul", ["H", "W"], ["cp_n"])

    b.const("cp_zero_f", "fp32", 0.0)
    b.const("cp_one_f", "fp32", 1.0)
    b.const("cp_two_f", "fp32", 2.0)
    b.const("cp_three_f", "fp32", 3.0)
    b.const("cp_half", "fp32", 0.5)
    b.const("cp_255", "fp32", 255.0)

    b.add("Cast", ["x0"], ["cp_feat_i8"], to=TensorProto.INT8)
    b.const("cp_one64", "i64", [1])
    b.const("cp_16", "i64", [16])
    b.const("cp_5", "i64", [5])
    b.add("Concat", ["FH", "cp_one64", "FW", "cp_one64", "cp_16"],
          ["cp_feat5_shape"], axis=0)
    b.add("Reshape", ["cp_feat_i8", "cp_feat5_shape"], ["cp_feat5"])
    b.const("cp_tile", "i64", [1, 2, 1, 2, 1])
    b.add("Tile", ["cp_feat5", "cp_tile"], ["cp_feat_tiled"])
    b.add("Concat", ["cp_n", "cp_16"], ["cp_feat_shape"], axis=0)
    b.add("Reshape", ["cp_feat_tiled", "cp_feat_shape"], ["cp_feat"])

    weights = np.zeros((16, 20), dtype=np.int8)
    for gate in range(5):
        for parity in range(4):
            weights[:, gate * 4 + parity] = p["weights"][gate * 4 + parity]
    b.tensor("cp_weights", weights)
    b.add("MatMulInteger", ["cp_feat", "cp_weights"], ["cp_acc"])

    # select[h,w,gate] = gate*4 + (w&1) + 2*(h&1), flattened to [N,5].
    squeeze_scalar(b, "H")
    squeeze_scalar(b, "W")
    scalar_const(b, "cp_zero64", 0)
    scalar_const(b, "cp_one64_s", 1)
    scalar_const(b, "cp_two64", 2)
    scalar_const(b, "cp_five64", 5)
    b.add("Range", ["cp_zero64", "W_scalar", "cp_one64_s"], ["cp_xs"])
    b.add("Range", ["cp_zero64", "H_scalar", "cp_one64_s"], ["cp_ys"])
    b.add("Range", ["cp_zero64", "cp_five64", "cp_one64_s"], ["cp_gates"])
    b.add("Mod", ["cp_xs", "cp_two64"], ["cp_xpar"])
    b.add("Mod", ["cp_ys", "cp_two64"], ["cp_ypar"])
    b.const("cp_ax0", "i64", [0])
    b.const("cp_ax1", "i64", [1])
    b.const("cp_ax2", "i64", [2])
    b.const("cp_ax01", "i64", [0, 1])
    b.add("Unsqueeze", ["cp_xpar", "cp_ax0"], ["cp_xpar2"])
    b.add("Unsqueeze", ["cp_ypar", "cp_ax1"], ["cp_ypar2"])
    b.add("Mul", ["cp_ypar2", "cp_two64"], ["cp_ypar4"])
    b.add("Add", ["cp_xpar2", "cp_ypar4"], ["cp_parity"])
    b.add("Unsqueeze", ["cp_parity", "cp_ax2"], ["cp_parity3"])
    b.const("cp_four64", "i64", [4])
    b.add("Mul", ["cp_gates", "cp_four64"], ["cp_gate4"])
    b.add("Unsqueeze", ["cp_gate4", "cp_ax01"], ["cp_gate4_3"])
    b.add("Add", ["cp_parity3", "cp_gate4_3"], ["cp_select_hw5"])
    b.add("Concat", ["cp_n", "cp_5"], ["cp_n5_shape"], axis=0)
    b.add("Reshape", ["cp_select_hw5", "cp_n5_shape"], ["cp_select"])
    b.const("cp_flat_shape", "i64", [-1])
    b.add("Reshape", ["cp_select_hw5", "cp_flat_shape"], ["cp_select_flat"])
    b.add("GatherElements", ["cp_acc", "cp_select"], ["cp_acc_sel"], axis=1)

    b.tensor("cp_scales", p["scales"])
    b.tensor("cp_biases", p["biases"])
    b.add("Gather", ["cp_scales", "cp_select_flat"], ["cp_scale_row"],
          axis=0)
    b.add("Gather", ["cp_biases", "cp_select_flat"], ["cp_bias_row"],
          axis=0)
    b.add("Reshape", ["cp_scale_row", "cp_n5_shape"], ["cp_scale"])
    b.add("Reshape", ["cp_bias_row", "cp_n5_shape"], ["cp_bias"])
    b.add("Cast", ["cp_acc_sel"], ["cp_acc_f"], to=TensorProto.FLOAT)
    b.add("Mul", ["cp_acc_f", "cp_scale"], ["cp_v_scaled"])
    b.add("Add", ["cp_v_scaled", "cp_bias"], ["cp_v"])
    # Build the gate as the same float32 chain as the numpy replay
    # (`1/(1+exp(-v))`); the fused Sigmoid kernel rounds differently on ties.
    b.add("Neg", ["cp_v"], ["cp_neg_v"])
    b.add("Exp", ["cp_neg_v"], ["cp_exp_v"])
    b.add("Add", ["cp_exp_v", "cp_one_f"], ["cp_den_v"])
    b.add("Div", ["cp_one_f", "cp_den_v"], ["cp_s_flat"])
    b.add("Concat", ["H", "W", "cp_5"], ["cp_hw5"], axis=0)
    b.add("Reshape", ["cp_s_flat", "cp_hw5"], ["cp_s"])

    a = _overlay_channels(b, "t1")
    bb = _overlay_channels(b, "t2")
    c = _overlay_channels(b, "t3")
    d = _overlay_channels(b, "t11")
    e = _overlay_channels(b, "t12")
    s0 = _gate(b, "cp_s", 0)
    s1 = _gate(b, "cp_s", 1)
    s2 = _gate(b, "cp_s", 2)
    s3 = _gate(b, "cp_s", 3)
    s4 = _gate(b, "cp_s", 4)

    def fma(out, a, factor, addend, tag):
        """One-rounding float32 FMA via a float64 product/sum.

        The DXIL's `fmul fast` + `fadd fast` contract, so the ONNX graph must
        round once exactly where the numpy replay does.
        """
        b.add("Cast", [a], [tag + "_a64"], to=TensorProto.DOUBLE)
        b.add("Cast", [factor], [tag + "_f64"], to=TensorProto.DOUBLE)
        b.add("Mul", [tag + "_a64", tag + "_f64"], [tag + "_p64"])
        b.add("Cast", [addend], [tag + "_b64"], to=TensorProto.DOUBLE)
        b.add("Add", [tag + "_p64", tag + "_b64"], [tag + "_s64"])
        b.add("Cast", [tag + "_s64"], [out], to=TensorProto.FLOAT)

    def blend(ac, aa, bc, cc, dc, ec, tag):
        def n(name):
            return tag + name

        b.add("Sub", [ac, bc], [n("ab")])
        fma(n("m0"), n("ab"), s2, bc, n("fma0"))
        b.add("Sub", [aa, n("m0")], [n("aa_m0")])
        b.add("Sub", [cc, aa], [n("c_aa")])
        fma(n("m1"), n("c_aa"), s0, n("aa_m0"), n("fma1"))
        fma(n("m2"), n("m1"), s1, n("m0"), n("fma2"))
        b.add("Sub", [dc, ec], [n("de")])
        fma(n("m3"), n("de"), s3, ec, n("fma3"))
        b.add("Sub", [n("m2"), n("m3")], [n("m2_m3")])
        fma(n("m"), n("m2_m3"), s4, n("m3"), n("fma4"))
        return n("m"), n("m2")

    m, m2 = [], []
    for channel, (ac, aa, bc, cc, dc, ec) in enumerate((
            (a[0], a[3], bb[0], c[0], d[0], e[0]),
            (a[1], bb[3], bb[1], c[1], d[1], e[1]),
            (a[2], c[3], bb[2], c[2], d[2], e[2]))):
        m_name, m2_name = blend(ac, aa, bc, cc, dc, ec,
                                "cp_ch%d_" % channel)
        m.append(m_name)
        m2.append(m2_name)

    if p["mode"] == 2:
        q = _overlay_channels(b, "t9")
        t10 = _overlay_channels(b, "t10")
        b.const("cp_ten", "fp32", 10.0)
        b.add("Mul", [q[3], "cp_ten"], ["cp_q10"])
        b.add("Clip", ["cp_q10", "cp_zero_f", "cp_one_f"], ["cp_f"])
        b.add("Mul", ["cp_f", "cp_f"], ["cp_f2"])
        b.add("Mul", ["cp_f", "cp_two_f"], ["cp_2f"])
        b.add("Sub", ["cp_three_f", "cp_2f"], ["cp_3_2f"])
        b.add("Mul", ["cp_f2", "cp_3_2f"], ["cp_w"])
        out = []
        for channel in range(3):
            b.add("Sub", [t10[channel], m2[channel]],
                  ["cp_o_t10_%d" % channel])
            b.add("Mul", ["cp_o_t10_%d" % channel, "cp_w"],
                  ["cp_o_w_%d" % channel])
            b.add("Add", [m2[channel], "cp_o_w_%d" % channel],
                  ["cp_out_%d" % channel])
            out.append("cp_out_%d" % channel)
    elif p["mode"] == 3:
        q = _overlay_channels(b, "t9")
        out = []
        for channel in range(3):
            b.add("Sub", ["cp_one_f", q[3]], ["cp_1qa_%d" % channel])
            fma("cp_out_%d" % channel, m[channel], "cp_1qa_%d" % channel,
                q[channel], "cp_fmaq_%d" % channel)
            out.append("cp_out_%d" % channel)
    elif p["mode"] in (4, 5):
        t8 = _overlay_channels(b, "t8")
        t10 = _overlay_channels(b, "t10")
        q = _overlay_channels(b, "t9") if p["mode"] == 5 else None
        pcol = []
        for channel in range(3):
            if p["mode"] == 4:
                pcol.append(t8[channel])
            else:
                b.add("Sub", ["cp_one_f", q[3]],
                      ["cp_1q_%d" % channel])
                b.add("Mul", [t8[channel], "cp_1q_%d" % channel],
                      ["cp_t8q_%d" % channel])
                b.add("Add", [q[channel], "cp_t8q_%d" % channel],
                      ["cp_p_%d" % channel])
                pcol.append("cp_p_%d" % channel)
        accum = None
        for channel in range(3):
            b.add("Sub", [pcol[channel], t10[channel]],
                  ["cp_diff_%d" % channel])
            b.add("Mul", ["cp_diff_%d" % channel, "cp_diff_%d" % channel],
                  ["cp_diff2_%d" % channel])
            if channel == 0:
                accum = "cp_diff2_0"
            else:
                previous = ("cp_diff2_0" if channel == 1 else
                            "cp_dsum_%d" % (channel - 1))
                accum = "cp_dsum_%d" % channel
                b.add("Add", [previous, "cp_diff2_%d" % channel], [accum])
        b.add("Sqrt", [accum], ["cp_dnorm"])
        b.const("cp_20", "fp32", 20.0)
        b.add("Mul", ["cp_dnorm", "cp_20"], ["cp_g20"])
        b.add("Clip", ["cp_g20", "cp_zero_f", "cp_one_f"], ["cp_g"])
        b.add("Mul", ["cp_g", "cp_g"], ["cp_g2"])
        b.add("Mul", ["cp_g", "cp_two_f"], ["cp_g2f"])
        b.add("Sub", ["cp_three_f", "cp_g2f"], ["cp_g3"])
        b.add("Mul", ["cp_g2", "cp_g3"], ["cp_w"])
        b.add("Sub", ["cp_one_f", "cp_w"], ["cp_keep"])
        out = []
        for channel in range(3):
            if p["mode"] == 4:
                # d38 uses the final full m, not m2.
                b.add("Sub", [m[channel], t8[channel]],
                      ["cp_m2t8_%d" % channel])
                b.add("Mul", ["cp_m2t8_%d" % channel, "cp_keep"],
                      ["cp_keep_m2_%d" % channel])
                b.add("Add", [t10[channel], "cp_keep_m2_%d" % channel],
                      ["cp_out_%d" % channel])
            else:
                b.add("Sub", ["cp_one_f", q[3]],
                      ["cp_1q2_%d" % channel])
                b.add("Mul", [m2[channel], "cp_1q2_%d" % channel],
                      ["cp_q2m_%d" % channel])
                b.add("Add", [q[channel], "cp_q2m_%d" % channel],
                      ["cp_q2_%d" % channel])
                b.add("Mul", ["cp_keep", pcol[channel]],
                      ["cp_kp_%d" % channel])
                b.add("Sub", [t10[channel], "cp_kp_%d" % channel],
                      ["cp_t10_kp_%d" % channel])
                b.add("Mul", ["cp_keep", "cp_q2_%d" % channel],
                      ["cp_kq_%d" % channel])
                b.add("Add", ["cp_t10_kp_%d" % channel,
                              "cp_kq_%d" % channel],
                      ["cp_out_%d" % channel])
            out.append("cp_out_%d" % channel)
    else:
        out = m

    # The texture unit performs the UNORM8 conversion on the stored float, so
    # compute the 8-bit rounding exactly; a float32 *255+0.5 rounds 241.49999
    # up to 242, while the hardware writes 241.
    b.const("cp_255d", "fp64", [255.0])
    b.const("cp_halfd", "fp64", [0.5])
    channels = []
    for source in (out[2], out[1], out[0]):
        b.add("Clip", [source, "cp_zero_f", "cp_one_f"],
              ["cp_clip_" + source])
        b.add("Cast", ["cp_clip_" + source], ["cp_clipd_" + source],
              to=TensorProto.DOUBLE)
        b.add("Mul", ["cp_clipd_" + source, "cp_255d"],
              ["cp_255d_" + source])
        b.add("Add", ["cp_255d_" + source, "cp_halfd"],
              ["cp_halfd_" + source])
        b.add("Floor", ["cp_halfd_" + source], ["cp_floord_" + source])
        b.add("Cast", ["cp_floord_" + source], ["cp_u8_" + source],
              to=TensorProto.UINT8)
        channels.append("cp_u8_" + source)
    b.add("Concat", ["H", "W", "cp_one64"], ["cp_alpha_shape"], axis=0)
    b.add("ConstantOfShape", ["cp_alpha_shape"], ["cp_alpha"],
          value=numpy_helper.from_array(np.array([255], dtype=np.uint8)))
    channels.append("cp_alpha")
    b.add("Concat", channels, ["u0"], axis=2)

    inputs = [helper.make_tensor_value_info(
        "x0", TensorProto.UINT8, ["FH", "FW", 16])]
    for slot in p["overlays"]:
        inputs.append(helper.make_tensor_value_info(
            "in_" + slot, TensorProto.UINT8, ["H", "W", 4]))
    outputs = [helper.make_tensor_value_info(
        "u0", TensorProto.UINT8, ["H", "W", 4])]
    graph = helper.make_graph(
        b.nodes, "xefg_composite_dynamic_d%04d" % p["dispatch"],
        inputs, outputs, b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _plans(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    warp = composite = None
    for number, dispatch in dynamic_index.numbered_dispatches(
            run_dir, frame, rows):
        if number == 13:
            plan, error = recon.warp_plan(run_dir, dispatch)
            if error:
                raise SystemExit("d13 warp: %s" % error)
            plan["capture_dispatch"] = dispatch["dispatch"]
            plan["dispatch"] = number
            warp = plan
        elif number == 38:
            plan, error = recon.composite_plan(run_dir, dispatch)
            if error:
                raise SystemExit("d38 composite: %s" % error)
            plan["capture_dispatch"] = dispatch["dispatch"]
            plan["dispatch"] = number
            composite = plan
    if warp is None or composite is None:
        raise SystemExit("warp or composite dispatch missing")
    return warp, composite


def _warp_feeds(plan):
    values = {"src_w": plan["src_w"], "src_h": plan["src_h"],
              "tex_w": plan["tex_w"], "out_rows": plan["out_rows"]}
    feeds = {"x0": plan["texture"], "x1": plan["flow"]}
    for name in WARP_FEEDS:
        feeds[name] = np.asarray([values[name]], dtype=np.int64)
    return feeds


def _composite_feeds(plan):
    feeds = {"x0": plan["feature"]}
    for slot, data in plan["overlays"].items():
        feeds["in_" + slot] = data
    return feeds


def _warp_fingerprint(plan):
    """The warp bakes no dimensions; every extent is a feed."""
    return hashlib.sha256(b"flow-warp").hexdigest()


def _composite_fingerprint(plan):
    digest = hashlib.sha256()
    for array in (plan["weights"], plan["scales"], plan["biases"]):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps({"mode": plan["mode"],
                              "overlays": sorted(plan["overlays"])},
                             sort_keys=True).encode())
    return digest.hexdigest()


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
    parser.add_argument("--runs", required=True,
                        help="comma-separated run directories; the first "
                             "builds the models")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame", type=int, default=1)
    args = parser.parse_args()

    import onnxruntime

    runs = [os.path.abspath(path) for path in args.runs.split(",") if path]
    if len(runs) < 2:
        raise SystemExit("pass at least two runs")
    os.makedirs(args.out, exist_ok=True)

    all_plans = {run: _plans(run, args.frame) for run in runs}
    warp0, comp0 = all_plans[runs[0]]
    model = build_dynamic_warp(warp0)
    onnx.checker.check_model(model)
    warp_path = os.path.join(args.out, "d0013_warp_dynamic.onnx")
    onnx.save(model, warp_path)
    model = build_dynamic_composite(comp0)
    onnx.checker.check_model(model)
    comp_path = os.path.join(args.out, "d0038_composite_dynamic.onnx")
    onnx.save(model, comp_path)
    print("built d0013_warp_dynamic.onnx and d0038_composite_dynamic.onnx")

    results, passed = {}, True
    for run in runs:
        warp, composite = all_plans[run]
        if _warp_fingerprint(warp) != _warp_fingerprint(warp0):
            raise SystemExit("warp dims changed between runs")
        if _composite_fingerprint(composite) != _composite_fingerprint(comp0):
            raise SystemExit("composite parameters differ in %s" % run)
        per_run = {}
        session = onnxruntime.InferenceSession(
            warp_path, providers=["CPUExecutionProvider"])
        got = session.run(["u0"], _warp_feeds(warp))[0]
        per_run[13] = _compare(got, warp["expected"])
        session = onnxruntime.InferenceSession(
            comp_path, providers=["CPUExecutionProvider"])
        got = session.run(["u0"], _composite_feeds(composite))[0]
        per_run[38] = _compare(got, composite["expected"])
        results[run] = per_run
        passed = passed and all(value.get("mismatch") == 0
                                for value in per_run.values())
        print("%-60s %s" % (os.path.basename(run),
                            "EXACT" if all(value.get("mismatch") == 0
                                           for value in per_run.values())
                            else "DIFF"))

    manifest = {
        "13": {
            "model": os.path.basename(warp_path),
            "feeds": {os.path.basename(run): {
                name: all_plans[run][0][name] for name in WARP_FEEDS
            } for run in runs},
        },
        "38": {
            "model": os.path.basename(comp_path),
            "feeds": {os.path.basename(run): {
                "height": all_plans[run][1]["height"],
                "width": all_plans[run][1]["width"],
                "feature_h": all_plans[run][1]["feature_h"],
                "feature_w": all_plans[run][1]["feature_w"],
            } for run in runs},
        },
    }
    report = {
        "runs": runs,
        "models": {
            "13": {"path": warp_path,
                   "sha256": hashlib.sha256(open(warp_path, "rb").read()).hexdigest()},
            "38": {"path": comp_path,
                   "sha256": hashlib.sha256(open(comp_path, "rb").read()).hexdigest()},
        },
        "manifest": manifest,
        "results": {run: {str(number): value
                          for number, value in per_run.items()}
                    for run, per_run in results.items()},
        "passed": passed,
    }
    path = os.path.join(args.out, "dynamic_tail_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("2 models x %d runs; %s; report %s"
          % (len(runs), "PASSED" if passed else "FAILED", path))
    return 0 if passed else 1
