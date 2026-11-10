#!/usr/bin/env python
"""D1 pilot: one dynamic-shape ONNX model for each of the 15 int8 convs.

The weights, bias, activation and requant constants are identical at every
captured resolution; only the spatial extents and `d02`'s phase change.  This
tool therefore builds one model per dispatch from the first run and executes
that same file against every supplied run.

The graph derives H and W from `x0`'s shape and takes the remaining
resolution-dependent scalars as feeds:

    valid_h, in_w, out_h, out_w, phase_x, phase_y

Every model is compared with the captured `u0` over the whole allocation; the
run fails unless all 15 x N cases are byte-exact.
"""
# Internal legacy research module; not a supported CLI entry point.
import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper

from . import dynamic_index, recon
from .dynamic_onnx import Builder, arithmetic_shift


CONV_DISPATCHES = (2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21)
DIM_FEEDS = ("valid_h", "in_w", "out_h", "out_w", "phase_x", "phase_y",
             "alloc_h", "alloc_w")


def _dynamic_extents(builder):
    """H and W of the bound input allocation, as int64 [1] tensors."""
    b = builder
    b.add("Shape", ["x0"], ["x_shape"])
    b.const("axis0", "i64", [0])
    b.const("step1", "i64", [1])
    for tag, index in (("H", 1), ("W", 2)):
        b.const(tag + "_start", "i64", [index])
        b.const(tag + "_end", "i64", [index + 1])
        b.add("Slice", ["x_shape", tag + "_start", tag + "_end",
                        "axis0", "step1"], [tag])
    return "H", "W"


def _valid_mask(builder):
    """[1, H, W, 1] int32 with 1 inside the logical in_w x valid_h band."""
    b = builder
    b.add("Min", ["valid_h", "H"], ["vh"])
    b.add("Min", ["in_w", "W"], ["vw"])
    b.const("one", "i64", [1])
    b.add("Concat", ["one", "vh", "vw", "one"], ["ones_shape"], axis=0)
    b.add("ConstantOfShape", ["ones_shape"], ["ones"],
          value=helper.make_tensor("ones_value", TensorProto.INT32, [1], [1]))
    b.add("Sub", ["H", "vh"], ["pad_h"])
    b.add("Sub", ["W", "vw"], ["pad_w"])
    b.const("pad_zeros4", "i64", [0, 0, 0, 0])
    b.const("zero1", "i64", [0])
    b.add("Concat", ["pad_zeros4", "zero1", "pad_h", "pad_w", "zero1"],
          ["valid_pads"], axis=0)
    b.const("zero32", "i32", [0])
    b.add("Pad", ["ones", "valid_pads", "zero32"], ["valid"])
    return "valid"


def _expand_input(builder, plan):
    """Unpack x0 into [1, blocks*lanes, H, W] int32 channels, dynamically."""
    b = builder
    template, blocks = plan["template"], plan["blocks"]
    src_bytes = plan["src_bytes"]
    H, W = "H", "W"

    b.add("Cast", ["x0"], ["x32"], to=TensorProto.INT32)
    b.add("Mul", ["x32", "valid"], ["masked"])
    need = 1 if template == "A" else (
        2 * blocks if src_bytes == 8 else blocks)
    b.const("arr_start", "i64", [0])
    b.const("arr_end", "i64", [need])
    b.const("arr_step", "i64", [1])
    b.add("Slice", ["masked", "arr_start", "arr_end", "axis0", "arr_step"],
          ["used"])

    if template == "A":
        # One 2-byte texel per block: [c0, c1, folded zero point, 0].
        b.const("ch0", "i64", [0])
        b.const("ch1", "i64", [1])
        b.const("ch2", "i64", [2])
        b.const("ch_axis", "i64", [3])
        b.add("Slice", ["used", "ch0", "ch1", "ch_axis", "arr_step"], ["c0"])
        b.add("Slice", ["used", "ch1", "ch2", "ch_axis", "arr_step"], ["c1"])
        b.const("zero", "i32", [0])
        b.const("zp", "i32", [plan["zero_point"]])
        b.add("Mul", ["valid", "zp"], ["zp_plane"])
        b.add("Mul", ["valid", "zero"], ["zero_plane"])
        b.add("Concat", ["c0", "c1", "zp_plane", "zero_plane"], ["lanes"], axis=3)
        b.const("tile", "i64", [1, 1, 1, blocks])
        b.add("Tile", ["lanes", "tile"], ["tiled"])
        b.add("Transpose", ["tiled"], ["xch"], perm=[0, 3, 1, 2])
        return "xch"

    if src_bytes == 8:
        b.const("sb_pre", "i64", [blocks, 2])
        b.const("sb_post", "i64", [2, 4])
        b.add("Concat", ["sb_pre", H, W, "sb_post"], ["shape_b"], axis=0)
        b.add("Reshape", ["used", "shape_b"], ["r"])
        b.add("Transpose", ["r"], ["t"], perm=[0, 1, 4, 5, 2, 3])
    else:
        b.const("sb_pre", "i64", [blocks])
        b.const("sb_post", "i64", [4, 4])
        b.add("Concat", ["sb_pre", H, W, "sb_post"], ["shape_b"], axis=0)
        b.add("Reshape", ["used", "shape_b"], ["r"])
        b.add("Transpose", ["r"], ["t"], perm=[0, 3, 4, 1, 2])
    b.const("sc_pre", "i64", [1, blocks * 16])
    b.add("Concat", ["sc_pre", H, W], ["shape_c"], axis=0)
    b.add("Reshape", ["t", "shape_c"], ["xch"])
    return "xch"


def _crop(builder, plan):
    """Crop the 3-padded tensor to the stride window; start includes phase."""
    b = builder
    stride_x, stride_y = plan["stride_x"], plan["stride_y"]
    channels = (plan["blocks"] * 16 if plan["template"] == "BC"
                else plan["blocks"] * 4)
    b.const("three", "i64", [3])
    b.add("Add", ["phase_y", "three"], ["start_y"])
    b.add("Add", ["phase_x", "three"], ["start_x"])
    for tag, out, stride in (("h", "out_h", stride_y), ("w", "out_w", stride_x)):
        b.const("stride_" + tag, "i64", [stride])
        b.add("Sub", [out, "one"], [tag + "_m1"])
        b.add("Mul", [tag + "_m1", "stride_" + tag], [tag + "_span"])
        b.add("Add", [tag + "_span", "three"], ["crop_" + tag])
    b.const("crop_pre", "i64", [0, 0])
    b.add("Concat", ["crop_pre", "start_y", "start_x"], ["crop_starts"], axis=0)
    b.const("crop_one", "i64", [1, channels])
    b.add("Add", ["start_y", "crop_h"], ["end_y"])
    b.add("Add", ["start_x", "crop_w"], ["end_x"])
    b.add("Concat", ["crop_one", "end_y", "end_x"], ["crop_ends"], axis=0)
    b.const("crop_axes", "i64", [0, 1, 2, 3])
    b.const("crop_steps", "i64", [1, 1, 1, 1])
    b.add("Slice", ["padded", "crop_starts", "crop_ends", "crop_axes",
                    "crop_steps"], ["crop"])


def build_dynamic_conv(plan):
    """One int8 conv with symbolic H/W and six runtime scalar feeds."""
    p = plan
    slices, groups = p["slices"], p["groups"]
    out_bytes = p["out_bytes"]

    b = Builder()
    _dynamic_extents(b)
    _valid_mask(b)
    xch = _expand_input(b, p)

    # The shader's window starts at 3 + phase after the 3-texel pad.
    b.const("pad", "i64", [0, 0, 3, 3, 0, 0, 3, 3])
    b.const("pad_value", "i32", [0])
    b.add("Pad", [xch, "pad", "pad_value"], ["padded"])
    _crop(b, p)
    b.add("Cast", ["crop"], ["crop8"], to=TensorProto.INT8)

    weights = p["weights"].transpose(0, 2, 5, 1, 4, 6, 3).reshape(
        slices * groups * 4, p["blocks"] * p["words_per_tap"] * 4, 3, 3)
    b.tensor("w", weights.astype(np.int8))
    b.add("ConvInteger", ["crop8", "w"], ["acc"],
          strides=[p["stride_y"], p["stride_x"]])

    bias = p["bias"].reshape(slices * groups * 4, 1, 1).astype(np.int32)
    b.tensor("bias", bias)
    b.add("Add", ["acc", "bias"], ["biased"])
    b.const("so_pre", "i64", [slices, groups, 4])
    b.add("Concat", ["so_pre", "out_h", "out_w"], ["shape_out"], axis=0)
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

    b.const("scale_c", "i32", [p["scale"]])
    b.add("Mul", ["pos", "scale_c"], ["pos_scaled"])
    arithmetic_shift(b, "pos_scaled", p["s1"], "pos")
    b.add("Mul", ["neg", "activation"], ["neg_scaled"])
    arithmetic_shift(b, "neg_scaled", p["s2"], "neg")
    b.add("Add", ["pos_shifted", "neg_shifted"], ["requant"])
    b.const("qlo", "i32", [-128])
    b.const("qhi", "i32", [127])
    b.add("Clip", ["requant", "qlo", "qhi"], ["qclamped"])
    b.add("Transpose", ["qclamped"], ["qzyx"], perm=[0, 2, 3, 1, 4])
    b.const("bytes_pre", "i64", [slices])
    b.const("bytes_post", "i64", [out_bytes])
    b.add("Concat", ["bytes_pre", "out_h", "out_w", "bytes_post"],
          ["shape_bytes"], axis=0)
    b.add("Reshape", ["qzyx", "shape_bytes"], ["bytes"])

    # Place the logical rect at row 1 of the padded allocation, exactly as the
    # static lowering does: [0,1,0,0,0, alloc_h-1-out_h, alloc_w-out_w, 0].
    # The output allocation is not the input allocation: four stages bind a
    # larger input texture than the texture they write (d2/d3/d4/d15), so its
    # extent is a feed rather than `Shape(x0)`.
    b.add("Sub", ["alloc_h", "one"], ["op_h_m1"])
    b.add("Sub", ["op_h_m1", "out_h"], ["h_tail"])
    b.add("Sub", ["alloc_w", "out_w"], ["w_tail"])
    b.const("op_pre", "i64", [0, 1, 0, 0, 0])
    b.const("op_post", "i64", [0])
    b.add("Concat", ["op_pre", "h_tail", "w_tail", "op_post"], ["out_pad"],
          axis=0)
    b.const("out_zero", "i32", [0])
    b.add("Pad", ["bytes", "out_pad", "out_zero"], ["u0_i32"])
    b.add("Cast", ["u0_i32"], ["u0"], to=TensorProto.INT8)

    inputs = [
        helper.make_tensor_value_info(
            "x0", TensorProto.INT8,
            [int(p["src"].shape[0]), "H", "W", int(p["src_bytes"])]),
    ]
    for name in DIM_FEEDS:
        inputs.append(helper.make_tensor_value_info(
            name, TensorProto.INT64, [1]))
    outputs = [
        helper.make_tensor_value_info(
            "u0", TensorProto.INT8,
            [int(slices), "AH", "AW", int(out_bytes)]),
    ]
    graph = helper.make_graph(
        b.nodes, "xefg_conv_dynamic_d%04d" % p["dispatch"], inputs, outputs,
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _conv_plans(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    params = recon.layer_parameters(rows)
    plans = {}
    for number, dispatch in dynamic_index.numbered_dispatches(
            run_dir, frame, rows):
        if number not in CONV_DISPATCHES:
            continue
        if not recon.conv_shaped(dispatch):
            raise SystemExit("d%04d is no longer conv-shaped" % number)
        plan, error = recon.conv_plan(run_dir, dispatch,
                                      params.get(dispatch["key"], {}))
        if error:
            raise SystemExit("d%04d: %s" % (number, error))
        plan["capture_dispatch"] = dispatch["dispatch"]
        plan["dispatch"] = number
        plans[number] = plan
    if sorted(plans) != list(CONV_DISPATCHES):
        raise SystemExit("found conv dispatches %s, expected %s"
                         % (sorted(plans), list(CONV_DISPATCHES)))
    return plans


def _dim_values(plan):
    """The resolution-dependent scalars the model takes as feeds."""
    return {
        "valid_h": plan["valid_h"],
        "in_w": plan["in_w"],
        "out_h": plan["out_h"],
        "out_w": plan["out_w"],
        "phase_x": plan["phase"][0],
        "phase_y": plan["phase"][1],
        "alloc_h": plan["H"],
        "alloc_w": plan["W"],
    }


def _feeds(plan):
    feeds = {"x0": plan["src"].astype(np.int8)}
    for name, value in _dim_values(plan).items():
        feeds[name] = np.asarray([value], dtype=np.int64)
    return feeds


def _model_fingerprint(plan):
    """Everything a shared model bakes; the dimensions and phase are feeds."""
    digest = hashlib.sha256()
    for array in (plan["weights"], plan["bias"], plan["activation"]):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps([
        plan["template"], plan["blocks"], plan["groups"], plan["slices"],
        plan["stride_x"], plan["stride_y"], plan["src_bytes"],
        plan["out_bytes"], plan["scale"], plan["s1"], plan["s2"],
        plan["lo"], plan["hi"], plan["zero_point"],
    ], sort_keys=True).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True,
                        help="comma-separated run directories; the first "
                             "builds the models")
    parser.add_argument("--out", required=True, help="model/report directory")
    parser.add_argument("--frame", type=int, default=1)
    args = parser.parse_args()

    import onnxruntime

    runs = [os.path.abspath(path) for path in args.runs.split(",") if path]
    if len(runs) < 2:
        raise SystemExit("pass at least two runs")
    os.makedirs(args.out, exist_ok=True)

    all_plans = {run: _conv_plans(run, args.frame) for run in runs}
    built = all_plans[runs[0]]
    paths = {}
    fingerprints = {}
    for dispatch, plan in sorted(built.items()):
        model = build_dynamic_conv(plan)
        onnx.checker.check_model(model)
        name = "d%04d_conv_dynamic.onnx" % dispatch
        path = os.path.join(args.out, name)
        onnx.save(model, path)
        paths[dispatch] = path
        fingerprints[dispatch] = _model_fingerprint(plan)
        print("built %s  inputs=%s" % (name, [i.name for i in model.graph.input]))

    results = {}
    passed = True
    for run in runs:
        plans = all_plans[run]
        per_run = {}
        for dispatch, plan in sorted(plans.items()):
            if _model_fingerprint(plan) != fingerprints[dispatch]:
                raise SystemExit("d%04d model parameters differ in %s"
                                 % (dispatch, run))
            session = onnxruntime.InferenceSession(
                paths[dispatch], providers=["CPUExecutionProvider"])
            got = session.run(["u0"], _feeds(plan))[0]
            expected = plan["expected"].astype(np.int8)
            if got.shape != expected.shape:
                per_run[dispatch] = {"error": "shape %s != %s"
                                     % (got.shape, expected.shape)}
                passed = False
                continue
            diff = got.astype(np.int16) - expected.astype(np.int16)
            mismatch = int((diff != 0).sum())
            per_run[dispatch] = {
                "mismatch": mismatch,
                "max_diff": int(np.abs(diff).max()) if mismatch else 0,
                "shape": list(got.shape),
            }
            passed = passed and mismatch == 0
        results[run] = per_run
        print("%-60s %s" % (os.path.basename(run),
                            "EXACT" if all(v.get("mismatch") == 0
                                           for v in per_run.values())
                            else "DIFF"))

    manifest = {}
    for dispatch in sorted(built):
        manifest[str(dispatch)] = {
            "model": os.path.basename(paths[dispatch]),
            "input": {"array": int(built[dispatch]["src"].shape[0]),
                      "src_bytes": int(built[dispatch]["src_bytes"])},
            "output": {"slices": int(built[dispatch]["slices"]),
                       "out_bytes": int(built[dispatch]["out_bytes"])},
            "feeds": {
                os.path.basename(run): _dim_values(all_plans[run][dispatch])
                for run in runs
            },
        }

    report = {
        "runs": runs,
        "models": {str(d): {"path": paths[d],
                            "sha256": hashlib.sha256(
                                open(paths[d], "rb").read()).hexdigest(),
                            "fingerprint": fingerprints[d]}
                   for d in sorted(paths)},
        "manifest": manifest,
        "results": {run: {str(d): value for d, value in per_run.items()}
                    for run, per_run in results.items()},
        "passed": passed,
    }
    path = os.path.join(args.out, "dynamic_conv_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("%d models x %d runs; %s; report %s"
          % (len(paths), len(runs), "PASSED" if passed else "FAILED", path))
    return 0 if passed else 1
