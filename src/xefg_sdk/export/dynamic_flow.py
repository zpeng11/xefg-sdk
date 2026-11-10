#!/usr/bin/env python
"""D1: dynamic-shape models for the 8 flow convs and 4 upsample dispatches.

The flow and upsample weights, scale/bias tables and clamps are identical at
every captured resolution. Their extents differ, and `flow_plan` also chooses
per-phase padding from the input/output sizes; those four-value pad tuples are
feeds. The upsample stage needs no extra feeds: its low-res shape comes from
`x0`, its high-res shape from `x1`, and the crop is the high-res extent.

Each model is compared with the captured `u0` over the whole allocation; the
run fails unless every case is byte-exact.
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
from .dynamic_onnx import Builder, shape_dim


FLOW_DISPATCHES = (26, 27, 28, 29, 30, 32, 34, 36)
UPSAMPLE_DISPATCHES = (31, 33, 35, 37)


def _flow_input(builder, plan):
    """[1, slices*16, IH, IW] int8 with the shader's packed-word semantics."""
    b = builder
    p = plan
    if p["add"] == 0:
        b.add("Cast", ["x0"], ["flow_u32"], to=TensorProto.INT32)
        b.const("flow_128", "i32", [128])
        b.add("GreaterOrEqual", ["flow_u32", "flow_128"], ["flow_ge"])
        b.const("flow_256", "i32", [256])
        b.add("Sub", ["flow_u32", "flow_256"], ["flow_minus"])
        b.add("Where", ["flow_ge", "flow_minus", "flow_u32"],
              ["flow_signed32"])
        b.add("Cast", ["flow_signed32"], ["flow_signed"],
              to=TensorProto.INT8)
    else:
        b.const("flow_words_pre", "i64", [p["slices"]])
        b.const("flow_words_post", "i64", [4, 4])
        b.add("Concat", ["flow_words_pre", "IH", "IW", "flow_words_post"],
              ["flow_words_shape"], axis=0)
        b.add("Reshape", ["x0", "flow_words_shape"], ["flow_words"])
        b.const("flow_split_words", "i64", [1, 1, 1, 1])
        b.add("Split", ["flow_words", "flow_split_words"],
              ["flow_word0", "flow_word1", "flow_word2", "flow_word3"],
              axis=3)
        b.const("flow_split_bytes", "i64", [1, 1, 1, 1])
        b.const("flow_add64", "i64", [p["add"] & 0xFFFFFFFF])
        b.const("flow_mod32", "i64", [1 << 32])
        b.const("flow_256_64", "i64", [256])
        b.const("flow_128_64", "i64", [128])
        words = []
        for word in range(4):
            b.add("Split", ["flow_word%d" % word, "flow_split_bytes"],
                  ["flow_w%d_b%d" % (word, byte) for byte in range(4)],
                  axis=-1)
            packed = []
            for byte in range(4):
                b.add("Cast", ["flow_w%d_b%d" % (word, byte)],
                      ["flow_w%d_b%d_64" % (word, byte)], to=TensorProto.INT64)
                if byte:
                    b.const("flow_w%d_shift_%d" % (word, byte), "i64",
                            [1 << (8 * byte)])
                    b.add("Mul", ["flow_w%d_b%d_64" % (word, byte),
                                  "flow_w%d_shift_%d" % (word, byte)],
                          ["flow_w%d_b%d_s" % (word, byte)])
                    packed.append("flow_w%d_b%d_s" % (word, byte))
                else:
                    packed.append("flow_w%d_b%d_64" % (word, byte))
            b.add("Add", [packed[0], packed[1]], ["flow_w%d_01" % word])
            b.add("Add", [packed[2], packed[3]], ["flow_w%d_23" % word])
            b.add("Add", ["flow_w%d_01" % word, "flow_w%d_23" % word],
                  ["flow_w%d_sum" % word])
            b.add("Add", ["flow_w%d_sum" % word, "flow_add64"],
                  ["flow_w%d_added" % word])
            b.add("Mod", ["flow_w%d_added" % word, "flow_mod32"],
                  ["flow_w%d_mod" % word])
            bytes_out = []
            for byte in range(4):
                source = "flow_w%d_mod" % word
                if byte:
                    b.const("flow_w%d_div_%d" % (word, byte), "i64",
                            [1 << (8 * byte)])
                    b.add("Div", [source, "flow_w%d_div_%d" % (word, byte)],
                          ["flow_w%d_d%d" % (word, byte)])
                    source = "flow_w%d_d%d" % (word, byte)
                b.add("Mod", [source, "flow_256_64"],
                      ["flow_w%d_byte%d" % (word, byte)])
                b.add("GreaterOrEqual", ["flow_w%d_byte%d" % (word, byte),
                                         "flow_128_64"],
                      ["flow_w%d_ge%d" % (word, byte)])
                b.add("Sub", ["flow_w%d_byte%d" % (word, byte), "flow_256_64"],
                      ["flow_w%d_minus%d" % (word, byte)])
                b.add("Where", ["flow_w%d_ge%d" % (word, byte),
                                "flow_w%d_minus%d" % (word, byte),
                                "flow_w%d_byte%d" % (word, byte)],
                      ["flow_w%d_signed%d" % (word, byte)])
                b.add("Cast", ["flow_w%d_signed%d" % (word, byte)],
                      ["flow_w%d_i8_%d" % (word, byte)], to=TensorProto.INT8)
                bytes_out.append("flow_w%d_i8_%d" % (word, byte))
            b.add("Concat", bytes_out, ["flow_word%d_i8" % word], axis=-1)
            words.append("flow_word%d_i8" % word)
        b.add("Concat", words, ["flow_bytes"], axis=3)
        b.const("flow_src_pre", "i64", [p["slices"]])
        b.const("flow_src_post", "i64", [16])
        b.add("Concat", ["flow_src_pre", "IH", "IW", "flow_src_post"],
              ["flow_src_shape"], axis=0)
        b.add("Reshape", ["flow_bytes", "flow_src_shape"], ["flow_signed"])

    b.add("Transpose", ["flow_signed"], ["flow_nhwc"], perm=[0, 3, 1, 2])
    b.const("flow_nchw_pre", "i64", [1, p["slices"] * 16])
    b.add("Concat", ["flow_nchw_pre", "IH", "IW"], ["flow_nchw_shape"],
          axis=0)
    b.add("Reshape", ["flow_nhwc", "flow_nchw_shape"], ["flow_x"])
    return "flow_x"


def build_dynamic_flow(plan):
    """One flow conv with symbolic IH/IW/OH/OW and one pads feed per phase."""
    p = plan
    b = Builder()
    shape_dim(b, "x0", 1, "IH")
    shape_dim(b, "x0", 2, "IW")
    source = _flow_input(b, p)
    b.tensor("flow_w", p["weights"])

    accumulators = []
    for index, (left, top, right, bottom) in enumerate(p["pads"]):
        b.const("flow_pad_value_%d" % index, "i8", [p["pad_value"]])
        b.add("Pad", [source, "pads_%d" % index,
                      "flow_pad_value_%d" % index],
              ["flow_padded_%d" % index])
        b.add("ConvInteger", ["flow_padded_%d" % index, "flow_w"],
              ["flow_acc_%d" % index], strides=[p["stride_y"], p["stride_x"]])
        accumulators.append("flow_acc_%d" % index)
    if len(accumulators) == 1:
        acc = accumulators[0]
    else:
        b.add("Max", accumulators, ["flow_acc"])
        acc = "flow_acc"

    # One float32 rounding, as the DXIL's contracted fmul/fadd: compute the
    # product in double, add the bias, then cast once to float32.
    b.add("Cast", [acc], ["flow_accd"], to=TensorProto.DOUBLE)
    b.tensor("flow_scale", p["scale"].reshape(1, -1, 1, 1))
    b.tensor("flow_bias", p["bias"].reshape(1, -1, 1, 1))
    b.add("Cast", ["flow_scale"], ["flow_scale_d"], to=TensorProto.DOUBLE)
    b.add("Cast", ["flow_bias"], ["flow_bias_d"], to=TensorProto.DOUBLE)
    b.add("Mul", ["flow_accd", "flow_scale_d"], ["flow_scaled_d"])
    b.add("Add", ["flow_scaled_d", "flow_bias_d"], ["flow_biased_d"])
    b.add("Cast", ["flow_biased_d"], ["flow_biased"], to=TensorProto.FLOAT)
    b.add("Cast", ["flow_biased"], ["flow_requant"], to=TensorProto.INT32)
    b.const("flow_lo", "i32", [p["lo"]])
    b.const("flow_hi", "i32", [p["hi"]])
    b.add("Clip", ["flow_requant", "flow_lo", "flow_hi"], ["flow_clipped"])
    b.add("Cast", ["flow_clipped"], ["flow_q"], to=TensorProto.INT8)
    b.const("flow_out_pre", "i64", [p["outs"], 16])
    b.add("Concat", ["flow_out_pre", "OH", "OW"], ["flow_out_shape"], axis=0)
    b.add("Reshape", ["flow_q", "flow_out_shape"], ["flow_q5"])
    b.add("Transpose", ["flow_q5"], ["u0"], perm=[0, 2, 3, 1])

    inputs = [
        helper.make_tensor_value_info(
            "x0", TensorProto.UINT8,
            [int(p["slices"]), "IH", "IW", 16]),
        helper.make_tensor_value_info("OH", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("OW", TensorProto.INT64, [1]),
    ]
    for index in range(len(p["pads"])):
        inputs.append(helper.make_tensor_value_info(
            "pads_%d" % index, TensorProto.INT64, [8]))
    outputs = [
        helper.make_tensor_value_info(
            "u0", TensorProto.INT8,
            [int(p["outs"]), "OH", "OW", 16]),
    ]
    graph = helper.make_graph(
        b.nodes, "xefg_flow_dynamic_d%04d" % p["dispatch"], inputs, outputs,
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def build_dynamic_upsample(plan):
    """2x fixed-point staging + residual + int8 conv, fully shape-derived."""
    p = plan
    b = Builder()
    shape_dim(b, "x0", 1, "IH")
    shape_dim(b, "x0", 2, "IW")
    shape_dim(b, "x1", 1, "OH")
    shape_dim(b, "x1", 2, "OW")
    channels = p["slices"] * 16

    b.const("up_src_pre", "i64", [p["slices"]])
    b.const("up_src_post", "i64", [16])
    b.add("Concat", ["up_src_pre", "IH", "IW", "up_src_post"],
          ["up_src_shape"], axis=0)
    b.add("Reshape", ["x0", "up_src_shape"], ["up_src"])
    b.add("Transpose", ["up_src"], ["up_src_hwc"], perm=[0, 3, 1, 2])
    b.const("up_nchw_pre", "i64", [1, channels])
    b.add("Concat", ["up_nchw_pre", "IH", "IW"], ["up_nchw_shape"], axis=0)
    b.add("Reshape", ["up_src_hwc", "up_nchw_shape"], ["up_src_nchw"])
    b.add("Cast", ["up_src_nchw"], ["up_src_f"], to=TensorProto.FLOAT)

    kernel = np.asarray([1, 3, 3, 1], dtype=np.float32)
    weights = np.zeros((channels, 1, 4, 4), dtype=np.float32)
    weights[:, 0] = np.outer(kernel, kernel)
    b.tensor("up_kernel", weights)
    b.add("ConvTranspose", ["up_src_f", "up_kernel"], ["up_raw"],
          strides=[2, 2], pads=[1, 1, 1, 1], group=channels)
    b.const("up_16", "fp32", 16.0)
    b.add("Div", ["up_raw", "up_16"], ["up_scaled"])
    b.add("Cast", ["up_scaled"], ["up_int"], to=TensorProto.INT32)
    b.const("up_crop_starts", "i64", [0, 0, 0, 0])
    b.const("up_crop_one", "i64", [1, channels])
    b.add("Concat", ["up_crop_one", "OH", "OW"], ["up_crop_ends"], axis=0)
    b.const("up_crop_axes", "i64", [0, 1, 2, 3])
    b.const("up_crop_steps", "i64", [1, 1, 1, 1])
    b.add("Slice", ["up_int", "up_crop_starts", "up_crop_ends",
                    "up_crop_axes", "up_crop_steps"], ["up_cropped"])

    b.const("up_high_pre", "i64", [p["slices"]])
    b.add("Concat", ["up_high_pre", "OH", "OW", "up_src_post"],
          ["up_high_shape"], axis=0)
    b.add("Reshape", ["x1", "up_high_shape"], ["up_high"])
    b.add("Transpose", ["up_high"], ["up_high_hwc"], perm=[0, 3, 1, 2])
    b.add("Concat", ["up_nchw_pre", "OH", "OW"], ["up_high_nchw"], axis=0)
    b.add("Reshape", ["up_high_hwc", "up_high_nchw"], ["up_high_nchw_t"])
    b.add("Cast", ["up_high_nchw_t"], ["up_high_i"], to=TensorProto.INT32)

    b.add("Add", ["up_cropped", "up_high_i"], ["up_sum"])
    b.const("up_lo", "i32", [0])
    b.const("up_hi", "i32", [127])
    b.add("Clip", ["up_sum", "up_lo", "up_hi"], ["up_clamped"])
    b.add("Cast", ["up_clamped"], ["up_staged"], to=TensorProto.INT8)

    b.tensor("up_conv_w", p["weights"])
    b.add("ConvInteger", ["up_staged", "up_conv_w"], ["up_acc"],
          pads=[1, 1, 1, 1])
    b.add("Cast", ["up_acc"], ["up_accf"], to=TensorProto.FLOAT)
    b.tensor("up_scale", p["scale"].reshape(1, -1, 1, 1))
    b.tensor("up_bias", p["bias"].reshape(1, -1, 1, 1))
    b.add("Mul", ["up_accf", "up_scale"], ["up_scaled_acc"])
    b.add("Add", ["up_scaled_acc", "up_bias"], ["up_biased"])
    b.add("Cast", ["up_biased"], ["up_requant"], to=TensorProto.INT32)
    b.const("up_qlo", "i32", [0])
    b.const("up_qhi", "i32", [127])
    b.add("Clip", ["up_requant", "up_qlo", "up_qhi"], ["up_qclamped"])
    b.add("Cast", ["up_qclamped"], ["up_q"], to=TensorProto.INT8)
    b.const("up_out_pre", "i64", [p["outs"], 16])
    b.add("Concat", ["up_out_pre", "OH", "OW"], ["up_out_shape"], axis=0)
    b.add("Reshape", ["up_q", "up_out_shape"], ["up_q5"])
    b.add("Transpose", ["up_q5"], ["u0"], perm=[0, 2, 3, 1])

    graph = helper.make_graph(
        b.nodes, "xefg_upsample_dynamic_d%04d" % p["dispatch"],
        [helper.make_tensor_value_info(
            "x0", TensorProto.INT8, [int(p["slices"]), "IH", "IW", 16]),
         helper.make_tensor_value_info(
            "x1", TensorProto.INT8, [int(p["slices"]), "OH", "OW", 16])],
        [helper.make_tensor_value_info(
            "u0", TensorProto.INT8, [int(p["outs"]), "OH", "OW", 16])],
        b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _plans(run_dir, frame=1):
    rows = recon.read_events(run_dir)
    flow, upsample = {}, {}
    for number, dispatch in dynamic_index.numbered_dispatches(
            run_dir, frame, rows):
        if number in FLOW_DISPATCHES:
            plan, error = recon.flow_plan(run_dir, dispatch)
            if error:
                raise SystemExit("d%04d flow: %s" % (number, error))
            plan["capture_dispatch"] = dispatch["dispatch"]
            plan["dispatch"] = number
            flow[number] = plan
        elif number in UPSAMPLE_DISPATCHES:
            plan, error = recon.upsample_plan(run_dir, dispatch)
            if error:
                raise SystemExit("d%04d upsample: %s" % (number, error))
            plan["capture_dispatch"] = dispatch["dispatch"]
            plan["dispatch"] = number
            upsample[number] = plan
    if sorted(flow) != list(FLOW_DISPATCHES):
        raise SystemExit("found flow dispatches %s" % sorted(flow))
    if sorted(upsample) != list(UPSAMPLE_DISPATCHES):
        raise SystemExit("found upsample dispatches %s" % sorted(upsample))
    return flow, upsample


def _flow_feeds(plan):
    feeds = {"x0": plan["src"]}
    feeds["OH"] = np.asarray([plan["height"]], dtype=np.int64)
    feeds["OW"] = np.asarray([plan["width"]], dtype=np.int64)
    for index, (left, top, right, bottom) in enumerate(plan["pads"]):
        feeds["pads_%d" % index] = np.asarray(
            [0, 0, top, left, 0, 0, bottom, right], dtype=np.int64)
    return feeds


def _upsample_feeds(plan):
    return {"x0": plan["low"], "x1": plan["high"]}


def _flow_fingerprint(plan):
    digest = hashlib.sha256()
    for array in (plan["weights"], plan["scale"], plan["bias"]):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps([
        plan["template"], plan["slices"], plan["outs"], plan["stride_x"],
        plan["stride_y"], plan["add"], plan["pad_value"], plan["lo"],
        plan["hi"],
    ], sort_keys=True).encode())
    return digest.hexdigest()


def _upsample_fingerprint(plan):
    digest = hashlib.sha256()
    for array in (plan["weights"], plan["scale"], plan["bias"]):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps([
        plan["slices"], plan["outs"], plan["groups"],
    ], sort_keys=True).encode())
    return digest.hexdigest()


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
    flow0, up0 = all_plans[runs[0]]
    paths, fingerprints = {}, {}
    for number, plan in sorted(flow0.items()):
        model = build_dynamic_flow(plan)
        onnx.checker.check_model(model)
        name = "d%04d_flow_dynamic.onnx" % number
        path = os.path.join(args.out, name)
        onnx.save(model, path)
        paths[number] = ("flow", path)
        fingerprints[number] = _flow_fingerprint(plan)
        print("built %s  inputs=%s" % (name, [i.name for i in model.graph.input]))
    for number, plan in sorted(up0.items()):
        model = build_dynamic_upsample(plan)
        onnx.checker.check_model(model)
        name = "d%04d_upsample_dynamic.onnx" % number
        path = os.path.join(args.out, name)
        onnx.save(model, path)
        paths[number] = ("upsample", path)
        fingerprints[number] = _upsample_fingerprint(plan)
        print("built %s  inputs=%s" % (name, [i.name for i in model.graph.input]))

    results = {}
    passed = True
    for run in runs:
        flow, upsample = all_plans[run]
        per_run = {}
        for number, plan in sorted(flow.items()):
            if _flow_fingerprint(plan) != fingerprints[number]:
                raise SystemExit("d%04d flow parameters differ in %s"
                                 % (number, run))
            session = onnxruntime.InferenceSession(
                paths[number][1], providers=["CPUExecutionProvider"])
            got = session.run(["u0"], _flow_feeds(plan))[0]
            expected = plan["expected"]
            per_run[number] = _compare(got, expected)
        for number, plan in sorted(upsample.items()):
            if _upsample_fingerprint(plan) != fingerprints[number]:
                raise SystemExit("d%04d upsample parameters differ in %s"
                                 % (number, run))
            session = onnxruntime.InferenceSession(
                paths[number][1], providers=["CPUExecutionProvider"])
            got = session.run(["u0"], _upsample_feeds(plan))[0]
            per_run[number] = _compare(got, plan["expected"])
        results[run] = per_run
        passed = passed and all(value.get("mismatch") == 0
                                for value in per_run.values())
        print("%-60s %s" % (os.path.basename(run),
                            "EXACT" if all(value.get("mismatch") == 0
                                           for value in per_run.values())
                            else "DIFF"))

    manifest = {}
    for number in sorted(paths):
        kind = paths[number][0]
        if kind == "flow":
            feeds = {os.path.basename(run): {
                "OH": all_plans[run][0][number]["height"],
                "OW": all_plans[run][0][number]["width"],
                "pads": [list(pad) for pad in all_plans[run][0][number]["pads"]],
            } for run in runs}
        else:
            feeds = {os.path.basename(run): {
                "in_h": all_plans[run][1][number]["in_h"],
                "in_w": all_plans[run][1][number]["in_w"],
                "height": all_plans[run][1][number]["height"],
                "width": all_plans[run][1][number]["width"],
            } for run in runs}
        manifest[str(number)] = {
            "family": kind,
            "model": os.path.basename(paths[number][1]),
            "feeds": feeds,
        }

    report = {
        "runs": runs,
        "models": {str(number): {
            "family": kind,
            "path": path,
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "fingerprint": fingerprints[number],
        } for number, (kind, path) in sorted(paths.items())},
        "manifest": manifest,
        "results": {run: {str(number): value
                          for number, value in per_run.items()}
                    for run, per_run in results.items()},
        "passed": passed,
    }
    path = os.path.join(args.out, "dynamic_flow_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("%d models x %d runs; %s; report %s"
          % (len(paths), len(runs), "PASSED" if passed else "FAILED", path))
    return 0 if passed else 1


def _compare(got, expected):
    expected = expected.astype(np.int8)
    if got.shape != expected.shape:
        return {"error": "shape %s != %s" % (got.shape, expected.shape)}
    diff = got.astype(np.int16) - expected.astype(np.int16)
    mismatch = int((diff != 0).sum())
    return {
        "mismatch": mismatch,
        "max_diff": int(np.abs(diff).max()) if mismatch else 0,
        "shape": list(got.shape),
    }
