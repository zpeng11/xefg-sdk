"""Diagnostic ONNX copies: change d24 coordinates or d25 weights, never the package."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from xefg_sdk import Config, UiMode, Xefg2xRuntime


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_rounding(model, stage, operation):
    source, prefix = ("Round", "n0025_") if stage == "d25" else ("Ceil", "n0024_")
    changed = []
    for node in model.graph.node:
        if node.op_type == source and node.output[0].startswith(prefix):
            node.op_type = operation
            changed.append(node.output[0])
    if not changed:
        raise ValueError("No selected rounding nodes found")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=4)
    parser.add_argument("--stage", choices=("d25", "d24"), default="d25")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tools, replay, metrics = load_tool("validate"), load_tool("replay"), load_tool("metrics")
    pairs = tools.read_json(args.pairing)
    selected = [p for p in pairs["pairs"] if p["classification"] == "generated"
                and (args.all or p["current_frame"] == args.frame)]
    if not selected:
        raise ValueError("No selected generated pair")
    for item in pairs["artifacts"]:
        if tools.artifact(args.run, item["file"], item["bytes"])["sha256"] != item["sha256"]:
            raise ValueError("Capture changed since pairing")
    inputs = tools.read_json(args.run / "application_inputs/manifest.json")
    prepared = []
    with Xefg2xRuntime.load(args.package) as runtime:
        for pair in selected:
            previous = replay.load_frame(args.run, inputs, pair["previous_frame"])
            current = replay.load_frame(args.run, inputs, pair["current_frame"])
            _, plan, feeds = runtime.prepare(previous, current, Config(ui_mode=UiMode.HUDLESS_UITEXTURE))
            variant = runtime.package.variants[plan.variant_id]
            model_path = runtime.package.path(variant["model"])
            output_name = variant["output"]
            official = np.fromfile(args.run / pair["file"], np.uint8).reshape(360, 640, 4)
            prepared.append((pair["current_frame"], feeds, official))
    mask = load_tool("rendered_pairing").mask_rects((360, 640), pairs["excluded_rects"])
    reports = []
    source_operation = "Round" if args.stage == "d25" else "Ceil"
    for operation in (source_operation, *[op for op in ("Round", "Floor", "Ceil") if op != source_operation]):
        model = onnx.load(model_path)
        changed = replace_rounding(model, args.stage, operation)
        onnx.checker.check_model(model)
        target = args.out / (operation.lower() + ".onnx")
        onnx.save(model, target)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(target), sess_options=options, providers=["CPUExecutionProvider"])
        frame_results = []
        for frame, feeds, official in prepared:
            output = session.run([output_name], feeds)[0]
            np.save(args.out / f"{operation.lower()}_{frame:05d}.npy", output, allow_pickle=False)
            result = metrics.compare(official, output, mask)
            frame_results.append({"frame": frame, "metrics": result})
            print(operation, frame, result["rgb_mae"], result["rgb_max"], flush=True)
        reports.append({"operation": operation, "nodes": changed, "frames": frame_results,
                        "model": tools.artifact(args.out, target.name)})
        del session
    report = {"schema": "xefg.sampling_probe/1", "frame": args.frame, "stage": args.stage,
              "source_model": tools.artifact(model_path.parent, model_path.name),
              "pairing": tools.artifact(args.pairing.parent, args.pairing.name), "variants": reports,
              "note": "Sensitivity experiment, not validation of a replacement rounding implementation."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
