"""Replay verified calibration pairs through the public SDK API."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from xefg_sdk import Config, FrameInputs, ImageView, UiMode, Xefg2xRuntime


def load_frame(run, manifest, index, reset=False):
    row = manifest["frames"][index]
    if row["index"] != index:
        raise ValueError("Input manifest frame order differs from frame IDs")
    height, width = manifest["height"], manifest["width"]
    mh, mw = manifest["motion_height"], manifest["motion_width"]
    def view(role, dtype, shape, format):
        return ImageView(np.fromfile(run / "application_inputs" / row["inputs"][role],
                                    dtype=dtype).reshape(shape), format)
    return FrameInputs(index,
        hudless_color=view("color", np.uint8, (height, width, 4), "BGRA8_UNORM"),
        ui=view("ui", np.uint8, (height, width, 4), "BGRA8_UNORM"),
        motion_vectors=view("motion", "<f2", (mh, mw, 2), "RG16_FLOAT"),
        depth=view("depth", "<f4", (mh, mw), "R32_FLOAT"), reset_history=reset)


def run_replay(args, tools):
    run = args.run.resolve()
    pairing_path = args.pairing.resolve()
    pairing = tools.read_json(pairing_path)
    if pairing["schema"] not in ("xefg.pairing/1", "xefg.pairing/2") or pairing["status"] != "paired":
        raise ValueError("Replay requires a complete pairing report")
    rendered = pairing.get("method") == "rendered_scene"
    if pairing["official_tags_requested"] and not rendered:
        raise ValueError("Use an untagged run for SDK quality comparison")
    if rendered and (pairing["schema"] != "xefg.pairing/2" or not pairing["official_tags_requested"]):
        raise ValueError("Rendered comparison requires tagged pairing v2")
    if Path(pairing["capture_directory"]).resolve() != run:
        raise ValueError("Pairing report belongs to another capture")
    tools.inspect_run(run)
    for item in pairing["artifacts"]:
        current = tools.artifact(run, item["file"], item["bytes"])
        if current["sha256"] != item["sha256"]:
            raise ValueError("Capture changed since pairing: " + item["file"])
    manifest = tools.read_json(run / "application_inputs/manifest.json")
    calibration = tools.read_json(run / ("scene.json" if rendered else "calibration/manifest.json"))
    meta = tools.read_json(run / tools.metadata_name(run))
    references = {f["index"]: f for f in calibration["frames"]}
    resets = {f["index"] for f in calibration["frames"] if f["reset_history"]}
    pairs = [p for p in pairing["pairs"] if p["classification"] == "generated"]
    if not pairs:
        raise ValueError("Pairing report contains no generated frames")
    if not args.all:
        pairs = pairs[:1]
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metrics = tools.load_tool("metrics")
    results = []
    package_files = []
    with Xefg2xRuntime.load(args.package) as runtime:
        package_files.append(tools.artifact(runtime.package.directory, "manifest.json"))
        for variant in runtime.package.variants.values():
            for name in [variant["model"], *variant["profiles"]]:
                package_files.append(tools.artifact(runtime.package.directory, name))
        for pair in pairs:
            previous_id, current_id = pair["previous_frame"], pair["current_frame"]
            if current_id in resets or previous_id != current_id - 1:
                raise ValueError("Pair cannot be replayed across reset/missing history")
            previous = load_frame(run, manifest, previous_id)
            current = load_frame(run, manifest, current_id)
            config = Config(ui_mode=UiMode.HUDLESS_UITEXTURE)
            runtime.prepare(previous, current, config)
            start = time.perf_counter()
            result = runtime.interpolate_pair(previous, current, config)
            elapsed = time.perf_counter() - start
            generated = result.generated_frame.data
            official = np.fromfile(run / pair["file"], np.uint8).reshape(generated.shape)
            mask = np.ones(generated.shape[:2], dtype=bool)
            marker = pairing["marker"]
            x, y = marker["x"], marker["y"]
            mask[y:y + marker["cell_height"], x:x + marker["cell_width"] * marker["bits"]] = False
            if rendered:
                mask = tools.load_tool("rendered_pairing").mask_rects(generated.shape, pairing["excluded_rects"])
            stem = f"frame_{current_id:05d}"
            np.save(output / (stem + ".npy"), generated, allow_pickle=False)
            Image.fromarray(metrics.rgb_image(official)).save(output / (stem + "_official.png"))
            Image.fromarray(metrics.rgb_image(generated)).save(output / (stem + "_sdk.png"))
            difference = metrics.difference_image(official, generated)
            Image.fromarray(difference).save(output / (stem + "_diff_x8.png"))
            if rendered:
                difference[~mask] = 0
                Image.fromarray(difference).save(output / (stem + "_valid_diff_x8.png"))
            results.append({"previous_frame": previous_id, "current_frame": current_id,
                            "present_sequence": pair["sequence"], "profile": result.profile_id,
                            "wall_seconds_including_first_load": elapsed,
                            "full": metrics.compare(official, generated),
                            "valid_region": metrics.compare(official, generated, mask),
                            "output": stem + ".npy"})
            if rendered:
                reference = np.fromfile(run / references[current_id]["midpoint"], np.uint8).reshape(generated.shape)
                Image.fromarray(metrics.rgb_image(reference)).save(output / (stem + "_native_mid.png"))
                results[-1]["sdk_vs_native_mid"] = metrics.compare(reference, generated, mask)
                results[-1]["official_vs_native_mid"] = metrics.compare(reference, official, mask)
            else:
                results[-1]["without_marker"] = results[-1]["valid_region"]
            print(f"frame {current_id}: valid RGB MAE {results[-1]['valid_region']['rgb_mae']:.6f}, {elapsed:.2f}s", flush=True)
    repo = Path(__file__).resolve().parents[2]
    def git(*command):
        return subprocess.check_output(["git", "-C", str(repo), *command], text=True).strip()
    report = {"schema": "xefg.replay_comparison/1", "status": "measured", "quality_gate": "not_configured",
              "capture_directory": str(run), "package_directory": str(args.package.resolve()),
              "command": sys.argv, "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
              "python": platform.python_version(),
              "versions": {name: importlib.metadata.version(name) for name in ("xefg-sdk", "numpy", "onnxruntime", "pillow")},
              "provider": {key: meta[key] for key in ("adapter", "xefg_family", "xefg_dll_sha256")},
              "pairing": tools.artifact(pairing_path.parent, pairing_path.name),
              "package_artifacts": package_files, "pairs": results,
              "comparison_region": "exclude official tags and frame marker" if rendered else "exclude frame marker",
              "excluded_rects": pairing.get("excluded_rects", []),
              "full_frame_note": "Full-frame metrics include debug overlays; use valid_region for comparison." if rendered else "No official debug overlay.",
              "output_artifacts": [tools.artifact(output, p.name) for p in sorted(output.iterdir())],
              "timing_note": "CPU replay wall time includes first model load; not a real-time benchmark."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# SDK vs official output", "", "Measured only; no quality pass/fail threshold.",
             "Metrics below use the valid region, excluding the recorded overlays.", "",
             "| Current frame | RGB MAE | Max | Changed pixels |", "|---|---|---|---|"]
    lines += [f"| {r['current_frame']} | {r['valid_region']['rgb_mae']:.6f} | {r['valid_region']['rgb_max']} | {r['valid_region']['different_pixel_fraction']:.6%} |" for r in results]
    if rendered:
        lines += ["", "## Native midpoint reference", "", "| Frame | SDK MAE | Official MAE |", "|---|---|---|"]
        lines += [f"| {r['current_frame']} | {r['sdk_vs_native_mid']['rgb_mae']:.6f} | {r['official_vs_native_mid']['rgb_mae']:.6f} |" for r in results]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
