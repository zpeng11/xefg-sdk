"""Replay verified calibration pairs through the public SDK API."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
from xefg_sdk.executors import OrtExecutor
from PIL import Image
from xefg_sdk import Config, FrameInputs, ImageView, UiMode, Xefg2xRuntime


MODE_UI = {1: UiMode.NONE, 3: UiMode.HUDLESS_UITEXTURE, 4: UiMode.BACKBUFFER_HUDLESS}
MODE_INPUTS = {1: "backbuffer", 3: "hudless+ui", 4: "hudless+backbuffer"}


def _manifest_sizes(manifest):
    """Return validated output/motion sizes from the input manifest."""
    width, height = int(manifest["width"]), int(manifest["height"])
    motion_width = int(manifest["motion_width"])
    motion_height = int(manifest["motion_height"])
    if width <= 0 or height <= 0 or motion_width <= 0 or motion_height <= 0:
        raise ValueError("Replay requires positive output and motion dimensions")
    if (motion_width, motion_height) != (width // 2, height // 2):
        raise ValueError("Replay requires half-resolution depth/motion")
    return width, height, motion_width, motion_height


def _selected_profile(package, mode, output_size, motion_size, profile_id):
    """Best-effort package metadata for the profile selected by the runtime."""
    variants = getattr(package, "variants", None)
    profiles = getattr(package, "profiles", None)
    if not isinstance(variants, dict) or not isinstance(profiles, dict):
        return None
    for variant_id, variant in variants.items():
        if variant.get("mode") != mode:
            continue
        for profile in profiles.get(variant_id, []):
            if profile.get("id") != profile_id:
                continue
            if (tuple(profile.get("output_size", ())) != tuple(output_size)
                    or tuple(profile.get("motion_size", ())) != tuple(motion_size)):
                raise ValueError("Selected profile dimensions differ from the replay inputs")
            return {"variant_id": variant_id, "profile_id": profile_id,
                    "output_size": list(profile["output_size"]),
                    "motion_size": list(profile["motion_size"]),
                    "model": variant.get("model"),
                    "model_sha256": variant.get("model_sha256")}
    return None



def load_frame(run, manifest, index, reset=False, mode=3):
    """Load one frame; the default mode 3 keeps older diagnostics compatible."""
    if mode not in MODE_UI:
        raise ValueError("Replay supports modes 1, 3 and 4")
    row = manifest["frames"][index]
    if row["index"] != index:
        raise ValueError("Input manifest frame order differs from frame IDs")
    height, width = manifest["height"], manifest["width"]
    mh, mw = manifest["motion_height"], manifest["motion_width"]
    def view(role, dtype, shape, format):
        return ImageView(np.fromfile(run / "application_inputs" / row["inputs"][role],
                                    dtype=dtype).reshape(shape), format)
    def color(role):
        return view(role, np.uint8, (height, width, 4), "BGRA8_UNORM")
    common = dict(frame_id=index,
                  depth=view("depth", "<f4", (mh, mw), "R32_FLOAT"),
                  motion_vectors=view("motion", "<f2", (mh, mw, 2), "RG16_FLOAT"),
                  reset_history=reset)
    if mode == 1:
        return FrameInputs(backbuffer_color=color("backbuffer"), **common)
    if mode == 3:
        return FrameInputs(hudless_color=color("color"), ui=color("ui"), **common)
    return FrameInputs(hudless_color=color("color"), backbuffer_color=color("backbuffer"), **common)


def _pairing_mode(pairing, capture_configuration):
    schema = pairing.get("schema")
    method = pairing.get("method")
    isolated = schema == "xefg.pairing/3" or method == "isolated_pair"
    if isolated:
        if schema != "xefg.pairing/3" or pairing.get("pairing_method") != "isolated_pair":
            raise ValueError("Isolated replay requires isolated pairing schema 3")
        if pairing.get("mode") != 1:
            raise ValueError("Isolated pairing report does not describe mode 1")
        source_pair = pairing.get("source_pair")
        if (not isinstance(source_pair, list) or len(source_pair) != 2
                or source_pair[1] != source_pair[0] + 1):
            raise ValueError("Isolated pairing report has no contiguous source pair")
        if capture_configuration.get("ui_mode") not in (None, 1):
            raise ValueError("Capture metadata is not mode 1")
        if (capture_configuration.get("pairing_method") not in (None, "isolated_pair")
                or capture_configuration.get("pair_frame") not in (None, source_pair[1])):
            raise ValueError("Capture metadata differs from the isolated pairing report")
        return 1, isolated
    mode = pairing.get("mode", pairing.get("ui_mode", capture_configuration.get("ui_mode", 3)))
    mode = int(mode)
    if mode not in (3, 4):
        raise ValueError("Rendered replay supports modes 3 and 4")
    if capture_configuration.get("ui_mode") is not None and int(capture_configuration["ui_mode"]) != mode:
        raise ValueError("Pairing mode differs from captured ui_mode")
    return mode, isolated


def _ui_region_mask(run, manifest, index, valid):
    """Current UI nonzero-alpha pixels, intersected with (never widening) the valid mask."""
    row = manifest["frames"][index]
    height, width = manifest["height"], manifest["width"]
    pixels = np.fromfile(run / "application_inputs" / row["inputs"]["ui"],
                         dtype=np.uint8).reshape(height, width, 4)
    return (pixels[:, :, 3] != 0) & valid


def run_replay(args, tools):
    run = args.run.resolve()
    pairing_path = args.pairing.resolve()
    pairing = tools.read_json(pairing_path)
    if pairing["schema"] not in ("xefg.pairing/1", "xefg.pairing/2", "xefg.pairing/3") \
            or pairing["status"] != "paired":
        raise ValueError("Replay requires a complete pairing report")
    isolated = pairing.get("schema") == "xefg.pairing/3" or pairing.get("method") == "isolated_pair"
    rendered = pairing.get("method") == "rendered_scene" or isolated
    if pairing.get("official_tags_requested") and not rendered:
        raise ValueError("Use an untagged run for SDK quality comparison")
    if rendered and pairing["schema"] not in ("xefg.pairing/2", "xefg.pairing/3"):
        raise ValueError("Rendered comparison requires pairing v2 (mode 3/4) or v3 (mode 1)")
    if rendered and not pairing.get("official_tags_requested"):
        raise ValueError("Rendered comparison requires official tagged pairing")
    if Path(pairing["capture_directory"]).resolve() != run:
        raise ValueError("Pairing report belongs to another capture")
    tools.inspect_run(run)
    for item in pairing["artifacts"]:
        current = tools.artifact(run, item["file"], item["bytes"])
        if current["sha256"] != item["sha256"]:
            raise ValueError("Capture changed since pairing: " + item["file"])
    manifest = tools.read_json(run / "application_inputs/manifest.json")
    width, height, motion_width, motion_height = _manifest_sizes(manifest)
    output_size = (width, height)
    motion_size = (motion_width, motion_height)
    meta = tools.read_json(run / tools.metadata_name(run))
    capture_configuration = meta.get("configuration", {})
    mode, isolated = _pairing_mode(pairing, capture_configuration)
    calibration = tools.read_json(run / ("scene.json" if rendered else "calibration/manifest.json"))
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
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    with Xefg2xRuntime.load(args.package, executor=OrtExecutor(session_options=session_options)) as runtime:
        package_files.append(tools.artifact(runtime.package.directory, "manifest.json"))
        for variant in runtime.package.variants.values():
            for name in [variant["model"], *variant["profiles"]]:
                package_files.append(tools.artifact(runtime.package.directory, name))
        for pair in pairs:
            previous_id, current_id = pair["previous_frame"], pair["current_frame"]
            if current_id in resets or previous_id != current_id - 1:
                raise ValueError("Pair cannot be replayed across reset/missing history")
            previous = load_frame(run, manifest, previous_id,
                                  reset=previous_id in resets, mode=mode)
            current = load_frame(run, manifest, current_id, mode=mode)
            config = Config(ui_mode=MODE_UI[mode])
            runtime.prepare(previous, current, config)
            start = time.perf_counter()
            result = runtime.interpolate_pair(previous, current, config)
            elapsed = time.perf_counter() - start
            generated = result.generated_frame.data
            if generated.shape != (height, width, 4):
                raise ValueError("SDK generated frame differs from the input manifest size")
            official = np.fromfile(run / pair["file"], np.uint8)
            if official.size != width * height * 4:
                raise ValueError("Official present differs from the input manifest size")
            official = official.reshape(height, width, 4)
            mask = np.ones((height, width), dtype=bool)
            marker = pairing.get("marker")
            if rendered:
                mask = tools.load_tool("rendered_pairing").mask_rects(
                    (height, width), pairing["excluded_rects"])
            elif marker:
                x, y = marker["x"], marker["y"]
                mask[y:y + marker["cell_height"], x:x + marker["cell_width"] * marker["bits"]] = False
            stem = f"frame_{current_id:05d}"
            np.save(output / (stem + ".npy"), generated, allow_pickle=False)
            Image.fromarray(metrics.rgb_image(official)).save(output / (stem + "_official.png"))
            Image.fromarray(metrics.rgb_image(generated)).save(output / (stem + "_sdk.png"))
            difference = metrics.difference_image(official, generated)
            Image.fromarray(difference).save(output / (stem + "_diff_x8.png"))
            if rendered:
                difference[~mask] = 0
                Image.fromarray(difference).save(output / (stem + "_valid_diff_x8.png"))
            source_current = pair.get("source_current_frame", current_id)
            source_previous = pair.get("source_previous_frame", previous_id)
            row = {"previous_frame": previous_id, "current_frame": current_id,
                   "local_previous_frame": pair.get("local_previous_frame", previous_id),
                   "local_current_frame": pair.get("local_current_frame", current_id),
                   "source_previous_frame": source_previous,
                   "source_current_frame": source_current,
                   "source_frame": source_current,
                   "previous_time": pair.get("previous_time",
                                             references.get(previous_id, {}).get("time")),
                   "current_time": pair.get("current_time",
                                            references.get(current_id, {}).get("time")),
                   "source_pair": pairing.get("source_pair", [source_previous, source_current]),
                   "mode": mode, "ui_mode": int(mode), "ui_mode_name": UiMode(mode).name,
                   "present_sequence": pair["sequence"], "profile": result.profile_id,
                   "output_size": [width, height],
                   "motion_size": [motion_width, motion_height],
                   "profile_details": _selected_profile(runtime.package, mode, output_size,
                                                        motion_size, result.profile_id),
                   "wall_seconds_including_first_load": elapsed,
                   "full": metrics.compare(official, generated),
                   "valid_region": metrics.compare(official, generated, mask),
                   "output": stem + ".npy"}
            if mode in (3, 4):
                ui_mask = _ui_region_mask(run, manifest, current_id, mask)
                row["ui_region"] = metrics.compare(official, generated, ui_mask) if ui_mask.any() else None
                row["ui_region_definition"] = "current UI alpha != 0 intersect valid region"
            if rendered:
                reference_row = references[current_id]
                midpoint = reference_row.get("midpoint")
                if not midpoint or not (run / midpoint).exists():
                    raise ValueError("Rendered replay requires the native midpoint reference")
                reference = np.fromfile(run / midpoint, np.uint8)
                if reference.size != width * height * 4:
                    raise ValueError("Native midpoint differs from the input manifest size")
                reference = reference.reshape(height, width, 4)
                Image.fromarray(metrics.rgb_image(reference)).save(output / (stem + "_native_mid.png"))
                row["sdk_vs_native_mid"] = metrics.compare(reference, generated, mask)
                row["official_vs_native_mid"] = metrics.compare(reference, official, mask)
            if not rendered:
                row["without_marker"] = row["valid_region"]
            results.append(row)
            print(f"frame {current_id} (source {source_current}, mode {mode}): "
                  f"valid RGB MAE {row['valid_region']['rgb_mae']:.6f}, {elapsed:.2f}s", flush=True)
    repo = Path(__file__).resolve().parents[2]
    def git(*command):
        return subprocess.check_output(["git", "-C", str(repo), *command], text=True).strip()
    comparison_note = "exclude official tags and frame marker" if rendered else "exclude frame marker"
    if isolated:
        comparison_note = "exclude official tag layout; mode 1 has no frame marker"
    report = {"schema": "xefg.replay_comparison/1", "status": "measured", "quality_gate": "not_configured",
              "capture_directory": str(run), "package_directory": str(args.package.resolve()),
              "command": sys.argv, "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
              "python": platform.python_version(),
              "mode": mode, "ui_mode": int(mode), "ui_mode_name": UiMode(mode).name,
              "output_size": [width, height], "motion_size": [motion_width, motion_height],
              "selected_profiles": sorted({row["profile"] for row in results}),
              "profile_details": next((row["profile_details"] for row in results
                                       if row.get("profile_details")), None),
              "source_pair": pairing.get("source_pair"),
              "configuration": {"ui_mode": int(mode), "ui_mode_name": UiMode(mode).name,
                                "frame_inputs": MODE_INPUTS[mode],
                                "output_size": [width, height],
                                "motion_size": [motion_width, motion_height],
                                "intra_op_threads": 1, "inter_op_threads": 1,
                                "pairing_schema": pairing["schema"],
                                "pairing_method": pairing.get("pairing_method", pairing.get("method")),
                                "capture": capture_configuration},
              "executor": {"provider": "CPUExecutionProvider", "intra_op_threads": 1, "inter_op_threads": 1},
              "versions": {name: importlib.metadata.version(name) for name in ("xefg-sdk", "numpy", "onnxruntime", "pillow")},
              "provider": {key: meta[key] for key in ("adapter", "xefg_family", "xefg_dll_sha256")},
              "pairing": tools.artifact(pairing_path.parent, pairing_path.name),
              "package_artifacts": package_files, "pairs": results,
              "comparison_region": comparison_note,
              "excluded_rects": pairing.get("excluded_rects", []),
              "full_frame_note": "Full-frame metrics include debug overlays; use valid_region for comparison." if rendered else "No official debug overlay.",
              "ui_region_note": "For modes 3/4, ui_region metrics use current UI nonzero-alpha pixels intersected with the valid region; the valid mask is not widened.",
              "output_artifacts": [tools.artifact(output, p.name) for p in sorted(output.iterdir())],
              "timing_note": "CPU replay wall time includes first model load; not a real-time benchmark."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# SDK vs official output", "", "Measured only; no quality pass/fail threshold.",
             f"Mode {mode} ({UiMode(mode).name}); inputs: {MODE_INPUTS[mode]}.",
             "Metrics below use the valid region, excluding the recorded overlays.", "",
             "| Current frame | Source frame | RGB MAE | Max | Changed pixels |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['current_frame']} | {r['source_current_frame']} | {r['valid_region']['rgb_mae']:.6f} "
              f"| {r['valid_region']['rgb_max']} | {r['valid_region']['different_pixel_fraction']:.6%} |"
              for r in results]
    if rendered:
        lines += ["", "## Native midpoint reference", "", "| Frame | SDK MAE | Official MAE |", "|---|---|---|"]
        lines += [f"| {r['current_frame']} | {r['sdk_vs_native_mid']['rgb_mae']:.6f} "
                  f"| {r['official_vs_native_mid']['rgb_mae']:.6f} |"
                  for r in results if "sdk_vs_native_mid" in r]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
