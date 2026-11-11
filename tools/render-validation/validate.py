"""Capture, pair and compare official XeSS-FG output with same-input SDK replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PIXEL_BYTES = {28: 4, 29: 4, 87: 4, 91: 4, 24: 4, 10: 8}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(root, name, expected_bytes=None):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact is outside capture directory: {name}")
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"{name}: expected {expected_bytes} bytes, found {size}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": name, "bytes": size, "sha256": digest.hexdigest()}


def metadata_name(run):
    return "meta.json" if (Path(run) / "meta.json").exists() else "trace/meta.json"


def inspect_run(run):
    run = Path(run).resolve()
    presents = read_json(run / "presents.json")
    inputs = read_json(run / "application_inputs/manifest.json")
    meta_path = metadata_name(run)
    meta = read_json(run / meta_path)
    if presents["schema"] not in ("xefg.present_capture/1", "xefg.present_capture/2"):
        raise ValueError("Unsupported present manifest schema")
    if inputs["schema"] != "xefg.application_inputs/1":
        raise ValueError("Unsupported application input schema")
    if presents["capture_error"]:
        raise ValueError(presents["capture_error"])
    if not presents["presents"]:
        raise ValueError("No native presents were captured")

    config = meta["configuration"]
    width, height = inputs["width"], inputs["height"]
    if (width, height) != (config["width"], config["height"]):
        raise ValueError("Input and capture dimensions differ")
    if inputs["formats"] != {"color": "BGRA8", "ui": "BGRA8", "backbuffer": "BGRA8",
                             "depth": "R32_FLOAT", "motion": "R16G16_FLOAT"}:
        raise ValueError("This stage expects the capture tool's BGRA8/float input formats")
    expected_count = config["warmup"] + config["frames"]
    if inputs["frame_count"] != expected_count:
        raise ValueError("Input frame count differs from capture configuration")
    if [frame["index"] for frame in inputs["frames"]] != list(range(expected_count)):
        raise ValueError("Application inputs must contain every frame, including warmup")

    files = [artifact(run, name) for name in
             ("presents.json", "application_inputs/manifest.json", meta_path)]
    for frame in inputs["frames"]:
        for role in ("color", "ui", "backbuffer", "depth", "motion"):
            pixels = (inputs["motion_width"] * inputs["motion_height"]
                      if role in ("depth", "motion") else width * height)
            files.append(artifact(run, "application_inputs/" + frame["inputs"][role], pixels * 4))

    observed = []
    ordinals = {}
    for sequence, present in enumerate(presents["presents"]):
        if present["sequence"] != sequence:
            raise ValueError("Present sequence is not contiguous")
        frame = present["observed_application_frame"]
        if frame not in range(config["warmup"], expected_count):
            raise ValueError(f"Present {sequence} has an out-of-range observed frame")
        ordinal = ordinals.get(frame, 0)
        if present["ordinal"] != ordinal:
            raise ValueError(f"Present {sequence} has an inconsistent ordinal")
        ordinals[frame] = ordinal + 1
        if (present["width"], present["height"]) != (width, height):
            raise ValueError(f"Present {sequence} dimensions differ from inputs")
        bpp = PIXEL_BYTES[present["format"]]
        files.append(artifact(run, present["file"], width * height * bpp))
        observed.append({**present, "classification": "unassigned"})

    statuses = presents["frame_status"]
    if [status["application_frame"] for status in statuses] != list(range(expected_count)):
        raise ValueError("Present status records must cover all application frames")
    issues = []
    if not presents["only_interpolation_requested"] or presents["only_interpolation_result"] != 0:
        issues.append("Interpolation-only debug mode was not successfully enabled")
    for status in statuses:
        if status["application_frame"] < config["warmup"]:
            continue
        if (status["present_result"] != 0 or status["status_result"] != 0
                or status["frame_generation_result"] != 0 or status["frames_presented"] == 0):
            issues.append(f"Official present status needs inspection: {status}")
    missing = sorted(set(range(config["warmup"], expected_count)) - set(ordinals))
    if missing:
        issues.append(f"No native present observed under application frame labels: {missing}")

    return {
        "schema": "xefg.render_validation/1",
        "stage": "official_capture_inventory",
        "status": "needs_attention" if issues else "inventory_complete",
        "capture_directory": str(run),
        "configuration": config,
        "provider": {key: meta[key] for key in
                     ("adapter", "vendor_id", "device_id", "xefg_family", "xefg_dll_sha256")},
        "input_frame_count": expected_count,
        "present_count": len(observed),
        "pairing": "unverified",
        "pairing_note": "Observed application labels are not proven source-frame IDs. "
                        "No generated/base classification or quality comparison is performed.",
        "issues": issues,
        "presents": observed,
        "frame_status": statuses,
        "artifacts": files,
    }


def write_report(run, output):
    report = inspect_run(run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Official capture inventory", "", f"Status: {report['status']}",
             f"Application inputs: {report['input_frame_count']} frames (including warmup)",
             f"Native presents: {report['present_count']}", "", report["pairing_note"], "",
             "## Issues", *(report["issues"] or ["None in the capture inventory."]), ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def capture(args):
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / "capture"
    command = [str(args.probe.resolve()), "--xess-dir", str(args.xess_bin.resolve()),
               "--out", str(run), "--width", "640", "--height", "360",
               "--mv-width", "320", "--mv-height", "180", "--frames", str(args.frames),
               "--warmup", "4", "--ui", "on", "--ui-mode", "hudless-uitexture",
               "--dump-inputs", "--no-model-dump"]
    if args.adapter:
        command += ["--adapter", args.adapter]
    for flag in ("calibration", "tag_interpolated", "all_presents", "static"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    command += ["--pacing-ms", str(args.pacing_ms)]
    if args.reset_frame is not None:
        command += ["--reset-frame", str(args.reset_frame)]
    invocation = {"command": command, "probe": artifact(args.probe.resolve().parent, args.probe.name)}
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n", encoding="utf-8")
    with (output / "capture.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return write_report(run, output / "report")


def render_capture(args):
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / "capture"
    command = [str(args.renderer.resolve()), "--out", str(run), "--scene", args.scene,
               "--frames", str(args.frames), "--warmup", "4", "--pacing-ms", str(args.pacing_ms)]
    if args.xess_bin:
        command += ["--xess-dir", str(args.xess_bin.resolve())]
    for flag in ("render_only", "warp", "debug", "no_tag"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    invocation = {"command": command, "renderer": artifact(args.renderer.resolve().parent, args.renderer.name)}
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n", encoding="utf-8")
    with (output / "capture.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.render_only:
        print(f"render-only output: {run}")
        return 0
    report = write_report(run, output / "inventory")
    print(f"{report['status']}: {report['present_count']} presents")
    return 1 if report["issues"] else 0


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inventory an existing capture with presents.json")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.add_argument("--out", type=Path, required=True, help="New report directory")
    live = commands.add_parser("capture", help="Run the probe without tensor snapshots, then inventory")
    live.add_argument("--probe", type=Path, required=True)
    live.add_argument("--xess-bin", type=Path, required=True)
    live.add_argument("--out", type=Path, required=True, help="New run directory")
    live.add_argument("--frames", type=int, default=8)
    live.add_argument("--adapter", default="")
    live.add_argument("--calibration", action="store_true")
    live.add_argument("--tag-interpolated", action="store_true")
    live.add_argument("--all-presents", action="store_true")
    live.add_argument("--static", action="store_true")
    live.add_argument("--reset-frame", type=int)
    live.add_argument("--pacing-ms", type=int, default=16)
    pair = commands.add_parser("pair", help="Pair calibration outputs using pixel evidence")
    pair.add_argument("--run", type=Path, required=True)
    pair.add_argument("--out", type=Path, required=True)
    replay = commands.add_parser("replay", help="Compare verified pairs with SDK CPU replay")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--pairing", type=Path, required=True)
    replay.add_argument("--package", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--all", action="store_true", help="Replay all verified pairs, rather than the first")
    render = commands.add_parser("render", help="Capture a real DX12 raster scene")
    render.add_argument("--renderer", type=Path, required=True)
    render.add_argument("--xess-bin", type=Path)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--scene", choices=("translation", "occlusion", "camera", "cut"), default="translation")
    render.add_argument("--frames", type=int, default=8)
    render.add_argument("--pacing-ms", type=int, default=16)
    for flag in ("render-only", "warp", "debug", "no-tag"):
        render.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    if args.command == "render":
        if not 1 <= args.frames <= 65532 or args.pacing_ms < 0:
            parser.error("render requires 1..65532 measured frames and nonnegative pacing")
        if not args.render_only and (not args.xess_bin or args.warp):
            parser.error("FG rendering requires --xess-bin and a hardware GPU")
        return render_capture(args)
    if args.command == "replay":
        load_tool("replay").run_replay(args, sys.modules[__name__])
        return 0
    if args.command == "pair":
        if (args.run / "scene.json").exists():
            report = load_tool("rendered_pairing").write_pairing(args.run, args.out, sys.modules[__name__])
        else:
            report = load_tool("pairing").write_pairing(args.run, args.out, inspect_run, artifact)
        print(f"{report['status']}: {report['counts']}")
        return 0 if report["status"] == "paired" else 1
    if args.command == "capture":
        if args.frames < 1:
            parser.error("--frames must be positive")
        if args.pacing_ms < 0:
            parser.error("--pacing-ms must be nonnegative")
        if args.reset_frame is not None and not 0 <= args.reset_frame < args.frames + 4:
            parser.error("--reset-frame must identify an application frame, including warmup")
        if args.calibration and args.frames + 4 > 65536:
            parser.error("calibration uses 16-bit application frame IDs")
    report = capture(args) if args.command == "capture" else write_report(args.run, args.out)
    print(f"{report['status']}: {report['present_count']} presents; pairing {report['pairing']}")
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
