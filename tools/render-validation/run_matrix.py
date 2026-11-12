"""Run fixed raster scenes and explicit modes serially, including numeric gates."""
import argparse
import json
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys


SCENES = ("translation", "occlusion", "camera", "cut")
COVERAGE_SCENES = ("depth_layers", "screen_cross", "camera_turn", "nonrigid", "transparency")
EXIT_CODES = {"pass": 0, "run_failed": 1, "quality_failed": 2, "coverage_failed": 3}


SIZES = ("640x360", "1280x720", "1920x1080", "2560x1440", "3840x2160")


def preflight_package(directory, sizes, modes):
    from xefg_sdk.package import ModelPackage
    from xefg_sdk.planner import Planner
    package = ModelPackage(directory)
    planner = Planner(package)
    rows = []
    for size in sizes:
        width, height = map(int, size.split("x"))
        for mode in modes:
            plan = planner.resolve(mode, (width, height), (width // 2, height // 2))
            variant = package.variants[plan.variant_id]
            profile_path = next(name for name in variant["profiles"]
                                if package.read_json(name)["id"] == plan.profile_id)
            rows.append({"size": size, "mode": mode, "profile": plan.profile_id,
                         "variant": plan.variant_id, "model_sha256": variant["model_sha256"],
                         "profile_file": profile_path,
                         "profile_sha256": hashlib.sha256(package.path(profile_path).read_bytes()).hexdigest()})
    return rows


def run_case(args, directory, scene, mode, pair_frame=None, size="640x360"):
    tools = Path(__file__).resolve().parent
    render = ["render", "--renderer", str(args.renderer.resolve()),
              "--xess-bin", str(args.xess_bin.resolve()), "--scene", scene,
              "--mode", str(mode), "--size", size, "--out", str(directory)]
    if pair_frame is not None:
        render += ["--pair-frame", str(pair_frame)]
    commands = [
        ("render", "validate.py", render),
        ("inputs", "check_render_inputs.py", ["--run", str(directory / "capture"),
         "--out", str(directory / "input-check.json")]),
        ("pair", "validate.py", ["pair", "--run", str(directory / "capture"),
         "--out", str(directory / "pairing")]),
        ("replay", "validate.py", ["replay", "--run", str(directory / "capture"),
         "--pairing", str(directory / "pairing/pairing.json"), "--package", str(args.package.resolve()),
         "--out", str(directory / "replay"), "--all"]),
        ("quality", "check_quality.py", ["--report", str(directory / "replay/report.json"),
         "--out", str(directory / "quality.json")]),
    ]
    width, height = map(int, size.split("x"))
    result = {"directory": str(directory), "size": size, "output_size": [width, height],
              "motion_size": [width // 2, height // 2],
              "source_pair": [pair_frame - 1, pair_frame] if pair_frame else None,
              "stages": [], "status": "run_failed", "completed": False}
    directory.parent.mkdir(parents=True, exist_ok=True)
    for stage, script, arguments in commands:
        command = [sys.executable, "-I", "-B", str(tools / script), *arguments]
        log_path = directory.parent / f"{directory.name}-{stage}.log"
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result["stages"].append({"stage": stage, "exit_code": process.returncode,
                                 "command": command, "log": str(log_path)})
        print(f"{size}/mode{mode}/{scene}/{pair_frame or 'sequence'}: {stage} exit {process.returncode}", flush=True)
        if process.returncode:
            # Only a successfully written inventory can explain reset warnings.
            if stage == "render" and process.returncode == 1 and (directory / "inventory/report.json").exists():
                continue
            if stage == "inputs" and process.returncode == 3 and (directory / "input-check.json").exists():
                # Retain output comparisons even when the scene missed its coverage target.
                inputs = json.loads((directory / "input-check.json").read_text(encoding="utf-8"))
                if inputs.get("coverage", {}).get("status") == "fail":
                    continue
            if stage == "quality" and process.returncode == 1 and (directory / "quality.json").exists():
                gate = json.loads((directory / "quality.json").read_text(encoding="utf-8"))
                if gate.get("schema") == "xefg.quality_gate/1" and any(
                        check["status"] == "fail" for check in gate.get("checks", [])):
                    result["status"] = "quality_failed"
            break
        if stage == "quality":
            gate_path = directory / "quality.json"
            if gate_path.exists():
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                if gate.get("schema") == "xefg.quality_gate/1" and gate.get("checks") and all(
                        check["status"] == "pass" for check in gate["checks"]):
                    result["status"] = "pass"
    result["completed"] = result["status"] in ("pass", "quality_failed")
    if result["completed"]:
        report = json.loads((directory / "replay/report.json").read_text(encoding="utf-8"))
        result["profiles"] = sorted({row["profile"] for row in report["pairs"] if "profile" in row})
        result["valid_pixel_counts"] = sorted({row["valid_region"]["pixels"] for row in report["pairs"]
                                               if "pixels" in row["valid_region"]})
        result["generated_frames"] = len(report["pairs"])
        result["max_frame_rgb_mae"] = max(row["valid_region"]["rgb_mae"] for row in report["pairs"])
        result["max_channel_error"] = max(row["valid_region"]["rgb_max"] for row in report["pairs"])
        result["max_over_2_fraction"] = max(row["valid_region"]["over_tolerance_pixel_fraction"] for row in report["pairs"])
    return result


def combined_status(rows):
    if any(row["status"] == "run_failed" for row in rows):
        return "run_failed"
    if any(row["status"] == "coverage_failed" for row in rows):
        return "coverage_failed"
    return "quality_failed" if any(row["status"] == "quality_failed" for row in rows) else "pass"


def scene_coverage(scene, cases):
    if scene not in COVERAGE_SCENES:
        return {"status": "not_applicable"}
    spec = importlib.util.spec_from_file_location("input_coverage", Path(__file__).with_name("check_render_inputs.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    frames = []
    for case in cases:
        path = Path(case["directory"]) / "input-check.json"
        if not path.exists():
            return {"status": "pending", "reason": "input report missing"}
        frames.extend(json.loads(path.read_text(encoding="utf-8"))["frames"])
    return tool.aggregate_coverage(scene, frames)


def write_summary(output, report):
    (output / "matrix.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Render validation matrix", "", f"Status: {report['status']}", "",
             "| Size | Mode | Scene | Status | Coverage | Measured frames | Worst MAE | Max channel | Worst >2 fraction |",
             "|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in report["scenes"]:
        lines.append(f"| {row['size']} | {row['mode']} | {row['scene']} | {row['status']} | {row['coverage']['status']} | {row['generated_frames']} | "
                     f"{row['max_frame_rgb_mae']} | {row['max_channel_error']} | {row['max_over_2_fraction']} |")
    lines += ["", "Byte-unit RGB metrics exclude only recorded official tags and any frame marker.",
              "Cut source frame 7 is skipped because history resets. See JSON for stage logs and source pairs."]
    (output / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--xess-bin", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", type=int, nargs="+", choices=(1, 3, 4), default=[3])
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=["640x360"])
    parser.add_argument("--scenes", nargs="+", choices=SCENES + COVERAGE_SCENES, default=list(SCENES))
    args = parser.parse_args()
    if any(len(set(values)) != len(values) for values in (args.modes, args.sizes, args.scenes)):
        parser.error("--modes, --sizes and --scenes must not contain duplicates")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    profiles = preflight_package(args.package, args.sizes, args.modes)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for size in args.sizes:
        root = output if args.sizes == ["640x360"] else output / size
        for mode in args.modes:
            for scene in args.scenes:
                directory = root / scene if args.sizes == ["640x360"] and args.modes == [3] else root / f"mode{mode}" / scene
                skipped = [{"source_frame": 7, "reason": "camera cut resets history"}] if scene == "cut" else []
                if mode == 1:
                    cases = [run_case(args, directory / f"pair_{frame:05d}", scene, mode, frame, size)
                             for frame in range(4, 12) if not (scene == "cut" and frame == 7)]
                else:
                    cases = [run_case(args, directory, scene, mode, size=size)]
                result = {"scene": scene, "mode": mode, "size": size, "status": combined_status(cases),
                          "completed": all(row["completed"] for row in cases), "cases": cases,
                          "skipped": skipped}
                result["coverage"] = scene_coverage(scene, cases)
                result["quality_status"] = result["status"]
                if result["status"] != "run_failed" and result["coverage"]["status"] in ("fail", "pending"):
                    result["status"] = "coverage_failed"
                measured = [row for row in cases if row["completed"]]
                result["generated_frames"] = sum(row["generated_frames"] for row in measured)
                for key in ("max_frame_rgb_mae", "max_channel_error", "max_over_2_fraction"):
                    result[key] = max((row[key] for row in measured), default=None)
                results.append(result)
                expected = len(args.sizes) * len(args.modes) * len(args.scenes)
                size_results = []
                for completed_size in args.sizes:
                    rows = [row for row in results if row["size"] == completed_size]
                    size_results.append({"size": completed_size,
                                         "status": combined_status(rows) if len(rows) == len(args.modes) * len(args.scenes) else ("running" if rows else "pending"),
                                         "generated_frames": sum(row["generated_frames"] for row in rows)})
                report = {"schema": "xefg.render_matrix/4", "modes": args.modes, "sizes": args.sizes,
                          "requested_scenes": args.scenes,
                          "package_profiles": profiles, "size_results": size_results,
                          "status": combined_status(results) if len(results) == expected else "running",
                          "scenes": results, "exit_codes": EXIT_CODES}
                write_summary(output, report)
    return EXIT_CODES[combined_status(results)]


if __name__ == "__main__":
    sys.exit(main())
