"""Run the four fixed raster scenes serially and retain per-stage results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


SCENES = ("translation", "occlusion", "camera", "cut")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--xess-bin", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    results = []
    for scene in SCENES:
        directory = output / scene
        commands = [
            ("render", "validate.py", ["render", "--renderer", str(args.renderer.resolve()),
             "--xess-bin", str(args.xess_bin.resolve()), "--scene", scene, "--out", str(directory)]),
            ("inputs", "check_render_inputs.py", ["--run", str(directory / "capture"),
             "--out", str(directory / "input-check.json")]),
            ("pair", "validate.py", ["pair", "--run", str(directory / "capture"),
             "--out", str(directory / "pairing")]),
            ("replay", "validate.py", ["replay", "--run", str(directory / "capture"),
             "--pairing", str(directory / "pairing/pairing.json"), "--package", str(args.package.resolve()),
             "--out", str(directory / "replay"), "--all"]),
        ]
        result = {"scene": scene, "stages": []}
        for stage, script, arguments in commands:
            command = [sys.executable, "-I", "-B", str(tools / script), *arguments]
            with (output / f"{scene}-{stage}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            result["stages"].append({"stage": stage, "exit_code": process.returncode, "command": command})
            print(f"{scene}: {stage} exit {process.returncode}", flush=True)
            if process.returncode:
                # History-reset experiments can report inventory warnings even
                # though the native process and captured resources are complete.
                if stage == "render" and (directory / "inventory/report.json").exists():
                    continue
                break
        result["completed"] = result["stages"][-1]["stage"] == "replay" and result["stages"][-1]["exit_code"] == 0
        if result["completed"]:
            report = json.loads((directory / "replay/report.json").read_text(encoding="utf-8"))
            result["generated_frames"] = len(report["pairs"])
            result["max_frame_rgb_mae"] = max(row["valid_region"]["rgb_mae"] for row in report["pairs"])
            result["max_channel_error"] = max(row["valid_region"]["rgb_max"] for row in report["pairs"])
        results.append(result)
        (output / "matrix.json").write_text(json.dumps({"schema": "xefg.render_matrix/1", "scenes": results},
                                                       indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if all(row["completed"] for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
