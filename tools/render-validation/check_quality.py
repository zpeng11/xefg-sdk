"""Apply explicit numeric regression limits to valid-region replay reports."""
import argparse
import json
from pathlib import Path


def evaluate(report, mae=0.01, maximum=8, fraction=0.0002):
    failures = []
    if report["status"] != "measured" or not report["pairs"]:
        raise ValueError("Expected a completed, nonempty replay report")
    for pair in report["pairs"]:
        values = pair["valid_region"]
        if values["tolerance"] != 2:
            raise ValueError("Regression fraction expects a two-byte tolerance")
        exceeded = []
        if values["rgb_mae"] > mae:
            exceeded.append("rgb_mae")
        if values["rgb_max"] > maximum:
            exceeded.append("rgb_max")
        if values["over_tolerance_pixel_fraction"] > fraction:
            exceeded.append("over_2_fraction")
        if exceeded:
            failures.append({"frame": pair["current_frame"], "exceeded": exceeded, "metrics": values})
    return {"status": "pass" if not failures else "fail", "frames": len(report["pairs"]),
            "limits": {"rgb_mae": mae, "rgb_max": maximum, "over_2_fraction": fraction},
            "failures": failures,
            "scope": "Numeric regression gate for the current 640x360 mode3 test scenes, not general perceptual quality."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mae", type=float, default=0.01)
    parser.add_argument("--maximum", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.0002)
    args = parser.parse_args()
    if args.mae < 0 or args.maximum < 0 or not 0 <= args.fraction <= 1:
        parser.error("Limits must be nonnegative and fraction must be in [0,1]")
    checks = [{"report": str(path.resolve()), **evaluate(json.loads(path.read_text(encoding="utf-8")),
               args.mae, args.maximum, args.fraction)} for path in args.report]
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"schema": "xefg.quality_gate/1", "checks": checks}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for check in checks:
        print(check["status"], check["frames"], check["report"])
    raise SystemExit(0 if all(c["status"] == "pass" for c in checks) else 1)
