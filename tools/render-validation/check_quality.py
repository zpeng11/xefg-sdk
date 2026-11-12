"""Apply explicit numeric regression limits to valid-region replay reports."""
import argparse
import json
import math
from pathlib import Path


def evaluate(report, mae=0.011, maximum=32, fraction=0.001):
    failures = []
    warnings = []
    if report["status"] != "measured" or not report["pairs"]:
        raise ValueError("Expected a completed, nonempty replay report")
    for pair in report["pairs"]:
        values = pair["valid_region"]
        if values["tolerance"] != 2:
            raise ValueError("Regression fraction expects a two-byte tolerance")
        if any(not math.isfinite(values[key]) or values[key] < 0 for key in
               ("rgb_mae", "rgb_max", "over_tolerance_pixel_fraction")):
            raise ValueError("Regression metrics must be finite and nonnegative")
        if 8 < values["rgb_max"] <= maximum:
            warnings.append({"frame": pair["current_frame"],
                             "source_frame": pair.get("source_current_frame", pair["current_frame"]),
                             "reason": "rgb_max_above_8", "rgb_max": values["rgb_max"]})
        exceeded = []
        if values["rgb_mae"] > mae:
            exceeded.append("rgb_mae")
        if values["rgb_max"] > maximum:
            exceeded.append("rgb_max")
        if values["over_tolerance_pixel_fraction"] > fraction:
            exceeded.append("over_2_fraction")
        if exceeded:
            failures.append({"frame": pair["current_frame"],
                             "source_frame": pair.get("source_current_frame", pair["current_frame"]),
                             "exceeded": exceeded, "metrics": values})
    return {"status": "pass" if not failures else "fail", "frames": len(report["pairs"]),
            "limits": {"rgb_mae": mae, "rgb_max": maximum, "over_2_fraction": fraction},
            "failures": failures, "warnings": warnings, "warning_threshold": {"rgb_max": 8},
            "ui_mode": report.get("ui_mode", 3),
            "output_size": report.get("output_size", [640, 360]),
            "scope": "Uniform numeric regression target for rendered mode1/3/4 scenes; no automatic resolution scaling."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mae", type=float, default=0.011)
    parser.add_argument("--maximum", type=int, default=32)
    parser.add_argument("--fraction", type=float, default=0.001)
    args = parser.parse_args()
    if not math.isfinite(args.mae) or args.mae < 0 or args.maximum < 0 or not 0 <= args.fraction <= 1:
        parser.error("Limits must be nonnegative and fraction must be in [0,1]")
    checks = [{"report": str(path.resolve()), **evaluate(json.loads(path.read_text(encoding="utf-8")),
               args.mae, args.maximum, args.fraction)} for path in args.report]
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"schema": "xefg.quality_gate/1", "checks": checks}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for check in checks:
        print(check["status"], check["frames"], check["report"], f"warnings={len(check['warnings'])}")
    raise SystemExit(0 if all(c["status"] == "pass" for c in checks) else 1)
