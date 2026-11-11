"""Summarize GPU-rendered inputs without loading the FG runtime or a model."""
import argparse
import json
from pathlib import Path

import numpy as np


def inspect_inputs(run):
    run = Path(run)
    manifest = json.loads((run / "application_inputs/manifest.json").read_text(encoding="utf-8"))
    scene = json.loads((run / "scene.json").read_text(encoding="utf-8"))
    if scene["schema"] != "xefg.render_scene/1":
        raise ValueError("Expected a rendered scene manifest")
    count = manifest["motion_width"] * manifest["motion_height"]
    rows = []
    for frame in manifest["frames"]:
        files = frame["inputs"]
        depth = np.fromfile(run / "application_inputs" / files["depth"], dtype="<f4")
        motion = np.fromfile(run / "application_inputs" / files["motion"], dtype="<f2").reshape(-1, 2).astype(np.float32)
        if depth.size != count or len(motion) != count:
            raise ValueError("Geometry resource size mismatch")
        if not np.isfinite(depth).all() or not np.isfinite(motion).all():
            raise ValueError("Non-finite depth or motion")
        if depth.min() < 0 or depth.max() > 1 or not np.any(depth < 1):
            raise ValueError("Expected nonempty normal device-depth scene in [0,1]")
        rows.append({"frame": frame["index"], "depth_min": float(depth.min()), "depth_max": float(depth.max()),
                     "geometry_pixels": int(np.count_nonzero(depth < 1)),
                     "motion_min_xy": motion.min(axis=0).tolist(), "motion_max_xy": motion.max(axis=0).tolist(),
                     "moving_pixels": int(np.count_nonzero(np.any(motion != 0, axis=1)))})
    return {"schema": "xefg.render_input_check/1", "frames": rows,
            "note": "Finite/range checks only; motion direction and units are covered by native tests."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_inputs(args.run)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Checked {len(report['frames'])} GPU input frames")
