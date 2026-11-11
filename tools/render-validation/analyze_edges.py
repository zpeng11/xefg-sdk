"""Measure where SDK/provider differences occur; this does not assign a model cause."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dilate(mask, radius):
    padded = np.pad(mask, radius)
    output = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            output |= padded[y:y+h, x:x+w]
    return output


def boundaries(values, threshold):
    edge = np.zeros(values.shape[:2], dtype=bool)
    dx = np.abs(values[:, 1:] - values[:, :-1])
    dy = np.abs(values[1:] - values[:-1])
    if values.ndim == 3:
        dx, dy = dx.max(axis=2), dy.max(axis=2)
    edge[:, 1:] |= dx > threshold
    edge[:, :-1] |= dx > threshold
    edge[1:] |= dy > threshold
    edge[:-1] |= dy > threshold
    return edge


def analyze(scene_dir):
    scene_dir = Path(scene_dir)
    run = scene_dir / "capture"
    inputs = json.loads((run / "application_inputs/manifest.json").read_text())
    pairing = json.loads((scene_dir / "pairing/pairing.json").read_text())
    width, height = inputs["width"], inputs["height"]
    mw, mh = inputs["motion_width"], inputs["motion_height"]
    mask = load_tool("rendered_pairing").mask_rects((height, width), pairing["excluded_rects"])
    rows = []
    for pair in pairing["pairs"]:
        if pair["classification"] != "generated":
            continue
        frame = pair["current_frame"]
        roles = inputs["frames"][frame]["inputs"]
        official = np.fromfile(run / pair["file"], np.uint8).reshape(height, width, 4)
        sdk = np.load(scene_dir / "replay" / f"frame_{frame:05d}.npy", allow_pickle=False)
        diff = np.abs(official[:, :, :3].astype(np.int16) - sdk[:, :, :3]).max(axis=2)
        motion = np.fromfile(run / "application_inputs" / roles["motion"], "<f2").reshape(mh, mw, 2).astype(np.float32)
        depth = np.fromfile(run / "application_inputs" / roles["depth"], "<f4").reshape(mh, mw)
        ui = np.fromfile(run / "application_inputs" / roles["ui"], np.uint8).reshape(height, width, 4)
        mv_edge = boundaries(motion, 0.25).repeat(2, 0).repeat(2, 1)
        depth_edge = boundaries(depth, 0.001).repeat(2, 0).repeat(2, 1)
        significant = (diff > 2) & mask
        nonzero = (diff > 0) & mask
        edge_band = dilate(mv_edge | depth_edge, 8)
        y, x = np.where(significant)
        magnitude = np.linalg.norm(motion, axis=2)
        rows.append({
            "frame": frame, "different_pixels": int(nonzero.sum()),
            "over_2_pixels": int(significant.sum()), "max_channel_error": int(diff[mask].max()),
            "over_2_bbox_xyxy": [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1] if len(x) else None,
            "edge_band_radius_output_pixels": 8,
            "edge_band_valid_pixels": int((edge_band & mask).sum()),
            "over_2_near_motion_or_depth_edge": int((significant & edge_band).sum()),
            "over_2_under_ui": int((significant & (ui[:, :, 3] > 0)).sum()),
            "motion_magnitude_max": float(magnitude.max()),
            "motion_near_zero_pixels": int((magnitude < 0.01).sum()),
            "motion_pixels": magnitude.size,
            "depth_range": [float(depth.min()), float(depth.max())],
        })
    return {"scene_directory": str(scene_dir.resolve()), "frames": rows,
            "interpretation": "Spatial correlation only; edge proximity does not identify the faulty dispatch."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(path.parent for path in args.matrix.rglob("capture")
                         if (path.parent / "replay/report.json").exists())
    reports = {str(directory.relative_to(args.matrix)): analyze(directory) for directory in directories}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(reports, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for scene, report in reports.items():
        rows = report["frames"]
        total = sum(r["over_2_pixels"] for r in rows)
        edge = sum(r["over_2_near_motion_or_depth_edge"] for r in rows)
        ui = sum(r["over_2_under_ui"] for r in rows)
        print(f"{scene}: >2 pixels={total}, near edges={edge}, under UI={ui}")
