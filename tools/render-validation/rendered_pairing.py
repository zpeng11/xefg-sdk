"""Pair a raster scene using official frame tags and submitted UI frame IDs."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


PROFILED_PROVIDER = "ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27"
TAG_RECTS = [[0, 0, 64, 64], [576, 0, 640, 64], [0, 296, 64, 360],
             [576, 296, 640, 360], [48, 0, 64, 360], [576, 0, 592, 360]]


def mask_rects(shape, rects):
    mask = np.ones(shape[:2], dtype=bool)
    height, width = shape[:2]
    for x0, y0, x1, y1 in rects:
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Excluded rectangle lies outside the image")
        mask[y0:y1, x0:x1] = False
    if not mask.any():
        raise ValueError("Excluded regions cover the whole image")
    return mask


def tag_evidence(pixels):
    fractions = []
    for x0, y0, x1, y1 in TAG_RECTS:
        region = pixels[y0:y1, x0:x1, :3]
        fractions.append(float(np.all(region == [255, 0, 255], axis=2).mean()))
    return all(value == 1 for value in fractions), fractions


def classify(present, frame, record, status, tagged, base_mae):
    if record is None or status is None:
        return "ambiguous", "UI marker has no matching submission"
    if present["native_present_result"] != 0 or status["present_result"] != 0 or status["status_result"] != 0:
        return "ambiguous", "Present/status API failed"
    results = record["submission_results"]
    if any(results[key] != 0 for key in ("depth", "motion", "color", "ui", "constants", "present_id")):
        return "ambiguous", "input submission API failed"
    success = status["frame_generation_result"] == 0 and bool(status["is_frame_gen_enabled"])
    if tagged:
        if record["reset_history"] or not success or record["previous_frame"] != frame - 1 or frame <= 0:
            return "ambiguous", "official tag conflicts with history or generation status"
        return "generated", "official generated tag, decoded current UI ID and successful contiguous submission"
    if base_mae == 0:
        return ("fallback" if record["reset_history"] or not success else "base"), "matches submitted current backbuffer"
    return "ambiguous", "neither verified official tag nor current base content"


def write_pairing(run, output, tools):
    run = Path(run).resolve()
    inventory = tools.inspect_run(run)
    native = tools.read_json(run / "presents.json")
    scene = tools.read_json(run / "scene.json")
    inputs = tools.read_json(run / "application_inputs/manifest.json")
    if scene["schema"] != "xefg.render_scene/1" or native["schema"] != "xefg.present_capture/2":
        raise ValueError("Rendered scene pairing requires scene v1 and present v2")
    if (inputs["width"], inputs["height"]) != (640, 360) or inventory["configuration"]["ui_mode"] != 3:
        raise ValueError("Rendered pairing currently supports mode 3 at 640x360")
    if inventory["provider"]["xefg_dll_sha256"] != PROFILED_PROVIDER:
        raise ValueError("Official debug tag layout is not calibrated for this provider")
    if not native["tag_interpolated_requested"] or native["tag_interpolated_result"] != 0:
        raise ValueError("Rendered pairing requires the official generated-frame tag")
    marker = scene["marker"]
    marker_rect = [marker["x"], marker["y"], marker["x"] + marker["bits"] * marker["cell_width"],
                   marker["y"] + marker["cell_height"]]
    rects = [*TAG_RECTS, marker_rect]
    mask = mask_rects((360, 640), rects)
    marker_mask = np.zeros((360, 640), dtype=bool)
    marker_mask[marker_rect[1]:marker_rect[3], marker_rect[0]:marker_rect[2]] = True
    if np.any(marker_mask & ~mask_rects((360, 640), TAG_RECTS)):
        raise ValueError("Frame marker overlaps the official tag layout")
    helper = tools.load_tool("pairing")
    frames = {row["index"]: row for row in scene["frames"]}
    if sorted(frames) != list(range(inputs["frame_count"])):
        raise ValueError("Scene submissions must cover all input frames")
    statuses = {row["application_frame"]: row for row in native["frame_status"]}
    for index, record in frames.items():
        if record["previous_frame"] != (index - 1 if index else None):
            raise ValueError("Scene submission history is not contiguous")
    artifacts = [*inventory["artifacts"], tools.artifact(run, "scene.json")]
    for row in frames.values():
        artifacts.append(tools.artifact(run, row["midpoint"], 640 * 360 * 4))
    hashes = {row["file"]: row["sha256"] for row in artifacts}
    rows = []
    for present in native["presents"]:
        if present["format"] != 87:
            raise ValueError("Rendered pairing expects BGRA8_UNORM")
        pixels = helper.image(run / present["file"])
        if hashlib.sha256(pixels.tobytes()).hexdigest() != hashes[present["file"]]:
            raise ValueError("Present changed after inventory")
        frame, levels = helper.decode_marker(pixels, marker)
        record, status = frames.get(frame), statuses.get(frame)
        tagged, fractions = tag_evidence(pixels)
        base_mae = None
        if record is not None:
            base_name = "application_inputs/" + inputs["frames"][frame]["inputs"]["backbuffer"]
            base = helper.image(run / base_name)
            if base_name in hashes and hashlib.sha256(base.tobytes()).hexdigest() != hashes[base_name]:
                raise ValueError("Base image changed after inventory")
            base_mae = float(np.abs(pixels[mask, :3].astype(np.int16) - base[mask, :3]).mean())
        classification, reason = classify(present, frame, record, status, tagged, base_mae)
        if frame is not None and frame < inventory["configuration"]["warmup"]:
            classification, reason = "ambiguous", "captured warmup content"
        rows.append({"sequence": present["sequence"], "file": present["file"], "sha256": hashes[present["file"]],
                     "current_frame": frame, "previous_frame": record["previous_frame"] if record else None,
                     "decoded_ui_frame": frame, "observed_application_frame": present["observed_application_frame"],
                     "classification": classification, "reason": reason, "marker_levels": levels,
                     "official_tag_detected": tagged, "tag_region_fractions": fractions,
                     "base_mae_without_overlays": base_mae, "official_status": status})
    generated = collections.Counter(r["current_frame"] for r in rows if r["classification"] == "generated")
    for row in rows:
        if row["classification"] == "generated" and generated[row["current_frame"]] != 1:
            row.update(classification="ambiguous", reason="duplicate generated output")
    warmup = inventory["configuration"]["warmup"]
    expected = {i for i, f in frames.items() if i >= warmup and not f["reset_history"]}
    paired = {r["current_frame"] for r in rows if r["classification"] == "generated"}
    missing = sorted(expected - paired)
    counts = dict(collections.Counter(r["classification"] for r in rows))
    report = {"schema": "xefg.pairing/2", "method": "rendered_scene", "capture_directory": str(run),
              "status": "paired" if not missing and not counts.get("ambiguous") else "incomplete",
              "counts": counts, "missing_generated_frames": missing, "official_tags_requested": True,
              "marker": marker, "excluded_rects": rects, "valid_pixel_count": int(mask.sum()),
              "mask_provider_sha256": PROFILED_PROVIDER, "artifacts": artifacts, "pairs": rows,
              "reference_usage": "native midpoint is quality-reference-only, never an identity candidate"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pairing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Rendered scene pairing", "", f"Status: {report['status']}", f"Counts: {counts}",
             f"Missing: {missing}", "", "| Present | UI frame | Kind | Reason |", "|---|---|---|---|"]
    lines += [f"| {r['sequence']} | {r['current_frame']} | {r['classification']} | {r['reason']} |" for r in rows]
    (output / "pairing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
