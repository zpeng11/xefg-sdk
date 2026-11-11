"""Pixel-evidence pairing for the versioned 640x360 calibration scene."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def image(path, width=640, height=360):
    return np.fromfile(path, dtype=np.uint8).reshape(height, width, 4)


def decode_marker(pixels, marker):
    if marker["encoding"] != "u16-complement-lsb" or marker["bits"] != 32:
        raise ValueError("Unsupported calibration marker")
    value = 0
    levels = []
    for bit in range(32):
        x = marker["x"] + bit * marker["cell_width"]
        y = marker["y"]
        # Use cell interiors so interpolation borders do not change the bit.
        block = pixels[y + 2:y + marker["cell_height"] - 2,
                       x + 2:x + marker["cell_width"] - 2, :3]
        level = float(np.median(block))
        levels.append(level)
        if 32 < level < 223:
            return None, levels
        value |= int(level >= marker["threshold"]) << bit
    frame = value & 0xffff
    return (frame if value >> 16 == frame ^ 0xffff else None), levels


def candidate_scores(pixels, candidates, roi):
    x0, y0, x1, y1 = roi
    actual = pixels[y0:y1, x0:x1, :3].astype(np.float32)
    return sorted([
        {"kind": kind, "frame": frame,
         "mae": float(np.abs(actual - candidate[y0:y1, x0:x1, :3]).mean())}
        for kind, frame, candidate in candidates
    ], key=lambda row: row["mae"])


def classify(frame, scores, status, reset, static, tagged, only_interpolation):
    if frame is None or status is None:
        return "ambiguous", "frame marker did not identify an input/status record"
    best, second = scores[:2]
    # Require a distinct winner. This is a pairing threshold, not a quality gate.
    distinct = (best["mae"] <= 4.0 and second["mae"] - best["mae"] >= 0.5
                and best["mae"] <= second["mae"] * 0.8)
    success = (status["present_result"] == 0 and status["status_result"] == 0
               and status["frame_generation_result"] == 0 and status["is_frame_gen_enabled"])
    if static:
        return "ambiguous", "static scene cannot establish source-time pairing"
    if not distinct or best["frame"] != frame:
        return "ambiguous", "motion reference is tied or disagrees with decoded current UI frame"
    if best["kind"] == "base":
        return ("fallback" if reset or not success else "base"), "current base reference is the distinct winner"
    if reset or not success:
        return "ambiguous", "midpoint content conflicts with reset or official status"
    if not (tagged or only_interpolation):
        return "ambiguous", "generated-frame identity needs official debug evidence"
    if frame <= 0:
        return "ambiguous", "no previous application frame exists"
    return "generated", "current midpoint is the distinct winner with official generation evidence"


def write_pairing(run, output, inspect_run, artifact):
    run = Path(run).resolve()
    inventory = inspect_run(run)
    native = read_json(run / "presents.json")
    calibration = read_json(run / "calibration/manifest.json")
    inputs = read_json(run / "application_inputs/manifest.json")
    if native["schema"] != "xefg.present_capture/2":
        raise ValueError("Pairing requires present capture v2")
    if calibration["schema"] != "xefg.calibration/1":
        raise ValueError("Unsupported calibration schema")
    if inventory["configuration"]["ui_mode"] != 3 or (inputs["width"], inputs["height"]) != (640, 360):
        raise ValueError("Calibration pairing currently uses mode 3 at 640x360")
    statuses = {row["application_frame"]: row for row in native["frame_status"]}
    frames = {row["index"]: row for row in calibration["frames"]}
    if set(frames) != set(range(inputs["frame_count"])):
        raise ValueError("Calibration references must cover every application frame")
    artifacts = list(inventory["artifacts"])
    artifacts.append(artifact(run, "calibration/manifest.json"))
    candidates = []
    for row in inputs["frames"]:
        index = row["index"]
        candidates.append(("base", index, image(run / "application_inputs" / row["inputs"]["backbuffer"])))
        name = "calibration/" + frames[index]["midpoint"]
        artifacts.append(artifact(run, name, 640 * 360 * 4))
        candidates.append(("midpoint", index, image(run / name)))
    hashes = {row["file"]: row["sha256"] for row in artifacts}
    rows = []
    for present in native["presents"]:
        if present["format"] != 87:
            raise ValueError("Pairing expects BGRA8_UNORM presents")
        pixels = image(run / present["file"])
        if hashlib.sha256(pixels.tobytes()).hexdigest() != hashes[present["file"]]:
            raise ValueError("Present changed after inventory: " + present["file"])
        frame, levels = decode_marker(pixels, calibration["marker"])
        scores = candidate_scores(pixels, candidates, calibration["motion_roi"])
        status = statuses.get(frame)
        reset = frames.get(frame, {}).get("reset_history", True)
        # Official marks are detected outside the motion ROI and frame-ID strip.
        purple = ((pixels[:48, :48, 0] == 255) & (pixels[:48, :48, 1] == 0)
                  & (pixels[:48, :48, 2] == 255))
        tag_requested = native["tag_interpolated_requested"] and native["tag_interpolated_result"] == 0
        tagged = bool(tag_requested and purple.all())
        only = native["only_interpolation_requested"] and native["only_interpolation_result"] == 0
        classification, reason = classify(frame, scores, status, reset, calibration["static"], tagged, only)
        if tag_requested and classification == "generated" and not tagged:
            classification, reason = "ambiguous", "requested official generated-frame tag was not detected"
        if present.get("native_present_result", 0) != 0:
            classification, reason = "ambiguous", "native Present did not report success"
        rows.append({"sequence": present["sequence"], "file": present["file"],
                     "sha256": hashes[present["file"]], "observed_application_frame": present["observed_application_frame"],
                     "decoded_ui_frame": frame, "previous_frame": frame - 1 if frame is not None else None,
                     "current_frame": frame, "classification": classification, "reason": reason,
                     "official_tag_detected": tagged, "marker_levels": levels,
                     "candidates": scores[:6], "official_status": status})
    generated = collections.Counter(row["current_frame"] for row in rows if row["classification"] == "generated")
    for row in rows:
        if row["classification"] == "generated" and generated[row["current_frame"]] != 1:
            row.update(classification="ambiguous", reason="duplicate generated output for current frame")
    warmup = inventory["configuration"]["warmup"]
    for row in rows:
        if row["classification"] == "generated" and row["current_frame"] < warmup:
            row.update(classification="ambiguous", reason="captured output belongs to a warmup frame")
    counts = dict(collections.Counter(row["classification"] for row in rows))
    expected = {index for index, frame in frames.items()
                if index >= warmup and not frame["reset_history"]}
    paired = {row["current_frame"] for row in rows if row["classification"] == "generated"}
    missing = sorted(expected - paired)
    report = {"schema": "xefg.pairing/1", "status": "paired" if not missing and not counts.get("ambiguous") else "incomplete",
              "capture_directory": str(run), "counts": counts, "missing_generated_frames": missing,
              "official_tags_requested": native["tag_interpolated_requested"],
              "marker": calibration["marker"], "motion_roi": calibration["motion_roi"],
              "pairing_rule": {"min_mae_gap": 0.5, "max_best_to_runner_up": 0.8, "max_reference_mae": 4.0,
                               "official_tag_roi": [0, 0, 48, 48], "official_tag_bgr": [255, 0, 255]},
              "artifacts": artifacts, "pairs": rows}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pairing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Calibration pairing", "", f"Status: {report['status']}", f"Counts: {counts}",
             f"Missing generated frames: {missing}", "", "| Present | UI frame | Classification | Reason |",
             "|---|---|---|---|"]
    lines += [f"| {r['sequence']} | {r['decoded_ui_frame']} | {r['classification']} | {r['reason']} |" for r in rows]
    (output / "pairing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
