"""Pair a raster scene using official frame tags and submitted UI frame IDs."""
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROFILED_PROVIDER = "ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27"
# Measured 640x360 layout; kept as the backwards-compatible default.
TAG_RECTS = [[0, 0, 64, 64], [576, 0, 640, 64], [0, 296, 64, 360],
             [576, 296, 640, 360], [48, 0, 64, 360], [576, 0, 592, 360]]
SUBMISSION_KEYS = ("depth", "motion", "color", "ui", "constants", "present_id")
MODES = (1, 3, 4)
ALLOWED_SIZES = ((640, 360), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))
# Explicit provider+size debug-tag layouts. Only layouts confirmed by a real
# provider capture are registered here; no layout is inferred or scaled.
TAG_LAYOUTS = {
    (PROFILED_PROVIDER, (640, 360)): TAG_RECTS,
    # Measured in all three modes, with a no-tag control.
    (PROFILED_PROVIDER, (1280, 720)): [
        [0, 0, 64, 64], [1216, 0, 1280, 64], [48, 64, 64, 656],
        [1216, 64, 1232, 656], [0, 656, 64, 720], [1216, 656, 1280, 720]],
    (PROFILED_PROVIDER, (2560, 1440)): [
        [0, 0, 64, 64], [2496, 0, 2560, 64], [48, 64, 64, 1376],
        [2496, 64, 2512, 1376], [0, 1376, 64, 1440], [2496, 1376, 2560, 1440]],
    (PROFILED_PROVIDER, (3840, 2160)): [
        [0, 0, 64, 64], [3776, 0, 3840, 64], [48, 64, 64, 2096],
        [3776, 64, 3792, 2096], [0, 2096, 64, 2160], [3776, 2096, 3840, 2160]],
    (PROFILED_PROVIDER, (1920, 1080)): [
        [0, 0, 64, 64], [1856, 0, 1920, 64], [48, 64, 64, 1016],
        [1856, 64, 1872, 1016], [0, 1016, 64, 1080], [1856, 1016, 1920, 1080]],
}


def register_tag_layout(provider, width, height, rects):
    """Register one confirmed provider/size debug-tag layout.

    Tests and future calibration runs use this to add an explicitly measured
    layout. New output sizes are never derived by scaling an existing entry.
    """
    size = (int(width), int(height))
    if size not in ALLOWED_SIZES:
        raise ValueError("Unsupported rendered size %dx%d" % size)
    if not rects:
        raise ValueError("Tag layout must contain at least one rectangle")
    normalized = [[int(value) for value in rect] for rect in rects]
    mask_rects((size[1], size[0]), normalized)
    TAG_LAYOUTS[(str(provider), size)] = normalized
    return normalized


def tag_layout(provider, size):
    """Return the confirmed layout for a provider and output size."""
    key = (str(provider), (int(size[0]), int(size[1])))
    try:
        return TAG_LAYOUTS[key]
    except KeyError:
        raise ValueError(
            "Official debug tag layout is not calibrated for provider %s at %dx%d"
            % (key[0], key[1][0], key[1][1])) from None


def _input_dimensions(inputs):
    """Validate the manifest's output/motion dimensions; return width, height."""
    width, height = inputs.get("width"), inputs.get("height")
    if (width, height) not in ALLOWED_SIZES:
        raise ValueError("Rendered pairing supports 640x360, 1280x720, 1920x1080, "
                         "2560x1440 and 3840x2160")
    if (inputs.get("motion_width"), inputs.get("motion_height")) != (width // 2, height // 2):
        raise ValueError("Rendered pairing requires half-resolution depth/motion")
    return width, height


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


def tag_evidence(pixels, provider=PROFILED_PROVIDER):
    """Check the confirmed layout for the pixel size; reject uncalibrated sizes."""
    height, width = pixels.shape[:2]
    rects = tag_layout(provider, (width, height))
    fractions = []
    for x0, y0, x1, y1 in rects:
        region = pixels[y0:y1, x0:x1, :3]
        fractions.append(float(np.all(region == [255, 0, 255], axis=2).mean()))
    return all(value == 1 for value in fractions), fractions


def classify(present, frame, record, status, tagged, base_mae):
    if record is None or status is None:
        return "ambiguous", "UI marker has no matching submission"
    if present["native_present_result"] != 0 or status["present_result"] != 0 or status["status_result"] != 0:
        return "ambiguous", "Present/status API failed"
    results = record["submission_results"]
    if any(results[key] is not None and results[key] != 0
           for key in ("depth", "motion", "color", "ui", "constants", "present_id")):
        return "ambiguous", "input submission API failed"
    success = status["frame_generation_result"] == 0 and bool(status["is_frame_gen_enabled"])
    if tagged:
        if record["reset_history"] or not success or record["previous_frame"] != frame - 1 or frame <= 0:
            return "ambiguous", "official tag conflicts with history or generation status"
        return "generated", "official generated tag, decoded current UI ID and successful contiguous submission"
    if base_mae == 0:
        return ("fallback" if record["reset_history"] or not success else "base"), "matches submitted current backbuffer"
    return "ambiguous", "neither verified official tag nor current base content"


def _mode_from_metadata(scene, inventory):
    configuration = inventory.get("configuration", {})
    ui_mode = configuration.get("ui_mode")
    raw = scene.get("mode", ui_mode)
    if raw is None or isinstance(raw, bool) or int(raw) != raw:
        raise ValueError("Rendered scene metadata does not identify a mode")
    mode = int(raw)
    if mode not in MODES:
        raise ValueError("Rendered pairing supports modes 1, 3 and 4")
    if ui_mode is not None and int(ui_mode) != mode:
        raise ValueError("Scene mode differs from the captured UI mode")
    return mode


def _validate_submissions(record, mode, label):
    results = record.get("submission_results")
    if not isinstance(results, dict) or set(SUBMISSION_KEYS) - set(results):
        raise ValueError(f"{label}: submission results are incomplete")
    if mode == 1:
        if results["color"] is not None or results["ui"] is not None:
            raise ValueError(f"{label}: mode 1 must not submit color or UI")
        checked = ("depth", "motion", "constants", "present_id")
    elif mode == 3:
        checked = SUBMISSION_KEYS
    else:
        if results["ui"] is not None:
            raise ValueError(f"{label}: mode 4 must not submit UI")
        checked = ("depth", "motion", "color", "constants", "present_id")
    for key in checked:
        if results[key] != 0:
            raise ValueError(f"{label}: {key} submission failed")


def _status_success(status):
    if not isinstance(status, dict):
        return False
    if status.get("present_result") != 0 or status.get("status_result") != 0:
        return False
    return (status.get("frame_generation_result") == 0
            and bool(status.get("is_frame_gen_enabled")))


def _status_reset(status):
    """First isolated submission: known reset warning with no presented frames.

    A provider may instead present the reset/base frame; that documented case
    reports generation success with at least one presented frame.
    """
    if not isinstance(status, dict):
        return False
    if status.get("present_result") != 0 or status.get("status_result") != 0:
        return False
    generation = status.get("frame_generation_result")
    presented = status.get("frames_presented", 0)
    if generation == 3 and presented == 0:
        return True
    return generation == 0 and presented >= 1


def _frame_time(record):
    value = record.get("time")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _write_report(output, report, rows, title):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pairing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    lines = ["# " + title, "", f"Status: {report['status']}", f"Counts: {report['counts']}",
             f"Mode: {report.get('mode')}", f"Missing: {report['missing_generated_frames']}", "",
             "| Present | Local frame | Source frame | Kind | Reason |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['sequence']} | {r.get('current_frame')} | {r.get('source_current_frame')} "
              f"| {r['classification']} | {r['reason']} |" for r in rows]
    (output / "pairing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pairing(run, output, tools):
    run = Path(run).resolve()
    inventory = tools.inspect_run(run)
    native = tools.read_json(run / "presents.json")
    scene = tools.read_json(run / "scene.json")
    inputs = tools.read_json(run / "application_inputs/manifest.json")
    if scene["schema"] != "xefg.render_scene/1" or native["schema"] != "xefg.present_capture/2":
        raise ValueError("Rendered scene pairing requires scene v1 and present v2")
    width, height = _input_dimensions(inputs)
    mode = _mode_from_metadata(scene, inventory)
    provider = inventory.get("provider", {}).get("xefg_dll_sha256")
    tag_layout(provider, (width, height))
    if not native.get("tag_interpolated_requested") or native.get("tag_interpolated_result") != 0:
        raise ValueError("Rendered pairing requires the official generated-frame tag")
    if mode == 1:
        return _write_isolated_pairing(run, output, tools, inventory, native, scene, inputs,
                                       width, height, provider)
    return _write_continuous_pairing(run, output, tools, inventory, native, scene, inputs, mode,
                                     width, height, provider)


def _write_continuous_pairing(run, output, tools, inventory, native, scene, inputs, mode,
                              width, height, provider):
    """Mode 3/4 sequence pairing: marker identity plus official tags."""
    layout = tag_layout(provider, (width, height))
    shape = (height, width)
    marker = scene.get("marker")
    if not isinstance(marker, dict):
        raise ValueError("Mode 3/4 rendered pairing requires the UI frame marker")
    marker_rect = [marker["x"], marker["y"], marker["x"] + marker["bits"] * marker["cell_width"],
                   marker["y"] + marker["cell_height"]]
    rects = [*layout, marker_rect]
    mask = mask_rects(shape, rects)
    marker_mask = np.zeros(shape, dtype=bool)
    marker_mask[marker_rect[1]:marker_rect[3], marker_rect[0]:marker_rect[2]] = True
    if np.any(marker_mask & ~mask_rects(shape, layout)):
        raise ValueError("Frame marker overlaps the official tag layout")
    helper = tools.load_tool("pairing")
    frames = {row["index"]: row for row in scene["frames"]}
    if sorted(frames) != list(range(inputs["frame_count"])):
        raise ValueError("Scene submissions must cover all input frames")
    for record in frames.values():
        _validate_submissions(record, mode, f"Scene frame {record['index']}")
    statuses = {row["application_frame"]: row for row in native["frame_status"]}
    for index, record in frames.items():
        if record["previous_frame"] != (index - 1 if index else None):
            raise ValueError("Scene submission history is not contiguous")
    artifacts = [*inventory["artifacts"], tools.artifact(run, "scene.json")]
    for row in frames.values():
        artifacts.append(tools.artifact(run, row["midpoint"], width * height * 4))
    hashes = {row["file"]: row["sha256"] for row in artifacts}
    rows = []
    for present in native["presents"]:
        if present["format"] != 87:
            raise ValueError("Rendered pairing expects BGRA8_UNORM")
        pixels = helper.image(run / present["file"], width, height)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != hashes[present["file"]]:
            raise ValueError("Present changed after inventory")
        frame, levels = helper.decode_marker(pixels, marker)
        record, status = frames.get(frame), statuses.get(frame)
        tagged, fractions = tag_evidence(pixels, provider)
        base_mae = None
        if record is not None:
            base_name = "application_inputs/" + inputs["frames"][frame]["inputs"]["backbuffer"]
            base = helper.image(run / base_name, width, height)
            if base_name in hashes and hashlib.sha256(base.tobytes()).hexdigest() != hashes[base_name]:
                raise ValueError("Base image changed after inventory")
            base_mae = float(np.abs(pixels[mask, :3].astype(np.int16) - base[mask, :3]).mean())
        classification, reason = classify(present, frame, record, status, tagged, base_mae)
        if frame is not None and frame < inventory["configuration"]["warmup"]:
            classification, reason = "ambiguous", "captured warmup content"
        row = {"sequence": present["sequence"], "file": present["file"], "sha256": hashes[present["file"]],
               "mode": mode, "current_frame": frame, "previous_frame": record["previous_frame"] if record else None,
               "source_current_frame": frame,
               "source_previous_frame": record["previous_frame"] if record else None,
               "source_pair": ([record["previous_frame"], frame] if record is not None else None),
               "current_time": _frame_time(record) if record else None,
               "previous_time": (_frame_time(frames[record["previous_frame"]])
                                 if record and record["previous_frame"] in frames else None),
               "decoded_ui_frame": frame, "observed_application_frame": present["observed_application_frame"],
               "ordinal": present.get("ordinal"), "classification": classification, "reason": reason,
               "marker_levels": levels, "official_tag_detected": tagged, "tag_region_fractions": fractions,
               "base_mae_without_overlays": base_mae, "official_status": status}
        rows.append(row)
    generated = collections.Counter(r["current_frame"] for r in rows if r["classification"] == "generated")
    for row in rows:
        if row["classification"] == "generated" and generated[row["current_frame"]] != 1:
            row.update(classification="ambiguous", reason="duplicate generated output")
    warmup = inventory["configuration"]["warmup"]
    expected = {i for i, f in frames.items() if i >= warmup and not f["reset_history"]}
    paired = {r["current_frame"] for r in rows if r["classification"] == "generated"}
    missing = sorted(expected - paired)
    counts = dict(collections.Counter(r["classification"] for r in rows))
    report = {"schema": "xefg.pairing/2", "method": "rendered_scene", "mode": mode,
              "ui_mode": inventory["configuration"].get("ui_mode", mode),
              "capture_directory": str(run),
              "status": "paired" if not missing and not counts.get("ambiguous") else "incomplete",
              "counts": counts, "missing_generated_frames": missing, "official_tags_requested": True,
              "marker": marker, "excluded_rects": rects, "valid_pixel_count": int(mask.sum()),
              "output_size": [width, height],
              "motion_size": [inputs["motion_width"], inputs["motion_height"]],
              "mask_provider_sha256": provider, "artifacts": artifacts, "pairs": rows,
              "crosshair_policy": "dynamic UI/crosshair pixels remain in the valid comparison region",
              "reference_usage": "native midpoint is quality-reference-only, never an identity candidate"}
    _write_report(output, report, rows, "Rendered scene pairing")
    return report


def _write_isolated_pairing(run, output, tools, inventory, native, scene, inputs,
                            width, height, provider):
    """Mode 1: one fresh two-frame context; the provider tag is the only identity."""
    layout = tag_layout(provider, (width, height))
    shape = (height, width)
    if scene.get("pairing_method") != "isolated_pair":
        raise ValueError("Mode 1 pairing requires pairing_method='isolated_pair'")
    source_pair = scene.get("source_pair")
    if (not isinstance(source_pair, list) or len(source_pair) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in source_pair)
            or source_pair[0] < 0 or source_pair[1] != source_pair[0] + 1):
        raise ValueError("Mode 1 pairing requires a contiguous source_pair")
    configuration = inventory.get("configuration", {})
    if configuration.get("ui_mode") != 1:
        raise ValueError("Mode 1 pairing requires captured ui_mode 1")
    if configuration.get("pair_frame") != source_pair[1]:
        raise ValueError("Mode 1 pair_frame differs from the recorded source pair")
    if configuration.get("warmup", 0) != 0:
        raise ValueError("Mode 1 isolated pair must capture both local frames")
    if inputs["frame_count"] != 2 or len(inputs["frames"]) != 2:
        raise ValueError("Mode 1 pairing requires exactly two input frames")
    if [row["index"] for row in inputs["frames"]] != [0, 1]:
        raise ValueError("Mode 1 pairing requires local indices 0 and 1")
    frames = {}
    for row in scene["frames"]:
        if row.get("index") in frames:
            raise ValueError("Mode 1 scene submissions contain a duplicate local index")
        frames[row["index"]] = row
    if len(scene["frames"]) != 2 or set(frames) != {0, 1}:
        raise ValueError("Mode 1 pairing requires exactly two scene submissions")
    if frames[0].get("previous_frame") is not None or not frames[0].get("reset_history"):
        raise ValueError("Mode 1 first local submission must start a reset context")
    if frames[1].get("previous_frame") != 0 or frames[1].get("reset_history"):
        raise ValueError("Mode 1 second local submission must continue without a reset")
    for index in (0, 1):
        _validate_submissions(frames[index], 1, f"Mode 1 local frame {index}")
    eligible = {(0, 1)}
    observed = {(frames[1]["previous_frame"], 1)} if not frames[1]["reset_history"] else set()
    if observed != eligible:
        raise ValueError("Mode 1 metadata does not describe the unique eligible pair (0, 1)")
    source_times = [_frame_time(frames[0]), _frame_time(frames[1])]
    if any(value is None for value in source_times):
        raise ValueError("Mode 1 scene submissions must report source times")
    if source_times != [float(value) for value in source_pair]:
        raise ValueError("Mode 1 source times differ exactly from source_pair")
    statuses = {}
    for row in native["frame_status"]:
        if row.get("application_frame") in statuses:
            raise ValueError("Mode 1 native frame statuses contain a duplicate local frame")
        statuses[row["application_frame"]] = row
    if len(native["frame_status"]) != 2 or set(statuses) != {0, 1}:
        raise ValueError("Mode 1 pairing requires exactly two native frame statuses")
    if not _status_reset(statuses[0]):
        raise ValueError("Mode 1 local reset status is not the known reset/base case")
    if not _status_success(statuses[1]):
        raise ValueError("Mode 1 local generated status is not successful")
    artifacts = [*inventory["artifacts"], tools.artifact(run, "scene.json")]
    for record in frames.values():
        midpoint = record.get("midpoint")
        if not midpoint:
            raise ValueError("Mode 1 pairing requires a native midpoint for every local frame")
        artifacts.append(tools.artifact(run, midpoint, width * height * 4))
    hashes = {row["file"]: row["sha256"] for row in artifacts}
    helper = tools.load_tool("pairing")
    bases = {}
    for index in (0, 1):
        name = "application_inputs/" + inputs["frames"][index]["inputs"]["backbuffer"]
        bases[index] = helper.image(run / name, width, height)
        if name in hashes and hashlib.sha256(bases[index].tobytes()).hexdigest() != hashes[name]:
            raise ValueError("Base image changed after inventory")
    rows = []
    by_sequence = {}
    for present in native["presents"]:
        if present["format"] != 87:
            raise ValueError("Mode 1 pairing expects BGRA8_UNORM")
        pixels = helper.image(run / present["file"], width, height)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != hashes[present["file"]]:
            raise ValueError("Present changed after inventory")
        tagged, fractions = tag_evidence(pixels, provider)
        matched = None
        if not tagged:
            for index in (0, 1):
                if np.array_equal(pixels, bases[index]):
                    matched = index
                    break
        row = {"sequence": present["sequence"], "file": present["file"],
               "sha256": hashes[present["file"]], "mode": 1,
               "observed_application_frame": present.get("observed_application_frame"),
               "ordinal": present.get("ordinal"), "decoded_ui_frame": None,
               "official_tag_detected": tagged, "tag_region_fractions": fractions,
               "matched_base_frame": matched, "native_present_result": present.get("native_present_result", 0),
               "observed_label_used_for_identity": False,
               "output_size": [width, height],
               "motion_size": [inputs["motion_width"], inputs["motion_height"]]}
        rows.append(row)
        by_sequence[present["sequence"]] = present
    tagged_rows = [row for row in rows if row["official_tag_detected"]]
    for row in rows:
        present = by_sequence[row["sequence"]]
        row["official_status"] = statuses.get(row["matched_base_frame"]) if row["matched_base_frame"] is not None else None
        if row["official_tag_detected"]:
            status = statuses[1]
            success = (_status_success(status)
                       and present.get("native_present_result", 0) == 0)
            if len(tagged_rows) > 1:
                row.update(classification="ambiguous",
                           reason="multiple official-tagged outputs cannot be assigned to the unique pair")
            elif not success:
                row.update(classification="ambiguous",
                           reason="official tag cannot be paired without successful local frame 1 status")
            else:
                row.update(classification="generated",
                           reason="unique official tag assigned to the only eligible source pair",
                           previous_frame=0, current_frame=1,
                           local_previous_frame=0, local_current_frame=1,
                           source_previous_frame=source_pair[0], source_current_frame=source_pair[1],
                           source_pair=list(source_pair), previous_time=source_times[0],
                           current_time=source_times[1], official_status=status)
        elif row["matched_base_frame"] is not None:
            local = row["matched_base_frame"]
            status = statuses[local]
            if status.get("present_result") != 0 or status.get("status_result") != 0:
                row.update(classification="ambiguous", reason="untagged base output failed Present/status")
            else:
                record = frames[local]
                row.update(classification="fallback" if record["reset_history"] else "base",
                           reason=f"exact match to local frame {local} submitted backbuffer",
                           previous_frame=record["previous_frame"], current_frame=local,
                           local_previous_frame=record["previous_frame"], local_current_frame=local,
                           source_previous_frame=(source_pair[local - 1] if local else None),
                           source_current_frame=source_pair[local],
                           source_pair=[source_pair[local - 1], source_pair[local]] if local else None,
                           previous_time=(source_times[local - 1] if local else None),
                           current_time=source_times[local])
        else:
            row.update(classification="ambiguous",
                       reason="untagged output does not match any submitted backbuffer; retained, not certified")
        if present.get("native_present_result", 0) != 0:
            row.update(classification="ambiguous", reason="native Present failed for this output")
    counts = dict(collections.Counter(row["classification"] for row in rows))
    generated = [row for row in rows if row["classification"] == "generated"]
    ambiguous = counts.get("ambiguous", 0)
    missing = [] if len(generated) == 1 and not ambiguous else [1]
    report = {"schema": "xefg.pairing/3", "method": "isolated_pair",
              "pairing_method": "isolated_pair", "mode": 1, "ui_mode": 1,
              "capture_directory": str(run),
              "status": "paired" if not missing and not ambiguous else "incomplete",
              "counts": counts, "missing_generated_frames": missing,
              "missing_generated_source_frames": ([] if not missing else [source_pair[1]]),
              "tagged_present_count": len(tagged_rows),
              "official_tags_requested": True,
              "source_pair": list(source_pair), "source_times": source_times,
              "local_frames": [0, 1],
              "eligible_pairs": [{"local_previous_frame": 0, "local_current_frame": 1,
                                  "source_previous_frame": source_pair[0],
                                  "source_current_frame": source_pair[1],
                                  "previous_time": source_times[0],
                                  "current_time": source_times[1]}],
              "excluded_rects": layout, "valid_pixel_count": int(mask_rects(shape, layout).sum()),
              "output_size": [width, height],
              "motion_size": [inputs["motion_width"], inputs["motion_height"]],
              "mask_provider_sha256": provider, "artifacts": artifacts, "pairs": rows,
              "identity_rule": "only the official tag identifies the sole eligible local pair; "
                               "observed labels, ordinals and SDK similarity are not used",
              "marker_used": False,
              "reference_usage": "native midpoint is quality-reference-only, never an identity candidate"}
    _write_report(output, report, rows, "Rendered isolated-pair pairing")
    return report
