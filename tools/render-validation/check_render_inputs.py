"""Summarize GPU-rendered inputs without loading the FG runtime or a model."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


def _load_scene_evidence():
    spec = importlib.util.spec_from_file_location(
        "scene_evidence", Path(__file__).with_name("scene_evidence.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scene_evidence = _load_scene_evidence()

SCHEMA = "xefg.render_input_check/2"
LEGACY_SCHEMA = "xefg.render_input_check/1"
COVERAGE_SCHEMA = "xefg.render_input_coverage/1"
EVIDENCE_SCENES = scene_evidence.NEW_SCENES
COVERAGE_SCENES = ("depth_layers", "screen_cross", "camera_turn", *EVIDENCE_SCENES)
QUANTILE_EDGES = (0.0, 0.5, 0.9, 0.97, 0.995, 1.0)
DEPTH_HISTOGRAM_EDGES = (0.0, 0.5, 0.9, 0.97, 0.995, 1.0)
# The final edge is +infinity. JSON cannot represent that portably, so the
# serialized edge list uses the string "infinity" and still has 6 boundaries.
NORMALIZED_MAGNITUDE_EDGES = (0.0, 0.0001, 0.005, 0.02, 0.1, math.inf)
SIGN_EPSILON_PIXELS = 0.01
# Edge/moving geometry ignores sub-epsilon floating-point jitter. The legacy
# moving_pixels field remains exact (!= 0) for backwards compatibility.
MOTION_EPSILON_PIXELS = 0.01
BACKGROUND_DEPTH = 1.0

COVERAGE_CRITERIA = {
    "depth_layers": {
        "near_depth_max": 0.9,
        "far_depth_min": 0.99,
        "foreground_fraction_min": 0.01,
        "min_measured_source_frames": 1,
    },
    "screen_cross": {
        "sign_epsilon_pixels": SIGN_EPSILON_PIXELS,
        "sign_fraction_min": 0.01,
        "normalized_magnitude_threshold": 0.02,
        "normalized_magnitude_fraction_min": 0.01,
        "large_normalized_magnitude_threshold": 0.08,
        "large_normalized_magnitude_fraction_min": 0.01,
        "edge_moving_geometry_fraction_min": 0.01,
        "edge_moving_fraction_range_min": 0.01,
        "min_measured_source_frames": 2,
    },
    "camera_turn": {
        "near_depth_max": 0.97,
        "far_depth_min": 0.99,
        "broad_normalized_magnitude_threshold": 0.005,
        "broad_foreground_fraction_min": 0.30,
        "near_far_mean_delta_min": 0.001,
        "min_measured_source_frames": 2,
    },
    "nonrigid": dict(scene_evidence.DEFAULT_NONRIGID_CRITERIA),
    "transparency": dict(scene_evidence.DEFAULT_TRANSPARENCY_CRITERIA),
}


def _fraction(count, total):
    return float(count) / float(total) if total else 0.0


def _mean(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else 0.0


def _quantiles(values, edges):
    return [float(np.quantile(values, edge)) for edge in edges]


def _histogram(values, edges):
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=np.float64))
    counts = [int(value) for value in counts]
    total = int(sum(counts))
    return {"edges": list(edges), "counts": counts,
            "fractions": [_fraction(count, total) for count in counts]}


def _json_edges(edges):
    return [("infinity" if math.isinf(edge) else float(edge)) for edge in edges]


def _sign_fractions(values, epsilon=SIGN_EPSILON_PIXELS):
    total = int(values.size)
    return {"positive": _fraction(np.count_nonzero(values > epsilon), total),
            "negative": _fraction(np.count_nonzero(values < -epsilon), total),
            "zero": _fraction(np.count_nonzero(np.abs(values) <= epsilon), total)}


def _edge_occupancy(mask):
    border = np.zeros(mask.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return {"band": "one_pixel_border", "border_width": 1,
            "band_pixels": int(np.count_nonzero(border)),
            "left": _mean(mask[:, 0]), "right": _mean(mask[:, -1]),
            "top": _mean(mask[0, :]), "bottom": _mean(mask[-1, :]),
            "any": _mean(mask[border])}


def _foreground_depth(depth, foreground):
    values = depth[foreground]
    return {"pixels": int(values.size), "fraction": _fraction(values.size, depth.size),
            "min": float(values.min()), "max": float(values.max()),
            "quantile_edges": list(QUANTILE_EDGES), "quantiles": _quantiles(values, QUANTILE_EDGES),
            "below_0.9_pixels": int(np.count_nonzero(values < 0.9)),
            "above_0.99_pixels": int(np.count_nonzero(values > 0.99)),
            "histogram": _histogram(values, DEPTH_HISTOGRAM_EDGES)}


def _motion_summary(depth, motion, motion_width, motion_height):
    mx = motion[:, 0]
    my = motion[:, 1]
    shape = (int(motion_height), int(motion_width))
    magnitude_pixels = np.hypot(mx, my)
    normalized = np.hypot(mx / float(motion_width), my / float(motion_height))
    foreground = depth < BACKGROUND_DEPTH
    foreground_flat = foreground.reshape(-1)
    foreground_normalized = normalized[foreground_flat]
    near = foreground & (depth < 0.97)
    far = foreground & (depth > 0.99)
    near_values = normalized[near.reshape(-1)]
    far_values = normalized[far.reshape(-1)]
    moving = magnitude_pixels > MOTION_EPSILON_PIXELS
    moving = moving.reshape(shape)
    moving_geometry = foreground & moving
    x_clip = np.abs(mx) >= motion_width
    y_clip = np.abs(my) >= motion_height
    histogram = _histogram(normalized, NORMALIZED_MAGNITUDE_EDGES)
    histogram["edges"] = _json_edges(NORMALIZED_MAGNITUDE_EDGES)
    above_002_count = np.count_nonzero(normalized > 0.02)
    above_005_count = np.count_nonzero(normalized > 0.005)
    above_008_count = np.count_nonzero(normalized > 0.08)
    foreground_above_005 = np.count_nonzero(foreground_normalized > 0.005)
    foreground_above_002 = np.count_nonzero(foreground_normalized > 0.02)
    foreground_above_008 = np.count_nonzero(foreground_normalized > 0.08)
    return {
        "motion_magnitude_pixels": {
            "min": float(magnitude_pixels.min()), "max": float(magnitude_pixels.max()),
            "quantile_edges": list(QUANTILE_EDGES),
            "quantiles": _quantiles(magnitude_pixels, QUANTILE_EDGES)},
        "motion_normalized": {
            "vector_scale": [int(motion_width), int(motion_height)],
            "histogram": histogram,
            "above_0.005_pixels": int(above_005_count),
            "above_0.005_fraction": _fraction(above_005_count, normalized.size),
            "above_0.02_pixels": int(above_002_count),
            "above_0.02_fraction": _fraction(above_002_count, normalized.size),
            "above_0.08_pixels": int(above_008_count),
            "above_0.08_fraction": _fraction(above_008_count, normalized.size)},
        "motion_sign": {"epsilon_pixels": SIGN_EPSILON_PIXELS,
                        "x": _sign_fractions(mx), "y": _sign_fractions(my)},
        "clip_limit": {
            "threshold": [int(motion_width), int(motion_height)],
            "x_pixels": int(np.count_nonzero(x_clip)), "y_pixels": int(np.count_nonzero(y_clip)),
            "any_pixels": int(np.count_nonzero(x_clip | y_clip)),
            "x_fraction": _fraction(np.count_nonzero(x_clip), mx.size),
            "y_fraction": _fraction(np.count_nonzero(y_clip), my.size),
            "any_fraction": _fraction(np.count_nonzero(x_clip | y_clip), mx.size)},
        "foreground_motion": {
            "pixels": int(foreground_normalized.size),
            "above_0.005_pixels": int(foreground_above_005),
            "above_0.005_fraction": _fraction(foreground_above_005, foreground_normalized.size),
            "above_0.02_pixels": int(foreground_above_002),
            "above_0.02_fraction": _fraction(foreground_above_002, foreground_normalized.size),
            "above_0.08_pixels": int(foreground_above_008),
            "above_0.08_fraction": _fraction(foreground_above_008, foreground_normalized.size)},
        "depth_bands": {
            "near": {"depth_below": 0.97, "pixels": int(near_values.size),
                     "normalized_magnitude_mean": (_mean(near_values) if near_values.size else None)},
            "far": {"depth_above": 0.99, "pixels": int(far_values.size),
                    "normalized_magnitude_mean": (_mean(far_values) if far_values.size else None)}},
        "edge_occupancy": {
            "motion_epsilon_pixels": MOTION_EPSILON_PIXELS,
            "geometry": _edge_occupancy(foreground),
            "moving": _edge_occupancy(moving),
            "moving_geometry": _edge_occupancy(moving_geometry)},
    }


def _aggregate_depth(depth):
    foreground = depth < BACKGROUND_DEPTH
    background_pixels = int(depth.size - np.count_nonzero(foreground))
    return foreground, {"value": BACKGROUND_DEPTH, "pixels": background_pixels,
                        "fraction": _fraction(background_pixels, depth.size)}


def _legacy_row(frame, depth, motion):
    return {"frame": frame["index"], "depth_min": float(depth.min()), "depth_max": float(depth.max()),
            "geometry_pixels": int(np.count_nonzero(depth < BACKGROUND_DEPTH)),
            "motion_min_xy": motion.min(axis=0).tolist(), "motion_max_xy": motion.max(axis=0).tolist(),
            "moving_pixels": int(np.count_nonzero(np.any(motion != 0, axis=1)))}


def aggregate_coverage(scene_name, frame_rows):
    """Evaluate preset input coverage from measured per-frame rows.

    ``frame_rows`` is the concatenation of report ``frames`` arrays. Rows with
    ``measured`` false (warmup, reset, or an isolated-pair local base frame) are
    ignored. Result status is one of pass, fail, not_applicable, pending.
    """
    scene = str(scene_name)
    if scene not in COVERAGE_SCENES:
        return {"schema": COVERAGE_SCHEMA, "scene": scene, "status": "not_applicable",
                "criteria": {}, "measured_frames": [], "measured_frame_count": 0,
                "observed": {}, "checks": [],
                "reason": "no preset coverage gate for this scene"}
    criteria = dict(COVERAGE_CRITERIA[scene])
    measured = [row for row in frame_rows if bool(row.get("measured"))]
    source_frames = sorted({int(row.get("source_frame", row.get("frame", -1))) for row in measured})
    common = {"schema": COVERAGE_SCHEMA, "scene": scene, "criteria": criteria,
              "measured_frames": source_frames, "measured_frame_count": len(measured)}
    if not measured:
        return {**common, "status": "pending", "observed": {}, "checks": [],
                "reason": "no measured frames; coverage requires measured input records"}
    isolated = all(bool(row.get("isolated")) for row in measured)
    if scene in EVIDENCE_SCENES:
        evidence = [row.get("scene_evidence") for row in measured]
        missing = [item for item in evidence
                   if not isinstance(item, dict) or item.get("kind") != scene]
        if missing:
            return {**common, "status": "fail",
                    "observed": {"missing_evidence_rows": len(missing)},
                    "checks": [_check("metadata_present", False,
                                      "scene_evidence for every measured frame")],
                    "reason": "missing scene evidence for a measured %s frame" % scene}
    if isolated and len(source_frames) < 2:
        return {**common, "status": "pending", "observed": {"isolated": True},
                "checks": [], "reason": "isolated pair needs aggregate across source times"}
    minimum = int(criteria["min_measured_source_frames"])
    if len(source_frames) < minimum:
        return {**common, "status": "pending",
                "observed": {"source_frames": source_frames, "required": minimum},
                "checks": [], "reason": "full sequence is required for this coverage preset"}
    if scene in EVIDENCE_SCENES:
        return _aggregate_evidence(scene, measured, criteria, common)
    checks = []
    try:
        if scene == "depth_layers":
            observed = _depth_layers_observed(measured)
            checks = [
                _check("near_foreground_fraction", observed["near_fraction_ok"],
                       criteria["foreground_fraction_min"]),
                _check("far_foreground_fraction", observed["far_fraction_ok"],
                       criteria["foreground_fraction_min"]),
            ]
        elif scene == "screen_cross":
            observed = _screen_cross_observed(measured)
            checks = [
                _check("bidirectional_x_sign", observed["sign_ok"],
                       "positive and negative x fractions >= %.3f" % criteria["sign_fraction_min"]),
                _check("normalized_magnitude_above_0.02", observed["magnitude_ok"],
                       "some frame fraction >= %.3f" % criteria["normalized_magnitude_fraction_min"]),
                _check("large_normalized_magnitude_above_0.08", observed["large_magnitude_ok"],
                       "some frame fraction >= %.3f"
                       % criteria["large_normalized_magnitude_fraction_min"]),
                _check("moving_geometry_at_both_edges", observed["moving_geometry_ok"],
                       "left and right frame fractions each >= %.3f"
                       % criteria["edge_moving_geometry_fraction_min"]),
                _check("edge_moving_occupancy_varies", observed["edge_moving_ok"],
                       "range >= %.3f" % criteria["edge_moving_fraction_range_min"]),
            ]
        else:
            observed = _camera_turn_observed(measured)
            checks = [
                _check("broad_foreground_motion", observed["broad_ok"],
                       "some frame fraction >= %.3f" % criteria["broad_foreground_fraction_min"]),
                _check("separated_near_far_bands", observed["separated_ok"],
                       "near depth < %.2f and far depth > %.2f both nonempty"
                       % (criteria["near_depth_max"], criteria["far_depth_min"])),
                _check("near_far_differing_means", observed["differing_ok"],
                       "absolute mean difference >= %.4f" % criteria["near_far_mean_delta_min"]),
            ]
    except (KeyError, TypeError, ValueError) as error:
        return {**common, "status": "pending", "observed": {}, "checks": [],
                "reason": "coverage fields missing: %s" % error}
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {**common, "status": status, "observed": observed, "checks": checks,
            "reason": None if status == "pass" else "one or more preset coverage checks failed"}


def _check(name, passed, threshold):
    return {"name": name, "status": "pass" if passed else "fail",
            "observed": bool(passed), "threshold": threshold}


def _aggregate_evidence(scene, measured, criteria, common):
    evidence = [row.get("scene_evidence") for row in measured]
    if any(not isinstance(item, dict) or item.get("kind") != scene for item in evidence):
        return {**common, "status": "fail",
                "observed": {"missing_evidence_rows": sum(
                    1 for item in evidence
                    if not isinstance(item, dict) or item.get("kind") != scene)},
                "checks": [_check("metadata_present", False,
                                  "scene_evidence for every measured frame")],
                "reason": "missing scene evidence for a measured %s frame" % scene}
    failed = [item for item in evidence if item.get("status") != "pass"]
    if failed:
        checks = []
        seen = set()
        for item in failed:
            for check in item.get("checks", []):
                key = (check.get("name"), check.get("status"))
                if key not in seen:
                    seen.add(key)
                    checks.append(check)
        return {**common, "status": "fail",
                "observed": {"failed_rows": len(failed),
                             "reasons": [item.get("reason") for item in failed]},
                "checks": checks,
                "reason": "one or more measured frames failed scene evidence"}
    if scene == "nonrigid":
        return _aggregate_nonrigid(measured, evidence, criteria, common)
    return _aggregate_transparency(measured, evidence, criteria, common)


def _matrices_close(left, right, tolerance):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return left.shape == (4, 4) and right.shape == (4, 4) and bool(
        np.all(np.abs(left - right) <= tolerance))


def _aggregate_nonrigid(measured, evidence, criteria, common):
    tolerance = float(criteria["static_matrix_tolerance"])
    matrices = [item["matrices"] for item in evidence]
    static_ok = True
    for key in ("world", "view", "projection"):
        reference = matrices[0][key]
        static_ok = static_ok and all(
            _matrices_close(reference, current[key], tolerance) for current in matrices[1:])
    for item in matrices:
        for current_key, previous_key in (("world", "previous_world"),
                                          ("view", "previous_view"),
                                          ("projection", "previous_projection")):
            static_ok = static_ok and _matrices_close(
                item[current_key], item[previous_key], tolerance)

    values = {}
    for item in evidence:
        for pair in item.get("pair_distances", []):
            values.setdefault((pair["a"], pair["b"]), []).append(float(pair["current"]))
    changed_pairs = 0
    for samples in values.values():
        if len(samples) < 2:
            continue
        delta = max(samples) - min(samples)
        relative = delta / max(min(abs(value) for value in samples), 1.0e-9)
        if delta > 1.0e-9 and relative >= criteria["vertex_pair_change_fraction_min"]:
            changed_pairs += 1
    pairs_ok = changed_pairs >= int(criteria["min_changed_pairs"])

    total_samples = sum(int(item.get("sample_count", 0)) for item in evidence)
    max_triangles = max((int(item.get("sampled_triangle_count", 0)) for item in evidence),
                        default=0)
    samples_ok = (total_samples >= int(criteria["min_triangle_samples"])
                  and max_triangles >= int(criteria["min_sampled_triangles"]))
    spatial_values = [float(item.get("mv_spatial_range", 0.0)) for item in evidence]
    spatial_ok = any(value >= criteria["mv_spatial_range_pixels_min"]
                     for value in spatial_values)
    checks = [
        _check("static_world_view", static_ok,
               "world/view/projection constant and previous==current within %.0e" % tolerance),
        _check("vertex_pair_distances_change", pairs_ok,
               ">= %d sampled vertex pairs change by >= %.1f%%"
               % (criteria["min_changed_pairs"],
                  100.0 * criteria["vertex_pair_change_fraction_min"])),
        _check("sampled_triangle_interiors", samples_ok,
               ">= %d samples and >= %d sampled triangles"
               % (criteria["min_triangle_samples"], criteria["min_sampled_triangles"])),
        _check("mv_spatially_distinct", spatial_ok,
               ">= %.3f px analytic motion spread"
               % criteria["mv_spatial_range_pixels_min"]),
    ]
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {**common, "status": status,
            "observed": {"changed_pairs": changed_pairs, "total_samples": total_samples,
                         "max_sampled_triangles": max_triangles,
                         "mv_spatial_ranges": spatial_values,
                         "max_motion_error": max(
                             (float(item.get("max_motion_error", 0.0)) for item in evidence),
                             default=0.0),
                         "max_depth_error": max(
                             (float(item.get("max_depth_error", 0.0)) for item in evidence),
                             default=0.0)},
            "checks": checks,
            "reason": None if status == "pass" else
            "nonrigid geometry/deformation coverage checks failed"}


def _aggregate_transparency(measured, evidence, criteria, common):
    checks = []
    observed = {}
    for kind in ("single", "double", "occluded"):
        samples = 0
        area = 0.0
        mismatches = 0
        for item in evidence:
            region = item.get("regions", {}).get(kind, {})
            samples += int(region.get("sample_count", 0))
            area = max(area, float(region.get("area_fraction", 0.0)))
            mismatches += (int(region.get("color_mismatches", 0))
                           + int(region.get("depth_mismatches", 0))
                           + int(region.get("motion_mismatches", 0)))
        name = "region_" + kind + "_layer" if kind != "occluded" else "region_occluded"
        check = _check(name,
                       samples >= int(criteria["min_samples_per_region"])
                       and area >= float(criteria["region_area_fraction_min"])
                       and mismatches == 0,
                       ">= %d interior samples, area >= %.3f, no mismatch"
                       % (criteria["min_samples_per_region"],
                          criteria["region_area_fraction_min"]))
        checks.append(check)
        observed[kind] = {"samples": samples, "max_area_fraction": area,
                          "mismatches": mismatches}

    centers = {}
    for item in evidence:
        for name, center in item.get("transparent_centers", {}).items():
            centers.setdefault(name, []).append(np.asarray(center, dtype=np.float64))
    movement = 0.0
    moving_rows = 0
    for points in centers.values():
        if len(points) < 2:
            continue
        moving_rows = max(moving_rows, len(points))
        array = np.asarray(points)
        differences = array[:, None, :] - array[None, :, :]
        movement = max(movement, float(np.max(np.linalg.norm(differences, axis=2))))
    temporal_ok = (moving_rows >= int(criteria["min_temporal_rows"])
                   and movement >= float(criteria["temporal_motion_pixels_min"]))
    checks.append(_check("temporal_movement", temporal_ok,
                         ">= %d rows and >= %.2f px panel movement"
                         % (criteria["min_temporal_rows"],
                            criteria["temporal_motion_pixels_min"])))
    observed["temporal_movement_pixels"] = movement
    observed["temporal_rows"] = moving_rows
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {**common, "status": status, "observed": observed, "checks": checks,
            "reason": None if status == "pass" else
            "transparency composition/coverage checks failed"}


def _weighted_fraction(rows, getter):
    total_pixels = 0
    total_hits = 0.0
    for row in rows:
        weight = int(row.get("pixels", 0)) or 1
        total_pixels += weight
        total_hits += float(getter(row)) * weight
    return _fraction(total_hits, total_pixels)


def _depth_layers_observed(rows):
    criteria = COVERAGE_CRITERIA["depth_layers"]
    foreground = near = far = 0
    for row in rows:
        depth = row["foreground_depth"]
        foreground += int(depth["pixels"])
        near += int(depth["below_0.9_pixels"])
        far += int(depth["above_0.99_pixels"])
    near_fraction = _fraction(near, foreground)
    far_fraction = _fraction(far, foreground)
    return {"foreground_pixels": foreground, "pixels": foreground,
            "near_pixels": near, "near_fraction": near_fraction,
            "far_pixels": far, "far_fraction": far_fraction,
            "near_fraction_ok": near_fraction >= criteria["foreground_fraction_min"],
            "far_fraction_ok": far_fraction >= criteria["foreground_fraction_min"]}


def _screen_cross_observed(rows):
    criteria = COVERAGE_CRITERIA["screen_cross"]
    positive = _weighted_fraction(rows, lambda row: row["motion_sign"]["x"]["positive"])
    negative = _weighted_fraction(rows, lambda row: row["motion_sign"]["x"]["negative"])
    positive_frames = []
    negative_frames = []
    for row in rows:
        source_frame = int(row.get("source_frame", row.get("frame", -1)))
        yes = float(row["motion_sign"]["x"]["positive"])
        no = float(row["motion_sign"]["x"]["negative"])
        if yes >= criteria["sign_fraction_min"] and yes > no:
            positive_frames.append(source_frame)
        if no >= criteria["sign_fraction_min"] and no > yes:
            negative_frames.append(source_frame)
    sign_ok = (bool(positive_frames) and bool(negative_frames)
               and any(left != right for left in positive_frames for right in negative_frames))
    magnitudes = [float(row["motion_normalized"]["above_0.02_fraction"]) for row in rows]
    large = [float(row["motion_normalized"]["above_0.08_fraction"]) for row in rows]
    magnitude_ok = any(value >= criteria["normalized_magnitude_fraction_min"] for value in magnitudes)
    large_magnitude_ok = any(value >= criteria["large_normalized_magnitude_fraction_min"]
                             for value in large)
    geometry = [float(row["edge_occupancy"]["moving_geometry"]["any"]) for row in rows]
    geometry_left = [float(row["edge_occupancy"]["moving_geometry"]["left"]) for row in rows]
    geometry_right = [float(row["edge_occupancy"]["moving_geometry"]["right"]) for row in rows]
    moving = [float(row["edge_occupancy"]["moving"]["any"]) for row in rows]
    moving_geometry_ok = (
        any(value >= criteria["edge_moving_geometry_fraction_min"] for value in geometry_left)
        and any(value >= criteria["edge_moving_geometry_fraction_min"] for value in geometry_right))
    edge_moving_range = max(moving) - min(moving)
    edge_moving_ok = edge_moving_range >= criteria["edge_moving_fraction_range_min"]
    return {"sign_ok": sign_ok, "positive_x_fraction": positive,
            "negative_x_fraction": negative,
            "positive_dominant_source_frames": positive_frames,
            "negative_dominant_source_frames": negative_frames,
            "magnitude_ok": magnitude_ok, "above_0.02_fractions": magnitudes,
            "large_magnitude_ok": large_magnitude_ok, "above_0.08_fractions": large,
            "moving_geometry_ok": moving_geometry_ok,
            "moving_geometry_edge_fractions": geometry,
            "moving_geometry_left_fractions": geometry_left,
            "moving_geometry_right_fractions": geometry_right,
            "edge_moving_ok": edge_moving_ok, "edge_moving_fractions": moving,
            "edge_moving_range": edge_moving_range}


def _camera_turn_observed(rows):
    criteria = COVERAGE_CRITERIA["camera_turn"]
    broad = [float(row["foreground_motion"]["above_0.005_fraction"]) for row in rows]
    broad_ok = any(value >= criteria["broad_foreground_fraction_min"] for value in broad)
    near_pixels = far_pixels = 0
    near_sum = far_sum = 0.0
    for row in rows:
        near = row["depth_bands"]["near"]
        far = row["depth_bands"]["far"]
        if int(near["pixels"]) and near["normalized_magnitude_mean"] is not None:
            near_pixels += int(near["pixels"])
            near_sum += float(near["normalized_magnitude_mean"]) * int(near["pixels"])
        if int(far["pixels"]) and far["normalized_magnitude_mean"] is not None:
            far_pixels += int(far["pixels"])
            far_sum += float(far["normalized_magnitude_mean"]) * int(far["pixels"])
    near_mean = near_sum / near_pixels if near_pixels else None
    far_mean = far_sum / far_pixels if far_pixels else None
    separated_ok = bool(near_pixels and far_pixels)
    difference = abs(near_mean - far_mean) if separated_ok else None
    differing_ok = difference is not None and difference >= criteria["near_far_mean_delta_min"]
    return {"broad_ok": broad_ok, "broad_fractions": broad,
            "near_pixels": near_pixels, "far_pixels": far_pixels,
            "near_mean_normalized_magnitude": near_mean,
            "far_mean_normalized_magnitude": far_mean,
            "separated_ok": separated_ok,
            "mean_difference": difference, "differing_ok": differing_ok}


def _scene_frame_map(scene):
    records = scene.get("frames")
    if not isinstance(records, list) or not records:
        return None
    mapping = {}
    for record in records:
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Scene frames need integer local indices")
        if index in mapping:
            raise ValueError("Scene frames contain a duplicate local index")
        mapping[index] = record
    return mapping


def inspect_inputs(run):
    run = Path(run)
    manifest = json.loads((run / "application_inputs/manifest.json").read_text(encoding="utf-8"))
    scene = json.loads((run / "scene.json").read_text(encoding="utf-8"))
    if scene["schema"] != "xefg.render_scene/1":
        raise ValueError("Expected a rendered scene manifest")
    motion_width = int(manifest["motion_width"])
    motion_height = int(manifest["motion_height"])
    width = int(manifest.get("width", motion_width * 2))
    height = int(manifest.get("height", motion_height * 2))
    count = motion_width * motion_height
    scene_frames = _scene_frame_map(scene)
    warmup = int(scene.get("warmup", manifest.get("warmup", 0)))
    mode = scene.get("mode")
    isolated = scene.get("pairing_method") == "isolated_pair" or mode == 1
    scene_name = scene.get("name", "")
    evidence_scene = scene_name in EVIDENCE_SCENES
    rows = []
    for frame in manifest["frames"]:
        files = frame["inputs"]
        depth = np.fromfile(run / "application_inputs" / files["depth"], dtype="<f4")
        motion = np.fromfile(run / "application_inputs" / files["motion"],
                             dtype="<f2").reshape(-1, 2).astype(np.float32)
        if depth.size != count or len(motion) != count:
            raise ValueError("Geometry resource size mismatch")
        if not np.isfinite(depth).all() or not np.isfinite(motion).all():
            raise ValueError("Non-finite depth or motion")
        if depth.min() < 0 or depth.max() > 1 or not np.any(depth < 1):
            raise ValueError("Expected nonempty normal device-depth scene in [0,1]")
        depth_grid = depth.reshape(motion_height, motion_width)
        row = _legacy_row(frame, depth, motion)
        if scene_frames is not None:
            index = frame["index"]
            if index not in scene_frames:
                raise ValueError("Scene frames do not describe input frame %s" % index)
            record = scene_frames[index]
            if "scene_frame" in record:
                source_frame = record["scene_frame"]
                if isinstance(source_frame, bool) or not isinstance(source_frame, int):
                    raise ValueError("Scene frame %s has a non-integer scene_frame" % index)
            elif isolated:
                raise ValueError("Isolated scene frame %s has no source scene_frame" % index)
            else:
                source_frame = index
            measured = index >= warmup and not bool(record.get("reset_history"))
            row.update({"source_frame": source_frame, "measured": measured,
                        "mode": int(mode) if isinstance(mode, int) else None,
                        "isolated": bool(isolated), "pixels": int(count)})
            foreground, background = _aggregate_depth(depth_grid)
            row.update({"foreground_depth": _foreground_depth(depth_grid, foreground),
                        "background_depth": background,
                        **_motion_summary(depth_grid, motion, motion_width, motion_height)})
            if evidence_scene and measured:
                color = None
                if scene_name == "transparency":
                    try:
                        color = _read_color(run, files, width, height)
                    except ValueError:
                        color = None
                row["scene_evidence"] = scene_evidence.build_frame_evidence(
                    scene_name, scene, index, depth_grid, motion.reshape(motion_height,
                                                                         motion_width, 2),
                    color, width, height, motion_width, motion_height,
                    COVERAGE_CRITERIA[scene_name])
        rows.append(row)
    if scene_frames is None:
        if evidence_scene:
            for row in rows:
                index = int(row["frame"])
                measured = index >= warmup
                row.update({"source_frame": index, "measured": measured,
                            "mode": int(mode) if isinstance(mode, int) else None,
                            "isolated": bool(isolated), "pixels": int(count)})
                if measured:
                    row["scene_evidence"] = scene_evidence._failure(
                        scene_name, "scene has no per-frame metadata")
            coverage = aggregate_coverage(scene_name, rows)
            return {"schema": SCHEMA, "frames": rows, "coverage": coverage,
                    "note": "New-scene evidence is unavailable without frame metadata."}
        return {"schema": LEGACY_SCHEMA, "frames": rows,
                "note": "Finite/range checks only; motion direction and units are covered by native tests."}
    coverage = aggregate_coverage(scene_name, rows)
    return {"schema": SCHEMA, "frames": rows, "coverage": coverage,
            "note": ("Input-only geometry/motion coverage; no SDK output or model quality is used."
                     " Warmup, reset history and isolated-pair base rows are retained but not measured.")}


def _read_color(run, files, width, height):
    name = files.get("color")
    if not name:
        raise ValueError("Transparency scene needs a dumped color input")
    pixels = np.fromfile(run / "application_inputs" / name, dtype=np.uint8)
    if pixels.size != width * height * 4:
        raise ValueError("Color resource size mismatch")
    return pixels.reshape(height, width, 4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = inspect_inputs(args.run)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    coverage = report.get("coverage", {})
    status = coverage.get("status", "not_applicable")
    isolated = any(row.get("isolated") for row in report["frames"])
    print("Checked %d GPU input frames; coverage %s" % (len(report["frames"]), status))
    if status == "fail" and not isolated:
        print("Input coverage gate failed: %s" % coverage.get("reason"), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
