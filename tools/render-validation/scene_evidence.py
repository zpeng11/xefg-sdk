"""Input-only evidence for the opt-in nonrigid and transparency scenes.

The native renderer publishes ``scene_extension`` metadata plus authoritative
per-frame object transforms.  This module rebuilds the small analytic geometry
needed to verify GPU depth/motion/color at sampled interiors.  It deliberately
does not rasterize whole frames: only a deterministic spread of triangle or
panel interiors is sampled.
"""
from __future__ import annotations

import math

import numpy as np


SCENE_EXTENSION_SCHEMA = "xefg.render_scene_extension/1"
NEW_SCENES = ("nonrigid", "transparency")
MAX_SAMPLES = 32
MAX_REGION_SAMPLES = 24

DEFAULT_NONRIGID_CRITERIA = {
    "min_triangle_samples": 8,
    "min_sampled_triangles": 4,
    "interior_margin_pixels": 0.1,
    "depth_tolerance": 5.0e-4,
    "mv_tolerance_pixels": 0.25,
    "static_matrix_tolerance": 1.0e-6,
    "vertex_pair_change_fraction_min": 0.01,
    "min_changed_pairs": 4,
    "mv_spatial_range_pixels_min": 0.25,
    "min_measured_source_frames": 2,
}

DEFAULT_TRANSPARENCY_CRITERIA = {
    "min_samples_per_region": 9,
    "region_area_fraction_min": 0.002,
    "color_tolerance": 2,
    "depth_tolerance": 5.0e-4,
    "mv_tolerance_pixels": 0.25,
    "interior_margin_pixels": 2.0,
    "temporal_motion_pixels_min": 1.0,
    "min_temporal_rows": 2,
    "min_measured_source_frames": 2,
}


class EvidenceError(ValueError):
    """Raised when required scene metadata or evidence is missing or invalid."""


def criteria_for(kind, overrides=None):
    base = dict(DEFAULT_NONRIGID_CRITERIA if kind == "nonrigid"
                else DEFAULT_TRANSPARENCY_CRITERIA)
    if overrides:
        base.update(overrides)
    return base


def _finite(value, label):
    if isinstance(value, bool) or value is None:
        raise EvidenceError("%s must be a finite number" % label)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise EvidenceError("%s must be a finite number" % label) from None
    if not math.isfinite(number):
        raise EvidenceError("%s must be a finite number" % label)
    return number


def _matrix(values, label):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (16,):
        raise EvidenceError("%s must contain 16 numbers" % label)
    if not np.isfinite(array).all():
        raise EvidenceError("%s must be finite" % label)
    return array.reshape(4, 4)


def _points(values, label):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise EvidenceError("%s must be a nonempty list of xyz points" % label)
    if not np.isfinite(array).all():
        raise EvidenceError("%s must be finite" % label)
    return array


def _boolean(value, label):
    if not isinstance(value, bool):
        raise EvidenceError("%s must be a boolean" % label)
    return value


def _frame_map(scene):
    frames = scene.get("frames")
    if not isinstance(frames, list) or not frames:
        raise EvidenceError("scene has no frame records")
    mapping = {}
    for record in frames:
        if not isinstance(record, dict):
            raise EvidenceError("scene frame record must be an object")
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise EvidenceError("scene frame index must be an integer")
        if index in mapping:
            raise EvidenceError("scene frame indices are duplicated")
        mapping[index] = record
    return mapping


def extension(scene, kind):
    extension_data = scene.get("scene_extension")
    if not isinstance(extension_data, dict):
        raise EvidenceError("missing scene_extension metadata for %s" % kind)
    if extension_data.get("schema") != SCENE_EXTENSION_SCHEMA:
        raise EvidenceError("unsupported scene_extension schema")
    version = extension_data.get("version")
    if isinstance(version, bool) or version != 1:
        raise EvidenceError("unsupported scene_extension version")
    if extension_data.get("scene") != kind:
        raise EvidenceError("scene_extension belongs to another scene")
    return extension_data


def _transform(points, matrix):
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ matrix)[:, :3]


def _project(points, matrix, width, height):
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    clip = homogeneous @ matrix
    w = clip[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :2] / w[:, None]
        depth = clip[:, 2] / w
        pixels = (ndc * np.array([0.5, -0.5]) + 0.5) * np.array([width, height], dtype=np.float64)
    return clip, pixels, depth, w


def _barycentric(point, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denominator = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denominator) < 1.0e-12:
        return None
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / denominator
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / denominator
    return np.array([1.0 - u - v, u, v])


def _segment_distance(point, a, b):
    edge = b - a
    length_squared = float(edge @ edge)
    if length_squared < 1.0e-12:
        return float(np.linalg.norm(point - a))
    t = float((point - a) @ edge) / length_squared
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(point - (a + t * edge)))


def _triangle_margin(point, triangle):
    return min(_segment_distance(point, triangle[i], triangle[(i + 1) % 3])
               for i in range(3))


def _perspective_attributes(current_clip, previous_clip, bary):
    inverse_w = 1.0 / current_clip[:, 3]
    denominator = float(bary @ inverse_w)
    if abs(denominator) < 1.0e-12:
        raise EvidenceError("degenerate perspective interpolation")
    # SV_Position.z (device depth) is interpolated linearly in screen space.
    depth = float(bary @ (current_clip[:, 2] / current_clip[:, 3]))
    previous_xy = (bary @ (previous_clip[:, :2] * inverse_w[:, None])) / denominator
    previous_w = float((bary @ (previous_clip[:, 3] * inverse_w)) / denominator)
    return depth, previous_xy, previous_w, denominator


def _previous_pixel(previous_xy, previous_w, width, height, current_pixel):
    if previous_w <= 0.05:
        return np.asarray(current_pixel, dtype=np.float64)
    ndc = previous_xy / previous_w
    return (ndc * np.array([0.5, -0.5]) + 0.5) * np.array([width, height], dtype=np.float64)


def _quantize(values):
    return np.clip(np.floor(np.asarray(values, dtype=np.float64) * 255.0 + 0.5), 0.0, 255.0)


def _failure(kind, reason):
    return {
        "kind": kind,
        "status": "fail",
        "reason": reason,
        "checks": [{"name": "metadata_present", "status": "fail",
                    "observed": False, "threshold": "valid scene_extension metadata"}],
        "samples": [],
        "sample_count": 0,
        "sampled_triangle_count": 0,
    }


def build_frame_evidence(kind, scene, index, depth, motion, color,
                         width, height, motion_width, motion_height, criteria=None):
    """Return one frame's evidence dict.  Never raises for metadata problems."""
    try:
        if kind == "nonrigid":
            return _nonrigid_evidence(scene, index, depth, motion, motion_width,
                                      motion_height, criteria)
        if kind == "transparency":
            return _transparency_evidence(scene, index, color, depth, motion, width,
                                          height, motion_width, motion_height, criteria)
        raise EvidenceError("unsupported scene kind %s" % kind)
    except EvidenceError as error:
        return _failure(kind, str(error))
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        return _failure(kind, "invalid metadata: %s" % error)


# ---------------------------------------------------------------------------
# Nonrigid: procedural deformed grid


def _grid_definition(extension_data):
    grid = extension_data.get("grid")
    if not isinstance(grid, dict):
        raise EvidenceError("nonrigid scene_extension has no grid")
    cells_x = grid.get("cells_x")
    cells_y = grid.get("cells_y")
    if (isinstance(cells_x, bool) or not isinstance(cells_x, int)
            or isinstance(cells_y, bool) or not isinstance(cells_y, int)
            or cells_x < 1 or cells_y < 1):
        raise EvidenceError("grid cells_x/cells_y must be positive integers")
    if (cells_x + 1) * (cells_y + 1) > 65536:
        raise EvidenceError("grid is too large")
    deformation = extension_data.get("deformation")
    if not isinstance(deformation, dict):
        raise EvidenceError("nonrigid scene_extension has no deformation")
    parameters = {}
    for name in ("A1", "kx", "ky", "w1", "w2", "A2", "k2", "w3", "phi2"):
        parameters[name] = _finite(deformation.get(name), "deformation." + name)
    return cells_x, cells_y, parameters


def _deformed_vertices(cells_x, cells_y, parameters, time):
    columns = cells_x + 1
    xs = -1.0 + 2.0 * np.arange(columns, dtype=np.float64) / cells_x
    ys = -1.0 + 2.0 * np.arange(cells_y + 1, dtype=np.float64) / cells_y
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = (parameters["A1"] * np.sin(parameters["kx"] * grid_x + parameters["w1"] * time)
         * np.cos(parameters["ky"] * grid_y + parameters["w2"] * time)
         + parameters["A2"] * np.sin(parameters["k2"] * (grid_x + grid_y)
                                      + parameters["w3"] * time + parameters["phi2"]))
    return np.stack([grid_x, grid_y, z], axis=-1).reshape(-1, 3)


def _grid_triangles(cells_x, cells_y):
    columns = cells_x + 1
    rows, columns_index = np.meshgrid(np.arange(cells_y), np.arange(cells_x), indexing="ij")
    a = (rows * columns + columns_index).reshape(-1)
    b = a + 1
    c = a + columns
    d = c + 1
    first = np.stack([a, b, d], axis=1)
    second = np.stack([a, d, c], axis=1)
    return np.concatenate([first, second], axis=0)


def _grid_object(frame):
    objects = frame.get("objects")
    if not isinstance(objects, list):
        raise EvidenceError("frame has no objects list")
    matches = []
    for record in objects:
        if not isinstance(record, dict):
            continue
        label = (str(record.get("mesh", "")) + " " + str(record.get("role", ""))).lower()
        if "grid" in label:
            matches.append(record)
    if len(matches) != 1:
        raise EvidenceError("nonrigid frame needs exactly one grid object")
    return matches[0]


def _triangle_pixels(triangle):
    return np.asarray(triangle, dtype=np.float64)


def _sample_triangle_interior(pixel_triangle, margin, width, height):
    """Return (pixel index, pixel-center, barycentric) or None."""
    if not np.isfinite(pixel_triangle).all():
        return None
    centroid = pixel_triangle.mean(axis=0)
    candidates = [
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        np.array([0.5, 0.25, 0.25]),
        np.array([0.25, 0.5, 0.25]),
        np.array([0.25, 0.25, 0.5]),
        np.array([0.6, 0.2, 0.2]),
        np.array([0.2, 0.6, 0.2]),
        np.array([0.2, 0.2, 0.6]),
    ]
    for bary in candidates:
        point = bary @ pixel_triangle
        x = int(math.floor(point[0]))
        y = int(math.floor(point[1]))
        if not (0 <= x < width and 0 <= y < height):
            continue
        center = np.array([x + 0.5, y + 0.5])
        screen_bary = _barycentric(center, *pixel_triangle)
        if screen_bary is None or screen_bary.min() < -1.0e-9:
            continue
        if _triangle_margin(center, pixel_triangle) < margin:
            continue
        # Prefer a sample reasonably near the requested barycentric point.
        if float(np.linalg.norm(center - centroid)) < 1.0e-9 and margin > 0.0:
            continue
        return np.array([x, y], dtype=int), center, screen_bary
    return None


def _nonrigid_evidence(scene, index, depth, motion, width, height, criteria):
    criteria = criteria_for("nonrigid", criteria)
    extension_data = extension(scene, "nonrigid")
    cells_x, cells_y, parameters = _grid_definition(extension_data)
    frames = _frame_map(scene)
    if index not in frames:
        raise EvidenceError("scene_extension has no record for frame %s" % index)
    frame = frames[index]
    deformation = frame.get("deformation")
    if not isinstance(deformation, dict):
        raise EvidenceError("frame %s has no deformation timing" % index)
    for name in ("time", "previous_time", "previous_is_current"):
        if name not in deformation:
            raise EvidenceError("frame %s deformation is missing %s" % (index, name))
    time = _finite(deformation["time"], "deformation.time")
    previous_time = _finite(deformation["previous_time"], "deformation.previous_time")
    previous_is_current = _boolean(deformation["previous_is_current"],
                                   "deformation.previous_is_current")

    grid_object = _grid_object(frame)
    world = _matrix(grid_object.get("world"), "grid world")
    previous_world = _matrix(grid_object.get("previous_world"), "grid previous_world")
    view = _matrix(frame.get("view"), "frame view")
    projection = _matrix(frame.get("projection"), "frame projection")
    previous_view = _matrix(frame.get("previous_view", frame.get("view")),
                            "frame previous_view")
    previous_projection = _matrix(frame.get("previous_projection", frame.get("projection")),
                                  "frame previous_projection")

    current_local = _deformed_vertices(cells_x, cells_y, parameters, time)
    previous_local = (current_local.copy() if previous_is_current
                      else _deformed_vertices(cells_x, cells_y, parameters, previous_time))
    triangles = _grid_triangles(cells_x, cells_y)

    current_world = _transform(current_local, world)
    previous_world_points = _transform(previous_local, previous_world)
    current_view_projection = view @ projection
    previous_view_projection = previous_view @ previous_projection
    current_clip, current_pixels, _, current_w = _project(
        current_world, current_view_projection, width, height)
    previous_clip, _, _, _ = _project(previous_world_points, previous_view_projection,
                                      width, height)

    margin = float(criteria["interior_margin_pixels"])
    stride = max(1, len(triangles) // 96)
    samples = []
    sampled_triangles = set()
    offsets = [0, stride // 2, stride // 3, stride // 5]
    for offset in offsets:
        for triangle_index in range(offset, len(triangles), stride):
            if len(samples) >= MAX_SAMPLES:
                break
            if triangle_index in sampled_triangles:
                continue
            triangle = triangles[triangle_index]
            if current_w[triangle].min() <= 0.05:
                continue
            pixel_triangle = current_pixels[triangle]
            found = _sample_triangle_interior(pixel_triangle, margin, width, height)
            if found is None:
                continue
            pixel_index, center, screen_bary = found
            try:
                expected_depth, previous_xy, previous_w, denominator = _perspective_attributes(
                    current_clip[triangle], previous_clip[triangle], screen_bary)
                previous_pixel = _previous_pixel(previous_xy, previous_w, width, height,
                                                 center)
            except EvidenceError:
                continue
            expected_motion = previous_pixel - center
            x, y = int(pixel_index[0]), int(pixel_index[1])
            actual_depth = float(depth[y, x])
            actual_motion = np.asarray(motion[y, x], dtype=np.float64)
            depth_error = abs(actual_depth - expected_depth)
            motion_error = float(np.max(np.abs(actual_motion - expected_motion)))
            samples.append({
                "triangle": int(triangle_index),
                "pixel": [x, y],
                "depth_expected": expected_depth,
                "depth_actual": actual_depth,
                "depth_error": depth_error,
                "motion_expected": expected_motion.tolist(),
                "motion_actual": actual_motion.tolist(),
                "motion_error": motion_error,
                "pass": (depth_error <= float(criteria["depth_tolerance"])
                         and motion_error <= float(criteria["mv_tolerance_pixels"])),
            })
            sampled_triangles.add(int(triangle_index))
        if len(samples) >= MAX_SAMPLES:
            break

    spatial_range = 0.0
    direction_count = 0
    max_motion = 0.0
    if samples:
        expected = np.asarray([sample["motion_expected"] for sample in samples])
        differences = expected[:, None, :] - expected[None, :, :]
        spatial_range = float(np.max(np.linalg.norm(differences, axis=2)))
        direction_count = int(np.unique(np.round(expected, 2), axis=0).shape[0])
        max_motion = float(np.max(np.linalg.norm(expected, axis=1)))

    pair_distances = _vertex_pair_distances(current_local, previous_local, triangles)
    checks = [
        _check("metadata_present", True, "valid scene_extension and frame deformation"),
        _check("samples_match_analysis", bool(samples) and all(s["pass"] for s in samples),
               "all %d sampled triangle interiors within tolerance" % len(samples)),
    ]
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    errors = [s for s in samples if not s["pass"]]
    reason = None if status == "pass" else (
        "nonrigid samples/motion do not match analytic deformation")
    return {
        "kind": "nonrigid",
        "status": status,
        "reason": reason,
        "checks": checks,
        "samples": samples,
        "sample_count": len(samples),
        "sampled_triangle_count": len(sampled_triangles),
        "failed_sample_count": len(errors),
        "max_depth_error": max((s["depth_error"] for s in samples), default=0.0),
        "max_motion_error": max((s["motion_error"] for s in samples), default=0.0),
        "mv_spatial_range": spatial_range,
        "mv_direction_count": direction_count,
        "max_motion_magnitude": max_motion,
        "pair_distances": pair_distances,
        "matrices": {
            "world": world.tolist(),
            "previous_world": previous_world.tolist(),
            "view": view.tolist(),
            "previous_view": previous_view.tolist(),
            "projection": projection.tolist(),
            "previous_projection": previous_projection.tolist(),
        },
        "deformation": {
            "time": time,
            "previous_time": previous_time,
            "previous_is_current": previous_is_current,
        },
    }


def _vertex_pair_distances(current_local, previous_local, triangles):
    """Deterministic spread of grid edges, for cross-frame deformation checks."""
    pairs = []
    seen = set()
    stride = max(1, len(triangles) // 24)
    for triangle_index in range(0, len(triangles), stride):
        triangle = triangles[triangle_index]
        for left, right in ((0, 1), (1, 2), (2, 0)):
            pair = (int(triangle[left]), int(triangle[right]))
            key = (min(pair), max(pair))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) >= 32:
                return [
                    {"a": a, "b": b,
                     "current": float(np.linalg.norm(current_local[a] - current_local[b])),
                     "previous": float(np.linalg.norm(previous_local[a] - previous_local[b]))}
                    for a, b in pairs
                ]
    return [
        {"a": a, "b": b,
         "current": float(np.linalg.norm(current_local[a] - current_local[b])),
         "previous": float(np.linalg.norm(previous_local[a] - previous_local[b]))}
        for a, b in pairs
    ]


# ---------------------------------------------------------------------------
# Transparency: flat premultiplied panels and projected region differences


def _object_quad(mesh):
    # Transparency scenes use flat panels only; the mesh label is informational.
    if not isinstance(mesh, str) or not mesh:
        raise EvidenceError("transparency object needs a mesh label")
    return np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                     [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)


def _validate_objects(extension_data, frame, index):
    objects = frame.get("objects")
    if not isinstance(objects, list) or not objects:
        raise EvidenceError("transparency frame %s has no objects" % index)
    records = {}
    for record in objects:
        if not isinstance(record, dict):
            raise EvidenceError("transparency object record must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise EvidenceError("transparency object needs a name")
        if name in records:
            raise EvidenceError("transparency object names are duplicated")
        records[name] = record
    draw_order = extension_data.get("draw_order")
    if not isinstance(draw_order, list) or not draw_order:
        raise EvidenceError("transparency scene_extension has no draw_order")
    if any(not isinstance(name, str) for name in draw_order):
        raise EvidenceError("draw_order entries must be names")
    if len(set(draw_order)) != len(draw_order):
        raise EvidenceError("draw_order contains duplicate names")
    if set(draw_order) != set(records):
        raise EvidenceError("draw_order must list every object exactly once")

    policy = extension_data.get("policy")
    if not isinstance(policy, dict):
        raise EvidenceError("transparency scene_extension has no policy")
    if policy.get("depth_motion") != "opaque_only":
        raise EvidenceError("transparency policy must be opaque_only depth/motion")
    if policy.get("transparent_depth_write") is not False:
        raise EvidenceError("transparent_depth_write must be false")
    if policy.get("transparent_writes_mv") is not False:
        raise EvidenceError("transparent_writes_mv must be false")
    blend = extension_data.get("blend")
    if not isinstance(blend, dict) or blend.get("mode") not in (
            "premultiplied_alpha", "premultiplied_source_over"):
        raise EvidenceError("transparency blend must be premultiplied alpha")

    parsed = {}
    sort_values = []
    for name in draw_order:
        record = records[name]
        for key in ("world", "previous_world", "flat_color_rgba",
                    "premultiplied_rgba", "transparent", "depth_write",
                    "writes_depth_motion", "sort_index", "role", "mesh"):
            if key not in record:
                raise EvidenceError("%s is missing %s" % (name, key))
        transparent = _boolean(record["transparent"], name + ".transparent")
        depth_write = _boolean(record["depth_write"], name + ".depth_write")
        writes_motion = _boolean(record["writes_depth_motion"],
                                 name + ".writes_depth_motion")
        if transparent and (depth_write or writes_motion):
            raise EvidenceError("%s writes depth/motion while transparent" % name)
        if not transparent and (not depth_write or not writes_motion):
            raise EvidenceError("%s is opaque but does not write depth/motion" % name)
        role = record["role"]
        if not isinstance(role, str) or not role:
            raise EvidenceError("%s needs a role" % name)
        sort_index = record["sort_index"]
        if isinstance(sort_index, bool) or not isinstance(sort_index, int):
            raise EvidenceError("%s.sort_index must be an integer" % name)
        sort_values.append(sort_index)
        flat = np.asarray(record["flat_color_rgba"], dtype=np.float64)
        premultiplied = np.asarray(record["premultiplied_rgba"], dtype=np.float64)
        if flat.shape != (4,) or premultiplied.shape != (4,):
            raise EvidenceError("%s colours must contain four components" % name)
        if not np.isfinite(flat).all() or not np.isfinite(premultiplied).all():
            raise EvidenceError("%s colours must be finite" % name)
        if flat.min() < -1.0e-6 or flat.max() > 1.0 + 1.0e-6:
            raise EvidenceError("%s flat colour must be in [0,1]" % name)
        alpha = flat[3]
        if premultiplied[3] < -1.0e-6 or premultiplied[3] > 1.0 + 1.0e-6:
            raise EvidenceError("%s premultiplied alpha must be in [0,1]" % name)
        if np.max(np.abs(premultiplied[:3] - flat[:3] * alpha)) > 2.0e-3:
            raise EvidenceError("%s premultiplied_rgba disagrees with flat colour" % name)
        if premultiplied[:3].min() < -2.0e-3 or premultiplied[:3].max() > alpha + 2.0e-3:
            raise EvidenceError("%s premultiplied_rgba exceeds alpha" % name)
        parsed[name] = {
            "name": name,
            "world": _matrix(record["world"], name + ".world"),
            "previous_world": _matrix(record["previous_world"], name + ".previous_world"),
            "flat": flat,
            "premultiplied": premultiplied,
            "transparent": transparent,
            "depth_write": depth_write,
            "writes_motion": writes_motion,
            "sort_index": sort_index,
            "role": role,
            "mesh": record["mesh"],
            "local_quad": _object_quad(record["mesh"]),
        }
    if any(left > right for left, right in zip(sort_values, sort_values[1:])):
        raise EvidenceError("draw_order is not back-to-front by sort_index")
    return parsed, list(draw_order)


def _object_projection(panel, world_matrix, view_projection, width, height):
    points = _transform(panel["local_quad"], world_matrix)
    clip, pixels, depth, w = _project(points, view_projection, width, height)
    return {"clip": clip, "pixels": pixels, "depth": depth, "w": w, "points": points,
            "local": panel["local_quad"]}


def _quad_hit_local(projection, point):
    pixels = projection["pixels"]
    clip = projection["clip"]
    local = projection["local"]
    for indices in ((0, 1, 3), (0, 3, 2)):
        triangle = pixels[list(indices)]
        bary = _barycentric(point, *triangle)
        if bary is None or bary.min() < -1.0e-9:
            continue
        clamped = np.maximum(bary, 0.0)
        triangle_clip = clip[list(indices)]
        try:
            depth, _, _, _ = _perspective_attributes(triangle_clip, triangle_clip, clamped)
        except EvidenceError:
            return None
        local_point = clamped @ local[list(indices)]
        return {"indices": indices, "bary": clamped, "depth": depth,
                "local": local_point}
    return None


def _quad_signed_distance(projection, point):
    polygon = projection["pixels"]
    inside = _quad_hit_local(projection, point) is not None
    distance = min(_segment_distance(point, polygon[i], polygon[(i + 1) % 4])
                   for i in range(4))
    return distance if inside else -distance


def _project_panel(panel, frame, index, width, height):
    view = _matrix(frame.get("view"), "frame view")
    projection = _matrix(frame.get("projection"), "frame projection")
    previous_view = _matrix(frame.get("previous_view", frame.get("view")),
                            "frame previous_view")
    previous_projection = _matrix(frame.get("previous_projection", frame.get("projection")),
                                  "frame previous_projection")
    current = _object_projection(panel, panel["world"], view @ projection, width, height)
    previous = _object_projection(panel, panel["previous_world"],
                                  previous_view @ previous_projection, width, height)
    current["local"] = panel["local_quad"]
    previous["local"] = panel["local_quad"]
    current["panel"] = panel
    previous["panel"] = panel
    return current, previous


def _composite(draw_order, panels, coverage, point):
    """Return quantized BGRA bytes, opaque depth, and frontmost opaque name."""
    destination = np.array([0.0, 0.0, 0.0, 255.0])
    destination_rgba = np.array([0.0, 0.0, 0.0, 1.0])
    depth_buffer = 1.0
    depth_source = None
    transparent_depths = []
    for name in draw_order:
        hit = coverage.get(name)
        if hit is None:
            continue
        panel = panels[name]
        depth = hit["depth"]
        if depth > depth_buffer + 1.0e-6:
            continue
        premultiplied = panel["premultiplied"]
        if not panel["transparent"]:
            destination_rgba = np.array([premultiplied[0], premultiplied[1],
                                         premultiplied[2], 1.0])
            destination = np.concatenate([_quantize(destination_rgba[:3]), [255.0]])
            depth_buffer = depth
            depth_source = name
        else:
            alpha = float(premultiplied[3])
            destination_rgba[:3] = premultiplied[:3] + destination_rgba[:3] * (1.0 - alpha)
            destination_rgba[3] = alpha + destination_rgba[3] * (1.0 - alpha)
            destination = np.concatenate([_quantize(destination_rgba[:3]),
                                          _quantize([destination_rgba[3]])])
            destination_rgba[:3] = destination[:3] / 255.0
            destination_rgba[3] = destination[3] / 255.0
            transparent_depths.append((name, depth))
    return destination, depth_buffer, depth_source, transparent_depths


def _region_kind(transparent_names, occluder_names):
    if transparent_names and occluder_names:
        return "occluded"
    if len(transparent_names) >= 2:
        return "double"
    if len(transparent_names) == 1:
        return "single"
    return None


def _stable_region(kind, panels, signed, margin):
    """Region membership must not sit near a panel boundary."""
    transparent = [name for name, panel in panels.items()
                   if panel["transparent"] and signed[name] >= margin]
    occluders = [name for name, panel in panels.items()
                 if not panel["transparent"] and not _is_background(panel)
                 and signed[name] >= margin]
    if kind == "occluded":
        return bool(occluders and transparent)
    if kind == "double":
        return len(transparent) >= 2 and not occluders
    if kind == "single":
        return (len(transparent) == 1 and not occluders
                and all(signed[name] <= -margin for name, panel in panels.items()
                        if panel["transparent"] and name not in transparent)
                and all(signed[name] <= -margin for name, panel in panels.items()
                        if not panel["transparent"] and not _is_background(panel)))
    return False


def _is_background(panel):
    labels = (str(panel.get("role", "")) + " " + str(panel.get("name", ""))).lower()
    return "backdrop" in labels or "background" in labels or "sky" in labels


def _transparency_evidence(scene, index, color, depth, motion, width, height,
                           motion_width, motion_height, criteria):
    criteria = criteria_for("transparency", criteria)
    extension_data = extension(scene, "transparency")
    frames = _frame_map(scene)
    if index not in frames:
        raise EvidenceError("scene_extension has no record for frame %s" % index)
    frame = frames[index]
    panels, draw_order = _validate_objects(extension_data, frame, index)
    if color is None:
        raise EvidenceError("transparency scene needs the BGRA color input")
    if color.shape != (height, width, 4):
        raise EvidenceError("color input size mismatch")

    projections = {}
    previous_projections = {}
    for name, panel in panels.items():
        current, previous = _project_panel(panel, frame, index, width, height)
        projections[name] = current
        previous_projections[name] = previous
    for panel in panels.values():
        panel["previous_geometry_projection"] = _object_projection(
            panel, panel["previous_world"],
            _matrix(frame.get("previous_view", frame.get("view")),
                    "frame previous_view") @
            _matrix(frame.get("previous_projection", frame.get("projection")),
                    "frame previous_projection"),
            motion_width, motion_height)
        panel["geometry_projection"] = _object_projection(
            panel, panel["world"],
            _matrix(frame.get("view"), "frame view") @
            _matrix(frame.get("projection"), "frame projection"),
            motion_width, motion_height)

    margin = float(criteria["interior_margin_pixels"])
    step = max(2, min(width, height) // 90)
    region_counts = {"single": 0, "double": 0, "occluded": 0}
    candidates = {"single": [], "double": [], "occluded": []}
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            point = np.array([x + 0.5, y + 0.5])
            coverage = {}
            for name in draw_order:
                hit = _quad_hit_local(projections[name], point)
                if hit is not None:
                    coverage[name] = hit
            transparent_names = [name for name, hit in coverage.items()
                                 if panels[name]["transparent"]]
            occluder_names = [name for name, hit in coverage.items()
                              if not panels[name]["transparent"] and not _is_background(panels[name])]
            kind = _region_kind(transparent_names, occluder_names)
            if kind is None:
                continue
            region_counts[kind] += 1
            signed = {name: _quad_signed_distance(projections[name], point)
                      for name in panels}
            if not _stable_region(kind, panels, signed, 0.0):
                continue
            if _stable_region(kind, panels, signed, margin):
                candidates[kind].append([x, y])

    def select(items):
        if not items:
            return []
        if len(items) <= MAX_REGION_SAMPLES:
            return items
        indices = np.linspace(0, len(items) - 1, MAX_REGION_SAMPLES).astype(int)
        return [items[int(i)] for i in indices]

    region_evidence = {}
    total_color_errors = 0
    total_depth_errors = 0
    total_motion_errors = 0
    for kind in ("single", "double", "occluded"):
        region = {"samples": [], "sample_count": 0, "pixel_count": region_counts[kind],
                  "area_fraction": (region_counts[kind] * step * step) / float(width * height),
                  "max_color_error": 0, "max_depth_error": 0, "max_motion_error": 0,
                  "color_mismatches": 0, "depth_mismatches": 0, "motion_mismatches": 0}
        for x, y in select(candidates[kind]):
            point = np.array([x + 0.5, y + 0.5])
            coverage = {}
            for name in draw_order:
                hit = _quad_hit_local(projections[name], point)
                if hit is not None:
                    coverage[name] = hit
            expected_bytes, _, _, transparent_depths = _composite(
                draw_order, panels, coverage, point)
            if kind == "double":
                order = [name for name, _ in transparent_depths]
                depths = dict(transparent_depths)
                for first, second in zip(order, order[1:]):
                    if depths[first] + 1.0e-6 < depths[second]:
                        raise EvidenceError("transparent draw order is front-to-back")
            actual_bgra = color[y, x]
            expected_bgra = np.array([expected_bytes[2], expected_bytes[1],
                                      expected_bytes[0], expected_bytes[3]])
            color_error = int(np.max(np.abs(actual_bgra.astype(np.int16)
                                            - expected_bgra.astype(np.int16))))
            region["max_color_error"] = max(region["max_color_error"], color_error)

            # Geometry is evaluated at the half-resolution pixel centre.
            geometry_x = min(motion_width - 1, max(0, x // 2))
            geometry_y = min(motion_height - 1, max(0, y // 2))
            geometry_point = np.array([geometry_x + 0.5, geometry_y + 0.5])
            opaque = {}
            for name in draw_order:
                panel = panels[name]
                if panel["transparent"]:
                    continue
                hit = _quad_hit_local(panel["geometry_projection"], geometry_point)
                if hit is not None:
                    opaque[name] = hit
            depth_source = None
            depth_buffer = 1.0
            for name in draw_order:
                hit = opaque.get(name)
                if hit is None:
                    continue
                if hit["depth"] <= depth_buffer + 1.0e-6:
                    depth_buffer = hit["depth"]
                    depth_source = name
            if depth_source is None:
                depth_error = 1.0
                motion_error = float("inf")
            else:
                panel = panels[depth_source]
                current_geometry = _quad_hit_local(panel["geometry_projection"],
                                                   geometry_point)
                if current_geometry is None:
                    depth_error = 1.0
                    motion_error = float("inf")
                else:
                    depth_error = abs(float(depth[geometry_y, geometry_x]) - depth_buffer)
                    local_point = current_geometry["local"]
                    previous_world = _transform(np.asarray([local_point]),
                                                panel["previous_world"])[0]
                    clip, pixels, _, w = _project(np.asarray([previous_world]),
                                                  _matrix(frame.get("previous_view",
                                                                    frame.get("view")),
                                                          "frame previous_view") @
                                                  _matrix(frame.get("previous_projection",
                                                                    frame.get("projection")),
                                                          "frame previous_projection"),
                                                  motion_width, motion_height)
                    previous_pixel = (pixels[0] if float(w[0]) > 0.05
                                      else geometry_point)
                    expected_motion = previous_pixel - geometry_point
                    actual_motion = np.asarray(motion[geometry_y, geometry_x],
                                               dtype=np.float64)
                    motion_error = float(np.max(np.abs(actual_motion - expected_motion)))
            region["max_depth_error"] = max(region["max_depth_error"], depth_error)
            region["max_motion_error"] = max(region["max_motion_error"],
                                             motion_error if math.isfinite(motion_error)
                                             else 1.0e9)
            color_bad = color_error > int(criteria["color_tolerance"])
            depth_bad = depth_error > float(criteria["depth_tolerance"])
            motion_bad = motion_error > float(criteria["mv_tolerance_pixels"])
            region["color_mismatches"] += int(color_bad)
            region["depth_mismatches"] += int(depth_bad)
            region["motion_mismatches"] += int(motion_bad)
            region["samples"].append({
                "pixel": [x, y],
                "geometry_pixel": [geometry_x, geometry_y],
                "expected_bgra": expected_bgra.tolist(),
                "actual_bgra": actual_bgra.tolist(),
                "color_error": color_error,
                "depth_expected": depth_buffer,
                "depth_actual": float(depth[geometry_y, geometry_x]),
                "depth_error": depth_error,
                "motion_expected": expected_motion.tolist() if depth_source else None,
                "motion_actual": motion[geometry_y, geometry_x].tolist(),
                "motion_error": motion_error if math.isfinite(motion_error) else None,
                "depth_source": depth_source,
                "pass": not (color_bad or depth_bad or motion_bad),
            })
        region["sample_count"] = len(region["samples"])
        total_color_errors += region["color_mismatches"]
        total_depth_errors += region["depth_mismatches"]
        total_motion_errors += region["motion_mismatches"]
        region_evidence[kind] = region

    centers = {}
    for name, panel in panels.items():
        if not panel["transparent"]:
            continue
        corners = projections[name]["pixels"]
        if not np.isfinite(corners).all():
            continue
        centers[name] = [float(corners[:, 0].mean()), float(corners[:, 1].mean())]

    checks = [
        _check("metadata_present", True,
               "valid scene_extension, objects and premultiplied materials"),
        _check("geometry_policy", True, "opaque-only depth/motion policy"),
        _check("blend_and_sorting",
               total_color_errors == 0 and all(
                   region["color_mismatches"] == 0 for region in region_evidence.values()),
               "premultiplied composite matches every sampled interior"),
        _check("opaque_depth_motion",
               total_depth_errors == 0 and total_motion_errors == 0,
               "depth/motion match frontmost opaque geometry only"),
    ]
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    reason = None if status == "pass" else (
        "transparency blend/sort mismatch" if total_color_errors else
        "transparency depth/motion mismatch")
    return {
        "kind": "transparency",
        "status": status,
        "reason": reason,
        "checks": checks,
        "regions": region_evidence,
        "region_counts": region_counts,
        "region_area_fractions": {kind: region_evidence[kind]["area_fraction"]
                                   for kind in region_evidence},
        "sample_count": sum(region["sample_count"] for region in region_evidence.values()),
        "color_mismatches": total_color_errors,
        "depth_mismatches": total_depth_errors,
        "motion_mismatches": total_motion_errors,
        "transparent_centers": centers,
        "draw_order": draw_order,
    }


def _check(name, passed, threshold):
    return {"name": name, "status": "pass" if passed else "fail",
            "observed": bool(passed), "threshold": threshold}
