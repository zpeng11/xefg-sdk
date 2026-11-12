import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("render_inputs", Path(__file__).parents[1] / "check_render_inputs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def write_run(root, specs, scene_name="translation", warmup=0, mode=3,
              pairing_method="sequential", legacy=False):
    root = Path(root)
    inputs = root / "application_inputs"
    inputs.mkdir(parents=True)
    first = np.asarray(specs[0]["depth"])
    height, width = first.shape
    manifest_frames = []
    scene_frames = []
    for order, item in enumerate(specs):
        index = int(item.get("index", order))
        depth = np.ascontiguousarray(np.asarray(item["depth"], dtype="<f4"))
        if "motion" in item:
            motion = np.ascontiguousarray(np.asarray(item["motion"], dtype="<f2"))
        else:
            motion = np.zeros((height, width, 2), dtype="<f2")
        depth_name = "depth_%05d.r32f" % index
        motion_name = "mv_%05d.rg16f" % index
        depth.reshape(-1).tofile(inputs / depth_name)
        motion.reshape(-1).tofile(inputs / motion_name)
        manifest_frames.append({"index": index,
                                "inputs": {"depth": depth_name, "motion": motion_name}})
        record = {"index": index, "previous_frame": None if index == 0 else index - 1,
                  "reset_history": bool(item.get("reset_history", False)),
                  "midpoint": "native_mid/frame_%05d.bgra" % index,
                  "time": float(index), "previous_time": float(max(index - 1, 0))}
        if "scene_frame" in item:
            record["scene_frame"] = int(item["scene_frame"])
        scene_frames.append(record)
    manifest = {"schema": "xefg.application_inputs/1", "width": width * 2, "height": height * 2,
                "motion_width": width, "motion_height": height,
                "frame_count": len(manifest_frames), "first_frame": 0,
                "last_frame": len(manifest_frames) - 1, "includes_warmup": True,
                "warmup": warmup, "frames": manifest_frames}
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    if legacy:
        scene = {"schema": "xefg.render_scene/1"}
    else:
        scene = {"schema": "xefg.render_scene/1", "name": scene_name, "revision": 1,
                 "mode": mode, "pairing_method": pairing_method,
                 "width": width * 2, "height": height * 2,
                 "motion_width": width, "motion_height": height, "warmup": warmup,
                 "measured_frames": len(manifest_frames) - warmup, "frames": scene_frames}
    (root / "scene.json").write_text(json.dumps(scene))
    return root


def depth_layers_depth():
    depth = np.full((10, 10), 0.95, dtype=np.float32)
    flat = depth.reshape(-1)
    flat[:3] = 0.5
    flat[3:5] = 0.995
    return depth


def screen_cross_frames():
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    positive = np.zeros((10, 10, 2), dtype=np.float32)
    positive[:, :, 0] = 0.9
    negative = np.zeros((10, 10, 2), dtype=np.float32)
    negative[1:-1, 1:-1, 0] = -0.9
    return [{"depth": depth, "motion": positive, "scene_frame": 4},
            {"depth": depth, "motion": negative, "scene_frame": 5}]


def camera_turn_frames():
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    depth.reshape(-1)[50:] = 0.995
    depth.reshape(-1)[50:60] = 0.98
    motion = np.zeros((10, 10, 2), dtype=np.float32)
    flat = motion.reshape(-1, 2)
    flat[:50, 0] = 0.1
    flat[50:60, 0] = 100.0
    flat[60:, 0] = 0.5
    zero = np.zeros((10, 10, 2), dtype=np.float32)
    return [{"depth": depth, "motion": motion, "scene_frame": 4},
            {"depth": depth, "motion": zero, "scene_frame": 5}]


class LegacyFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "application_inputs").mkdir()
        (self.root / "scene.json").write_text(json.dumps({"schema": "xefg.render_scene/1"}))
        manifest = dict(motion_width=2, motion_height=1,
                        frames=[dict(index=0, inputs=dict(depth="depth.raw", motion="mv.raw"))])
        (self.root / "application_inputs/manifest.json").write_text(json.dumps(manifest))
        self.depth = self.root / "application_inputs/depth.raw"
        np.array([0.5, 1], "<f4").tofile(self.depth)
        np.zeros((1, 2, 2), "<f2").tofile(self.root / "application_inputs/mv.raw")

    def test_normal_device_depth(self):
        report = checker.inspect_inputs(self.root)
        self.assertEqual(report["schema"], "xefg.render_input_check/1")
        self.assertNotIn("coverage", report)
        self.assertEqual(report["frames"][0]["geometry_pixels"], 1)
        self.assertEqual(report["frames"][0]["moving_pixels"], 0)

    def test_depth_not_reinterpreted_as_linear(self):
        np.array([1, 100], "<f4").tofile(self.depth)
        with self.assertRaisesRegex(ValueError, "device-depth"):
            checker.inspect_inputs(self.root)

    def test_nan_rejected(self):
        np.array([float("nan"), 1], "<f4").tofile(self.depth)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            checker.inspect_inputs(self.root)


class RenderInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_foreground_depth_and_motion_statistics(self):
        depth = np.array([[0.2, 0.6, 0.95], [0.999, 1.0, 1.0]], dtype=np.float32)
        motion = np.array([[[0.02, 0.0], [-0.02, 0.0], [0.5, 0.002]],
                           [[3.0, 2.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
        write_run(self.root, [{"depth": depth, "motion": motion}], scene_name="translation")
        row = checker.inspect_inputs(self.root)["frames"][0]
        self.assertEqual((row["source_frame"], row["measured"], row["pixels"]), (0, True, 6))
        foreground = row["foreground_depth"]
        self.assertEqual(foreground["pixels"], 4)
        self.assertAlmostEqual(foreground["fraction"], 4 / 6)
        self.assertEqual((foreground["below_0.9_pixels"], foreground["above_0.99_pixels"]), (2, 1))
        self.assertEqual(foreground["histogram"]["counts"], [1, 1, 1, 0, 1])
        self.assertEqual(foreground["histogram"]["edges"], [0.0, 0.5, 0.9, 0.97, 0.995, 1.0])
        self.assertEqual(len(foreground["quantiles"]), 6)
        self.assertAlmostEqual(foreground["quantiles"][0], 0.2)
        self.assertAlmostEqual(foreground["quantiles"][-1], 0.999)
        self.assertEqual(row["background_depth"]["pixels"], 2)
        self.assertAlmostEqual(row["background_depth"]["fraction"], 2 / 6)
        normalized = row["motion_normalized"]
        self.assertEqual(normalized["histogram"]["edges"][-1], "infinity")
        self.assertEqual(normalized["histogram"]["counts"], [2, 0, 2, 0, 2])
        self.assertAlmostEqual(normalized["above_0.02_fraction"], 2 / 6)
        self.assertAlmostEqual(normalized["above_0.08_fraction"], 2 / 6)
        self.assertAlmostEqual(row["motion_sign"]["x"]["positive"], 3 / 6)
        self.assertAlmostEqual(row["motion_sign"]["x"]["negative"], 1 / 6)
        self.assertAlmostEqual(row["motion_sign"]["y"]["positive"], 1 / 6)
        self.assertEqual(row["clip_limit"]["x_pixels"], 1)
        self.assertEqual(row["clip_limit"]["y_pixels"], 1)
        self.assertEqual(row["clip_limit"]["any_pixels"], 1)
        self.assertEqual(row["edge_occupancy"]["geometry"]["band"], "one_pixel_border")
        self.assertAlmostEqual(row["edge_occupancy"]["geometry"]["any"], 4 / 6)
        self.assertAlmostEqual(row["edge_occupancy"]["moving"]["any"], 4 / 6)
        self.assertAlmostEqual(row["edge_occupancy"]["moving_geometry"]["any"], 4 / 6)
        self.assertAlmostEqual(row["foreground_motion"]["above_0.005_fraction"], 1.0)
        self.assertAlmostEqual(row["foreground_motion"]["above_0.08_fraction"], 0.5)
        self.assertEqual(row["depth_bands"]["near"]["pixels"], 3)
        self.assertEqual(row["depth_bands"]["far"]["pixels"], 1)
        self.assertAlmostEqual(row["depth_bands"]["near"]["normalized_magnitude_mean"],
                               0.06, delta=1e-5)
        self.assertAlmostEqual(row["depth_bands"]["far"]["normalized_magnitude_mean"],
                               float(np.hypot(1.0, 1.0)), delta=1e-5)

    def test_edge_moving_ignores_sub_epsilon_noise(self):
        depth = np.full((4, 4), 0.5, dtype=np.float32)
        noise = np.full((4, 4, 2), 0.001, dtype=np.float32)
        write_run(self.root, [{"depth": depth, "motion": noise}], scene_name="translation")
        row = checker.inspect_inputs(self.root)["frames"][0]
        self.assertEqual(row["moving_pixels"], 16)
        self.assertEqual(row["edge_occupancy"]["moving"]["any"], 0.0)
        self.assertEqual(row["edge_occupancy"]["moving_geometry"]["any"], 0.0)
        self.assertEqual(row["edge_occupancy"]["motion_epsilon_pixels"], 0.01)

    def test_sequential_warmup_and_reset_are_not_measured(self):
        depth = np.full((4, 4), 0.5, dtype=np.float32)
        write_run(self.root, [
            {"depth": depth, "scene_frame": 0, "reset_history": True},
            {"depth": depth, "scene_frame": 5},
            {"depth": depth, "scene_frame": 6, "reset_history": True}],
            scene_name="translation", warmup=1)
        report = checker.inspect_inputs(self.root)
        self.assertEqual([row["source_frame"] for row in report["frames"]], [0, 5, 6])
        self.assertEqual([row["measured"] for row in report["frames"]], [False, True, False])
        self.assertEqual(report["coverage"]["status"], "not_applicable")

    def test_missing_scene_frame_falls_back_for_legacy_sequential(self):
        depth = np.full((4, 4), 0.5, dtype=np.float32)
        write_run(self.root, [{"depth": depth}, {"depth": depth}], scene_name="translation")
        rows = checker.inspect_inputs(self.root)["frames"]
        self.assertEqual([row["source_frame"] for row in rows], [0, 1])
        self.assertTrue(all(row["measured"] for row in rows))

    def test_isolated_pair_uses_local_one_source(self):
        depth = np.full((4, 4), 0.5, dtype=np.float32)
        write_run(self.root, [
            {"depth": depth, "scene_frame": 7, "reset_history": True},
            {"depth": depth, "scene_frame": 8}],
            scene_name="depth_layers", mode=1, pairing_method="isolated_pair")
        report = checker.inspect_inputs(self.root)
        self.assertEqual([row["source_frame"] for row in report["frames"]], [7, 8])
        self.assertEqual([row["measured"] for row in report["frames"]], [False, True])
        self.assertTrue(all(row["isolated"] for row in report["frames"]))
        self.assertEqual(report["coverage"]["status"], "pending")
        self.assertIn("aggregate", report["coverage"]["reason"])

    def test_depth_layers_coverage_pass_and_fail(self):
        depth = depth_layers_depth()
        write_run(self.root, [{"depth": depth, "scene_frame": 4},
                              {"depth": depth, "scene_frame": 5}], scene_name="depth_layers")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "pass")
        self.assertEqual(coverage["observed"]["foreground_pixels"], 200)
        self.assertAlmostEqual(coverage["observed"]["near_fraction"], 0.03)
        self.assertAlmostEqual(coverage["observed"]["far_fraction"], 0.02)
        flat = np.full((10, 10), 0.95, dtype=np.float32)
        failed = Path(self.temp.name) / "failed"
        write_run(failed, [{"depth": flat, "scene_frame": 4}], scene_name="depth_layers")
        self.assertEqual(checker.inspect_inputs(failed)["coverage"]["status"], "fail")

    def test_screen_cross_coverage_pass(self):
        write_run(self.root, screen_cross_frames(), scene_name="screen_cross")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "pass")
        self.assertEqual(coverage["observed"]["positive_dominant_source_frames"], [4])
        self.assertEqual(coverage["observed"]["negative_dominant_source_frames"], [5])
        self.assertGreaterEqual(coverage["observed"]["above_0.08_fractions"][0], 0.01)
        self.assertGreaterEqual(coverage["observed"]["moving_geometry_left_fractions"][0], 0.01)
        self.assertGreaterEqual(coverage["observed"]["moving_geometry_right_fractions"][0], 0.01)
        self.assertGreaterEqual(coverage["observed"]["edge_moving_range"], 0.01)

    def test_screen_cross_requires_large_motion_above_old_translation(self):
        depth = np.full((10, 10), 0.5, dtype=np.float32)
        positive = np.zeros((10, 10, 2), dtype=np.float32)
        positive[:, :, 0] = 0.5
        negative = np.zeros((10, 10, 2), dtype=np.float32)
        negative[1:-1, 1:-1, 0] = -0.5
        write_run(self.root, [{"depth": depth, "motion": positive, "scene_frame": 4},
                              {"depth": depth, "motion": negative, "scene_frame": 5}],
                  scene_name="screen_cross")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertTrue(coverage["observed"]["magnitude_ok"])
        self.assertFalse(coverage["observed"]["large_magnitude_ok"])
        self.assertEqual([check["status"] for check in coverage["checks"]
                          if check["name"] == "large_normalized_magnitude_above_0.08"], ["fail"])

    def test_screen_cross_requires_both_moving_edges(self):
        depth = np.full((10, 10), 0.5, dtype=np.float32)
        positive = np.zeros((10, 10, 2), dtype=np.float32)
        positive[:, :5, 0] = 0.9
        negative = np.zeros((10, 10, 2), dtype=np.float32)
        negative[2:-2, 2:-2, 0] = -0.9
        write_run(self.root, [{"depth": depth, "motion": positive, "scene_frame": 4},
                              {"depth": depth, "motion": negative, "scene_frame": 5}],
                  scene_name="screen_cross")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertGreaterEqual(coverage["observed"]["moving_geometry_left_fractions"][0], 0.01)
        self.assertLess(coverage["observed"]["moving_geometry_right_fractions"][0], 0.01)

    def test_screen_cross_static_foreground_edges_do_not_count(self):
        depth = np.full((10, 10), 0.5, dtype=np.float32)
        positive = np.zeros((10, 10, 2), dtype=np.float32)
        positive[1:-1, 1:-1, 0] = 0.9
        negative = np.zeros((10, 10, 2), dtype=np.float32)
        negative[2:-2, 2:-2, 0] = -0.9
        write_run(self.root, [{"depth": depth, "motion": positive, "scene_frame": 4},
                              {"depth": depth, "motion": negative, "scene_frame": 5}],
                  scene_name="screen_cross")
        report = checker.inspect_inputs(self.root)
        coverage = report["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertEqual(report["frames"][0]["edge_occupancy"]["geometry"]["left"], 1.0)
        self.assertEqual(report["frames"][0]["edge_occupancy"]["moving_geometry"]["left"], 0.0)
        self.assertFalse(coverage["observed"]["moving_geometry_ok"])
        self.assertFalse(coverage["observed"]["edge_moving_ok"])

    def test_screen_cross_single_frame_is_pending(self):
        write_run(self.root, screen_cross_frames()[:1], scene_name="screen_cross")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "pending")
        self.assertIn("sequence", coverage["reason"])

    def test_camera_turn_coverage_near_far(self):
        write_run(self.root, camera_turn_frames(), scene_name="camera_turn")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "pass")
        self.assertEqual(coverage["observed"]["near_pixels"], 100)
        self.assertEqual(coverage["observed"]["far_pixels"], 80)
        self.assertAlmostEqual(coverage["observed"]["near_mean_normalized_magnitude"],
                               0.005, delta=1e-5)
        self.assertAlmostEqual(coverage["observed"]["far_mean_normalized_magnitude"],
                               0.025, delta=1e-5)
        self.assertAlmostEqual(coverage["observed"]["mean_difference"], 0.02, delta=1e-4)

    def test_camera_turn_requires_differing_near_far_means(self):
        depth = np.full((10, 10), 0.5, dtype=np.float32)
        depth.reshape(-1)[50:] = 0.995
        motion = np.zeros((10, 10, 2), dtype=np.float32)
        motion[:, :, 0] = 0.1
        zero = np.zeros((10, 10, 2), dtype=np.float32)
        write_run(self.root, [{"depth": depth, "motion": motion, "scene_frame": 4},
                              {"depth": depth, "motion": zero, "scene_frame": 5}],
                  scene_name="camera_turn")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertTrue(coverage["observed"]["broad_ok"])
        self.assertTrue(coverage["observed"]["separated_ok"])
        self.assertFalse(coverage["observed"]["differing_ok"])

    def test_aggregate_coverage_filters_measured_and_combines_isolated_reports(self):
        depth = np.full((10, 10), 0.5, dtype=np.float32)
        positive = np.zeros((10, 10, 2), dtype=np.float32)
        positive[:, :, 0] = 0.9
        negative = np.zeros((10, 10, 2), dtype=np.float32)
        negative[1:-1, 1:-1, 0] = -0.9
        first = Path(self.temp.name) / "first"
        second = Path(self.temp.name) / "second"
        for target, previous, current, motion in ((first, 3, 4, positive),
                                                  (second, 5, 6, negative)):
            write_run(target, [
                {"depth": depth, "motion": np.zeros_like(motion),
                 "scene_frame": previous, "reset_history": True},
                {"depth": depth, "motion": motion, "scene_frame": current}],
                scene_name="screen_cross", mode=1, pairing_method="isolated_pair")
        rows = checker.inspect_inputs(first)["frames"] + checker.inspect_inputs(second)["frames"]
        coverage = checker.aggregate_coverage("screen_cross", rows)
        self.assertEqual(coverage["status"], "pass")
        self.assertEqual(coverage["measured_frame_count"], 2)
        self.assertEqual(coverage["measured_frames"], [4, 6])
        self.assertEqual(coverage["observed"]["positive_dominant_source_frames"], [4])
        self.assertEqual(coverage["observed"]["negative_dominant_source_frames"], [6])

    def test_cli_sequential_coverage_fail_exit3_and_isolated_pending_exit0(self):
        flat = np.full((10, 10), 0.95, dtype=np.float32)
        write_run(self.root / "sequential", [{"depth": flat, "scene_frame": 4}],
                  scene_name="depth_layers")
        report_path = self.root / "sequential.json"
        self.assertEqual(checker.main(["--run", str(self.root / "sequential"),
                                       "--out", str(report_path)]), 3)
        self.assertEqual(json.loads(report_path.read_text())["coverage"]["status"], "fail")
        depth = depth_layers_depth()
        write_run(self.root / "isolated", [
            {"depth": depth, "scene_frame": 7, "reset_history": True},
            {"depth": depth, "scene_frame": 8}],
            scene_name="depth_layers", mode=1, pairing_method="isolated_pair")
        isolated_path = self.root / "isolated.json"
        self.assertEqual(checker.main(["--run", str(self.root / "isolated"),
                                       "--out", str(isolated_path)]), 0)
        self.assertEqual(json.loads(isolated_path.read_text())["coverage"]["status"], "pending")


NONRIGID_PARAMETERS = {
    "A1": 0.25, "kx": 2.1, "ky": 1.6, "w1": 1.2, "w2": 0.75,
    "A2": 0.05, "k2": 1.3, "w3": 0.9, "phi2": 0.4,
}


def _scale_translate(sx, sy, sz, tx, ty, tz):
    matrix = np.eye(4)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = sx, sy, sz
    matrix[3, :3] = [tx, ty, tz]
    return matrix


def _perspective(fov_degrees, aspect, near=0.1, far=100.0):
    f = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, -far / (far - near), -1.0],
        [0.0, 0.0, -(far * near) / (far - near), 0.0],
    ])


def _deformed_grid(cells_x, cells_y, parameters, time):
    columns = cells_x + 1
    xs = -1.0 + 2.0 * np.arange(columns) / cells_x
    ys = -1.0 + 2.0 * np.arange(cells_y + 1) / cells_y
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = (parameters["A1"] * np.sin(parameters["kx"] * grid_x + parameters["w1"] * time)
         * np.cos(parameters["ky"] * grid_y + parameters["w2"] * time)
         + parameters["A2"] * np.sin(parameters["k2"] * (grid_x + grid_y)
                                      + parameters["w3"] * time + parameters["phi2"]))
    return np.stack([grid_x, grid_y, z], axis=-1).reshape(-1, 3)


def _grid_triangles(cells_x, cells_y):
    columns = cells_x + 1
    rows, cols = np.meshgrid(np.arange(cells_y), np.arange(cells_x), indexing="ij")
    a = (rows * columns + cols).reshape(-1)
    b, c, d = a + 1, a + columns, a + columns + 1
    return np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)])


def _barycentric(point, a, b, c):
    v0, v1, v2 = b - a, c - a, point - a
    denominator = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denominator) < 1.0e-12:
        return None
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / denominator
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / denominator
    return np.array([1.0 - u - v, u, v])


def _clip_points(points, matrix):
    return np.concatenate([points, np.ones((len(points), 1))], axis=1) @ matrix


def _transform_points(points, matrix):
    return _clip_points(points, matrix)[:, :3]


def _raster_geometry(local_current, local_previous, triangles, world, previous_world,
                     view, projection, previous_view, previous_projection, width, height):
    depth = np.ones((height, width), dtype=np.float64)
    motion = np.zeros((height, width, 2), dtype=np.float64)
    current_clip = _clip_points(_transform_points(local_current, world), view @ projection)
    previous_clip = _clip_points(_transform_points(local_previous, previous_world),
                                 previous_view @ previous_projection)
    with np.errstate(divide="ignore", invalid="ignore"):
        current_pixels = ((current_clip[:, :2] / current_clip[:, 3, None])
                          * np.array([0.5, -0.5]) + 0.5) * np.array([width, height])
    for triangle in triangles:
        if current_clip[triangle, 3].min() <= 0.05:
            continue
        pixel_triangle = current_pixels[triangle]
        minimum = np.floor(pixel_triangle.min(axis=0)).astype(int) - 1
        maximum = np.ceil(pixel_triangle.max(axis=0)).astype(int) + 1
        for y in range(max(0, minimum[1]), min(height, maximum[1] + 1)):
            for x in range(max(0, minimum[0]), min(width, maximum[0] + 1)):
                center = np.array([x + 0.5, y + 0.5])
                bary = _barycentric(center, *pixel_triangle)
                if bary is None or bary.min() < -1.0e-9:
                    continue
                inverse_w = 1.0 / current_clip[triangle, 3]
                denominator = float(bary @ inverse_w)
                current_depth = float(bary @ (current_clip[triangle, 2]
                                              / current_clip[triangle, 3]))
                previous_xy = ((bary @ (previous_clip[triangle, :2]
                                        * inverse_w[:, None])) / denominator)
                previous_w = float((bary @ (previous_clip[triangle, 3] * inverse_w))
                                   / denominator)
                if previous_w > 0.05:
                    previous_pixel = (((previous_xy / previous_w)
                                       * np.array([0.5, -0.5]) + 0.5)
                                      * np.array([width, height]))
                else:
                    previous_pixel = center
                if current_depth < depth[y, x]:
                    depth[y, x] = current_depth
                    motion[y, x] = previous_pixel - center
    return depth.astype("<f4"), motion.astype("<f2")


def _write_manifest_and_scene(root, scene, frames, motion_width, motion_height):
    root = Path(root)
    inputs = root / "application_inputs"
    inputs.mkdir(parents=True)
    manifest_frames = []
    for frame in frames:
        index = frame["index"]
        depth_name = "depth_%05d.r32f" % index
        motion_name = "mv_%05d.rg16f" % index
        frame["depth"].astype("<f4").reshape(-1).tofile(inputs / depth_name)
        frame["motion"].astype("<f2").reshape(-1).tofile(inputs / motion_name)
        manifest_frame = {"index": index,
                          "inputs": {"depth": depth_name, "motion": motion_name}}
        if "color" in frame:
            color_name = "color_%05d.bgra" % index
            frame["color"].astype(np.uint8).reshape(-1).tofile(inputs / color_name)
            manifest_frame["inputs"]["color"] = color_name
        manifest_frames.append(manifest_frame)
    manifest = {
        "schema": "xefg.application_inputs/1",
        "width": motion_width * 2, "height": motion_height * 2,
        "motion_width": motion_width, "motion_height": motion_height,
        "frame_count": len(frames), "first_frame": 0,
        "last_frame": len(frames) - 1, "includes_warmup": True, "warmup": 0,
        "frames": manifest_frames,
    }
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    (root / "scene.json").write_text(json.dumps(scene))
    return root


def _nonrigid_run(root, actual_previous="deformed", metadata_previous="deformed",
                  zero_deformation=False, vary_view=False, missing_extension=False,
                  missing_deformation=None):
    motion_width, motion_height = 64, 36
    times = (0, 1, 2)
    parameters = dict(NONRIGID_PARAMETERS)
    if zero_deformation:
        parameters["A1"] = 0.0
        parameters["A2"] = 0.0
    cells_x, cells_y = 8, 6
    world = _scale_translate(2.7, 1.5, 1.0, 0.0, 0.0, -4.2)
    base_view = np.eye(4)
    projection = _perspective(50.0, motion_width / motion_height)
    local = {time: _deformed_grid(cells_x, cells_y, parameters, time) for time in times}
    triangles = _grid_triangles(cells_x, cells_y)
    frames = []
    records = []
    for order, time in enumerate(times):
        previous_time = time - 1 if order else time
        reset = order == 0
        current = local[time]
        actual_prev = (current if reset or actual_previous == "current"
                       else local[previous_time])
        view = base_view
        if vary_view and order == len(times) - 1:
            view = _scale_translate(1.0, 1.0, 1.0, 0.10, 0.0, 0.0)
        depth, motion = _raster_geometry(current, actual_prev, triangles, world, world,
                                         view, projection, view, projection,
                                         motion_width, motion_height)
        index = order
        frame = {"index": index, "depth": depth, "motion": motion}
        frames.append(frame)
        record = {
            "index": index, "scene_frame": index,
            "previous_frame": None if index == 0 else index - 1,
            "reset_history": reset, "midpoint": "native_mid/frame_%05d.bgra" % index,
            "time": float(time), "previous_time": float(previous_time),
            "view": view.reshape(-1).tolist(),
            "projection": projection.reshape(-1).tolist(),
            "objects": [{
                "name": "grid", "mesh": "deformed_grid", "role": "deformed_grid",
                "world": world.reshape(-1).tolist(),
                "previous_world": world.reshape(-1).tolist(),
            }],
            "deformation": {
                "time": float(time), "previous_time": float(previous_time),
                "previous_is_current": bool(reset or metadata_previous == "current"),
            },
        }
        if missing_deformation == index:
            record.pop("deformation")
        records.append(record)
    scene = {
        "schema": "xefg.render_scene/1", "name": "nonrigid", "revision": 1,
        "mode": 3, "pairing_method": "sequential",
        "width": motion_width * 2, "height": motion_height * 2,
        "motion_width": motion_width, "motion_height": motion_height,
        "warmup": 0, "measured_frames": len(frames), "frames": records,
    }
    if not missing_extension:
        scene["scene_extension"] = {
            "schema": "xefg.render_scene_extension/1", "version": 1,
            "scene": "nonrigid",
            "grid": {"cells_x": cells_x, "cells_y": cells_y},
            "deformation": parameters,
            "policy": {"depth_motion": "opaque_only",
                       "transparent_depth_write": False,
                       "transparent_writes_mv": False},
        }
    return _write_manifest_and_scene(root, scene, frames, motion_width, motion_height)


def _panel_world(scale, translation):
    return _scale_translate(scale[0], scale[1], 1.0, translation[0], translation[1],
                            translation[2])


def _transparency_objects(static_panels, time):
    far_x = -0.1 + (0.0 if static_panels else 0.06 * math.sin(0.9 * time))
    near_x = -0.1 + (0.0 if static_panels else -0.045 * math.sin(0.9 * time + 0.5))
    return [
        {"name": "backdrop", "role": "opaque_backdrop", "transparent": False,
         "rgba": (0.10, 0.15, 0.24, 1.0), "scale": (1.4, 1.4), "translation": (0.0, 0.0, 0.9)},
        {"name": "blocker", "role": "opaque_blocker", "transparent": False,
         "rgba": (0.20, 0.24, 0.26, 1.0), "scale": (0.5, 0.4), "translation": (0.35, -0.1, 0.5)},
        {"name": "far", "role": "transparent_far", "transparent": True,
         "rgba": (0.80, 0.22, 0.18, 0.55), "scale": (0.7, 0.35),
         "translation": (far_x, 0.05, 0.7)},
        {"name": "near", "role": "transparent_near", "transparent": True,
         "rgba": (0.16, 0.60, 0.78, 0.45), "scale": (0.4, 0.25),
         "translation": (near_x, 0.02, 0.6)},
    ]


def _panel_coverage(matrix, point_ndc):
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]) @ matrix[:2, :2]
    corners = corners + matrix[3, :2]
    for indices in ((0, 1, 3), (0, 3, 2)):
        bary = _barycentric(point_ndc, *corners[list(indices)])
        if bary is not None and bary.min() >= 0.0:
            return True
    return False


def _quantize(values):
    return np.clip(np.floor(np.asarray(values, dtype=np.float64) * 255.0 + 0.5), 0.0, 255.0)


def _render_transparency_color(objects, draw_order, width, height, blend=True):
    image = np.zeros((height, width, 4), dtype=np.uint8)
    prepared = []
    for item in objects:
        rgba = np.asarray(item["rgba"], dtype=np.float64)
        premultiplied = np.array([rgba[0] * rgba[3], rgba[1] * rgba[3],
                                  rgba[2] * rgba[3], rgba[3]])
        prepared.append((item, _panel_world(item["scale"], item["translation"]),
                         premultiplied))
    for y in range(height):
        for x in range(width):
            point = np.array([(x + 0.5) / width * 2.0 - 1.0,
                              1.0 - (y + 0.5) / height * 2.0])
            destination = np.array([0.0, 0.0, 0.0, 1.0])
            depth_buffer = 1.0
            for name in draw_order:
                item, world, premultiplied = next(entry for entry in prepared
                                                  if entry[0]["name"] == name)
                if not _panel_coverage(world, point):
                    continue
                depth = float(world[3, 2])
                if depth > depth_buffer:
                    continue
                if item["transparent"]:
                    if blend:
                        alpha = float(premultiplied[3])
                        destination[:3] = premultiplied[:3] + destination[:3] * (1.0 - alpha)
                        destination[3] = alpha + destination[3] * (1.0 - alpha)
                        stored = _quantize(destination)
                        destination = stored / 255.0
                    else:
                        stored = _quantize([premultiplied[0], premultiplied[1],
                                            premultiplied[2], 1.0])
                        destination = stored / 255.0
                else:
                    stored = _quantize([premultiplied[0], premultiplied[1],
                                        premultiplied[2], 1.0])
                    destination = stored / 255.0
                    depth_buffer = depth
            image[y, x] = [int(destination[2] * 255.0), int(destination[1] * 255.0),
                           int(destination[0] * 255.0), int(destination[3] * 255.0)]
    return image


def _render_transparency_geometry(objects, width, height, include_transparent):
    depth = np.ones((height, width), dtype=np.float64)
    for item in objects:
        if item["transparent"] and not include_transparent:
            continue
        world = _panel_world(item["scale"], item["translation"])
        for y in range(height):
            for x in range(width):
                point = np.array([(x + 0.5) / width * 2.0 - 1.0,
                                  1.0 - (y + 0.5) / height * 2.0])
                if _panel_coverage(world, point):
                    depth[y, x] = min(depth[y, x], float(world[3, 2]))
    return depth.astype("<f4"), np.zeros((height, width, 2), dtype="<f2")


def _transparency_run(root, static_panels=False, no_blend=False, order_override=None,
                      transparent_depth=False, alpha_only=False, policy_error=False,
                      missing_extension=False):
    motion_width, motion_height = 64, 36
    width, height = motion_width * 2, motion_height * 2
    times = (0, 1, 2)
    frames = []
    records = []
    for index, time in enumerate(times):
        objects = _transparency_objects(static_panels, time)
        draw_order = ["backdrop", "blocker", "far", "near"]
        color = _render_transparency_color(objects, order_override or draw_order,
                                           width, height, blend=not no_blend)
        depth, motion = _render_transparency_geometry(objects, motion_width, motion_height,
                                                      transparent_depth)
        frames.append({"index": index, "depth": depth, "motion": motion, "color": color})
        object_records = []
        for sort_index, item in enumerate(objects):
            rgba = item["rgba"]
            record = {
                "name": item["name"], "role": item["role"],
                "transparent": item["transparent"],
                "flat_color_rgba": list(rgba),
                "premultiplied_rgba": [rgba[0] * rgba[3], rgba[1] * rgba[3],
                                       rgba[2] * rgba[3], rgba[3]],
                "depth_write": not item["transparent"],
                "writes_depth_motion": not item["transparent"],
                "sort_index": sort_index, "mesh": "panel",
                "world": _panel_world(item["scale"], item["translation"]).reshape(-1).tolist(),
                "previous_world": _panel_world(item["scale"],
                                               item["translation"]).reshape(-1).tolist(),
            }
            if alpha_only and item["name"] == "far":
                record.pop("premultiplied_rgba")
            if policy_error and item["name"] == "near":
                record["depth_write"] = True
                record["writes_depth_motion"] = True
            object_records.append(record)
        records.append({
            "index": index, "scene_frame": index,
            "previous_frame": None if index == 0 else index - 1,
            "reset_history": False, "midpoint": "native_mid/frame_%05d.bgra" % index,
            "time": float(time), "previous_time": float(max(time - 1, 0)),
            "view": np.eye(4).reshape(-1).tolist(),
            "projection": np.eye(4).reshape(-1).tolist(),
            "objects": object_records,
        })
    scene = {
        "schema": "xefg.render_scene/1", "name": "transparency", "revision": 1,
        "mode": 3, "pairing_method": "sequential",
        "width": width, "height": height,
        "motion_width": motion_width, "motion_height": motion_height,
        "warmup": 0, "measured_frames": len(frames), "frames": records,
    }
    if not missing_extension:
        scene["scene_extension"] = {
            "schema": "xefg.render_scene_extension/1", "version": 1,
            "scene": "transparency",
            "blend": {"mode": "premultiplied_alpha", "source": "premultiplied"},
            "policy": {"depth_motion": "opaque_only",
                       "transparent_depth_write": False,
                       "transparent_writes_mv": False},
            "draw_order": ["backdrop", "blocker", "far", "near"],
            "objects": [{"name": item["name"], "role": item["role"],
                         "transparent": item["transparent"]}
                        for item in _transparency_objects(False, 0.0)],
        }
    return _write_manifest_and_scene(root, scene, frames, motion_width, motion_height)


class NonrigidEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_nonrigid_deformed_grid_passes(self):
        _nonrigid_run(self.root)
        report = checker.inspect_inputs(self.root)
        coverage = report["coverage"]
        self.assertEqual(coverage["status"], "pass")
        evidence = [row["scene_evidence"] for row in report["frames"]
                    if row.get("measured")]
        self.assertEqual(len(evidence), 2)
        self.assertTrue(all(item["status"] == "pass" for item in evidence))
        self.assertGreaterEqual(evidence[0]["sample_count"], 8)
        self.assertGreaterEqual(evidence[0]["sampled_triangle_count"], 4)
        self.assertGreater(evidence[0]["mv_spatial_range"], 0.25)
        observed = coverage["observed"]
        self.assertGreaterEqual(observed["changed_pairs"], 4)

    def test_nonrigid_zero_previous_deformation_fails(self):
        _nonrigid_run(self.root, actual_previous="current")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        names = {check["name"] for check in coverage["checks"]
                 if check["status"] == "fail"}
        self.assertIn("samples_match_analysis", names)

    def test_nonrigid_broken_previous_metadata_fails(self):
        _nonrigid_run(self.root, metadata_previous="current")
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        names = {check["name"] for check in coverage["checks"]
                 if check["status"] == "fail"}
        self.assertIn("samples_match_analysis", names)

    def test_nonrigid_missing_frame_deformation_fails_closed(self):
        _nonrigid_run(self.root, missing_deformation=2)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("metadata_present",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_nonrigid_missing_extension_fails_closed(self):
        _nonrigid_run(self.root, missing_extension=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("metadata_present",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_nonrigid_static_view_violation_fails(self):
        _nonrigid_run(self.root, vary_view=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("static_world_view",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_nonrigid_without_deformation_fails(self):
        _nonrigid_run(self.root, zero_deformation=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        names = {check["name"] for check in coverage["checks"]
                 if check["status"] == "fail"}
        self.assertIn("vertex_pair_distances_change", names)

    def test_missing_evidence_cannot_pass(self):
        coverage = checker.aggregate_coverage(
            "nonrigid", [{"source_frame": 1, "measured": True, "frame": 1,
                          "isolated": True}])
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("metadata_present",
                      {check["name"] for check in coverage["checks"]})


class TransparencyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_transparency_premultiplied_regions_pass(self):
        _transparency_run(self.root)
        report = checker.inspect_inputs(self.root)
        coverage = report["coverage"]
        self.assertEqual(coverage["status"], "pass")
        observed = coverage["observed"]
        for kind in ("single", "double", "occluded"):
            self.assertGreaterEqual(observed[kind]["samples"], 9)
            self.assertEqual(observed[kind]["mismatches"], 0)
            self.assertGreaterEqual(observed[kind]["max_area_fraction"], 0.002)
        self.assertGreaterEqual(observed["temporal_movement_pixels"], 1.0)

    def test_transparency_no_blending_fails(self):
        _transparency_run(self.root, no_blend=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("blend_and_sorting",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_transparency_wrong_sorting_fails(self):
        _transparency_run(self.root,
                          order_override=["backdrop", "blocker", "near", "far"])
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("blend_and_sorting",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_transparency_transparent_depth_write_fails(self):
        _transparency_run(self.root, transparent_depth=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("opaque_depth_motion",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_transparency_alpha_only_metadata_fails(self):
        _transparency_run(self.root, alpha_only=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("metadata_present",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_transparency_geometry_policy_error_fails(self):
        _transparency_run(self.root, policy_error=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("metadata_present",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_transparency_static_panels_fail_temporal_movement(self):
        _transparency_run(self.root, static_panels=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("temporal_movement",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_transparency_missing_extension_fails_closed(self):
        _transparency_run(self.root, missing_extension=True)
        coverage = checker.inspect_inputs(self.root)["coverage"]
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("metadata_present",
                      {check["name"] for check in coverage["checks"]
                       if check["status"] == "fail"})

    def test_missing_transparency_evidence_cannot_pass(self):
        coverage = checker.aggregate_coverage(
            "transparency", [{"source_frame": 1, "measured": True, "frame": 1}])
        self.assertEqual(coverage["status"], "fail")
        self.assertIn("metadata_present",
                      {check["name"] for check in coverage["checks"]})


if __name__ == "__main__":
    unittest.main()