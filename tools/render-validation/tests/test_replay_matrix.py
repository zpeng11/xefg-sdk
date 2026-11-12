import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "replay_matrix", Path(__file__).parents[1] / "replay_matrix.py")
replay_matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_matrix)


def _matrix(path, rows, size_results=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    value = {"schema": "xefg.render_matrix/4", "scenes": rows}
    if size_results is not None:
        value["size_results"] = size_results
    (path / "matrix.json").write_text(json.dumps(value), encoding="utf-8")
    return path / "matrix.json"


def _row(scene, size, mode, cases, generated_frames=8):
    return {"scene": scene, "size": size, "mode": mode, "generated_frames":
            generated_frames, "cases": [{"directory": str(case)} for case in cases]}


def _package(path):
    path = Path(path)
    (path / "models").mkdir(parents=True)
    (path / "profiles").mkdir(parents=True)
    (path / "models/mode1.onnx").write_bytes(b"model")
    (path / "profiles/mode1-640x360.json").write_bytes(b"profile")
    manifest = {"variants": {"mode1": {
        "mode": 1, "model": "models/mode1.onnx",
        "profiles": ["profiles/mode1-640x360.json"]}}}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


class MappingTests(unittest.TestCase):
    def test_uses_scene_rows_and_case_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = [root / "run" / "640x360" / "mode1" / "translation"
                     / ("pair_%05d" % frame) for frame in range(4, 12)]
            matrix = _matrix(root / "source", [_row("translation", "640x360", 1, cases)],
                             [{"size": "640x360", "generated_frames": 8}])
            index, sources = replay_matrix.discover_source_index([matrix])
            key = replay_matrix.combo_key("translation", "640x360", 1)
            self.assertEqual(index[key]["cases"], [str(case.resolve()) for case in cases])
            self.assertEqual(sources[0]["size_results"][0]["generated_frames"], 8)

    def test_size_results_disagreeing_with_rows_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix = _matrix(root / "source",
                             [_row("translation", "640x360", 3, [root / "a"], 8)],
                             [{"size": "640x360", "generated_frames": 7}])
            with self.assertRaises(replay_matrix.PreflightError):
                replay_matrix.discover_source_index([matrix])

    def test_preflight_accepts_111_sources_and_24_known_renders(self):
        index = {}
        for key in replay_matrix.expected_combo_keys():
            scene, size, mode = replay_matrix.parse_combo_key(key)
            if key in replay_matrix.allowed_missing_keys():
                continue
            index[key] = {"scene": scene, "size": size, "mode": mode,
                          "cases": ["case-%d" % number for number in
                                    range(replay_matrix.expected_case_count(scene, mode))],
                          "matrix": "source"}
        report = replay_matrix.preflight(index)
        self.assertEqual(report["expected_combos"], 135)
        self.assertEqual(report["expected_frames"], 1065)
        self.assertEqual(report["found_combos"], 111)
        self.assertEqual(len(report["render_combos"]), 24)

    def test_duplicate_combo_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = _matrix(root / "left",
                           [_row("translation", "640x360", 3, [root / "left-case"])])
            right = _matrix(root / "right",
                            [_row("translation", "640x360", 3, [root / "right-case"])])
            with self.assertRaises(replay_matrix.DuplicateComboError):
                replay_matrix.discover_source_index([left, right])

    def test_repeating_the_same_source_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix = _matrix(root / "source",
                             [_row("translation", "640x360", 3, [root / "case"])])
            with self.assertRaises(replay_matrix.DuplicateComboError):
                replay_matrix.discover_source_index([matrix, matrix])


class ExpectedInventoryTests(unittest.TestCase):
    def test_cut_skips_source_frame_seven_in_every_mode(self):
        self.assertEqual(replay_matrix.expected_pair_frames("cut"),
                         [4, 5, 6, 8, 9, 10, 11])
        self.assertEqual(replay_matrix.expected_frames("cut"), 7)
        self.assertEqual(replay_matrix.expected_case_count("cut", 1), 7)
        self.assertEqual(replay_matrix.expected_case_count("cut", 3), 1)
        self.assertEqual(replay_matrix.expected_frames("translation"), 8)

    def test_full_inventory_is_135_combinations_and_1065_frames(self):
        self.assertEqual(len(replay_matrix.expected_combo_keys()), 135)
        self.assertEqual(replay_matrix.expected_total_frames(), 1065)
        self.assertEqual(len(replay_matrix.allowed_missing_keys()), 24)

    def test_unexpected_missing_combination_fails_preflight(self):
        with self.assertRaises(replay_matrix.PreflightError):
            replay_matrix.preflight({})


class ResumeTests(unittest.TestCase):
    def test_package_fingerprint_changes_with_package_content(self):
        with tempfile.TemporaryDirectory() as temp:
            package = _package(Path(temp) / "package")
            first = replay_matrix.package_identity(package)
            self.assertEqual(first["identity"],
                             replay_matrix.package_identity(package)["identity"])
            (package / "models/mode1.onnx").write_bytes(b"changed")
            second = replay_matrix.package_identity(package)
            self.assertNotEqual(first["identity"], second["identity"])

    def test_stale_package_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            package = _package(Path(temp) / "package")
            identity = replay_matrix.package_identity(package)
            state = {"schema": replay_matrix.STATE_SCHEMA, "package": identity,
                     "config_sha256": "config", "source_index_sha256": "source",
                     "combos": {}}
            self.assertEqual(replay_matrix.validate_resume(
                state, identity, "config", "source", False, True), "verified")
            changed = dict(identity)
            changed["identity"] = "different"
            with self.assertRaises(replay_matrix.StalePackageError):
                replay_matrix.validate_resume(state, changed, "config", "source",
                                              False, True)
            with self.assertRaises(replay_matrix.MatrixError):
                replay_matrix.validate_resume(state, identity, "other", "source",
                                              False, True)
            with self.assertRaises(replay_matrix.MatrixError):
                replay_matrix.validate_resume(state, identity, "config", None,
                                              False, False)

    def test_complete_resume_is_readable_without_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            package = _package(Path(temp) / "package")
            identity = replay_matrix.package_identity(package)
            key = replay_matrix.combo_key("translation", "640x360", 3)
            checkpoint_path = out / "results" / "combos" / (key + ".json")
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint = {"schema": replay_matrix.CHECKPOINT_SCHEMA, "key": key,
                          "status": "pass", "generated_frames": 8,
                          "package_identity": identity["identity"],
                          "config_sha256": "config"}
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            state = {"schema": replay_matrix.STATE_SCHEMA, "package": identity,
                     "config_sha256": "config", "source_index_sha256": "source",
                     "combos": {key: {"checkpoint": "results/combos/" + key + ".json",
                                      "complete": True}}}
            self.assertEqual(replay_matrix.validate_resume(
                state, identity, "config", None, True, False), "stored")
            done = replay_matrix.completed_checkpoints(
                out, state, [key], identity, "config")
            self.assertEqual(done[key]["status"], "pass")
            stale = dict(checkpoint)
            stale["package_identity"] = "old"
            checkpoint_path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaises(replay_matrix.StalePackageError):
                replay_matrix.completed_checkpoints(out, state, [key], identity, "config")


class ImageSelectionTests(unittest.TestCase):
    LIMITS = {"mae": 0.015, "maximum": 40, "fraction": 0.0015}

    @staticmethod
    def _pair(frame, mae, maximum, fraction):
        return {"current_frame": frame, "source_current_frame": frame,
                "valid_region": {"rgb_mae": mae, "rgb_max": maximum,
                                 "over_tolerance_pixel_fraction": fraction}}

    def test_selects_worst_frame_per_metric_and_distinct_cause(self):
        pairs = [self._pair(4, 0.02, 20, 0.001),
                 self._pair(5, 0.001, 50, 0.0001),
                 self._pair(6, 0.0001, 5, 0.005)]
        selected = replay_matrix.select_image_frames(pairs, self.LIMITS)
        by_frame = {item["source_frame"]: item["reasons"] for item in selected}
        self.assertIn("max_rgb_mae", by_frame[4])
        self.assertIn("max_rgb_max", by_frame[5])
        self.assertIn("max_over_2_fraction", by_frame[6])
        self.assertIn("cause:rgb_mae", by_frame[4])
        self.assertIn("cause:rgb_max", by_frame[5])
        self.assertIn("cause:over_2_fraction", by_frame[6])
        self.assertIn("cause:rgb_max_above_8", by_frame[5])

    def test_equal_frames_are_deduplicated(self):
        pairs = [self._pair(4, 0.02, 40, 0.001)]
        selected = replay_matrix.select_image_frames(pairs, self.LIMITS)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source_frame"], 4)
        self.assertIn("max_rgb_mae", selected[0]["reasons"])
        self.assertIn("max_rgb_max", selected[0]["reasons"])

    def test_empty_pairs_select_no_images(self):
        self.assertEqual(replay_matrix.select_image_frames([], self.LIMITS), [])

    def test_normalized_row_has_no_retained_artifact_pointer(self):
        row = replay_matrix.normalize_pair(
            {"current_frame": 4, "output": "frame_00004.npy",
             "valid_region": {"rgb_mae": 0}})
        self.assertNotIn("output", row)
        self.assertEqual(row["artifact_retention"]["state"], "compacted")


class CoverageAggregationTests(unittest.TestCase):
    def test_coverage_status_controls_combo_status(self):
        pairs = [{"source_current_frame": frame} for frame in range(4, 12)]
        cases = [{"case": "capture", "label": "capture", "status": "pass",
                  "pairs": pairs, "input_check": {"frames": []},
                  "quality": {"failures": [], "warnings": []}}]
        limits = {"mae": 0.015, "maximum": 40, "fraction": 0.0015}
        for coverage_status, expected in (("pending", "coverage_failed"),
                                          ("fail", "coverage_failed"),
                                          ("pass", "pass"),
                                          ("not_applicable", "pass")):
            with self.subTest(coverage_status=coverage_status):
                tool = mock.Mock()
                tool.aggregate_coverage.return_value = {
                    "status": coverage_status, "measured_frames": []}
                with mock.patch.object(replay_matrix, "_load_tool", return_value=tool), \
                        mock.patch.object(replay_matrix, "retain_images", return_value=[]):
                    result = replay_matrix.aggregate_combo(
                        "depth_layers__640x360__mode3", "depth_layers", "640x360",
                        3, "existing", cases, Path("unused"), limits)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["coverage"]["status"], coverage_status)
                self.assertEqual(result["quality"]["status"], "pass")


class InputEvidenceTests(unittest.TestCase):
    def test_fingerprints_cover_every_role_by_source_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            inputs = capture / "application_inputs"
            inputs.mkdir(parents=True)
            frames = []
            for index, source in ((0, 3), (1, 4)):
                names = {"color": "color_%05d.bgra" % index,
                         "ui": "ui_%05d.bgra" % index,
                         "backbuffer": "backbuffer_%05d.bgra" % index,
                         "depth": "depth_%05d.r32f" % index,
                         "motion": "mv_%05d.rg16f" % index}
                for name in names.values():
                    (inputs / name).write_bytes(b"x")
                frames.append({"index": index, "inputs": names})
            (inputs / "manifest.json").write_text(json.dumps({"frames": frames}),
                                                  encoding="utf-8")
            (capture / "scene.json").write_text(json.dumps({"frames": [
                {"index": 0, "scene_frame": 3}, {"index": 1, "scene_frame": 4}]}),
                encoding="utf-8")
            artifacts = [{"file": "application_inputs/" + name,
                          "sha256": "hash-" + name, "bytes": 1}
                         for name in frames[0]["inputs"].values()]
            pairing = {"artifacts": artifacts}
            rows = replay_matrix.input_fingerprints(capture, pairing, "pair_00004")
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["source_frame"] for row in rows}, {3, 4})
            self.assertEqual({row["role"] for row in rows},
                             {"color", "ui", "backbuffer", "depth", "motion"})
            self.assertTrue(all(row["source"] == "pairing_artifacts"
                                for row in rows if row["local_frame"] == 0))
            self.assertTrue(all(row["source"] == "recomputed"
                                for row in rows if row["local_frame"] == 1))

    def test_mode3_versus_mode4_equality_uses_compact_fingerprints(self):
        def combo(mode, hashes):
            return {"key": "translation__640x360__mode%d" % mode,
                    "scene": "translation", "size": "640x360", "mode": mode,
                    "status": "pass", "input_fingerprints": [
                        {"source_frame": 4, "role": role, "sha256": digest}
                        for role, digest in hashes.items()]}
        hashes = {"color": "c", "ui": "u", "backbuffer": "b",
                  "depth": "d", "motion": "m"}
        rows = replay_matrix._input_equality([combo(3, hashes), combo(4, hashes)])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ui_exact"])
        self.assertTrue(rows[0]["comparable_roles_equal"])
        different = dict(hashes, ui="other")
        rows = replay_matrix._input_equality(
            [combo(3, hashes), combo(4, different)])
        self.assertFalse(rows[0]["ui_exact"])
        self.assertFalse(rows[0]["comparable_roles_equal"])


class DeletionGuardTests(unittest.TestCase):
    def test_only_owned_work_is_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            source = Path(temp) / "source" / "capture"
            package = out / "package"
            work = out / "work" / "combo"
            for path in (source, package, work):
                path.mkdir(parents=True)
                (path / "file.txt").write_text("x", encoding="utf-8")
            replay_matrix.remove_owned(work, out, protected=[source, package])
            self.assertFalse(work.exists())
            self.assertTrue(source.exists())
            self.assertTrue(package.exists())

    def test_source_package_out_and_results_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            source = Path(temp) / "source"
            package = out / "package"
            results = out / "results"
            for path in (source, package, results):
                path.mkdir(parents=True)
            for path in (source, package, results, out):
                with self.assertRaises(replay_matrix.UnsafeDeleteError):
                    replay_matrix.remove_owned(path, out, protected=[source, package])

    def test_protected_path_inside_work_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            source = out / "work" / "combo" / "capture"
            source.mkdir(parents=True)
            with self.assertRaises(replay_matrix.UnsafeDeleteError):
                replay_matrix.remove_owned(out / "work" / "combo", out,
                                           protected=[source])


if __name__ == "__main__":
    unittest.main()
