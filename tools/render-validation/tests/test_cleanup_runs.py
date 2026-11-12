import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "cleanup_runs", Path(__file__).parents[1] / "cleanup_runs.py")
cleanup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cleanup
spec.loader.exec_module(cleanup)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scan = self.root / "runs"
        self.scan.mkdir()
        self.current = self.root / "current"
        self.current.mkdir()
        self.keep = self.scan / "xefg-protected"
        self.write(self.keep / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        self.package = self.current / "package"
        self.make_package()

    def write(self, path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (dict, list)):
            path.write_text(json.dumps(data), encoding="utf-8")
        elif isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(str(data), encoding="utf-8")
        return path

    def make_package(self):
        model = self.package / "models" / "mode3.onnx"
        profile = self.package / "profiles" / "mode3.json"
        self.write(model, b"toy onnx model")
        self.write(profile, {"schema_version": 1, "variant": "mode3"})
        manifest = {
            "schema_version": 1,
            "variants": {
                "mode3": {
                    "model": "models/mode3.onnx",
                    "model_sha256": file_sha256(model),
                    "profiles": ["profiles/mode3.json"],
                }
            },
        }
        self.write(self.package / "manifest.json", manifest)
        self.package_profile = {
            "variant": "mode3",
            "model_sha256": file_sha256(model),
            "profile_file": "profiles/mode3.json",
            "profile_sha256": file_sha256(profile),
        }

    def matrix(self, rows=135, frames=1065, status_cycle=("pass", "quality_failed", "coverage_failed"),
               special=None):
        special = special or {}
        base, remainder = divmod(frames, rows)
        scenes = []
        for index in range(rows):
            status = status_cycle[index % len(status_cycle)]
            scenes.append({
                "status": special.get(index, status),
                "completed": special.get(index, status) not in cleanup.BAD_MATRIX_STATUSES,
                "generated_frames": base + (1 if index < remainder else 0),
            })
        top_status = "run_failed" if any(row["status"] == "run_failed" for row in scenes) else "quality_failed"
        return {
            "schema": "xefg.render_matrix/4",
            "status": top_status,
            "package_profiles": [self.package_profile],
            "scenes": scenes,
        }

    def replay_matrix(self, frames=1065, special=None, nan_frame=False):
        special = special or {}
        base, remainder = divmod(frames, 135)
        combos = []
        for index in range(135):
            generated = base + (1 if index < remainder else 0)
            status = special.get(index, "complete" if index % 2 else "quality_failed")
            frame_rows = [{
                "index": index,
                "rgb_mae": 0.01,
                "rgb_max": 3,
                "over_tolerance_pixel_fraction": 0.0001,
            }]
            if nan_frame and index == 0:
                frame_rows[0]["rgb_mae"] = float("nan")
            combos.append({
                "scene": "translation",
                "size": "640x360",
                "mode": 3,
                "status": status,
                "expected_frames": generated,
                "generated_frames": generated,
                "coverage": {"status": "pass", "fraction": 1.0},
                "quality": {"status": "pass", "worst_mae": 0.01},
                "frames": frame_rows,
            })
        return {
            "schema": "xefg.replay_matrix/1",
            "status": "quality_failed",
            "config": {"source": "render-matrix"},
            "combos": combos,
            "summary": {
                "combos": 135,
                "expected_combos": 135,
                "frames": 1065,
                "expected_frames": 1065,
                "status_counts": {
                    "pass": sum(1 for combo in combos if combo["status"] == "pass"),
                    "quality_failed": sum(1 for combo in combos
                                          if combo["status"] == "quality_failed"),
                    "coverage_failed": sum(1 for combo in combos
                                           if combo["status"] == "coverage_failed"),
                    "run_failed": sum(1 for combo in combos
                                      if combo["status"] == "run_failed"),
                },
            },
        }

    def package_identity(self):
        manifest = json.loads((self.package / "manifest.json").read_text(encoding="utf-8"))
        files = {}
        for variant in manifest["variants"].values():
            for relative in [variant["model"], *variant.get("profiles", [])]:
                path = self.package / relative
                files[relative] = {"bytes": path.stat().st_size,
                                   "sha256": file_sha256(path)}
        return {
            "directory": str(self.package),
            "manifest_sha256": file_sha256(self.package / "manifest.json"),
            "files": files,
            "identity": "toy-package-identity",
        }

    def write_status(self, completed_combos=135, completed_frames=1065, status="complete"):
        return self.write(self.current / "results" / "status.json", {
            "schema": "xefg.replay_matrix_status/1",
            "status": status,
            "expected_combos": 135,
            "completed_combos": completed_combos,
            "expected_frames": 1065,
            "reported_frames": completed_frames,
            "counts": {"pass": 1, "quality_failed": 0, "coverage_failed": 0,
                       "run_failed": 0},
        })

    def write_matrix(self, data, name="matrix.json"):
        path = self.current / "results" / name
        self.write(path, data)
        return path

    def inventory(self, exact_paths=(), protected=()):
        scope = [cleanup.ScopeRule(str(self.scan), "xefg-*", "run")]
        for path in exact_paths:
            scope.append(cleanup.ScopeRule(str(path), "", "exact"))
        returns = cleanup.run_inventory(
            self.current, scope, list(exact_paths),
            [self.current, self.keep, *protected],
            force=True, wait_seconds=0)
        return returns

    def load_manifest(self, manifest_path):
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    def test_inventory_is_read_only_and_classifies_exact_entries(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        self.write(old / "nested" / "report.json", {"schema": "xefg.report/1"})
        empty = self.scan / "xefg-empty"
        empty.mkdir()
        log = self.write(self.scan / "xefg-old.log", b"xefg old log\n")
        unknown = self.scan / "xefg-unknown"
        self.write(unknown / "random.txt", b"nothing project-specific\n")
        self.write(self.scan / "not-xefg", b"ignore me\n")

        manifest, output = self.inventory()
        self.assertTrue(output.is_file())
        self.assertTrue(output.is_relative_to(self.current / "cleanup"))
        self.assertTrue(old.exists())
        self.assertTrue(log.exists())
        entries = {Path(entry["path"]).name: entry for entry in manifest["entries"]}
        self.assertEqual(entries["xefg-old"]["classification"], "delete")
        self.assertEqual(entries["xefg-old"]["kind"], "directory")
        self.assertGreater(entries["xefg-old"]["bytes"], 0)
        self.assertEqual(entries["xefg-empty"]["classification"], "delete")
        self.assertEqual(entries["xefg-old.log"]["classification"], "delete")
        self.assertEqual(entries["xefg-unknown"]["classification"], "skip")
        self.assertIn("unknown ownership", entries["xefg-unknown"]["reason"])
        self.assertEqual(entries["xefg-protected"]["classification"], "keep")
        self.assertNotIn("not-xefg", entries)
        self.assertIn("st_dev", entries["xefg-old"]["identity"])
        self.assertIn("st_ino", entries["xefg-old"]["identity"])
        self.assertEqual(manifest["summary"]["delete_count"], 3)
        self.assertEqual(manifest["summary"]["keep_count"], 1)
        self.assertEqual(manifest["summary"]["skip_count"], 1)

    def test_inventory_skips_reparse_entries(self):
        target = self.scan / "xefg-link"
        self.write(target / "matrix.json", {"schema": "xefg.render_matrix/4"})
        real_lstat = cleanup.lstat_path

        def fake_lstat(value):
            st, reparse = real_lstat(value)
            if os.path.normcase(os.path.basename(os.fspath(value))) == "xefg-link":
                return st, True
            return st, reparse

        with patch.object(cleanup, "lstat_path", side_effect=fake_lstat):
            manifest, _ = self.inventory()
        entry = next(row for row in manifest["entries"] if Path(row["path"]).name == "xefg-link")
        self.assertEqual(entry["classification"], "skip")
        self.assertIn("reparse", entry["reason"])

    def test_inventory_never_creates_current_root(self):
        missing = self.root / "missing-current"
        scope = [cleanup.ScopeRule(str(self.scan), "xefg-*", "run")]
        with self.assertRaises(cleanup.CleanupError):
            cleanup.run_inventory(missing, scope, [], [missing], wait_seconds=0)
        self.assertFalse(missing.exists())

    def test_apply_requires_complete_matrix_and_allows_quality_failure(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        unknown = self.scan / "xefg-unknown"
        self.write(unknown / "random.txt", b"leave me\n")
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.matrix())
        report = cleanup.run_apply(
            manifest_path, self.current, matrix_path, self.package)
        self.assertFalse(old.exists())
        self.assertTrue(unknown.exists())
        self.assertTrue(self.keep.exists())
        self.assertEqual(report["validation"]["matrix"]["rows"], 135)
        self.assertEqual(report["validation"]["matrix"]["generated_frames"], 1065)
        self.assertEqual(report["actual"]["deleted_entries"], 1)
        self.assertEqual(report["projected"]["bytes"], report["actual"]["deleted_bytes"])
        self.assertTrue(report["disk_before"])
        self.assertTrue(report["disk_after"])
        self.assertTrue(Path(report["report_path"]).is_file())

    def test_apply_refuses_run_failed_matrix(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.matrix(special={0: "run_failed"}))
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertIn("run failure", str(caught.exception))
        self.assertTrue(old.exists())

    def test_apply_refuses_incomplete_counts(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        too_few = self.write_matrix(self.matrix(rows=134, frames=1065), "few.json")
        with self.assertRaises(cleanup.CleanupError):
            cleanup.run_apply(manifest_path, self.current, too_few, self.package)
        wrong_frames = self.write_matrix(self.matrix(rows=135, frames=1064), "frames.json")
        with self.assertRaises(cleanup.CleanupError):
            cleanup.run_apply(manifest_path, self.current, wrong_frames, self.package)
        self.assertTrue(old.exists())

    def test_apply_refuses_package_hash_mismatch(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.matrix())
        (self.package / "models" / "mode3.onnx").write_bytes(b"tampered")
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertIn("hash mismatch", str(caught.exception))
        self.assertTrue(old.exists())

    def test_apply_refuses_changed_identity(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.matrix())
        self.write(old / "added.bin", b"changed after inventory")
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertIn("since inventory", str(caught.exception))
        self.assertTrue(old.exists())

    def test_apply_refuses_tampered_paths_outside_scope_or_protected(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.matrix())
        original = self.load_manifest(manifest_path)
        outside = self.root / "outside"
        self.write(outside / "matrix.json", {"schema": "xefg.render_matrix/4"})
        tampered = json.loads(json.dumps(original))
        tampered["entries"].append({
            "path": str(outside), "classification": "delete", "kind": "directory",
            "bytes": 1, "files": 1, "dirs": 0,
            "identity": {"kind": "directory", "st_dev": 0, "st_ino": 0,
                         "st_mtime_ns": 0, "bytes": 1, "files": 1, "dirs": 0},
            "evidence": [],
        })
        tampered_path = self.current / "cleanup" / "tampered-outside.json"
        self.write(tampered_path, tampered)
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(tampered_path, self.current, matrix_path, self.package)
        self.assertIn("outside the approved inventory scope", str(caught.exception))
        self.assertTrue(outside.exists())

        protected = json.loads(json.dumps(original))
        protected["entries"].append({
            "path": str(self.keep), "classification": "delete", "kind": "directory",
            "bytes": 1, "files": 1, "dirs": 0,
            "identity": {"kind": "directory", "st_dev": 0, "st_ino": 0,
                         "st_mtime_ns": 0, "bytes": 1, "files": 1, "dirs": 0},
            "evidence": [],
        })
        protected_path = self.current / "cleanup" / "tampered-protected.json"
        self.write(protected_path, protected)
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(protected_path, self.current, matrix_path, self.package)
        self.assertIn("protected", str(caught.exception))
        self.assertTrue(self.keep.exists())

    def test_apply_dry_run_and_explicit_manifest_requirement(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.matrix())
        report = cleanup.run_apply(
            manifest_path, self.current, matrix_path, self.package, dry_run=True)
        self.assertTrue(old.exists())
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["actual"]["deleted_entries"], 0)
        self.assertGreater(report["projected"]["bytes"], 0)

        copied = self.write(self.root / "copied-manifest.json", self.load_manifest(manifest_path))
        with self.assertRaises(cleanup.CleanupError):
            cleanup.run_apply(copied, self.current, matrix_path, self.package)

    def test_apply_accepts_final_replay_matrix_schema(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.replay_matrix())
        self.write_status()
        report = cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertFalse(old.exists())
        self.assertEqual(report["validation"]["matrix"]["kind"], "replay")
        self.assertEqual(report["validation"]["matrix"]["rows"], 135)
        self.assertEqual(report["validation"]["matrix"]["generated_frames"], 1065)
        self.assertEqual(report["validation"]["matrix"]["status_file"]["completed_combos"], 135)
        self.assertEqual(report["validation"]["package"]["matrix_profile_rows"], 0)

    def test_apply_validates_replay_package_file_identity(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix = self.replay_matrix()
        matrix["package"] = self.package_identity()
        matrix_path = self.write_matrix(matrix)
        self.write_status()
        report = cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertFalse(old.exists())
        self.assertEqual(report["validation"]["package"]["matrix_profile_rows"], 0)
        self.assertTrue(report["validation"]["package"]["matrix_profile_link"])

        old_again = self.scan / "xefg-old-again"
        self.write(old_again / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_again = self.inventory()
        tampered = self.replay_matrix()
        tampered["package"] = self.package_identity()
        tampered["package"]["files"]["models/mode3.onnx"]["sha256"] = "0" * 64
        tampered_path = self.write_matrix(tampered, "tampered.json")
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(manifest_again, self.current, tampered_path, self.package)
        self.assertIn("package hash does not match", str(caught.exception))
        self.assertTrue(old_again.exists())

    def test_apply_refuses_nonfinite_frame_metrics(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.replay_matrix(nan_frame=True))
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertIn("non-finite", str(caught.exception))
        self.assertTrue(old.exists())

    def test_apply_refuses_status_file_incomplete(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.replay_matrix())
        self.write_status(completed_combos=134)
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertIn("completed combinations", str(caught.exception))
        self.assertTrue(old.exists())

    def test_apply_refuses_replay_run_failed_combo(self):
        old = self.scan / "xefg-old"
        self.write(old / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        _, manifest_path = self.inventory()
        matrix_path = self.write_matrix(self.replay_matrix(special={0: "run_failed"}))
        with self.assertRaises(cleanup.CleanupError) as caught:
            cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertIn("run failure", str(caught.exception))
        self.assertTrue(old.exists())

    def test_exact_paths_are_added_to_manifest_scope(self):
        artifact = self.scan / "ptrcheck.obj"
        self.write(artifact, b"\x64\x86" + b"coff payload")
        scope = [cleanup.ScopeRule(str(self.scan), "xefg-*", "run")]
        manifest, manifest_path = cleanup.run_inventory(
            self.current, scope, [artifact], [self.current, self.keep], force=True)
        exact_rules = [rule for rule in manifest["scope"] if rule["kind"] == "exact"]
        self.assertEqual([Path(rule["root"]) for rule in exact_rules], [artifact])
        entry = next(row for row in manifest["entries"] if row["path"] == str(artifact))
        self.assertEqual(entry["classification"], "delete")
        matrix_path = self.write_matrix(self.matrix())
        cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertFalse(artifact.exists())

    def test_temp_build_and_shader_dump_dirs_are_owned(self):
        build = self.scan / "xefg-renderer-build"
        self.write(build / "build.bat",
                   '@call "C:\\Users\\Eleven\\Documents\\xefg-sdk\\build.bat"')
        self.write(build / "renderer.obj", b"\x64\x86binary")
        disasm = self.scan / "xefg-disasm"
        self.write(disasm / "c15.txt", "; Input signature:\n; Name Index Mask\n")
        verify = self.scan / "xefg-verify"
        self.write(verify / "dump" / "dump1.txt", "; Input signature:\n")
        verify2 = self.scan / "xefg-verify-1758"
        self.write(verify2 / "probe-opsonly" / "trace" / "graph.json",
                   {"run": {"directory": "D:\\xefg-evidence\\run"}})
        manifest, _ = self.inventory()
        entries = {Path(entry["path"]).name: entry for entry in manifest["entries"]}
        for name in ("xefg-renderer-build", "xefg-disasm", "xefg-verify",
                     "xefg-verify-1758"):
            self.assertEqual(entries[name]["classification"], "delete", name)

    def make_junction(self, link, target):
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            output = result.stdout.decode("utf-8", "replace")
            self.skipTest(f"cannot create junction: {output}")

    def test_owned_local_junction_is_link_only_delete(self):
        work = self.scan / "xefg-work"
        self.write(work / "matrix.json", {"schema": "xefg.render_matrix/4", "scenes": []})
        target = self.root / "xefg-target"
        self.write(target / "keep.txt", b"target payload must remain")
        link = work / "trace" / "tensors"
        self.make_junction(link, target)
        manifest, manifest_path = self.inventory()
        entry = next(row for row in manifest["entries"] if row["path"] == str(work))
        self.assertEqual(entry["classification"], "delete")
        self.assertEqual(entry["reparse_policy"], "delete_links_only")
        self.assertEqual(len(entry["reparse_links"]), 1)
        link_info = entry["reparse_links"][0]
        self.assertTrue(link_info["target_local"])
        self.assertTrue(link_info["target_owned"])
        self.assertEqual(Path(link_info["target"]), target)
        matrix_path = self.write_matrix(self.matrix())
        cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertFalse(work.exists())
        self.assertTrue((target / "keep.txt").is_file())
        self.assertEqual((target / "keep.txt").read_bytes(), b"target payload must remain")

    def test_exact_file_candidate_is_verified_and_deleted(self):
        artifact = self.scan / "ptrcheck.obj"
        self.write(artifact, b"\x64\x86" + b"coff payload")
        matched, manifest_path = self.inventory(exact_paths=[artifact])
        entry = next(row for row in matched["entries"] if row["path"] == str(artifact))
        self.assertEqual(entry["classification"], "delete")
        matrix_path = self.write_matrix(self.matrix())
        report = cleanup.run_apply(manifest_path, self.current, matrix_path, self.package)
        self.assertFalse(artifact.exists())
        self.assertEqual(report["actual"]["deleted_entries"], 1)

        mismatch = self.scan / "ptrcheck.obj"
        self.write(mismatch, b"not coff")
        manifest, _ = self.inventory(exact_paths=[mismatch])
        entry = next(row for row in manifest["entries"] if row["path"] == str(mismatch))
        self.assertEqual(entry["classification"], "skip")
        self.assertIn("content does not match", entry["reason"])


if __name__ == "__main__":
    unittest.main()
