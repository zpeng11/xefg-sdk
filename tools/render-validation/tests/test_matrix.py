import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("matrix", Path(__file__).parents[1] / "run_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class MatrixTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(matrix, "preflight_package", return_value=[])
        self.preflight = patcher.start()
        self.addCleanup(patcher.stop)

    def test_failed_scene_does_not_hide_other_scene_results(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "matrix"
            argv = ["run_matrix.py","--renderer","renderer.exe","--xess-bin","sdk",
                    "--package","package","--out",str(output)]
            with patch.object(matrix.sys,"argv",argv), patch.object(matrix.subprocess,"run",return_value=SimpleNamespace(returncode=2)) as run:
                self.assertEqual(matrix.main(),1)
            self.assertEqual(run.call_count,4)
            report = json.loads((output / "matrix.json").read_text())
            self.assertEqual(len(report["scenes"]),4)
            self.assertTrue(all(not row["completed"] for row in report["scenes"]))
            self.assertTrue(all(row["cases"][0]["stages"][0]["exit_code"] == 2 for row in report["scenes"]))

    def test_modes_cover_all_pairs_and_reset_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "matrix"
            argv = ["run_matrix.py", "--renderer", "renderer.exe", "--xess-bin", "sdk",
                    "--package", "package", "--out", str(output), "--modes", "1", "3", "4"]
            case = {"status": "pass", "completed": True, "generated_frames": 1,
                    "max_frame_rgb_mae": 0, "max_channel_error": 0, "max_over_2_fraction": 0}
            with patch.object(matrix.sys, "argv", argv), patch.object(matrix, "run_case", return_value=case) as run:
                self.assertEqual(matrix.main(), 0)
            self.assertEqual(run.call_count, 31 + 4 + 4)
            report = json.loads((output / "matrix.json").read_text())
            self.assertEqual(len(report["scenes"]), 12)
            cut = report["scenes"][3]
            self.assertEqual(cut["mode"], 1)
            self.assertEqual(len(cut["cases"]), 7)
            self.assertEqual(cut["skipped"][0]["source_frame"], 7)

    def test_quality_failure_has_distinct_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = ["run_matrix.py", "--renderer", "renderer.exe", "--xess-bin", "sdk",
                    "--package", "package", "--out", str(Path(temp) / "matrix")]
            case = {"status": "quality_failed", "completed": True, "generated_frames": 8,
                    "max_frame_rgb_mae": 0.1, "max_channel_error": 10, "max_over_2_fraction": 0.01}
            with patch.object(matrix.sys, "argv", argv), patch.object(matrix, "run_case", return_value=case):
                self.assertEqual(matrix.main(), 2)
            self.assertEqual(matrix.combined_status([case, {"status": "run_failed"}]), "run_failed")

    def test_quality_stage_and_mode_are_forwarded(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "scene"
            args = SimpleNamespace(renderer=Path("renderer.exe"), xess_bin=Path("sdk"), package=Path("package"))
            def execute(command, **kwargs):
                if "replay" in command:
                    (directory / "replay").mkdir(parents=True)
                    values = {"rgb_mae": 0, "rgb_max": 0, "over_tolerance_pixel_fraction": 0}
                    (directory / "replay/report.json").write_text(json.dumps({"pairs": [{"valid_region": values}]}))
                if any("check_quality.py" in word for word in command):
                    (directory / "quality.json").write_text(json.dumps({"schema": "xefg.quality_gate/1", "checks": [{"status": "pass"}]}))
                return SimpleNamespace(returncode=0)
            with patch.object(matrix.subprocess, "run", side_effect=execute) as run:
                result = matrix.run_case(args, directory, "camera", 1, 5)
            self.assertEqual(result["status"], "pass")
            self.assertEqual([row["stage"] for row in result["stages"]], ["render", "inputs", "pair", "replay", "quality"])
            command = run.call_args_list[0].args[0]
            self.assertEqual(command[command.index("--mode") + 1], "1")
            self.assertEqual(command[command.index("--pair-frame") + 1], "5")

    def test_quality_process_failure_is_not_numeric_failure_without_report(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(renderer=Path("renderer.exe"), xess_bin=Path("sdk"), package=Path("package"))
            def execute(command, **kwargs):
                return SimpleNamespace(returncode=1 if any("check_quality.py" in word for word in command) else 0)
            with patch.object(matrix.subprocess, "run", side_effect=execute):
                result = matrix.run_case(args, Path(temp) / "scene", "camera", 4)
            self.assertEqual(result["status"], "run_failed")
            self.assertFalse(result["completed"])

    def test_native_failure_cannot_be_waived_by_inventory_file(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "scene"
            (directory / "inventory").mkdir(parents=True)
            (directory / "inventory/report.json").write_text("{}")
            args = SimpleNamespace(renderer=Path("renderer.exe"), xess_bin=Path("sdk"), package=Path("package"))
            with patch.object(matrix.subprocess, "run", return_value=SimpleNamespace(returncode=2)) as run:
                result = matrix.run_case(args, directory, "cut", 3)
            self.assertEqual(result["status"], "run_failed")
            self.assertEqual(run.call_count, 1)

    def test_two_sizes_have_separate_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "matrix"
            argv = ["run_matrix.py", "--renderer", "renderer.exe", "--xess-bin", "sdk",
                    "--package", "package", "--out", str(output), "--sizes", "640x360", "1920x1080"]
            case = {"status": "pass", "completed": True, "generated_frames": 1,
                    "max_frame_rgb_mae": 0, "max_channel_error": 0, "max_over_2_fraction": 0}
            with patch.object(matrix.sys, "argv", argv), patch.object(matrix, "run_case", return_value=case) as run:
                self.assertEqual(matrix.main(), 0)
            self.assertEqual(run.call_count, 8)
            self.assertEqual(run.call_args_list[0].args[1], output / "640x360/mode3/translation")
            self.assertEqual(run.call_args_list[4].kwargs["size"], "1920x1080")
            report = json.loads((output / "matrix.json").read_text())
            self.assertEqual(len(report["size_results"]), 2)

    def test_missing_profile_stops_before_any_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "matrix"
            argv = ["run_matrix.py", "--renderer", "renderer.exe", "--xess-bin", "sdk",
                    "--package", "package", "--out", str(output)]
            self.preflight.side_effect = ValueError("missing profile")
            with patch.object(matrix.sys, "argv", argv), patch.object(matrix.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "missing profile"):
                    matrix.main()
                run.assert_not_called()
            self.assertFalse(output.exists())

    def test_new_output_only(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = ["run_matrix.py","--renderer","renderer.exe","--xess-bin","sdk",
                    "--package","package","--out",temp]
            with patch.object(matrix.sys,"argv",argv), patch.object(matrix.subprocess,"run") as run:
                with self.assertRaises(FileExistsError):
                    matrix.main()
                run.assert_not_called()


class PackagePreflightTests(unittest.TestCase):
    def test_profiles_resolve_exact_size_and_model_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = b"test model identity"
            (root / "model.onnx").write_bytes(model)
            profiles = []
            for width, height in ((640, 360), (1920, 1080)):
                name = f"profile-{width}.json"
                (root / name).write_text(json.dumps({"schema_version": 1, "variant": "mode3",
                    "id": f"mode3-{width}", "output_size": [width, height],
                    "motion_size": [width // 2, height // 2], "feeds": {}}))
                profiles.append(name)
            (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "variants": {
                "mode3": {"mode": 3, "model": "model.onnx", "model_sha256": hashlib.sha256(model).hexdigest(),
                          "profiles": profiles}}}))
            rows = matrix.preflight_package(root, ["640x360", "1920x1080"], [3])
            self.assertEqual([r["profile"] for r in rows], ["mode3-640", "mode3-1920"])
            self.assertEqual(len(rows[0]["profile_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "found 0"):
                matrix.preflight_package(root, ["1280x720"], [3])
            with self.assertRaisesRegex(ValueError, "found 0"):
                matrix.preflight_package(root, ["640x360"], [4])
            (root / "model.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "model hash"):
                matrix.preflight_package(root, ["640x360"], [3])


if __name__ == "__main__":
    unittest.main()
