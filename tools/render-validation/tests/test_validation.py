import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("validation", Path(__file__).parents[1] / "validate.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "application_inputs").mkdir()
        (self.root / "trace").mkdir()
        self.inputs = {
            "schema": "xefg.application_inputs/1", "width": 4, "height": 2,
            "motion_width": 2, "motion_height": 1, "frame_count": 2,
            "formats": {"color": "BGRA8", "ui": "BGRA8", "backbuffer": "BGRA8",
                        "depth": "R32_FLOAT", "motion": "R16G16_FLOAT"},
            "frames": [],
        }
        for index in range(2):
            files = {}
            for role in self.inputs["formats"]:
                name = f"{role}_{index}.raw"
                size = 8 if role in ("motion", "depth") else 32
                (self.root / "application_inputs" / name).write_bytes(bytes(size))
                files[role] = name
            self.inputs["frames"].append({"index": index, "inputs": files})
        self.meta = {"configuration": {"width": 4, "height": 2, "warmup": 1, "frames": 1},
                     "adapter": "fixture", "vendor_id": 0, "device_id": 0,
                     "xefg_family": "fixture", "xefg_dll_sha256": "fixture"}
        self.presents = {
            "schema": "xefg.present_capture/1", "only_interpolation_requested": True,
            "only_interpolation_result": 0, "capture_error": "",
            "frame_status": [{"application_frame": i, "present_result": 0,
                              "status_result": 0, "frames_presented": 1,
                              "frame_generation_result": 0} for i in range(2)],
            "presents": [{"sequence": 0, "observed_application_frame": 1,
                          "ordinal": 0, "width": 4, "height": 2, "format": 87,
                          "file": "present.bgra"}],
        }
        (self.root / "present.bgra").write_bytes(bytes(32))

    def inspect(self):
        for name, data in (("presents.json", self.presents), ("trace/meta.json", self.meta),
                           ("application_inputs/manifest.json", self.inputs)):
            (self.root / name).write_text(json.dumps(data), encoding="utf-8")
        return validation.inspect_run(self.root)

    def test_inventory_does_not_infer_generated_frame(self):
        report = self.inspect()
        self.assertEqual(report["status"], "inventory_complete")
        self.assertEqual(report["pairing"], "unverified")
        self.assertEqual(report["presents"][0]["classification"], "unassigned")
        self.assertEqual(len(report["artifacts"]), 14)
        self.assertEqual(len(report["artifacts"][-1]["sha256"]), 64)

    def test_root_metadata_without_fake_trace(self):
        self.inspect()
        (self.root / "trace/meta.json").rename(self.root / "meta.json")
        report = validation.inspect_run(self.root)
        names = {item["file"] for item in report["artifacts"]}
        self.assertIn("meta.json", names)
        self.assertNotIn("trace/meta.json", names)

    def test_v2_inventory_keeps_pairing_unverified(self):
        self.presents["schema"] = "xefg.present_capture/2"
        self.presents["tag_interpolated_requested"] = True
        self.presents["tag_interpolated_result"] = 0
        self.assertEqual(self.inspect()["pairing"], "unverified")

    def test_truncated_present(self):
        (self.root / "present.bgra").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "expected 32 bytes"):
            self.inspect()

    def test_missing_input(self):
        (self.root / "application_inputs/depth_0.raw").unlink()
        with self.assertRaises(FileNotFoundError):
            self.inspect()

    def test_capture_error(self):
        self.presents["capture_error"] = "readback failed"
        with self.assertRaisesRegex(ValueError, "readback failed"):
            self.inspect()

    def test_debug_feature_failure(self):
        self.presents["only_interpolation_result"] = -1
        self.assertEqual(self.inspect()["status"], "needs_attention")

    def test_all_presents_not_classified(self):
        self.presents["only_interpolation_requested"] = False
        self.assertEqual(self.inspect()["status"], "needs_attention")

    def test_generation_failure(self):
        self.presents["frame_status"][1]["frame_generation_result"] = -1
        self.assertEqual(self.inspect()["status"], "needs_attention")

    def test_warmup_status_is_not_a_measured_failure(self):
        self.presents["frame_status"][0]["frame_generation_result"] = 1
        self.assertEqual(self.inspect()["status"], "inventory_complete")

    def test_missing_status(self):
        self.presents["frame_status"].pop()
        with self.assertRaisesRegex(ValueError, "cover all"):
            self.inspect()

    def test_sequence_gap(self):
        self.presents["presents"][0]["sequence"] = 1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.inspect()

    def test_ordinal_gap(self):
        self.presents["presents"][0]["ordinal"] = 1
        with self.assertRaisesRegex(ValueError, "ordinal"):
            self.inspect()

    def test_input_frame_gap(self):
        self.inputs["frames"].pop(0)
        with self.assertRaisesRegex(ValueError, "every frame"):
            self.inspect()

    def test_artifact_path_escape(self):
        self.presents["presents"][0]["file"] = "../outside.bgra"
        with self.assertRaisesRegex(ValueError, "outside capture"):
            self.inspect()

    def test_report_never_overwrites(self):
        self.inspect()
        output = self.root / "report"
        validation.write_report(self.root, output)
        original = (output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            validation.write_report(self.root, output)
        self.assertEqual((output / "report.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
