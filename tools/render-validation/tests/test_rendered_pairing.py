import importlib.util
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("rendered", Path(__file__).parents[1] / "rendered_pairing.py")
rendered = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rendered)


class RenderedPairingTests(unittest.TestCase):
    def setUp(self):
        self.present = {"native_present_result": 0}
        self.record = {"previous_frame": 3, "reset_history": False,
                       "submission_results": dict.fromkeys(("depth", "motion", "color", "ui", "constants", "present_id"), 0)}
        self.status = {"present_result": 0, "status_result": 0,
                       "frame_generation_result": 0, "is_frame_gen_enabled": True}

    def classify(self, tagged=True, base_mae=0):
        return rendered.classify(self.present, 4, self.record, self.status, tagged, base_mae)[0]

    def test_tagged_static_content_is_generated(self):
        self.assertEqual(self.classify(), "generated")

    def test_reset_cannot_be_generated(self):
        self.record["reset_history"] = True
        self.assertEqual(self.classify(), "ambiguous")
        self.assertEqual(self.classify(tagged=False), "fallback")

    def test_base_content_alone_is_not_generated(self):
        self.assertEqual(self.classify(tagged=False), "base")
        self.assertEqual(self.classify(tagged=False, base_mae=5), "ambiguous")

    def test_failed_submission(self):
        for key in self.record["submission_results"]:
            self.record["submission_results"][key] = -1
            self.assertEqual(self.classify(), "ambiguous")
            self.record["submission_results"][key] = 0

    def test_wrong_history(self):
        self.record["previous_frame"] = 2
        self.assertEqual(self.classify(), "ambiguous")

    def test_failed_native_present(self):
        self.present["native_present_result"] = -1
        self.assertEqual(self.classify(), "ambiguous")

    def test_marker_and_all_tag_regions_excluded(self):
        rects = [*rendered.TAG_RECTS, [96, 32, 352, 40]]
        mask = rendered.mask_rects((360, 640), rects)
        self.assertFalse(mask[36, 100])
        self.assertFalse(mask[180, 50])
        self.assertTrue(mask[180, 320])
        self.assertGreater(mask.sum(), 180000)
        with self.assertRaises(ValueError):
            rendered.mask_rects((360, 640), [[0, 0, 700, 360]])

    def test_complete_rendered_report_and_duplicate_rejection(self):
        spec = importlib.util.spec_from_file_location("validation", Path(__file__).parents[1] / "validate.py")
        validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "application_inputs").mkdir()
            (root / "native_mid").mkdir()
            marker = dict(x=96, y=32, cell_width=8, cell_height=8, bits=32,
                          encoding="u16-complement-lsb", threshold=128)
            inputs, frames, statuses = [], [], []
            for i in range(2):
                pixels = np.full((360, 640, 4), 20 + i, dtype=np.uint8)
                pixels.tofile(root / "application_inputs" / f"{i}.bgra")
                pixels.tofile(root / "native_mid" / f"{i}.bgra")
                inputs.append(dict(index=i, inputs={"backbuffer": f"{i}.bgra"}))
                frames.append(dict(index=i, previous_frame=i-1 if i else None, reset_history=i == 0,
                                   midpoint=f"native_mid/{i}.bgra", submission_results=self.record["submission_results"]))
                statuses.append(dict(application_frame=i, **self.status))
            actual = np.full((360, 640, 4), 30, dtype=np.uint8)
            for x0,y0,x1,y1 in rendered.TAG_RECTS:
                actual[y0:y1,x0:x1,:3] = [255,0,255]
            code = 1 | (65534 << 16)
            for bit in range(32):
                actual[32:40,96+bit*8:104+bit*8,:3] = 255 if code & (1 << bit) else 0
            actual.tofile(root / "present.bgra")
            present = dict(sequence=0,file="present.bgra",format=87,observed_application_frame=0, native_present_result=0)
            native = dict(schema="xefg.present_capture/2", tag_interpolated_requested=True,
                          tag_interpolated_result=0, frame_status=statuses,presents=[present])
            for name,data in (("presents.json",native), ("scene.json",dict(schema="xefg.render_scene/1", marker=marker,frames=frames)),
                               ("application_inputs/manifest.json",dict(width=640,height=360,motion_width=320,
                                                                        motion_height=180,frame_count=2,frames=inputs))):
                (root / name).write_text(json.dumps(data),encoding="utf-8")
            def inventory(run):
                return dict(configuration=dict(ui_mode=3,warmup=1),provider=dict(xefg_dll_sha256=rendered.PROFILED_PROVIDER),
                            artifacts=[validation.artifact(root,"present.bgra")])
            tools = SimpleNamespace(inspect_run=inventory,read_json=validation.read_json,
                                    artifact=validation.artifact,load_tool=validation.load_tool)
            report = rendered.write_pairing(root,root / "pairing",tools)
            self.assertEqual(report["status"],"paired")
            self.assertEqual(report["pairs"][0]["current_frame"],1)
            self.assertEqual(report["pairs"][0]["observed_application_frame"],0)
            native["presents"].append({**present,"sequence":1})
            (root / "presents.json").write_text(json.dumps(native),encoding="utf-8")
            report = rendered.write_pairing(root,root / "duplicates",tools)
            self.assertEqual(report["counts"],{"ambiguous":2})
            self.assertEqual(report["missing_generated_frames"],[1])

    def test_full_tag_layout_required(self):
        pixels = np.zeros((360, 640, 4), dtype=np.uint8)
        for x0, y0, x1, y1 in rendered.TAG_RECTS:
            pixels[y0:y1, x0:x1, :3] = [255, 0, 255]
        self.assertTrue(rendered.tag_evidence(pixels)[0])
        pixels[0, 0, 0] = 254
        self.assertFalse(rendered.tag_evidence(pixels)[0])

    def test_mode4_null_ui_submission_is_not_a_failure(self):
        record = {"previous_frame": 3, "reset_history": False,
                  "submission_results": {"depth": 0, "motion": 0, "color": 0, "ui": None,
                                         "constants": 0, "present_id": 0}}
        self.assertEqual(rendered.classify(self.present, 4, record, self.status, True, None)[0],
                         "generated")
        rendered._validate_submissions(record, 4, "mode4")

    def _mode1_capture(self, root, tagged=True, second_tag=False, status1=None,
                       extra_unmatched=False, color=None):
        spec = importlib.util.spec_from_file_location("validation", Path(__file__).parents[1] / "validate.py")
        validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation)
        (root / "application_inputs").mkdir()
        (root / "native_mid").mkdir()
        inputs, frames, bases = [], [], []
        for i in range(2):
            base = np.full((360, 640, 4), 20 + i * 40, dtype=np.uint8)
            base[:, :, 3] = 255
            bases.append(base)
            base.tofile(root / "application_inputs" / f"bb{i}.bgra")
            base.tofile(root / "application_inputs" / f"color{i}.bgra")
            np.zeros((360, 640, 4), np.uint8).tofile(root / "application_inputs" / f"ui{i}.bgra")
            np.zeros((180, 320), "<f4").tofile(root / "application_inputs" / f"d{i}.r32f")
            np.zeros((180, 320, 2), "<f2").tofile(root / "application_inputs" / f"m{i}.rg16f")
            base.tofile(root / "native_mid" / f"mid{i}.bgra")
            inputs.append({"index": i, "inputs": {"color": f"color{i}.bgra", "ui": f"ui{i}.bgra",
                         "backbuffer": f"bb{i}.bgra", "depth": f"d{i}.r32f", "motion": f"m{i}.rg16f"}})
            frames.append({"index": i, "scene_frame": 4 + i,
                           "previous_frame": None if i == 0 else 0, "reset_history": i == 0,
                           "midpoint": f"native_mid/mid{i}.bgra", "time": float(4 + i),
                           "previous_time": float(3 + i),
                           "submission_results": {"depth": 0, "motion": 0,
                               "color": color, "ui": None, "constants": 0, "present_id": 0}})
        output = bases[1].copy()
        if tagged:
            for x0, y0, x1, y1 in rendered.TAG_RECTS:
                output[y0:y1, x0:x1, :3] = [255, 0, 255]
        output.tofile(root / "present1.bgra")
        bases[0].tofile(root / "base0.bgra")
        presents = [{"sequence": 0, "observed_application_frame": 99, "ordinal": 0,
                     "width": 640, "height": 360, "format": 87, "file": "base0.bgra",
                     "native_present_result": 0},
                    {"sequence": 1, "observed_application_frame": 88, "ordinal": 0,
                     "width": 640, "height": 360, "format": 87, "file": "present1.bgra",
                     "native_present_result": 0}]
        if second_tag:
            output.tofile(root / "present2.bgra")
            presents.append({**presents[-1], "sequence": 2, "file": "present2.bgra",
                             "observed_application_frame": 42})
        if extra_unmatched:
            np.full((360, 640, 4), 99, np.uint8).tofile(root / "noise.bgra")
            presents.append({**presents[-1], "sequence": len(presents), "file": "noise.bgra",
                             "observed_application_frame": 7})
        status1 = status1 or {"present_result": 0, "status_result": 0, "frames_presented": 1,
                              "frame_generation_result": 0, "is_frame_gen_enabled": True}
        statuses = [{"application_frame": 0, "present_result": 0, "status_result": 0,
                     "frames_presented": 0, "frame_generation_result": 3,
                     "is_frame_gen_enabled": True},
                    {"application_frame": 1, **status1}]
        scene = {"schema": "xefg.render_scene/1", "mode": 1,
                 "pairing_method": "isolated_pair", "source_pair": [4, 5],
                 "marker": dict(x=96, y=32, cell_width=8, cell_height=8, bits=32,
                                encoding="u16-complement-lsb", threshold=128),
                 "frames": frames}
        native = {"schema": "xefg.present_capture/2", "tag_interpolated_requested": True,
                  "tag_interpolated_result": 0, "capture_error": "",
                  "frame_status": statuses, "presents": presents}
        manifest = {"schema": "xefg.application_inputs/1", "width": 640, "height": 360,
                    "motion_width": 320, "motion_height": 180, "frame_count": 2,
                    "frames": inputs}
        for name, data in (("scene.json", scene), ("presents.json", native),
                           ("application_inputs/manifest.json", manifest)):
            (root / name).write_text(json.dumps(data), encoding="utf-8")
        files = [str(path.relative_to(root)).replace("\\", "/")
                 for path in sorted(root.rglob("*")) if path.is_file()]
        artifacts = [validation.artifact(root, name) for name in files]
        def inventory(run):
            return {"configuration": {"ui_mode": 1, "mode": 1, "pair_frame": 5,
                                      "warmup": 0, "frames": 2},
                    "provider": {"xefg_dll_sha256": rendered.PROFILED_PROVIDER},
                    "artifacts": artifacts}
        tools = SimpleNamespace(inspect_run=inventory, read_json=validation.read_json,
                                artifact=validation.artifact, load_tool=validation.load_tool)
        return tools

    def test_mode1_isolated_pair_uses_only_official_tag_and_source_times(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root)
            report = rendered.write_pairing(root, root / "pairing", tools)
            self.assertEqual(report["schema"], "xefg.pairing/3")
            self.assertEqual(report["status"], "paired")
            self.assertEqual(report["counts"], {"generated": 1, "fallback": 1})
            self.assertFalse(report["marker_used"])
            self.assertNotIn("marker", report)
            generated = next(row for row in report["pairs"] if row["classification"] == "generated")
            self.assertEqual(generated["previous_frame"], 0)
            self.assertEqual(generated["current_frame"], 1)
            self.assertEqual(generated["source_previous_frame"], 4)
            self.assertEqual(generated["source_current_frame"], 5)
            self.assertEqual(generated["previous_time"], 4.0)
            self.assertEqual(generated["current_time"], 5.0)
            self.assertEqual(generated["observed_application_frame"], 88)

    def test_mode1_zero_or_multiple_tags_never_certified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root, tagged=False)
            report = rendered.write_pairing(root, root / "zero", tools)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["tagged_present_count"], 0)
            self.assertEqual(report["missing_generated_frames"], [1])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root, second_tag=True)
            report = rendered.write_pairing(root, root / "many", tools)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["tagged_present_count"], 2)
            self.assertTrue(all(row["classification"] == "ambiguous"
                                for row in report["pairs"] if row["official_tag_detected"]))

    def test_mode1_unexplained_and_failed_outputs_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root, extra_unmatched=True)
            report = rendered.write_pairing(root, root / "extra", tools)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["counts"]["ambiguous"], 1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root)
            scene = json.loads((root / "scene.json").read_text())
            native = json.loads((root / "presents.json").read_text())
            native["presents"][1]["native_present_result"] = -1
            (root / "presents.json").write_text(json.dumps(native))
            report = rendered.write_pairing(root, root / "present", tools)
            self.assertEqual(report["status"], "incomplete")

    def test_mode1_reset_and_submission_contract_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root, color=0)
            with self.assertRaisesRegex(ValueError, "must not submit color"):
                rendered.write_pairing(root, root / "color", tools)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root, status1={"present_result": 0, "status_result": 0,
                "frames_presented": 1, "frame_generation_result": 3,
                "is_frame_gen_enabled": True})
            with self.assertRaisesRegex(ValueError, "generated status"):
                rendered.write_pairing(root, root / "status", tools)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._mode1_capture(root)
            scene = json.loads((root / "scene.json").read_text())
            scene["frames"][1]["time"] = 4.5
            (root / "scene.json").write_text(json.dumps(scene))
            with self.assertRaisesRegex(ValueError, "source times differ exactly"):
                rendered.write_pairing(root, root / "times", tools)


    def _synthetic_rendered_capture(self, root, width, height, rects, provider):
        spec = importlib.util.spec_from_file_location("validation", Path(__file__).parents[1] / "validate.py")
        validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation)
        (root / "application_inputs").mkdir()
        (root / "native_mid").mkdir()
        marker = dict(x=width // 8, y=height // 8, cell_width=8, cell_height=8, bits=32,
                      encoding="u16-complement-lsb", threshold=128)
        inputs, frames, statuses = [], [], []
        for i in range(2):
            pixels = np.full((height, width, 4), 20 + i, dtype=np.uint8)
            pixels[:, :, 3] = 255
            pixels.tofile(root / "application_inputs" / f"bb{i}.bgra")
            pixels.tofile(root / "native_mid" / f"mid{i}.bgra")
            inputs.append(dict(index=i, inputs={"backbuffer": f"bb{i}.bgra"}))
            frames.append(dict(index=i, previous_frame=i - 1 if i else None,
                               reset_history=i == 0, midpoint=f"native_mid/mid{i}.bgra",
                               submission_results=dict(self.record["submission_results"])))
            statuses.append(dict(application_frame=i, **self.status))
        actual = np.full((height, width, 4), 30, dtype=np.uint8)
        for x0, y0, x1, y1 in rects:
            actual[y0:y1, x0:x1, :3] = [255, 0, 255]
        code = 1 | (65534 << 16)
        for bit in range(32):
            x = marker["x"] + bit * marker["cell_width"]
            level = 255 if code & (1 << bit) else 0
            actual[marker["y"]:marker["y"] + marker["cell_height"],
                   x:x + marker["cell_width"], :3] = level
        actual.tofile(root / "present.bgra")
        present = dict(sequence=0, file="present.bgra", format=87,
                       observed_application_frame=0, native_present_result=0)
        native = dict(schema="xefg.present_capture/2", tag_interpolated_requested=True,
                      tag_interpolated_result=0, frame_status=statuses, presents=[present])
        scene = dict(schema="xefg.render_scene/1", marker=marker, frames=frames)
        manifest = dict(width=width, height=height, motion_width=width // 2,
                        motion_height=height // 2, frame_count=2, frames=inputs)
        for name, data in (("presents.json", native), ("scene.json", scene),
                           ("application_inputs/manifest.json", manifest)):
            (root / name).write_text(json.dumps(data), encoding="utf-8")
        def inventory(run):
            return dict(configuration=dict(ui_mode=3, warmup=1),
                        provider=dict(xefg_dll_sha256=provider),
                        artifacts=[validation.artifact(root, "present.bgra")])
        return SimpleNamespace(inspect_run=inventory, read_json=validation.read_json,
                               artifact=validation.artifact, load_tool=validation.load_tool)

    def test_synthetic_1080_layout_pairs_and_reads_full_size(self):
        provider = "synthetic-1080-provider"
        rects = [[0, 0, 64, 64], [1856, 0, 1920, 64], [0, 1016, 64, 1080],
                 [1856, 1016, 1920, 1080]]
        rendered.register_tag_layout(provider, 1920, 1080, rects)
        self.addCleanup(rendered.TAG_LAYOUTS.pop, (provider, (1920, 1080)), None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._synthetic_rendered_capture(root, 1920, 1080, rects, provider)
            report = rendered.write_pairing(root, root / "pairing", tools)
            self.assertEqual(report["status"], "paired")
            self.assertEqual(report["output_size"], [1920, 1080])
            self.assertEqual(report["motion_size"], [960, 540])
            marker = report["marker"]
            marker_pixels = marker["cell_width"] * marker["bits"] * marker["cell_height"]
            tag_pixels = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
            self.assertEqual(report["valid_pixel_count"],
                             1920 * 1080 - tag_pixels - marker_pixels)
            self.assertEqual(report["pairs"][0]["current_frame"], 1)

    def test_uncalibrated_non640_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            rendered.tag_evidence(np.zeros((1080, 1920, 4), np.uint8), "unmeasured-provider")
        rects = [[0, 0, 64, 64]]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self._synthetic_rendered_capture(root, 1920, 1080, rects,
                                                     "unmeasured-provider")
            with self.assertRaisesRegex(ValueError, "not calibrated"):
                rendered.write_pairing(root, root / "pairing", tools)

    def test_rendered_dimensions_require_allowed_size_and_half_motion(self):
        with self.assertRaisesRegex(ValueError, "supports"):
            rendered._input_dimensions({"width": 1024, "height": 768,
                                        "motion_width": 512, "motion_height": 384})
        with self.assertRaisesRegex(ValueError, "half-resolution"):
            rendered._input_dimensions({"width": 1920, "height": 1080,
                                        "motion_width": 1920, "motion_height": 1080})
        self.assertEqual(rendered._input_dimensions(
            {"width": 1920, "height": 1080, "motion_width": 960, "motion_height": 540}),
            (1920, 1080))


if __name__ == "__main__":
    unittest.main()
