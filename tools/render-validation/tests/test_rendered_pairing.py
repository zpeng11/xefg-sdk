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
                               ("application_inputs/manifest.json",dict(width=640,height=360,frame_count=2,frames=inputs))):
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


if __name__ == "__main__":
    unittest.main()
