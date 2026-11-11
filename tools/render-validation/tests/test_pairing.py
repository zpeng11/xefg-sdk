import importlib.util
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("pairing", Path(__file__).parents[1] / "pairing.py")
pairing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairing)


class PairingTests(unittest.TestCase):
    def test_marker_roundtrip(self):
        marker = dict(x=32, y=32, cell_width=8, cell_height=8, bits=32,
                      encoding="u16-complement-lsb", threshold=128)
        for frame in (0, 1, 7, 255, 65535):
            pixels = np.zeros((360, 640, 4), dtype=np.uint8)
            code = frame | ((frame ^ 65535) << 16)
            for bit in range(32):
                pixels[32:40, 32 + bit * 8:40 + bit * 8, :3] = 255 if code & (1 << bit) else 0
            self.assertEqual(pairing.decode_marker(pixels, marker)[0], frame)
            pixels[32:40, 32:40, :3] = 128
            self.assertIsNone(pairing.decode_marker(pixels, marker)[0])

    def classify(self, kind="midpoint", frame=4, reset=False, static=False, success=True, gap=True):
        scores = [{"kind": kind, "frame": frame, "mae": 1},
                  {"kind": "base", "frame": 3, "mae": 10 if gap else 1.1}]
        status = dict(present_result=0, status_result=0,
                      frame_generation_result=0 if success else 3, is_frame_gen_enabled=True)
        return pairing.classify(4, scores, status, reset, static, False, True)[0]

    def test_generated(self):
        self.assertEqual(self.classify(), "generated")

    def test_base(self):
        self.assertEqual(self.classify(kind="base"), "base")

    def test_reset_fallback(self):
        self.assertEqual(self.classify(kind="base", reset=True, success=False), "fallback")

    def test_reset_midpoint_is_not_generated(self):
        self.assertEqual(self.classify(reset=True), "ambiguous")

    def test_wrong_candidate_frame(self):
        self.assertEqual(self.classify(frame=5), "ambiguous")

    def test_close_candidates(self):
        self.assertEqual(self.classify(gap=False), "ambiguous")

    def test_static_is_not_temporal_evidence(self):
        self.assertEqual(self.classify(static=True), "ambiguous")

    def test_bad_image_is_not_paired_by_relative_rank(self):
        scores = [{"kind": "midpoint", "frame": 4, "mae": 100},
                  {"kind": "base", "frame": 4, "mae": 125}]
        status = dict(present_result=0, status_result=0, frame_generation_result=0,
                      is_frame_gen_enabled=True)
        self.assertEqual(pairing.classify(4, scores, status, False, False, True, True)[0],
                         "ambiguous")

    def test_missing_status(self):
        self.assertEqual(pairing.classify(4, [], None, False, False, True, True)[0], "ambiguous")

    def test_no_identity_evidence(self):
        scores = [{"kind": "midpoint", "frame": 4, "mae": 1},
                  {"kind": "base", "frame": 4, "mae": 10}]
        status = dict(present_result=0, status_result=0, frame_generation_result=0,
                      is_frame_gen_enabled=True)
        self.assertEqual(pairing.classify(4, scores, status, False, False, False, False)[0],
                         "ambiguous")

    def test_marker_complement_rejects_wrong_bits(self):
        marker = dict(x=32, y=32, cell_width=8, cell_height=8, bits=32,
                      encoding="u16-complement-lsb", threshold=128)
        pixels = np.zeros((360, 640, 4), dtype=np.uint8)
        self.assertIsNone(pairing.decode_marker(pixels, marker)[0])

    def test_pairing_uses_pixels_not_observed_label(self):
        marker = dict(x=32, y=32, cell_width=8, cell_height=8, bits=32,
                      encoding="u16-complement-lsb", threshold=128)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "application_inputs").mkdir()
            (root / "calibration").mkdir()
            frames = []
            references = []
            for index in range(3):
                base = np.full((360, 640, 4), index * 80, dtype=np.uint8)
                mid = np.full_like(base, max(0, index * 80 - 40))
                base.tofile(root / "application_inputs" / f"{index}.bgra")
                mid.tofile(root / "calibration" / f"{index}.bgra")
                frames.append({"index": index, "inputs": {"backbuffer": f"{index}.bgra"}})
                references.append({"index": index, "midpoint": f"{index}.bgra", "reset_history": index == 0})
            actual = np.full((360, 640, 4), 120, dtype=np.uint8)
            code = 2 | ((2 ^ 65535) << 16)
            for bit in range(32):
                actual[32:40, 32 + bit * 8:40 + bit * 8, :3] = 255 if code & (1 << bit) else 0
            actual.tofile(root / "output.bgra")
            native = {"schema": "xefg.present_capture/2", "frame_status": [
                dict(application_frame=i, present_result=0, status_result=0,
                     frame_generation_result=0, is_frame_gen_enabled=True) for i in range(3)],
                "tag_interpolated_requested": False, "tag_interpolated_result": 0,
                "only_interpolation_requested": True, "only_interpolation_result": 0,
                "presents": [dict(sequence=0, file="output.bgra", format=87, observed_application_frame=1)]}
            for name, data in (("presents.json", native),
                               ("application_inputs/manifest.json", dict(width=640, height=360, frame_count=3, frames=frames)),
                               ("calibration/manifest.json", dict(schema="xefg.calibration/1", marker=marker,
                                  motion_roi=[80, 80, 560, 300], static=False, frames=references))):
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            def inspect(run):
                return {"configuration": {"ui_mode": 3, "warmup": 2},
                        "artifacts": [{"file": "output.bgra", "sha256": hashlib.sha256(actual.tobytes()).hexdigest()}]}
            def artifact(run, name, expected=None):
                return {"file": name, "sha256": "test"}
            report = pairing.write_pairing(root, root / "report", inspect, artifact)
            self.assertEqual(report["status"], "paired")
            self.assertEqual(report["pairs"][0]["current_frame"], 2)
            self.assertEqual(report["pairs"][0]["observed_application_frame"], 1)
            native["presents"].append({**native["presents"][0], "sequence": 1})
            (root / "presents.json").write_text(json.dumps(native), encoding="utf-8")
            duplicate = pairing.write_pairing(root, root / "duplicates", inspect, artifact)
            self.assertEqual(duplicate["counts"], {"ambiguous": 2})
            self.assertEqual(duplicate["missing_generated_frames"], [2])

    def test_unsigned_difference_does_not_wrap(self):
        actual = np.zeros((2, 2, 4), dtype=np.uint8)
        reference = np.full_like(actual, 255)
        scores = pairing.candidate_scores(actual, [("base", 0, reference)], [0, 0, 2, 2])
        self.assertEqual(scores[0]["mae"], 255)


if __name__ == "__main__":
    unittest.main()
