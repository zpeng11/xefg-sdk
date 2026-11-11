import importlib.util
from pathlib import Path
import unittest
import tempfile
import json

import numpy as np
import onnx


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


edges = load("analyze_edges")
sampling = load("probe_sampling")


class EdgeDiagnosticTests(unittest.TestCase):
    def test_analysis_uses_manifest_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("capture/application_inputs", "pairing", "replay"):
                (root / name).mkdir(parents=True)
            pixels = np.zeros((8, 16, 4), np.uint8)
            pixels.tofile(root / "capture/official.bgra")
            changed = pixels.copy()
            changed[4, 8, 0] = 5
            np.save(root / "replay/frame_00000.npy", changed)
            roles = {"ui": "ui.raw", "motion": "motion.raw", "depth": "depth.raw"}
            for role, values in (("ui", pixels), ("motion", np.zeros((4, 8, 2), "<f2")),
                                 ("depth", np.ones((4, 8), "<f4"))):
                values.tofile(root / "capture/application_inputs" / roles[role])
            manifest = {"width": 16, "height": 8, "motion_width": 8, "motion_height": 4,
                        "frames": [{"inputs": roles}]}
            (root / "capture/application_inputs/manifest.json").write_text(json.dumps(manifest))
            (root / "pairing/pairing.json").write_text(json.dumps({"excluded_rects": [], "pairs": [
                {"classification": "generated", "current_frame": 0, "file": "official.bgra"}]}))
            result = edges.analyze(root)["frames"][0]
            self.assertEqual(result["over_2_pixels"], 1)
            self.assertEqual(result["over_2_bbox_xyxy"], [8, 4, 9, 5])

    def test_boundaries_mark_both_sides(self):
        values = np.zeros((4, 4), dtype=np.float32)
        values[:, 2:] = 1
        boundary = edges.boundaries(values, 0.5)
        self.assertTrue(boundary[:, 1:3].all())
        self.assertFalse(boundary[:, 0].any())
        self.assertFalse(boundary[:, 3].any())

    def test_vector_boundary_uses_both_components(self):
        values = np.zeros((3, 3, 2), dtype=np.float32)
        values[1:, :, 1] = 1
        self.assertTrue(edges.boundaries(values, 0.5)[:2].all())

    def test_dilation_does_not_wrap(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        result = edges.dilate(mask, 1)
        self.assertEqual(result.sum(), 4)
        self.assertFalse(result[-1].any())

    def test_rounding_probe_changes_only_selected_dispatch(self):
        graph = onnx.helper.make_graph([
            onnx.helper.make_node("Ceil", ["a"], ["n0024_sp_cx"]),
            onnx.helper.make_node("Ceil", ["b"], ["unrelated"]),
            onnx.helper.make_node("Round", ["c"], ["n0025_wx"]),
        ], "test", [], [])
        model = onnx.helper.make_model(graph)
        changed = sampling.replace_rounding(model, "d24", "Round")
        self.assertEqual(changed, ["n0024_sp_cx"])
        self.assertEqual([n.op_type for n in model.graph.node], ["Round", "Ceil", "Round"])
        with self.assertRaises(ValueError):
            sampling.replace_rounding(model, "d24", "Floor")


if __name__ == "__main__":
    unittest.main()
