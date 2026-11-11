import importlib.util
from pathlib import Path
import unittest

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
