import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("render_inputs", Path(__file__).parents[1] / "check_render_inputs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class RenderInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "application_inputs").mkdir()
        (self.root / "scene.json").write_text(json.dumps({"schema": "xefg.render_scene/1"}))
        manifest = dict(motion_width=2,motion_height=1,frames=[dict(index=0,inputs=dict(depth="depth.raw",motion="mv.raw"))])
        (self.root / "application_inputs/manifest.json").write_text(json.dumps(manifest))
        self.depth = self.root / "application_inputs/depth.raw"
        np.array([0.5,1],"<f4").tofile(self.depth)
        np.zeros((1,2,2),"<f2").tofile(self.root / "application_inputs/mv.raw")

    def test_normal_device_depth(self):
        report = checker.inspect_inputs(self.root)
        self.assertEqual(report["frames"][0]["geometry_pixels"],1)
        self.assertEqual(report["frames"][0]["moving_pixels"],0)

    def test_depth_not_reinterpreted_as_linear(self):
        np.array([1,100],"<f4").tofile(self.depth)
        with self.assertRaisesRegex(ValueError,"device-depth"):
            checker.inspect_inputs(self.root)

    def test_nan_rejected(self):
        np.array([float("nan"),1],"<f4").tofile(self.depth)
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            checker.inspect_inputs(self.root)


if __name__ == "__main__":
    unittest.main()
