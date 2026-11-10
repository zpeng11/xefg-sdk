"""Exercise the real ORT adapter through the public runtime API."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime  # noqa: F401  (full test extra installs ONNX Runtime)
from onnx import TensorProto, helper

from xefg_sdk import FrameInputs, ImageView, Xefg2xRuntime


class OrtExecutorTests(unittest.TestCase):
    def test_runtime_with_real_onnx_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
                ("previous_color", TensorProto.UINT8, [2, 4, 4]),
                ("current_color", TensorProto.UINT8, [2, 4, 4]),
                ("motion_vectors", TensorProto.FLOAT16, [1, 2, 2]),
                ("depth", TensorProto.FLOAT, [1, 2]),
            )]
            graph = helper.make_graph(
                [helper.make_node("Identity", ["current_color"], ["generated_frame"])],
                "executor_smoke", inputs,
                [helper.make_tensor_value_info("generated_frame", TensorProto.UINT8, [2, 4, 4])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
            model.ir_version = 8
            onnx.save(model, root / "model.onnx")
            (root / "profile.json").write_text(json.dumps({
                "schema_version": 1, "id": "smoke", "variant": "mode1",
                "output_size": [4, 2], "motion_size": [2, 1], "feeds": {},
            }), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "variants": {"mode1": {
                    "mode": 1, "model": "model.onnx",
                    "model_sha256": hashlib.sha256((root / "model.onnx").read_bytes()).hexdigest(),
                    "output": "generated_frame", "profiles": ["profile.json"],
                    "inputs": [v.name for v in inputs],
                }},
            }), encoding="utf-8")
            frames = []
            for n in range(2):
                frames.append(FrameInputs(
                    n, backbuffer_color=ImageView(np.full((2, 4, 4), n, np.uint8), "BGRA8_UNORM"),
                    depth=ImageView(np.ones((1, 2), np.float32), "R32_FLOAT"),
                    motion_vectors=ImageView(np.zeros((1, 2, 2), np.float16), "RG16_FLOAT")))
            with Xefg2xRuntime.load(root) as runtime:
                result = runtime.interpolate_pair(*frames)
            np.testing.assert_array_equal(result.generated_frame.data, frames[1].backbuffer_color.data)
            self.assertEqual(result.status, "generated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
