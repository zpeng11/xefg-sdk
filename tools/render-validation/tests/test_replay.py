import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load("replay")
validation = load("validate")


class ReplayTests(unittest.TestCase):
    def test_raw_frame_formats_and_current_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "application_inputs").mkdir()
            rows = []
            for index in range(2):
                files = {}
                for role, data in {
                    "color": np.full((2, 4, 4), index + 10, np.uint8),
                    "ui": np.full((2, 4, 4), index + 20, np.uint8),
                    "depth": np.full((1, 2), index + 30, "<f4"),
                    "motion": np.full((1, 2, 2), index + 40, "<f2"),
                }.items():
                    name = f"{index}_{role}.raw"
                    data.tofile(root / "application_inputs" / name)
                    files[role] = name
                rows.append({"index": index, "inputs": files})
            manifest = dict(width=4, height=2, motion_width=2, motion_height=1, frames=rows)
            previous = replay.load_frame(root, manifest, 0)
            current = replay.load_frame(root, manifest, 1, reset=True)
            self.assertEqual(previous.frame_id, 0)
            self.assertEqual(current.frame_id, 1)
            self.assertEqual(current.ui.data[0, 0, 0], 21)
            self.assertEqual(current.motion_vectors.data.dtype, np.float16)
            self.assertEqual(current.motion_vectors.data[0, 0, 0], 41)
            self.assertEqual(current.depth.data.dtype, np.float32)
            self.assertTrue(current.reset_history)
            self.assertEqual(current.hudless_color.format, "BGRA8_UNORM")

    def test_rendered_pairing_runs_masked_three_way_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "application_inputs").mkdir()
            (root / "native_mid").mkdir()
            frames = []
            for index in range(2):
                files = {}
                for role, data in {"color": np.zeros((2, 4, 4), np.uint8),
                                   "ui": np.zeros((2, 4, 4), np.uint8),
                                   "depth": np.ones((1, 2), "<f4"),
                                   "motion": np.zeros((1, 2, 2), "<f2")}.items():
                    name = f"{index}_{role}.raw"
                    data.tofile(root / "application_inputs" / name)
                    files[role] = name
                frames.append(dict(index=index, inputs=files))
            official = np.zeros((2, 4, 4), np.uint8)
            official[:, 0, :3] = 255
            official.tofile(root / "official.bgra")
            np.zeros_like(official).tofile(root / "native_mid/1.bgra")
            documents = {
                "application_inputs/manifest.json": dict(width=4,height=2,motion_width=2,motion_height=1,frames=frames),
                "scene.json": dict(frames=[dict(index=0,reset_history=True),dict(index=1,reset_history=False,midpoint="native_mid/1.bgra")]),
                "meta.json": dict(adapter="test",xefg_family="test",xefg_dll_sha256="test"),
            }
            for name,data in documents.items():
                (root / name).write_text(json.dumps(data),encoding="utf-8")
            pairing = dict(schema="xefg.pairing/2",method="rendered_scene",status="paired",official_tags_requested=True,
                           capture_directory=str(root),artifacts=[],excluded_rects=[[0,0,1,2]],
                           marker=dict(x=0,y=0,cell_height=1,cell_width=1,bits=1),
                           pairs=[dict(classification="generated",previous_frame=0,current_frame=1,sequence=0,file="official.bgra")])
            path = root / "pairing.json"
            path.write_text(json.dumps(pairing),encoding="utf-8")
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text("{}")
            runtime = SimpleNamespace(package=SimpleNamespace(directory=package,variants={}))
            runtime.prepare = lambda *args: None
            calls = []
            def interpolate(previous,current,config):
                calls.append((previous.frame_id,current.frame_id))
                return SimpleNamespace(generated_frame=SimpleNamespace(data=np.zeros_like(official)),profile_id="test")
            runtime.interpolate_pair = interpolate
            args = SimpleNamespace(run=root,pairing=path,out=root / "report",all=True,package=package)
            with patch.object(replay.Xefg2xRuntime,"load") as load_runtime, patch.object(validation,"inspect_run"):
                load_runtime.return_value.__enter__.return_value = runtime
                report = replay.run_replay(args,validation)
            self.assertEqual(calls,[(0,1)])
            row = report["pairs"][0]
            self.assertGreater(row["full"]["rgb_mae"],0)
            self.assertEqual(row["valid_region"]["rgb_mae"],0)
            self.assertEqual(row["official_vs_native_mid"]["rgb_mae"],0)
            self.assertEqual(row["sdk_vs_native_mid"]["rgb_mae"],0)
            self.assertTrue((args.out / "frame_00001_native_mid.png").exists())

    def test_incomplete_pairing_never_loads_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "pairing.json"
            path.write_text(json.dumps({"schema": "xefg.pairing/1", "status": "incomplete"}))
            args = SimpleNamespace(run=root, pairing=path, out=root / "out")
            with patch.object(replay.Xefg2xRuntime, "load") as runtime:
                with self.assertRaisesRegex(ValueError, "complete pairing"):
                    replay.run_replay(args, validation)
                runtime.assert_not_called()

    def test_tagged_run_never_loads_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "pairing.json"
            path.write_text(json.dumps({"schema": "xefg.pairing/1", "status": "paired",
                                        "official_tags_requested": True}))
            args = SimpleNamespace(run=root, pairing=path, out=root / "out")
            with patch.object(replay.Xefg2xRuntime, "load") as runtime:
                with self.assertRaisesRegex(ValueError, "untagged"):
                    replay.run_replay(args, validation)
                runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
