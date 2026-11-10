"""Positive tests for the XeFG input provenance inventory.

The fixtures are deliberately tiny traces written in the same JSONL shape as
XefgProbe.  They let every assertion be derived from content, not from a
provider run:

  * aliases resolve by frame-relative node id even though dispatch ids are
    global across frames;
  * previous/current colour is proved by ``prev(f) == current(f-1)``;
  * motion/depth frame identity is only claimed when application inputs match;
  * a UI-off trace binds current colour to d38.t9 instead of a UI texture;
  * an app-input manifest can shift trace frame N onto a different application
    frame without the tool assuming N == application frame;
  * alias and missing-tensor failures are reported, not hidden.

Install the package first (``python -m pip install -e ".[test]"``), then run:

    python -I -B -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import tempfile
import unittest

from xefg_sdk.export import input_provenance


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pointer(data):
    return "0x" + sha(data)[:12]


class Fixture:
    """A tiny trace plus optional application inputs and tensor snapshots."""

    def __init__(self, root, frames=(1, 2, 3), colors=None, motion=None,
                 depth=None, ui=b"ui-static", lut=b"lut-constant", mode=3,
                 ui_off=False, trace_config=None):
        self.root = root
        self.trace_dir = os.path.join(root, "trace")
        self.tensors = os.path.join(self.trace_dir, "tensors")
        os.makedirs(self.tensors)
        self.frames = list(frames)
        self.colors = colors or {
            frame: ("color-%d" % frame).encode()
            for frame in range(min(self.frames) - 1, max(self.frames) + 2)}
        self.motion = motion or {
            frame: ("mv-%d" % frame).encode()
            for frame in range(min(self.frames) - 1, max(self.frames) + 2)}
        self.depth = depth or {
            frame: ("depth-%d" % frame).encode()
            for frame in range(min(self.frames) - 1, max(self.frames) + 2)}
        self.ui = ui
        self.lut = lut
        self.mode = mode
        self.ui_off = ui_off
        self.configuration = {
            "width": 64, "height": 36, "motion_width": 32,
            "motion_height": 18, "scene": "standard", "scene_revision": 1,
            "static": False, "ui": not ui_off, "ui_mode": 1 if ui_off else 0,
            "frames": len(self.frames), "warmup": 0,
            "velocity": [12, 4], "flip_mv": False,
        }
        if trace_config:
            self.configuration.update(trace_config)
        self.trace_path = os.path.join(self.trace_dir, "trace.jsonl")
        self._write_trace()
        self._write_meta()

    def _write_tensor(self, dispatch, slot, data, fmt=87, width=64,
                      height=36, dimension="texture2d"):
        name = "d%04d_%s_in.bin" % (dispatch, slot)
        with open(os.path.join(self.tensors, name), "wb") as handle:
            handle.write(data)
        return name

    def _srv(self, slot, resource, data, dispatch, fmt=87, width=64,
             height=36):
        name = self._write_tensor(dispatch, slot, data, fmt, width, height)
        return {
            "parameter": 0,
            "kind": "srv",
            "slot": slot,
            "index": 0,
            "resource": resource,
            "format": fmt,
            "dimension": "texture2d",
            "width": width,
            "height": height,
            "array": 1,
            "mips": 1,
            "file": name,
        }

    def _resources(self, node, dispatch, frame):
        current = self.colors[frame]
        previous = self.colors[frame - 1]
        motion = self.motion[frame]
        depth = self.depth[frame]
        if node == 1:
            return [
                self._srv("t0", pointer(previous), previous, dispatch),
                self._srv("t1", pointer(current), current, dispatch),
            ]
        if node == 24:
            return [
                self._srv("t0", pointer(motion), motion, dispatch,
                          fmt=34, width=32, height=18),
                self._srv("t1", pointer(depth), depth, dispatch,
                          fmt=41, width=32, height=18),
            ]
        if node == 25:
            return [
                self._srv("t2", pointer(previous), previous, dispatch),
                self._srv("t3", pointer(current), current, dispatch),
                self._srv("t4", pointer(motion), motion, dispatch,
                          fmt=34, width=32, height=18),
                self._srv("t5", pointer(depth), depth, dispatch,
                          fmt=41, width=32, height=18),
                self._srv("t6", pointer(self.lut), self.lut, dispatch,
                          fmt=61, width=32, height=32),
            ]
        if node == 38:
            bound_ui = current if self.ui_off else self.ui
            resources = [
                self._srv("t9", pointer(bound_ui), bound_ui, dispatch),
                self._srv("t11", pointer(previous), previous, dispatch),
                self._srv("t12", pointer(current), current, dispatch),
            ]
            if self.ui_off:
                resources.extend([
                    self._srv("t8", pointer(current), current, dispatch),
                    self._srv("t10", pointer(current), current, dispatch),
                ])
            return resources
        return []

    def _write_trace(self):
        sequence = 0
        with open(self.trace_path, "w", encoding="utf-8") as handle:
            for frame in self.frames:
                for node in range(1, 39):
                    dispatch = (frame - 1) * 38 + node
                    sequence += 1
                    constants = []
                    if node == 38:
                        constants = [{
                            "parameter": 8,
                            "values": [0] * 14 + [self.mode, 0, 0, 0],
                        }]
                    event = {
                        "seq": sequence,
                        "type": "dispatch",
                        "dispatch": dispatch,
                        "frame": frame,
                        "present": frame - 1,
                        "constants": constants,
                        "resources": self._resources(node, dispatch, frame),
                    }
                    handle.write(json.dumps(event) + "\n")

    def _write_meta(self):
        with open(os.path.join(self.trace_dir, "meta.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({
                "xefg_family": "fg78",
                "configuration": self.configuration,
            }, handle)

    def write_application_inputs(self, app_frames, directory="app",
                                 ui_off=None):
        """Write a manifest and input files; returns the application root."""
        if ui_off is None:
            ui_off = self.ui_off
        root = os.path.join(self.root, directory)
        inputs = os.path.join(root, "inputs")
        os.makedirs(inputs)
        records = []
        for frame in app_frames:
            names = {
                "color": "color_%05d.bgra" % frame,
                "ui": "ui_%05d.bgra" % frame,
                "backbuffer": "backbuffer_%05d.bgra" % frame,
                "depth": "depth_%05d.r32f" % frame,
                "motion": "mv_%05d.rg16f" % frame,
            }
            payloads = {
                "color": self.colors[frame],
                "ui": b"ui-off-zero" if ui_off else self.ui,
                "backbuffer": b"backbuffer-%d" % frame,
                "depth": self.depth[frame],
                "motion": self.motion[frame],
            }
            for key, name in names.items():
                with open(os.path.join(inputs, name), "wb") as handle:
                    handle.write(payloads[key])
            records.append({
                "index": frame,
                "inputs": {key: "inputs/" + name
                           for key, name in names.items()},
            })
        with open(os.path.join(root, "manifest.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"schema": "xefg.application_inputs/1",
                       "configuration": self.configuration,
                       "frames": records}, handle)
        return root


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def test_aliases_and_trace_relation_without_application_inputs(self):
        fixture = Fixture(self.root)
        expected_lut = sha(fixture.lut)
        report = input_provenance.build_report(
            fixture.trace_path, expected_lut_sha256=expected_lut)

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["d38_mode"]["distinct"], [3])
        inputs = report["boundary"]["inputs"]
        self.assertEqual(inputs["current_color"]["status"], "confirmed")
        self.assertEqual(inputs["current_color"]["frame_shift"], 0)
        self.assertEqual(inputs["prev_color"]["status"], "confirmed")
        self.assertEqual(inputs["prev_color"]["frame_shift"], -1)
        self.assertEqual(inputs["motion"]["status"], "unresolved")
        self.assertIsNone(inputs["motion"]["frame_shift"])
        self.assertEqual(inputs["depth"]["status"], "unresolved")
        self.assertEqual(inputs["ui"]["status"], "frame_invariant")
        self.assertIsNone(inputs["ui"]["frame_shift"])
        self.assertEqual(inputs["lut"]["status"], "confirmed")
        self.assertTrue(inputs["lut"]["matches_expected_sha256"])

        for entry in inputs.values():
            self.assertEqual(entry["alias_status"], "confirmed")

        relation = report["frame_identity"]["prev_color_trace_relation"]
        self.assertEqual(len(relation), 2)
        self.assertTrue(all(item["match"] for item in relation))
        self.assertEqual(report["frame_identity"]["trace_to_application"], {})

    def test_application_inputs_resolve_motion_and_depth(self):
        fixture = Fixture(self.root)
        app = fixture.write_application_inputs((0, 1, 2, 3))
        report = input_provenance.build_report(
            fixture.trace_path, application_inputs=app,
            expected_lut_sha256=sha(fixture.lut))

        self.assertEqual(report["status"], "confirmed")
        self.assertEqual(report["frame_identity"]["trace_to_application"],
                         {"1": 1, "2": 2, "3": 3})
        inputs = report["boundary"]["inputs"]
        self.assertEqual(inputs["prev_color"]["frame_shift"], -1)
        self.assertEqual(inputs["current_color"]["frame_shift"], 0)
        self.assertEqual(inputs["motion"]["frame_shift"], 0)
        self.assertEqual(inputs["motion"]["status"], "confirmed")
        self.assertEqual(inputs["depth"]["frame_shift"], 0)
        self.assertEqual(inputs["depth"]["status"], "confirmed")
        self.assertEqual(inputs["ui"]["status"], "frame_invariant")
        self.assertEqual(inputs["ui"]["frame_shift"], None)
        self.assertEqual(inputs["lut"]["status"], "confirmed")
        for item in inputs["motion"]["per_frame"]:
            self.assertEqual(item["application_frames"], [item["frame"]])

    def test_application_frame_ids_are_discovered_not_assumed(self):
        frames = (1, 2, 3)
        colors = {frame: ("app-color-%d" % (frame + 4)).encode()
                  for frame in range(0, 9)}
        motion = {frame: ("app-mv-%d" % (frame + 4)).encode()
                  for frame in range(0, 9)}
        depth = {frame: ("app-depth-%d" % (frame + 4)).encode()
                 for frame in range(0, 9)}
        fixture = Fixture(
            self.root, frames=frames,
            colors={frame: colors[frame + 4] for frame in range(0, 4)},
            motion={frame: motion[frame + 4] for frame in range(0, 4)},
            depth={frame: depth[frame + 4] for frame in range(0, 4)},
            ui=b"ui-static")
        # Rewrite the fixture's app-side content for application frames 4..8.
        fixture.colors = colors
        fixture.motion = motion
        fixture.depth = depth
        app = fixture.write_application_inputs((4, 5, 6, 7, 8))

        report = input_provenance.build_report(
            fixture.trace_path, application_inputs=app,
            expected_lut_sha256=sha(fixture.lut))
        self.assertEqual(report["status"], "confirmed")
        self.assertEqual(report["frame_identity"]["trace_to_application"],
                         {"1": 5, "2": 6, "3": 7})
        inputs = report["boundary"]["inputs"]
        self.assertEqual(inputs["current_color"]["frame_shift"], 0)
        self.assertEqual(inputs["prev_color"]["frame_shift"], -1)
        self.assertEqual(inputs["motion"]["frame_shift"], 0)
        self.assertEqual(inputs["depth"]["frame_shift"], 0)

    def test_ui_off_d38_t9_is_current_color_placeholder(self):
        fixture = Fixture(self.root, mode=1, ui_off=True)
        app = fixture.write_application_inputs((0, 1, 2, 3))
        report = input_provenance.build_report(
            fixture.trace_path, application_inputs=app,
            expected_lut_sha256=sha(fixture.lut))

        self.assertEqual(report["status"], "confirmed")
        self.assertEqual(report["d38_mode"]["distinct"], [1])
        ui = report["boundary"]["inputs"]["ui"]
        self.assertEqual(ui["status"], "ui_off_current_color_placeholder")
        self.assertEqual(ui["frame_shift"], 0)
        self.assertEqual(ui["frames_matching_current_color"], [1, 2, 3])
        self.assertTrue(any("not an application UI input"
                            in note for note in report["notes"]))

    def test_alias_content_mismatch_is_reported(self):
        fixture = Fixture(self.root)
        # Corrupt the d38.t12 snapshot of frame 2 while keeping the pointer
        # alias: the alias group must report a content mismatch.
        dispatch = (2 - 1) * 38 + 38
        name = "d%04d_t12_in.bin" % dispatch
        with open(os.path.join(fixture.tensors, name), "wb") as handle:
            handle.write(b"not-current-color")

        report = input_provenance.build_report(
            fixture.trace_path, expected_lut_sha256=sha(fixture.lut))
        self.assertEqual(report["status"], "mismatch")
        current = report["boundary"]["inputs"]["current_color"]
        self.assertEqual(current["alias_status"], "mismatch")
        self.assertTrue(report["boundary"]["alias_mismatch"])

    def test_missing_tensor_is_recorded_not_fatal(self):
        fixture = Fixture(self.root)
        dispatch = (1 - 1) * 38 + 38
        os.remove(os.path.join(fixture.tensors,
                               "d%04d_t9_in.bin" % dispatch))
        report = input_provenance.build_report(
            fixture.trace_path, expected_lut_sha256=sha(fixture.lut))
        self.assertEqual(report["status"], "partial")
        ui = report["boundary"]["inputs"]["ui"]
        self.assertEqual(ui["alias_status"], "partial")
        self.assertEqual(ui["status"], "unavailable")
        frame_one = next(item for item in ui["per_frame"]
                         if item["frame"] == 1)
        self.assertEqual(frame_one["status"], "unavailable")
        self.assertEqual(frame_one["aliases"][0]["status"],
                         "missing_tensor")

    def test_static_content_is_frame_invariant_not_guessed(self):
        static_color = b"static-color"
        static_motion = b"static-motion"
        static_depth = b"static-depth"
        fixture = Fixture(
            self.root,
            colors={frame: static_color for frame in (0, 1, 2, 3)},
            motion={frame: static_motion for frame in (0, 1, 2, 3)},
            depth={frame: static_depth for frame in (0, 1, 2, 3)},
            ui=b"ui-static")
        app = fixture.write_application_inputs((0, 1, 2, 3))
        report = input_provenance.build_report(
            fixture.trace_path, application_inputs=app,
            expected_lut_sha256=sha(fixture.lut))

        self.assertEqual(report["status"], "confirmed")
        inputs = report["boundary"]["inputs"]
        self.assertEqual(inputs["current_color"]["status"], "frame_invariant")
        self.assertIsNone(inputs["current_color"]["frame_shift"])
        self.assertEqual(inputs["motion"]["status"], "frame_invariant")
        self.assertIsNone(inputs["motion"]["frame_shift"])
        self.assertEqual(inputs["depth"]["status"], "frame_invariant")
        self.assertEqual(inputs["ui"]["status"], "frame_invariant")
        self.assertEqual(report["frame_identity"]["trace_to_application"], {})

    def test_lut_expected_hash_mismatch_is_reported(self):
        fixture = Fixture(self.root)
        report = input_provenance.build_report(
            fixture.trace_path, expected_lut_sha256="0" * 64)
        self.assertEqual(report["status"], "mismatch")
        lut = report["boundary"]["inputs"]["lut"]
        self.assertEqual(lut["status"], "mismatch")
        self.assertFalse(lut["matches_expected_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
