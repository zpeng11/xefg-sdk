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
                 ui_off=False, trace_config=None, backbuffers=None,
                 trace_backbuffers=None):
        self.root = root
        self.trace_dir = os.path.join(root, "trace")
        self.tensors = os.path.join(self.trace_dir, "tensors")
        os.makedirs(self.tensors)
        self.frames = list(frames)
        first_frame, last_frame = min(self.frames) - 1, max(self.frames) + 1
        self.colors = colors or {
            frame: ("color-%d" % frame).encode()
            for frame in range(first_frame, last_frame + 1)}
        self.motion = motion or {
            frame: ("mv-%d" % frame).encode()
            for frame in range(first_frame, last_frame + 1)}
        self.depth = depth or {
            frame: ("depth-%d" % frame).encode()
            for frame in range(first_frame, last_frame + 1)}
        self.backbuffers = backbuffers or {
            frame: ("backbuffer-%d" % frame).encode()
            for frame in range(first_frame, last_frame + 1)}
        # The trace may deliberately bind a different backbuffer than the
        # application input for a frame, which lets a test prove that a
        # previous-frame binding is reported as a mismatch.
        self.trace_backbuffers = dict(self.backbuffers)
        if trace_backbuffers:
            self.trace_backbuffers.update(trace_backbuffers)
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

    def rewrite_tensor(self, frame, node, slot, data):
        """Replace one captured tensor without changing the trace payload."""
        dispatch = (frame - 1) * 38 + node
        name = "d%04d_%s_in.bin" % (dispatch, slot)
        with open(os.path.join(self.tensors, name), "wb") as handle:
            handle.write(data)

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
            if self.mode == 4:
                backbuffer = self.trace_backbuffers[frame]
                return [
                    self._srv("t8", pointer(current), current, dispatch),
                    self._srv("t9", pointer(backbuffer), backbuffer, dispatch),
                    self._srv("t10", pointer(backbuffer), backbuffer, dispatch),
                    self._srv("t11", pointer(previous), previous, dispatch),
                    self._srv("t12", pointer(current), current, dispatch),
                ]
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
                                 ui_off=None, native=False):
        """Write a manifest and input files; returns the application root.

        ``native=True`` mirrors the capture tool's layout:
        ``<root>/application_inputs/manifest.json`` with file names relative to
        that directory.  The default layout keeps the older oracle fixture
        shape (``<root>/manifest.json`` plus ``inputs/``).
        """
        if ui_off is None:
            ui_off = self.ui_off
        root = os.path.join(self.root, directory)
        if native:
            inputs = os.path.join(root, "application_inputs")
            manifest_dir = inputs
            prefix = ""
        else:
            inputs = os.path.join(root, "inputs")
            manifest_dir = root
            prefix = "inputs/"
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
                "backbuffer": self.backbuffers.get(
                    frame, b"backbuffer-%d" % frame),
                "depth": self.depth[frame],
                "motion": self.motion[frame],
            }
            for key, name in names.items():
                with open(os.path.join(inputs, name), "wb") as handle:
                    handle.write(payloads[key])
            records.append({
                "index": frame,
                "inputs": {key: prefix + name
                           for key, name in names.items()},
            })
        manifest = {
            "schema": "xefg.application_inputs/1",
            "configuration": self.configuration,
            "frames": records,
        }
        if native:
            ordered = sorted(app_frames)
            manifest.update({
                "frame_count": len(ordered),
                "first_frame": ordered[0] if ordered else None,
                "last_frame": ordered[-1] if ordered else None,
                "traced_frames": len(self.frames),
                "frame_semantics": (
                    "index is the application frame including warmup; color "
                    "is the hudless color, backbuffer is the premultiplied UI "
                    "over it"),
            })
        with open(os.path.join(manifest_dir, "manifest.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(manifest, handle)
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


class Mode4ProvenanceTests(unittest.TestCase):
    """Mode 4: t8/current colour plus t10/current backbuffer, never a UI input."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def _report(self, fixture, **kwargs):
        return input_provenance.build_report(
            fixture.trace_path, expected_lut_sha256=sha(fixture.lut),
            **kwargs)

    def test_mode4_uses_t8_and_current_backbuffer_without_ui(self):
        fixture = Fixture(self.root, mode=4)
        app = fixture.write_application_inputs((0, 1, 2, 3))
        report = self._report(fixture, application_inputs=app)

        self.assertEqual(report["status"], "confirmed")
        self.assertEqual(report["d38_mode"]["distinct"], [4])
        self.assertEqual(report["d38_mode"]["spec_mode"], 4)
        self.assertEqual(report["d38_mode"]["boundary_spec"], "mode4")

        inputs = report["boundary"]["inputs"]
        self.assertNotIn("ui", inputs)
        self.assertEqual(set(inputs),
                         {"prev_color", "current_color", "motion", "depth",
                          "backbuffer", "lut"})

        current = inputs["current_color"]
        self.assertEqual(current["status"], "confirmed")
        self.assertEqual(current["frame_shift"], 0)
        self.assertEqual(current["alias_status"], "confirmed")
        self.assertIn({"dispatch": 38, "slot": "t8"}, current["slots"])

        backbuffer = inputs["backbuffer"]
        self.assertEqual(backbuffer["kind"], "backbuffer")
        self.assertEqual(backbuffer["status"], "confirmed")
        self.assertEqual(backbuffer["frame_shift"], 0)
        self.assertEqual(backbuffer["slots"],
                         [{"dispatch": 38, "slot": "t10"}])
        for item in backbuffer["per_frame"]:
            self.assertEqual(item["application_frames"], [item["frame"]])
            self.assertEqual(item["observed_shifts"], [0])

        self.assertEqual(inputs["prev_color"]["frame_shift"], -1)
        self.assertEqual(inputs["motion"]["frame_shift"], 0)
        self.assertEqual(inputs["depth"]["frame_shift"], 0)

        unused = report["boundary"]["unused_slots"]
        self.assertEqual([entry["slot"] for entry in unused], ["d38.t9"])
        self.assertEqual(unused[0]["alias_status"], "confirmed")
        for item in unused[0]["per_frame"]:
            self.assertEqual(item["application_frames"], [item["frame"]])

    def test_mode4_changing_ui_backbuffer_matches_current_not_previous(self):
        # The default fixture gives every frame a different backbuffer, which
        # is the changing-UI case: current and previous backbuffer content
        # differ, so a wrong temporal binding cannot match by accident.
        fixture = Fixture(self.root, mode=4)
        for frame in (1, 2, 3):
            self.assertNotEqual(fixture.backbuffers[frame],
                                fixture.backbuffers[frame - 1])
        app = fixture.write_application_inputs((0, 1, 2, 3))
        report = self._report(fixture, application_inputs=app)

        self.assertEqual(report["status"], "confirmed")
        for item in report["boundary"]["inputs"]["backbuffer"]["per_frame"]:
            self.assertEqual(item["application_frames"], [item["frame"]])
            self.assertEqual(item["observed_shifts"], [0])

    def test_mode4_previous_frame_backbuffer_binding_is_a_mismatch(self):
        # The trace binds frame 2's t9/t10 to frame 1's backbuffer, while the
        # application inputs still describe frame 2.  The current-backbuffer
        # expectation must fail rather than silently accepting the alias.
        fixture = Fixture(
            self.root, mode=4,
            trace_backbuffers={2: b"backbuffer-1"})
        app = fixture.write_application_inputs((0, 1, 2, 3))
        report = self._report(fixture, application_inputs=app)

        self.assertEqual(report["status"], "mismatch")
        backbuffer = report["boundary"]["inputs"]["backbuffer"]
        self.assertEqual(backbuffer["status"], "mismatch")
        frame_two = next(item for item in backbuffer["per_frame"]
                         if item["frame"] == 2)
        self.assertEqual(frame_two["application_frames"], [1])
        self.assertEqual(frame_two["observed_shifts"], [-1])

    def test_mode4_previous_colour_at_t8_breaks_the_alias(self):
        fixture = Fixture(self.root, mode=4)
        fixture.rewrite_tensor(2, 38, "t8", fixture.colors[1])
        report = self._report(fixture)

        current = report["boundary"]["inputs"]["current_color"]
        self.assertEqual(current["alias_status"], "mismatch")
        self.assertEqual(report["status"], "mismatch")

    def test_mode4_frames_without_tensor_snapshots_stay_partial(self):
        # The trace may contain more frames than the retained tensor snapshots
        # (for example traced frames 3..5 after tensors for 1..2).  Those later
        # frames must remain unavailable evidence, not be dropped to force a
        # confirmed top-level status.
        fixture = Fixture(self.root, mode=4, frames=(1, 2, 3, 4, 5))
        for frame in (3, 4, 5):
            for node in range(1, 39):
                dispatch = (frame - 1) * 38 + node
                prefix = "d%04d_" % dispatch
                for name in os.listdir(fixture.tensors):
                    if name.startswith(prefix):
                        os.remove(os.path.join(fixture.tensors, name))
        app = fixture.write_application_inputs((0, 1, 2, 3, 4, 5))
        report = self._report(fixture, application_inputs=app)

        self.assertEqual(report["status"], "partial")
        current = report["boundary"]["inputs"]["current_color"]
        self.assertEqual(current["alias_status"], "partial")
        per_frame = {item["frame"]: item for item in current["per_frame"]}
        for frame in (1, 2):
            self.assertEqual(per_frame[frame]["status"], "unique")
            self.assertEqual(per_frame[frame]["observed_shifts"], [0])
        for frame in (3, 4, 5):
            self.assertEqual(per_frame[frame]["status"], "unavailable")
            self.assertIsNone(per_frame[frame]["content_sha256"])

    def test_native_application_inputs_manifest_is_discovered(self):
        fixture = Fixture(self.root, mode=4)
        root = fixture.write_application_inputs(
            (0, 1, 2, 3), native=True)
        report = self._report(fixture, application_inputs=root)

        self.assertEqual(report["status"], "confirmed")
        self.assertEqual(
            report["application_inputs"]["manifest"],
            os.path.join(root, "application_inputs", "manifest.json"))
        self.assertEqual(report["application_inputs"]["errors"], [])
        self.assertEqual(report["application_inputs"]["kinds"]["backbuffer"],
                         4)

    def test_manifest_with_unusable_paths_falls_back_to_raw_files(self):
        fixture = Fixture(self.root, mode=4)
        root = fixture.write_application_inputs(
            (0, 1, 2, 3), native=True)
        manifest_path = os.path.join(
            root, "application_inputs", "manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for record in manifest["frames"]:
            for key in record["inputs"]:
                record["inputs"][key] = "missing_" + record["inputs"][key]
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        report = self._report(fixture, application_inputs=root)
        self.assertEqual(report["status"], "confirmed")
        self.assertTrue(any("falling back to raw filename discovery" in error
                            for error in report["application_inputs"]["errors"]))
        self.assertEqual(report["application_inputs"]["kinds"]["backbuffer"],
                         4)

    def test_mode1_still_uses_the_ui_boundary(self):
        fixture = Fixture(self.root, mode=1)
        app = fixture.write_application_inputs((0, 1, 2, 3))
        report = self._report(fixture, application_inputs=app)

        self.assertEqual(report["d38_mode"]["spec_mode"], 1)
        self.assertEqual(report["d38_mode"]["boundary_spec"], "mode1_3")
        self.assertIn("ui", report["boundary"]["inputs"])
        self.assertNotIn("backbuffer", report["boundary"]["inputs"])
        self.assertEqual(report["boundary"]["unused_slots"], [])


CAPTURE_ENV = "XEFG_MODE4_CAPTURE"


@unittest.skipUnless(
    os.environ.get(CAPTURE_ENV),
    "set %s to a mode 4 capture root to run the capture-backed report"
    % CAPTURE_ENV)
class Mode4CaptureProvenanceTests(unittest.TestCase):
    """Opt-in check against a real capture; no source-tree paths are assumed."""

    def test_real_mode4_capture_report(self):
        root = os.environ[CAPTURE_ENV]
        trace = os.path.join(root, "trace", "trace.jsonl")
        self.assertTrue(os.path.isfile(trace), trace)
        report = input_provenance.build_report(
            trace, application_inputs=root)

        self.assertEqual(report["d38_mode"]["distinct"], [4])
        self.assertEqual(report["d38_mode"]["spec_mode"], 4)
        inputs = report["boundary"]["inputs"]
        self.assertNotIn("ui", inputs)
        self.assertIn({"dispatch": 38, "slot": "t8"},
                      inputs["current_color"]["slots"])
        # The trace extends past the retained tensor snapshots, so the
        # aggregate status is partial; the traced frames carry the actual
        # shift evidence checked below.
        self.assertEqual(inputs["backbuffer"]["hypothesis_frame_shift"], 0)

        # The capture retains tensors for its traced frames (1 and 2); later
        # trace frames are reported as unavailable rather than guessed.
        for name in ("current_color", "backbuffer", "motion", "depth"):
            traced = [item for item in inputs[name]["per_frame"]
                      if item["frame"] in (1, 2)]
            self.assertEqual(len(traced), 2)
            for item in traced:
                self.assertEqual(item["application_frames"], [item["frame"]])
                self.assertEqual(item["observed_shifts"], [0])

        unused = report["boundary"]["unused_slots"]
        self.assertEqual([entry["slot"] for entry in unused], ["d38.t9"])
        self.assertNotIn("ui", inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
