import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

spec = importlib.util.spec_from_file_location(
    "compare_tensors", Path(__file__).parents[1] / "compare_tensors.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def resource(slot, kind, dimension, fmt, width, height, array=1, size=0, file=None):
    return {"slot": slot, "kind": kind, "dimension": dimension, "format": fmt,
            "width": width, "height": height, "array": array, "bytes": size,
            "file": file}


def d24_dispatch(ordinal=24, frame=8, key="d24-key", file="d0024_u0_out.bin"):
    return {
        "type": "dispatch", "dispatch": ordinal, "frame": frame,
        "key": key, "sha256": "a" * 64, "group": [40, 23, 1],
        "constants": [{"parameter": 3, "values": [320, 180, 640, 360, 0, 0, 0, 0, 0, 0]}],
        "resources": [
            resource("t0", "srv", "texture2d", 34, 320, 180),
            resource("t1", "srv", "texture2d", 41, 320, 180),
            resource("u0", "uav", "buffer", 39, 1843200, 1, 1, 1843200, file),
        ],
    }


def d25_dispatch(ordinal=25, frame=8, key="d25-key"):
    return {
        "type": "dispatch", "dispatch": ordinal, "frame": frame,
        "key": key, "sha256": "b" * 64, "group": [40, 12, 1],
        "constants": [{"parameter": 3, "values": list(range(51))}],
        "resources": [
            resource("t0", "srv", "buffer", 39, 471040, 1, 1, 471040),
            resource("t8", "srv", "buffer", 39, 1843200, 1, 1, 1843200),
            resource("u0", "uav", "texture2d", 3, 320, 180, file="d0025_u0_out.bin"),
            resource("u1", "uav", "texture2d", 87, 640, 360, file="d0025_u1_out.bin"),
            resource("u2", "uav", "texture2d", 87, 640, 360, file="d0025_u2_out.bin"),
            resource("u3", "uav", "texture2d", 87, 640, 360, file="d0025_u3_out.bin"),
            resource("u4", "uav", "texture2d", 87, 640, 360, file="d0025_u4_out.bin"),
            resource("t2", "srv", "texture2d", 87, 640, 360),
            resource("t3", "srv", "texture2d", 87, 640, 360),
            resource("t4", "srv", "texture2d", 34, 320, 180),
            resource("t5", "srv", "texture2d", 41, 320, 180),
            resource("t6", "srv", "texture2d", 61, 32, 32),
            resource("b1", "cbv", "buffer", 0, 2560, 1, 1, 256),
            resource("b2", "cbv", "buffer", 0, 1280, 1, 1, 1280),
            resource("b3", "cbv", "buffer", 0, 256, 1, 1, 256),
            resource("b4", "cbv", "buffer", 0, 256, 1, 1, 256),
        ],
    }


def debug_entry(node, semantic, slot, name, shape):
    return {"node": node, "semantic": semantic, "slot": slot,
            "graph_name": name, "dtype": "uint8", "shape": shape}


def debug_map(with_d25_u3=True):
    outputs = [
        debug_entry(24, "splat", "u0", "n0024_u0", ["O0"]),
        debug_entry(25, "warp + flow init", "u0", "n0025_u0", ["O0", "O1", "O2"]),
        debug_entry(25, "warp + flow init", "u1", "n0025_u1", ["O0", "O1", "O2"]),
        debug_entry(25, "warp + flow init", "u2", "n0025_u2", ["O0", "O1", "O2"]),
        debug_entry(38, "generated_frame", "u0", "n0038_u0", ["O0", "O1", "O2"]),
    ]
    if with_d25_u3:
        outputs.append(debug_entry(25, "warp + flow init", "u3", "n0025_u3",
                                   ["O0", "O1", "O2"]))
    outputs[-1 if with_d25_u3 else -1]["public_name"] = None
    for entry in outputs:
        if entry["node"] == 38:
            entry["public_name"] = "generated_frame"
    return {"model": "debug/mode3.onnx", "outputs": outputs}


def write_trace(root, rows):
    trace = Path(root) / "trace"
    (trace / "tensors").mkdir(parents=True)
    with (trace / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return Path(root)


def write_bytes(root, name, size, value=0):
    path = Path(root) / "trace" / "tensors" / name
    path.write_bytes(bytes([value]) * size)
    return path


class MappingTests(unittest.TestCase):
    def test_selects_by_shape_not_global_ordinal(self):
        rows = [d24_dispatch(ordinal=31), d25_dispatch(ordinal=24)]
        selected = compare.select_stage_dispatch(rows, "d24", 8)
        self.assertEqual(selected["dispatch"], 31)

    def test_ambiguous_dispatch_mapping_fails(self):
        rows = [d24_dispatch(ordinal=24), d24_dispatch(ordinal=31, key="other")]
        with self.assertRaisesRegex(compare.ComparisonError, "ambiguous"):
            compare.select_stage_dispatch(rows, "d24", 8)

    def test_missing_dispatch_fails(self):
        rows = [d25_dispatch(ordinal=25)]
        with self.assertRaisesRegex(compare.ComparisonError, "not found"):
            compare.select_stage_dispatch(rows, "d24", 8)

    def test_missing_debug_slot_fails(self):
        with self.assertRaisesRegex(compare.ComparisonError, "missing debug map entries"):
            compare.match_debug_entries(debug_map(with_d25_u3=False), "d25")

    def test_duplicate_debug_slot_fails(self):
        mapping = debug_map()
        mapping["outputs"].append(
            debug_entry(25, "warp + flow init", "u0", "n0025_duplicate", ["O0"]))
        with self.assertRaisesRegex(compare.ComparisonError, "duplicate debug slot"):
            compare.match_debug_entries(mapping, "d25")

    def test_ambiguous_debug_node_fails(self):
        mapping = debug_map(with_d25_u3=False)
        mapping["outputs"].append(
            debug_entry(26, "warp + flow init", "u3", "n0026_u3", ["O0", "O1", "O2"]))
        with self.assertRaisesRegex(compare.ComparisonError, "ambiguous debug node"):
            compare.match_debug_entries(mapping, "d25")

    def test_ordinal_binding_mismatch_fails(self):
        rows = [d24_dispatch(ordinal=31)]
        entries = compare.match_debug_entries(debug_map(), "d24")
        with self.assertRaisesRegex(compare.ComparisonError, "does not match debug node"):
            compare.bind_dispatch_to_debug(rows[0], entries, "d24")

    def test_global_ordinal_binds_through_frame_local_position(self):
        rows = [d24_dispatch(ordinal=290), d25_dispatch(ordinal=291)]
        self.assertEqual(compare.frame_local_index(rows, rows[0]), 1)
        entries = [debug_entry(1, "splat", "u0", "n0024_u0", ["O0"])]
        self.assertEqual(compare.bind_dispatch_to_debug(rows[0], entries, "d24", rows), 1)

    def test_missing_debug_shape_fails(self):
        mapping = debug_map()
        mapping["outputs"][0]["shape"] = []
        with self.assertRaisesRegex(compare.ComparisonError, "missing shape"):
            compare.match_debug_entries(mapping, "d24")

    def test_ambiguous_final_debug_entry_fails(self):
        mapping = debug_map()
        mapping["outputs"].append(
            debug_entry(38, "generated_frame", "u0", "n0038_duplicate", ["O0"]))
        mapping["outputs"][-1]["public_name"] = "generated_frame"
        with self.assertRaisesRegex(compare.ComparisonError, "ambiguous or missing final"):
            compare.debug_final_entry(mapping)


class ShapeAndTensorTests(unittest.TestCase):
    def test_missing_official_tensor_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            run = write_trace(temp, [d24_dispatch()])
            dispatch = compare.select_stage_dispatch(
                compare.recon.read_events(str(run)), "d24", 8)
            with self.assertRaisesRegex(compare.ComparisonError, "missing official tensor"):
                compare.load_official(run, dispatch, "u0")

    def test_missing_d24_shape_constants_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            row = d24_dispatch()
            row["constants"] = []
            run = write_trace(temp, [row])
            with self.assertRaisesRegex(compare.ComparisonError, "shape unavailable"):
                compare.stage_shapes(run, row, "d24")

    def test_missing_texture_shape_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            row = d25_dispatch()
            for item in row["resources"]:
                if item["slot"] == "t4":
                    item["width"] = 0
                    item["height"] = 0
            run = write_trace(temp, [row])
            with self.assertRaisesRegex(compare.ComparisonError, "d25 shape unavailable"):
                compare.stage_shapes(run, row, "d25")

    def test_d25_logical_shapes_and_unmapped_u4(self):
        with tempfile.TemporaryDirectory() as temp:
            run = write_trace(temp, [d25_dispatch()])
            write_bytes(run, "d0025_u0_out.bin", 921600)
            write_bytes(run, "d0025_u1_out.bin", 921600)
            write_bytes(run, "d0025_u2_out.bin", 921600)
            write_bytes(run, "d0025_u3_out.bin", 921600)
            write_bytes(run, "d0025_u4_out.bin", 921600)
            shapes = compare.stage_shapes(run, d25_dispatch(), "d25")
            self.assertEqual(shapes["u0"]["logical_shape"], [180, 320, 16])
            self.assertEqual(shapes["u1"]["logical_shape"], [360, 640, 4])
            entries = compare.match_debug_entries(debug_map(), "d25")
            unmapped = compare._unmapped_slots(run, d25_dispatch(), entries)
            self.assertEqual(unmapped[0]["slot"], "u4")
            self.assertTrue(unmapped[0]["all_zero"])

    def test_debug_tensor_shape_mismatch_fails(self):
        official = np.zeros(32, dtype=np.uint8)
        debug = np.zeros(24, dtype=np.uint8)
        with self.assertRaisesRegex(compare.ComparisonError, "logical shape"):
            compare.compare_bytes(official, debug, [2, 2, 2, 4], 32, "uint8")

    def test_declared_debug_rank_mismatch_fails(self):
        with self.assertRaisesRegex(compare.ComparisonError, "declares rank"):
            compare._check_debug_shape({"graph_name": "n", "shape": ["O0", "O1"]},
                                       np.zeros((2, 2, 4), dtype=np.uint8),
                                       [2, 2, 4])

    def test_first_divergence_and_logical_padding(self):
        official = np.arange(40, dtype=np.uint8)
        debug = official[:32].copy()
        debug[5] = 99
        result = compare.compare_bytes(official, debug, [2, 2, 2, 4], 40, "uint8")
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["first_divergence"]["flat_index"], 5)
        self.assertEqual(result["first_divergence"]["coord"], [0, 0, 1, 1])
        self.assertEqual(result["padding_bytes"], 8)
        self.assertFalse(result["padding_compared"])

    def test_full_allocation_padding_is_compared(self):
        official = np.arange(40, dtype=np.uint8)
        debug = official.copy()
        debug[35] = 7
        result = compare.compare_bytes(official, debug, [2, 2, 2, 4], 40, "uint8")
        self.assertEqual(result["mismatch_count"], 0)
        self.assertTrue(result["padding_compared"])
        self.assertEqual(result["padding_mismatch"], 1)

    def test_int8_debug_values_are_signed(self):
        official = np.array([0, 255], dtype=np.uint8)
        debug = np.array([0, -1], dtype=np.int8)
        result = compare.compare_bytes(official, debug, [2], 2, "int8")
        self.assertEqual(result["mismatch_count"], 0)


class TraceMetaAndIdentityTests(unittest.TestCase):
    def test_explicit_incomplete_trace_is_refused(self):
        with self.assertRaisesRegex(compare.ComparisonError, "complete is false"):
            compare.validate_trace_meta(
                {"schema": "xefg.tensor_trace/1", "complete": False}, 8)

    def test_new_schema_requires_complete_true(self):
        with self.assertRaisesRegex(compare.ComparisonError, "not true"):
            compare.validate_trace_meta({"schema": "xefg.tensor_trace/1"}, 8)

    def test_trace_and_wait_errors_are_refused(self):
        for key in ("trace_error", "wait_error"):
            with self.assertRaisesRegex(compare.ComparisonError, key):
                compare.validate_trace_meta(
                    {"schema": "xefg.tensor_trace/1", "complete": True, key: "boom"}, 8)

    def test_tensor_frame_must_match(self):
        with self.assertRaisesRegex(compare.ComparisonError, "tensor_frame"):
            compare.validate_trace_meta(
                {"schema": "xefg.tensor_trace/1", "complete": True,
                 "tensor_frame": [7, 8]}, 9)
        result = compare.validate_trace_meta(
            {"schema": "xefg.tensor_trace/1", "complete": True,
             "tensor_frame": [7, 8]}, 8)
        self.assertTrue(result["complete"])

    def test_old_meta_allowed_for_inspection_only(self):
        self.assertIsNone(compare.validate_trace_meta({}, 8)["complete"])
        with self.assertRaisesRegex(compare.ComparisonError, "requires a complete"):
            compare.validate_trace_meta({}, 8, require_complete=True)

    def test_reference_program_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = write_trace(Path(temp) / "reference", [d24_dispatch(key="ref-key")])
            package = Path(temp) / "package"
            package.mkdir()
            (package / "manifest.json").write_text(json.dumps(
                {"source": {"captures": [{"run": str(reference)}]}}))
            found = compare.reference_program_hashes(package, "d24")
            self.assertEqual(found["status"], "available")
            self.assertEqual(found["key"], "ref-key")
            with self.assertRaisesRegex(compare.ComparisonError, "program hash mismatch"):
                compare.validate_program_identity(d24_dispatch(key="new-key"), found, "d24")

    def test_missing_reference_program_hash_is_reported_not_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "package"
            package.mkdir()
            (package / "manifest.json").write_text(json.dumps({"source": {"captures": []}}))
            found = compare.reference_program_hashes(package, "d24")
            self.assertEqual(found["status"], "unavailable")
            result = compare.validate_program_identity(d24_dispatch(), found, "d24")
            self.assertEqual(result["status"], "unavailable")


class FinalAndOutputTests(unittest.TestCase):
    def test_final_comparison_reports_first_divergence(self):
        saved = np.zeros((2, 2, 4), dtype=np.uint8)
        debug = saved.copy()
        debug[1, 0, 2] = 5
        result = compare.compare_final(debug, saved)
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["first_divergence"]["coord"], [1, 0, 2])

    def test_final_shape_mismatch_fails(self):
        with self.assertRaisesRegex(compare.ComparisonError, "shape"):
            compare.compare_final(np.zeros((2, 2, 4), dtype=np.uint8),
                                  np.zeros((3, 2, 4), dtype=np.uint8))

    def test_missing_saved_debug_output_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            entries = compare.match_debug_entries(debug_map(), "d24")
            with self.assertRaisesRegex(compare.ComparisonError, "missing debug output"):
                compare.load_debug_outputs(temp, entries)


if __name__ == "__main__":
    unittest.main()
