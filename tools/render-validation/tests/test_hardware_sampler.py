"""Tests for the offline hardware sampler diagnostic.

These tests are self-contained: they use a synthetic d25 input dictionary and
never read an external capture or run the native helper.  They pin the packet
framing, truncation rejection, id/count layout, float32 UV echo, the BGRA8
upload -> RGBA sample/Load meaning, and the explicit one-case CLI contract.
"""
import argparse
import contextlib
import importlib.util
import io
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xefg_sdk.export import recon

REPO = Path(__file__).resolve().parents[3]
DRIVER = REPO / "tools" / "render-validation" / "probe_hardware_sampler.py"
spec = importlib.util.spec_from_file_location("probe_hardware_sampler", DRIVER)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _synthetic_data():
    """A tiny 2x2-cell d25 input with 4x4 t2/t3 BGRA textures."""
    c = [0] * 51
    c[0], c[1], c[2], c[3] = 0, 0, 4, 4
    c[4], c[5], c[6], c[7] = 0, 0, 4, 4
    c[36], c[37], c[38], c[39] = 2, 2, 4, 4
    c[42], c[43] = _f32_bits(0.25), _f32_bits(0.25)
    c[46] = 2
    c[47] = _f32_bits(1.0)
    c[48] = 0
    c[50] = _f32_bits(0.5)

    tex2 = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    tex3 = (255 - np.arange(4 * 4 * 4, dtype=np.uint8)).reshape(4, 4, 4)
    return {
        "tw": 2, "th": 2, "ow": 4, "oh": 4,
        "plane_w": 2, "plane_h": 2, "stride": 2, "c": c,
        "t0": np.zeros((2, 2, 4), dtype=np.float32),
        "t2": tex2, "t3": tex3,
        "t4": np.zeros((2, 2, 2), dtype=np.float32),
        "t5": np.zeros((2, 2), dtype=np.float32),
        "t6": np.ones((32, 32), dtype=np.float32),
        "plane0": np.zeros((2, 2), dtype=np.uint32),
        "plane1": np.zeros((2, 2), dtype=np.uint32),
        "weights": np.arange(80 * 16, dtype=np.int8).reshape(80, 16),
        "scale": np.linspace(0.01, 0.16, 16, dtype=np.float32),
        "bias": np.linspace(-1.0, 1.0, 16, dtype=np.float32),
        "u0": np.zeros((2, 2, 16), dtype=np.uint8),
        "u1": np.zeros((4, 4, 4), dtype=np.uint8),
        "u2": np.zeros((4, 4, 4), dtype=np.uint8),
        "u3": np.zeros((4, 4, 4), dtype=np.uint8),
        "u4": np.zeros((4, 4, 4), dtype=np.uint8),
    }


def _all_cells(data):
    return np.ones((data["th"], data["tw"]), dtype=bool)


def _packets(data, mask=None):
    mask = _all_cells(data) if mask is None else mask
    tables = probe._collect_calls(data)
    return probe._pack_requests(tables, data, mask), tables


def _sample_records(request_records, normalized=0.25, units=63.75, truncated=63):
    records = np.zeros(len(request_records), dtype=probe.SAMPLE_DTYPE)
    records["id"] = request_records["id"]
    records["op"] = request_records["op"]
    records["u"] = request_records["u"]
    records["v"] = request_records["v"]
    records["normalized"] = np.float32(normalized)
    records["units"] = np.float32(units)
    records["truncated"] = int(truncated)
    return records


class PacketLayoutTests(unittest.TestCase):
    def test_all_sample_and_load_calls_are_present_for_all_four_lanes(self):
        data = _synthetic_data()
        packets, tables = _packets(data)
        cells = data["th"] * data["tw"]

        for tex in ("t2", "t3"):
            packet = packets[tex]
            self.assertEqual(packet["width"], data["ow"])
            self.assertEqual(packet["height"], data["oh"])
            self.assertEqual(len(packet["records"]), 12 * cells)
            self.assertEqual(sum(span["count"] for span in packet["spans"]
                                 if span["op"] == probe.OP_SAMPLE), 8 * cells)
            self.assertEqual(sum(span["count"] for span in packet["spans"]
                                 if span["op"] == probe.OP_LOAD), 4 * cells)
            self.assertEqual(list(packet["records"]["id"]),
                             list(range(12 * cells)))

            sample_roles = {span["role"] for span in packet["spans"]
                            if span["op"] == probe.OP_SAMPLE}
            load_roles = {span["role"] for span in packet["spans"]
                          if span["op"] == probe.OP_LOAD}
            if tex == "t2":
                self.assertEqual(sample_roles, {"centre2", "warp2"})
                self.assertEqual(load_roles, {"self2"})
            else:
                self.assertEqual(sample_roles, {"centre3", "warp3"})
                self.assertEqual(load_roles, {"self3"})
            for lane in range(4):
                for role in sample_roles | load_roles:
                    self.assertTrue(any(span["ax"] == lane & 1
                                        and span["ay"] == lane >> 1
                                        and span["role"] == role
                                        for span in packet["spans"]))

            # The two textures together produce the contract's 16 Sample and
            # 8 Load calls per cell, and every id is unique.
            self.assertEqual(len(set(packet["records"]["id"].tolist())),
                             len(packet["records"]))
        self.assertEqual(len(tables["t2"]), 12)
        self.assertEqual(len(tables["t3"]), 12)

    def test_request_packet_header_and_record_sizes(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        for tex in ("t2", "t3"):
            records = packets[tex]["records"]
            raw = probe.request_bytes(records)
            self.assertEqual(probe.REQUEST_DTYPE.itemsize, 32)
            self.assertEqual(len(raw), 16 + 32 * len(records))
            magic, version, count, stride = struct.unpack_from("<4I", raw, 0)
            self.assertEqual((magic, version, count, stride),
                             (0x58534647, 1, len(records), 32))
            parsed = probe.read_requests_bytes(raw)
            self.assertEqual(parsed.dtype.itemsize, 32)
            self.assertTrue(np.array_equal(parsed["id"], records["id"]))
            self.assertTrue(np.array_equal(parsed["op"], records["op"]))
            self.assertTrue(np.array_equal(parsed["u"].view("<u4"),
                                          records["u"].view("<u4")))

    def test_direct_load_roles_carry_the_shader_integer_coordinates(self):
        data = _synthetic_data()
        packets, tables = _packets(data)
        entry = next(item for item in tables["t2"]
                     if item["op"] == probe.OP_LOAD and item["role"] == "self2"
                     and item["ax"] == 1 and item["ay"] == 0)
        span = next(item for item in packets["t2"]["spans"]
                    if item["op"] == probe.OP_LOAD and item["role"] == "self2"
                    and item["ax"] == 1 and item["ay"] == 0)
        block = packets["t2"]["records"][span["start"]:span["start"] + span["count"]]
        cells_y, cells_x = np.nonzero(_all_cells(data))
        self.assertTrue(np.array_equal(block["x"], entry["x"][cells_y, cells_x]))
        self.assertTrue(np.array_equal(block["y"], entry["y"][cells_y, cells_x]))
        self.assertTrue(np.array_equal(block["u"], np.zeros(span["count"], np.float32)))
        self.assertTrue(np.array_equal(block["v"], np.zeros(span["count"], np.float32)))


class UvEchoTests(unittest.TestCase):
    def test_uv_is_the_float32_pos_times_reciprocal(self):
        data = _synthetic_data()
        packets, tables = _packets(data)
        entry = tables["t2"][0]
        span = packets["t2"]["spans"][0]
        cells_y, cells_x = np.nonzero(_all_cells(data))
        block = packets["t2"]["records"][span["start"]:span["start"] + span["count"]]
        expected_u = (entry["pos_x"].astype(np.float32)
                      * (np.float32(1.0) / np.float32(data["ow"]))).astype(np.float32)
        expected_v = (entry["pos_y"].astype(np.float32)
                      * (np.float32(1.0) / np.float32(data["oh"]))).astype(np.float32)
        self.assertTrue(np.array_equal(block["u"].view("<u4"),
                                      expected_u[cells_y, cells_x].view("<u4")))
        self.assertTrue(np.array_equal(block["v"].view("<u4"),
                                      expected_v[cells_y, cells_x].view("<u4")))
        self.assertEqual(block["u"].dtype, np.dtype("<f4"))
        self.assertEqual(block["v"].dtype, np.dtype("<f4"))

    def test_echo_requires_identical_float32_uv_bits(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        request = packets["t2"]["records"][:15]
        records = _sample_records(request)
        probe._assert_echo(request, records, "test")
        records["u"][3] = np.nextafter(records["u"][3], np.float32(2.0))
        with self.assertRaises(probe.ProbeError):
            probe._assert_echo(request, records, "test")

    def test_echo_rejects_wrong_ids_and_ops(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        request = packets["t2"]["records"][:10]
        records = _sample_records(request)
        records["id"][4] = 99
        with self.assertRaises(probe.ProbeError):
            probe._assert_echo(request, records, "test")
        records = _sample_records(request)
        records["op"][4] = probe.OP_LOAD
        with self.assertRaises(probe.ProbeError):
            probe._assert_echo(request, records, "test")


class SamplePacketTests(unittest.TestCase):
    def test_sample_header_count_stride_and_ids(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        request = packets["t2"]["records"]
        records = _sample_records(request)
        raw = probe.sample_bytes(records)
        self.assertEqual(probe.SAMPLE_DTYPE.itemsize, 80)
        magic, version, count, stride = struct.unpack_from("<4I", raw, 0)
        self.assertEqual((magic, version, count, stride),
                         (0x58534647, 1, len(records), 80))
        self.assertEqual(len(raw), 16 + 80 * len(records))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.bin"
            path.write_bytes(raw)
            parsed = probe.read_samples(path, expected_count=len(records))
            self.assertEqual(list(parsed["id"]), list(range(len(records))))
            self.assertEqual(parsed["normalized"].shape, (len(records), 4))
            self.assertEqual(parsed["units"].shape, (len(records), 4))
            self.assertEqual(parsed["truncated"].shape, (len(records), 4))
            self.assertEqual(parsed["padding"].shape, (len(records), 4))

    def test_truncated_packets_are_rejected(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        request = packets["t2"]["records"]
        records = _sample_records(request)
        with tempfile.TemporaryDirectory() as temp:
            request_path = Path(temp) / "requests.bin"
            request_path.write_bytes(probe.request_bytes(request))
            raw = request_path.read_bytes()
            request_path.write_bytes(raw[:-1])
            with self.assertRaises(probe.ProbeError):
                probe.read_requests(request_path)
            request_path.write_bytes(raw[:8])
            with self.assertRaises(probe.ProbeError):
                probe.read_requests(request_path)
            request_path.write_bytes(raw + b"\0")
            with self.assertRaises(probe.ProbeError):
                probe.read_requests(request_path)

            sample_path = Path(temp) / "samples.bin"
            sample_path.write_bytes(probe.sample_bytes(records))
            sample_raw = sample_path.read_bytes()
            sample_path.write_bytes(sample_raw[:-1])
            with self.assertRaises(probe.ProbeError):
                probe.read_samples(sample_path)
            sample_path.write_bytes(sample_raw[:16])
            with self.assertRaises(probe.ProbeError):
                probe.read_samples(sample_path)
            sample_path.write_bytes(sample_raw)
            with self.assertRaises(probe.ProbeError):
                probe.read_samples(sample_path, expected_count=len(records) + 1)

    def test_wrong_magic_version_and_stride_are_rejected(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        request = packets["t2"]["records"]
        raw = bytearray(probe.request_bytes(request))
        for offset, value in ((0, 0xDEADBEEF), (4, 2), (12, 64)):
            bad = bytearray(raw)
            struct.pack_into("<I", bad, offset, value)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "requests.bin"
                path.write_bytes(bytes(bad))
                with self.assertRaises(probe.ProbeError):
                    probe.read_requests(path)

        records = _sample_records(request)
        sample_raw = bytearray(probe.sample_bytes(records))
        for offset, value in ((0, 0), (4, 9), (12, 32)):
            bad = bytearray(sample_raw)
            struct.pack_into("<I", bad, offset, value)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "samples.bin"
                path.write_bytes(bytes(bad))
                with self.assertRaises(probe.ProbeError):
                    probe.read_samples(path)

    def test_duplicate_ids_are_rejected(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        records = _sample_records(packets["t2"]["records"])
        records["id"][1] = records["id"][0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.bin"
            path.write_bytes(probe.sample_bytes(records))
            with self.assertRaises(probe.ProbeError):
                probe.read_samples(path)


class ReplayMeasurementTests(unittest.TestCase):
    def test_measured_units_replay_matches_the_exact_replay(self):
        data = _synthetic_data()
        packets, tables = _packets(data)
        records = {}
        for tex in ("t2", "t3"):
            request = packets[tex]["records"]
            block = _sample_records(request)
            cells_y = packets[tex]["cells_y"]
            cells_x = packets[tex]["cells_x"]
            for entry, span in zip(tables[tex], packets[tex]["spans"]):
                piece = block[span["start"]:span["start"] + span["count"]]
                if entry["op"] == probe.OP_SAMPLE:
                    units = recon._warp_init_sample(
                        data[tex],
                        entry["pos_x"][cells_y, cells_x],
                        entry["pos_y"][cells_y, cells_x])
                else:
                    units = probe._direct_texture_load(
                        data[tex], entry["x"][cells_y, cells_x],
                        entry["y"][cells_y, cells_x])
                piece["units"] = np.asarray(units, dtype=np.float32)
                piece["truncated"] = np.trunc(piece["units"]).astype(np.int32)
                piece["normalized"] = (piece["units"] / np.float32(255.0)).astype(np.float32)
            records[tex] = block

        lookup, _raw = probe._build_lookup(records, packets, data, "units",
                                           want_raw=True)
        measured = probe._run_measured(data, lookup)
        exact = probe._replay_d25(data, probe._exact_sample_hook(data),
                                  probe._exact_load_hook(data))
        self.assertTrue(np.array_equal(measured["got0"], exact["got0"]))
        self.assertEqual(measured["got0"].tolist(), exact["got0"].tolist())
        measured_packs = probe._pack_outputs(
            measured["pack"], data["th"], data["tw"], data["oh"], data["ow"],
            "half_up")
        exact_packs = probe._pack_outputs(
            exact["pack"], data["th"], data["tw"], data["oh"], data["ow"],
            "half_up")
        for name in ("u1", "u2", "u3"):
            self.assertTrue(np.array_equal(measured_packs[name], exact_packs[name]))
        self.assertEqual(measured["values"].tolist(), exact["values"].tolist())

    def test_packing_report_exposes_candidates_and_selected_pack(self):
        data = _synthetic_data()
        exact = probe._replay_d25(data, probe._exact_sample_hook(data),
                                  probe._exact_load_hook(data))
        exact_packs = probe._pack_outputs(
            exact["pack"], data["th"], data["tw"], data["oh"], data["ow"],
            "half_up")
        official = probe._official_arrays(data)
        mask = _all_cells(data)
        report = probe._packing_report(exact, official, exact, exact_packs,
                                       mask, data["oh"], data["ow"])
        self.assertEqual(set(report["candidates"]), set(probe.PACKING_CANDIDATES))
        self.assertEqual(report["production_unorm8"]["candidate"], "half_up")
        self.assertEqual(report["selected_candidate"], "half_up")
        self.assertIn("u1", report["candidates"]["trunc"]["counts_official"])
        self.assertEqual(set(report["selected_packs"]), {"u1", "u2", "u3"})
        self.assertIn("u0", report["selected_counts_official"])
        self.assertIn("u1", report["selected_counts_official"])

    def test_lane_contribution_audit_covers_all_four_lanes(self):
        data = _synthetic_data()
        exact = probe._replay_d25(data, probe._exact_sample_hook(data),
                                  probe._exact_load_hook(data))
        audit = probe._lane_contribution_details(exact, exact, data, 1, 1, 4)
        self.assertTrue(audit["matches_accumulator_delta"])
        self.assertEqual(audit["delta_acc"], 0)
        self.assertEqual(audit["delta_acc_from_lane_changes"], 0)
        self.assertEqual([lane["lane"] for lane in audit["lanes"]],
                         [0, 1, 2, 3])
        self.assertTrue(all(not lane["changes"] for lane in audit["lanes"]))

    def test_saved_measured_arrays_and_full_u0_difference_are_reported(self):
        data = _synthetic_data()
        exact = probe._replay_d25(data, probe._exact_sample_hook(data),
                                  probe._exact_load_hook(data))
        exact_packs = probe._pack_outputs(
            exact["pack"], data["th"], data["tw"], data["oh"], data["ow"],
            "half_up")
        official = {"u0": exact["got0"]}
        mask = _all_cells(data)
        mismatches = probe._u0_mismatch_details(exact, exact, official, mask)
        self.assertEqual(mismatches["total_bytes"], data["th"] * data["tw"] * 16)
        self.assertEqual(mismatches["region_bytes"], data["th"] * data["tw"] * 16)
        self.assertEqual(mismatches["count"], 0)
        with tempfile.TemporaryDirectory() as temp:
            case_dir = Path(temp) / "case"
            case_dir.mkdir()
            saved = probe._save_reconstructed_arrays(
                case_dir, exact, exact_packs, exact)
            self.assertEqual(set(saved), {
                "u0_measured_units", "u0_production_baseline",
                "u1_production_unorm8", "u2_production_unorm8",
                "u3_production_unorm8"})
            for name, entry in saved.items():
                path = Path(temp) / entry["file"]
                self.assertTrue(path.is_file())
                self.assertEqual(np.load(path, allow_pickle=False).shape,
                                 tuple(entry["shape"]))
                self.assertEqual(entry["sha256"], probe.sha256_file(path))


class BgraMeaningTests(unittest.TestCase):
    def test_direct_load_returns_rgba_from_bgra_memory(self):
        texture = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
        loaded = probe._direct_texture_load(texture, 0, 0)
        self.assertEqual(loaded.tolist(), [30.0, 20.0, 10.0, 40.0])

    def test_one_texel_sample_returns_rgba_from_bgra_memory(self):
        texture = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
        sampled = recon._warp_init_sample(
            texture, np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32))
        self.assertEqual(sampled[0, 0].tolist(), [30.0, 20.0, 10.0, 40.0])

    def test_hardware_units_keep_the_native_rgba_order(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        records = {
            tex: _sample_records(packets[tex]["records"], units=42.5)
            for tex in ("t2", "t3")
        }
        span = packets["t2"]["spans"][0]
        records["t2"]["units"][span["start"]:span["start"] + span["count"]] = \
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        lookup, raw = probe._build_lookup(records, packets, data, "units",
                                          want_raw=True)
        key = (span["tex"], span["role"], span["ax"], span["ay"])
        y, x = int(packets["t2"]["cells_y"][0]), int(packets["t2"]["cells_x"][0])
        self.assertEqual(lookup[key][y, x].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(raw[key]["units"][y, x].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_feature_modes_use_measured_values_not_recomputed_filtering(self):
        data = _synthetic_data()
        packets, _tables = _packets(data)
        records = {}
        for tex in ("t2", "t3"):
            block = _sample_records(packets[tex]["records"],
                                    normalized=np.float32(0.2),
                                    units=np.float32(51.0),
                                    truncated=50)
            records[tex] = block
        units_lookup, _ = probe._build_lookup(records, packets, data, "units")
        trunc_lookup, _ = probe._build_lookup(records, packets, data, "trunc")
        norm_lookup, _ = probe._build_lookup(records, packets, data, "norm255")
        span = packets["t2"]["spans"][0]
        key = (span["tex"], span["role"], span["ax"], span["ay"])
        y, x = int(packets["t2"]["cells_y"][0]), int(packets["t2"]["cells_x"][0])
        self.assertEqual(units_lookup[key][y, x].tolist(), [51.0] * 4)
        self.assertEqual(trunc_lookup[key][y, x].tolist(), [50.0] * 4)
        expected = np.float32(np.float32(0.2) * np.float32(255.0))
        self.assertEqual(norm_lookup[key][y, x].tolist(), [float(expected)] * 4)

    def test_region_mask_covers_all_lanes_around_a_target(self):
        data = _synthetic_data()
        spec = {"name": "tiny", "targets": ((1, 1, 2),)}
        mask = probe._region_mask(data, spec, "targets", radius=0)
        self.assertEqual(mask.tolist(), [[False, False], [False, True]])
        packets, _tables = _packets(data, mask)
        for tex in ("t2", "t3"):
            self.assertEqual(len(packets[tex]["records"]), 12)
            spans = packets[tex]["spans"]
            self.assertEqual({span["ax"] for span in spans}, {0, 1})
            self.assertEqual({span["ay"] for span in spans}, {0, 1})


class CaseSpecTests(unittest.TestCase):
    def test_case_spec_has_no_builtin_capture_or_target(self):
        spec = probe._case_spec("PATH/TO/CAPTURE", 3)
        self.assertEqual(spec["name"], "CAPTURE_frame3")
        self.assertEqual(spec["run"], Path("PATH/TO/CAPTURE"))
        self.assertEqual(spec["frame"], 3)
        self.assertEqual(spec["targets"], ())
        self.assertEqual(spec["default_region"], "full")
        self.assertFalse(spec["require_complete"])

    def test_case_spec_uses_explicit_name_targets_and_completeness(self):
        spec = probe._case_spec(
            "PATH/TO/CAPTURE", 8, "frame8", ((34, 51, 4), (135, 211, 3)), True)
        self.assertEqual(spec["name"], "frame8")
        self.assertEqual(spec["targets"], ((34, 51, 4), (135, 211, 3)))
        self.assertEqual(spec["default_region"], "targets")
        self.assertTrue(spec["require_complete"])

    def test_case_spec_rejects_invalid_names_frames_and_targets(self):
        for name in ("", ".", "..", "a/b", "a\\b", "a:b", "a*b"):
            with self.assertRaises(probe.ProbeError):
                probe._case_spec("PATH/TO/CAPTURE", 1, name)
        with self.assertRaises(probe.ProbeError):
            probe._case_spec("PATH/TO/CAPTURE", -1)
        for target in ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 16)):
            with self.assertRaises(probe.ProbeError):
                probe._case_spec("PATH/TO/CAPTURE", 1, targets=(target,))

    def test_parse_target_accepts_only_y_x_ch(self):
        self.assertEqual(probe._parse_target("34,51,4"), (34, 51, 4))
        for text in ("1,2", "1,2,3,4", "a,b,c", "-1,2,3", "1,-2,3", "1,2,16"):
            with self.assertRaises(argparse.ArgumentTypeError):
                probe._parse_target(text)

    def test_spec_from_manifest_case_restores_gate_expectation(self):
        case = {
            "name": "frame8",
            "run": "PATH/TO/CAPTURE",
            "frame": 8,
            "targets": [[34, 51, 4]],
            "require_complete": True,
            "gate": {"dispatch": 291, "key": "123", "sha256": "abc"},
        }
        spec = probe._spec_from_case(case)
        self.assertEqual(spec["name"], "frame8")
        self.assertEqual(spec["run"], Path("PATH/TO/CAPTURE"))
        self.assertEqual(spec["targets"], ((34, 51, 4),))
        self.assertTrue(spec["require_complete"])
        self.assertEqual(spec["expect"],
                         {"dispatch": 291, "key": "123", "sha256": "abc"})

    def test_driver_contains_no_old_capture_defaults(self):
        source = DRIVER.read_text(encoding="utf-8")
        for marker in ("E:/xefg-", "D:/xefg-", "xefg-nonrigid-tensors",
                       "xefg-capture-standalone", "TARGET_SPEC", "CONTROL_SPECS"):
            self.assertNotIn(marker, source)


class CliTests(unittest.TestCase):
    def test_prepare_cli_accepts_one_explicit_case(self):
        parser = probe.build_parser()
        args = parser.parse_args([
            "prepare",
            "--run", "PATH/TO/CAPTURE",
            "--frame", "8",
            "--out", "PATH/TO/NEW/BUNDLE",
            "--name", "frame8",
            "--target", "34,51,4",
            "--target", "135,211,3",
            "--require-complete",
        ])
        self.assertEqual(args.run, Path("PATH/TO/CAPTURE"))
        self.assertEqual(args.frame, 8)
        self.assertEqual(args.target, [(34, 51, 4), (135, 211, 3)])
        spec = probe._spec_from_args(parser, args)
        self.assertEqual(spec["name"], "frame8")
        self.assertEqual(spec["default_region"], "targets")
        self.assertTrue(spec["require_complete"])

    def test_prepare_cli_requires_run_and_frame(self):
        parser = probe.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["prepare", "--out", "PATH/TO/NEW/BUNDLE"])

    def test_run_cli_can_execute_an_existing_bundle_without_case_args(self):
        parser = probe.build_parser()
        args = parser.parse_args([
            "run",
            "--out", "PATH/TO/BUNDLE",
            "--exe", "PATH/TO/XefgSamplerProbe.exe",
        ])
        self.assertIsNone(args.run)
        self.assertIsNone(args.frame)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                probe._spec_from_args(parser, args)

    def test_analyze_cli_needs_only_the_bundle(self):
        parser = probe.build_parser()
        args = parser.parse_args(["analyze", "--out", "PATH/TO/BUNDLE"])
        self.assertEqual(args.out, Path("PATH/TO/BUNDLE"))


if __name__ == "__main__":
    unittest.main()
