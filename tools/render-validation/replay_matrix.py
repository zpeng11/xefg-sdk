"""Resumable full rendered-replay matrix over the repaired SDK package.

The runner maps existing rotation captures from one or more render-matrix
runs, renders only the known-missing combinations, replays every capture
through the public SDK, applies the numeric quality gate and retains a compact
report plus representative official/SDK/difference images.

It never writes to, or deletes, a source capture matrix or package.  Only the
``work`` directory below ``--out`` is removed, and only after a combination has
been checkpointed and its images copied.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


TOOLS = Path(__file__).resolve().parent
SCHEMA = "xefg.replay_matrix/1"
STATUS_SCHEMA = "xefg.replay_matrix_status/1"
STATE_SCHEMA = "xefg.replay_matrix_state/1"
CHECKPOINT_SCHEMA = "xefg.replay_combo/1"
SOURCE_SCHEMAS = ("xefg.render_matrix/3", "xefg.render_matrix/4")
SCENES = ("translation", "occlusion", "camera", "cut", "depth_layers",
          "screen_cross", "camera_turn", "nonrigid", "transparency")
SIZES = ("640x360", "1280x720", "1920x1080", "2560x1440", "3840x2160")
MODES = (1, 3, 4)
EXPECTED_COMBOS = 135
EXPECTED_FRAMES = 1065
CUT_SKIP_SOURCE_FRAME = 7
SOURCE_FRAMES = tuple(range(4, 12))
ALLOWED_MISSING_SCENES = ("nonrigid", "transparency")
ALLOWED_MISSING_SIZES = SIZES[1:]
DEFAULT_QUALITY = {"mae": 0.015, "maximum": 40, "fraction": 0.0015}
EXIT_CODES = {"pass": 0, "run_failed": 1, "quality_failed": 2, "coverage_failed": 3}
COMPLETE_STATUSES = ("pass", "quality_failed", "coverage_failed")
METRIC_KEYS = (("max_rgb_mae", "rgb_mae"),
               ("max_rgb_max", "rgb_max"),
               ("max_over_2_fraction", "over_tolerance_pixel_fraction"))
FRAME_FIELDS = ("previous_frame", "current_frame", "local_previous_frame",
                "local_current_frame", "source_previous_frame", "source_current_frame",
                "source_frame", "previous_time", "current_time", "source_pair", "mode",
                "ui_mode", "ui_mode_name", "present_sequence", "profile", "output_size",
                "motion_size", "profile_details", "wall_seconds_including_first_load",
                "full", "valid_region", "ui_region", "ui_region_definition",
                "sdk_vs_native_mid", "official_vs_native_mid")


class MatrixError(Exception):
    """Base error for a run that cannot proceed safely."""


class DuplicateComboError(MatrixError):
    """Two source matrices claim the same scene/size/mode combination."""


class StalePackageError(MatrixError):
    """The package changed since the resumable state was written."""


class UnsafeDeleteError(MatrixError):
    """A delete request escaped the owned work directory."""


class PreflightError(MatrixError):
    """The expected 135-combination/1065-frame inventory is not satisfied."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.%d.tmp" % (path.name, os.getpid(), time.time_ns()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_tool(name):
    spec = importlib.util.spec_from_file_location("replay_matrix_" + name,
                                                  TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def combo_key(scene, size, mode):
    return "%s__%s__mode%d" % (scene, size, int(mode))


def parse_combo_key(key):
    scene, size, mode = key.split("__")
    return scene, size, int(mode.replace("mode", ""))


def expected_pair_frames(scene):
    """Mode 1 source pair ends; the camera-cut crossing at frame 7 is skipped."""
    return [frame for frame in SOURCE_FRAMES
            if not (scene == "cut" and frame == CUT_SKIP_SOURCE_FRAME)]


def expected_frames(scene):
    return len(expected_pair_frames(scene))


def expected_case_count(scene, mode):
    return len(expected_pair_frames(scene)) if int(mode) == 1 else 1


def expected_combo_keys(scenes=SCENES, sizes=SIZES, modes=MODES):
    return [combo_key(scene, size, mode)
            for size in sizes for mode in modes for scene in scenes]


def expected_total_frames(scenes=SCENES, sizes=SIZES, modes=MODES):
    return sum(expected_frames(scene) for scene in scenes
               for _size in sizes for _mode in modes)


def allowed_missing_keys():
    """Only the two new scenes at the four non-640 sizes may need rendering."""
    return {combo_key(scene, size, mode) for scene in ALLOWED_MISSING_SCENES
            for size in ALLOWED_MISSING_SIZES for mode in MODES}


def _matrix_path(path):
    path = Path(path)
    return path / "matrix.json" if path.is_dir() else path


def _validate_source_rows(matrix, matrix_path):
    rows = matrix.get("scenes")
    if not isinstance(rows, list) or not rows:
        raise PreflightError("%s: matrix has no scene rows" % matrix_path)
    by_size = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PreflightError("%s: malformed scene row" % matrix_path)
        scene, size, mode = row.get("scene"), row.get("size"), row.get("mode")
        if scene not in SCENES or size not in SIZES or mode not in MODES:
            raise PreflightError("%s: unsupported scene/size/mode %r/%r/%r"
                                 % (matrix_path, scene, size, mode))
        cases = row.get("cases")
        if not isinstance(cases, list) or not cases:
            raise PreflightError("%s: %s/%s/mode%s has no case paths"
                                 % (matrix_path, scene, size, mode))
        for case in cases:
            if not isinstance(case, dict) or not case.get("directory"):
                raise PreflightError("%s: case without directory" % matrix_path)
        by_size.setdefault(size, []).append(row)
    size_results = matrix.get("size_results")
    if size_results is not None:
        if not isinstance(size_results, list):
            raise PreflightError("%s: size_results must be a list" % matrix_path)
        reported = {row.get("size"): row.get("generated_frames") for row in size_results
                    if isinstance(row, dict)}
        for size, size_rows in by_size.items():
            actual = sum(int(row.get("generated_frames") or 0) for row in size_rows)
            if size in reported and reported[size] is not None and int(reported[size]) != actual:
                raise PreflightError("%s: size_results %s reports %s frames, rows total %d"
                                     % (matrix_path, size, reported[size], actual))


def discover_source_index(source_matrices):
    """Map every source matrix scene row to its case directories.

    ``scenes[].cases[].directory`` is authoritative.  ``size_results`` is only
    cross-checked against the rows, never used to guess paths.
    """
    index = {}
    sources = []
    seen = set()
    for raw in source_matrices:
        matrix_path = _matrix_path(raw).resolve()
        if matrix_path in seen:
            raise DuplicateComboError("duplicate --source-matrix: %s" % matrix_path)
        seen.add(matrix_path)
        if not matrix_path.is_file():
            raise PreflightError("source matrix not found: %s" % matrix_path)
        matrix = _read_json(matrix_path)
        if matrix.get("schema") not in SOURCE_SCHEMAS:
            raise PreflightError("%s: unsupported schema %r"
                                 % (matrix_path, matrix.get("schema")))
        _validate_source_rows(matrix, matrix_path)
        sources.append({"matrix": str(matrix_path),
                        "sha256": _sha256_file(matrix_path),
                        "schema": matrix["schema"],
                        "size_results": matrix.get("size_results", [])})
        for row in matrix["scenes"]:
            key = combo_key(row["scene"], row["size"], row["mode"])
            if key in index:
                raise DuplicateComboError(
                    "%s is defined by both %s and %s"
                    % (key, index[key]["matrix"], matrix_path))
            index[key] = {"scene": row["scene"], "size": row["size"],
                          "mode": int(row["mode"]),
                          "cases": [str(Path(case["directory"]).resolve())
                                    for case in row["cases"]],
                          "matrix": str(matrix_path)}
    return index, sources


def source_index_fingerprint(index, sources):
    return _fingerprint({"sources": sources,
                         "combos": {key: {"scene": value["scene"],
                                          "size": value["size"],
                                          "mode": value["mode"],
                                          "cases": value["cases"]}
                                    for key, value in sorted(index.items())}})


def preflight(index):
    expected = set(expected_combo_keys())
    extra = sorted(set(index) - expected)
    if extra:
        raise PreflightError("source matrices contain unexpected combinations: %s" % extra)
    missing = sorted(expected - set(index))
    allowed = allowed_missing_keys()
    unexpected = [key for key in missing if key not in allowed]
    if unexpected:
        raise PreflightError("unexpected missing combinations (only %d known renders allowed): %s"
                             % (len(allowed), unexpected))
    for key, entry in index.items():
        wanted = expected_case_count(entry["scene"], entry["mode"])
        if len(entry["cases"]) != wanted:
            raise PreflightError("%s: expected %d case directories, found %d"
                                 % (key, wanted, len(entry["cases"])))
    combos = len(expected)
    frames = expected_total_frames()
    if combos != EXPECTED_COMBOS or frames != EXPECTED_FRAMES:
        raise PreflightError("internal inventory is %d combos/%d frames, expected %d/%d"
                             % (combos, frames, EXPECTED_COMBOS, EXPECTED_FRAMES))
    return {"expected_combos": combos, "expected_frames": frames,
            "found_combos": len(index), "found_frames": sum(
                expected_frames(value["scene"]) for value in index.values()),
            "missing_before_render": missing, "allowed_missing": sorted(missing),
            "unexpected_missing": [], "extra_source_combos": [],
            "render_combos": missing}


def package_identity(directory):
    """Hash the package manifest, every model and every profile."""
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise MatrixError("package manifest not found: %s" % manifest_path)
    manifest = _read_json(manifest_path)
    variants = manifest.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise MatrixError("package manifest has no variants")
    files = {}
    for variant_id, variant in sorted(variants.items()):
        names = [variant.get("model"), *variant.get("profiles", [])]
        if not variant.get("model") or not isinstance(variant.get("profiles"), list):
            raise MatrixError("package variant %s is incomplete" % variant_id)
        for name in names:
            path = directory / name
            if not path.is_file():
                raise MatrixError("package file missing: %s" % name)
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        declared = variant.get("model_sha256")
        if declared and files[variant["model"]]["sha256"] != declared:
            raise MatrixError("package model hash differs from manifest: %s" % variant["model"])
    payload = {"manifest_sha256": _sha256_file(manifest_path), "files": files}
    return {"directory": str(directory),
            "manifest_sha256": payload["manifest_sha256"],
            "files": files, "identity": _fingerprint(payload)}


def build_config(source_matrices, package, renderer, xess_bin, limits):
    return {"schema": SCHEMA, "scenes": list(SCENES), "sizes": list(SIZES),
            "modes": list(MODES), "expected_combos": EXPECTED_COMBOS,
            "expected_frames": EXPECTED_FRAMES, "source_frames": list(SOURCE_FRAMES),
            "cut_skip_source_frame": CUT_SKIP_SOURCE_FRAME,
            "source_matrices": [str(_matrix_path(path).resolve())
                                for path in source_matrices],
            "package_directory": str(Path(package).resolve()),
            "renderer": str(Path(renderer).resolve()),
            "xess_bin": str(Path(xess_bin).resolve()),
            "quality_limits": dict(limits),
            "executor": {"provider": "CPUExecutionProvider",
                         "intra_op_threads": 1, "inter_op_threads": 1},
            "serial": True,
            "allowed_missing_scenes": list(ALLOWED_MISSING_SCENES),
            "allowed_missing_sizes": list(ALLOWED_MISSING_SIZES)}


def validate_resume(state, identity, config_sha256, source_index_sha256,
                    all_complete, sources_available):
    """Reject stale package/config/source mappings before reusing checkpoints."""
    if state.get("schema") != STATE_SCHEMA:
        raise MatrixError("resume state has an unsupported schema")
    recorded = (state.get("package") or {}).get("identity")
    if recorded != identity["identity"]:
        raise StalePackageError("package changed since the resumable state was written")
    if state.get("config_sha256") != config_sha256:
        raise MatrixError("run configuration changed since the resumable state was written")
    if sources_available:
        if source_index_sha256 != state.get("source_index_sha256"):
            raise MatrixError("source matrix index changed since the resumable state was written")
        return "verified"
    if not all_complete:
        raise MatrixError("resume needs readable source matrices for unfinished combinations")
    return "stored"


def completed_checkpoints(out, state, expected_keys, identity, config_sha256):
    done = {}
    for key in expected_keys:
        item = (state.get("combos") or {}).get(key)
        if item and item.get("complete"):
            path = Path(out) / item.get("checkpoint", "")
        else:
            # A checkpoint can be durable before state.json is replaced.
            path = Path(out) / "results" / "combos" / (key + ".json")
        if not path.is_file():
            continue
        checkpoint = _read_json(path)
        if checkpoint.get("package_identity") != identity["identity"]:
            raise StalePackageError("checkpoint for %s used a different package" % key)
        if checkpoint.get("config_sha256") != config_sha256:
            raise MatrixError("checkpoint for %s used a different configuration" % key)
        if checkpoint.get("status") not in COMPLETE_STATUSES:
            continue
        done[key] = checkpoint
    return done


def _metric(pair, name):
    return float((pair.get("valid_region") or {}).get(name, 0.0))


def failure_causes(pair, limits):
    values = pair.get("valid_region") or {}
    causes = []
    if values.get("rgb_mae", 0) > limits["mae"]:
        causes.append("rgb_mae")
    if values.get("rgb_max", 0) > limits["maximum"]:
        causes.append("rgb_max")
    if values.get("over_tolerance_pixel_fraction", 0) > limits["fraction"]:
        causes.append("over_2_fraction")
    if 8 < values.get("rgb_max", 0) <= limits["maximum"]:
        causes.append("rgb_max_above_8")
    return causes


def _frame_identity(pair):
    return int(pair.get("source_current_frame", pair.get("current_frame", -1)))


def select_image_frames(pairs, limits):
    """Worst frame per metric plus the worst frame for each distinct cause."""
    if not pairs:
        return []
    chosen = {}

    def choose(pair, reason):
        frame = _frame_identity(pair)
        chosen.setdefault(frame, set()).add(reason)

    for reason, metric in METRIC_KEYS:
        choose(max(pairs, key=lambda pair: _metric(pair, metric)), reason)
    cause_metric = {"rgb_mae": "rgb_mae", "rgb_max": "rgb_max",
                    "over_2_fraction": "over_tolerance_pixel_fraction",
                    "rgb_max_above_8": "rgb_max"}
    for cause in sorted({cause for pair in pairs for cause in failure_causes(pair, limits)}):
        choose(max(pairs, key=lambda pair: _metric(pair, cause_metric[cause])),
               "cause:" + cause)
    return [{"source_frame": frame, "reasons": sorted(reasons)}
            for frame, reasons in sorted(chosen.items())]


def normalize_pair(pair, selected_reasons=None):
    row = {key: pair.get(key) for key in FRAME_FIELDS if key in pair}
    row["artifact_retention"] = {
        "state": "compacted",
        "detail": "replay npy/work artifacts removed; representative PNGs retained"}
    if selected_reasons:
        row["selected_for_images"] = sorted(selected_reasons)
    return row


def _owned_root(out, kind):
    return (Path(out).resolve() / kind).resolve()


def assert_owned_work(path, out, protected=()):
    path = Path(path).resolve()
    out = Path(out).resolve()
    roots = [_owned_root(out, "work"), _owned_root(out, "images"),
             _owned_root(out, "results")]
    if path == out or not any(path == root or path.is_relative_to(root) for root in roots):
        raise UnsafeDeleteError("refusing to delete outside owned output: %s" % path)
    if path in roots:
        raise UnsafeDeleteError("refusing to delete an owned root directory: %s" % path)
    for item in protected:
        item = Path(item).resolve()
        if path == item or path.is_relative_to(item) or item.is_relative_to(path):
            raise UnsafeDeleteError("refusing to delete protected source/package: %s" % item)
    return path


def remove_owned(path, out, protected=()):
    """Delete only a path below the owned work/images/results directories."""
    path = assert_owned_work(path, out, protected)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    return path


def _copy_image(source, destination):
    source = Path(source)
    if not source.is_file():
        raise MatrixError("representative image missing: %s" % source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _relative(out, path):
    return Path(path).resolve().relative_to(Path(out).resolve()).as_posix()


def _image_sources(replay_dir, pair):
    stem = "frame_%05d" % int(pair["current_frame"])
    return {"official": Path(replay_dir) / (stem + "_official.png"),
            "sdk": Path(replay_dir) / (stem + "_sdk.png"),
            "diff": Path(replay_dir) / (stem + "_valid_diff_x8.png"),
            "diff_full": Path(replay_dir) / (stem + "_diff_x8.png")}


def retain_images(out, scene, size, mode, pairs, image_sources, limits):
    """Copy official/SDK/diff PNGs for the selected representative frames."""
    images = []
    selected = select_image_frames(pairs, limits)
    for item in selected:
        frame = item["source_frame"]
        sources = image_sources.get(frame)
        if sources is None:
            raise MatrixError("selected frame %s has no replay images" % frame)
        directory = (Path(out) / "images" / scene / size / ("mode%d" % mode)
                     / ("frame_%05d" % frame))
        official = _copy_image(sources["official"], directory / "official.png")
        sdk = _copy_image(sources["sdk"], directory / "sdk.png")
        diff_source = sources["diff"] if sources["diff"].is_file() else sources["diff_full"]
        diff = _copy_image(diff_source, directory / "diff.png")
        images.append({"source_frame": frame, "reasons": item["reasons"],
                       "official": _relative(out, official),
                       "sdk": _relative(out, sdk), "diff": _relative(out, diff)})
    return images


def _source_frame_map(capture, pairing):
    manifest = _read_json(Path(capture) / "application_inputs/manifest.json")
    scene_path = Path(capture) / "scene.json"
    scene = _read_json(scene_path) if scene_path.is_file() else {}
    scene_frames = {row.get("index"): row for row in scene.get("frames", [])
                    if isinstance(row, dict)}
    mapping = {}
    for row in manifest["frames"]:
        index = row["index"]
        record = scene_frames.get(index, {})
        mapping[index] = int(record.get("scene_frame", index))
    return manifest, mapping


def input_fingerprints(capture, pairing_report, label):
    """Fingerprint every recorded application input by role and source frame."""
    capture = Path(capture)
    manifest, source_frames = _source_frame_map(capture, pairing_report)
    artifacts = {item["file"]: item for item in pairing_report.get("artifacts", [])
                 if isinstance(item, dict) and item.get("file")}
    rows = []
    for frame in manifest["frames"]:
        index = frame["index"]
        for role, name in sorted(frame["inputs"].items()):
            relative = "application_inputs/" + name
            artifact = artifacts.get(relative)
            if artifact and artifact.get("sha256"):
                digest, size = artifact["sha256"], artifact.get("bytes")
            else:
                path = capture / relative
                digest, size = _sha256_file(path), path.stat().st_size
            rows.append({"case": label, "local_frame": index,
                         "source_frame": source_frames[index], "role": role,
                         "file": name, "bytes": size, "sha256": digest,
                         "source": "pairing_artifacts" if artifact else "recomputed"})
    return rows


def _pair_status(pair, limits):
    return failure_causes(pair, limits)


def _run_replay(capture, pairing, package, output, log_path):
    output = Path(output)
    if output.exists():
        raise MatrixError("refusing to reuse replay output: %s" % output)
    command = [sys.executable, "-I", "-B", str(TOOLS / "validate.py"), "replay",
               "--run", str(capture), "--pairing", str(pairing),
               "--package", str(Path(package).resolve()),
               "--out", str(output), "--all"]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return {"exit_code": process.returncode, "command": command, "log": str(log_path)}


def _inspect_inputs(capture):
    tool = _load_tool("check_render_inputs")
    return tool.inspect_inputs(Path(capture))


def process_existing_case(case_directory, key, label, args, limits, out, work):
    case_directory = Path(case_directory)
    capture = case_directory / "capture"
    pairing_path = case_directory / "pairing" / "pairing.json"
    result = {"case": str(case_directory), "label": label, "status": "run_failed",
              "input_check": None, "input_check_recomputed": True,
              "replay": None, "pairs": [], "image_sources": {},
              "fingerprints": [], "error": None}
    if not capture.is_dir() or not pairing_path.is_file():
        result["error"] = "capture or pairing report missing"
        return result
    try:
        # Old input-check schemas are deliberately not trusted or skipped.
        input_check = _inspect_inputs(capture)
        result["input_check"] = input_check
        _write_json_atomic(Path(work) / key / "inputs" / (label + ".json"), input_check)
    except Exception as error:  # noqa: BLE001 - recorded as a runner failure
        result["error"] = "input check failed: %s" % error
        return result

    replay_dir = Path(work) / key / "replay" / label
    if replay_dir.exists():
        remove_owned(replay_dir, out, protected=[args.package, *args.source_matrix])
    try:
        replay = _run_replay(capture, pairing_path, args.package, replay_dir,
                             Path(work) / key / "logs" / (label + "-replay.log"))
    except Exception as error:  # noqa: BLE001
        result["error"] = "replay launch failed: %s" % error
        return result
    result["replay"] = replay
    report_path = replay_dir / "report.json"
    if replay["exit_code"] != 0 or not report_path.is_file():
        result["error"] = "replay exited %d" % replay["exit_code"]
        return result
    # replay.run_replay re-hashes every pairing artifact before measuring.
    result["content_hashes_verified"] = True

    report = _read_json(report_path)
    try:
        quality = _load_tool("check_quality").evaluate(
            report, limits["mae"], limits["maximum"], limits["fraction"])
    except Exception as error:  # noqa: BLE001
        result["error"] = "quality evaluation failed: %s" % error
        return result
    result["quality"] = quality
    result["pairs"] = report["pairs"]
    result["image_sources"] = {_frame_identity(pair): _image_sources(replay_dir, pair)
                               for pair in report["pairs"]}
    pairing_report = _read_json(pairing_path)
    result["fingerprints"] = input_fingerprints(capture, pairing_report, label)
    result["status"] = ("quality_failed" if quality["status"] == "fail"
                        else "pass")
    return result


def process_new_case(directory, scene, mode, pair_frame, args, limits, out, work, key):
    directory = Path(directory)
    if directory.exists():
        remove_owned(directory, out, protected=[args.package, *args.source_matrix])
    matrix = _load_tool("run_matrix")
    case = matrix.run_case(args, directory, scene, mode, pair_frame,
                           _size_from_key(key))
    result = {"case": str(directory), "label": directory.name, "status": "run_failed",
              "input_check": None, "input_check_recomputed": False,
              "replay": None, "pairs": [], "image_sources": {},
              "fingerprints": [], "error": None, "runner": case}
    if case.get("status") == "run_failed":
        result["error"] = "render/pair/replay runner failed"
        return result
    capture = directory / "capture"
    pairing_path = directory / "pairing" / "pairing.json"
    replay_dir = directory / "replay"
    report_path = replay_dir / "report.json"
    if not report_path.is_file() or not pairing_path.is_file():
        result["error"] = "runner did not produce replay/pairing reports"
        return result
    try:
        check_path = directory / "input-check.json"
        if check_path.is_file() and _read_json(check_path).get("schema") == "xefg.render_input_check/2":
            input_check = _read_json(check_path)
        else:
            input_check = _inspect_inputs(capture)
            result["input_check_recomputed"] = True
        result["input_check"] = input_check
        _write_json_atomic(Path(work) / key / "inputs" / (directory.name + ".json"),
                           input_check)
        report = _read_json(report_path)
        quality = _load_tool("check_quality").evaluate(
            report, limits["mae"], limits["maximum"], limits["fraction"])
        result["quality"] = quality
        result["pairs"] = report["pairs"]
        result["image_sources"] = {_frame_identity(pair): _image_sources(replay_dir, pair)
                                   for pair in report["pairs"]}
        result["fingerprints"] = input_fingerprints(
            capture, _read_json(pairing_path), directory.name)
        result["content_hashes_verified"] = True
        result["status"] = ("quality_failed" if quality["status"] == "fail"
                            else "pass")
    except Exception as error:  # noqa: BLE001
        result["error"] = "new capture validation failed: %s" % error
    return result


def _size_from_key(key):
    return key.split("__")[1]


def aggregate_combo(key, scene, size, mode, source, case_results, out, limits):
    input_frames = []
    pairs = []
    image_sources = {}
    fingerprints = []
    quality_failures = []
    quality_warnings = []
    cases = []
    errors = []
    recomputed = False
    for result in case_results:
        if result.get("input_check"):
            input_frames.extend(result["input_check"]["frames"])
        pairs.extend(result.get("pairs", []))
        image_sources.update(result.get("image_sources", {}))
        fingerprints.extend(result.get("fingerprints", []))
        recomputed = recomputed or bool(result.get("input_check_recomputed"))
        cases.append({"case": result["case"], "label": result["label"],
                      "status": result["status"], "error": result.get("error"),
                      "case_retention": ("retained_source" if source == "existing"
                                         else "removed_owned_work"),
                      "replay_exit": (result.get("replay") or {}).get("exit_code"),
                      "input_check_schema": (result.get("input_check") or {}).get("schema"),
                      "input_check_recomputed": result.get("input_check_recomputed", True),
                      "content_hashes_verified": result.get("content_hashes_verified", False)})
        if result.get("status") == "run_failed":
            errors.append("%s: %s" % (result["label"], result.get("error")))
        for item in (result.get("quality") or {}).get("failures", []):
            quality_failures.append(item)
        for item in (result.get("quality") or {}).get("warnings", []):
            quality_warnings.append(item)
    coverage = _load_tool("check_render_inputs").aggregate_coverage(scene, input_frames)
    status = "pass"
    if errors:
        status = "run_failed"
    elif coverage.get("status") == "fail":
        status = "coverage_failed"
    elif quality_failures:
        status = "quality_failed"

    images = []
    if status in COMPLETE_STATUSES and pairs:
        try:
            images = retain_images(out, scene, size, mode, pairs, image_sources, limits)
        except Exception as error:  # noqa: BLE001
            errors.append("image retention failed: %s" % error)
            status = "run_failed"

    selected = {item["source_frame"]: item["reasons"] for item in images}
    normalized = [normalize_pair(pair, selected.get(_frame_identity(pair)))
                  for pair in sorted(pairs, key=_frame_identity)]
    if status != "run_failed" and len(normalized) != expected_frames(scene):
        errors.append("generated %d of %d expected frames"
                      % (len(normalized), expected_frames(scene)))
        status = "run_failed"
    causes = {}
    for pair in pairs:
        for cause in _pair_status(pair, limits):
            causes.setdefault(cause, []).append(_frame_identity(pair))
    if coverage.get("status") == "fail":
        causes["input_coverage"] = [item.get("source_frame") for item in
                                    coverage.get("measured_frames", [])]
    if errors:
        causes["runner"] = errors
    return {"schema": CHECKPOINT_SCHEMA, "key": key, "scene": scene, "size": size,
            "mode": int(mode), "source": source, "status": status,
            "expected_frames": expected_frames(scene), "generated_frames": len(normalized),
            "cases": cases, "errors": errors,
            "content_hashes_verified": bool(cases) and all(
                case["content_hashes_verified"] for case in cases),
            "coverage": coverage, "quality": {
                "status": "fail" if quality_failures else "pass",
                "limits": dict(limits), "failures": quality_failures,
                "warnings": quality_warnings,
                "scope": "check_quality.evaluate over compact replay metrics"},
            "failure_causes": [{"cause": cause, "source_frames": sorted(set(frames))}
                               for cause, frames in sorted(causes.items())],
            "input_check": {"schema": "xefg.render_input_check/2",
                            "frames": input_frames, "coverage": coverage,
                            "recomputed": recomputed},
            "frames": normalized, "images": images,
            "input_fingerprints": fingerprints}


def _input_equality(combos):
    """Compare mode 3 and mode 4 application inputs per source frame.

    Works from compact checkpoints alone, so a finished resume remains
    readable even after every source capture has been removed.
    """
    rows = []
    by_combo = {combo["key"]: combo for combo in combos}
    for key, mode4 in sorted(by_combo.items()):
        if int(mode4["mode"]) != 4:
            continue
        mode3 = by_combo.get(combo_key(mode4["scene"], mode4["size"], 3))
        if not mode3 or mode3["status"] == "run_failed" or mode4["status"] == "run_failed":
            continue
        left = {(item["source_frame"], item["role"]): item["sha256"]
                for item in mode3["input_fingerprints"]}
        right = {(item["source_frame"], item["role"]): item["sha256"]
                 for item in mode4["input_fingerprints"]}
        for source_frame in sorted({frame for frame, _role in left} & {frame for frame, _role in right}):
            roles = {}
            for role in ("color", "ui", "backbuffer", "depth", "motion"):
                a, b = left.get((source_frame, role)), right.get((source_frame, role))
                roles[role] = {"mode3": a, "mode4": b,
                               "equal": None if a is None or b is None else a == b}
            rows.append({"scene": mode4["scene"], "size": mode4["size"],
                         "source_frame": source_frame, "roles": roles,
                         "ui_exact": roles["ui"]["equal"],
                         "comparable_roles_equal": all(
                             value["equal"] is not False for value in roles.values())})
    return rows


def build_matrix(out, state, checkpoints, inventory, equality, identity, config,
                 config_sha256, resumed, source_index_verification):
    order = {key: position for position, key in enumerate(expected_combo_keys())}
    combos = [checkpoints[key] for key in sorted(checkpoints, key=lambda item: order[item])]
    counts = {status: sum(1 for combo in combos if combo["status"] == status)
              for status in ("pass", "quality_failed", "coverage_failed", "run_failed")}
    status = ("run_failed" if counts["run_failed"] else
              "coverage_failed" if counts["coverage_failed"] else
              "quality_failed" if counts["quality_failed"] else "pass")
    if len(combos) != EXPECTED_COMBOS:
        status = "running"
    size_results = []
    for size in SIZES:
        rows = [combo for combo in combos if combo["size"] == size]
        size_results.append({"size": size,
                             "status": ("running" if len(rows) < len(SCENES) * len(MODES)
                                        else ("run_failed" if any(row["status"] == "run_failed" for row in rows)
                                              else "coverage_failed" if any(row["status"] == "coverage_failed" for row in rows)
                                              else "quality_failed" if any(row["status"] == "quality_failed" for row in rows)
                                              else "pass")),
                             "generated_frames": sum(row["generated_frames"] for row in rows),
                             "combos": len(rows)})
    measured = [combo for combo in combos if combo["frames"]]
    return {"schema": SCHEMA, "status": status, "generated_at": _now(),
            "command": sys.argv, "package": identity, "config": config,
            "resume": {"resumed": resumed, "package_identity": identity["identity"],
                       "config_sha256": config_sha256,
                       "source_index_sha256": state.get("source_index_sha256"),
                       "source_index_verification": source_index_verification},
            "inventory": inventory, "size_results": size_results,
            "input_equality": equality, "combos": combos,
            "summary": {"combos": len(combos), "expected_combos": EXPECTED_COMBOS,
                        "frames": sum(combo["generated_frames"] for combo in combos),
                        "expected_frames": EXPECTED_FRAMES,
                        "status_counts": counts,
                        "max_frame_rgb_mae": max((max(
                            pair["valid_region"]["rgb_mae"] for pair in combo["frames"])
                            for combo in measured), default=None),
                        "max_channel_error": max((max(
                            pair["valid_region"]["rgb_max"] for pair in combo["frames"])
                            for combo in measured), default=None),
                        "max_over_2_fraction": max((max(
                            pair["valid_region"]["over_tolerance_pixel_fraction"]
                            for pair in combo["frames"]) for combo in measured), default=None),
                        "images": [{"combo": combo["key"], "scene": combo["scene"],
                                    "size": combo["size"], "mode": combo["mode"],
                                    **image} for combo in combos
                                   for image in combo["images"]]},
            "artifact_retention": {
                "policy": "compact",
                "detail": "per-frame numeric metrics retained; replay npy/work removed; "
                          "only representative official/sdk/diff PNGs retained under images/",
                "work": "only out/work is owned and may be removed"}}


def write_matrix_markdown(out, matrix):
    lines = ["# Full rendered replay matrix", "",
             "Status: %s" % matrix["status"],
             "Combinations: %d/%d; frames: %d/%d" % (
                 matrix["summary"]["combos"], EXPECTED_COMBOS,
                 matrix["summary"]["frames"], EXPECTED_FRAMES), "",
             "| Size | Mode | Scene | Status | Source | Frames | Coverage | Quality | Worst MAE | Max | >2 fraction | Images |",
             "|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|"]
    for combo in matrix["combos"]:
        worst_mae = max((pair["valid_region"]["rgb_mae"] for pair in combo["frames"]),
                        default=None)
        worst_max = max((pair["valid_region"]["rgb_max"] for pair in combo["frames"]),
                        default=None)
        worst_over = max((pair["valid_region"]["over_tolerance_pixel_fraction"]
                          for pair in combo["frames"]), default=None)
        lines.append("| %s | %d | %s | %s | %s | %d/%d | %s | %s | %s | %s | %s | %d |" % (
            combo["size"], combo["mode"], combo["scene"], combo["status"],
            combo["source"], combo["generated_frames"], combo["expected_frames"],
            combo["coverage"].get("status"), combo["quality"]["status"],
            "n/a" if worst_mae is None else "%.6f" % worst_mae,
            "n/a" if worst_max is None else worst_max,
            "n/a" if worst_over is None else "%.6g" % worst_over,
            len(combo["images"])))
    lines += ["", "Max-error representative images are linked from `results/matrix.json` "
              "under `images` (official/sdk/diff PNGs).", ""]
    image_rows = [(combo["key"], image) for combo in matrix["combos"]
                  for image in combo["images"]]
    if image_rows:
        lines += ["## Representative images", "",
                  "| Combination | Source frame | Causes | Images |",
                  "|---|---:|---|---|"]
        for key, image in image_rows:
            lines.append("| %s | %d | %s | [official](%s) [sdk](%s) [diff](%s) |" % (
                key, image["source_frame"], ", ".join(image["reasons"]),
                "../" + image["official"], "../" + image["sdk"],
                "../" + image["diff"]))
        lines.append("")
    (Path(out) / "results" / "matrix.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")


def write_status(out, status, completed, frames, current, counts, error=None):
    _write_json_atomic(Path(out) / "results" / "status.json", {
        "schema": STATUS_SCHEMA, "status": status, "updated_at": _now(),
        "expected_combos": EXPECTED_COMBOS, "completed_combos": completed,
        "expected_frames": EXPECTED_FRAMES, "reported_frames": frames,
        "current": current, "counts": counts, "last_error": error})


def _store_checkpoint(out, state, checkpoint, identity, config_sha256):
    key = checkpoint["key"]
    path = Path(out) / "results" / "combos" / (key + ".json")
    checkpoint["package_identity"] = identity["identity"]
    checkpoint["config_sha256"] = config_sha256
    checkpoint["completed_at"] = _now()
    _write_json_atomic(path, checkpoint)
    state.setdefault("combos", {})[key] = {
        "checkpoint": _relative(out, path),
        "complete": checkpoint["status"] in COMPLETE_STATUSES,
        "status": checkpoint["status"],
        "frames": checkpoint["generated_frames"]}
    return checkpoint


def _prepare_roots(out, resume):
    out = Path(out).resolve()
    results = out / "results"
    state_path = results / "state.json"
    if resume:
        if not state_path.is_file():
            if results.exists() and any(results.iterdir()):
                raise MatrixError("results exist but state.json is missing")
            resume = False
        else:
            (out / "images").mkdir(parents=True, exist_ok=True)
            (out / "work").mkdir(parents=True, exist_ok=True)
            return out, results, True
    if not resume:
        if results.exists() and any(results.iterdir()):
            raise MatrixError("output results already exist; use --resume")
        work = out / "work"
        if work.exists() and any(work.iterdir()):
            raise MatrixError("owned work directory is not empty; use --resume")
    (out / "images").mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (out / "work").mkdir(parents=True, exist_ok=True)
    return out, results, resume


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    out, results, resumed = _prepare_roots(out, args.resume)
    identity = package_identity(args.package)
    limits = {"mae": args.mae, "maximum": args.maximum, "fraction": args.fraction}
    config = build_config(args.source_matrix, args.package, args.renderer,
                          args.xess_bin, limits)
    config_sha256 = _fingerprint(config)
    state_path = results / "state.json"

    expected = expected_combo_keys()
    state = None
    index = None
    sources = None
    source_sha256 = None
    verification = "fresh"
    done = {}
    if resumed:
        state = _read_json(state_path)
        complete_keys = {key for key, item in (state.get("combos") or {}).items()
                         if item.get("complete")}
        for key in expected:
            if key in complete_keys:
                continue
            orphan = results / "combos" / (key + ".json")
            if orphan.is_file() and _read_json(orphan).get("status") in COMPLETE_STATUSES:
                complete_keys.add(key)
        all_complete = set(expected) <= complete_keys
        try:
            index, sources = discover_source_index(args.source_matrix)
            source_sha256 = source_index_fingerprint(index, sources)
            sources_available = True
        except MatrixError:
            if not all_complete:
                raise
            sources_available = False
        verification = validate_resume(state, identity, config_sha256, source_sha256,
                                       all_complete, sources_available)
        done = completed_checkpoints(out, state, expected, identity, config_sha256)
    else:
        index, sources = discover_source_index(args.source_matrix)
        source_sha256 = source_index_fingerprint(index, sources)
        inventory = preflight(index)
        if not args.dry_run:
            _load_tool("run_matrix").preflight_package(args.package, SIZES, MODES)
        state = {"schema": STATE_SCHEMA, "created_at": _now(), "package": identity,
                 "config": config, "config_sha256": config_sha256,
                 "source_index": {key: {"scene": value["scene"], "size": value["size"],
                                        "mode": value["mode"], "cases": value["cases"]}
                                  for key, value in index.items()},
                 "sources": sources, "source_index_sha256": source_sha256,
                 "combos": {}}
        _write_json_atomic(state_path, state)

    if resumed:
        if index is None:
            inventory = {"expected_combos": EXPECTED_COMBOS,
                         "expected_frames": EXPECTED_FRAMES,
                         "found_combos": len(done), "found_frames": sum(
                             combo["generated_frames"] for combo in done.values()),
                         "missing_before_render": [],
                         "allowed_missing": [], "unexpected_missing": [],
                         "render_combos": []}
        else:
            inventory = preflight(index)

    if args.dry_run:
        write_status(out, "preflight", len(done),
                     sum(combo["generated_frames"] for combo in done.values()),
                     None, {}, None)
        print(json.dumps({"status": "preflight", "inventory": inventory,
                          "resume_verification": verification}, indent=2))
        return 0

    protected = [Path(args.package), *[Path(path) for path in args.source_matrix],
                 *[Path(case) for entry in (index or {}).values()
                   for case in entry["cases"]]]
    progress = {"completed": len(done),
                "frames": sum(combo["generated_frames"] for combo in done.values()),
                "counts": {status: 0 for status in
                           ("pass", "quality_failed", "coverage_failed", "run_failed")}}
    for combo in done.values():
        progress["counts"][combo["status"]] = progress["counts"].get(combo["status"], 0) + 1
    write_status(out, "running", progress["completed"], progress["frames"], None,
                 progress["counts"], None)

    render_keys = set(inventory["render_combos"]) if index is not None else set()
    for key in expected:
        if key in done:
            continue
        scene, size, mode = parse_combo_key(key)
        current = {"key": key, "scene": scene, "size": size, "mode": mode,
                   "stage": "start"}
        write_status(out, "running", progress["completed"], progress["frames"],
                     current, progress["counts"], None)
        try:
            work = out / "work" / key
            if work.exists():
                remove_owned(work, out, protected=protected)
            work.mkdir(parents=True, exist_ok=True)
            if key in render_keys:
                cases = []
                for pair_frame in (expected_pair_frames(scene) if mode == 1 else [None]):
                    label = ("pair_%05d" % pair_frame if pair_frame is not None
                             else "sequence")
                    current = {"key": key, "scene": scene, "size": size, "mode": mode,
                               "stage": "render/pair/replay", "case": label}
                    write_status(out, "running", progress["completed"],
                                 progress["frames"], current, progress["counts"], None)
                    directory = work / "new" / label
                    cases.append(process_new_case(directory, scene, mode, pair_frame,
                                                  args, limits, out, work, key))
                checkpoint = aggregate_combo(key, scene, size, mode, "rendered", cases,
                                             out, limits)
                checkpoint["content_hashes_verified"] = all(
                    case.get("content_hashes_verified") for case in cases)
            else:
                entry = index.get(key) if index is not None else None
                if entry is None:
                    raise MatrixError("missing source mapping for %s" % key)
                cases = []
                for case_path in entry["cases"]:
                    label = Path(case_path).name
                    current = {"key": key, "scene": scene, "size": size, "mode": mode,
                               "stage": "input-check/replay", "case": label}
                    write_status(out, "running", progress["completed"],
                                 progress["frames"], current, progress["counts"], None)
                    cases.append(process_existing_case(case_path, key, label, args,
                                                       limits, out, work))
                checkpoint = aggregate_combo(key, scene, size, mode, "existing", cases,
                                             out, limits)
            _store_checkpoint(out, state, checkpoint, identity, config_sha256)
            _write_json_atomic(state_path, state)
            if checkpoint["status"] in COMPLETE_STATUSES:
                remove_owned(out / "work" / key, out, protected=protected)
            done[key] = checkpoint
            progress["completed"] += 1
            progress["frames"] += checkpoint["generated_frames"]
            progress["counts"][checkpoint["status"]] = (
                progress["counts"].get(checkpoint["status"], 0) + 1)
            write_status(out, "running", progress["completed"], progress["frames"],
                         {"key": key, "stage": "complete",
                          "status": checkpoint["status"]}, progress["counts"], None)
        except Exception as error:  # noqa: BLE001 - continue every combination
            failed = {"schema": CHECKPOINT_SCHEMA, "key": key, "scene": scene,
                      "size": size, "mode": int(mode),
                      "source": "rendered" if key in render_keys else "existing",
                      "status": "run_failed",
                      "expected_frames": expected_frames(scene), "generated_frames": 0,
                      "cases": [], "errors": [str(error)],
                      "coverage": {"status": "pending", "reason": str(error)},
                      "quality": {"status": "fail", "limits": dict(limits),
                                  "failures": [], "warnings": []},
                      "failure_causes": [{"cause": "runner", "source_frames": []}],
                      "input_check": {"schema": "xefg.render_input_check/2",
                                      "frames": [], "coverage": {"status": "pending"},
                                      "recomputed": True},
                      "frames": [], "images": [], "input_fingerprints": []}
            _store_checkpoint(out, state, failed, identity, config_sha256)
            _write_json_atomic(state_path, state)
            done[key] = failed
            progress["completed"] += 1
            progress["counts"]["run_failed"] += 1
            write_status(out, "running", progress["completed"], progress["frames"],
                         {"key": key, "stage": "failed"}, progress["counts"],
                         str(error))
            print("FAILED %s: %s" % (key, error), flush=True)

    checkpoints = {key: done[key] for key in expected if key in done}
    equality = _input_equality(list(checkpoints.values()))
    _write_json_atomic(results / "input-equality.json", {
        "schema": "xefg.replay_input_equality/1", "rows": equality,
        "note": "mode 3 vs 4 fingerprints by source frame; UI hashes come from "
                "pairing artifacts"})
    matrix = build_matrix(out, state, checkpoints, inventory, equality, identity,
                          config, config_sha256, resumed, verification)
    _write_json_atomic(results / "matrix.json", matrix)
    write_matrix_markdown(out, matrix)
    write_status(out, matrix["status"], len(checkpoints),
                 matrix["summary"]["frames"], None, matrix["summary"]["status_counts"], None)
    return EXIT_CODES.get(matrix["status"], 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", type=Path, action="append", required=True,
                        help="Render-matrix directory or matrix.json; repeatable")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--xess-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse/complete compact checkpoints from a previous run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight the source mapping and write status only")
    parser.add_argument("--mae", type=float, default=DEFAULT_QUALITY["mae"])
    parser.add_argument("--maximum", type=int, default=DEFAULT_QUALITY["maximum"])
    parser.add_argument("--fraction", type=float, default=DEFAULT_QUALITY["fraction"])
    args = parser.parse_args(argv)
    if not args.source_matrix:
        parser.error("at least one --source-matrix is required")
    if len({Path(path).resolve() for path in args.source_matrix}) != len(args.source_matrix):
        parser.error("--source-matrix must not contain duplicates")
    if args.mae < 0 or args.maximum < 0 or not 0 <= args.fraction <= 1:
        parser.error("quality limits must be nonnegative and fraction in [0,1]")
    try:
        return run(args)
    except MatrixError as error:
        print("error: %s" % error, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
