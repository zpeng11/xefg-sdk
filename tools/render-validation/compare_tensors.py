"""Compare official d24/d25 captured tensors with the debug ONNX model.

Focused diagnostic for one Mode 3 frame.  It deliberately does not build a
general tensor framework: only the splat (d24) and warp + flow-init (d25)
stages are selected, by program key plus output-resource shapes.  A debug
output-map entry is accepted only when its node, producer, semantic and slot
agree with that dispatch; a global dispatch ordinal is never used as the
selection rule.

Official tensors are read only from ``<run>/trace/tensors``.  They are never
reconstructed from a final image.  Debug ONNX inference is disabled unless
``--execute`` is passed, so an old/reduced trace can be inspected first.

Usage:
    python -I -B tools/render-validation/compare_tensors.py \
      --run RUN --package PACKAGE --frame 8 --out OUT \
      --normal-replay REPLAY/frame_00008.npy
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from xefg_sdk.export import recon


MODE = 3
STAGES = {
    "d24": {
        "semantic": "splat",
        "slots": ("u0",),
        "predicate": recon.splat_shaped,
    },
    "d25": {
        "semantic": "warp + flow init",
        "slots": ("u0", "u1", "u2", "u3"),
        "predicate": recon.warp_init_shaped,
    },
}


class ComparisonError(ValueError):
    """A required tensor, shape or unambiguous stage mapping is missing."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_root(path):
    """Accept either the capture root or its ``trace`` directory."""
    path = Path(path).resolve()
    if (path / "trace" / "trace.jsonl").exists():
        return path
    if (path / "trace.jsonl").exists():
        return path.parent
    raise ComparisonError("capture has no trace/trace.jsonl: %s" % path)


def trace_frames(rows):
    return sorted({row.get("frame") for row in rows
                   if row.get("type") == "dispatch" and row.get("frame") is not None})


def read_trace_meta(run):
    path = Path(run) / "trace" / "meta.json"
    return read_json(path) if path.is_file() else {}


def validate_trace_meta(meta, frame, require_complete=False):
    """Hard gate a tensor trace: an explicit false/absent complete is refused."""
    schema = meta.get("schema")
    complete = meta.get("complete")
    if complete is False:
        raise ComparisonError("trace meta.complete is false; the tensor trace is incomplete")
    if schema == "xefg.tensor_trace/1" and complete is not True:
        raise ComparisonError(
            "trace meta.complete is not true for schema xefg.tensor_trace/1")
    if require_complete and complete is not True:
        raise ComparisonError(
            "trace meta.complete is not true; --execute requires a complete tensor trace")
    for key in ("trace_error", "wait_error"):
        value = meta.get(key)
        if value:
            raise ComparisonError("trace meta.%s is set: %s" % (key, value))
    tensor_frame = meta.get("tensor_frame")
    if tensor_frame is not None:
        frames = set(tensor_frame) if isinstance(tensor_frame, (list, tuple)) else {tensor_frame}
        if frame not in frames:
            raise ComparisonError(
                "trace meta.tensor_frame %s does not include frame %s"
                % (sorted(frames), frame))
    return {
        "schema": schema,
        "complete": complete,
        "tensor_frame": tensor_frame,
        "trace_error": meta.get("trace_error"),
        "wait_error": meta.get("wait_error"),
        "tensor_files": meta.get("tensor_files"),
        "tensor_bytes": meta.get("tensor_bytes"),
    }


def _manifest_reference_runs(package):
    manifest = read_json(Path(package) / "manifest.json")
    result = []
    for capture in manifest.get("source", {}).get("captures", []):
        run = capture.get("run")
        if run and (Path(run) / "trace" / "trace.jsonl").is_file():
            result.append(Path(run))
    return result


def reference_program_hashes(package, stage, reference_run=None):
    """Find the package's reference dispatch identity for one stage, if retained."""
    if reference_run is not None:
        runs = [Path(reference_run)]
    else:
        runs = _manifest_reference_runs(package)
    predicate = STAGES[stage]["predicate"]
    found = []
    for run in runs:
        try:
            rows = recon.read_events(str(run))
        except OSError:
            continue
        identities = set()
        for row in rows:
            if row.get("type") != "dispatch":
                continue
            try:
                if predicate(row):
                    identities.add((row.get("key"), row.get("sha256")))
            except (KeyError, TypeError, IndexError):
                continue
        if len(identities) == 1:
            key, sha = identities.pop()
            return {"status": "available", "key": key, "sha256": sha,
                    "reference_run": str(run)}
        if len(identities) > 1:
            return {"status": "ambiguous", "reference_run": str(run),
                    "identities": sorted(identities)}
    return {"status": "unavailable",
            "reason": "no retained package reference trace for this stage",
            "reference_runs": [str(run) for run in runs]}


def validate_program_identity(dispatch, reference, stage):
    if reference.get("status") != "available":
        return {"status": reference.get("status", "unavailable"),
                "reference": reference,
                "note": "Program-hash identity is checked only when the package "
                        "retains a reference dispatch."}
    if (dispatch.get("key"), dispatch.get("sha256")) != (
            reference.get("key"), reference.get("sha256")):
        raise ComparisonError(
            "%s program hash mismatch: trace %s/%s vs package reference %s/%s"
            % (stage, dispatch.get("key"), dispatch.get("sha256"),
               reference.get("key"), reference.get("sha256")))
    return {"status": "matched", "key": reference["key"],
            "sha256": reference["sha256"],
            "reference_run": reference.get("reference_run")}


def frame_local_index(rows, dispatch):
    """One-based position of a dispatch inside its own frame.

    The native trace may use global ordinals across frames; the debug output
    map uses the per-frame node number.  This is only a cross-check after the
    stage was selected by key/resources/shapes.
    """
    frame_rows = [row for row in rows
                  if row.get("type") == "dispatch" and row.get("frame") == dispatch.get("frame")]
    for index, row in enumerate(frame_rows, 1):
        if row is dispatch or row.get("seq") == dispatch.get("seq"):
            return index
    raise ComparisonError("selected dispatch is not present in its trace frame")


def dispatch_signature(dispatch):
    """Stable key + resource description used to identify a dispatch."""
    resources = []
    for item in dispatch.get("resources", []):
        resources.append({
            "slot": item.get("slot"),
            "index": item.get("index", 0),
            "kind": item.get("kind"),
            "dimension": item.get("dimension"),
            "format": item.get("format"),
            "width": item.get("width"),
            "height": item.get("height"),
            "array": item.get("array"),
            "bytes": item.get("bytes"),
            "file": item.get("file"),
        })
    resources.sort(key=lambda item: (str(item["slot"]), int(item["index"] or 0),
                                     str(item["kind"])))
    return {
        "dispatch": dispatch.get("dispatch"),
        "frame_local_dispatch": dispatch.get("_frame_local"),
        "frame": dispatch.get("frame"),
        "key": dispatch.get("key"),
        "sha256": dispatch.get("sha256"),
        "group": dispatch.get("group"),
        "resources": resources,
    }


def stage_candidates(rows, stage, frame):
    if stage not in STAGES:
        raise ComparisonError("unsupported stage %s" % stage)
    predicate = STAGES[stage]["predicate"]
    candidates = []
    for row in rows:
        if row.get("type") != "dispatch":
            continue
        if row.get("frame") != frame:
            continue
        try:
            if predicate(row):
                candidates.append(row)
        except (KeyError, TypeError, IndexError):
            continue
    return candidates


def select_stage_dispatch(rows, stage, frame):
    """Select exactly one dispatch by shader/resources/shapes, not ordinal."""
    candidates = stage_candidates(rows, stage, frame)
    available = trace_frames(rows)
    if not candidates:
        raise ComparisonError(
            "%s dispatch not found in frame %s (trace frames %s); selection "
            "used program resources and shapes, not dispatch ordinal"
            % (stage, frame, available))
    if len(candidates) > 1:
        detail = ", ".join(
            "dispatch %s key %s" % (row.get("dispatch"), row.get("key"))
            for row in candidates)
        raise ComparisonError("%s dispatch mapping is ambiguous: %s" % (stage, detail))
    return candidates[0]


def match_debug_entries(debug_map, stage):
    """Return exactly the declared debug entries for one bounded stage."""
    if stage not in STAGES:
        raise ComparisonError("unsupported stage %s" % stage)
    semantic = STAGES[stage]["semantic"]
    entries = [entry for entry in debug_map.get("outputs", [])
               if entry.get("semantic") == semantic]
    if not entries:
        raise ComparisonError("missing debug map entries for %s (%s)" % (stage, semantic))
    nodes = {entry.get("node") for entry in entries}
    if len(nodes) != 1:
        raise ComparisonError("ambiguous debug node for %s: %s" % (stage, sorted(nodes)))
    by_slot = {}
    for entry in entries:
        slot = entry.get("slot")
        if slot in by_slot:
            raise ComparisonError("duplicate debug slot %s for %s" % (slot, stage))
        by_slot[slot] = entry
    expected = list(STAGES[stage]["slots"])
    missing = [slot for slot in expected if slot not in by_slot]
    extra = sorted(slot for slot in by_slot if slot not in expected)
    if missing:
        raise ComparisonError("missing debug map entries for %s slots %s" % (stage, missing))
    if extra:
        raise ComparisonError("unexpected debug slots for %s: %s" % (stage, extra))
    for slot in expected:
        entry = by_slot[slot]
        if not entry.get("graph_name"):
            raise ComparisonError("missing graph output name for %s.%s" % (stage, slot))
        if not entry.get("dtype"):
            raise ComparisonError("missing dtype for %s.%s" % (stage, slot))
        if not entry.get("shape"):
            raise ComparisonError("missing shape in debug map for %s.%s" % (stage, slot))
    return [by_slot[slot] for slot in expected]


def bind_dispatch_to_debug(dispatch, entries, stage, rows=None):
    """Cross-check the shader-selected dispatch against the debug output map."""
    nodes = {entry.get("node") for entry in entries}
    if len(nodes) != 1:
        raise ComparisonError("ambiguous debug node for %s: %s" % (stage, sorted(nodes)))
    node = nodes.pop()
    local = dispatch.get("dispatch")
    if rows is not None:
        local = frame_local_index(rows, dispatch)
        dispatch["_frame_local"] = local
    if local != node:
        raise ComparisonError(
            "%s mapping mismatch: trace frame-local dispatch %s (global %s, key %s) "
            "does not match debug node %s"
            % (stage, local, dispatch.get("dispatch"), dispatch.get("key"), node))
    return node


def _resources(dispatch):
    return {item["slot"]: item for item in dispatch.get("resources", [])
            if item.get("slot")}


def _output_resource(dispatch, slot):
    resource = _resources(dispatch).get(slot)
    if resource is None:
        raise ComparisonError("missing output resource %s on dispatch %s"
                              % (slot, dispatch.get("dispatch")))
    if resource.get("kind") != "uav":
        raise ComparisonError("resource %s is not a UAV output" % slot)
    return resource


def _format_bytes(resource):
    if resource.get("dimension") == "buffer":
        return None
    value = recon.FORMAT_BYTES.get(resource.get("format"))
    if value is None:
        raise ComparisonError("unknown format %s for slot %s"
                              % (resource.get("format"), resource.get("slot")))
    return int(value)


def resource_bytes(resource):
    """Byte size implied by the trace's resource declaration."""
    if resource.get("dimension") == "buffer":
        value = int(resource.get("bytes") or 0)
        if value <= 0:
            value = int(resource.get("width") or 0)
        if value <= 0:
            raise ComparisonError("missing buffer size for slot %s" % resource.get("slot"))
        return value
    width = int(resource.get("width") or 0)
    height = int(resource.get("height") or 0)
    array = int(resource.get("array") or 1)
    if width <= 0 or height <= 0:
        raise ComparisonError("missing texture shape for slot %s" % resource.get("slot"))
    return width * height * array * _format_bytes(resource)


def resource_shape(resource):
    """Logical shape implied by the trace resource declaration."""
    if resource.get("dimension") == "buffer":
        return (resource_bytes(resource),)
    width = int(resource.get("width") or 0)
    height = int(resource.get("height") or 0)
    array = int(resource.get("array") or 1)
    channels = _format_bytes(resource)
    if width <= 0 or height <= 0:
        raise ComparisonError("missing texture shape for slot %s" % resource.get("slot"))
    if array == 1:
        return (height, width, channels)
    return (array, height, width, channels)


def _logical_bytes(shape):
    return int(np.prod(shape, dtype=np.int64))


def stage_shapes(run, dispatch, stage):
    """Resolve logical and allocation sizes for one stage.

    d24 is a single byte-address buffer with two output planes; its logcial
    shape is plane-major, matching the captured allocation order.  d25 has four
    written texture outputs.  Padding bytes are reported separately and are
    never silently treated as logical data.
    """
    if stage == "d24":
        plan, error = recon.splat_plan(str(run), dispatch)
        if error is None:
            dst_w, dst_h = int(plan["dst_w"]), int(plan["dst_h"])
            logical = (2, dst_h, dst_w, 4)
            allocation = int(plan["allocation_words"]) * 4
            source = "recon.splat_plan"
        else:
            constants = recon.constants_of(dispatch)
            if len(constants) < 4:
                raise ComparisonError("d24 shape unavailable: %s" % error)
            dst_w, dst_h = int(constants[2]), int(constants[3])
            logical = (2, dst_h, dst_w, 4)
            allocation = resource_bytes(_output_resource(dispatch, "u0"))
            source = "trace constants"
        if min(dst_w, dst_h) <= 0:
            raise ComparisonError("d24 shape unavailable: zero destination from trace constants")
        size = _logical_bytes(logical)
        return {"u0": {
            "logical_shape": list(logical),
            "logical_bytes": size,
            "allocation_bytes": allocation,
            "padding_bytes": max(0, allocation - size),
            "shape_source": source,
        }}

    plan, error = recon.warp_init_plan(str(run), dispatch)
    if error is None:
        tw, th = int(plan["tw"]), int(plan["th"])
        ow, oh = int(plan["ow"]), int(plan["oh"])
        source = "recon.warp_init_plan"
    else:
        slots = _resources(dispatch)
        t4, t2 = slots.get("t4"), slots.get("t2")
        if t4 is None or t2 is None:
            raise ComparisonError("d25 shape unavailable: %s" % error)
        tw, th = int(t4.get("width") or 0), int(t4.get("height") or 0)
        ow, oh = int(t2.get("width") or 0), int(t2.get("height") or 0)
        source = "trace resources"
    if min(tw, th, ow, oh) <= 0:
        raise ComparisonError("d25 shape unavailable: zero cell or output shape")
    shapes = {
        "u0": (th, tw, 16),
        "u1": (oh, ow, 4),
        "u2": (oh, ow, 4),
        "u3": (oh, ow, 4),
    }
    result = {}
    expected_bytes = {"u0": 16, "u1": 4, "u2": 4, "u3": 4}
    for slot, shape in shapes.items():
        resource = _output_resource(dispatch, slot)
        channels = _format_bytes(resource)
        if channels != expected_bytes[slot]:
            raise ComparisonError(
                "%s.%s format has %s bytes per texel, expected %d"
                % (stage, slot, channels, expected_bytes[slot]))
        expected = resource_bytes(resource)
        size = _logical_bytes(shape)
        if expected < size:
            raise ComparisonError("%s.%s resource is %d bytes, logical shape needs %d"
                                  % (stage, slot, expected, size))
        result[slot] = {
            "logical_shape": list(shape),
            "logical_bytes": size,
            "allocation_bytes": expected,
            "padding_bytes": expected - size,
            "shape_source": source,
        }
    return result


def load_official(run, dispatch, slot):
    resource = _output_resource(dispatch, slot)
    name = resource.get("file")
    if not name:
        raise ComparisonError("missing official tensor file for %s.%s"
                              % (dispatch.get("dispatch"), slot))
    path = Path(run) / "trace" / "tensors" / name
    if not path.is_file():
        raise ComparisonError("missing official tensor %s" % path)
    raw = np.fromfile(path, dtype=np.uint8)
    expected = resource_bytes(resource)
    if raw.size != expected:
        raise ComparisonError(
            "official tensor %s has %d bytes, trace resource declares %d"
            % (path.name, raw.size, expected))
    return raw, resource


def _debug_bytes(array):
    value = np.asarray(array)
    if value.dtype == np.uint8:
        return np.ascontiguousarray(value).reshape(-1)
    if value.dtype == np.int8:
        return np.ascontiguousarray(value).view(np.uint8).reshape(-1)
    raise ComparisonError("unsupported debug tensor dtype %s" % value.dtype)


def compare_bytes(official, debug, shape, allocation_bytes, dtype_name):
    """Compare logical bytes, keep padding explicit, and report first divergence."""
    official = np.asarray(official, dtype=np.uint8).reshape(-1)
    debug_bytes = _debug_bytes(debug)
    logical_bytes = _logical_bytes(shape)
    if official.size < logical_bytes:
        raise ComparisonError("official tensor is shorter than logical shape %s" % (shape,))
    if debug_bytes.size < logical_bytes:
        raise ComparisonError(
            "debug tensor is %d bytes, logical shape %s needs %d"
            % (debug_bytes.size, shape, logical_bytes))
    if debug_bytes.size not in (logical_bytes, official.size):
        raise ComparisonError(
            "debug tensor shape is ambiguous: %d bytes, logical %d, official %d"
            % (debug_bytes.size, logical_bytes, official.size))
    if official.size != allocation_bytes and official.size != logical_bytes:
        raise ComparisonError(
            "official tensor is %d bytes, expected logical %d or allocation %d"
            % (official.size, logical_bytes, allocation_bytes))

    official_logical = official[:logical_bytes]
    debug_logical = debug_bytes[:logical_bytes]
    if dtype_name == "int8":
        left = official_logical.view(np.int8).astype(np.int16)
        right = debug_logical.view(np.int8).astype(np.int16)
    else:
        left = official_logical.astype(np.int16)
        right = debug_logical.astype(np.int16)
    difference = left - right
    mismatch = int(np.count_nonzero(difference))
    maxdiff = int(np.abs(difference).max()) if difference.size else 0
    first = None
    if mismatch:
        flat = int(np.flatnonzero(difference)[0])
        coord = [int(value) for value in np.unravel_index(flat, shape)]
        first = {
            "flat_index": flat,
            "coord": coord,
            "official": int(left[flat]),
            "debug": int(right[flat]),
        }

    official_padding = official[logical_bytes:]
    padding_mismatch = None
    padding_compared = bool(official_padding.size and debug_bytes.size == official.size)
    if padding_compared:
        padding_mismatch = int(np.count_nonzero(
            official_padding != debug_bytes[logical_bytes:]))

    return {
        "logical_shape": list(shape),
        "logical_bytes": logical_bytes,
        "allocation_bytes": int(allocation_bytes),
        "padding_bytes": int(official_padding.size),
        "padding_compared": padding_compared,
        "padding_mismatch": padding_mismatch,
        "debug_bytes": int(debug_bytes.size),
        "mismatch_count": mismatch,
        "maxdiff": maxdiff,
        "first_divergence": first,
        "status": "identical" if mismatch == 0 else "different",
    }


def compare_final(debug_final, normal_replay):
    official = np.asarray(debug_final)
    saved = np.asarray(normal_replay)
    if official.shape != saved.shape:
        raise ComparisonError(
            "final debug output shape %s differs from saved replay %s"
            % (official.shape, saved.shape))
    if official.dtype != np.uint8 or saved.dtype != np.uint8:
        raise ComparisonError("final debug and saved replay must both be uint8")
    difference = official.astype(np.int16) - saved.astype(np.int16)
    mismatch = int(np.count_nonzero(difference))
    maxdiff = int(np.abs(difference).max()) if difference.size else 0
    first = None
    if mismatch:
        flat = int(np.flatnonzero(difference.reshape(-1))[0])
        coord = [int(value) for value in np.unravel_index(flat, difference.shape)]
        first = {"coord": coord, "debug": int(official[tuple(coord)]),
                 "saved": int(saved[tuple(coord)])}
    rgb = np.abs(difference[..., :3].astype(np.float64))
    return {
        "shape": list(official.shape),
        "mismatch_count": mismatch,
        "maxdiff": maxdiff,
        "rgb_mae": float(rgb.mean()),
        "first_divergence": first,
        "status": "identical" if mismatch == 0 else "different",
    }


def debug_final_entry(debug_map):
    entries = [entry for entry in debug_map.get("outputs", [])
               if entry.get("public_name") == "generated_frame"
               or entry.get("semantic") == "generated_frame"]
    if len(entries) != 1:
        raise ComparisonError("ambiguous or missing final debug output entry")
    entry = entries[0]
    if not entry.get("graph_name"):
        raise ComparisonError("final debug entry has no graph output name")
    return entry


def load_debug_outputs(directory, entries):
    directory = Path(directory)
    values = {}
    for entry in entries:
        name = entry["graph_name"]
        path = directory / (name + ".npy")
        if not path.is_file():
            raise ComparisonError("missing debug output %s" % path)
        values[name] = np.load(path, allow_pickle=False)
    return values


def _load_sibling(name):
    # replay.py imports PIL for image saving, but load_frame does not use it.
    # Keep the one-frame diagnostic runnable in the validated ORT environment
    # even when Pillow is absent there.
    if name == "replay":
        try:
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            import types
            pil = types.ModuleType("PIL")
            image = types.ModuleType("PIL.Image")
            image.fromarray = lambda *args, **kwargs: None
            pil.Image = image
            sys.modules["PIL"] = pil
            sys.modules["PIL.Image"] = image
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_debug_onnx(run, package, frame, previous_frame, entries, final_entry):
    """Run the debug model on the captured application inputs.

    This is intentionally explicit: it is only called with ``--execute`` after
    the real trace exists.  The runtime plan/feeds come from the same
    ``replay.load_frame`` and ``runtime.prepare`` path used by validated replay.
    """
    import onnxruntime as ort
    from xefg_sdk import Config, UiMode, Xefg2xRuntime

    replay = _load_sibling("replay")
    manifest = read_json(Path(run) / "application_inputs" / "manifest.json")
    previous_id = frame - 1 if previous_frame is None else previous_frame
    previous = replay.load_frame(run, manifest, previous_id, mode=MODE)
    current = replay.load_frame(run, manifest, frame, mode=MODE)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    with Xefg2xRuntime.load(package) as runtime:
        _, _plan, feeds = runtime.prepare(
            previous, current, Config(ui_mode=UiMode(MODE)))
    debug_info = read_json(Path(package) / "manifest.json")["debug"]["mode%d" % MODE]
    model = Path(package) / debug_info["model"]
    names = [entry["graph_name"] for entry in entries + [final_entry]]
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    values = session.run(names, feeds)
    return dict(zip(names, values))


def run_recon(run, dispatch, stage, shapes, debug_values=None):
    """Replay d24/d25 from captured official inputs, separating propagation."""
    capture = {}
    if stage == "d24":
        result, error = recon.reconstruct_splat(str(run), dispatch, capture=capture)
    else:
        result, error = recon.reconstruct_warp_init(str(run), dispatch, capture=capture)
    if error is not None:
        return {"status": "unavailable", "error": str(error)}
    if len(result) == 5:
        mismatch, total, dims, note, maxdiff = result
    else:
        mismatch, total, dims, note = result
        official, _resource = load_official(run, dispatch, "u0")
        got = np.asarray(capture["u0"], dtype=np.uint8).reshape(-1)
        maxdiff = int(np.abs(official.astype(np.int16)
                             - got.astype(np.int16)).max()) if got.size else 0
    report = {
        "status": "measured",
        "vs_official": {
            "mismatch_count": int(mismatch),
            "total_bytes": int(total),
            "maxdiff": int(maxdiff),
            "dims": [int(value) for value in dims],
            "note": note,
        },
        "vs_debug": None,
    }
    if debug_values is not None:
        comparisons = {}
        for slot, array in capture.items():
            entry = debug_values.get(slot)
            if entry is None:
                continue
            info = shapes[slot]
            if entry.get("shape"):
                _check_debug_shape({"graph_name": slot, "shape": entry["shape"]},
                                   entry["array"], info["logical_shape"])
            comparisons[slot] = compare_bytes(
                np.asarray(array).reshape(-1), entry["array"],
                info["logical_shape"], info["allocation_bytes"], entry["dtype"])
        report["vs_debug"] = comparisons
    return report


def _entry_node(entries):
    nodes = {entry.get("node") for entry in entries}
    if len(nodes) != 1:
        raise ComparisonError("ambiguous debug node: %s" % sorted(nodes))
    return nodes.pop()


def _check_debug_shape(entry, array, logical_shape):
    """Reject a debug array whose declared rank/shape does not match the trace."""
    value = np.asarray(array)
    declared_rank = len(entry.get("shape") or [])
    if declared_rank not in (1, len(logical_shape)):
        raise ComparisonError(
            "%s declares rank %d but %s has rank %d"
            % (entry.get("graph_name"), declared_rank, logical_shape, len(logical_shape)))
    if declared_rank != 1 and tuple(value.shape) != tuple(logical_shape):
        raise ComparisonError(
            "debug tensor %s shape %s differs from logical shape %s"
            % (entry.get("graph_name"), value.shape, tuple(logical_shape)))


def _stage_report(run, dispatch, entries, shapes, debug_values):
    report = {
        "dispatch": dispatch_signature(dispatch),
        "debug_node": _entry_node(entries),
        "debug_entries": [
            {"slot": entry["slot"], "graph_name": entry["graph_name"],
             "dtype": entry["dtype"], "shape": entry["shape"]}
            for entry in entries
        ],
        "outputs": {},
    }
    first_slot = None
    first = None
    for entry in entries:
        slot = entry["slot"]
        info = shapes[slot]
        official, resource = load_official(run, dispatch, slot)
        output = {
            "official_file": resource.get("file"),
            "official_bytes": int(official.size),
            "logical_shape": info["logical_shape"],
            "logical_bytes": info["logical_bytes"],
            "allocation_bytes": info["allocation_bytes"],
            "padding_bytes": info["padding_bytes"],
            "shape_source": info["shape_source"],
        }
        if debug_values is not None:
            _check_debug_shape(entry, debug_values[slot]["array"], info["logical_shape"])
            output.update(compare_bytes(
                official, debug_values[slot]["array"], info["logical_shape"],
                info["allocation_bytes"], debug_values[slot]["dtype"]))
            if first is None and output.get("first_divergence"):
                first_slot = slot
                first = output["first_divergence"]
        else:
            output["status"] = "official_only"
        report["outputs"][slot] = output
    report["first_divergence_slot"] = first_slot
    report["first_divergence"] = first
    report["unmapped_official_slots"] = _unmapped_slots(run, dispatch, entries)
    return report


def _unmapped_slots(run, dispatch, entries):
    mapped = {entry["slot"] for entry in entries}
    result = []
    for slot, resource in sorted(_resources(dispatch).items()):
        if resource.get("kind") != "uav" or slot in mapped:
            continue
        name = resource.get("file")
        size = resource_bytes(resource)
        zero = None
        if name:
            path = Path(run) / "trace" / "tensors" / name
            if path.is_file():
                zero = not bool(np.fromfile(path, dtype=np.uint8).any())
        result.append({"slot": slot, "file": name, "bytes": size, "all_zero": zero})
    return result


def _git_state():
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                         text=True).strip()
        status = subprocess.check_output(["git", "-C", str(repo), "status", "--short"],
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, status


def _debug_maps(package, debug_map_path=None):
    if debug_map_path is not None:
        return read_json(debug_map_path)
    info = read_json(Path(package) / "manifest.json")["debug"]["mode%d" % MODE]
    return read_json(Path(package) / info["outputs"])


def _package_artifacts(package):
    package = Path(package)
    manifest = read_json(package / "manifest.json")
    names = ["manifest.json"]
    for mode in ("mode1", "mode3", "mode4"):
        info = manifest.get("debug", {}).get(mode)
        if info:
            names.extend([info["model"], info["outputs"]])
    return [{"file": name, "bytes": (package / name).stat().st_size,
             "sha256": _sha256(package / name)} for name in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="capture root containing trace/ and application_inputs/")
    parser.add_argument("--package", type=Path, required=True,
                        help="package directory containing manifest.json and debug outputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=8)
    parser.add_argument("--previous-frame", type=int, default=None)
    parser.add_argument("--stages", nargs="+", choices=sorted(STAGES),
                        default=["d24", "d25"])
    parser.add_argument("--debug-map", type=Path, default=None,
                        help="override debug/mode3.outputs.json")
    parser.add_argument("--reference-run", type=Path, default=None,
                        help="retained reference capture used to validate program hashes")
    parser.add_argument("--normal-replay", type=Path, default=None,
                        help="saved normal-package replay frame_XXXXX.npy")
    parser.add_argument("--debug-final", type=Path, default=None,
                        help="already-saved debug n0038_u0 .npy (no inference)")
    parser.add_argument("--debug-outputs", type=Path, default=None,
                        help="directory of saved n0024/n0025 .npy outputs")
    parser.add_argument("--execute", action="store_true",
                        help="run debug ONNX; requires the real trace")
    parser.add_argument("--no-recon", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and args.debug_outputs is not None:
        raise ComparisonError("--execute and --debug-outputs are mutually exclusive")
    if args.execute and args.normal_replay is None:
        raise ComparisonError(
            "--execute requires --normal-replay so the debug final output is "
            "compared with the saved normal replay before intermediates")
    if args.out.exists():
        raise ComparisonError("output directory already exists: %s" % args.out)
    args.out.mkdir(parents=True)

    run = capture_root(args.run)
    meta_report = validate_trace_meta(read_trace_meta(run), args.frame,
                                      require_complete=args.execute)
    rows = recon.read_events(str(run))
    debug_map = _debug_maps(args.package, args.debug_map)
    final_entry = debug_final_entry(debug_map)

    stages = {}
    selected = {}
    identities = {}
    for stage in args.stages:
        dispatch = select_stage_dispatch(rows, stage, args.frame)
        entries = match_debug_entries(debug_map, stage)
        node = bind_dispatch_to_debug(dispatch, entries, stage, rows)
        dispatch["_run"] = str(run)
        shapes = stage_shapes(run, dispatch, stage)
        reference = reference_program_hashes(args.package, stage, args.reference_run)
        identities[stage] = validate_program_identity(dispatch, reference, stage)
        selected[stage] = (dispatch, entries, node, shapes)

    debug_values = None
    if args.debug_outputs is not None:
        debug_values = {}
        for stage in args.stages:
            entries = selected[stage][1]
            loaded = load_debug_outputs(args.debug_outputs, entries)
            for entry in entries:
                debug_values[entry["graph_name"]] = {
                    "array": loaded[entry["graph_name"]],
                    "dtype": entry["dtype"],
                    "slot": entry["slot"],
                    "shape": entry["shape"],
                }
    elif args.execute:
        entries = [entry for stage in args.stages for entry in selected[stage][1]]
        loaded = run_debug_onnx(run, args.package, args.frame, args.previous_frame,
                                entries, final_entry)
        debug_values = {}
        for stage in args.stages:
            for entry in selected[stage][1]:
                name = entry["graph_name"]
                debug_values[name] = {"array": loaded[name], "dtype": entry["dtype"],
                                      "slot": entry["slot"], "shape": entry["shape"]}
        debug_values["final"] = {"array": loaded[final_entry["graph_name"]],
                                 "dtype": "uint8", "slot": final_entry.get("slot"),
                                 "shape": final_entry.get("shape")}

    final_result = {"status": "skipped", "reason": "no --normal-replay supplied"}
    debug_final = None
    if args.debug_final is not None:
        if not args.debug_final.is_file():
            raise ComparisonError("missing debug final: %s" % args.debug_final)
        debug_final = np.load(args.debug_final, allow_pickle=False)
    elif debug_values is not None and "final" in debug_values:
        debug_final = debug_values["final"]["array"]
    if args.normal_replay is not None:
        if not args.normal_replay.is_file():
            raise ComparisonError("missing saved normal replay: %s" % args.normal_replay)
        if debug_final is None:
            final_result = {
                "status": "skipped",
                "reason": "debug final unavailable; pass --debug-final or --execute",
            }
        else:
            final_result = compare_final(
                debug_final, np.load(args.normal_replay, allow_pickle=False))
            final_result["debug_source"] = (
                str(args.debug_final.resolve()) if args.debug_final is not None
                else "debug ONNX " + final_entry["graph_name"])
            final_result["normal_replay"] = str(args.normal_replay.resolve())

    for stage in args.stages:
        dispatch, entries, node, shapes = selected[stage]
        stage_values = None
        if debug_values is not None:
            stage_values = {}
            for entry in entries:
                stage_values[entry["slot"]] = debug_values[entry["graph_name"]]
        report = _stage_report(run, dispatch, entries, shapes, stage_values)
        report["program_identity"] = identities[stage]
        if not args.no_recon:
            report["recon"] = run_recon(run, dispatch, stage, shapes, stage_values)
        stages[stage] = report

    commit, status = _git_state()
    report = {
        "schema": "xefg.tensor_comparison/1",
        "status": "measured" if debug_values is not None else "incomplete",
        "command": sys.argv,
        "frame": args.frame,
        "mode": MODE,
        "run": str(run),
        "package": str(Path(args.package).resolve()),
        "git_commit": commit,
        "git_status": status,
        "python": platform.python_version(),
        "executor": {"provider": "CPUExecutionProvider", "intra_op_threads": 1,
                     "inter_op_threads": 1},
        "debug_map": {"model": debug_map.get("model"),
                      "final_output": final_entry["graph_name"]},
        "trace_meta": meta_report,
        "final_replay": final_result,
        "stages": stages,
        "package_artifacts": _package_artifacts(args.package),
        "limitations": [
            "Bounded to d24 (splat) and d25 (warp + flow init) for one Mode 3 frame.",
            "Official tensors come only from <run>/trace/tensors; no tensor is inferred from a final image.",
            "Debug ONNX inference is disabled unless --execute is passed.",
            "d24 logical bytes are two plane-major planes; an allocation tail is reported as padding.",
            "d25 reports u0..u3 from the debug map; a bound-but-unmapped u4 is listed separately and not compared.",
            "Recon replay is standalone and may be unavailable when retained disassembly is missing.",
            "Any missing tensor, missing shape or ambiguous stage mapping is a hard failure.",
        ],
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Tensor comparison", "",
             "Frame %d, mode %d. Debug comparison: %s." % (
                 args.frame, MODE, report["status"]), ""]
    lines += ["## Final replay control", ""]
    if final_result.get("status") == "skipped":
        lines += ["Skipped: " + final_result["reason"] + "."]
    else:
        lines += ["| Mismatch bytes | Max diff | RGB MAE |", "|---:|---:|---:|",
                  "| %d | %d | %.9f |" % (
                      final_result["mismatch_count"], final_result["maxdiff"],
                      final_result["rgb_mae"])]
    for stage in args.stages:
        row = stages[stage]
        lines += ["", "## %s" % stage, "",
                  "Dispatch %s, key `%s`, debug node %s." % (
                      row["dispatch"]["dispatch"], row["dispatch"]["key"],
                      row["debug_node"]), "",
                  "| Slot | Status | Mismatch | Max diff | First divergence |",
                  "|---|---|---:|---:|---|"]
        for slot, output in row["outputs"].items():
            first = output.get("first_divergence")
            where = "-" if not first else "index %d coord %s" % (
                first["flat_index"], first["coord"])
            lines.append("| %s | %s | %s | %s | %s |" % (
                slot, output.get("status"), output.get("mismatch_count", "-"),
                output.get("maxdiff", "-"), where))
        recon_row = row.get("recon")
        if recon_row:
            if recon_row["status"] == "unavailable":
                lines += ["", "Recon unavailable: " + recon_row["error"]]
            else:
                vs = recon_row["vs_official"]
                lines += ["", "Standalone recon vs official: %d/%d bytes differ, max %d. %s"
                          % (vs["mismatch_count"], vs["total_bytes"],
                             vs["maxdiff"], vs["note"])]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        print("compare_tensors: %s" % error, file=sys.stderr)
        raise SystemExit(2)
