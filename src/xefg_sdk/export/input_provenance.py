#!/usr/bin/env python
"""Resolve the application-input boundary of the fg78 XeFG graph.

The trace exposes the provider's dispatches, the bound resource pointers for
each SRV slot and optional tensor snapshots.  This tool answers three separate
questions without inferring any transformation that was not observed:

  * do the documented slot aliases really name the same resource in every
    frame (for example ``d1.t0 == d25.t2 == d38.t11``);
  * what frame does each aliased texture contain - the previous application
    frame, the current one, or a frame-invariant buffer such as the synthetic
    UI; and
  * is the 32x32 ``d25.t6`` LUT the same provider-owned constant on every
    frame.

Frame identity is only *proved* when the trace content can be matched against
application inputs saved by ``XefgOracle`` (``manifest.json`` plus ``inputs/``)
or by a scene-only exporter that links the repository's exact ``scene.cpp``.
Native capture output uses ``application_inputs/manifest.json`` with file
names relative to that directory; both layouts are accepted.  Without a
comparison, previous/current colour can still be related to one another inside
the trace (``prev(f) == current(f-1)``), but motion vectors and depth at frame
``f`` cannot be distinguished from those at ``f-1``; the report records
``unresolved`` rather than guessing.  The UI is genuinely static in the legacy
scene, so its content is frame-invariant and its current/previous identity is
not observable from content alone.

The observed d38 mode selects the boundary map.  All supported modes share:

    prev_color     d1.t0   d25.t2   d38.t11
    current_color  d1.t1   d25.t3   d38.t12
    motion         d24.t0  d25.t4
    depth          d24.t1  d25.t5
    lut            d25.t6

The mode 1/3 report also inspects d38.t9 (UI in mode 3, a placeholder in mode 1).
Mode 4 adds d38.t8 to current_color and d38.t10 as the current backbuffer; its
unused t9 binding is reported separately.
"""
# Internal provenance helper retained for synthetic tests; not a supported CLI
# entry point.
import argparse
import hashlib
import json
import os
import re
import sys

SCHEMA = "xefg.input_provenance/1"
KNOWN_LUT_SHA256 = (
    "49e55b012bc0575066e0342bf533f142d5895348152a2c9fe31d00406ded1e5d"
)
LEGACY_D38_MODES = (1, 3)
SUPPORTED_D38_MODES = (1, 3, 4)
# Compatibility alias; supported modes are listed separately.
BOUNDARY_D38_MODES = LEGACY_D38_MODES

__all__ = [
    "SCHEMA",
    "KNOWN_LUT_SHA256",
    "LEGACY_D38_MODES",
    "SUPPORTED_D38_MODES",
    "BOUNDARY_D38_MODES",
    "BOUNDARY_INPUTS",
    "MODE_BOUNDARY_INPUTS",
    "load_trace",
    "load_application_inputs",
    "build_report",
    "main",
]


class InputSpec:
    """One named input in the semantic boundary."""

    def __init__(self, name, kind, slots, expected_shift=None, application=True):
        self.name = name
        self.kind = kind
        self.slots = tuple(slots)
        self.expected_shift = expected_shift
        self.application = application

    def as_dict(self):
        return {
            "kind": self.kind,
            "slots": [{"dispatch": d, "slot": s} for d, s in self.slots],
            "hypothesis_frame_shift": self.expected_shift,
            "application_input": self.application,
        }


BOUNDARY_INPUTS = (
    InputSpec("prev_color", "color", ((1, "t0"), (25, "t2"), (38, "t11")),
              expected_shift=-1),
    InputSpec("current_color", "color", ((1, "t1"), (25, "t3"), (38, "t12")),
              expected_shift=0),
    InputSpec("motion", "motion", ((24, "t0"), (25, "t4")),
              expected_shift=0),
    InputSpec("depth", "depth", ((24, "t1"), (25, "t5")),
              expected_shift=0),
    InputSpec("ui", "ui", ((38, "t9"),), expected_shift=None),
    InputSpec("lut", "lut", ((25, "t6"),), expected_shift=None,
              application=False),
)

MODE4_BOUNDARY_INPUTS = (
    InputSpec("prev_color", "color", ((1, "t0"), (25, "t2"), (38, "t11")),
              expected_shift=-1),
    InputSpec("current_color", "color",
              ((1, "t1"), (25, "t3"), (38, "t12"), (38, "t8")),
              expected_shift=0),
    InputSpec("motion", "motion", ((24, "t0"), (25, "t4")),
              expected_shift=0),
    InputSpec("depth", "depth", ((24, "t1"), (25, "t5")),
              expected_shift=0),
    InputSpec("backbuffer", "backbuffer", ((38, "t10"),), expected_shift=0),
    InputSpec("lut", "lut", ((25, "t6"),), expected_shift=None,
              application=False),
)

MODE_BOUNDARY_INPUTS = {
    1: BOUNDARY_INPUTS,
    3: BOUNDARY_INPUTS,
    4: MODE4_BOUNDARY_INPUTS,
}

_APP_PATTERNS = (
    ("color", re.compile(r"^color_(\d+)\.bgra$", re.IGNORECASE)),
    ("ui", re.compile(r"^ui_(\d+)\.bgra$", re.IGNORECASE)),
    ("backbuffer", re.compile(r"^backbuffer_(\d+)\.bgra$", re.IGNORECASE)),
    ("depth", re.compile(r"^depth_(\d+)\.r32f$", re.IGNORECASE)),
    ("motion", re.compile(r"^mv_(\d+)\.rg16f$", re.IGNORECASE)),
    ("motion", re.compile(r"^motion_(\d+)\.rg16f$", re.IGNORECASE)),
)

_MANIFEST_FIELDS = (
    ("color", ("color",)),
    ("ui", ("ui",)),
    ("backbuffer", ("backbuffer",)),
    ("depth", ("depth",)),
    ("motion", ("motion", "mv", "motion_vectors")),
)


def _sha256_file(path, cache):
    path = os.path.abspath(path)
    if path in cache:
        return cache[path]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    cache[path] = digest.hexdigest()
    return cache[path]


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_trace(path):
    """Parse ``trace.jsonl`` into dispatch records keyed by (frame, dispatch).

    Only SRV resources are retained; UAV output names are irrelevant here.  A
    duplicate (frame, dispatch) event is reported instead of overwriting the
    first one, so a malformed fixture cannot silently hide a slot.
    """
    trace_path = os.path.abspath(path)
    events = {}
    frames = set()
    errors = []
    duplicates = []
    dispatch_count = 0
    with open(trace_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                errors.append("line %d: invalid JSON: %s" % (line_number, exc))
                continue
            if event.get("type") != "dispatch":
                continue
            frame = _as_int(event.get("frame"))
            dispatch = _as_int(event.get("dispatch"))
            if frame is None or dispatch is None:
                errors.append("line %d: dispatch without frame/dispatch id"
                              % line_number)
                continue
            dispatch_count += 1
            resources = []
            lookup = {}
            for raw in event.get("resources") or ():
                if raw.get("kind") != "srv":
                    continue
                slot = raw.get("slot")
                if slot in lookup:
                    duplicates.append(
                        "frame %d dispatch %d: duplicate SRV slot %s"
                        % (frame, dispatch, slot))
                    continue
                resource = {
                    "parameter": raw.get("parameter"),
                    "slot": slot,
                    "index": raw.get("index"),
                    "resource": raw.get("resource"),
                    "format": raw.get("format"),
                    "dimension": raw.get("dimension"),
                    "width": raw.get("width"),
                    "height": raw.get("height"),
                    "array": raw.get("array"),
                    "mips": raw.get("mips"),
                    "file": raw.get("file"),
                }
                resources.append(resource)
                lookup[slot] = resource
            record = {
                "frame": frame,
                "dispatch": dispatch,
                "line": line_number,
                "seq": event.get("seq"),
                "present": event.get("present"),
                "pso": event.get("pso"),
                "sha256": event.get("sha256"),
                "constants": event.get("constants") or [],
                "resources": resources,
                "lookup": lookup,
            }
            key = (frame, dispatch)
            if key in events:
                duplicates.append(
                    "frame %d dispatch %d: duplicate dispatch event"
                    % (frame, dispatch))
                continue
            events[key] = record
            frames.add(frame)
    # Dispatch ids are global and monotonically increase across frames
    # (frame 1: 1..38, frame 2: 39..76, ...).  The boundary slot map is
    # expressed in frame-relative node ids (d1, d24, d25, d38), so assign the
    # ordinal inside each frame.
    by_node = {}
    for frame in sorted(frames):
        frame_events = sorted(
            (record for (event_frame, _), record in events.items()
             if event_frame == frame),
            key=lambda record: record["dispatch"])
        for node, record in enumerate(frame_events, 1):
            record["node"] = node
            by_node[(frame, node)] = record
    return {
        "path": trace_path,
        "events": events,
        "by_node": by_node,
        "frames": sorted(frames),
        "dispatch_count": dispatch_count,
        "errors": errors,
        "duplicates": duplicates,
    }


def _trace_meta(trace_path):
    directory = os.path.dirname(os.path.abspath(trace_path))
    for name in ("meta.json", "manifest.json"):
        data = _read_json(os.path.join(directory, name))
        if isinstance(data, dict):
            return data, os.path.join(directory, name)
    return {}, None


def _manifest_candidates(root):
    """Return manifest paths to try, most specific layout first."""
    if os.path.isfile(root):
        return [root]
    return [
        os.path.join(root, "manifest.json"),
        # Native oracle/capture layout: <root>/application_inputs/manifest.json
        os.path.join(root, "application_inputs", "manifest.json"),
        os.path.join(root, "inputs", "manifest.json"),
    ]


def _find_manifest(root):
    for candidate in _manifest_candidates(root):
        if os.path.isfile(candidate):
            return candidate
    return None


def _manifest_records(manifest, base, errors):
    """Convert a manifest's declared frame inputs into file records.

    The native ``xefg.application_inputs/1`` format stores relative names
    directly in each frame's ``inputs`` object (for example ``color_00000.bgra``
    next to the manifest).  The older oracle fixture layout prefixes them with
    ``inputs/``.  Both are resolved against the manifest's own directory.
    """
    records = []
    frames = manifest.get("frames")
    if isinstance(frames, dict):
        frames = [dict(value, index=key) if isinstance(value, dict) else value
                  for key, value in frames.items()]
    if not isinstance(frames, (list, tuple)):
        errors.append("manifest has no frames list")
        return records
    for item in frames:
        if not isinstance(item, dict):
            errors.append("manifest frame is not an object")
            continue
        frame = _as_int(item.get("index"))
        if frame is None:
            frame = _as_int(item.get("frame"))
        if frame is None:
            errors.append("manifest frame without an index")
            continue
        inputs = item.get("inputs")
        if not isinstance(inputs, dict):
            # Some manifests declare the keys directly on the frame object.
            inputs = {key: value for key, value in item.items()
                      if key not in ("index", "frame")}
        for kind, aliases in _MANIFEST_FIELDS:
            field = None
            relative = None
            for alias in aliases:
                if inputs.get(alias):
                    field = alias
                    relative = inputs[alias]
                    break
            if not relative:
                continue
            file_path = os.path.join(
                base, str(relative).replace("/", os.sep))
            records.append({"frame": frame, "kind": kind,
                            "path": file_path,
                            "source": "manifest.inputs.%s" % field})
    return records


def _raw_records(directories, errors, seen=None):
    """Auto-discover raw application input files by filename."""
    records = []
    seen = seen if seen is not None else set()
    for directory in directories:
        try:
            names = os.listdir(directory)
        except OSError as exc:
            errors.append("cannot list %s: %s" % (directory, exc))
            continue
        for name in names:
            for kind, pattern in _APP_PATTERNS:
                match = pattern.match(name)
                if not match:
                    continue
                file_path = os.path.abspath(os.path.join(directory, name))
                if file_path in seen:
                    break
                seen.add(file_path)
                records.append({
                    "frame": int(match.group(1)),
                    "kind": kind,
                    "path": file_path,
                    "source": "filename",
                })
                break
    return records


def _raw_directories(root, manifest_path):
    directories = []
    candidates = [root]
    if manifest_path:
        candidates.insert(0, os.path.dirname(manifest_path))
    if os.path.isdir(root):
        candidates.extend([
            os.path.join(root, "inputs"),
            os.path.join(root, "application_inputs"),
        ])
    for directory in candidates:
        directory = os.path.abspath(directory)
        if os.path.isdir(directory) and directory not in directories:
            directories.append(directory)
    return directories


def load_application_inputs(path):
    """Load application inputs from a manifest or raw frame directory.

    Accepts ``manifest.json`` itself, an oracle/capture root containing
    ``manifest.json`` or ``application_inputs/manifest.json``, or a directory
    of raw ``color_*.bgra`` / ``ui_*.bgra`` / ``backbuffer_*.bgra`` /
    ``depth_*.r32f`` / ``mv_*.rg16f`` files.  A manifest whose declared paths
    are unusable falls back to filename discovery instead of silently
    reporting an empty inventory.  Every file that exists is hashed; missing
    files are recorded but do not abort the inventory.
    """
    root = os.path.abspath(path)
    manifest_path = _find_manifest(root)
    base = os.path.dirname(manifest_path) if manifest_path else root

    records = []
    errors = []
    if manifest_path:
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            errors.append("cannot read manifest %s" % manifest_path)
        else:
            records = _manifest_records(manifest, base, errors)
            if not records:
                errors.append(
                    "manifest %s declared no usable input records"
                    % manifest_path)
            elif not any(os.path.isfile(record["path"])
                         for record in records):
                errors.append(
                    "manifest %s paths do not exist; falling back to raw "
                    "filename discovery" % manifest_path)
                records = []

    if not records:
        records = _raw_records(
            _raw_directories(root, manifest_path), errors)
        if not records and not manifest_path:
            errors.append(
                "no manifest.json and no colour/ui/backbuffer/depth/mv files "
                "in %s" % root)

    cache = {}
    by_kind = {}
    frames = set()
    missing = []
    for record in records:
        frames.add(record["frame"])
        if not os.path.isfile(record["path"]):
            missing.append({"frame": record["frame"],
                            "kind": record["kind"],
                            "path": record["path"]})
            continue
        try:
            record["sha256"] = _sha256_file(record["path"], cache)
        except OSError as exc:
            errors.append("cannot hash %s: %s" % (record["path"], exc))
            continue
        by_kind.setdefault(record["kind"], {}).setdefault(
            record["sha256"], set()).add(record["frame"])
    for kind in by_kind:
        by_kind[kind] = {sha: sorted(values)
                         for sha, values in by_kind[kind].items()}
    return {
        "root": root,
        "manifest": manifest_path,
        "frames": sorted(frames),
        "by_kind": by_kind,
        "files": [{key: value for key, value in record.items()
                   if key != "sha256"} for record in records],
        "errors": errors,
        "missing": missing,
        "sha256_count": len(cache),
    }


def _resolve_aliases(trace, spec, tensor_dir, hash_cache):
    """Resolve one input's alias slots and hash the captured tensors."""
    per_frame = []
    missing = []
    for frame in trace["frames"]:
        aliases = []
        pointers = set()
        hashes = set()
        hashed_any = False
        for dispatch, slot in spec.slots:
            event = trace["by_node"].get((frame, dispatch))
            resource = event["lookup"].get(slot) if event else None
            if resource is None:
                entry = {
                    "node": dispatch,
                    "dispatch": event["dispatch"] if event else None,
                    "slot": slot,
                    "line": event["line"] if event else None,
                    "status": "missing",
                }
                aliases.append(entry)
                missing.append({"frame": frame, "node": dispatch,
                                "slot": slot})
                continue
            entry = {
                "node": dispatch,
                "dispatch": event["dispatch"],
                "slot": slot,
                "line": event["line"],
                "seq": event.get("seq", event["line"]),
                "resource": resource.get("resource"),
                "format": resource.get("format"),
                "dimension": resource.get("dimension"),
                "width": resource.get("width"),
                "height": resource.get("height"),
                "tensor": os.path.basename(str(resource.get("file")))
                if resource.get("file") else None,
            }
            if resource.get("resource") is not None:
                pointers.add(str(resource["resource"]))
            file_name = resource.get("file")
            if file_name:
                file_path = os.path.join(tensor_dir,
                                         os.path.basename(str(file_name)))
                entry["tensor_path"] = file_path
                if os.path.isfile(file_path):
                    try:
                        digest = _sha256_file(file_path, hash_cache)
                        entry["sha256"] = digest
                        entry["bytes"] = os.path.getsize(file_path)
                        hashes.add(digest)
                        hashed_any = True
                    except OSError as exc:
                        entry["status"] = "unreadable"
                        entry["error"] = str(exc)
                else:
                    entry["status"] = "missing_tensor"
            else:
                entry["status"] = "no_tensor_file"
            aliases.append(entry)
        pointer_consistent = len(pointers) <= 1
        content_consistent = len(hashes) <= 1
        present = [entry for entry in aliases
                   if entry.get("status") not in
                   ("missing", "missing_tensor", "unreadable", "no_tensor_file")]
        if len(present) != len(aliases):
            status = "unavailable"
        elif not pointer_consistent or not content_consistent:
            status = "mismatch"
        elif hashed_any:
            status = "confirmed"
        else:
            status = "resource_only"
        per_frame.append({
            "frame": frame,
            "aliases": aliases,
            "pointer_values": sorted(pointers),
            "content_sha256_values": sorted(hashes),
            "content_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
            "present_aliases": len(present),
            "all_aliases": len(aliases),
            "pointer_consistent": pointer_consistent,
            "content_consistent": content_consistent,
            "status": status,
        })

    statuses = {item["status"] for item in per_frame}
    if "mismatch" in statuses:
        alias_status = "mismatch"
    elif statuses <= {"confirmed"}:
        alias_status = "confirmed"
    elif statuses <= {"resource_only"}:
        alias_status = "resource_only"
    elif statuses <= {"unavailable"}:
        alias_status = "unavailable"
    else:
        alias_status = "partial"
    return {
        "name": spec.name,
        "kind": spec.kind,
        "slots": [{"dispatch": d, "slot": s} for d, s in spec.slots],
        "hypothesis_frame_shift": spec.expected_shift,
        "alias_status": alias_status,
        "missing_aliases": missing,
        "per_frame": per_frame,
    }


def _annotate_application_candidates(resolved, app):
    """Attach matching application frames to every per-frame content hash."""
    kind_index = app["by_kind"].get(resolved["kind"], {})
    for item in resolved["per_frame"]:
        digest = item.get("content_sha256")
        item["application_frames"] = (
            list(kind_index.get(digest, ())) if digest else [])


def _reference_frame_map(resolved_inputs, app):
    """Choose the strongest trace -> application frame mapping available.

    Colour is the natural reference, but a static colour scene repeats the
    same texels on every frame.  Motion vectors or depth can then supply the
    mapping when their own content is non-static.
    """
    if app is None:
        return {}, None
    for name in ("current_color", "motion", "depth"):
        resolved = resolved_inputs.get(name)
        if not resolved:
            continue
        mapping = {}
        usable = True
        # Missing snapshots cannot constrain this map, but remain in the report.
        frame_items = [item for item in resolved["per_frame"]
                       if item.get("application_frames")]
        if not frame_items:
            continue
        for item in frame_items:
            candidates = item.get("application_frames") or []
            if len(candidates) != 1:
                usable = False
                break
            mapping[item["frame"]] = candidates[0]
        if usable and len(set(mapping.values())) == len(mapping):
            return mapping, "application_inputs:%s" % name
    return {}, None


def _trace_prev_relation(resolved_inputs):
    """Check ``prev_color(f) == current_color(f-1)`` using trace hashes."""
    previous = {item["frame"]: item.get("content_sha256")
                for item in resolved_inputs["prev_color"]["per_frame"]}
    current = {item["frame"]: item.get("content_sha256")
               for item in resolved_inputs["current_color"]["per_frame"]}
    checked = []
    for frame in sorted(previous):
        if frame - 1 not in current:
            continue
        if previous[frame] is None or current[frame - 1] is None:
            continue
        checked.append({
            "frame": frame,
            "previous": previous[frame],
            "previous_frame": frame - 1,
            "current": current[frame - 1],
            "match": previous[frame] == current[frame - 1],
        })
    return checked


def _per_frame_status(item, reference_map, kind_available):
    digest = item.get("content_sha256")
    if digest is None:
        return "unavailable"
    if not kind_available:
        return "application_inputs_missing"
    candidates = item.get("application_frames") or []
    if not candidates:
        return "unmatched"
    if not reference_map:
        return "frame_invariant" if len(candidates) > 1 else "no_reference"
    reference = reference_map.get(item["frame"])
    if reference is None:
        return "no_reference"
    item["reference_application_frame"] = reference
    item["observed_shifts"] = sorted({c - reference for c in candidates})
    if len(candidates) > 1:
        return "frame_invariant"
    return "unique"


def _ui_off_analysis(resolved, current_color, ui_off, reference_map=None):
    """Describe d38.t9 when the application did not supply a UI texture.

    XeFG still binds a texture to the slot.  In the profiled mode1 UI-off
    capture that texture aliases the current application colour.  This is not
    an application UI input and must not be compared against zeroed UI files.
    """
    reference_map = reference_map or {}
    current_hashes = {item["frame"]: item.get("content_sha256")
                      for item in current_color["per_frame"]}
    matches_current = []
    for item in resolved["per_frame"]:
        digest = item.get("content_sha256")
        reference = reference_map.get(item["frame"])
        if reference is not None:
            item["reference_application_frame"] = reference
        if digest is None:
            item["status"] = "unavailable"
        elif digest == current_hashes.get(item["frame"]):
            item["status"] = "ui_off_current_color"
            item["current_color_sha256"] = current_hashes[item["frame"]]
            if reference is not None:
                item["application_frames"] = [reference]
            matches_current.append(item["frame"])
        else:
            item["status"] = "ui_off_other_binding"
    statuses = {item["status"] for item in resolved["per_frame"]}
    if "unavailable" in statuses:
        status = "unavailable"
        shift = None
    elif statuses == {"ui_off_current_color"}:
        status = "ui_off_current_color_placeholder"
        shift = 0
    elif statuses <= {"ui_off_current_color"}:
        status = "ui_off_current_color_placeholder"
        shift = 0
    else:
        status = "ui_off_unexpected_binding"
        shift = None
    return {
        "kind": resolved["kind"],
        "alias_status": resolved["alias_status"],
        "status": status,
        "frame_shift": shift,
        "frame_shift_method": (
            "d38.t9 aliases current_color because the application UI is off"
            if shift == 0 else "ui disabled; slot binding not a UI input"),
        "ui_off": ui_off,
        "frames_matching_current_color": matches_current,
        "slots": resolved["slots"],
        "per_frame": resolved["per_frame"],
    }


def _analyse_input(resolved, spec, app, reference_map, reference_source,
                   trace_relation, configuration=None, current_color=None):
    """Combine alias evidence, frame matching and the trace relation."""
    name = resolved["name"]
    per_frame = resolved["per_frame"]
    kind_available = bool(app and app["by_kind"].get(spec.kind))
    frame_statuses = []
    observed_shifts = set()
    configuration = configuration or {}
    ui_off = (name == "ui" and configuration.get("ui") is False)

    if ui_off:
        return _ui_off_analysis(resolved, current_color, ui_off, reference_map)

    for item in per_frame:
        if name == "lut" or app is None:
            continue
        status = _per_frame_status(item, reference_map, kind_available)
        item["status"] = status
        frame_statuses.append(status)
        for shift in item.get("observed_shifts") or ():
            observed_shifts.add(shift)

    if name == "lut":
        hashes = [item["content_sha256"] for item in per_frame
                  if item.get("content_sha256")]
        stable = len(set(hashes)) == 1 if hashes else False
        expected = resolved.get("expected_lut_sha256")
        matches_expected = None
        if expected is not None and hashes:
            matches_expected = all(value == expected for value in hashes)
        if not hashes:
            status = "unavailable"
        elif not stable:
            status = "mismatch"
        elif matches_expected is False:
            status = "mismatch"
        else:
            status = "confirmed"
        return {
            "kind": resolved["kind"],
            "alias_status": resolved["alias_status"],
            "status": status,
            "frame_shift": None,
            "frame_shift_method": "not_frame_dependent",
            "sha256": hashes[0] if hashes else None,
            "expected_sha256": expected,
            "matches_expected_sha256": matches_expected,
            "stable_across_frames": stable,
            "frames_with_tensor": len(hashes),
            "slots": resolved["slots"],
            "per_frame": per_frame,
        }

    if app is None:
        if any(item["status"] == "unavailable" for item in per_frame):
            return {
                "kind": spec.kind,
                "alias_status": resolved["alias_status"],
                "status": "unavailable",
                "frame_shift": None,
                "frame_shift_method": "tensor snapshot missing",
                "slots": resolved["slots"],
                "per_frame": per_frame,
            }
        if name == "current_color":
            return {
                "kind": spec.kind,
                "alias_status": resolved["alias_status"],
                "status": "confirmed",
                "frame_shift": 0,
                "frame_shift_method": "reference_slot",
                "trace_relation": trace_relation,
                "slots": resolved["slots"],
                "per_frame": per_frame,
            }
        if name == "prev_color":
            checked = trace_relation
            failed = [item for item in checked if not item["match"]]
            if checked and not failed:
                status, shift, method = (
                    "confirmed", -1, "trace_hash_relation")
            elif not checked:
                status, shift, method = (
                    "unresolved", None, "trace has no adjacent frame pair")
            else:
                status, shift, method = ("mismatch", None,
                                         "trace_hash_relation")
            return {
                "kind": spec.kind,
                "alias_status": resolved["alias_status"],
                "status": status,
                "frame_shift": shift,
                "frame_shift_method": method,
                "trace_relation": trace_relation,
                "slots": resolved["slots"],
                "per_frame": per_frame,
            }
        if name == "ui":
            hashes = {item["content_sha256"] for item in per_frame
                      if item.get("content_sha256")}
            if len(hashes) == 1:
                status, reason = "frame_invariant", "ui content identical"
            else:
                status, reason = "unresolved", "ui content varies"
            return {
                "kind": spec.kind,
                "alias_status": resolved["alias_status"],
                "status": status,
                "frame_shift": None,
                "frame_shift_method": reason,
                "slots": resolved["slots"],
                "per_frame": per_frame,
            }
        return {
            "kind": spec.kind,
            "alias_status": resolved["alias_status"],
            "status": "unresolved",
            "frame_shift": None,
            "frame_shift_method": "application inputs required",
            "slots": resolved["slots"],
            "per_frame": per_frame,
        }

    if "mismatch" in frame_statuses:
        status = "mismatch"
    elif frame_statuses and all(value == "unavailable"
                                for value in frame_statuses):
        status = "unavailable"
    elif not kind_available:
        status = "unavailable"
    elif frame_statuses and all(value == "frame_invariant"
                                for value in frame_statuses):
        status = "frame_invariant"
    elif frame_statuses and all(value == "unique"
                                for value in frame_statuses):
        status = "confirmed"
    elif frame_statuses and all(value in ("unique", "frame_invariant")
                                for value in frame_statuses):
        status = "partial"
    elif frame_statuses and all(value == "unmatched"
                                for value in frame_statuses):
        status = "mismatch"
    elif frame_statuses and all(value in ("no_reference", "frame_invariant")
                                for value in frame_statuses):
        status = "unresolved"
    else:
        status = "partial"

    frame_shift = None
    expected = spec.expected_shift
    if status == "confirmed":
        if len(observed_shifts) == 1:
            frame_shift = next(iter(observed_shifts))
            if expected is not None and frame_shift != expected:
                status = "mismatch"
        elif len(observed_shifts) > 1:
            # Unique matches must agree on the temporal offset.
            status = "mismatch"
    if status == "frame_invariant" and expected is None:
        frame_shift = None
    if status == "mismatch":
        frame_shift = None

    return {
        "kind": spec.kind,
        "alias_status": resolved["alias_status"],
        "status": status,
        "frame_shift": frame_shift,
        "hypothesis_frame_shift": expected,
        "observed_frame_shifts": sorted(observed_shifts),
        "frame_shift_method": (
            "application_input_hash" if status in ("confirmed", "mismatch")
            else "content_matches_multiple_frames"
            if status == "frame_invariant" else "unresolved"),
        "reference_source": reference_source,
        "slots": resolved["slots"],
        "per_frame": per_frame,
    }


def _d38_modes(trace):
    per_frame = {}
    for event in trace["events"].values():
        if event.get("node") != 38:
            continue
        for constant in event.get("constants") or ():
            values = constant.get("values") or ()
            if len(values) >= 15:
                mode = _as_int(values[14])
                if mode is not None:
                    per_frame[event["frame"]] = mode
                break
    return {
        "per_frame": {str(frame): value
                      for frame, value in sorted(per_frame.items())},
        "distinct": sorted(set(per_frame.values())),
    }


def _select_boundary_inputs(modes):
    """Select an observed boundary map and warn on mixed or unknown modes."""
    distinct = set(modes.get("distinct") or ())
    if distinct == {4}:
        return MODE4_BOUNDARY_INPUTS, 4, None
    if distinct and 4 in distinct:
        return BOUNDARY_INPUTS, None, (
            "trace mixes d38 mode 4 with %s; using the mode 1/3 boundary map"
            % sorted(distinct - {4}))
    unsupported = distinct - set(SUPPORTED_D38_MODES)
    if unsupported:
        return BOUNDARY_INPUTS, None, (
            "d38 modes %s are not supported; using the mode 1/3 boundary map"
            % sorted(unsupported))
    if distinct and distinct <= set(LEGACY_D38_MODES):
        selected = next(iter(distinct)) if len(distinct) == 1 else None
        return BOUNDARY_INPUTS, selected, None
    # No d38 mode constant: preserve the original mode 1/3 default.
    return BOUNDARY_INPUTS, None, None


def _unused_slot_evidence(trace, tensors_dir, hash_cache, app, reference_map):
    """Describe mode 4's d38.t9 binding without calling it a UI input."""
    spec = InputSpec("unused_d38_t9", "backbuffer", ((38, "t9"),))
    resolved = _resolve_aliases(trace, spec, tensors_dir, hash_cache)
    if app is not None:
        _annotate_application_candidates(resolved, app)
    for item in resolved["per_frame"]:
        reference = reference_map.get(item["frame"])
        if reference is not None:
            item["reference_application_frame"] = reference
    return [{
        "slot": "d38.t9",
        "kind": "backbuffer",
        "reason": (
            "mode 4 does not sample the UI slot; the binding aliases the "
            "current backbuffer bound at d38.t10 and is not a required input"),
        "alias_status": resolved["alias_status"],
        "per_frame": resolved["per_frame"],
    }]


def build_report(trace_path, tensors_dir=None, application_inputs=None,
                 expected_lut_sha256=KNOWN_LUT_SHA256):
    """Build the provenance report for one trace.

    ``tensors_dir`` defaults to ``<trace dir>/tensors``.  ``application_inputs``
    is optional; without it the report still proves the slot aliases, the
    previous/current colour relation and the LUT, and marks inputs that need
    frame matching unresolved.  The d38 mode actually present in the trace
    selects the slot map, so mode 4's t8/t10 boundary does not inherit the
    mode 1/3 UI requirement.
    """
    trace = load_trace(trace_path)
    if tensors_dir is None:
        tensors_dir = os.path.join(os.path.dirname(trace["path"]), "tensors")
    tensors_dir = os.path.abspath(tensors_dir)
    app = (load_application_inputs(application_inputs)
           if application_inputs else None)
    meta, meta_path = _trace_meta(trace_path)

    modes = _d38_modes(trace)
    boundary_inputs, spec_mode, mode_warning = _select_boundary_inputs(modes)

    hash_cache = {}
    resolved_inputs = {}
    for spec in boundary_inputs:
        resolved = _resolve_aliases(trace, spec, tensors_dir, hash_cache)
        if spec.name == "lut":
            resolved["expected_lut_sha256"] = expected_lut_sha256
        resolved_inputs[spec.name] = resolved
        if app is not None and spec.application:
            _annotate_application_candidates(resolved, app)

    reference_map, reference_source = _reference_frame_map(
        resolved_inputs, app)
    trace_relation = _trace_prev_relation(resolved_inputs)

    family = None
    configuration = {}
    if isinstance(meta, dict):
        family = meta.get("xefg_family")
        configuration = meta.get("configuration") or {}

    inputs = {}
    for spec in boundary_inputs:
        inputs[spec.name] = _analyse_input(
            resolved_inputs[spec.name], spec, app, reference_map,
            reference_source, trace_relation,
            configuration=configuration,
            current_color=resolved_inputs["current_color"])
        inputs[spec.name].setdefault(
            "missing_aliases", resolved_inputs[spec.name].get(
                "missing_aliases", []))

    unused_slots = (_unused_slot_evidence(
        trace, tensors_dir, hash_cache, app, reference_map)
        if spec_mode == 4 else [])

    warnings = []
    if mode_warning:
        warnings.append(mode_warning)
    if family and family != "fg78":
        warnings.append(
            "trace family is %s; the slot map is profiled for fg78" % family)
    configured_frames = configuration.get("frames")
    if (isinstance(configured_frames, int)
            and configured_frames != len(trace["frames"])):
        warnings.append(
            "meta configuration.frames=%d but the trace contains %d frames; "
            "frame identity below comes from content matching, not metadata"
            % (configured_frames, len(trace["frames"])))

    mismatch = any(value["status"] == "mismatch"
                   for value in inputs.values())
    unresolved = any(value["status"] in
                     ("unresolved", "unavailable", "partial",
                      "ui_off_unexpected_binding")
                     for value in inputs.values())
    if trace["errors"] or trace["duplicates"]:
        unresolved = True
    if mode_warning:
        unresolved = True
    alias_mismatch = any(value["alias_status"] == "mismatch"
                         for value in inputs.values())
    alias_unavailable = any(value["alias_status"] == "unavailable"
                            for value in inputs.values())
    mismatch = mismatch or alias_mismatch
    if mismatch:
        top_status = "mismatch"
    elif unresolved:
        top_status = "partial"
    else:
        top_status = "confirmed"

    if spec_mode == 4:
        mode_notes = [
            "Mode 4 samples d38.t8 as current hudless colour alongside "
            "d1.t1/d25.t3/d38.t12, and d38.t10 as the current application "
            "backbuffer.",
            "Mode 4 requires no separate UI input; d38.t9 aliases the same "
            "backbuffer as d38.t10 and is recorded as unused evidence, not as "
            "a required UI input.",
        ]
    else:
        mode_notes = [
            ("d38.t9 is not an application UI input when the application UI "
             "is off; in the profiled mode1 capture it aliases current_color."
             if configuration.get("ui") is False else
             "The UI texture is frame-invariant in this scene; its current or "
             "previous identity cannot be distinguished from content."),
        ]

    reported_modes = dict(modes)
    reported_modes.update({
        "spec_mode": spec_mode,
        "boundary_spec": "mode4" if spec_mode == 4 else "mode1_3",
        "supported_modes": list(SUPPORTED_D38_MODES),
    })

    return {
        "schema": SCHEMA,
        "status": top_status,
        "trace": {
            "path": trace["path"],
            "meta_path": meta_path,
            "family": family,
            "frames": trace["frames"],
            "dispatch_count": trace["dispatch_count"],
            "errors": trace["errors"],
            "duplicates": trace["duplicates"],
        },
        "configuration": configuration,
        "d38_mode": reported_modes,
        "boundary": {
            "inputs": inputs,
            "unused_slots": unused_slots,
            "alias_mismatch": alias_mismatch,
            "alias_unavailable": alias_unavailable,
        },
        "frame_identity": {
            "trace_to_application": {str(frame): value
                                     for frame, value in sorted(reference_map.items())},
            "source": reference_source,
            "prev_color_trace_relation": trace_relation,
        },
        "application_inputs": ({
            "root": app["root"],
            "manifest": app["manifest"],
            "frames": app["frames"],
            "kinds": {kind: len(values)
                      for kind, values in app["by_kind"].items()},
            "files": len(app["files"]),
            "missing": app["missing"],
            "errors": app["errors"],
        } if app is not None else None),
        "warnings": warnings,
        "notes": [
            "prev_color/current_color textures are the raw application "
            "hudless-colour frames, evidenced by matching SHA-256 values.",
            "History enters as an explicit previous-frame colour texture: "
            "d1 reads the same raw colours that d25 and d38 receive; no "
            "transformed history buffer was observed at this boundary.",
            "motion and depth SRVs are the raw application buffers when the "
            "application-input hashes match their frame.",
        ] + mode_notes + [
            "The LUT is provider-owned (d25.t6), not an application input.",
        ],
    }


def _print_summary(report, stream):
    print("status: %s" % report["status"], file=stream)
    trace = report["trace"]
    print("frames: %s  dispatches: %s" % (trace["frames"],
                                          trace["dispatch_count"]),
          file=stream)
    modes = report["d38_mode"]
    if modes["distinct"]:
        print("d38 mode: %s" % modes["distinct"], file=stream)
    for name, entry in report["boundary"]["inputs"].items():
        shift = entry.get("frame_shift")
        print("  %-14s status=%-16s shift=%-4s aliases=%s"
              % (name, entry["status"],
                 "none" if shift is None else shift,
                 entry.get("alias_status")), file=stream)
    mapping = report["frame_identity"]["trace_to_application"]
    if mapping:
        print("trace->application frame: %s" % mapping, file=stream)
    for warning in report.get("warnings") or ():
        print("warning: %s" % warning, file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inventory XeFG input aliases, frame shifts and the LUT.")
    parser.add_argument("--trace", required=True,
                        help="trace.jsonl from XefgProbe")
    parser.add_argument("--tensors",
                        help="tensor snapshot directory (default: <trace>/tensors)")
    parser.add_argument("--application-inputs",
                        help="oracle output root, manifest.json, or raw frame dir")
    parser.add_argument("--expected-lut-sha",
                        default=KNOWN_LUT_SHA256,
                        help="expected d25.t6 SHA-256 (default: profiled LUT)")
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero unless the report is confirmed")
    args = parser.parse_args(argv)

    try:
        report = build_report(
            args.trace, tensors_dir=args.tensors,
            application_inputs=args.application_inputs,
            expected_lut_sha256=args.expected_lut_sha,
        )
    except (OSError, ValueError) as exc:
        print("input provenance failed: %s" % exc, file=sys.stderr)
        return 2

    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    _print_summary(report, sys.stderr)

    if report["status"] == "mismatch":
        return 1
    if args.strict and report["status"] != "confirmed":
        return 1
    return 0
