#!/usr/bin/env python
"""Offline hardware sampler diagnostic for the d25 warp + flow-init stage.

This driver prepares and analyses the exact native IO contract agreed with the
native probe owner.  It never edits or writes the production reconstruction and
never writes inside a capture directory.  The d25 replay below is a copy of
``recon.reconstruct_warp_init`` with two hooks (sampling and direct Load); the
copy is checked byte-for-byte against the production function before a bundle
is written.  The production path is used read-only.

Native contract (the frozen helper owns the implementation):

    exe --texture PATH --width W --height H --requests PATH --out NEWDIR
        [--debug] [--adapter text]

``requests`` header: four little-endian uint32
``{magic=0x58534647, version=1, count, stride=32}``.  Each 32-byte record is
``{u32 id, u32 op (0=Sample, 1=Load), f32 u, f32 v, i32 x, i32 y,
u32 reserved0, u32 reserved1}``.  ``samples.bin`` repeats the header with
``stride=80``; each record is ``{u32 id, u32 op, f32 u, f32 v,
f32 normalizedRGBA[4], f32 unitsRGBA[4]=saturate(normalized)*255,
i32 truncatedRGBA[4], u32 padding[4]}``.

The texture is uploaded as raw BGRA8 (DXGI_FORMAT 87) with no CPU swizzle; the
GPU sampler returns RGBA.  Sample UVs are generated exactly as the d25 replay:
``float32(pos_x * float32(1 / width))`` (and the same for y).  The driver
records those float32 bits and the analysis rejects any non-identical echo.

The capture gate selects exactly one d25 dispatch for the requested frame by
shader/resources/shapes, then chooses the root-signature event by that
dispatch's program SHA (not by a reused root-signature pointer).  Its static
sampler must be MIN_MAG_MIP_LINEAR (filter 21), U/V = CLAMP (3), W = WRAP (1).
t2/t3 must be format 87, one mip, one array slice.  The dispatch, key and
program SHA actually selected are stored in the bundle and re-checked during
analysis.  No capture path or frame is built in.

Feature replay uses the hardware-measured normalized/units/truncated arrays,
not a float64 resampling of the texture.  ``units`` is the default because the
d25 source arithmetic is in 0..255 units; ``trunc`` and ``norm255`` are
reported as variants.  Output colour writes are reported as conversion
candidates (half-even, half-up, truncate); the driver does not pretend the
hardware UNORM store is byte-exact and never changes the production algorithm.

Exact CLI
---------

Each invocation handles one capture frame.  ``--run`` is the capture root
containing ``trace/``; ``--frame`` is the frame number.  ``--name`` is an
optional bundle case name.
``--target`` is repeatable and takes ``Y,X,CH``; with ``--region auto`` a run
with targets uses the target neighbourhoods, and a run without targets uses
the full frame.  ``--require-complete`` requires ``trace/meta.json`` to mark
the trace complete.

Prepare a bundle for one target frame::

    python -I -B tools/render-validation/probe_hardware_sampler.py prepare \\
      --run PATH/TO/CAPTURE --frame 8 --name target_case \\
      --out PATH/TO/NEW/BUNDLE \\
      --target 34,51,4 --target 135,211,3 --target 137,65,2

Prepare a full-frame bundle for one control frame::

    python -I -B tools/render-validation/probe_hardware_sampler.py prepare \\
      --run PATH/TO/CAPTURE --frame 1 --name control_case \\
      --out PATH/TO/NEW/BUNDLE --region full

Run the frozen native helper over a new or already prepared bundle.  A new
``--out`` is prepared first and therefore also needs ``--run``/``--frame``;
an existing bundle is executed only (its ``manifest.json`` is required)::

    python -I -B tools/render-validation/probe_hardware_sampler.py run \\
      --out PATH/TO/BUNDLE \\
      --exe PATH/TO/XefgSamplerProbe.exe \\
      --adapter "GPU ADAPTER NAME"

``--adapter`` is passed verbatim to the frozen helper, which matches adapter
name text; omit it to let the helper choose the first device.  ``--debug`` is
forwarded unchanged.

Analyse the produced ``samples.bin`` files, writing ``analysis.json``::

    python -I -B tools/render-validation/probe_hardware_sampler.py analyze \\
      --out PATH/TO/BUNDLE

The native command for each texture is also recorded in the bundle manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from xefg_sdk.export import recon  # noqa: E402

_REQUEST_MAGIC = 0x58534647
_REQUEST_VERSION = 1
_REQUEST_STRIDE = 32
_SAMPLE_MAGIC = 0x58534647
_SAMPLE_VERSION = 1
_SAMPLE_STRIDE = 80

REQUEST_HEADER = struct.Struct("<4I")
SAMPLE_HEADER = struct.Struct("<4I")
REQUEST_DTYPE = np.dtype([
    ("id", "<u4"), ("op", "<u4"), ("u", "<f4"), ("v", "<f4"),
    ("x", "<i4"), ("y", "<i4"), ("reserved0", "<u4"), ("reserved1", "<u4"),
])
SAMPLE_DTYPE = np.dtype([
    ("id", "<u4"), ("op", "<u4"), ("u", "<f4"), ("v", "<f4"),
    ("normalized", "<f4", (4,)), ("units", "<f4", (4,)),
    ("truncated", "<i4", (4,)), ("padding", "<u4", (4,)),
])
assert REQUEST_DTYPE.itemsize == _REQUEST_STRIDE
assert SAMPLE_DTYPE.itemsize == _SAMPLE_STRIDE

OP_SAMPLE = 0
OP_LOAD = 1
SUBS = ((0, 0), (1, 0), (0, 1), (1, 1))

SAMPLER_FILTER_MIN_MAG_MIP_LINEAR = 21
SAMPLER_ADDRESS_CLAMP = 3
SAMPLER_ADDRESS_WRAP = 1

PACKING_CANDIDATES = ("half_up", "half_even", "trunc")
FEATURE_MODES = ("units", "trunc", "norm255")


class ProbeError(ValueError):
    """The bundle, capture or packet is not the approved diagnostic input."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _repo_head():
    head = (REPO / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = REPO / ".git" / head[5:].strip()
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
    return head


def _json_dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_case_name(name):
    invalid = set('\\/:*?"<>|')
    if (not name or name in (".", "..")
            or any(char in invalid or ord(char) < 32 for char in name)):
        raise ProbeError("invalid case name %r" % (name,))
    return name


def _parse_target(text):
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target must be Y,X,CH")
    try:
        y, x, ch = (int(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("target must be three integers: Y,X,CH")
    if y < 0 or x < 0 or not 0 <= ch < 16:
        raise argparse.ArgumentTypeError("target must be Y>=0, X>=0, 0<=CH<16")
    return (y, x, ch)


def _case_spec(run, frame, name=None, targets=(), require_complete=False):
    """Build one explicit case spec; no capture path or target is built in."""
    frame = int(frame)
    if frame < 0:
        raise ProbeError("frame must be non-negative")
    if name is None:
        name = "%s_frame%d" % (Path(run).name or "case", frame)
    _validate_case_name(name)
    normalized = []
    for target in targets:
        y, x, ch = (int(value) for value in target)
        if y < 0 or x < 0 or not 0 <= ch < 16:
            raise ProbeError("target %s must be Y>=0, X>=0, 0<=CH<16"
                             % ((y, x, ch),))
        normalized.append((y, x, ch))
    targets = tuple(normalized)
    return {
        "name": name,
        "run": Path(run),
        "frame": frame,
        "targets": targets,
        "default_region": "targets" if targets else "full",
        "require_complete": bool(require_complete),
    }


def _spec_from_case(case):
    """Rebuild the gate spec for one manifest case during analysis."""
    gate = case.get("gate") or {}
    expected = {field: gate[field] for field in ("dispatch", "key", "sha256")
                if gate.get(field) is not None}
    return {
        "name": case["name"],
        "run": Path(case["run"]),
        "frame": int(case["frame"]),
        "targets": tuple(tuple(int(value) for value in item)
                         for item in case.get("targets", ())),
        "require_complete": bool(case.get("require_complete", False)),
        "expect": expected,
    }


def _load_compare():
    """Load the retained compare helper without assuming cwd is the repo."""
    path = Path(__file__).resolve().with_name("compare_tensors.py")
    spec = importlib.util.spec_from_file_location("_probe_compare_tensors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_capture(spec):
    """Load one case and enforce the selected d25 program/sampler gate.

    The root signature is looked up by the program SHA.  The trace reuses one
    root-signature pointer for several programs, so a pointer-only match can
    pick the wrong sampler; that is the error this gate prevents.
    """
    run = Path(spec["run"])
    compare = _load_compare()
    try:
        rows = recon.read_events(str(run))
    except OSError as error:
        raise ProbeError("cannot read trace for %s: %s" % (run, error))
    meta_path = run / "trace" / "meta.json"
    meta = compare.read_json(meta_path) if meta_path.is_file() else {}
    compare.validate_trace_meta(meta, spec["frame"],
                                require_complete=spec.get("require_complete", False))
    dispatch = compare.select_stage_dispatch(rows, "d25", spec["frame"])
    expected = spec.get("expect") or {}
    for field in ("dispatch", "key", "sha256"):
        if field not in expected:
            continue
        actual = dispatch.get(field)
        wanted = expected[field]
        if field == "dispatch":
            actual, wanted = int(actual), int(wanted)
        if actual != wanted:
            raise ProbeError("%s: selected d25 %s %r != expected %r"
                             % (spec["name"], field, actual, wanted))
    program_sha = dispatch.get("sha256")
    if not program_sha:
        raise ProbeError("%s: selected d25 dispatch has no program SHA"
                         % spec["name"])

    resources = {entry.get("slot"): entry for entry in dispatch.get("resources", [])}
    for slot in ("t2", "t3"):
        entry = resources.get(slot)
        if entry is None:
            raise ProbeError("%s: missing %s" % (spec["name"], slot))
        if entry.get("dimension") != "texture2d" or int(entry.get("format")) != 87:
            raise ProbeError("%s: %s is not BGRA8 (format 87): %s"
                             % (spec["name"], slot, entry))
        if int(entry.get("array", 1)) != 1 or int(entry.get("mips", 1)) != 1:
            raise ProbeError("%s: %s is not single-mip/single-slice" % (spec["name"], slot))
        if int(entry.get("first_mip", 0)) != 0 or int(entry.get("first_slice", 0)) != 0:
            raise ProbeError("%s: %s starts at a mip/slice offset" % (spec["name"], slot))

    roots = [row for row in rows
             if row.get("type") == "root_signature"
             and row.get("sha256") == program_sha]
    same_pointer = [row for row in roots
                    if row.get("root_signature") == dispatch.get("root_signature")]
    if same_pointer:
        roots = same_pointer
    if not roots:
        raise ProbeError("%s: no root signature with program SHA %s"
                         % (spec["name"], program_sha))
    # A reused pointer can have several SHAs; by construction every remaining
    # entry matches the program SHA.  Refuse contradictory sampler layouts.
    layouts = {json.dumps(row.get("static_samplers", []), sort_keys=True)
               for row in roots}
    if len(layouts) != 1:
        raise ProbeError("%s: ambiguous root signature for program SHA" % spec["name"])
    root = roots[0]
    samplers = [sampler for sampler in root.get("static_samplers", [])
                if int(sampler.get("register", -1)) == 0
                and int(sampler.get("space", -1)) == 0]
    if len(samplers) != 1:
        raise ProbeError("%s: expected one static sampler at s0" % spec["name"])
    sampler = samplers[0]
    if int(sampler.get("filter", -1)) != SAMPLER_FILTER_MIN_MAG_MIP_LINEAR:
        raise ProbeError("%s: sampler filter %s is not MIN_MAG_MIP_LINEAR (21)"
                         % (spec["name"], sampler.get("filter")))
    if int(sampler.get("address_u", -1)) != SAMPLER_ADDRESS_CLAMP:
        raise ProbeError("%s: sampler AddressU is not CLAMP" % spec["name"])
    if int(sampler.get("address_v", -1)) != SAMPLER_ADDRESS_CLAMP:
        raise ProbeError("%s: sampler AddressV is not CLAMP" % spec["name"])
    if int(sampler.get("address_w", -1)) != SAMPLER_ADDRESS_WRAP:
        raise ProbeError("%s: sampler AddressW is not WRAP" % spec["name"])

    data, error = recon._warp_init_inputs(str(run), dispatch)
    if error:
        raise ProbeError("%s: %s" % (spec["name"], error))
    gate = {
        "run": str(run),
        "frame": spec["frame"],
        "dispatch": int(dispatch.get("dispatch")),
        "key": dispatch.get("key"),
        "sha256": dispatch.get("sha256"),
        "root_signature": dispatch.get("root_signature"),
        "root_signature_program_sha256": root.get("sha256"),
        "root_signature_sha256": root.get("sha256"),
        "static_sampler": sampler,
        "sampler_filter_name": "MIN_MAG_MIP_LINEAR",
        "t2": {key: resources["t2"].get(key) for key in
               ("format", "width", "height", "array", "mips", "first_mip", "first_slice")},
        "t3": {key: resources["t3"].get(key) for key in
               ("format", "width", "height", "array", "mips", "first_mip", "first_slice")},
        "trace_meta": meta,
    }
    return rows, dispatch, data, gate


# ---------------------------------------------------------------------------
# d25 replay copied from recon.reconstruct_warp_init.
#
# Provenance: src/xefg_sdk/export/recon.py at the repository revision recorded
# in the bundle manifest.  Only the four sample calls and the two direct self
# loads are routed through hooks; every coordinate, clamp, feature slot,
# integer GEMV term and truncation is the production expression.  The exact
# hooks must reproduce recon.reconstruct_warp_init byte-for-byte, and prepare
# verifies that before writing anything.
# ---------------------------------------------------------------------------
def _replay_d25(data, sample, load, finalize=True):
    c = data["c"]
    tw, th, ow, oh = data["tw"], data["th"], data["ow"], data["oh"]
    plane_w, plane_h, stride = data["plane_w"], data["plane_h"], data["stride"]
    t0, t2, t3 = data["t0"], data["t2"], data["t3"]
    t4, t5, t6 = data["t4"], data["t5"], data["t6"]
    plane0, plane1 = data["plane0"], data["plane1"]
    weights, scale, bias = data["weights"], data["scale"], data["bias"]

    tx = np.broadcast_to(np.arange(tw, dtype=np.int32)[None, :], (th, tw))
    ty = np.broadcast_to(np.arange(th, dtype=np.int32)[:, None], (th, tw))

    def depth(x, y):
        x = np.clip(x, 0, tw - 1)
        y = np.clip(y, 0, th - 1)
        word = plane0[y, x]
        packed = (np.float32(1.0)
                  - ((word >> np.uint32(16)) & np.uint32(0x7FFF)).astype(np.float32)
                  / np.float32(32768.0))
        return np.where(word == 0, t5[y, x] - np.float32(1000.0), packed)

    A = depth(tx - 1, ty - 1)
    B = depth(tx + 1, ty - 1)
    C = depth(tx - 1, ty + 1)
    D = depth(tx + 1, ty + 1)
    centre = depth(tx, ty)
    row_sel = np.maximum(-A, -B) > np.maximum(-C, -D)
    dx = np.where(row_sel, np.where(A < B, -1, 1), np.where(C < D, -1, 1))
    dy = np.where(row_sel, -1, 1)
    keep = np.maximum(np.maximum(-A, -B), np.maximum(-C, -D)) > -centre
    xp = np.clip(tx + np.where(keep, dx, 0), 0, tw - 1)
    yp = np.clip(ty + np.where(keep, dy, 0), 0, th - 1)
    word0 = plane0[yp, xp]
    word1 = plane1[yp, xp]
    f0 = recon._half_to_f32((word0 & np.uint32(0xFFFF)).astype("<u2"))
    f1 = np.where(word1 == 0, np.float32(0.0),
                  recon._half_to_f32((word1 & np.uint32(0xFFFF)).astype("<u2")))
    f0 = f0.astype(np.float32)
    f1 = f1.astype(np.float32)
    fallback_flow = t4[yp, xp]

    span_x = np.float32(c[36] - 1) / np.float32(c[38] - 1)
    span_y = np.float32(c[37] - 1) / np.float32(c[39] - 1)
    uv_a = (np.float32(recon._splat_f32(c[42])),
            np.float32(recon._splat_f32(c[43])))
    motion_scale = np.float32(recon._splat_f32(c[47]))
    ratio = np.float32(recon._splat_f32(c[50])) / (
        np.float32(1.0) - np.float32(recon._splat_f32(c[50])))
    lo_a = (np.float32(c[0]) + np.float32(0.5001),
            np.float32(c[1]) + np.float32(0.5001))
    hi_a = (np.float32(c[0] + c[2]) - np.float32(0.5001),
            np.float32(c[1] + c[3]) - np.float32(0.5001))
    lo_b = (np.float32(c[4]) + np.float32(0.5001),
            np.float32(c[5]) + np.float32(0.5001))
    hi_b = (np.float32(c[4] + c[6]) - np.float32(0.5001),
            np.float32(c[5] + c[7]) - np.float32(0.5001))

    def t0_bilinear(x, y):
        return recon._warp_init_motion(t0, x, y, span_x, span_y, plane_w,
                                       plane_h, int(c[38]), int(c[39]))

    features = []
    pack_lanes = {}
    for ax, ay in SUBS:
        px = (2 * tx + ax).astype(np.float32) + np.float32(0.5)
        py = (2 * ty + ay).astype(np.float32) + np.float32(0.5)
        motion = t0_bilinear(2 * tx + ax, 2 * ty + ay)
        pos_a = (np.clip(recon._warp_init_add_scaled(px, motion_scale,
                                                     motion[..., 0], "a"),
                         lo_a[0], hi_a[0]),
                 np.clip(recon._warp_init_add_scaled(py, motion_scale,
                                                     motion[..., 1], "a"),
                         lo_a[1], hi_a[1]))
        pos_b = (np.clip(recon._warp_init_add_scaled(px, motion_scale,
                                                     motion[..., 2], "b"),
                         lo_b[0], hi_b[0]),
                 np.clip(recon._warp_init_add_scaled(py, motion_scale,
                                                     motion[..., 3], "b"),
                         lo_b[1], hi_b[1]))

        rx = np.float32(2.0) * px * uv_a[0] - np.float32(1.0)
        ry = np.float32(2.0) * py * uv_a[1] - np.float32(1.0)
        radius = np.maximum(np.abs(rx), np.abs(ry))
        ramp = np.clip(np.float32(5.0) * (radius - np.float32(0.8))
                       + np.float32(0.25), 0, 1)
        shrink = np.float32(0.099) * ramp
        qx = px - np.float32(2.0) * f0
        qy = py - np.float32(2.0) * f1
        moved = np.maximum(np.abs(np.float32(2.0) * qx * uv_a[0] - np.float32(1.0)),
                           np.abs(np.float32(2.0) * qy * uv_a[1] - np.float32(1.0)))
        confidence = np.float32(1.0) - np.clip(
            (moved - np.float32(0.9) - shrink) / (np.float32(0.1) - shrink), 0, 1)
        gate = t6[py.astype(np.int32) & 31, px.astype(np.int32) & 31]
        fallback = recon._warp_init_fallback(word0, confidence < gate)
        fx = np.where(fallback, np.float32(0.5) * fallback_flow[..., 0], f0)
        fy = np.where(fallback, np.float32(0.5) * fallback_flow[..., 1], f1)
        two_ratio = np.float32(2.0) * ratio
        pos_c = (np.clip(recon._warp_init_add_scaled(px, two_ratio, fx, "c"),
                         lo_a[0], hi_a[0]),
                 np.clip(recon._warp_init_add_scaled(py, two_ratio, fy, "c"),
                         lo_a[1], hi_a[1]))
        pos_d = (np.clip(recon._warp_init_add_scaled(px, np.float32(-2.0), fx, "d"),
                         lo_b[0], hi_b[0]),
                 np.clip(recon._warp_init_add_scaled(py, np.float32(-2.0), fy, "d"),
                         lo_b[1], hi_b[1]))

        centre2 = sample("t2", pos_c[0], pos_c[1], "centre2", ax, ay)
        centre3 = sample("t3", pos_d[0], pos_d[1], "centre3", ax, ay)
        warp2 = np.clip(sample("t2", pos_a[0], pos_a[1], "warp2", ax, ay), 0, 255)
        warp3 = np.clip(sample("t3", pos_b[0], pos_b[1], "warp3", ax, ay), 0, 255)
        xi = px.astype(np.int32)
        yi = py.astype(np.int32)
        self2 = load("t2", xi, yi, "self2", ax, ay)
        self3 = load("t3", xi, yi, "self3", ax, ay)
        flag = np.where(fallback, np.float32(0.0), np.float32(1.0)).astype(np.float32)
        features.append(np.concatenate([
            centre2[..., :3], centre3[..., :3], warp2[..., :3], warp3[..., :3],
            flag[..., None], self2[..., :3], self3[..., :3]], axis=-1))

        one = np.stack([warp2[..., 0], warp2[..., 1], warp2[..., 2],
                        centre3[..., 0]], axis=-1)
        two = np.stack([warp3[..., 0], warp3[..., 1], warp3[..., 2],
                        centre3[..., 1]], axis=-1)
        three = np.stack([centre2[..., 0], centre2[..., 1], centre2[..., 2],
                          centre3[..., 2]], axis=-1)
        pack_lanes[(ax, ay)] = (one, two, three)

    if not finalize:
        return {"pack": pack_lanes}

    quantised = np.zeros((19, 4, th, tw), dtype=np.int64)
    units_lanes = []
    for lane, feature in enumerate(features):
        units = feature.copy()
        units[..., 12] *= np.float32(255.0)
        if int(c[48]) == 1:
            for slot in range(19):
                if slot != 12:
                    units[..., slot] = recon._warp_init_curve(units[..., slot])
        units_lanes.append(units.astype(np.float64).copy())
        quantised[:, lane] = np.moveaxis(
            np.clip(np.trunc(units) - 128, -128, 127).astype(np.int64), -1, 0)

    got0 = np.zeros((th, tw, 16), dtype=np.uint8)
    accs = np.zeros((16, th, tw), dtype=np.int64)
    values = np.zeros((16, th, tw), dtype=np.float64)
    for channel in range(16):
        acc = np.zeros((th, tw), dtype=np.int64)
        for group in range(19):
            row = 4 * group + (channel >> 2)
            weight = weights[row, (channel & 3) * 4:(channel & 3) * 4 + 4].astype(np.int64)
            acc += (quantised[group] * weight[:, None, None]).sum(axis=0)
        row = 76 + (channel >> 2)
        weight = weights[row, (channel & 3) * 4:(channel & 3) * 4 + 4].astype(np.int64)
        acc += np.int64(-128) * weight.sum()
        value = scale[channel] * acc.astype(np.float32) + bias[channel]
        value = value.astype(np.float32)
        got0[..., channel] = np.clip(np.trunc(value.astype(np.float32)),
                                     0, 127).astype(np.uint8)
        accs[channel] = acc
        values[channel] = value.astype(np.float64)

    return {
        "got0": got0, "pack": pack_lanes, "quantised": quantised,
        "accs": accs, "values": values, "units_lanes": units_lanes,
        "features": features,
    }


def _exact_sample_hook(data):
    def sample(tex, pos_x, pos_y, role, ax, ay):
        del role, ax, ay
        return recon._warp_init_sample(data[tex], pos_x, pos_y)
    return sample


def _direct_load(tex, x, y):
    """Direct Load in the shader's RGBA order, from a BGRA8 texture."""
    return tex[y, x][..., [2, 1, 0, 3]].astype(np.float32)


def _exact_load_hook(data):
    def load(tex, x, y, role, ax, ay):
        del role, ax, ay
        return _direct_load(data[tex], x, y)
    return load


def _direct_texture_load(tex, x, y):
    """Drop-in replacement used by callers that only have texture arrays."""
    return _direct_load(tex, x, y)


def _unorm_candidate(units, candidate):
    values = np.asarray(units, dtype=np.float32)
    if candidate == "half_even":
        rounded = np.rint(values.astype(np.float64))
    elif candidate == "half_up":
        rounded = np.floor(values.astype(np.float64) + 0.5)
    elif candidate == "trunc":
        rounded = np.floor(values.astype(np.float64))
    else:
        raise ProbeError("unknown UNORM candidate %s" % candidate)
    return np.clip(rounded, 0, 255).astype(np.uint8)


def _pack_outputs(pack_lanes, th, tw, oh, ow, candidate):
    got1 = np.zeros((oh, ow, 4), dtype=np.uint8)
    got2 = np.zeros((oh, ow, 4), dtype=np.uint8)
    got3 = np.zeros((oh, ow, 4), dtype=np.uint8)
    for (ax, ay), (one, two, three) in pack_lanes.items():
        got1[ay::2, ax::2] = _unorm_candidate(one, candidate)[..., [2, 1, 0, 3]]
        got2[ay::2, ax::2] = _unorm_candidate(two, candidate)[..., [2, 1, 0, 3]]
        got3[ay::2, ax::2] = _unorm_candidate(three, candidate)[..., [2, 1, 0, 3]]
    return {"u1": got1, "u2": got2, "u3": got3}


# ---------------------------------------------------------------------------
# Call tracing, request packetisation and strict packet readers.
# ---------------------------------------------------------------------------
def _collect_calls(data):
    """Run the copied d25 arithmetic with recording sample/load hooks."""
    tables = {"t2": [], "t3": []}
    zeros = np.zeros((data["th"], data["tw"], 4), dtype=np.float64)

    def sample(tex, pos_x, pos_y, role, ax, ay):
        tables[tex].append({"op": OP_SAMPLE, "role": role, "ax": ax, "ay": ay,
                            "pos_x": pos_x, "pos_y": pos_y})
        return zeros

    def load(tex, x, y, role, ax, ay):
        tables[tex].append({"op": OP_LOAD, "role": role, "ax": ax, "ay": ay,
                            "x": x, "y": y})
        return zeros

    _replay_d25(data, sample, load, finalize=False)
    return tables


def _region_mask(data, spec, region, radius):
    th, tw = data["th"], data["tw"]
    if region == "full":
        return np.ones((th, tw), dtype=bool)
    if region != "targets":
        raise ProbeError("unknown region %s" % region)
    targets = spec.get("targets") or ()
    if not targets:
        raise ProbeError("%s has no local targets" % spec["name"])
    mask = np.zeros((th, tw), dtype=bool)
    for y, x, _ch in targets:
        if not (0 <= y < th and 0 <= x < tw):
            raise ProbeError("target %s outside %dx%d" % ((y, x), tw, th))
        y0, y1 = max(0, y - radius), min(th - 1, y + radius)
        x0, x1 = max(0, x - radius), min(tw - 1, x + radius)
        mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def _uv_float32(pos, size):
    reciprocal = np.float32(1.0) / np.float32(size)
    return (np.asarray(pos, dtype=np.float32) * reciprocal).astype(np.float32)


def _pack_requests(tables, data, mask):
    oh, ow = data["oh"], data["ow"]
    cells_y, cells_x = np.nonzero(mask)
    count_cells = int(cells_y.size)
    packets = {}
    for tex in ("t2", "t3"):
        width, height = ow, oh
        records = []
        spans = []
        next_id = 0
        for entry in tables[tex]:
            if entry["op"] == OP_SAMPLE:
                pos_x = entry["pos_x"][cells_y, cells_x]
                pos_y = entry["pos_y"][cells_y, cells_x]
                u = _uv_float32(pos_x, width)
                v = _uv_float32(pos_y, height)
                x = np.zeros(count_cells, dtype=np.int32)
                y = np.zeros(count_cells, dtype=np.int32)
            else:
                u = np.zeros(count_cells, dtype=np.float32)
                v = np.zeros(count_cells, dtype=np.float32)
                x = entry["x"][cells_y, cells_x].astype(np.int32)
                y = entry["y"][cells_y, cells_x].astype(np.int32)
            block = np.empty(count_cells, dtype=REQUEST_DTYPE)
            ids = np.arange(next_id, next_id + count_cells, dtype=np.uint32)
            block["id"] = ids
            block["op"] = entry["op"]
            block["u"] = u
            block["v"] = v
            block["x"] = x
            block["y"] = y
            block["reserved0"] = 0
            block["reserved1"] = 0
            records.append(block)
            spans.append({
                "tex": tex, "op": entry["op"], "role": entry["role"],
                "ax": entry["ax"], "ay": entry["ay"],
                "start": int(next_id), "count": int(count_cells),
            })
            next_id += count_cells
        packets[tex] = {
            "records": np.concatenate(records) if records else np.empty(0, REQUEST_DTYPE),
            "spans": spans,
            "width": width,
            "height": height,
            "cells_y": cells_y,
            "cells_x": cells_x,
            "mask": mask,
        }
    return packets


def request_bytes(records):
    records = np.ascontiguousarray(records, dtype=REQUEST_DTYPE)
    return REQUEST_HEADER.pack(_REQUEST_MAGIC, _REQUEST_VERSION, len(records),
                               _REQUEST_STRIDE) + records.tobytes()


def sample_bytes(records):
    records = np.ascontiguousarray(records, dtype=SAMPLE_DTYPE)
    return SAMPLE_HEADER.pack(_SAMPLE_MAGIC, _SAMPLE_VERSION, len(records),
                              _SAMPLE_STRIDE) + records.tobytes()


def read_requests(path):
    return read_requests_bytes(Path(path).read_bytes(), str(path))


def read_requests_bytes(raw, label="<requests>"):
    if len(raw) < REQUEST_HEADER.size:
        raise ProbeError("%s: truncated request header" % label)
    magic, version, count, stride = REQUEST_HEADER.unpack_from(raw, 0)
    if magic != _REQUEST_MAGIC:
        raise ProbeError("%s: request magic 0x%08x" % (label, magic))
    if version != _REQUEST_VERSION:
        raise ProbeError("%s: request version %s" % (label, version))
    if stride != _REQUEST_STRIDE:
        raise ProbeError("%s: request stride %s" % (label, stride))
    expected = REQUEST_HEADER.size + int(count) * _REQUEST_STRIDE
    if len(raw) != expected:
        raise ProbeError("%s: request length %d != %d for count %d"
                         % (label, len(raw), expected, count))
    return np.frombuffer(raw, dtype=REQUEST_DTYPE, count=count,
                         offset=REQUEST_HEADER.size).copy()


def read_samples(path, expected_count=None):
    return read_samples_bytes(Path(path).read_bytes(), str(path), expected_count)


def read_samples_bytes(raw, label="<samples>", expected_count=None):
    if len(raw) < SAMPLE_HEADER.size:
        raise ProbeError("%s: truncated sample header" % label)
    magic, version, count, stride = SAMPLE_HEADER.unpack_from(raw, 0)
    if magic != _SAMPLE_MAGIC:
        raise ProbeError("%s: sample magic 0x%08x" % (label, magic))
    if version != _SAMPLE_VERSION:
        raise ProbeError("%s: sample version %s" % (label, version))
    if stride != _SAMPLE_STRIDE:
        raise ProbeError("%s: sample stride %s" % (label, stride))
    expected = SAMPLE_HEADER.size + int(count) * _SAMPLE_STRIDE
    if len(raw) != expected:
        raise ProbeError("%s: sample length %d != %d for count %d"
                         % (label, len(raw), expected, count))
    if expected_count is not None and int(count) != int(expected_count):
        raise ProbeError("%s: sample count %s != expected %s"
                         % (label, count, expected_count))
    records = np.frombuffer(raw, dtype=SAMPLE_DTYPE, count=count,
                            offset=SAMPLE_HEADER.size).copy()
    order = np.argsort(records["id"], kind="stable")
    records = records[order]
    ids = records["id"]
    if ids.size and (ids[0] != 0 or not np.array_equal(
            ids, np.arange(ids.size, dtype=np.uint32))):
        raise ProbeError("%s: sample ids are not exactly 0..count-1" % label)
    return records


def _assert_echo(expected, actual, label):
    if len(expected) != len(actual):
        raise ProbeError("%s: echo count %d != %d"
                         % (label, len(actual), len(expected)))
    for field in ("id", "op", "u", "v"):
        left = np.ascontiguousarray(expected[field])
        right = np.ascontiguousarray(actual[field])
        if field in ("u", "v"):
            same = np.array_equal(left.view("<u4"), right.view("<u4"))
        else:
            same = np.array_equal(left, right)
        if not same:
            index = int(np.nonzero(left.view("<u4") != right.view("<u4"))[0][0]) \
                if field in ("u", "v") else int(np.nonzero(left != right)[0][0])
            raise ProbeError("%s: %s echo differs at record %d (%r vs %r)"
                             % (label, field, index, left[index], right[index]))


def _write_case_files(case_dir, run, dispatch, data):
    files = {}
    for slot in ("t2", "t3"):
        raw, resource = recon.tensor(str(run), dispatch, slot)
        if raw is None:
            raise ProbeError("%s: missing tensor bytes" % slot)
        width = int(resource["width"])
        height = int(resource["height"])
        if len(raw) != width * height * 4:
            raise ProbeError("%s: %d bytes is not %dx%d BGRA8"
                             % (slot, len(raw), width, height))
        # Byte-for-byte upload: no CPU channel swizzle.  The GPU sampler
        # returns RGBA from this BGRA8 (format 87) resource.
        target = case_dir / ("texture_%s.bin" % slot)
        target.write_bytes(raw.tobytes())
        files["texture_%s" % slot] = {
            "file": target.name,
            "width": width,
            "height": height,
            "format": int(resource["format"]),
            "bytes": len(raw),
            "sha256": sha256_file(target),
        }
    return files


def _official_arrays(data):
    th, tw, oh, ow = data["th"], data["tw"], data["oh"], data["ow"]
    return {
        "u0": data["u0"].reshape(th, tw, 16),
        "u1": data["u1"].reshape(oh, ow, 4),
        "u2": data["u2"].reshape(oh, ow, 4),
        "u3": data["u3"].reshape(oh, ow, 4),
    }


def _output_mask(mask, oh, ow):
    output = np.zeros((oh, ow), dtype=bool)
    for ax, ay in SUBS:
        output[ay::2, ax::2] = mask
    return output


def _counts_against(got, official, candidate_mask):
    delta = got.astype(np.int16) - official.astype(np.int16)
    selected = delta[candidate_mask]
    return {
        "mismatch": int((selected != 0).sum()),
        "max_abs_diff": int(np.abs(selected).max()) if selected.size else 0,
    }


def _case_counts(result, packs, official, mask, oh, ow):
    out_mask = _output_mask(mask, oh, ow)
    return {
        "u0": _counts_against(result["got0"], official["u0"], mask),
        "u1": _counts_against(packs["u1"], official["u1"], out_mask),
        "u2": _counts_against(packs["u2"], official["u2"], out_mask),
        "u3": _counts_against(packs["u3"], official["u3"], out_mask),
    }


def _flatten_counts(counts):
    return {key: counts[key]["mismatch"] for key in ("u0", "u1", "u2", "u3")}


def _prepare_case(spec, case_dir, region, radius):
    rows, dispatch, data, gate = _gate_capture(spec)
    del rows
    th, tw, oh, ow = data["th"], data["tw"], data["oh"], data["ow"]
    mask = _region_mask(data, spec, region, radius)
    cells = int(mask.sum())

    production = {}
    production_result, error = recon.reconstruct_warp_init(
        str(spec["run"]), dispatch, capture=production)
    if error:
        raise ProbeError("%s: production replay failed: %s" % (spec["name"], error))
    exact = _replay_d25(data, _exact_sample_hook(data), _exact_load_hook(data))
    exact_packs = _pack_outputs(exact["pack"], th, tw, oh, ow, "half_up")
    if not np.array_equal(exact["got0"], production["u0"]):
        raise ProbeError("%s: copied d25 u0 does not match production" % spec["name"])
    for name in ("u1", "u2", "u3"):
        if not np.array_equal(exact_packs[name], production[name]):
            raise ProbeError("%s: copied d25 %s does not match production"
                             % (spec["name"], name))

    official = _official_arrays(data)
    baseline_counts = _case_counts(exact, exact_packs, official, mask, oh, ow)

    tables = _collect_calls(data)
    packets = _pack_requests(tables, data, mask)
    files = _write_case_files(case_dir, spec["run"], dispatch, data)
    request_manifest = {}
    for tex in ("t2", "t3"):
        packet = packets[tex]
        raw = request_bytes(packet["records"])
        path = case_dir / ("requests_%s.bin" % tex)
        path.write_bytes(raw)
        samples = sum(span["count"] for span in packet["spans"] if span["op"] == OP_SAMPLE)
        loads = sum(span["count"] for span in packet["spans"] if span["op"] == OP_LOAD)
        request_manifest[tex] = {
            "file": path.name,
            "sha256": sha256_bytes(raw),
            "count": int(len(packet["records"])),
            "samples": int(samples),
            "loads": int(loads),
            "stride": _REQUEST_STRIDE,
            "width": packet["width"],
            "height": packet["height"],
            "spans": packet["spans"],
        }
        native_dir = case_dir / ("native_%s" % tex)
        request_manifest[tex]["native_out"] = native_dir.name
        request_manifest[tex]["native_samples"] = str(native_dir / "samples.bin")

    provenance = {
        "trace_sha256": sha256_file(Path(spec["run"]) / "trace" / "trace.jsonl"),
        "t2_sha256": files["texture_t2"]["sha256"],
        "t3_sha256": files["texture_t3"]["sha256"],
    }
    return {
        "name": spec["name"],
        "run": str(spec["run"]),
        "frame": spec["frame"],
        "targets": [list(item) for item in spec.get("targets", ())],
        "require_complete": bool(spec.get("require_complete", False)),
        "region": {"mode": region, "radius": int(radius), "cells": cells,
                   "shape_th_tw": [th, tw]},
        "shapes": {"t2": [oh, ow, 4], "t3": [oh, ow, 4],
                   "u0": [th, tw, 16], "u1_u2_u3": [oh, ow, 4]},
        "gate": gate,
        "provenance": provenance,
        "files": files,
        "requests": request_manifest,
        "uv_formula": "float32(pos * float32(1.0 / size))",
        "baseline": {
            "production_note": production_result[3],
            "production_counts": list(production_result[0])
            if isinstance(production_result[0], (list, tuple)) else None,
            "copied_replay_matches_production": True,
            "official_counts": _flatten_counts(baseline_counts),
            "official_max_abs_diff": {key: baseline_counts[key]["max_abs_diff"]
                                      for key in ("u0", "u1", "u2", "u3")},
        },
    }


def prepare(out, spec, region="auto", radius=2):
    _validate_case_name(spec["name"])
    chosen = spec["default_region"] if region == "auto" else region
    if chosen == "targets" and not spec.get("targets"):
        raise ProbeError("%s has no local targets; use --region full"
                         % spec["name"])
    out = Path(out)
    if out.exists():
        raise ProbeError("--out must be a new directory: %s" % out)
    out.mkdir(parents=True)
    manifest = {
        "schema": "xefg.hardware_sampler/1",
        "repo_head": _repo_head(),
        "recon_sha256": sha256_file(REPO / "src" / "xefg_sdk" / "export" / "recon.py"),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "contract": {
            "request_magic": _REQUEST_MAGIC,
            "request_version": _REQUEST_VERSION,
            "request_stride": _REQUEST_STRIDE,
            "sample_magic": _SAMPLE_MAGIC,
            "sample_version": _SAMPLE_VERSION,
            "sample_stride": _SAMPLE_STRIDE,
            "texture_format": 87,
            "texture_upload": "raw BGRA8, no CPU swizzle; GPU sample returns RGBA",
            "uv": "float32(pos * float32(1.0 / size))",
            "targets": [list(item) for item in spec.get("targets", ())],
        },
        "cases": [],
    }
    case_dir = out / spec["name"]
    case_dir.mkdir()
    case_manifest = _prepare_case(spec, case_dir, chosen, radius)
    manifest["cases"].append(case_manifest)
    print("prepared %s: %s cells, t2 %d requests, t3 %d requests"
          % (spec["name"], case_manifest["region"]["cells"],
             case_manifest["requests"]["t2"]["count"],
             case_manifest["requests"]["t3"]["count"]), flush=True)
    _json_dump(out / "manifest.json", manifest)
    return manifest


def execute_native(out, exe, debug=False, adapter=None):
    out = Path(out)
    manifest = _read_json(out / "manifest.json")
    exe = Path(exe)
    if not exe.is_file():
        raise ProbeError("native helper not found: %s" % exe)
    for case in manifest["cases"]:
        case_dir = out / case["name"]
        for tex in ("t2", "t3"):
            request = case["requests"][tex]
            native_dir = case_dir / request["native_out"]
            if native_dir.exists():
                raise ProbeError("native output already exists: %s" % native_dir)
            command = [
                str(exe),
                "--texture", str(case_dir / case["files"]["texture_%s" % tex]["file"]),
                "--width", str(request["width"]),
                "--height", str(request["height"]),
                "--requests", str(case_dir / request["file"]),
                "--out", str(native_dir),
            ]
            if debug:
                command.append("--debug")
            if adapter is not None:
                command.extend(["--adapter", str(adapter)])
            print("run:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            samples = native_dir / "samples.bin"
            if not samples.is_file():
                raise ProbeError("native helper did not write %s" % samples)
            records = read_samples(samples, expected_count=request["count"])
            print("  %s: %d sample records" % (samples, len(records)), flush=True)


def _build_lookup(records, packets, data, mode, want_raw=False):
    th, tw = data["th"], data["tw"]
    mode_lookup = {}
    raw_lookup = {}
    for tex in ("t2", "t3"):
        packet = packets[tex]
        cells_y, cells_x = packet["cells_y"], packet["cells_x"]
        for span in packet["spans"]:
            start = span["start"]
            count = span["count"]
            block = records[tex][start:start + count]
            if not np.array_equal(block["id"], np.arange(start, start + count,
                                                        dtype=np.uint32)):
                raise ProbeError("sample ids are not the contiguous request span")
            if mode == "units":
                values = block["units"].astype(np.float32)
            elif mode == "trunc":
                values = block["truncated"].astype(np.float32)
            elif mode == "norm255":
                values = (block["normalized"].astype(np.float32)
                          * np.float32(255.0))
            else:
                raise ProbeError("unknown feature mode %s" % mode)
            key = (span["tex"], span["role"], span["ax"], span["ay"])
            array = np.zeros((th, tw, 4), dtype=np.float32)
            array[cells_y, cells_x] = values
            mode_lookup[key] = array
            if want_raw:
                raw = {}
                for name, field in (("normalized", "normalized"),
                                    ("units", "units"),
                                    ("truncated", "truncated")):
                    raw_array = np.zeros((th, tw, 4), dtype=np.float32)
                    raw_array[cells_y, cells_x] = block[field].astype(np.float32)
                    raw[name] = raw_array
                raw_lookup[key] = raw
    return mode_lookup, raw_lookup


def _run_measured(data, lookup):
    def sample(tex, pos_x, pos_y, role, ax, ay):
        del pos_x, pos_y
        return lookup[(tex, role, ax, ay)]

    def load(tex, x, y, role, ax, ay):
        del x, y
        return lookup[(tex, role, ax, ay)]

    return _replay_d25(data, sample, load)


def _packing_report(result, official, baseline, baseline_packs, mask, oh, ow):
    baseline_arrays = {"u0": baseline["got0"], "u1": baseline_packs["u1"],
                       "u2": baseline_packs["u2"], "u3": baseline_packs["u3"]}
    candidates = {}
    for candidate in PACKING_CANDIDATES:
        packs = _pack_outputs(result["pack"], result["got0"].shape[0],
                              result["got0"].shape[1], oh, ow, candidate)
        counts = _case_counts(result, packs, official, mask, oh, ow)
        baseline_counts = _case_counts(result, packs, baseline_arrays,
                                       mask, oh, ow)
        candidates[candidate] = {
            "counts_official": _flatten_counts(counts),
            "counts_vs_baseline": _flatten_counts(baseline_counts),
            "max_abs_diff_official": {key: counts[key]["max_abs_diff"]
                                      for key in ("u0", "u1", "u2", "u3")},
            "max_abs_diff_vs_baseline": {
                key: baseline_counts[key]["max_abs_diff"]
                for key in ("u0", "u1", "u2", "u3")},
            "packs": packs,
            "counts": counts,
        }
    # The production replay's unorm8 is floor(x + 0.5); keep it first and
    # report it alongside every measured-units candidate.  No candidate is
    # chosen by fitting the official image.
    production = candidates["half_up"]
    return {
        "production_unorm8": {
            "candidate": "half_up",
            "counts_official": production["counts_official"],
            "counts_vs_baseline": production["counts_vs_baseline"],
            "max_abs_diff_official": production["max_abs_diff_official"],
            "max_abs_diff_vs_baseline": production["max_abs_diff_vs_baseline"],
        },
        "candidates": {name: {
            "counts_official": value["counts_official"],
            "counts_vs_baseline": value["counts_vs_baseline"],
            "max_abs_diff_official": value["max_abs_diff_official"],
            "max_abs_diff_vs_baseline": value["max_abs_diff_vs_baseline"],
        } for name, value in candidates.items()},
        "selected_candidate": "half_up",
        "selection_rule": "original production unorm8, reported alongside all "
                          "measured-units candidates; not a fit-based choice",
        "selected_counts_official": production["counts_official"],
        "selected_counts_vs_baseline": production["counts_vs_baseline"],
        "selected_packs": production["packs"],
        "note": "UNORM store conversion candidates; production half_up is kept "
                "as its own variant, not selected by official-image fit",
    }


def _calls_for_lane(raw_lookup, lane, y, x):
    ax, ay = SUBS[lane]
    calls = {}
    for tex, role in (("t2", "centre2"), ("t3", "centre3"),
                      ("t2", "warp2"), ("t3", "warp3"),
                      ("t2", "self2"), ("t3", "self3")):
        values = raw_lookup[(tex, role, ax, ay)]
        calls[role] = {
            "tex": tex,
            "normalized_rgba": [float(value) for value in values["normalized"][y, x]],
            "units_rgba": [float(value) for value in values["units"][y, x]],
            "truncated_rgba": [int(value) for value in values["truncated"][y, x]],
        }
    return calls


def _lane_contribution_details(result, baseline, data, y, x, ch):
    """All four lanes' quantised changes and their GEMV contribution.

    A d25 output channel reads all four lanes of every feature group, so the
    channel's ``channel & 3`` lane is not the only possible source of a byte
    change.  The per-lane contributions are summed and checked against the
    replay's accumulator difference.
    """
    weights = data["weights"]
    group = ch >> 2
    lane_base = (ch & 3) * 4
    lanes = []
    total = 0
    for lane in range(4):
        changes = []
        lane_total = 0
        for feature_group in range(19):
            baseline_value = int(baseline["quantised"][feature_group, lane, y, x])
            measured_value = int(result["quantised"][feature_group, lane, y, x])
            if measured_value == baseline_value:
                continue
            row = 4 * feature_group + group
            weight = int(weights[row, lane_base + lane])
            contribution = (measured_value - baseline_value) * weight
            lane_total += contribution
            changes.append({
                "group": int(feature_group),
                "slot": int(feature_group),
                "baseline_quantised": baseline_value,
                "measured_quantised": measured_value,
                "delta_quantised": measured_value - baseline_value,
                "weight": weight,
                "weight_row": int(row),
                "weight_column": int(lane_base + lane),
                "delta_acc_contribution": int(contribution),
            })
        lanes.append({
            "lane": lane,
            "changes": changes,
            "delta_acc": int(lane_total),
        })
        total += lane_total
    delta_acc = int(result["accs"][ch, y, x] - baseline["accs"][ch, y, x])
    if total != delta_acc:
        raise ProbeError(
            "target %s: lane-change GEMV delta %d != accumulator delta %d"
            % ((y, x, ch), total, delta_acc))
    return {
        "delta_acc": delta_acc,
        "delta_acc_from_lane_changes": int(total),
        "matches_accumulator_delta": True,
        "delta_value_f32": float(result["values"][ch, y, x]
                                 - baseline["values"][ch, y, x]),
        "lanes": lanes,
    }


def _target_details(result, baseline, packing, official, data, raw_lookup, target):
    y, x, ch = target
    th, tw = data["th"], data["tw"]
    lane = ch & 3
    ax, ay = SUBS[lane]
    oy, ox = 2 * y + ay, 2 * x + ax
    units = result["units_lanes"][lane][y, x]
    quantised = result["quantised"][:, lane, y, x]
    baseline_units = baseline["units_lanes"][lane][y, x]
    baseline_quantised = baseline["quantised"][:, lane, y, x]
    frac = units - np.floor(units)
    value_changes = []
    index_changes = []
    for slot in range(19):
        if float(units[slot]) != float(baseline_units[slot]):
            value_changes.append({
                "slot": int(slot),
                "baseline_units": float(baseline_units[slot]),
                "measured_units": float(units[slot]),
                "delta_units": float(units[slot]) - float(baseline_units[slot]),
            })
        if int(quantised[slot]) != int(baseline_quantised[slot]):
            index_changes.append({
                "slot": int(slot),
                "baseline_quantised": int(baseline_quantised[slot]),
                "measured_quantised": int(quantised[slot]),
            })
    detail = {
        "coord_y_x_ch": [int(y), int(x), int(ch)],
        "lane": lane,
        "official_byte": int(official["u0"][y, x, ch]),
        "baseline_byte": int(baseline["got0"][y, x, ch]),
        "hardware_byte": int(result["got0"][y, x, ch]),
        "value_f32": float(result["values"][ch, y, x]),
        "acc": int(result["accs"][ch, y, x]),
        "scale": float(data["scale"][ch]),
        "bias": float(data["bias"][ch]),
        "margin_to_official_floor": float(int(official["u0"][y, x, ch])
                                          - result["values"][ch, y, x]),
        "margin_above_own_floor": float(result["values"][ch, y, x]
                                        - int(result["got0"][y, x, ch])),
        "units": [float(value) for value in units],
        "quantised": [int(value) for value in quantised],
        "baseline_units": [float(value) for value in baseline_units],
        "baseline_quantised": [int(value) for value in baseline_quantised],
        "feature_value_changes_vs_baseline": value_changes,
        "feature_index_changes_vs_baseline": index_changes,
        "lane_quantised_contributions": _lane_contribution_details(
            result, baseline, data, y, x, ch),
        "feature_frac": [float(value) for value in frac],
        "calls": _calls_for_lane(raw_lookup, lane, y, x),
        "outputs": {},
    }
    one, two, three = result["pack"][(ax, ay)]
    for name, raw in (("u1", one), ("u2", two), ("u3", three)):
        units_rgba = raw[y, x]
        candidate_bytes = _unorm_candidate(units_rgba, packing["selected_candidate"])
        detail["outputs"][name] = {
            "pixel_y_x": [int(oy), int(ox)],
            "pack_units_rgba": [float(value) for value in units_rgba],
            "selected_candidate_bgra": [int(value)
                                        for value in candidate_bytes[[2, 1, 0, 3]]],
            "official_bgra": [int(value) for value in official[name][oy, ox]],
            "baseline_bgra": [int(value)
                              for value in packing["selected_packs"][name][oy, ox]],
        }
    # The reference probe reports the scale/bias leverage per slot; expose it
    # without changing the production replay.
    weights = data["weights"]
    group = ch >> 2
    lane_in_group = ch & 3
    detail["scale"] = float(data["scale"][ch])
    detail["bias"] = float(data["bias"][ch])
    detail["slot_weights"] = [
        int(weights[4 * g + group, lane_in_group * 4 + lane]) for g in range(19)]
    detail["slot_value_leverage"] = [
        float(data["scale"][ch]) * value for value in detail["slot_weights"]]
    return detail


def _nearby_controls(result, baseline, official, data, targets, radius=2):
    th, tw = data["th"], data["tw"]
    controls = []
    for y, x, ch in targets:
        lane = ch & 3
        for yy in range(max(0, y - radius), min(th, y + radius + 1)):
            for xx in range(max(0, x - radius), min(tw, x + radius + 1)):
                if (yy, xx) == (y, x):
                    continue
                if int(result["got0"][yy, xx, ch]) != int(official["u0"][yy, xx, ch]):
                    continue
                values = result["units_lanes"][lane][yy, xx]
                value = float(result["values"][ch, yy, xx])
                controls.append({
                    "ref_target": [int(y), int(x), int(ch)],
                    "coord_y_x": [int(yy), int(xx)],
                    "official_byte": int(official["u0"][yy, xx, ch]),
                    "baseline_byte": int(baseline["got0"][yy, xx, ch]),
                    "hardware_byte": int(result["got0"][yy, xx, ch]),
                    "value": value,
                    "margin_above_floor": value - float(result["got0"][yy, xx, ch]),
                    "distance_to_next_int": float(np.ceil(value) - value)
                    if value > np.floor(value) else 0.0,
                    "nearest_feature_frac": float(np.min(np.minimum(
                        values % 1.0, 1.0 - (values % 1.0)))),
                })
    return controls


def _u0_mismatch_details(result, baseline, official, mask, limit=1000):
    got = result["got0"].astype(np.int16)
    expected = official["u0"].astype(np.int16)
    baseline_bytes = baseline["got0"].astype(np.int16)
    selected = (got != expected) & mask[..., None]
    coords = np.argwhere(selected)
    details = []
    for y, x, ch in coords[:limit]:
        details.append({
            "coord_y_x_ch": [int(y), int(x), int(ch)],
            "official_byte": int(expected[y, x, ch]),
            "baseline_byte": int(baseline_bytes[y, x, ch]),
            "hardware_byte": int(got[y, x, ch]),
            "value_f32": float(result["values"][ch, y, x]),
        })
    return {
        "count": int(len(coords)),
        "region_bytes": int(mask.sum() * 16),
        "total_bytes": int(expected.size),
        "listed": int(len(details)),
        "items": details,
    }


def _save_reconstructed_arrays(case_dir, result, production_packs, baseline):
    directory = Path(case_dir) / "reconstructed"
    if directory.exists():
        raise ProbeError("reconstructed output already exists: %s" % directory)
    directory.mkdir()
    arrays = {
        "u0_measured_units": result["got0"],
        "u1_production_unorm8": production_packs["u1"],
        "u2_production_unorm8": production_packs["u2"],
        "u3_production_unorm8": production_packs["u3"],
        "u0_production_baseline": baseline["got0"],
    }
    saved = {}
    for name, array in arrays.items():
        path = directory / ("%s.npy" % name)
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
        saved[name] = {
            "file": str(path.relative_to(Path(case_dir).parent)),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": sha256_file(path),
            "note": ("measured GPU units" if name.startswith("u0_measured")
                     else "production unorm8 packing with measured GPU units"
                     if "production_unorm8" in name
                     else "production baseline replay"),
        }
    return saved


def _analyse_case(out, case):
    spec = _spec_from_case(case)
    rows, dispatch, data, gate = _gate_capture(spec)
    del rows
    for key, path in (("trace_sha256", Path(spec["run"]) / "trace" / "trace.jsonl"),
                      ("t2_sha256", Path(spec["run"]) / "trace" / "tensors"
                       / dispatch_tensor_file(dispatch, "t2")),
                      ("t3_sha256", Path(spec["run"]) / "trace" / "tensors"
                       / dispatch_tensor_file(dispatch, "t3"))):
        if sha256_file(path) != case["provenance"][key]:
            raise ProbeError("%s: %s changed since prepare" % (case["name"], key))
    if gate["sha256"] != case["gate"]["sha256"]:
        raise ProbeError("%s: program SHA changed since prepare" % case["name"])

    th, tw, oh, ow = data["th"], data["tw"], data["oh"], data["ow"]
    station_mask = _region_mask(data, spec, case["region"]["mode"],
                                case["region"]["radius"])
    tables = _collect_calls(data)
    packets = _pack_requests(tables, data, station_mask)
    for tex in ("t2", "t3"):
        expected = request_bytes(packets[tex]["records"])
        path = Path(out) / case["name"] / case["requests"][tex]["file"]
        if sha256_file(path) != sha256_bytes(expected):
            raise ProbeError("%s: %s does not match the reproducible request packet"
                             % (case["name"], path))

    records = {}
    echo = {}
    for tex in ("t2", "t3"):
        request = case["requests"][tex]
        path = Path(out) / case["name"] / request["native_samples"]
        records[tex] = read_samples(path, expected_count=request["count"])
        _assert_echo(packets[tex]["records"], records[tex],
                     "%s/%s" % (case["name"], tex))
        echo[tex] = {
            "status": "exact",
            "records": int(len(records[tex])),
            "request_samples": int(request["samples"]),
            "request_loads": int(request["loads"]),
            "uv_float32_bits": "identical",
            "id_op_echo": "identical",
        }

    exact = _replay_d25(data, _exact_sample_hook(data), _exact_load_hook(data))
    exact_packs = _pack_outputs(exact["pack"], th, tw, oh, ow, "half_up")
    production = {}
    production_result, error = recon.reconstruct_warp_init(
        str(spec["run"]), dispatch, capture=production)
    if error:
        raise ProbeError("%s: production replay failed: %s" % (case["name"], error))
    if not np.array_equal(exact["got0"], production["u0"]):
        raise ProbeError("%s: copied replay drifted from production" % case["name"])
    for name in ("u1", "u2", "u3"):
        if not np.array_equal(exact_packs[name], production[name]):
            raise ProbeError("%s: copied %s drifted from production"
                             % (case["name"], name))
    official = _official_arrays(data)
    baseline_counts = _case_counts(exact, exact_packs, official, station_mask, oh, ow)

    variants = {}
    raw_lookup = {}
    selected_result = None
    unit_packing = None
    for mode in FEATURE_MODES:
        lookup, raw = _build_lookup(records, packets, data, mode,
                                    want_raw=(mode == FEATURE_MODES[0]))
        if mode == FEATURE_MODES[0]:
            raw_lookup = raw
        result = _run_measured(data, lookup)
        packing = _packing_report(result, official, exact, exact_packs,
                                  station_mask, oh, ow)
        if mode == "units":
            selected_result = result
            unit_packing = packing
        all_official = _case_counts(result, packing["selected_packs"], official,
                                    station_mask, oh, ow)
        all_baseline = _case_counts(result, packing["selected_packs"], {
            "u0": exact["got0"], "u1": exact_packs["u1"],
            "u2": exact_packs["u2"], "u3": exact_packs["u3"]},
            station_mask, oh, ow)
        variants[mode] = {
            "counts_official": _flatten_counts(all_official),
            "counts_vs_baseline": _flatten_counts(all_baseline),
            "max_abs_diff_official": {key: all_official[key]["max_abs_diff"]
                                      for key in ("u0", "u1", "u2", "u3")},
            "max_abs_diff_vs_baseline": {key: all_baseline[key]["max_abs_diff"]
                                         for key in ("u0", "u1", "u2", "u3")},
            "packing": {key: value for key, value in packing.items()
                        if key != "selected_packs"},
        }

    details = []
    controls = []
    if case["targets"]:
        for target in case["targets"]:
            details.append(_target_details(selected_result, exact, unit_packing,
                                           official, data, raw_lookup,
                                           tuple(target)))
        controls = _nearby_controls(selected_result, exact, official, data,
                                    [tuple(item) for item in case["targets"]],
                                    radius=case["region"]["radius"])
    saved_arrays = _save_reconstructed_arrays(
        Path(out) / case["name"], selected_result,
        unit_packing["selected_packs"], exact)
    u0_mismatches = _u0_mismatch_details(selected_result, exact, official,
                                          station_mask)

    return {
        "name": case["name"],
        "run": case["run"],
        "frame": case["frame"],
        "region": case["region"],
        "gate": gate,
        "provenance": case["provenance"],
        "echo": echo,
        "request_counts": {tex: case["requests"][tex]["count"]
                           for tex in ("t2", "t3")},
        "baseline": {
            "official_counts": _flatten_counts(baseline_counts),
            "official_max_abs_diff": {key: baseline_counts[key]["max_abs_diff"]
                                      for key in ("u0", "u1", "u2", "u3")},
            "production_note": production_result[3],
        },
        "variants": variants,
        "u0_mismatches": u0_mismatches,
        "saved_arrays": saved_arrays,
        "targets": details,
        "nearby_matching_controls": controls,
    }


def dispatch_tensor_file(dispatch, slot):
    for resource in dispatch.get("resources", []):
        if resource.get("slot") == slot and resource.get("file"):
            return resource["file"]
    raise ProbeError("dispatch has no %s tensor file" % slot)


def analyze(out):
    out = Path(out)
    manifest = _read_json(out / "manifest.json")
    current_driver = sha256_file(Path(__file__).resolve())
    if manifest.get("recon_sha256") != sha256_file(
            REPO / "src" / "xefg_sdk" / "export" / "recon.py"):
        raise ProbeError("recon.py changed since prepare; rerun prepare")
    target = out / "analysis.json"
    if target.exists():
        raise ProbeError("analysis already exists: %s" % target)
    report = {
        "schema": "xefg.hardware_sampler_analysis/1",
        "repo_head": manifest.get("repo_head"),
        "recon_sha256": manifest.get("recon_sha256"),
        "driver_sha256_manifest": manifest.get("driver_sha256"),
        "driver_sha256_current": current_driver,
        "driver_changed_since_prepare": manifest.get("driver_sha256") != current_driver,
        "provenance_note": "request packets are regenerated and hash-compared; "
                           "a changed analysis driver does not change the bundle",
        "contract": manifest.get("contract"),
        "cases": [],
    }
    for case in manifest["cases"]:
        report["cases"].append(_analyse_case(out, case))
        print("analysed %s" % case["name"], flush=True)
    _json_dump(target, report)
    return report


def _add_case_args(parser, required):
    parser.add_argument("--run", type=Path, required=required,
                        help="capture root containing trace/")
    parser.add_argument("--frame", type=int, required=required,
                        help="one capture frame to diagnose")
    parser.add_argument("--name", default=None,
                        help="bundle case name; default is RUNBASENAME_frameN")
    parser.add_argument("--target", action="append", type=_parse_target,
                        metavar="Y,X,CH",
                        help="target cell and output channel; repeatable")
    parser.add_argument("--region", default="auto",
                        choices=("auto", "targets", "full"),
                        help="auto uses targets when --target is given, else full")
    parser.add_argument("--radius", type=int, default=2,
                        help="target neighbourhood radius in cells")
    parser.add_argument("--require-complete", action="store_true",
                        help="require trace/meta.json to mark the trace complete")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="write a new request bundle")
    prepare_parser.add_argument("--out", required=True, type=Path,
                                help="new bundle directory")
    _add_case_args(prepare_parser, required=True)
    run_parser = sub.add_parser("run", help="prepare if needed and execute the native helper")
    run_parser.add_argument("--out", required=True, type=Path,
                            help="new or already prepared bundle directory")
    _add_case_args(run_parser, required=False)
    run_parser.add_argument("--exe", required=True, type=Path)
    run_parser.add_argument("--debug", action="store_true")
    run_parser.add_argument("--adapter", default=None)
    analyze_parser = sub.add_parser("analyze", help="analyse samples.bin")
    analyze_parser.add_argument("--out", required=True, type=Path,
                                help="prepared bundle directory")
    return parser


def _spec_from_args(parser, args):
    if args.run is None or args.frame is None:
        parser.error("--run and --frame are required for a new bundle")
    return _case_spec(args.run, args.frame, args.name,
                      tuple(args.target or ()), args.require_complete)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        prepare(args.out, _spec_from_args(parser, args), args.region, args.radius)
        return 0
    if args.command == "run":
        if not (args.out / "manifest.json").is_file():
            prepare(args.out, _spec_from_args(parser, args),
                    args.region, args.radius)
        execute_native(args.out, args.exe, args.debug, args.adapter)
        return 0
    if args.command == "analyze":
        analyze(args.out)
        return 0
    raise ProbeError("unknown command")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProbeError as error:
        print("error: %s" % error, file=sys.stderr)
        sys.exit(2)
