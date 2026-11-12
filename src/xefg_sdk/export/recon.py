#!/usr/bin/env python
"""Reconstruct the model's int8 dispatches from captured tensors, byte-exactly.

Usage:
    python -I trace/recon.py <run-dir> [--frame F] [--dispatch N] [--all]

Every captured operator family is replayed here. The conv template below is decoded from the captured
weights and root constants; the flow/refinement, 2x-upsample and buffer-GEMV
families are decoded from each program's own disassembly and allocations.

Verified against the fg78 two-times model. All conv programs share one shape:

    acc[z][g][w] = bias[z][g][w]
                 + sum_blocks sum_taps sum_word dot4AddPacked(a[block,tap,word], w[...])

    pos = max(clamp(acc, lo, hi), 0);  neg = min(clamp(acc, lo, hi), 0)
    out = clamp(((pos*scale + r1) >> s1) + ((neg*activation + r2) >> s2), -128, 127)

What differs per program is the input packing and the weight block size, and both
follow from the captured buffers:

  - `groups` comes from the bias buffer: t2 holds `groups*4` int32 per output
    slice, and the output texel is `groups` dwords (groups*4 int8 channels).
  - the weight buffer holds `groups * 9 taps * 4 input words` uint4 per block per
    output slice; the block count is whatever is left over.
  - the input view decides how the four words of a tap are gathered:
      * two-byte texels (R8G8_SINT): one texel gives two channels and a folded
        zero point fills the third lane (template A, one block);
      * eight-byte texels (R32G32_UINT): two array slices, two dwords each;
      * sixteen-byte texels (R32G32B32A32 read as raw ints): one texel, four
        dwords, so one array slice per block.

Sampling: the window is `stride*out + phase + tap`, with `stride` from the in/out
ratio and `phase` read from the program's own `textureLoad` coordinate (the
constant term of `stride*threadId + phase`, with cbuffer loads resolved from the
captured root constants). It is never chosen by trying phases against the
expected output. Only the logical top-left region of each (padded) allocation is
meaningful; a tap that leaves it contributes a zero word, which also zeroes the
folded zero point. The written row is the logical row + 1.

The replay is compared against the whole captured allocation, not only the
nonzero texels. Clamp bounds come from the pipeline-builder parameter capture
(d150/d151).
"""
# Internal legacy research module; not a supported CLI entry point.
import argparse
import json
import os
import re
import struct

import numpy as np

from . import d25_bound, dxil_ir

TAPS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
FORMAT_BYTES = {3: 16, 17: 8, 34: 4, 41: 4, 52: 2, 61: 1, 87: 4}

# The buffer GEMV walks a 3x3 window in raster order, two packed words per tap.
GEMV_TAPS = [(dx, dy) for dy in range(3) for dx in (-1, 0, 1)]

# One conv template covers 4 groups x (9 taps x 4 words) uint4 per input slice.
FLOW_UNITS = 144


def read_events(run_dir):
    rows = []
    with open(os.path.join(run_dir, "trace", "trace.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def layer_parameters(rows):
    out = {}
    for row in rows:
        if row.get("type") == "layer_parameters":
            out[row["key"]] = {k: int(v) for k, v in row.get("parameters", [])}
    return out


def constants_of(dispatch):
    values = [0] * 16
    for entry in dispatch.get("constants", []):
        if len(values) < len(entry["values"]):
            values.extend([0] * (len(entry["values"]) - len(values)))
        for i, v in enumerate(entry["values"]):
            values[i] = v
    return values


def tensor(run_dir, dispatch, slot):
    for r in dispatch["resources"]:
        if r["slot"] == slot and r.get("file"):
            path = os.path.join(run_dir, "trace", "tensors", r["file"])
            if os.path.exists(path):
                return np.fromfile(path, dtype=np.uint8), r
    return None, None


def _substitute_constants(expr, constants):
    """Replace `cbufferLoadLegacy` nodes with the captured root constants."""
    if isinstance(expr, tuple):
        if expr and expr[0] == "cbuf":
            if len(expr) != 3:
                raise RuntimeError("unresolved cbuffer value")
            index = expr[1] * 4 + expr[2]
            if index >= len(constants):
                raise RuntimeError("cbuffer index %d out of range" % index)
            return ("c", constants[index] & 0xFFFFFFFF)
        return tuple(_substitute_constants(x, constants) for x in expr)
    if isinstance(expr, list):
        return [_substitute_constants(x, constants) for x in expr]
    return expr


def _zero_unknowns(expr):
    """Replace unresolved loop/phi values with 0 to read a staging loop's origin."""
    if isinstance(expr, tuple):
        if expr and expr[0] == "unknown":
            return ("c", 0)
        return tuple(_zero_unknowns(x) for x in expr)
    if isinstance(expr, list):
        return [_zero_unknowns(x) for x in expr]
    return expr


def _conv_phase(run_dir, dispatch, constants, stride_x, stride_y):
    """Read the input-window phase from the program's own load coordinates.

    Two shapes occur. A direct conv loads `stride*threadId + phase + tap`, so the
    constant term is the phase. A stride-1 program with a groupshared tile
    stages `gid*NumThreads + base + tile`, and the conv reads the tile at the
    thread's own position; the same `base` is then the phase. Both are read from
    the DXIL, never by trying phases against the captured output.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    ir = dxil_ir.IR(path)
    if not ir.loads:
        return None, "no textureLoad in program"
    nthreads = dxil_ir.num_threads(path)
    bases_x, bases_y = [], []
    for name in ir.loads:
        load = ir.expr("%" + name)
        if load[0] != "res-tex":
            return None, "unsupported textureLoad form"
        x = _substitute_constants(load[1][1], constants)
        y = _substitute_constants(load[1][2], constants)
        direct = True
        try:
            bx, cx = dxil_ir.affine(x, nthreads)
            by, cy = dxil_ir.affine(y, nthreads)
        except (RuntimeError, IndexError):
            direct = False
        if direct and cx[0] == stride_x and cy[1] == stride_y:
            bases_x.append(bx)
            bases_y.append(by)
            continue
        # Staging loop: the coordinate depends on the group, not the thread, and
        # the loop index is a phi the symbolic reader cannot fold. Its value at
        # loop index 0 is the tile origin; the conv body reads the tile at the
        # thread position, so that origin is the phase.
        try:
            bx, cx = dxil_ir.affine(_zero_unknowns(x), nthreads)
            by, cy = dxil_ir.affine(_zero_unknowns(y), nthreads)
        except (RuntimeError, IndexError) as error:
            return None, "input coordinate is not affine: %s" % error
        if (cx[0] != 0 or cy[1] != 0
                or cx[3] != stride_x * nthreads[0]
                or cy[4] != stride_y * nthreads[1]):
            return None, "input coordinate is not stride-aligned"
        bases_x.append(bx)
        bases_y.append(by)
    if not bases_x:
        return None, "no usable input coordinate"
    # Taps add 0..2 on each axis, so the smallest base is the window origin.
    return (min(bases_x), min(bases_y)), None


def conv_shaped(dispatch):
    if flow_shaped(dispatch) or gemv_shaped(dispatch) or upsample_shaped(dispatch):
        return False
    slots = {r["slot"] for r in dispatch["resources"]}
    if not {"t0", "t1", "t2", "t3", "u0"} <= slots:
        return False
    for r in dispatch["resources"]:
        if r["slot"] in ("t1", "t2", "t3") and r.get("dimension") != "buffer":
            return False
        if r["slot"] == "u0" and r.get("dimension") != "texture2d":
            return False
    return True


def requant(acc, activation, scale, s1, s2, lo, hi):
    clamped = np.clip(acc, lo, hi)
    pos = np.maximum(clamped, 0)
    neg = np.minimum(clamped, 0)
    r1, r2 = 1 << ((s1 - 1) & 31), 1 << ((s2 - 1) & 31)
    return np.clip(((pos * scale + r1) >> s1) + ((neg * activation + r2) >> s2),
                   -128, 127)


def sample(template, src_bytes, blocks, groups, words_per_tap, weights, bias, activation, pad,
           valid, out_w, out_h, stride_x, stride_y, phase_x, phase_y,
           scale, s1, s2, lo, hi, zero_point, slices, H, W, out_bytes):
    words = np.zeros((blocks, 4, 9, out_h, out_w, 4), dtype=np.int64)
    for i in range(blocks):
        for tap, (dx, dy) in enumerate(TAPS):
            ys = slice(3 + phase_y + dy, 3 + phase_y + dy + stride_y * out_h, stride_y)
            xs = slice(3 + phase_x + dx, 3 + phase_x + dx + stride_x * out_w, stride_x)
            if template == "A":
                words[i, 0, tap, ..., 0] = pad[0, ys, xs, 0]
                words[i, 0, tap, ..., 1] = pad[0, ys, xs, 1]
                # an out-of-bounds tap zeroes the whole word, zero point included
                words[i, 0, tap, ..., 2] = zero_point * valid[0, ys, xs]
            elif src_bytes == 8:
                for k, s in enumerate((2 * i, 2 * i, 2 * i + 1, 2 * i + 1)):
                    offset = 0 if k % 2 == 0 else 4
                    words[i, k, tap] = pad[s, ys, xs, offset:offset + 4]
            else:
                for k in range(4):
                    words[i, k, tap] = pad[i, ys, xs, 4 * k:4 * k + 4]

    acc = np.zeros((slices, groups, out_h, out_w, 4), dtype=np.int32)
    for i in range(blocks):
        for tap in range(9):
            for k in range(words_per_tap):
                term = np.einsum("yxl,zgwl->zgyxw", words[i, k, tap],
                                 weights[:, i, :, tap, k, :, :]).astype(np.int32)
                # dx.op.dot4AddPacked is i32, and the tap chain adds in i32 too.
                acc = (acc + term).astype(np.int32)
    acc = (acc + bias[:, :, None, None, :].astype(np.int32)).astype(np.int32)
    value = requant(acc, activation[:, :, None, None, :], scale, s1, s2, lo, hi)
    got = np.zeros((slices, H, W, out_bytes), dtype=np.int64)
    got[:, 1:1 + out_h, :out_w] = value.transpose(0, 2, 3, 1, 4).reshape(
        slices, out_h, out_w, out_bytes)
    return got


def conv_plan(run_dir, dispatch, params):
    """Derive one int8 conv's layout, weights and captured output.

    Shared by the numpy replay and the ONNX exporter, so both read the same
    block/tap/quantisation layout instead of re-deriving it.
    """
    c = constants_of(dispatch)
    in_w, in_h = c[2], c[3]
    out_w, out_h = c[4], c[5]
    scale = c[9]
    s1, s2 = c[10] & 0xFF, (c[10] >> 8) & 0xFF
    zero_point = c[11]
    lo, hi = params.get("d150", -131072), params.get("d151", 131071)

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, _ = tensor(run_dir, dispatch, "t1")
    t2, _ = tensor(run_dir, dispatch, "t2")
    t3, _ = tensor(run_dir, dispatch, "t3")
    u0, r4 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, u0)):
        return None, "tensor files missing"
    if r4.get("dimension") != "texture2d":
        return None, "output is not a texture"
    out_bytes = FORMAT_BYTES.get(r4["format"])
    if out_bytes not in (8, 16):
        return None, "unsupported output format %d" % r4["format"]

    slices = r4["array"]
    H, W = r4["height"], r4["width"]
    groups = out_bytes // 4
    if t2.size != slices * groups * 16:
        return None, "bias size does not match %d groups" % groups
    if t3.size != t2.size:
        return None, "activation size does not match bias"
    bias = t2.view(np.int32).reshape(slices, groups, 4).astype(np.int64)
    activation = t3.view(np.int32).reshape(slices, groups, 4).astype(np.int64)
    expected = u0.view(np.int8).reshape(slices, H, W, out_bytes).astype(np.int64)

    src_bytes = FORMAT_BYTES.get(r0["format"])
    if src_bytes == 2:
        template, words_per_tap = "A", 1
    elif src_bytes in (8, 16):
        template, words_per_tap = "BC", 4
    else:
        return None, "unsupported input format %d" % r0["format"]
    per_block = groups * 9 * words_per_tap
    if t1.size % (slices * per_block * 16):
        return None, "weight size does not match the block layout"
    blocks = t1.size // (slices * per_block * 16)
    weights = t1.view(np.int8).reshape(slices, blocks, groups, 9, words_per_tap,
                                       4, 4).astype(np.int64)

    src = t0.view(np.int8).reshape(r0["array"], r0["height"], r0["width"],
                                    src_bytes).astype(np.int64)
    # BC programs bound-check `y >= in_h + 1` instead of `in_h`: the producer
    # wrote logical row 0 at row 1, so the shader's valid band is one row
    # taller. Row 0 is rejected by the shader itself and is zero in the dump.
    valid_h = in_h + 1 if template == "BC" else in_h

    # Ratios are exact except for odd heights (23 -> 12 is stride 2), so round.
    stride_x = max(1, int(round(in_w / max(out_w, 1))))
    stride_y = max(1, int(round(in_h / max(out_h, 1))))

    phase, error = _conv_phase(run_dir, dispatch, c, stride_x, stride_y)
    if error:
        return None, error
    phase_x, phase_y = phase
    plan = {
        "template": template,
        "src_bytes": src_bytes,
        "words_per_tap": words_per_tap,
        "blocks": blocks,
        "groups": groups,
        "slices": slices,
        "H": H,
        "W": W,
        "out_w": out_w,
        "out_h": out_h,
        "in_w": in_w,
        "in_h": in_h,
        "valid_h": valid_h,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "phase": (phase_x, phase_y),
        "weights": weights,
        "bias": bias,
        "activation": activation,
        "scale": scale,
        "s1": s1,
        "s2": s2,
        "lo": lo,
        "hi": hi,
        "zero_point": zero_point,
        "src": src,
        "expected": expected,
        "out_bytes": out_bytes,
    }
    return plan, None


def reconstruct(run_dir, dispatch, params):
    plan, error = conv_plan(run_dir, dispatch, params)
    if error:
        return None, error
    src = plan["src"]
    valid_h, in_w = plan["valid_h"], plan["in_w"]
    masked = np.zeros_like(src)
    masked[:, :valid_h, :in_w] = src[:, :valid_h, :in_w]
    pad = np.zeros((src.shape[0], src.shape[1] + 6, src.shape[2] + 6, src.shape[3]),
                   dtype=np.int64)
    pad[:, 3:3 + src.shape[1], 3:3 + src.shape[2]] = masked
    valid = np.zeros(pad.shape[:3], dtype=np.int64)
    valid[:, 3:3 + valid_h, 3:3 + in_w] = 1

    phase_x, phase_y = plan["phase"]
    got = sample(plan["template"], plan["src_bytes"], plan["blocks"], plan["groups"],
                 plan["words_per_tap"], plan["weights"], plan["bias"],
                 plan["activation"], pad, valid, plan["out_w"], plan["out_h"],
                 plan["stride_x"], plan["stride_y"], phase_x, phase_y,
                 plan["scale"], plan["s1"], plan["s2"], plan["lo"], plan["hi"],
                 plan["zero_point"], plan["slices"], plan["H"], plan["W"],
                 plan["out_bytes"])
    expected = plan["expected"]
    # Whole allocation: the captured output is fully initialised, so padding and
    # rows the program never writes must be zero in both.
    mismatch = int((got != expected).sum())
    return (mismatch, int(expected.size),
            (plan["in_w"], plan["in_h"], plan["out_w"], plan["out_h"]),
            plan["template"], plan["blocks"], plan["groups"], plan["stride_x"],
            plan["phase"]), None


def flow_shaped(dispatch):
    """The flow/refinement template: four root constants, int8 weight table,
    float scale/bias table, one Texture2DArray input and one output."""
    if upsample_shaped(dispatch):
        return False
    slots = {r["slot"]: r for r in dispatch["resources"]}
    if not {"t0", "t1", "t2", "t3", "u0"} <= set(slots):
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    if len(dispatch["constants"][0]["values"]) != 4:
        return False
    for slot in ("t1", "t2", "t3"):
        if slots[slot].get("dimension") != "buffer":
            return False
    for slot in ("t0", "u0"):
        if slots[slot].get("dimension") != "texture2d":
            return False
    return True


def _disasm_path(run_dir, dispatch):
    return os.path.join(run_dir, "trace", "disasm",
                        "%s-%s.ll" % (dispatch["key"], dispatch["sha256"][:16]))


def _tensor(run_dir, dispatch, slot):
    for r in dispatch["resources"]:
        if r["slot"] == slot and r.get("file"):
            path = os.path.join(run_dir, "trace", "tensors", r["file"])
            if os.path.exists(path):
                return np.fromfile(path, dtype=np.uint8), r
    return None, None


def reconstruct_flow(run_dir, dispatch):
    """Replay one flow/refinement dispatch from its disassembly and dumps.

    The program is a 4-group int8 3x3 conv over one or more input slices. What
    varies per program is the output position each thread writes (one texel, a
    1x2 or 2x2 block), the input-window origin, whether four phase windows are
    reduced with a max, the requant clamp bounds and the input's +0x80 bias.
    All of it is read from the DXIL, not assumed.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    ir = dxil_ir.IR(path)
    if not ir.stores:
        return None, "no integer textureStore"

    t0, r0 = _tensor(run_dir, dispatch, "t0")
    t1, _ = _tensor(run_dir, dispatch, "t1")
    t2, _ = _tensor(run_dir, dispatch, "t2")
    t3, _ = _tensor(run_dir, dispatch, "t3")
    u0, r4 = _tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, u0)):
        return None, "tensor files missing"

    nthreads = dxil_ir.num_threads(path)
    grid = tuple(dispatch["group"][i] * nthreads[i] for i in range(3))
    if min(grid) == 0:
        return None, "empty grid"
    slices, outs = r0["array"], r4["array"]
    in_w, in_h = r0["width"], r0["height"]
    width, height = r4["width"], r4["height"]
    nw = t1.size // 16
    if nw != outs * slices * FLOW_UNITS:
        return None, "weight size does not match %d x %d slices" % (outs, slices)
    if t2.size < outs * 4 * 16 or t3.size < outs * 4 * 16:
        return None, "scale/bias table smaller than %d slices" % outs

    source = t0.view(np.uint32).reshape(-1, 4)
    weights = t1.view(np.int8).reshape(nw, 4, 4).astype(np.int32)
    scale = t2.view(np.float32).reshape(-1)
    bias = t3.view(np.float32).reshape(-1)
    expected = u0.view(np.uint8).reshape(outs, height, width, 16)

    tigx = np.arange(nthreads[0])[:, None, None, None, None, None]
    tigy = np.arange(nthreads[1])[None, :, None, None, None, None]
    tigz = np.arange(nthreads[2])[None, None, :, None, None, None]
    gidx = np.arange(grid[0] // nthreads[0])[None, None, None, :, None, None]
    gidy = np.arange(grid[1] // nthreads[1])[None, None, None, None, :, None]
    gidz = np.arange(grid[2] // nthreads[2])[None, None, None, None, None, :]
    axes = [a.ravel() for a in np.broadcast_arrays(tigx, tigy, tigz, gidx, gidy, gidz)]
    count = axes[0].size

    flat_cache = {}

    def flat(expr):
        # Keep the expression in the entry: that pins its id(), so a temporary
        # tuple freed and reallocated at the same address cannot alias here.
        key = id(expr)
        hit = flat_cache.get(key)
        if hit is not None and hit[0] is expr:
            return hit[1]
        base, coefs = dxil_ir.affine(expr, nthreads)
        out = np.full(count, base, dtype=np.int64)
        for coef, axis in zip(coefs, axes):
            out += coef * axis
        flat_cache[key] = (expr, out)
        return out

    written = np.zeros(expected.shape, dtype=bool)
    got = np.zeros(expected.shape, dtype=np.uint8)
    word_cache = {}

    for store in ir.stores:
        m = re.match(r'^call void @dx\.op\.textureStore\.i32\((.*)\)$', store)
        if not m:
            return None, "unsupported store form"
        args = dxil_ir.split_args(m.group(1))
        if len(args) != 10:
            return None, "unsupported store arity"
        try:
            ox, oy, oz = flat(ir.expr(args[2])), flat(ir.expr(args[3])), flat(ir.expr(args[4]))
        except RuntimeError:
            return None, "output coordinate is not affine"
        valid = ((ox >= 0) & (ox < width) & (oy >= 0) & (oy < height)
                 & (oz >= 0) & (oz < outs))

        packs = [ir.expr(a) for a in args[5:9]]
        for group in range(4):
            for lane in range(4):
                core, sc, bs, lo, hi = dxil_ir.strip_requant(packs[group][1][lane])
                if core is None:
                    return None, "unrecognised requant"
                if lo is None or hi is None:
                    return None, "missing clamp bounds"
                sidx = flat(sc[1]) * 4 + (sc[2] if isinstance(sc[2], int) else flat(sc[2]))
                bidx = flat(bs[1]) * 4 + (bs[2] if isinstance(bs[2], int) else flat(bs[2]))

                leaves = []

                def walk(node):
                    if node[0] == 'max':
                        walk(node[1])
                        walk(node[2])
                    else:
                        leaves.append(node)

                walk(core)
                candidates = []
                for leaf in leaves:
                    base, terms = dxil_ir.linearize(leaf)
                    looped = not (base[0] == 'c' and base[1] == 0)
                    acc = np.zeros(count, dtype=np.int32)
                    for si in range(slices if looped else 1):
                        for a, b in terms:
                            try:
                                tex, add = dxil_ir.unpack_word(a)
                            except RuntimeError:
                                return None, "input staged in groupshared"
                            _, tx, ty, tz = tex[1]
                            word = tex[2]
                            key = (id(a), si)
                            entry = word_cache.get(key)
                            if entry is not None and entry[0] is a:
                                word_bytes = entry[1]
                            else:
                                px, py = flat(tx), flat(ty)
                                pz = (np.full(count, si, dtype=np.int64)
                                      if looped else flat(tz))
                                ok = ((px >= 0) & (px < in_w) & (py >= 0)
                                      & (py < in_h) & (pz >= 0) & (pz < slices))
                                raw = np.zeros((count, 4), dtype=np.uint32)
                                at = np.where(ok, (pz * in_h + py) * in_w + px, 0)
                                raw[ok] = source[at[ok]]
                                added = ((raw.astype(np.uint64) + np.uint64(add & 0xFFFFFFFF))
                                         & 0xFFFFFFFF).astype(np.uint32)
                                word_bytes = added.view(np.int8).reshape(count, 16)
                                word_cache[key] = (a, word_bytes)
                            wb = word_bytes[:, word * 4:word * 4 + 4].astype(np.int32)
                            base_idx = flat(b[1])
                            widx = (oz * slices + si) * FLOW_UNITS + (base_idx % FLOW_UNITS)
                            wlane = b[2] if isinstance(b[2], int) else flat(b[2])
                            if widx.min() < 0 or widx.max() >= nw:
                                return None, 'weight index out of range'
                            w = weights[widx, wlane]
                            term = (wb * w).sum(axis=1).astype(np.int32)
                            acc = (acc.astype(np.int64) + term).astype(np.int32)
                    candidates.append(acc)
                value = candidates[0] if len(candidates) == 1 else np.maximum.reduce(candidates)
                if sidx.min() < 0 or sidx.max() >= scale.size:
                    return None, 'scale index out of range'
                if bidx.min() < 0 or bidx.max() >= bias.size:
                    return None, 'bias index out of range'
                # The DXIL emits `fmul fast` + `fadd fast`; fast-math lets the
                # backend contract them into one FMA, so the requant has a
                # single float32 rounding. The float64 expression below is the
                # exact product + sum rounded once to float32. At 2560x1440 this
                # fixes one d29 byte whose two-round value is exactly 4.0 while
                # the FMA value is 3.9999998.
                f = np.float32(value.astype(np.float64) * scale[sidx].astype(np.float64)
                               + bias[bidx].astype(np.float64))
                result = np.clip(np.trunc(f.astype(np.float64)), lo, hi).astype(np.int64)

                channel = group * 4 + lane
                got[oz[valid], oy[valid], ox[valid], channel] = result[valid].astype(np.uint8)
                written[oz[valid], oy[valid], ox[valid], channel] = True

    total = expected.size
    if written.sum() != total:
        missing = total - int(written.sum())
        return (missing, total, (in_w, in_h, width, height, slices, outs), "unwritten bytes"), None
    mismatch = int((got != expected).sum())
    return (mismatch, total, (in_w, in_h, width, height, slices, outs), ""), None


def _flow_right_pad(in_dim, out_dim, stride, origin):
    """Smallest right/bottom pad that makes a 3x3 conv produce `out_dim`."""
    left = -origin
    for right in range(0, stride + 4):
        if (in_dim + left + right - 3) // stride + 1 == out_dim:
            return right
    return None


def flow_plan(run_dir, dispatch):
    """Derive a dense 3x3 int8 convolution for one flow/refinement dispatch.

    The replay above walks the DXIL store by store. This plan proves the same
    body is a regular convolution and returns the tensors the ONNX exporter
    needs:

      * `d26`-`d29` are 2x downscales: four phase windows, reduced with max;
      * `d30`, `d32`, `d34`, `d36` are 1:1 convolutions (the several stores
        only distribute output texels across threads).

    Every term's weight index, output lane, tap and input affine map is checked
    against the captured IR, so a layout change fails here instead of producing
    a plausible but wrong graph. The input bias (`0x80808080` on `d26`) is a
    packed-word add; out-of-bounds texels therefore read the bias word, not
    zero, and the plan carries the matching pad value.
    """
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if not {"t0", "t1", "t2", "t3", "u0"} <= set(slots):
        return None, "flow slots missing"
    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, _ = tensor(run_dir, dispatch, "t1")
    t2, _ = tensor(run_dir, dispatch, "t2")
    t3, _ = tensor(run_dir, dispatch, "t3")
    u0, r4 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, u0)):
        return None, "tensor files missing"
    if FORMAT_BYTES.get(r0.get("format")) != 16 or FORMAT_BYTES.get(r4.get("format")) != 16:
        return None, "flow tensors are not 16-byte texels"

    slices, outs = r0["array"], r4["array"]
    in_w, in_h = r0["width"], r0["height"]
    width, height = r4["width"], r4["height"]
    stride_x = max(1, int(round(in_w / max(width, 1))))
    stride_y = max(1, int(round(in_h / max(height, 1))))
    if stride_x != stride_y or stride_x not in (1, 2):
        return None, "unsupported stride %dx%d" % (stride_x, stride_y)

    nw = t1.size // 16
    if t1.size % 16 or nw != outs * slices * FLOW_UNITS:
        return None, "weight size does not match %d x %d slices" % (outs, slices)
    if t2.size < outs * 16 * 4 or t3.size < outs * 16 * 4:
        return None, "scale/bias table smaller than %d slices" % outs
    wbuf = t1.view(np.int8).reshape(nw, 16)
    scale = np.ascontiguousarray(t2.view("<f4")[:outs * 16], dtype=np.float32)
    bias = np.ascontiguousarray(t3.view("<f4")[:outs * 16], dtype=np.float32)

    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    ir = dxil_ir.IR(path)
    if not ir.stores:
        return None, "no integer textureStore"
    nthreads = dxil_ir.num_threads(path)

    adds, los, his, phases, leaf_counts = set(), set(), set(), set(), set()
    for store in ir.stores:
        match = re.match(r"^call void @dx\.op\.textureStore\.i32\((.*)\)$", store)
        if not match:
            return None, "unsupported store form"
        args = dxil_ir.split_args(match.group(1))
        if len(args) != 10:
            return None, "unsupported store arity"
        out_x = dxil_ir.affine(ir.expr(args[2]), nthreads)
        out_y = dxil_ir.affine(ir.expr(args[3]), nthreads)
        packs = [ir.expr(a) for a in args[5:9]]
        for group in range(4):
            for lane in range(4):
                core, sc, bs, lo, hi = dxil_ir.strip_requant(packs[group][1][lane])
                if core is None or lo is None or hi is None:
                    return None, "unrecognised requant"
                los.add(lo)
                his.add(hi)
                if sc[0] != "w" or bs[0] != "w":
                    return None, "scale/bias is not a raw buffer load"
                for out_slice in range(outs):
                    env = {"tig": (0, 0, out_slice), "gid": (0, 0, 0),
                           "tid": (0, 0, out_slice)}
                    sidx = dxil_ir.evaluate(sc[1], env) * 4 + (
                        sc[2] if isinstance(sc[2], int) else dxil_ir.evaluate(sc[2], env))
                    bidx = dxil_ir.evaluate(bs[1], env) * 4 + (
                        bs[2] if isinstance(bs[2], int) else dxil_ir.evaluate(bs[2], env))
                    expected_index = out_slice * 16 + group * 4 + lane
                    if sidx != expected_index or bidx != expected_index:
                        return None, "scale/bias index %d/%d != %d" % (
                            sidx, bidx, expected_index)

                leaves = []
                def walk(node):
                    if node[0] == "max":
                        walk(node[1])
                        walk(node[2])
                    else:
                        leaves.append(node)

                walk(core)
                leaf_counts.add(len(leaves))
                for leaf in leaves:
                    _, terms = dxil_ir.linearize(leaf)
                    records = []
                    for word_expr, weight_expr in terms:
                        try:
                            texel, add = dxil_ir.unpack_word(word_expr)
                        except RuntimeError:
                            return None, "input staged in groupshared"
                        in_x = dxil_ir.affine(texel[1][1], nthreads)
                        in_y = dxil_ir.affine(texel[1][2], nthreads)
                        if (in_x[1] != [stride_x * c for c in out_x[1]]
                                or in_y[1] != [stride_y * c for c in out_y[1]]):
                            return None, "input map is not a regular convolution"
                        adds.add(add & 0xFFFFFFFF)
                        records.append((
                            in_x[0] - stride_x * out_x[0],
                            in_y[0] - stride_y * out_y[0],
                            texel[2], weight_expr))
                    if not records:
                        return None, "empty accumulation"
                    min_x = min(r[0] for r in records)
                    min_y = min(r[1] for r in records)
                    phases.add((min_x, min_y))
                    for dx, dy, word, weight_expr in records:
                        tap_x, tap_y = dx - min_x, dy - min_y
                        if not (0 <= tap_x < 3 and 0 <= tap_y < 3):
                            return None, "tap outside the 3x3 window"
                        tap = tap_y * 3 + tap_x
                        widx = dxil_ir.affine(weight_expr[1], nthreads)[0] % FLOW_UNITS
                        wlane = weight_expr[2] if isinstance(weight_expr[2], int) else (
                            dxil_ir.evaluate(weight_expr[2],
                                             {"tig": (0, 0, 0), "gid": (0, 0, 0),
                                              "tid": (0, 0, 0)}))
                        expected_widx = group + 16 * tap + 4 * word
                        if widx != expected_widx or wlane != lane:
                            return None, "weight layout %d/%d != %d/%d" % (
                                widx, wlane, expected_widx, lane)

    if len(adds) != 1 or len(los) != 1 or len(his) != 1:
        return None, "mixed input bias or clamp bounds"
    add = next(iter(adds))
    if add == 0:
        pad_value = 0
    elif add == 0x80808080:
        pad_value = -128
    else:
        return None, "unsupported input bias 0x%08x" % add

    if stride_x == 2:
        if leaf_counts != {4} or phases != {(-1, -1), (-1, 0), (0, -1), (0, 0)}:
            return None, "downscale phases are not the four 2x2 windows"
        template = "flow-max4"
    else:
        if leaf_counts != {1} or phases != {(-1, -1)}:
            return None, "same-size window is not a centred 3x3"
        template = "flow-same"

    ordered = sorted(phases)
    pads = []
    for origin_x, origin_y in ordered:
        right = _flow_right_pad(in_w, width, stride_x, origin_x)
        bottom = _flow_right_pad(in_h, height, stride_y, origin_y)
        if right is None or bottom is None:
            return None, "no padding reproduces %dx%d" % (width, height)
        pads.append((-origin_x, -origin_y, right, bottom))

    weights = np.zeros((outs * 16, slices * 16, 3, 3), dtype=np.int8)
    for out_slice in range(outs):
        for in_slice in range(slices):
            for group in range(4):
                for lane in range(4):
                    out_channel = out_slice * 16 + group * 4 + lane
                    for tap in range(9):
                        dy, dx = divmod(tap, 3)
                        for word in range(4):
                            row = ((out_slice * slices + in_slice) * FLOW_UNITS
                                   + group + 16 * tap + 4 * word)
                            weights[out_channel,
                                    in_slice * 16 + word * 4:in_slice * 16 + word * 4 + 4,
                                    dy, dx] = wbuf[row, lane * 4:lane * 4 + 4]

    plan = {
        "template": template,
        "slices": slices,
        "outs": outs,
        "in_w": in_w,
        "in_h": in_h,
        "width": width,
        "height": height,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "phases": ordered,
        "pads": pads,
        "add": add,
        "pad_value": pad_value,
        "lo": next(iter(los)),
        "hi": next(iter(his)),
        "weights": weights,
        "scale": scale,
        "bias": bias,
        "src": t0.reshape(slices, in_h, in_w, 16),
        "expected": u0.view(np.int8).reshape(outs, height, width, 16),
        "stores": len(ir.stores),
        "leaves": next(iter(leaf_counts)),
    }
    return plan, None


def gemv_plan(run_dir, dispatch):
    """Derive the buffer GEMV as a gather, byte pack and one MatMulInteger.

    The packed activation is shared by every output slice: the same 3x3 window
    of t0/t4 feeds all weight matrices. `values` are the six saturated fp16
    channels of each 12-byte window; the two t4 bytes and the fill byte complete
    the 18 packed words. The plan carries gather indices, the out-of-range mask,
    and the dense `[72, slices*16]` weight matrix.
    """
    if not gemv_shaped(dispatch):
        return None, "not a buffer GEMV"
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "-131072" not in text or "131071" not in text:
        return None, "GEMV clamp literals not found"
    nthreads = dxil_ir.num_threads(path)
    if nthreads[2] != 1:
        return None, "GEMV thread depth is not 1"

    c = constants_of(dispatch)
    stride, xmax, ymax = c[0], c[2], c[3] + 1
    ow, oh = c[4], c[5]
    cscale, sbyte, kfill = c[9], c[10], c[11]
    s1, s2 = sbyte & 31, (sbyte >> 8) & 31
    r1 = 1 << ((sbyte + 31) & 31)
    r2 = 1 << (((sbyte >> 8) + 31) & 31)

    t0, _ = tensor(run_dir, dispatch, "t0")
    t1, _ = tensor(run_dir, dispatch, "t1")
    t2, _ = tensor(run_dir, dispatch, "t2")
    t3, _ = tensor(run_dir, dispatch, "t3")
    t4, _ = tensor(run_dir, dispatch, "t4")
    u0, u0r = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, t4, u0)):
        return None, "tensor files missing"
    if u0r.get("dimension") != "texture2d" or u0r.get("format") != 3:
        return None, "GEMV output is not a 16-byte-texel 2D array"
    slices, height, width = u0r["array"], u0r["height"], u0r["width"]
    if t2.size != slices * 4 * 16 or t3.size != t2.size:
        return None, "GEMV bias/activation table does not match %d slices" % slices
    if t1.size != slices * 4 * 18 * 16:
        return None, "GEMV weight size does not match the 8x18 layout"
    gx, gy, gz = dispatch["group"][0] * nthreads[0], \
        dispatch["group"][1] * nthreads[1], dispatch["group"][2]
    if gz != slices or gx < ow or gy < oh:
        return None, "GEMV grid does not cover the output tile"

    tx = np.arange(gx, dtype=np.int64)[:, None]
    ty = np.arange(gy, dtype=np.int64)[None, :]
    index_halves = np.zeros((gx, gy, len(GEMV_TAPS), 6), dtype=np.int64)
    index_t4 = np.zeros((gx, gy, len(GEMV_TAPS), 2), dtype=np.int64)
    in_range = np.zeros((gx, gy, len(GEMV_TAPS)), dtype=np.int32)
    for tap, (dx, dy) in enumerate(GEMV_TAPS):
        x, y = 2 * tx + dx, 2 * ty + dy
        ok = (x >= 0) & (x < xmax) & (y >= 1) & (y < ymax)
        lin = stride * np.clip(y, 1, ymax - 1) + np.clip(x, 0, xmax - 1)
        index_halves[:, :, tap, :] = 8 * lin[..., None] + np.arange(6)
        index_t4[:, :, tap, :] = 4 * lin[..., None] + np.arange(2)
        in_range[:, :, tap] = ok.astype(np.int32)
    if index_halves.max() >= t0.size // 2 or index_t4.max() >= t4.size:
        return None, "GEMV gather leaves the captured buffer"

    table = t1.view(np.int8).reshape(-1, 4)
    dense = np.zeros((72, slices * 16), dtype=np.int8)
    for channel in range(slices * 4):
        matrix = table[4 * 18 * channel:4 * 18 * (channel + 1)].reshape(18, 4, 4)
        for word in range(18):
            for lane in range(4):
                dense[word * 4:word * 4 + 4, channel * 4 + lane] = matrix[word, lane]
    bias = t2.view("<i4").reshape(-1, 4).reshape(-1).astype(np.int32)
    activation = t3.view("<i4").reshape(-1, 4).reshape(-1).astype(np.int32)

    plan = {
        "template": "buffer-gemv",
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "ow": ow,
        "oh": oh,
        "height": height,
        "width": width,
        "kfill": kfill & 0xFF,
        "cscale": int(cscale),
        "s1": s1,
        "s2": s2,
        "r1": r1,
        "r2": r2,
        "index_halves": index_halves.reshape(-1, 6),
        "index_t4": index_t4.reshape(-1, 2),
        "in_range": in_range.reshape(-1, 1),
        "weights": dense,
        "bias": bias,
        "activation": activation,
        "src0": t0.view("<f2"),
        "src1": t4.view(np.int8),
        "expected": u0.reshape(gz, height, width, 16),
    }
    return plan, None


def gemv_shaped(dispatch):
    """The buffer-domain GEMV: two byte-address SRVs plus weight/bias/activation
    tables and a multi-slice 16-byte-texel UAV, with twelve root constants."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if not {"t0", "t1", "t2", "t3", "t4", "u0"} <= set(slots):
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    if len(dispatch["constants"][0]["values"]) != 12:
        return False
    for slot in ("t0", "t1", "t2", "t3", "t4"):
        if slots[slot].get("dimension") != "buffer":
            return False
    u0 = slots["u0"]
    return (u0.get("dimension") == "texture2d" and u0.get("array", 1) > 1
            and u0.get("format") == 3)


def _sat8(value):
    """fptosi + IMax/IMin: truncate to -128..127, then keep the byte."""
    value = np.clip(value, -128, 127)
    return (value & 127) | ((value >> 31) << 7)


def _gemv_pack(low, values, kfill):
    """Pack one GEMV tap into the two uint32 words the shader consumes.

    `low` holds the two verbatim bytes from t4 and `values` the five saturated
    int8 features (the sixth half is unused). Every feature is masked before
    its shift: a negative feature is an int64 in [-128, -1], and an unmasked
    left shift would sign-extend into the neighbouring byte. Static content
    keeps the first feature non-negative, so only motion exposes that.
    """
    word0 = (low | ((values[:, 0] & 0xFF).astype(np.uint32) << 16)
             | ((values[:, 1] & 0xFF).astype(np.uint32) << 24))
    word1 = ((values[:, 2] & 0xFF)
             | ((values[:, 3] & 0xFF) << 8)
             | ((values[:, 4] & 0xFF) << 16)
             | ((kfill & 0xFF) << 24))
    return word0, word1.astype(np.uint32)


def reconstruct_gemv(run_dir, dispatch, capture=None):
    """Replay the buffer GEMV that feeds the flow family.

    Per output thread: pack a 3x3 window of 8-byte words (two bytes copied from
    t4, six saturated fp16 values from t0, the folded zero point in the last
    lane), run 18 packed dot4 steps per channel group, and requantise with the
    integer path. The clamp bounds are IR literals in this program; they are
    checked against its disassembly rather than assumed.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "-131072" not in text or "131071" not in text:
        return None, "GEMV clamp literals not found"
    nthreads = dxil_ir.num_threads(path)
    if nthreads[2] != 1:
        return None, "GEMV thread depth is not 1"

    c = constants_of(dispatch)
    stride, xmax, ymax = c[0], c[2], c[3] + 1
    ow, oh = c[4], c[5]
    cscale, sbyte, kfill = c[9], c[10], c[11]
    s1, s2 = sbyte & 31, (sbyte >> 8) & 31
    r1 = 1 << ((sbyte + 31) & 31)
    r2 = 1 << (((sbyte >> 8) + 31) & 31)

    t0, _ = tensor(run_dir, dispatch, "t0")
    t1, _ = tensor(run_dir, dispatch, "t1")
    t2, _ = tensor(run_dir, dispatch, "t2")
    t3, _ = tensor(run_dir, dispatch, "t3")
    t4, _ = tensor(run_dir, dispatch, "t4")
    u0, u0r = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, t4, u0)):
        return None, "tensor files missing"
    if u0r.get("dimension") != "texture2d" or u0r.get("format") != 3:
        return None, "GEMV output is not a 16-byte-texel 2D array"
    slices, height, width = u0r["array"], u0r["height"], u0r["width"]
    if t2.size != slices * 4 * 16 or t3.size != t2.size:
        return None, "GEMV bias/activation table does not match %d slices" % slices
    if t1.size % (slices * 4 * 18 * 16):
        return None, "GEMV weight size does not match the 8x18 layout"

    expected = u0.reshape(slices, height, width, 16)
    weights = t1.view("<u4").reshape(-1, 4)
    bias = t2.view("<i4").reshape(-1, 4)
    activation = t3.view("<i4").reshape(-1, 4)

    gx, gy, gz = dispatch["group"][0] * nthreads[0], \
        dispatch["group"][1] * nthreads[1], dispatch["group"][2]
    if gz != slices or gx < ow or gy < oh:
        return None, "GEMV grid does not cover the output tile"
    tx = np.arange(gx, dtype=np.int64)[:, None]
    ty = np.arange(gy, dtype=np.int64)[None, :]

    packed = np.zeros((gz, gx, gy, 18), dtype=np.uint32)
    for s in range(gz):
        for p, (dx, dy) in enumerate(GEMV_TAPS):
            x, y = 2 * tx + dx, 2 * ty + dy
            ok = (x >= 0) & (x < xmax) & (y >= 1) & (y < ymax)
            lin = (stride * np.clip(y, 1, ymax - 1) + np.clip(x, 0, xmax - 1)).ravel()
            offset = 16 * lin
            raw = t0[np.minimum(offset, t0.size - 16)[:, None] + np.arange(12)[None, :]]
            d0 = raw[:, 0:4].copy().view("<u4").ravel()
            d1 = raw[:, 4:8].copy().view("<u4").ravel()
            d2 = raw[:, 8:12].copy().view("<u4").ravel()
            halves = np.empty((offset.size, 6), dtype=np.uint16)
            halves[:, 0] = (d0 & 0xFFFF).astype(np.uint16)
            halves[:, 1] = (d0 >> 16).astype(np.uint16)
            halves[:, 2] = (d1 & 0xFFFF).astype(np.uint16)
            halves[:, 3] = (d1 >> 16).astype(np.uint16)
            halves[:, 4] = (d2 & 0xFFFF).astype(np.uint16)
            values = _sat8(np.trunc(halves.view(np.float16).astype(np.float32)).astype(np.int64))
            q = t4[np.minimum(4 * lin, t4.size - 2)[:, None] + np.arange(2)[None, :]]
            low = q[:, 0].astype(np.uint32) | (q[:, 1].astype(np.uint32) << 8)
            word0, word1 = _gemv_pack(low, values, kfill)
            ok = ok.ravel()
            packed[s, :, :, 2 * p] = np.where(ok, word0, 0).reshape(gx, gy)
            packed[s, :, :, 2 * p + 1] = np.where(ok, word1, 0).reshape(gx, gy)

    words = packed.view(np.int8).reshape(gz, gx, gy, 18, 4).astype(np.int64)
    table = weights.view(np.int8).reshape(-1, 4).astype(np.int64)
    acc = np.empty((gz, gx, gy, 4, 4), dtype=np.int64)
    for s in range(gz):
        for g in range(4):
            channel = 4 * s + g
            matrix = table[4 * 18 * channel:4 * 18 * (channel + 1)].reshape(18, 4, 4)
            acc[s, :, :, g, :] = np.einsum("xywb,wcb->xyc", words[s], matrix)
            acc[s, :, :, g, :] += bias[channel][None, None, :]

    clamped = np.clip(acc, -131072, 131071)
    positive, negative = np.maximum(clamped, 0), np.minimum(clamped, 0)
    act = np.stack([activation[4 * s:4 * s + 4] for s in range(gz)])
    value = (((positive * cscale + r1) >> s1)
             + ((negative * act[:, None, None] + r2) >> s2))
    out = np.clip(value, -128, 127).astype(np.int64) & 0xFF

    got = np.zeros((slices, height, width, 16), dtype=np.uint8)
    for s in range(gz):
        tile = out[s].transpose(1, 0, 2, 3).reshape(gy, gx, 16)
        got[s, 1:1 + oh, :ow] = tile[:oh, :ow]

    if capture is not None:
        capture.update({"u0": got, "expected": expected, "out": out,
                        "packed": packed, "words": words, "acc": acc,
                        "value": value})
    total = int(expected.size)
    mismatch = int((got != expected).sum())
    note = ""
    written = ow * oh * gz * 16
    if written != total:
        note = ("%d/%d bytes inside the %dx%d store tile; the rest of the %dx%d "
                "allocation is untouched (zero here)" % (written, total, ow, oh, width, height))
    dims = (xmax, ymax, width, height, slices, ow, oh)
    return (mismatch, total, dims, note), None


# The buffer-output template always advances the weight table by 20 dwords per
# reduction step; d22 only consumes the first 16 of those accumulators.
BUFFER_OUTPUT_WEIGHT_STEP = 20


def buffer_output_plan(run_dir, dispatch):
    """Derive the buffer-output GEMV as MatMulInteger plus a static scatter.

    Each output cell reduces `slices * words_per_texel` steps; a step reads one
    uint32 word from t0 and one from t1, so the packed input vector is a gather
    of four-byte groups. The weights form `[steps*4, accumulators]`, the seed is
    added, the integer requant is the usual clamp/scale/shift, and the byte
    layout is a fixed permutation of `(accumulator, ty, tx)`.
    """
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if set(slots) != {"t0", "t1", "t2", "t3", "u0"}:
        return None, "buffer-output slots missing"
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "dot4AddPacked" not in text or "rawBufferStore.i32" not in text:
        return None, "not a packed-GEMV buffer-output program"
    if "<2 x i32>, <2 x i32>, <2 x i32>, i32, i32, i32" not in text:
        return None, "unexpected root-constant layout"
    ir = dxil_ir.IR(path)
    unit, error = _buffer_output_unit(ir, text)
    if error:
        return None, error
    nacc = text.count("i8 15, i32 4") * 4
    if nacc not in (16, 20):
        return None, "unsupported accumulator count %d" % nacc
    if unit != (16 if nacc == 20 else 8):
        return None, "store unit %d does not match %d accumulators" % (unit, nacc)

    c = constants_of(dispatch)
    row_stride, row_guard, scale, shift = c[2], c[5], c[7], c[8]
    if row_stride <= 0 or row_guard <= 1:
        return None, "empty output grid"
    nthreads = dxil_ir.num_threads(path)
    gx = dispatch["group"][0] * nthreads[0]
    gy = dispatch["group"][1] * nthreads[1]
    if min(gx, gy) <= 0:
        return None, "empty grid"

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, r1 = tensor(run_dir, dispatch, "t1")
    t2, _ = tensor(run_dir, dispatch, "t2")
    t3, _ = tensor(run_dir, dispatch, "t3")
    u0, r4 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, u0)):
        return None, "tensor files missing"
    if r0.get("format") != r1.get("format"):
        return None, "input formats differ"
    texel = FORMAT_BYTES.get(r0.get("format"))
    if texel not in (8, 16):
        return None, "unsupported input format %d" % r0.get("format")
    wpc = texel // 4
    slices, h0, w0 = r0["array"], r0["height"], r0["width"]
    h1, w1 = r1["height"], r1["width"]
    if r1["array"] != slices:
        return None, "input slice counts differ"
    if t2.size % (4 * BUFFER_OUTPUT_WEIGHT_STEP):
        return None, "weight size is not a whole number of steps"
    trip = t2.size // (4 * BUFFER_OUTPUT_WEIGHT_STEP)
    if trip != slices * wpc:
        return None, "weight steps %d != %d slices x %d words" % (trip, slices, wpc)
    if t3.size < nacc * 4:
        return None, "seed buffer smaller than %d accumulators" % nacc
    if t0.size != slices * h0 * w0 * texel or t1.size != slices * h1 * w1 * texel:
        return None, "input extent does not match its dump"
    if gx > w0 or gy + 1 > h0 or gx > w1 or gy + 1 > h1:
        return None, "grid exceeds the input texture extent"

    weights = t2.view("<u4").reshape(trip, BUFFER_OUTPUT_WEIGHT_STEP)[:, :nacc]
    lanes = _s8_words(weights)
    dense = np.zeros((trip * 4, nacc), dtype=np.int8)
    for step in range(trip):
        for byte in range(4):
            dense[step * 4 + byte] = lanes[step, :, byte]
    dense = np.concatenate([dense, dense], axis=0)

    steps = np.arange(trip)
    z, word = steps // wpc, steps % wpc
    tx = np.arange(gx)
    ty = np.arange(gy)

    def gather_index(height, width):
        base = (((z[:, None, None] * height + (ty + 1)[None, :, None]) * width
                 + tx[None, None, :]) * texel + word[:, None, None] * 4)
        base = np.transpose(base, (1, 2, 0))          # [gy, gx, trip]
        return (base[..., None] + np.arange(4)).reshape(gx * gy, trip * 4)

    keep = (2 * ty) < (row_guard - 1)
    base = row_stride * ty[:, None] + tx[None, :]
    half, quarter = unit // 2, unit // 4
    scatter = np.zeros((gy, gx, nacc), dtype=np.int64)
    for k in range(min(nacc, 16)):
        scatter[:, :, k] = (unit * base + half * row_stride * ((k % 4) // 2)
                            + half * (k % 2) + k // 4)
    for k in range(16, nacc):
        scatter[:, :, k] = (unit * base + half * row_stride * ((k - 16) // 2)
                            + quarter + half * ((k - 16) % 2))
    keep3 = np.broadcast_to(keep[:, None, None], (gy, gx, nacc))
    if scatter[keep3].min() < 0 or scatter[keep3].max() >= u0.size:
        return None, "scatter index outside the output allocation"

    plan = {
        "template": "buffer-output",
        "nacc": nacc,
        "unit": unit,
        "gx": gx,
        "gy": gy,
        "trip": trip,
        "texel": texel,
        "scale": int(scale),
        "shift": int(shift),
        "src0": t0.view(np.int8),
        "src1": t1.view(np.int8),
        "index0": gather_index(h0, w0),
        "index1": gather_index(h1, w1),
        "weights": dense,
        "seed": t3.view("<i4")[:nacc].astype(np.int32),
        "scatter_index": scatter[keep3][:, None].astype(np.int64),
        "update_index": np.arange(gy * gx * nacc, dtype=np.int64).reshape(
            gy, gx, nacc)[keep3],
        "expected": u0,
    }
    return plan, None


def buffer_output_shaped(dispatch):
    """The buffer-output GEMV: two texture-array inputs, a weight table at t2,
    an accumulator seed at t3, and a byte-address output buffer at u0."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if set(slots) != {"t0", "t1", "t2", "t3", "u0"}:
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    if len(dispatch["constants"][0]["values"]) != 9:
        return False
    if slots["t0"].get("dimension") != "texture2d" or slots["t0"].get("array", 1) <= 1:
        return False
    if slots["t1"].get("dimension") != "texture2d":
        return False
    if slots["t1"].get("format") != slots["t0"].get("format"):
        return False
    if any(slots[k].get("dimension") != "buffer" for k in ("t2", "t3", "u0")):
        return False
    return FORMAT_BYTES.get(slots["t0"].get("format")) in (8, 16)


def _s8_words(words):
    """Unpack each uint32 word into its four little-endian int8 lanes."""
    words = np.asarray(words, dtype=np.uint32)
    lanes = np.empty(words.shape + (4,), dtype=np.int64)
    for i in range(4):
        byte = (words >> np.uint32(8 * i)) & np.uint32(0xFF)
        lanes[..., i] = (byte.astype(np.int64) ^ 128) - 128
    return lanes


def _buffer_output_unit(ir, text):
    """Read the output cell size from the first store's index expression.

    Both programs compute `index = unit * (c2*ty + tx)`, so the outer shift of
    the first rawBufferStore's index is the cell size (16 for d10, 8 for d22).
    """
    m = re.search(r"call void @dx\.op\.rawBufferStore\.i32\(([^)]*)\)", text)
    if not m:
        return None, "no rawBufferStore"
    args = dxil_ir.split_args(m.group(1))
    if len(args) < 3:
        return None, "unsupported store arity"
    expr = ir.expr(args[2])
    if expr[0] == "shl" and expr[2][0] == "c":
        return 1 << (expr[2][1] & 31), None
    return None, "store index is not a constant shift"


def reconstruct_buffer_output(run_dir, dispatch):
    """Replay one buffer-output GEMV from its disassembly and tensor dumps.

    Each thread owns one (ty, tx) output cell. The reduction runs over
    `slices * words_per_texel` steps; at each step it loads one uint32 word from
    t0 and one from t1 at the same (tx, ty+1, slice), and for every accumulator
    adds `dot4(t0_word, weight) + dot4(t1_word, weight)`. The seed is t3, the
    requant is the integer path, and the 16/20 results are packed into the byte
    layout the store indices describe (including d10's four single-byte tail
    stores). The replay compares the whole u0 allocation.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "dot4AddPacked" not in text or "rawBufferStore.i32" not in text:
        return None, "not a packed-GEMV buffer-output program"
    if "<2 x i32>, <2 x i32>, <2 x i32>, i32, i32, i32" not in text:
        return None, "unexpected root-constant layout"
    ir = dxil_ir.IR(path)
    unit, error = _buffer_output_unit(ir, text)
    if error:
        return None, error
    nacc = text.count("i8 15, i32 4") * 4
    if nacc not in (16, 20):
        return None, "unsupported accumulator count %d" % nacc
    if unit != (16 if nacc == 20 else 8):
        return None, "store unit %d does not match %d accumulators" % (unit, nacc)

    c = constants_of(dispatch)
    row_stride, row_guard, scale, shift = c[2], c[5], c[7], c[8]
    if row_stride <= 0 or row_guard <= 1:
        return None, "empty output grid"
    nthreads = dxil_ir.num_threads(path)
    gx = dispatch["group"][0] * nthreads[0]
    gy = dispatch["group"][1] * nthreads[1]
    if min(gx, gy) <= 0:
        return None, "empty grid"

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, r1 = tensor(run_dir, dispatch, "t1")
    t2, r2 = tensor(run_dir, dispatch, "t2")
    t3, r3 = tensor(run_dir, dispatch, "t3")
    u0, r4 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, u0)):
        return None, "tensor files missing"
    if r0.get("format") != r1.get("format"):
        return None, "input formats differ"
    texel_bytes = FORMAT_BYTES.get(r0.get("format"))
    if texel_bytes not in (8, 16):
        return None, "unsupported input format %d" % r0.get("format")
    wpc = texel_bytes // 4
    slices, height0, width0 = r0["array"], r0["height"], r0["width"]
    height1, width1 = r1["height"], r1["width"]
    if r1["array"] != slices:
        return None, "input slice counts differ"
    if t2.size % (4 * BUFFER_OUTPUT_WEIGHT_STEP):
        return None, "weight size is not a whole number of steps"
    trip = t2.size // (4 * BUFFER_OUTPUT_WEIGHT_STEP)
    if trip != slices * wpc:
        return None, "weight steps %d != %d slices x %d words" % (trip, slices, wpc)
    if t3.size < nacc * 4:
        return None, "seed buffer smaller than %d accumulators" % nacc
    if t0.size != slices * height0 * width0 * texel_bytes:
        return None, "t0 extent does not match its dump"
    if t1.size != slices * height1 * width1 * texel_bytes:
        return None, "t1 extent does not match its dump"
    if gx > width0 or gy + 1 > height0 or gx > width1 or gy + 1 > height1:
        return None, "grid exceeds the input texture extent"

    seed = t3.view("<i4")[:nacc].astype(np.int64)
    weights = t2.view("<u4").reshape(trip, BUFFER_OUTPUT_WEIGHT_STEP)[:, :nacc]
    source0 = t0.view("<u4").reshape(slices, height0, width0, wpc)
    source1 = t1.view("<u4").reshape(slices, height1, width1, wpc)
    steps = np.arange(trip)
    z, word = steps // wpc, steps % wpc
    tx = np.arange(gx)
    ty = np.arange(gy)
    at = (z[:, None, None], (ty + 1)[None, :, None], tx[None, None, :], word[:, None, None])
    a = _s8_words(source0[at])
    b = _s8_words(source1[at])
    w = _s8_words(weights)
    acc = seed[:, None, None] + np.einsum("jxyb,jkb->kxy", a, w) \
        + np.einsum("jxyb,jkb->kxy", b, w)

    rounding = 1 << ((shift + 31) & 31)
    q = np.clip((np.clip(acc, -131072, 131071) * scale + rounding) >> shift,
                -128, 127).astype(np.uint8)

    keep = (2 * ty) < (row_guard - 1)
    base = row_stride * ty[:, None] + tx[None, :]
    half, quarter = unit // 2, unit // 4
    out = np.zeros(u0.size, dtype=np.uint8)
    max_index = -1
    for k in range(min(nacc, 16)):
        index = unit * base + half * row_stride * ((k % 4) // 2) + half * (k % 2) + k // 4
        selected = index[keep]
        if selected.min() < 0 or selected.max() >= u0.size:
            return None, "store index outside the output allocation"
        out[selected] = q[k][keep]
        max_index = max(max_index, int(selected.max()))
    for k in range(16, nacc):
        index = (unit * base + half * row_stride * ((k - 16) // 2)
                 + quarter + half * ((k - 16) % 2))
        selected = index[keep]
        if selected.min() < 0 or selected.max() >= u0.size:
            return None, "store index outside the output allocation"
        out[selected] = q[k][keep]
        max_index = max(max_index, int(selected.max()))

    mismatch = int((out != u0).sum())
    written = max_index + 1
    dims = (width0, height0, slices, row_stride, int(keep.sum()), nacc, u0.size)
    note = ""
    if written < u0.size:
        note = ("%d/%d bytes written; the tail is untouched (zero here)"
                % (written, u0.size))
    return (mismatch, u0.size, dims, note), None


def splat_shaped(dispatch):
    """The motion-vector splat: exactly two texture SRVs (RG motion, R depth),
    one byte-address UAV and ten root constants. The two output planes are
    fixed byte offsets inside that one UAV, so there is no u1 binding."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if set(slots) != {"t0", "t1", "u0"}:
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    if len(dispatch["constants"][0]["values"]) != 10:
        return False
    return (slots["t0"].get("dimension") == "texture2d"
            and slots["t0"].get("format") == 34
            and slots["t1"].get("dimension") == "texture2d"
            and slots["t1"].get("format") == 41
            and slots["u0"].get("dimension") == "buffer")


def _splat_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _splat_s32(bits):
    return bits - 0x100000000 if bits >= 0x80000000 else bits


def _f32_to_half_bits(value):
    """Float32 -> fp16 bit pattern, following the splat's DXIL step by step.

    The program does not call f32tof16; it adds 0x1000 to the float bits, then
    assembles the half sign/exponent/mantissa, the subnormal shift and the
    inf/nan sentinel itself. Reproducing that chain keeps the replay faithful
    for denormals and signed zero, where a plain float16 cast can differ.
    """
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 4096) & 0xFFFFFFFF
    exponent = (rounded >> 23) & 0xFF
    mantissa = rounded & 0x7FFFFF
    sign = ((rounded >> 16) & 0x8000).astype(np.uint32)
    normal = np.where(exponent > 112,
                      (((exponent + 16) & 0x1F) << 10) | (mantissa >> 13),
                      np.uint64(0)).astype(np.uint32)
    subnormal = np.where((exponent - 102) < 11,
                         ((((mantissa + 8384512) >> ((125 - exponent) & 31)) + 1) >> 1),
                         np.uint64(0)).astype(np.uint32)
    special = np.where(exponent > 143, np.uint64(32767), np.uint64(0)).astype(np.uint32)
    return sign | normal | subnormal | special


def _splat_depth_key(depth):
    """The DXIL depth key: trunc((1-depth) * 32767), clamped 0..32767, << 16.

    `fptoui` is undefined for negative inputs. This run's depth is >= 1 and the
    captured keys show the hardware saturating those to 0, which is what the
    replay does; a wrong choice here shows up as a byte mismatch, not a pass.
    """
    value = np.float32(1.0) - np.ascontiguousarray(depth, dtype=np.float32)
    value = np.trunc(np.float32(value * np.float32(32767.0)))
    value = np.nan_to_num(value, nan=0.0, posinf=32767.0, neginf=0.0)
    return np.clip(value, 0.0, 32767.0).astype(np.uint32) << np.uint32(16)


def _splat_load(texture, src_w, src_h, off_x, off_y, view):
    """Sample a texture SRV at (offset + tid) for every source texel.

    Out-of-bounds textureLoad returns zero in DXIL; the replay zero-fills the
    same coordinates instead of clamping to the edge.
    """
    tex_w, tex_h = texture["width"], texture["height"]
    xs = off_x + np.arange(src_w, dtype=np.int64)[None, :]
    ys = off_y + np.arange(src_h, dtype=np.int64)[:, None]
    ok = (xs >= 0) & (xs < tex_w) & (ys >= 0) & (ys < tex_h)
    source = view(tex_h, tex_w)
    out = np.zeros((src_h, src_w) + source.shape[2:], dtype=source.dtype)
    out[ok] = source[np.clip(ys, 0, tex_h - 1), np.clip(xs, 0, tex_w - 1)][ok]
    return out


def splat_plan(run_dir, dispatch):
    """Load and prepare the motion-vector splat without scattering it.

    Returns the source textures, the offset-applied motion/depth windows, the
    destination coordinates, the packed keys and the captured allocation. The
    ONNX lowering consumes the same plan; `reconstruct_splat` only performs the
    atomic-max scatter.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "atomicrmw umax i32 addrspace(3)" not in text:
        return None, "no groupshared UMax"
    if "dx.op.atomicBinOp.i32" not in text or "i32 7" not in text:
        return None, "no global UMax atomic"
    if "<2 x i32>, <2 x i32>, <2 x float>, <2 x i32>, <2 x i32>" not in text:
        return None, "unexpected root-constant layout"

    c = constants_of(dispatch)
    src_w, src_h, dst_w, dst_h = c[0], c[1], c[2], c[3]
    scale_x, scale_y = _splat_f32(c[4]), _splat_f32(c[5])
    motion_x, motion_y = _splat_s32(c[6]), _splat_s32(c[7])
    depth_x, depth_y = _splat_s32(c[8]), _splat_s32(c[9])
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        return None, "empty source or destination"

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, r1 = tensor(run_dir, dispatch, "t1")
    u0, r2 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, u0)):
        return None, "tensor files missing"
    if r0.get("dimension") != "texture2d" or r0.get("format") != 34:
        return None, "motion SRV is not R16G16_FLOAT"
    if r1.get("dimension") != "texture2d" or r1.get("format") != 41:
        return None, "depth SRV is not R32_FLOAT"
    if r2.get("dimension") != "buffer" or u0.size % 4:
        return None, "output is not a byte-address buffer"

    tex_w, tex_h = r0.get("width"), r0.get("height")
    if t0.size != tex_w * tex_h * 4:
        return None, "motion texture is not R16G16 at its declared size"
    if t1.size != tex_w * tex_h * 4:
        return None, "depth texture is not R32 at its declared size"

    motion_tex = t0.view(np.float16).reshape(tex_h, tex_w, 2)
    depth_tex = t1.view(np.float32).reshape(tex_h, tex_w)
    motion = _splat_load(
        r0, src_w, src_h, motion_x, motion_y,
        lambda h, w: motion_tex)
    depth = _splat_load(
        r1, src_w, src_h, depth_x, depth_y,
        lambda h, w: depth_tex[..., None])[..., 0]

    finite = np.isfinite(motion[..., 0]) & np.isfinite(motion[..., 1])
    disp_x = np.where(finite, motion[..., 0] * np.float32(scale_x), np.float32(0.0))
    disp_y = np.where(finite, motion[..., 1] * np.float32(scale_y), np.float32(0.0))

    tid_x = np.arange(src_w, dtype=np.float32)[None, :]
    tid_y = np.arange(src_h, dtype=np.float32)[:, None]
    if src_w == dst_w and src_h == dst_h:
        # DXIL Round_ni(disp + 0.5): floor after the float32 addition.
        # Ties go toward +infinity, including negative half-integers.
        dst_x = np.arange(src_w, dtype=np.int64)[None, :] + np.floor(
            disp_x + np.float32(0.5)).astype(np.int64)
        dst_y = np.arange(src_h, dtype=np.int64)[:, None] + np.floor(
            disp_y + np.float32(0.5)).astype(np.int64)
    else:
        # The rescale branch uses Round_ne with no half-texel subtraction.
        # Preserve the shader's multiply-then-divide float32 operation order.
        dst_x = np.rint((tid_x + np.float32(0.5) + disp_x)
                       * np.float32(dst_w) / np.float32(src_w)).astype(np.int64)
        dst_y = np.rint((tid_y + np.float32(0.5) + disp_y)
                       * np.float32(dst_h) / np.float32(src_h)).astype(np.int64)

    depth_key = _splat_depth_key(depth)
    key_x = np.uint32(0x80000000) | depth_key | _f32_to_half_bits(disp_x)
    key_y = np.uint32(0x80000000) | depth_key | _f32_to_half_bits(disp_y)

    plane = dst_w * dst_h
    allocation = u0.view(np.uint32)
    if allocation.size < 2 * plane:
        return None, "allocation is smaller than the two output planes"
    return {
        "dispatch": dispatch["dispatch"],
        "src_w": src_w, "src_h": src_h, "dst_w": dst_w, "dst_h": dst_h,
        "tex_w": tex_w, "tex_h": tex_h,
        "offset_x": motion_x, "offset_y": motion_y,
        "depth_offset_x": depth_x, "depth_offset_y": depth_y,
        "scale_x": np.float32(scale_x), "scale_y": np.float32(scale_y),
        "motion_tex": motion_tex, "depth_tex": depth_tex,
        "motion": motion, "depth": depth,
        "finite": finite, "disp_x": disp_x, "disp_y": disp_y,
        "dst_x": dst_x, "dst_y": dst_y,
        "key_x": key_x, "key_y": key_y,
        "expected": u0, "allocation_words": int(allocation.size),
    }, None


def reconstruct_splat(run_dir, dispatch, capture=None):
    """Replay the motion-vector splat from its disassembly and tensor dumps.

    Every source pixel scatters, into a 2x2 destination footprint, one key per
    output plane:

        key = 0x80000000 | depth_key << 16 | fp16(displacement)

    The destination is `tid + round(mv + 0.5)` at the same size, or the
    half-texel-centre rescale `round((tid + 0.5 + mv) * dst / src)` otherwise.
    The program's groupshared 16x16 tile is an atomic-max optimisation; because
    UMax is order-independent and every key carries the valid bit, replaying the
    global scatter directly is equivalent and is what this function does.
    `capture`, when given, receives the replayed `u0` bytes for a host runtime.
    """
    plan, error = splat_plan(run_dir, dispatch)
    if error:
        return None, error
    dst_w, dst_h = plan["dst_w"], plan["dst_h"]
    plane = dst_w * dst_h
    got = np.zeros(plan["allocation_words"], dtype=np.uint32)
    for ox in (0, 1):
        for oy in (0, 1):
            xx = plan["dst_x"] + ox
            yy = plan["dst_y"] + oy
            ok = (plan["finite"] & (xx >= 0) & (xx < dst_w)
                  & (yy >= 0) & (yy < dst_h))
            np.maximum.at(got, yy[ok] * dst_w + xx[ok], plan["key_x"][ok])
            np.maximum.at(got, plane + yy[ok] * dst_w + xx[ok], plan["key_y"][ok])

    if capture is not None:
        capture["u0"] = got.view(np.uint8)
    allocation = plan["expected"].view(np.uint32)
    mismatch = int((got != allocation).sum())
    dims = (plan["src_w"], plan["src_h"], dst_w, dst_h, 2 * plane, allocation.size)
    note = ""
    if allocation.size > 2 * plane:
        note = ("%d uints written; the %d-uint tail is untouched (zero in this "
                "capture)" % (2 * plane, allocation.size - 2 * plane))
    return (mismatch, int(allocation.size), dims, note), None


def downsample_plan(run_dir, dispatch):
    """Derive the int8 pyramid downsample as static gathers and ordered adds.

    The fine plane is `trunc(sum(4x4 luma)/16 - 128)` per 8x8 thread group; the
    coarse plane folds each group's four 4x4 quadrants of those **sums** with
    `trunc(sum/256 - 128)`. Both sums keep the DXIL's x-then-y (fine) and
    y-then-x (coarse) addition order, which the plan pins as index tensors.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    for marker in ("; NumThreads=(8,8,1)",
                   "0x3FCB367A00000000", "0x3FE6E2EB20000000",
                   "0x3FB27BB300000000",
                   "6.250000e-02", "3.906250e-03", "-1.280000e+02"):
        if marker not in text:
            return None, "unexpected downsample program (%s)" % marker
    if text.count("call void @dx.op.textureStore.i32") != 2:
        return None, "unexpected downsample store count"

    c = constants_of(dispatch)
    x0, y0, w0, h0 = c[0], c[1], c[2], c[3]
    x1, y1, w1, h1 = c[4], c[5], c[6], c[7]
    off_x, off_y = c[8], c[9]
    if min(w0, h0, w1, h1) <= 0:
        return None, "empty source region"
    group = dispatch.get("group") or []
    if len(group) != 3 or group[2] != 1 or min(group[0], group[1]) <= 0:
        return None, "unexpected group size"
    group_w, group_h = group[0], group[1]
    fine_w, fine_h = group_w * 8, group_h * 8
    coarse_w, coarse_h = group_w * 2, group_h * 2

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, r1 = tensor(run_dir, dispatch, "t1")
    u0, r2 = tensor(run_dir, dispatch, "u0")
    u1, r3 = tensor(run_dir, dispatch, "u1")
    if any(x is None for x in (t0, t1, u0, u1)):
        return None, "tensor files missing"
    if any(r.get("dimension") != "texture2d" for r in (r0, r1, r2, r3)):
        return None, "unexpected resource dimension"
    if t0.size != r0["width"] * r0["height"] * 4:
        return None, "t0 is not BGRA8"
    if t1.size != r1["width"] * r1["height"] * 4:
        return None, "t1 is not BGRA8"
    if u0.size != r2["width"] * r2["height"] * 2:
        return None, "u0 is not two int8 channels"
    if u1.size != r3["width"] * r3["height"] * 2:
        return None, "u1 is not two int8 channels"

    def tap_grid(region, tex_w, tex_h):
        x, y, w, h = region
        gx = np.arange(fine_w, dtype=np.int64)[None, :]
        gy = np.arange(fine_h, dtype=np.int64)[:, None]
        index = np.zeros((fine_h, fine_w, 16), dtype=np.int64)
        mask = np.zeros((fine_h, fine_w, 16), dtype=np.float32)
        slot = 0
        for dx in range(4):
            for dy in range(4):
                sx = x + 4 * gx - off_x + dx
                sy = y + 4 * gy - off_y + dy
                ok = ((sx >= x) & (sx < x + w) & (sy >= y) & (sy < y + h))
                index[:, :, slot] = (np.clip(sy, 0, tex_h - 1) * tex_w
                                     + np.clip(sx, 0, tex_w - 1))
                mask[:, :, slot] = ok.astype(np.float32)
                slot += 1
        return index, mask

    index0, mask0 = tap_grid((x0, y0, w0, h0), r0["width"], r0["height"])
    index1, mask1 = tap_grid((x1, y1, w1, h1), r1["width"], r1["height"])

    fold = np.zeros((coarse_h, coarse_w, 16), dtype=np.int64)
    for cy in range(coarse_h):
        for cx in range(coarse_w):
            gy, gyq = cy // 2, cy & 1
            gx, gxq = cx // 2, cx & 1
            slot = 0
            for dy in range(4):
                for dx in range(4):
                    ty = 8 * gy + 4 * gyq + dy
                    tx = 8 * gx + 4 * gxq + dx
                    fold[cy, cx, slot] = ty * fine_w + tx
                    slot += 1

    plan = {
        "template": "downsample",
        "fine_w": fine_w,
        "fine_h": fine_h,
        "coarse_w": coarse_w,
        "coarse_h": coarse_h,
        "index0": index0,
        "mask0": mask0,
        "index1": index1,
        "mask1": mask1,
        "fold": fold,
        "src0": t0.reshape(r0["height"], r0["width"], 4),
        "src1": t1.reshape(r1["height"], r1["width"], 4),
        "expected0": u0,
        "expected1": u1,
        "alloc0": (r2["height"], r2["width"]),
        "alloc1": (r3["height"], r3["width"]),
    }
    return plan, None


def downsample_shaped(dispatch):
    """Int8 pyramid downsample: two BGRA8 SRVs and two int8 UAV planes."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if set(slots) != {"t0", "t1", "u0", "u1"}:
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    if len(dispatch["constants"][0]["values"]) != 10:
        return False
    return (all(r.get("dimension") == "texture2d" for r in slots.values())
            and slots["t0"].get("format") == 87
            and slots["t1"].get("format") == 87
            and slots["u0"].get("format") == 52
            and slots["u1"].get("format") == 52)


def _downsample_luma(raw, width, height):
    """BGRA8 UNORM load -> Rec.709 luma in 0..255, in the DXIL's operation order.

    The DXIL's `fmul fast` + `fadd fast` luma multiplies/adds contract to two
    FMAs: `fma(green, 0.7152, red*0.2126)` then `fma(blue, 0.0722, value)`.
    The red product is rounded to float32 first; each FMA rounds once through
    a float64 intermediate, matching the GPU (the non-FMA order differs by
    1 byte on the smooth scene at a trunc boundary).
    """
    channels = raw.view(np.uint8).reshape(height, width, 4).astype(np.float32)
    channels /= np.float32(255.0)
    red, green, blue = channels[..., 2], channels[..., 1], channels[..., 0]
    red_term = red * np.float32(0.2126)
    value = np.float32(np.float64(green) * np.float64(np.float32(0.7152))
                       + np.float64(red_term))
    value = np.float32(np.float64(blue) * np.float64(np.float32(0.0722))
                       + np.float64(value))
    return value * np.float32(255.0)


def _downsample_sum(luma, grid_w, grid_h, x0, y0, x1, y1, off_x, off_y):
    """Sum each cell's 4x4 source window, in the DXIL's x-then-y tap order.

    The window origin is `region + 4*cell - offset`, and a tap outside the
    region contributes zero (the shader's select), not a clamped edge value.
    """
    total = np.zeros((grid_h, grid_w), dtype=np.float32)
    tex_h, tex_w = luma.shape
    gx = np.arange(grid_w, dtype=np.int64)[None, :]
    gy = np.arange(grid_h, dtype=np.int64)[:, None]
    for dx in range(4):
        for dy in range(4):
            sx = x0 + 4 * gx - off_x + dx
            sy = y0 + 4 * gy - off_y + dy
            ok = ((sx >= x0) & (sx < x1) & (sy >= y0) & (sy < y1))
            value = luma[np.clip(sy, 0, tex_h - 1), np.clip(sx, 0, tex_w - 1)]
            total = total + np.where(ok, value, np.float32(0.0))
    return total


def _downsample_fold(block, group_w, group_h):
    """Second stage: fold each group's 4x4 quadrants of the 8x8 cells, then
    place one value per quadrant at `2*group + quadrant`."""
    out = np.zeros((group_h * 2, group_w * 2), dtype=np.float32)
    gx = np.arange(group_w, dtype=np.int64)[None, :]
    gy = np.arange(group_h, dtype=np.int64)[:, None]
    for tig_y in range(2):
        for tig_x in range(2):
            acc = np.zeros((group_h, group_w), dtype=np.float32)
            for dy in range(4):
                for dx in range(4):
                    ty = 8 * gy + 4 * tig_y + dy
                    tx = 8 * gx + 4 * tig_x + dx
                    acc = acc + block[ty, tx]
            quantized = np.trunc(acc * np.float32(1.0 / 256.0)
                                 + np.float32(-128.0)).astype(np.int64)
            out[tig_y::2, tig_x::2] = quantized
    return out


def reconstruct_downsample(run_dir, dispatch):
    """Replay the int8 pyramid downsample (dispatch 1).

    The first stage writes one 2-channel cell per thread: the mean of a 4x4
    Rec.709 luma window, quantised to `trunc(sum/16 - 128)`. The second stage
    folds each group's four 4x4 quadrants of those cells into a 2x2 coarser
    plane with `trunc(sum/256 - 128)`. Both planes are R8G8_SINT; only the
    logical region is written, and the allocation tails stay untouched (zero
    in this capture).
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    for marker in ("; NumThreads=(8,8,1)",
                   "0x3FCB367A00000000", "0x3FE6E2EB20000000",
                   "0x3FB27BB300000000",
                   "6.250000e-02", "3.906250e-03", "-1.280000e+02"):
        if marker not in text:
            return None, "unexpected downsample program (%s)" % marker
    if text.count("call void @dx.op.textureStore.i32") != 2:
        return None, "unexpected downsample store count"

    c = constants_of(dispatch)
    x0, y0, w0, h0 = c[0], c[1], c[2], c[3]
    x1, y1, w1, h1 = c[4], c[5], c[6], c[7]
    off_x, off_y = c[8], c[9]
    if min(w0, h0, w1, h1) <= 0:
        return None, "empty source region"

    group = dispatch.get("group") or []
    if len(group) != 3 or group[2] != 1 or min(group[0], group[1]) <= 0:
        return None, "unexpected group size"
    group_w, group_h = group[0], group[1]
    fine_w, fine_h = group_w * 8, group_h * 8
    coarse_w, coarse_h = group_w * 2, group_h * 2

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, r1 = tensor(run_dir, dispatch, "t1")
    u0, r2 = tensor(run_dir, dispatch, "u0")
    u1, r3 = tensor(run_dir, dispatch, "u1")
    if any(x is None for x in (t0, t1, u0, u1)):
        return None, "tensor files missing"
    if r0.get("dimension") != "texture2d" or r1.get("dimension") != "texture2d":
        return None, "colour inputs are not textures"
    if r2.get("dimension") != "texture2d" or r3.get("dimension") != "texture2d":
        return None, "int8 outputs are not textures"
    if t0.size != r0["width"] * r0["height"] * 4:
        return None, "t0 is not BGRA8"
    if t1.size != r1["width"] * r1["height"] * 4:
        return None, "t1 is not BGRA8"
    if u0.size != r2["width"] * r2["height"] * 2:
        return None, "u0 is not two int8 channels"
    if u1.size != r3["width"] * r3["height"] * 2:
        return None, "u1 is not two int8 channels"
    if r3["width"] < fine_w or r3["height"] < fine_h:
        return None, "u1 allocation is smaller than the logical plane"
    if r2["width"] < coarse_w or r2["height"] < coarse_h:
        return None, "u0 allocation is smaller than the logical plane"

    luma0 = _downsample_luma(t0, r0["width"], r0["height"])
    luma1 = _downsample_luma(t1, r1["width"], r1["height"])
    sum0 = _downsample_sum(luma0, fine_w, fine_h, x0, y0, x0 + w0, y0 + h0,
                          off_x, off_y)
    sum1 = _downsample_sum(luma1, fine_w, fine_h, x1, y1, x1 + w1, y1 + h1,
                          off_x, off_y)

    fine0 = np.trunc(sum0 * np.float32(1.0 / 16.0)
                     + np.float32(-128.0)).astype(np.int64)
    fine1 = np.trunc(sum1 * np.float32(1.0 / 16.0)
                     + np.float32(-128.0)).astype(np.int64)
    coarse0 = _downsample_fold(sum0, group_w, group_h)
    coarse1 = _downsample_fold(sum1, group_w, group_h)

    got1 = np.zeros((r3["height"], r3["width"], 2), dtype=np.int8)
    got1[:fine_h, :fine_w, 0] = fine0
    got1[:fine_h, :fine_w, 1] = fine1
    got0 = np.zeros((r2["height"], r2["width"], 2), dtype=np.int8)
    got0[:coarse_h, :coarse_w, 0] = coarse0
    got0[:coarse_h, :coarse_w, 1] = coarse1

    mismatch = int((got0.view(np.uint8).reshape(-1) != u0).sum())
    mismatch += int((got1.view(np.uint8).reshape(-1) != u1).sum())
    total = int(u0.size + u1.size)
    dims = (r0["width"], r0["height"], fine_w, fine_h, coarse_w, coarse_h)
    written = (fine_w * fine_h + coarse_w * coarse_h) * 2
    note = ""
    if written < total:
        note = ("%d/%d bytes written; the tails are untouched (zero here)"
                % (written, total))
    return (mismatch, total, dims, note), None


# d38 is byte-exact. The former 1-2 LSB edge residual came from two model
# errors that mostly cancelled: the packed dot is signed int8 (the weights were
# read as uint8), and each m1 channel uses its own overlay alpha (t1.a, t2.a,
# t3.a) rather than t1.a for all three. Fixing both makes every captured scene
# exact, so no tolerance is allowed.
FINAL_COMPOSITE_MAX_DIFF = 0


def final_composite_shaped(dispatch):
    """The final composite: five packed-GEMV channels, 18 root constants and
    the 13 SRV/UAV inputs that write the leaf frame."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if set(slots) != {"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
                      "t9", "t10", "t11", "t12", "u0"}:
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    if len(dispatch["constants"][0]["values"]) != 18:
        return False
    return (slots["t5"].get("dimension") == "buffer"
            and slots["t6"].get("dimension") == "buffer"
            and slots["t7"].get("dimension") == "buffer"
            and slots["u0"].get("dimension") == "texture2d")


def _composite_rgba(run_dir, dispatch, slot, width, height):
    data, resource = tensor(run_dir, dispatch, slot)
    if data is None or resource is None:
        return None
    if data.size != width * height * 4:
        return None
    value = data.reshape(height, width, 4).astype(np.float32)
    value /= np.float32(255.0)
    return value[..., 2], value[..., 1], value[..., 0], value[..., 3]


def composite_plan(run_dir, dispatch):
    """Derive the final composite as a 20-column packed dot and a pointwise mix.

    The feature texture is repeated 2x2, each pixel parity selects one of four
    weight rows per gate, and the five sigmoid values drive the DXIL's m0..m3
    blend. `mode` is a static root constant, so the plan records which overlay
    textures the graph needs.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "; NumThreads=(8,8,1)" not in text:
        return None, "unexpected thread group"
    if text.count("dx.op.dot4AddPacked.i32") < 20:
        return None, "unexpected packed-dot count"
    if text.count("dx.op.unary.f32(i32 21") != 5:
        return None, "unexpected gate count"

    c = constants_of(dispatch)
    if len(c) < 18:
        return None, "18 root constants were not captured"
    mode = c[14]

    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    u0, ru = tensor(run_dir, dispatch, "u0")
    t0, r0 = tensor(run_dir, dispatch, "t0")
    t5, _ = tensor(run_dir, dispatch, "t5")
    t6, _ = tensor(run_dir, dispatch, "t6")
    t7, _ = tensor(run_dir, dispatch, "t7")
    if any(x is None for x in (u0, t0, t5, t6, t7)):
        return None, "tensor files missing"
    width, height = ru["width"], ru["height"]
    if u0.size != width * height * 4:
        return None, "u0 is not BGRA8"
    if r0.get("format") != 3 or r0["width"] * 2 != width or r0["height"] * 2 != height:
        return None, "feature texture is not a half-resolution uint4 grid"
    if t0.size != r0["width"] * r0["height"] * 16:
        return None, "feature texture is not 16 bytes per texel"
    if t5.size != 20 * 16 or t6.size != 5 * 4 * 4 or t7.size != 5 * 4 * 4:
        return None, "unexpected weight/scale/bias sizes"

    needed = ["t1", "t2", "t3", "t11", "t12"]
    if mode == 2:
        needed += ["t9", "t10"]
    elif mode == 3:
        needed += ["t9"]
    elif mode in (4, 5):
        needed += ["t8", "t10"]
        if mode == 5:
            needed += ["t9"]
    overlays = {}
    for slot in needed:
        data, view = tensor(run_dir, dispatch, slot)
        if data is None or view.get("dimension") != "texture2d":
            return None, "composite input %s missing" % slot
        if data.size != width * height * 4:
            return None, "%s is not BGRA8" % slot
        overlays[slot] = data.reshape(height, width, 4)

    parity = ((np.arange(width)[None, :] & 1)
              | ((np.arange(height)[:, None] & 1) << 1)).astype(np.int32)
    parity = np.broadcast_to(parity, (height, width)).reshape(-1)
    select = np.zeros((height * width, 5), dtype=np.int64)
    for gate in range(5):
        select[:, gate] = gate * 4 + parity

    plan = {
        "template": "final-composite",
        "mode": int(mode),
        "width": width,
        "height": height,
        "feature_w": r0["width"],
        "feature_h": r0["height"],
        "feature": t0.reshape(r0["height"], r0["width"], 16),
        "weights": t5.view(np.int8).reshape(20, 16),
        "scales": t6.view("<f4").reshape(5, 4).reshape(-1).astype(np.float32),
        "biases": t7.view("<f4").reshape(5, 4).reshape(-1).astype(np.float32),
        "select": select,
        "overlays": overlays,
        "expected": u0.reshape(height, width, 4),
    }
    return plan, None


def _unorm8(value):
    """The texture unit's float -> UNORM8 conversion, computed exactly."""
    return np.floor(np.clip(value, 0, 1).astype(np.float64) * 255.0 + 0.5)


def _fma(a, b, c):
    """One-rounding float32 fused multiply-add.

    The DXIL marks the blend multiplies and adds `fast`, so the GPU contracts
    them into FMAs. A float64 product/sum followed by one float32 cast
    reproduces that rounding for the captured range.
    """
    return (np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
            + np.asarray(c, dtype=np.float64)).astype(np.float32)


def reconstruct_final_composite(run_dir, dispatch, capture=None):
    """Replay the final composite (`d38`) from its DXIL and tensor dumps.

    Each 2x2 output block shares a 16-byte feature vector. For pixel parity
    `p = (x&1) | ((y&1)<<1)`, five signed packed-int8 dots select the weight
    entries `4*k+p`; `t6`/`t7` hold the matching float scale/bias. The five
    sigmoid gates then drive the five-way composite below. The mode constant
    `c14` selects the final UI/leaf composition (modes 2..5 in the DXIL); this
    run uses mode 3. The blend contracts to one-rounding FMAs and the stored
    float is converted to UNORM8 exactly. The whole allocation is compared.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "; NumThreads=(8,8,1)" not in text:
        return None, "unexpected thread group"
    if text.count("dx.op.dot4AddPacked.i32") < 20:
        return None, "unexpected packed-dot count"
    if text.count("dx.op.unary.f32(i32 21") != 5:
        return None, "unexpected gate count"
    if "dx.op.textureStore.f32" not in text:
        return None, "no float texture store"

    c = constants_of(dispatch)
    if len(c) < 18:
        return None, "18 root constants were not captured"
    mode = c[14]

    u0, ru = tensor(run_dir, dispatch, "u0")
    t0, r0 = tensor(run_dir, dispatch, "t0")
    if u0 is None or t0 is None:
        return None, "tensor files missing"
    if ru.get("dimension") != "texture2d" or r0.get("dimension") != "texture2d":
        return None, "unexpected input/output dimension"
    width, height = ru["width"], ru["height"]
    if u0.size != width * height * 4:
        return None, "u0 is not BGRA8"
    if r0.get("format") != 3 or r0["width"] * 2 != width or r0["height"] * 2 != height:
        return None, "feature texture is not a half-resolution uint4 grid"
    if t0.size != r0["width"] * r0["height"] * 16:
        return None, "feature texture is not 16 bytes per texel"

    t5, r5 = tensor(run_dir, dispatch, "t5")
    t6, r6 = tensor(run_dir, dispatch, "t6")
    t7, r7 = tensor(run_dir, dispatch, "t7")
    if any(x is None for x in (t5, t6, t7)):
        return None, "weight/scale/bias buffers missing"
    if t5.size != 20 * 16 or t6.size != 5 * 4 * 4 or t7.size != 5 * 4 * 4:
        return None, "unexpected weight/scale/bias sizes"

    feat = t0.reshape(r0["height"], r0["width"], 16)
    feat = np.repeat(np.repeat(feat, 2, axis=0), 2, axis=1).astype(np.int32)
    weights = t5.view(np.int8).reshape(20, 16).astype(np.int32)
    scales = t6.view(np.float32).reshape(5, 4)
    biases = t7.view(np.float32).reshape(5, 4)

    ys = np.arange(height, dtype=np.int64)[:, None]
    xs = np.arange(width, dtype=np.int64)[None, :]
    parity = np.broadcast_to((xs & 1) | ((ys & 1) << 1), (height, width))

    acc = np.zeros((height, width, 5), dtype=np.int32)
    for k in range(5):
        for p in range(4):
            mask = parity == p
            acc[..., k][mask] = feat[mask] @ weights[4 * k + p]

    v = np.empty((height, width, 5), dtype=np.float32)
    for k in range(5):
        v[..., k] = (acc[..., k].astype(np.float32) * scales[k][parity]
                     + biases[k][parity])
    with np.errstate(over="ignore"):
        s = [np.float32(1.0) / (np.float32(1.0) + np.exp(-v[..., k]))
             for k in range(5)]

    a = _composite_rgba(run_dir, dispatch, "t1", width, height)
    b = _composite_rgba(run_dir, dispatch, "t2", width, height)
    cc = _composite_rgba(run_dir, dispatch, "t3", width, height)
    d = _composite_rgba(run_dir, dispatch, "t11", width, height)
    e = _composite_rgba(run_dir, dispatch, "t12", width, height)
    if any(x is None for x in (a, b, cc, d, e)):
        return None, "composite input tensor missing"

    ar, ag, ab, aa = a
    br, bg, bb, ba = b
    cr, cg, cb, ca = cc
    dr, dg, db, _ = d
    er, eg, eb, _ = e

    # `fmul fast` + `fadd fast` contract to one FMA in every blend step.
    m0 = [_fma(ac - bc, s[2], bc) for ac, bc in ((ar, br), (ag, bg), (ab, bb))]
    # The DXIL uses t1.a for R, t2.a for G and t3.a for B, not one shared alpha.
    m1 = [_fma(x - al, s[0], al - m0[i])
          for i, (x, al) in enumerate(zip((cr, cg, cb), (aa, ba, ca)))]
    m2 = [_fma(m1[i], s[1], m0[i]) for i in range(3)]
    m3 = [_fma(dc - ec, s[3], ec) for dc, ec in ((dr, er), (dg, eg), (db, eb))]
    m = [_fma(m2[i] - m3[i], s[4], m3[i]) for i in range(3)]

    if mode == 2:
        q = _composite_rgba(run_dir, dispatch, "t9", width, height)
        t10 = _composite_rgba(run_dir, dispatch, "t10", width, height)
        if q is None or t10 is None:
            return None, "mode 2 overlay missing"
        f = np.clip(q[3] * np.float32(10.0), 0, 1)
        w = f * f * (np.float32(3.0) - np.float32(2.0) * f)
        out = [m2[i] + (t10[i] - m2[i]) * w for i in range(3)]
    elif mode == 3:
        q = _composite_rgba(run_dir, dispatch, "t9", width, height)
        if q is None:
            return None, "mode 3 overlay missing"
        out = [_fma(m[i], np.float32(1.0) - q[3], q[i]) for i in range(3)]
    elif mode in (4, 5):
        t8 = _composite_rgba(run_dir, dispatch, "t8", width, height)
        t10 = _composite_rgba(run_dir, dispatch, "t10", width, height)
        q = _composite_rgba(run_dir, dispatch, "t9", width, height)
        if t8 is None or t10 is None or (mode == 5 and q is None):
            return None, "mode %d overlay missing" % mode
        if mode == 4:
            p = [t8[i] for i in range(3)]
        else:
            p = [q[i] + t8[i] * (np.float32(1.0) - q[3]) for i in range(3)]
        diff = [p[i] - t10[i] for i in range(3)]
        g = np.clip(np.sqrt(diff[0] * diff[0] + diff[1] * diff[1]
                            + diff[2] * diff[2]) * np.float32(20.0), 0, 1)
        w = g * g * (np.float32(3.0) - np.float32(2.0) * g)
        keep = np.float32(1.0) - w
        if mode == 4:
            # The d38 mode-4 branch uses the final full m (%223), not the
            # intermediate m2 (%205): t10 + keep * (m - t8).
            out = [t10[i] + (m[i] - t8[i]) * keep for i in range(3)]
        else:
            q2 = [q[i] + m2[i] * (np.float32(1.0) - q[3]) for i in range(3)]
            out = [t10[i] - keep * p[i] + keep * q2[i] for i in range(3)]
    else:
        out = list(m)

    # The DXIL writes the float and the texture unit converts it to UNORM8.
    # Compute that 8-bit rounding exactly: a float32 `*255 + 0.5` rounds
    # 241.49999 up to 242, while the hardware produces 241.
    got = np.stack([
        _unorm8(out[2]), _unorm8(out[1]), _unorm8(out[0]),
        np.full((height, width), 255.0),
    ], axis=-1).astype(np.int64)
    ref = u0.reshape(height, width, 4).astype(np.int64)
    if capture is not None:
        capture.update({"got": got, "ref": ref, "out": out, "acc": acc,
                        "v": v, "s": s, "a": a, "b": b, "cc": cc, "d": d,
                        "e": e, "m0": m0, "m1": m1, "m2": m2, "m3": m3,
                        "m": m, "mode": mode})
    diff = got - ref
    mismatch = int((diff != 0).sum())
    max_diff = int(np.abs(diff).max())
    pixels = int((diff != 0).any(axis=-1).sum())
    total = int(u0.size)
    dims = (width, height, 5, mode)
    note = ""
    if mismatch:
        note = "%d LSB max on %d pixel(s)" % (max_diff, pixels)
    return (mismatch, total, dims, note, max_diff, pixels), None


def _half_to_f32(values):
    return np.ascontiguousarray(values).view("<f2").astype(np.float32)


def _half_pixel_grid(dst_w, dst_h, scale_x, scale_y, ratio_x, ratio_y):
    """ONNX half-pixel source coordinates and the four bilinear weights.

    The shader computes `src = scale * (dst - (ratio-1)/2)`, clamps at zero and
    floors; the weights are the four products of the fractional distances. All
    operands stay float32, and the accumulation order is pinned in _bilinear.
    """
    y, x = np.mgrid[0:dst_h, 0:dst_w]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    ax = np.maximum(np.float32(scale_x)
                    * (x - np.float32(0.5) * (np.float32(ratio_x) - np.float32(1.0))),
                    np.float32(0.0))
    ay = np.maximum(np.float32(scale_y)
                    * (y - np.float32(0.5) * (np.float32(ratio_y) - np.float32(1.0))),
                    np.float32(0.0))
    ix = np.floor(ax).astype(np.int32)
    iy = np.floor(ay).astype(np.int32)
    w0 = (np.float32(iy + 1) - ay) * (np.float32(ix + 1) - ax)
    w1 = (np.float32(iy + 1) - ay) * (ax - np.float32(ix))
    w2 = (ay - np.float32(iy)) * (np.float32(ix + 1) - ax)
    w3 = (ay - np.float32(iy)) * (ax - np.float32(ix))
    return ix, iy, (w0, w1, w2, w3)


def _resample_corners(source, ix, iy, src_h, src_w):
    """Clamp the four bilinear taps to the logical source rectangle."""
    cy0 = np.clip(iy, 0, src_h - 1)
    cy1 = np.clip(iy + 1, 0, src_h - 1)
    cx0 = np.clip(ix, 0, src_w - 1)
    cx1 = np.clip(ix + 1, 0, src_w - 1)
    return [source[cy0, cx0], source[cy0, cx1], source[cy1, cx0], source[cy1, cx1]]


def _bilinear(corners, weights):
    """The exact DXIL accumulation order: (((c0*w0 + c1*w1) + c2*w2) + c3*w3)."""
    value = corners[0] * weights[0]
    value = corners[1] * weights[1] + value
    value = corners[2] * weights[2] + value
    return corners[3] * weights[3] + value


def _hex_double(text, literal):
    """Decode a DXIL double literal such as 0x3FA2A071C0000000 to float32."""
    if literal not in text:
        return None
    return np.float32(struct.unpack(">d", bytes.fromhex(literal[2:]))[0])


def resample_plan(run_dir, dispatch):
    """Derive a half-pixel bilinear resample as gather + baked weights.

    The source is a strided buffer of int8 or f16 cells; the four clamped taps
    and their float32 weights are static, so the plan carries one gather index
    tensor and four weight planes. d11/d12 scale channels 0..3; d23 fuses the
    f16 residual with a single-rounding multiply-add. The f16 output pack stays
    a harness step: ONNX `Cast` to float16 does not reproduce the shader's
    half-up bit chain on d12.
    """
    if not resample_shaped(dispatch):
        return None, "not a bilinear resample"
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    c = constants_of(dispatch)
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    nconst = len(dispatch["constants"][0]["values"]) if dispatch.get("constants") else 0
    src_w, src_h, dst_w, dst_h = c[2], c[3], c[4], c[7]
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        return None, "empty resample grid"

    if nconst == 14:
        # d12: f16 cells, channels 0..3 scaled by 4, rows start at 1.
        scale = (_splat_f32(c[10]), _splat_f32(c[11]))
        ratio = (_splat_f32(c[12]), _splat_f32(c[13]))
        gain = _splat_f32(c[12])
        channels, start_row, cell_bytes = 5, 1, 16
        source_f16, source_cell = True, 8
    elif "t1" in slots:
        # d23: int8 cells plus the f16 residual, single-rounding FMA.
        scale = (_splat_f32(c[8]), _splat_f32(c[9]))
        ratio = (_splat_f32(c[10]), _splat_f32(c[11]))
        gain = _hex_double(text, "0x3F9C01F4C0000000")
        if gain is None:
            return None, "K2 literal not found"
        channels, start_row, cell_bytes = 4, 0, 8
        source_f16, source_cell = False, 4
    else:
        # d11: int8 cells, channels 0..3 dequantised by K1.
        scale = (_splat_f32(c[8]), _splat_f32(c[9]))
        ratio = (_splat_f32(c[10]), _splat_f32(c[11]))
        gain = _hex_double(text, "0x3FA2A071C0000000")
        if gain is None:
            return None, "K1 literal not found"
        channels, start_row, cell_bytes = 5, 0, 16
        source_f16, source_cell = False, 8

    t0, r0 = tensor(run_dir, dispatch, "t0")
    u0, r4 = tensor(run_dir, dispatch, "u0")
    if t0 is None or u0 is None:
        return None, "tensor files missing"
    if r0.get("dimension") != "buffer" or r4.get("dimension") != "buffer":
        return None, "input or output is not a byte-address buffer"
    stride = c[0]
    if stride <= 0:
        return None, "empty source stride"
    source_units = t0.size // 2 if source_f16 else t0.size
    residual = "t1" in slots
    t1 = None
    if residual:
        t1, r1 = tensor(run_dir, dispatch, "t1")
        if t1 is None or r1.get("dimension") != "buffer":
            return None, "residual buffer missing"
        if t1.size % (dst_w * 16):
            return None, "residual buffer does not match the f16 row stride"
        residual_rows = t1.size // (dst_w * 16)
        if 1 + dst_h > residual_rows:
            return None, "residual rows exceed the buffer"

    ix, iy, weights = _half_pixel_grid(dst_w, dst_h, scale[0], scale[1],
                                       ratio[0], ratio[1])
    cx0 = np.clip(ix, 0, src_w - 1)
    cx1 = np.clip(ix + 1, 0, src_w - 1)
    cy0 = np.clip(iy, 0, src_h - 1)
    cy1 = np.clip(iy + 1, 0, src_h - 1)
    cell_units = source_cell
    corners = [(cy0, cx0), (cy0, cx1), (cy1, cx0), (cy1, cx1)]
    gather = np.zeros((dst_h, dst_w, 4, channels), dtype=np.int64)
    for index, (cy, cx) in enumerate(corners):
        for channel in range(channels):
            gather[:, :, index, channel] = (
                (cy.astype(np.int64) * stride + cx.astype(np.int64)) * cell_units
                + channel)
    if gather.max() >= source_units:
        return None, "resample gather leaves the source buffer"

    residual_index = None
    if residual:
        residual_index = np.zeros((dst_h, dst_w, channels), dtype=np.int64)
        for channel in range(channels):
            residual_index[:, :, channel] = (
                ((1 + np.arange(dst_h, dtype=np.int64)[:, None]) * dst_w
                 + np.arange(dst_w, dtype=np.int64)[None, :]) * 8 + channel)
        if residual_index.max() >= t1.size // 2:
            return None, "residual gather leaves the buffer"

    rows_alloc = u0.size // (dst_w * cell_bytes)
    if u0.size % (dst_w * cell_bytes) or start_row + dst_h > rows_alloc:
        return None, "output rows exceed the allocation"

    plan = {
        "template": "resample",
        "src_w": src_w,
        "src_h": src_h,
        "dst_w": dst_w,
        "dst_h": dst_h,
        "channels": channels,
        "start_row": start_row,
        "cell_bytes": cell_bytes,
        "rows_alloc": rows_alloc,
        "source_f16": source_f16,
        "residual": residual,
        "gain": np.float32(gain),
        "gather": gather,
        "weights": [np.asarray(w, dtype=np.float32)[:, :, None, None]
                    for w in weights],
        "gain_vector": np.asarray(
            [np.float32(gain)] * min(4, channels)
            + [np.float32(1.0)] * max(0, channels - 4),
            dtype=np.float32).reshape(1, 1, 1, channels),
        "residual_index": residual_index,
        "src": t0.view("<f2") if source_f16 else t0.view(np.int8),
        "res": None if t1 is None else t1.view("<f2"),
        "expected": u0,
    }
    return plan, None


def resample_shaped(dispatch):
    """Half-pixel bilinear resample: int8/f16 buffer in, packed f16 buffer out."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if set(slots) not in ({"t0", "u0"}, {"t0", "t1", "u0"}):
        return False
    if slots["t0"].get("dimension") != "buffer" or slots["u0"].get("dimension") != "buffer":
        return False
    if "t1" in slots and slots["t1"].get("dimension") != "buffer":
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    return len(dispatch["constants"][0]["values"]) in (13, 14)


def reconstruct_resample(run_dir, dispatch):
    """Replay the three half-pixel bilinear resamples (d11, d12, d23).

    d11 and d12 upsample a 5-channel int8/f16 buffer with the same half-pixel
    core (d11 also dequantises channels 0..3 with K1; d12 scales them by 4).
    d23 upsamples a 4-channel int8 buffer and adds the f16 residual from d12's
    output with a single-rounding FMA. The output cell size is 16 bytes for the
    two 5-channel programs and 8 bytes for d23.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    c = constants_of(dispatch)
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    nconst = len(dispatch["constants"][0]["values"]) if dispatch.get("constants") else 0
    src_w, src_h, dst_w, dst_h = c[2], c[3], c[4], c[7]
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        return None, "empty resample grid"

    if nconst == 14:
        # d12: 40x23 f16 -> 92x160 f16, channels 0..3 scaled by 4.
        scale = (_splat_f32(c[10]), _splat_f32(c[11]))
        ratio = (_splat_f32(c[12]), _splat_f32(c[13]))
        gain = _splat_f32(c[12])
        channels, start_row, cell_bytes = 5, 1, 16
    elif "t1" in slots:
        # d23: 80x46 int8 + f16 residual -> 92x160 f16.
        scale = (_splat_f32(c[8]), _splat_f32(c[9]))
        ratio = (_splat_f32(c[10]), _splat_f32(c[11]))
        gain = _hex_double(text, "0x3F9C01F4C0000000")
        if gain is None:
            return None, "K2 literal not found"
        channels, start_row, cell_bytes = 4, 0, 8
    else:
        # d11: 20x12 int8 -> 23x40 f16, channels 0..3 dequantised by K1.
        scale = (_splat_f32(c[8]), _splat_f32(c[9]))
        ratio = (_splat_f32(c[10]), _splat_f32(c[11]))
        gain = _hex_double(text, "0x3FA2A071C0000000")
        if gain is None:
            return None, "K1 literal not found"
        channels, start_row, cell_bytes = 5, 0, 16

    t0, r0 = tensor(run_dir, dispatch, "t0")
    u0, r4 = tensor(run_dir, dispatch, "u0")
    if t0 is None or u0 is None:
        return None, "tensor files missing"
    if r0.get("dimension") != "buffer" or r4.get("dimension") != "buffer":
        return None, "input or output is not a byte-address buffer"
    stride = c[0]
    if stride <= 0:
        return None, "empty source stride"
    if nconst == 14:
        # 16-byte f16 cells; eight half channels, the first five are live.
        if t0.size % (stride * 16):
            return None, "source size does not match the f16 cell layout"
        rows = t0.size // (stride * 16)
        raw = np.frombuffer(t0, dtype="<u2").reshape(rows, stride, 8)
        source = np.stack([_half_to_f32(raw[..., k]) for k in range(5)], axis=-1)
    elif "t1" in slots:
        # 4-byte int8 cells; only the logical src_h x stride block is read. The
        # required size comes from the dispatch constants, so it scales with
        # resolution; the former 14720-byte cap was a 640x360-only limit.
        need = src_h * stride * 4
        if t0.size < need:
            return None, "source size does not match the int8 cell layout"
        source = np.frombuffer(t0, dtype=np.int8)[:need].reshape(src_h, stride, 4)
        source = source.astype(np.float32)
    else:
        # 8-byte int8 cells; only the first five channels are live. The bound
        # buffer can be larger than the logical source (it is a shared
        # allocation), so read the logical prefix instead of requiring the
        # whole allocation to divide evenly like a standalone texture.
        need = src_h * stride * 8
        if t0.size < need:
            return None, "source size does not match the int8 cell layout"
        source = np.frombuffer(t0, dtype=np.int8)[:need].reshape(src_h, stride, 8)[:, :, :5]
        source = source.astype(np.float32)

    ix, iy, weights = _half_pixel_grid(dst_w, dst_h, scale[0], scale[1],
                                       ratio[0], ratio[1])
    corners = _resample_corners(source, ix, iy, src_h, src_w)
    if u0.size % (dst_w * cell_bytes):
        return None, "output size does not match the cell layout"
    rows_alloc = u0.size // (dst_w * cell_bytes)
    if start_row + dst_h > rows_alloc:
        return None, "output rows exceed the allocation"
    out = np.zeros(u0.size, dtype=np.uint8)
    texels = out.view("<u2").reshape(rows_alloc, dst_w, cell_bytes // 2)

    residual = None
    if "t1" in slots:
        t1, r1 = tensor(run_dir, dispatch, "t1")
        if t1 is None:
            return None, "tensor files missing"
        if r1.get("dimension") != "buffer" or t1.size % (dst_w * 16):
            return None, "residual buffer does not match the f16 row stride"
        res_rows = t1.size // (dst_w * 16)
        if 1 + dst_h > res_rows:
            return None, "residual rows exceed the buffer"
        raw = np.frombuffer(t1, dtype="<u2").reshape(res_rows, dst_w, 8)
        residual = np.stack([_half_to_f32(raw[1:1 + dst_h, :, k]) for k in range(channels)],
                            axis=-1)

    for k in range(channels):
        value = _bilinear([corner[..., k] for corner in corners], weights)
        if residual is None:
            if k < 4:
                value = value * gain
        else:
            value = (np.float64(value) * np.float64(gain)
                     + np.float64(residual[..., k])).astype(np.float32)
        texels[start_row:start_row + dst_h, :, k] = _f32_to_half_bits(value)

    mismatch = int((out != u0).sum())
    written = (start_row + dst_h) * dst_w * cell_bytes
    dims = (src_w, src_h, dst_w, dst_h, channels, u0.size)
    note = ""
    if written < u0.size:
        note = ("%d/%d bytes written; the tail is untouched (zero here)"
                % (written, u0.size))
    return (mismatch, u0.size, dims, note), None


def warp_plan(run_dir, dispatch):
    """Derive the flow-warp as two bilinear grid samples and a byte pack.

    The flow buffer supplies two 2-channel positions; each int8 texture channel
    is bilinearly sampled at its position with the DXIL's operation order, then
    truncated/clamped and packed as two bytes per texel at row+1.
    """
    if not warp_shaped(dispatch):
        return None, "not a flow warp"
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    c = constants_of(dispatch)
    src_w, src_h = c[0], c[1]
    if min(src_w, src_h) <= 0:
        return None, "empty warp grid"

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, r1 = tensor(run_dir, dispatch, "t1")
    u0, r2 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, u0)):
        return None, "tensor files missing"
    if r0.get("dimension") != "texture2d":
        return None, "warp source is not a texture"
    if r1.get("dimension") != "buffer" or r2.get("dimension") != "buffer":
        return None, "flow or output is not a byte-address buffer"
    tex_w, tex_h = r0["width"], r0["height"]
    if t0.size != tex_w * tex_h * 2:
        return None, "warp texture is not two int8 channels per texel"
    if t1.size % (src_w * 16):
        return None, "flow buffer does not match the f16 row stride"
    rows = t1.size // (src_w * 16)
    if 1 + src_h > rows:
        return None, "flow rows exceed the buffer"
    if u0.size % (src_w * 4):
        return None, "output size does not match the packed layout"
    out_rows = u0.size // (src_w * 4)
    if 1 + src_h > out_rows:
        return None, "output rows exceed the allocation"

    flow_index = np.zeros((src_h, src_w, 4), dtype=np.int64)
    for channel in range(4):
        flow_index[:, :, channel] = (
            ((1 + np.arange(src_h, dtype=np.int64)[:, None]) * src_w
             + np.arange(src_w, dtype=np.int64)[None, :]) * 8 + channel)

    plan = {
        "template": "flow-warp",
        "src_w": src_w,
        "src_h": src_h,
        "tex_w": tex_w,
        "tex_h": tex_h,
        "out_rows": out_rows,
        "flow_index": flow_index,
        "texture": t0.view(np.int8),
        "flow": t1.view("<f2"),
        "expected": u0,
    }
    return plan, None


def warp_shaped(dispatch):
    """Flow-warp: a 2-channel int8 texture sampled by the f16 flow buffer."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if set(slots) != {"t0", "t1", "u0"}:
        return False
    if slots["t0"].get("dimension") != "texture2d":
        return False
    if slots["t1"].get("dimension") != "buffer" or slots["u0"].get("dimension") != "buffer":
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    return len(dispatch["constants"][0]["values"]) == 4


def _warp_grid(texture, flow):
    """Bilinear grid-sample of one int8 channel at the f16 flow position."""
    src_h, src_w = texture.shape[:2]
    xi = np.arange(src_w, dtype=np.int32)[None, :]
    yi = np.arange(src_h, dtype=np.int32)[:, None]
    ix = np.floor(flow[0]).astype(np.int32)
    iy = np.floor(flow[1]).astype(np.int32)
    fx = flow[0] - np.floor(flow[0])
    fy = flow[1] - np.floor(flow[1])
    x0 = np.clip(ix + xi, 0, src_w - 1)
    x1 = np.clip(ix + xi + 1, 0, src_w - 1)
    y0 = np.clip(iy + yi, 0, src_h - 1)
    y1 = np.clip(iy + yi + 1, 0, src_h - 1)
    top = texture[y0, x1] * fx + texture[y0, x0] * (np.float32(1.0) - fx)
    return ((texture[y1, x0] * (np.float32(1.0) - fx) - top)
            + texture[y1, x1] * fx) * fy + top


def reconstruct_warp(run_dir, dispatch):
    """Replay the flow-warp stage (d13).

    d12's f16 output supplies two 2-channel flows; each channel of the int8
    texture (the downsample program's output) is bilinearly sampled at the
    corresponding flow position, truncated to int8 and packed two bytes per
    texel.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    c = constants_of(dispatch)
    src_w, src_h = c[0], c[1]
    if min(src_w, src_h) <= 0:
        return None, "empty warp grid"

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, r1 = tensor(run_dir, dispatch, "t1")
    u0, r2 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, u0)):
        return None, "tensor files missing"
    if r0.get("dimension") != "texture2d":
        return None, "warp source is not a texture"
    if r1.get("dimension") != "buffer" or r2.get("dimension") != "buffer":
        return None, "flow or output is not a byte-address buffer"
    tex_w, tex_h = r0["width"], r0["height"]
    if t0.size != tex_w * tex_h * 2:
        return None, "warp texture is not two int8 channels per texel"
    texture = np.frombuffer(t0, dtype=np.int8).reshape(tex_h, tex_w, 2)
    texture = texture[:src_h, :src_w].astype(np.float32)

    if t1.size % (src_w * 16):
        return None, "flow buffer does not match the f16 row stride"
    rows = t1.size // (src_w * 16)
    if 1 + src_h > rows:
        return None, "flow rows exceed the buffer"
    raw = np.frombuffer(t1, dtype="<u2").reshape(rows, src_w, 8)
    flow = [_half_to_f32(raw[1:1 + src_h, :, k]) for k in range(4)]

    low = np.clip(_warp_grid(texture[..., 0], (flow[0], flow[1])).astype(np.int32),
                  -128, 127)
    high = np.clip(_warp_grid(texture[..., 1], (flow[2], flow[3])).astype(np.int32),
                   -128, 127)

    if u0.size % (src_w * 4):
        return None, "output size does not match the packed layout"
    out_rows = u0.size // (src_w * 4)
    if 1 + src_h > out_rows:
        return None, "output rows exceed the allocation"
    out = np.zeros(u0.size, dtype=np.uint8)
    packed = out.view("<u4").reshape(out_rows, src_w)
    packed[1:1 + src_h] = ((low.astype("<u4") & np.uint32(0xFF))
                           | ((high.astype("<u4") & np.uint32(0xFF)) << np.uint32(8)))

    mismatch = int((out != u0).sum())
    written = (1 + src_h) * src_w * 4
    dims = (src_w, src_h, src_w, src_h, 2, u0.size)
    note = ""
    if written < u0.size:
        note = ("%d/%d bytes written; the tail is untouched (zero here)"
                % (written, u0.size))
    return (mismatch, u0.size, dims, note), None


# ---- warp + flow init (d25) -------------------------------------------------
# The only operator that samples the scene through the texture unit's bilinear
# filter. Float32 UVs and half-to-even 8-bit subtexel weights feed an integer
# four-tap sum. The measured fg78/RX6800XT sampler rounds that sum to 13
# fractional bits in byte units before feature quantisation. The current
# measured evidence is summarized in tools/render-validation/VALIDATION.md; the
# per-size colour-store bound lives in `d25_bound.py`.
WARP_INIT_MAX_DIFF = d25_bound.WARP_INIT_MAX_DIFF
WARP_INIT_MAX_FRACTION = d25_bound.WARP_INIT_MAX_FRACTION


def warp_init_shaped(dispatch):
    """Warp + flow init: the 51-constant program with four CBV tables."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    need = {"t0", "t8", "t2", "t3", "t4", "t5", "t6",
            "u0", "u1", "u2", "u3", "u4", "b2", "b3", "b4"}
    if not need <= set(slots):
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    values = dispatch["constants"][0]["values"]
    if len(values) < 51:
        return False
    if slots["t0"].get("dimension") != "buffer" or slots["t8"].get("dimension") != "buffer":
        return False
    if slots["b2"].get("dimension") != "buffer":
        return False
    if any(slots[k].get("dimension") != "texture2d"
           for k in ("t2", "t3", "t4", "t5", "t6", "u0", "u1", "u2", "u3", "u4")):
        return False
    return (slots["t2"]["width"] == 2 * slots["t4"]["width"]
            and slots["t2"]["height"] == 2 * slots["t4"]["height"]
            and slots["u0"]["width"] == slots["t4"]["width"]
            and slots["u0"]["height"] == slots["t4"]["height"])


def _warp_init_sample(tex, pos_x, pos_y):
    """Bilinear sample of a BGRA8 texture, in shader order and 0..255 units.

    The sampler is MIN_MAG_MIP_LINEAR with clamp addressing. This is the
    texture-unit model measured by the hardware sampler probe: the shader
    normalises the pixel position with a
    float32 reciprocal (`uv = float32(pos * float32(1 / size))`) and the texture
    unit multiplies that float32 uv by the size as an exact float64 product
    before the usual float64 `- 0.5` texel shift. The 8-bit subtexel weights are
    round-half-to-even (`np.rint`) on `frac * 256`, clipped to 0..256, and the
    four fixed-point products are summed, then rounded half-to-even from 16
    to 13 fractional bits. Returning exact byte units preserves the measured
    feature truncation; it does not claim bit-exact sampler float conversion.
    """
    h, w = tex.shape[:2]
    uv_x = (pos_x.astype(np.float32)
            * (np.float32(1.0) / np.float32(w))).astype(np.float32)
    uv_y = (pos_y.astype(np.float32)
            * (np.float32(1.0) / np.float32(h))).astype(np.float32)
    tex_x = uv_x.astype(np.float64) * float(w) - 0.5
    tex_y = uv_y.astype(np.float64) * float(h) - 0.5
    x0 = np.floor(tex_x)
    y0 = np.floor(tex_y)
    ix = np.clip(np.rint((tex_x - x0) * 256.0).astype(np.int64), 0, 256)
    iy = np.clip(np.rint((tex_y - y0) * 256.0).astype(np.int64), 0, 256)
    x0i = np.clip(x0.astype(np.int32), 0, w - 1)
    y0i = np.clip(y0.astype(np.int32), 0, h - 1)
    x1i = np.clip(x0i + 1, 0, w - 1)
    y1i = np.clip(y0i + 1, 0, h - 1)
    src = tex[..., [2, 1, 0, 3]].astype(np.int64)
    total = (src[y0i, x0i] * ((256 - iy) * (256 - ix))[..., None]
             + src[y0i, x1i] * ((256 - iy) * ix)[..., None]
             + src[y1i, x0i] * (iy * (256 - ix))[..., None]
             + src[y1i, x1i] * (iy * ix)[..., None])
    return np.rint(total.astype(np.float64) / 8.0) / 8192.0


def _warp_init_curve(units):
    """PQ inverse EOTF + 2.2 gamma, in 0..255 units (RC12.x == 1 only)."""
    x = np.clip(units.astype(np.float32) / np.float32(255.0), 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.exp(np.log(x) * np.float32(0.012683313339948654))
        a = np.maximum(t - np.float32(0.8359375), np.float32(0.0))
        b = np.float32(18.8515625) - np.float32(18.6875) * t
        u = np.where(b > 0, a / np.maximum(b, np.float32(1e-30)), np.float32(0.0))
        v = np.exp(np.log(np.maximum(u, np.float32(1e-30)))
                   * np.float32(2.853361129760742))
        y = np.minimum(v * np.float32(4.589730739593506), np.float32(1.0))
    return np.where(x > 0, np.maximum(y, np.float32(0.0)), np.float32(0.0)) * np.float32(255.0)


def _warp_init_inputs(run_dir, dispatch):
    """Decode and validate the captured inputs of the warp + flow-init stage.

    Shared by the numpy replay and `warp_init_plan`, so the ONNX lowering and the
    reference kernel cannot disagree about a shape, a constant or a slot role.
    """
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    c = constants_of(dispatch)
    tw, th = slots["t4"]["width"], slots["t4"]["height"]
    ow, oh = 2 * tw, 2 * th
    if (slots["t2"]["width"], slots["t2"]["height"]) != (ow, oh):
        return None, "source textures are not 2x the cell grid"

    t0b, r0 = tensor(run_dir, dispatch, "t0")
    t8b, r8 = tensor(run_dir, dispatch, "t8")
    t2b, r2 = tensor(run_dir, dispatch, "t2")
    t3b, r3 = tensor(run_dir, dispatch, "t3")
    t4b, r4 = tensor(run_dir, dispatch, "t4")
    t5b, r5 = tensor(run_dir, dispatch, "t5")
    t6b, r6 = tensor(run_dir, dispatch, "t6")
    b2b, _ = tensor(run_dir, dispatch, "b2")
    b3b, _ = tensor(run_dir, dispatch, "b3")
    b4b, _ = tensor(run_dir, dispatch, "b4")
    u0b, _ = tensor(run_dir, dispatch, "u0")
    u1b, _ = tensor(run_dir, dispatch, "u1")
    u2b, _ = tensor(run_dir, dispatch, "u2")
    u3b, _ = tensor(run_dir, dispatch, "u3")
    u4b, _ = tensor(run_dir, dispatch, "u4")
    arrays = (t0b, t8b, t2b, t3b, t4b, t5b, t6b, b2b, b3b, b4b, u0b, u1b, u2b, u3b, u4b)
    if any(x is None for x in arrays):
        return None, "tensor files missing"
    if (u0b.size != tw * th * 16 or u1b.size != ow * oh * 4
            or u2b.size != ow * oh * 4 or u3b.size != ow * oh * 4):
        return None, "output tensors do not match the dispatch's shapes"

    plane_w, plane_h = int(c[36]), int(c[37])
    stride = int(c[46])
    if min(plane_w, plane_h, stride) <= 0 or t0b.size % (plane_h * stride * 8):
        return None, "t0 is not a whole number of 160x92 fp16 planes"
    planes = t0b.size // (plane_h * stride * 8)
    t0 = _half_to_f32(np.frombuffer(t0b.tobytes(), dtype="<u2")
                      .reshape(planes, plane_h, stride, 4))[0]

    if t2b.size != ow * oh * 4 or t3b.size != ow * oh * 4:
        return None, "colour textures are not BGRA8 at 2x the cell grid"
    t2 = t2b.reshape(oh, ow, 4)
    t3 = t3b.reshape(oh, ow, 4)
    if t4b.size != tw * th * 4 or t5b.size != tw * th * 4:
        return None, "motion/depth inputs do not match the cell grid"
    t4 = _half_to_f32(np.frombuffer(t4b.tobytes(), dtype="<u2").reshape(th, tw, 2))
    t5 = np.frombuffer(t5b.tobytes(), dtype="<f4").reshape(th, tw)
    if t6b.size != 32 * 32:
        return None, "the gate LUT is not 32x32"
    t6 = t6b.reshape(32, 32).astype(np.float32) / np.float32(255.0)

    plane_words = tw * th
    if t8b.size < 8 * plane_words:
        return None, "t8 does not hold two key planes"
    t8 = np.frombuffer(t8b.tobytes(), dtype="<u4")
    plane0 = t8[:plane_words].reshape(th, tw)
    plane1 = t8[plane_words:2 * plane_words].reshape(th, tw)

    weights = np.frombuffer(b2b.tobytes(), dtype=np.int8)
    if weights.size < 80 * 16:
        return None, "the weight table is smaller than 80 uint4 rows"
    weights = weights[:80 * 16].reshape(80, 16)
    scale = np.frombuffer(b3b.tobytes(), dtype="<f4")[:16]
    bias = np.frombuffer(b4b.tobytes(), dtype="<f4")[:16]
    if scale.size != 16 or bias.size != 16:
        return None, "scale/bias tables are shorter than 16 floats"

    data = {
        "tw": tw, "th": th, "ow": ow, "oh": oh, "c": c,
        "plane_w": plane_w, "plane_h": plane_h, "stride": stride,
        "t0": t0, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6,
        "plane0": plane0, "plane1": plane1,
        "weights": weights, "scale": scale, "bias": bias,
        "u0": u0b, "u1": u1b, "u2": u2b, "u3": u3b, "u4": u4b,
    }
    return data, None


def _warp_init_motion_axis(x, span, plane, target):
    """`span*x` in float32: the shader's t0 coordinate for one axis.

    Equivalent formulations (`x*(plane-1)/(target-1)`, reciprocal multiply,
    double intermediate) can differ in the last bit, so this is a hook for
    `trace/fit_coords.py`.
    """
    return span * x.astype(np.float32)


def _warp_init_motion(t0, x, y, span_x, span_y, plane_w, plane_h,
                      target_w, target_h):
    """The shader's manual float32 bilinear of the t0 motion plane.

    `x`/`y` are integer pixel coordinates; the spans map the output grid onto
    the plane. The accumulation order is the DXIL's, left to right. It is a
    module-global so `trace/fit_coords.py` can compare rounding orders.
    """
    u = _warp_init_motion_axis(x, span_x, plane_w, target_w)
    v = _warp_init_motion_axis(y, span_y, plane_h, target_h)
    fu = np.floor(u)
    fv = np.floor(v)
    fx = (u - fu).astype(np.float32)
    fy = (v - fv).astype(np.float32)
    ex = np.clip(fu.astype(np.int32), 0, plane_w - 1)
    ey = np.clip(fv.astype(np.int32), 0, plane_h - 1)
    ex1 = np.clip(ex + 1, 0, plane_w - 1)
    ey1 = np.clip(ey + 1, 0, plane_h - 1)
    one = np.float32(1.0)
    w00 = (one - fy) * (one - fx)
    w10 = (one - fy) * fx
    w01 = fy * (one - fx)
    w11 = fy * fx
    return (((t0[ey, ex] * w00[..., None] + t0[ey, ex1] * w10[..., None])
             + t0[ey1, ex] * w01[..., None]) + t0[ey1, ex1] * w11[..., None])


def _warp_init_fallback(word0, gate_fallback):
    """The DXIL's full-fallback predicate for the flow source.

    The shader branches on the picked neighbour's packed word before decoding
    the flow: an empty word skips the gate and takes the `0.5*t4` fallback with
    flag 0. The gate can also select the same fallback, so the two conditions
    are ORed. On static content no picked word is empty.
    """
    return (word0 == 0) | gate_fallback


def _warp_init_add_scaled(base, scale, value, tag=""):
    """`base + scale*value` in float32.

    The DXIL may contract this to a single fused multiply-add, which rounds once
    instead of twice. It is a module-global so `trace/fit_coords.py` can test
    the fused form without touching the replay.
    """
    return base + scale * value


def reconstruct_warp_init(run_dir, dispatch, capture=None):
    """Replay the warp + flow-init stage (d25).

    Per 2x2 output cell it searches the splat depth plane for a neighbour, picks
    the t8 flow (or the t4 fallback when the picked word is empty or the t6 gate
    loses), warps t2/t3 four times,
    quantises 19 feature slots into int8 and runs the 76->16 GEMV from the CBVs.
    `capture`, when given, receives the replayed `u0..u3` arrays for a host
    runtime; `u4` is bound but never written and is not included.
    """
    data, error = _warp_init_inputs(run_dir, dispatch)
    if error:
        return None, error
    c = data["c"]
    tw, th, ow, oh = data["tw"], data["th"], data["ow"], data["oh"]
    plane_w, plane_h, stride = (data["plane_w"], data["plane_h"],
                                data["stride"])
    t0, t2, t3 = data["t0"], data["t2"], data["t3"]
    t4, t5, t6 = data["t4"], data["t5"], data["t6"]
    plane0, plane1 = data["plane0"], data["plane1"]
    weights, scale, bias = data["weights"], data["scale"], data["bias"]
    u0b, u1b, u2b, u3b, u4b = (data["u0"], data["u1"], data["u2"],
                              data["u3"], data["u4"])

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
    # The DXIL branches on the chosen neighbour's packed word before decoding
    # the flow: an empty word0 skips the gate and takes the 0.5*t4 fallback.
    # On static content no cell is empty, so this branch only fires on motion.
    f0 = _half_to_f32((word0 & np.uint32(0xFFFF)).astype("<u2"))
    f1 = np.where(word1 == 0, np.float32(0.0),
                  _half_to_f32((word1 & np.uint32(0xFFFF)).astype("<u2")))
    f0 = f0.astype(np.float32)
    f1 = f1.astype(np.float32)
    fallback_flow = t4[yp, xp]

    span_x = np.float32(c[36] - 1) / np.float32(c[38] - 1)
    span_y = np.float32(c[37] - 1) / np.float32(c[39] - 1)
    uv_a = (np.float32(_splat_f32(c[42])), np.float32(_splat_f32(c[43])))
    motion_scale = np.float32(_splat_f32(c[47]))
    ratio = np.float32(_splat_f32(c[50])) / (
        np.float32(1.0) - np.float32(_splat_f32(c[50])))
    lo_a = (np.float32(c[0]) + np.float32(0.5001),
            np.float32(c[1]) + np.float32(0.5001))
    hi_a = (np.float32(c[0] + c[2]) - np.float32(0.5001),
            np.float32(c[1] + c[3]) - np.float32(0.5001))
    lo_b = (np.float32(c[4]) + np.float32(0.5001),
            np.float32(c[5]) + np.float32(0.5001))
    hi_b = (np.float32(c[4] + c[6]) - np.float32(0.5001),
            np.float32(c[5] + c[7]) - np.float32(0.5001))

    def t0_bilinear(x, y):
        return _warp_init_motion(t0, x, y, span_x, span_y, plane_w, plane_h,
                                 int(c[38]), int(c[39]))

    def unorm8(units):
        return np.clip(np.floor(units + 0.5), 0, 255).astype(np.uint8)

    got0 = np.zeros((th, tw, 16), dtype=np.uint8)
    got1 = np.zeros((oh, ow, 4), dtype=np.uint8)
    got2 = np.zeros((oh, ow, 4), dtype=np.uint8)
    got3 = np.zeros((oh, ow, 4), dtype=np.uint8)
    features = []
    subs = ((0, 0), (1, 0), (0, 1), (1, 1))
    for ax, ay in subs:
        px = (2 * tx + ax).astype(np.float32) + np.float32(0.5)
        py = (2 * ty + ay).astype(np.float32) + np.float32(0.5)
        motion = t0_bilinear(2 * tx + ax, 2 * ty + ay)
        pos_a = (np.clip(_warp_init_add_scaled(px, motion_scale, motion[..., 0], "a"),
                         lo_a[0], hi_a[0]),
                 np.clip(_warp_init_add_scaled(py, motion_scale, motion[..., 1], "a"),
                         lo_a[1], hi_a[1]))
        pos_b = (np.clip(_warp_init_add_scaled(px, motion_scale, motion[..., 2], "b"),
                         lo_b[0], hi_b[0]),
                 np.clip(_warp_init_add_scaled(py, motion_scale, motion[..., 3], "b"),
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
        # An empty picked word is a full fallback regardless of the gate; the
        # two paths select the same 0.5*t4 flow, but both set flag 0.
        fallback = _warp_init_fallback(word0, confidence < gate)
        fx = np.where(fallback, np.float32(0.5) * fallback_flow[..., 0], f0)
        fy = np.where(fallback, np.float32(0.5) * fallback_flow[..., 1], f1)
        two_ratio = np.float32(2.0) * ratio
        pos_c = (np.clip(_warp_init_add_scaled(px, two_ratio, fx, "c"),
                         lo_a[0], hi_a[0]),
                 np.clip(_warp_init_add_scaled(py, two_ratio, fy, "c"),
                         lo_a[1], hi_a[1]))
        pos_d = (np.clip(_warp_init_add_scaled(px, np.float32(-2.0), fx, "d"),
                         lo_b[0], hi_b[0]),
                 np.clip(_warp_init_add_scaled(py, np.float32(-2.0), fy, "d"),
                         lo_b[1], hi_b[1]))

        centre2 = _warp_init_sample(t2, pos_c[0], pos_c[1])
        centre3 = _warp_init_sample(t3, pos_d[0], pos_d[1])
        warp2 = np.clip(_warp_init_sample(t2, pos_a[0], pos_a[1]), 0, 255)
        warp3 = np.clip(_warp_init_sample(t3, pos_b[0], pos_b[1]), 0, 255)
        xi = px.astype(np.int32)
        yi = py.astype(np.int32)
        self2 = t2[yi, xi][..., [2, 1, 0, 3]].astype(np.float32)
        self3 = t3[yi, xi][..., [2, 1, 0, 3]].astype(np.float32)
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
        got1[ay::2, ax::2] = unorm8(one)[..., [2, 1, 0, 3]]
        got2[ay::2, ax::2] = unorm8(two)[..., [2, 1, 0, 3]]
        got3[ay::2, ax::2] = unorm8(three)[..., [2, 1, 0, 3]]

    quantised = np.zeros((19, 4, th, tw), dtype=np.int64)
    for lane, feature in enumerate(features):
        units = feature.copy()
        units[..., 12] *= np.float32(255.0)
        if int(c[48]) == 1:
            for slot in range(19):
                if slot != 12:
                    units[..., slot] = _warp_init_curve(units[..., slot])
        quantised[:, lane] = np.moveaxis(
            np.clip(np.trunc(units) - 128, -128, 127).astype(np.int64), -1, 0)

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
        got0[..., channel] = np.clip(np.trunc(value.astype(np.float32)), 0, 127).astype(np.uint8)

    if capture is not None:
        capture.update({"u0": got0, "u1": got1, "u2": got2, "u3": got3})
    expected0 = u0b.reshape(th, tw, 16)
    expected1 = u1b.reshape(oh, ow, 4)
    expected2 = u2b.reshape(oh, ow, 4)
    expected3 = u3b.reshape(oh, ow, 4)
    deltas = (got0.astype(np.int16) - expected0.astype(np.int16),
              got1.astype(np.int16) - expected1.astype(np.int16),
              got2.astype(np.int16) - expected2.astype(np.int16),
              got3.astype(np.int16) - expected3.astype(np.int16))
    mismatch = int(sum((d != 0).sum() for d in deltas))
    total_bytes = int(sum(d.size for d in deltas))
    max_diff = int(max(np.abs(d).max() for d in deltas))
    untouched = "u4 is bound but never written (zero here); " if not u4b.any() else ""
    bad = [int((d != 0).sum()) for d in deltas]
    note = ("%su0 %d, u1 %d, u2 %d, u3 %d bytes differ, %d LSB max"
            % (untouched, bad[0], bad[1], bad[2], bad[3], max_diff))
    if bad[3] == 0 and max_diff <= 1:
        note += ("; u3 is exact and the remainder is the 8-bit subtexel "
                 "weight term in the motion-warp u1/u2 path")
    dims = (tw, th, ow, oh, 16, total_bytes)
    return (mismatch, total_bytes, dims, note, max_diff), None


def warp_init_plan(run_dir, dispatch):
    """Derive d25 as a standard-op graph: integer texel filter and int8 GEMV.

    Every runtime term stays a function of the captured inputs; the plan bakes
    only geometry (cell grids, clamp indices), the folded 76->16 weight matrix
    and the scalar constants the DXIL derives from the root CBV. The texture
    filter is an integer four-tap sum, so the lowering does not need a float
    sampler: `_warp_init_sample` and the graph run the same arithmetic.
    """
    if not warp_init_shaped(dispatch):
        return None, "not a warp + flow init"
    data, error = _warp_init_inputs(run_dir, dispatch)
    if error:
        return None, error
    c = data["c"]
    if int(c[48]) == 1:
        return None, "the PQ/gamma curve path (RC12.x == 1) is not lowered"
    tw, th, ow, oh = data["tw"], data["th"], data["ow"], data["oh"]
    plane_w, plane_h, stride = (data["plane_w"], data["plane_h"],
                                data["stride"])
    if stride < plane_w:
        return None, "t0 row stride is narrower than its logical plane"
    if int(c[38]) <= 1 or int(c[39]) <= 1:
        return None, "degenerate t0 resample span"

    # The packed flow is the low 16 bits of each t8 plane. The graph rebuilds
    # the fp16 value arithmetically (ONNX has no bitcast); exponent 31 has no
    # finite form and must not occur in the captured splat.
    plane0, plane1 = data["plane0"], data["plane1"]
    words = np.concatenate([plane0.reshape(-1), plane1.reshape(-1)])
    low = (words & np.uint32(0xFFFF)).astype(np.uint32)
    exponent = ((low >> np.uint32(10)) & np.uint32(0x1F)).astype(np.int32)
    live = (words != 0) & (low != 0)
    if np.any(exponent[live] == 31):
        return None, "the splat plane holds a non-finite fp16 flow"

    # A[cell, 4*group + lane] x B[4*group + lane, channel]. Rows 76..79 hold
    # the folded -128 group, so A carries four constant -128 columns.
    weights = data["weights"]
    dense = np.zeros((80, 16), dtype=np.int8)
    for channel in range(16):
        lane = (channel & 3) * 4
        for group in range(19):
            row = 4 * group + (channel >> 2)
            dense[4 * group:4 * group + 4, channel] = weights[row, lane:lane + 4]
        row = 76 + (channel >> 2)
        dense[76:80, channel] = weights[row, lane:lane + 4]

    tx = np.broadcast_to(np.arange(tw, dtype=np.int32)[None, :], (th, tw))
    ty = np.broadcast_to(np.arange(th, dtype=np.int32)[:, None], (th, tw))

    def cell_index(x, y):
        return (np.clip(y, 0, th - 1) * tw
                + np.clip(x, 0, tw - 1)).astype(np.int64)

    depth_index = {
        "A": cell_index(tx - 1, ty - 1), "B": cell_index(tx + 1, ty - 1),
        "C": cell_index(tx - 1, ty + 1), "D": cell_index(tx + 1, ty + 1),
        "centre": cell_index(tx, ty),
    }

    lanes = []
    for ax, ay in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi = (2 * tx + ax).astype(np.int64)
        yi = (2 * ty + ay).astype(np.int64)
        lanes.append({
            "offset_x": ax, "offset_y": ay,
            "self_index": (yi * ow + xi).astype(np.int64),
            "gate_index": ((yi & 31) * 32 + (xi & 31)).astype(np.int64),
        })

    ratio_in = np.float32(_splat_f32(c[50]))
    if np.float32(1.0) - ratio_in == np.float32(0.0):
        return None, "degenerate flow ratio"
    constants = {
        "span_x": np.float32(plane_w - 1) / np.float32(int(c[38]) - 1),
        "span_y": np.float32(plane_h - 1) / np.float32(int(c[39]) - 1),
        "uv_x": np.float32(_splat_f32(c[42])),
        "uv_y": np.float32(_splat_f32(c[43])),
        "motion_scale": np.float32(_splat_f32(c[47])),
        "ratio": ratio_in / (np.float32(1.0) - ratio_in),
        "lo_ax": np.float32(c[0]) + np.float32(0.5001),
        "lo_ay": np.float32(c[1]) + np.float32(0.5001),
        "hi_ax": np.float32(c[0] + c[2]) - np.float32(0.5001),
        "hi_ay": np.float32(c[1] + c[3]) - np.float32(0.5001),
        "lo_bx": np.float32(c[4]) + np.float32(0.5001),
        "lo_by": np.float32(c[5]) + np.float32(0.5001),
        "hi_bx": np.float32(c[4] + c[6]) - np.float32(0.5001),
        "hi_by": np.float32(c[5] + c[7]) - np.float32(0.5001),
    }

    plan = {
        "template": "warp + flow init",
        "tw": tw, "th": th, "ow": ow, "oh": oh,
        "plane_w": plane_w, "plane_h": plane_h, "stride": stride,
        "depth_index": depth_index,
        "lanes": lanes,
        "constants": constants,
        "t0": np.ascontiguousarray(data["t0"], dtype=np.float32).reshape(-1),
        "t2": np.ascontiguousarray(data["t2"], dtype=np.uint8).reshape(-1),
        "t3": np.ascontiguousarray(data["t3"], dtype=np.uint8).reshape(-1),
        "t4": np.ascontiguousarray(data["t4"], dtype=np.float32).reshape(-1),
        "t5": np.ascontiguousarray(data["t5"], dtype=np.float32).reshape(-1),
        "t6": np.ascontiguousarray(data["t6"], dtype=np.float32).reshape(-1),
        "t8": words[:2 * tw * th].astype(np.uint32),
        "weights": dense,
        "scale": np.ascontiguousarray(data["scale"], dtype=np.float32),
        "bias": np.ascontiguousarray(data["bias"], dtype=np.float32),
        "expected": [
            data["u0"].reshape(th, tw, 16),
            data["u1"].reshape(oh, ow, 4),
            data["u2"].reshape(oh, ow, 4),
            data["u3"].reshape(oh, ow, 4),
        ],
    }
    return plan, None


def upsample_plan(run_dir, dispatch):
    """Derive the 2x fixed-point bilinear staging and the 3x3 int8 conv.

    The groupshared tile is an implementation detail: for a global high-res
    coordinate `(X, Y)` the staged value is
    `trunc(sum_{dx,dy} wx*wy*low[(Y+1)/2+dy-1, (X+1)/2+dx-1] / 16)`, where
    `wx = 3 if dx == (X+1)&1 else 1`. That is a half-pixel 2x bilinear upsample
    with the separable integer kernel `[1,3,3,1]/16`, followed by the co-located
    high-res residual, a `0..127` clamp, and a 3x3 packed int8 conv. The plan
    therefore needs neither the tile shape nor the groupshared layout.
    """
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if not {"t0", "t1", "t2", "t3", "t4", "u0"} <= set(slots):
        return None, "upsample slots missing"
    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, _ = tensor(run_dir, dispatch, "t1")
    t2, _ = tensor(run_dir, dispatch, "t2")
    t3, _ = tensor(run_dir, dispatch, "t3")
    t4, r4 = tensor(run_dir, dispatch, "t4")
    u0, r5 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, t4, u0)):
        return None, "tensor files missing"
    if any(r.get("dimension") != "texture2d" for r in (r0, r4, r5)):
        return None, "input/output is not a texture2darray"
    if FORMAT_BYTES.get(r0.get("format")) != 16 or FORMAT_BYTES.get(r4.get("format")) != 16:
        return None, "staging tensors are not 16-byte texels"
    out_bytes = FORMAT_BYTES.get(r5.get("format"))
    if out_bytes not in (8, 16):
        return None, "unsupported output format %d" % r5.get("format")
    groups = out_bytes // 4
    if groups != 4:
        return None, "upsample template expects four output groups"
    slices, outs = r0["array"], r5["array"]
    if slices != outs:
        return None, "upsample slices %d != output slices %d" % (slices, outs)
    in_w, in_h = r0["width"], r0["height"]
    width, height = r5["width"], r5["height"]
    if (r4["width"], r4["height"]) != (width, height) or r4["array"] != outs:
        return None, "high-res input does not match the output"
    if in_w * 2 != width or in_h * 2 not in (height, height + 1):
        return None, "low-res input is not half the output extent"
    if t1.size != outs * slices * FLOW_UNITS * 16:
        return None, "weight size does not match %d x %d slices" % (outs, slices)
    if t2.size != outs * groups * 16 or t3.size != outs * groups * 16:
        return None, "scale/bias table does not match %d output slices" % outs

    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "dot4AddPacked" not in text or "addrspace(3)" not in text:
        return None, "not the groupshared staging template"

    low = t0.view(np.int8).reshape(slices, in_h, in_w, 16)
    high = t4.view(np.int8).reshape(slices, height, width, 16)
    weights = t1.view(np.int8).reshape(outs, slices, 36, groups, 4, 4)
    scale = np.ascontiguousarray(t2.view("<f4").reshape(outs, groups, 4),
                                 dtype=np.float32).reshape(-1)
    bias = np.ascontiguousarray(t3.view("<f4").reshape(outs, groups, 4),
                                dtype=np.float32).reshape(-1)
    expected = u0.view(np.int8).reshape(outs, height, width, 16)

    dense = np.zeros((outs * 16, slices * 16, 3, 3), dtype=np.int8)
    for out_slice in range(outs):
        for in_slice in range(slices):
            for tap in range(9):
                tap_y, tap_x = divmod(tap, 3)
                for word in range(4):
                    block = weights[out_slice, in_slice, tap * 4 + word]
                    for group in range(groups):
                        out_base = out_slice * 16 + group * 4
                        in_base = in_slice * 16 + word * 4
                        dense[out_base:out_base + 4, in_base:in_base + 4,
                              tap_y, tap_x] = block[group]

    plan = {
        "template": "upsample",
        "slices": slices,
        "outs": outs,
        "groups": groups,
        "in_w": in_w,
        "in_h": in_h,
        "width": width,
        "height": height,
        "low": low,
        "high": high,
        "weights": dense,
        "scale": scale,
        "bias": bias,
        "expected": expected,
    }
    return plan, None


def upsample_shaped(dispatch):
    """2x-upsample + int8 conv: low-res t0, same-size t4, and a high-res output."""
    slots = {r["slot"]: r for r in dispatch.get("resources", [])}
    if not {"t0", "t1", "t2", "t3", "t4", "u0"} <= set(slots):
        return False
    if len(dispatch.get("constants", [])) != 1:
        return False
    if len(dispatch["constants"][0]["values"]) != 4:
        return False
    if any(slots[k].get("dimension") != "buffer" for k in ("t1", "t2", "t3")):
        return False
    if any(slots[k].get("dimension") != "texture2d" for k in ("t0", "t4", "u0")):
        return False
    if FORMAT_BYTES.get(slots["u0"]["format"]) not in (8, 16):
        return False
    out = slots["u0"]
    if (slots["t4"]["width"], slots["t4"]["height"]) != (out["width"], out["height"]):
        return False
    if slots["t4"]["array"] != out["array"]:
        return False
    outs, slices = out["array"], slots["t0"]["array"]
    if slots["t1"].get("width", 0) != outs * slices * 144 * 16:
        return False
    if slots["t2"].get("width", 0) != outs * 64 or slots["t3"].get("width", 0) != outs * 64:
        return False
    c = dispatch["constants"][0]["values"]
    if (c[0], c[1]) != (out["width"], out["height"]):
        return False
    return (slots["t0"]["width"] * 2 == out["width"]
            and slots["t0"]["height"] * 2 in (out["height"], out["height"] + 1))


def _sdiv(value, divisor):
    """C-style signed division: truncate toward zero, not floor."""
    return np.where(value >= 0, value // divisor, -((-value) // divisor))


def _tile_shape(path, nthreads):
    """Solve the groupshared tile size for the per-thread output block."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    m = re.search(r"addrspace\(3\) global \[(\d+) x i32\]", text)
    if not m:
        return None, "no groupshared tile"
    texels = int(m.group(1)) // 4
    for bx in (1, 2, 3, 4):
        for by in (1, 2, 3, 4):
            if (bx * nthreads[0] + 2) * (by * nthreads[1] + 2) == texels:
                return (bx, by), None
    return None, "tile %d does not fit (bx*NTx+2)(by*NTy+2)" % texels


def reconstruct_upsample(run_dir, dispatch):
    """Replay one 2x-upsample + int8 conv dispatch.

    Staging bilinearly upsamples t0 into a groupshared tile (9/3/3/1, truncating
    divide by 16), adds the co-located t4 texel, clamps to 0..127 and zero-fills
    the tile border; the conv then runs the same packed int8 template over the
    tile. Every geometric constant comes from the DXIL and the allocations.
    """
    path = _disasm_path(run_dir, dispatch)
    if not os.path.exists(path):
        return None, "no disassembly"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "dot4AddPacked" not in text or "addrspace(3)" not in text:
        return None, "not an upsample program"

    t0, r0 = tensor(run_dir, dispatch, "t0")
    t1, _ = tensor(run_dir, dispatch, "t1")
    t2, _ = tensor(run_dir, dispatch, "t2")
    t3, _ = tensor(run_dir, dispatch, "t3")
    t4, r4 = tensor(run_dir, dispatch, "t4")
    u0, r5 = tensor(run_dir, dispatch, "u0")
    if any(x is None for x in (t0, t1, t2, t3, t4, u0)):
        return None, "tensor files missing"
    for resource in (r0, r4, r5):
        if resource.get("dimension") != "texture2d":
            return None, "input/output is not a texture2darray"
    out_bytes = FORMAT_BYTES.get(r5["format"])
    if out_bytes not in (8, 16):
        return None, "unsupported output format %d" % r5["format"]
    groups = out_bytes // 4
    slices, outs = r0["array"], r5["array"]
    in_w, in_h = r0["width"], r0["height"]
    width, height = r5["width"], r5["height"]
    if (r4["width"], r4["height"]) != (width, height):
        return None, "high-res input does not match the output extent"
    if r4["array"] != r5["array"]:
        return None, "high-res input does not match the output slices"
    if r0["width"] * 2 != width or r0["height"] * 2 not in (height, height + 1):
        return None, "low-res input is not half the output extent"

    nthreads = dxil_ir.num_threads(path)
    block, error = _tile_shape(path, nthreads)
    if error:
        return None, error
    bx, by = block
    span_x, span_y = bx * nthreads[0], by * nthreads[1]
    tile_w, tile_h = span_x + 2, span_y + 2
    grid = tuple(dispatch["group"][i] * nthreads[i] for i in range(3))
    groups_x, groups_y = grid[0] // nthreads[0], grid[1] // nthreads[1]
    if grid[2] != outs:
        return None, "grid.z %d != %d output slices" % (grid[2], outs)
    if t1.size != outs * slices * 144 * 16:
        return None, "weight size does not match %d x %d slices" % (outs, slices)
    if t2.size < outs * groups * 16 or t3.size < outs * groups * 16:
        return None, "scale/bias table smaller than %d slices" % outs

    low = t0.view(np.int8).reshape(slices, in_h, in_w, 4, 4).astype(np.int64)
    high = t4.view(np.int8).reshape(slices, height, width, 4, 4).astype(np.int64)
    weights = t1.view(np.int8).astype(np.int64).reshape(outs, slices, 36, groups, 4, 4)
    scale = t2.view(np.float32).reshape(outs, groups, 4)
    bias = t3.view(np.float32).reshape(outs, groups, 4)
    expected = u0.reshape(outs, height, width, groups, 4)

    columns = np.arange(tile_w)
    rows = np.arange(tile_h)
    u_, kx = columns >> 1, columns & 1
    v_, ky = rows >> 1, rows & 1
    group_x = np.arange(groups_x)
    group_y = np.arange(groups_y)
    x0 = (span_x // 2) * group_x[:, None] + u_[None, :] - 1
    y0 = (span_y // 2) * group_y[:, None] + v_[None, :] - 1
    hx = span_x * group_x[:, None] + columns[None, :] - 1
    hy = span_y * group_y[:, None] + rows[None, :] - 1

    staged = np.zeros((groups_x, groups_y, slices, tile_w, tile_h, 4, 4), dtype=np.int64)
    for si in range(slices):
        acc = np.zeros((groups_x, groups_y, tile_w, tile_h, 4, 4), dtype=np.int64)
        for dx in (0, 1):
            wx = np.where(dx == kx, 3, 1)
            for dy in (0, 1):
                wy = np.where(dy == ky, 3, 1)
                xs, ys = x0 + dx, y0 + dy
                ok = (((xs >= 0) & (xs < in_w))[:, None, :, None]
                      & ((ys >= 0) & (ys < in_h))[None, :, None, :])
                xc = np.clip(xs, 0, in_w - 1)
                yc = np.clip(ys, 0, in_h - 1)
                value = low[si][np.ix_(yc.ravel(), xc.ravel())].reshape(
                    groups_y, tile_h, groups_x, tile_w, 4, 4).transpose(2, 0, 3, 1, 4, 5)
                weight = wx[None, None, :, None] * wy[None, None, None, :]
                acc += value * (weight * ok)[..., None, None]
        acc = _sdiv(acc, 16)
        inside_x = (hx >= 0) & (hx < width)
        inside_y = (hy >= 0) & (hy < height)
        high_value = high[si][np.ix_(np.clip(hy, 0, height - 1).ravel(),
                                     np.clip(hx, 0, width - 1).ravel())].reshape(
            groups_y, tile_h, groups_x, tile_w, 4, 4).transpose(2, 0, 3, 1, 4, 5)
        inside = (inside_x[:, None, :, None] & inside_y[None, :, None, :])[..., None, None]
        staged[:, :, si] = np.where(inside, np.clip(acc + high_value, 0, 127), 0)

    out = np.zeros((outs, height, width, groups, 4), dtype=np.int64)
    for group_x_index in range(groups_x):
        for group_y_index in range(groups_y):
            for oz in range(outs):
                acc = np.zeros((span_y, span_x, groups, 4), dtype=np.int64)
                for si in range(slices):
                    tile = np.zeros((tile_w + 2, tile_h + 2, 4, 4), dtype=np.int64)
                    tile[0:tile_w, 0:tile_h] = staged[group_x_index, group_y_index, si]
                    for tap in range(9):
                        dx, dy = tap % 3, tap // 3
                        inp = tile[dx:dx + span_x, dy:dy + span_y]
                        for word in range(4):
                            acc += np.tensordot(
                                inp[:, :, word, :].transpose(1, 0, 2),
                                weights[oz, si, tap * 4 + word], axes=([2], [2]))
                value = (acc.astype(np.float32) * scale[oz][None, None]
                         + bias[oz][None, None])
                value = np.clip(np.trunc(value.astype(np.float64)), 0, 127).astype(np.int64)
                xs = span_x * group_x_index + np.arange(span_x)
                ys = span_y * group_y_index + np.arange(span_y)
                mx, my = xs < width, ys < height
                out[oz][ys[my][:, None], xs[mx][None, :]] = value[np.ix_(my, mx)]

    mismatch = int((out != expected).sum())
    dims = (in_w, in_h, width, height, slices, outs)
    return (mismatch, int(expected.size), dims, ""), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--dispatch", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    rows = read_events(args.run_dir)
    params = layer_parameters(rows)
    dispatches = [r for r in rows if r["type"] == "dispatch" and r["frame"] == args.frame]
    if args.dispatch:
        dispatches = [d for d in dispatches if d["dispatch"] == args.dispatch]
    elif not args.all:
        dispatches = dispatches[:1]

    failures = 0

    gemv = [d for d in dispatches if gemv_shaped(d)]
    print("%d dispatches, %d buffer GEMV" % (len(dispatches), len(gemv)))
    exact = 0
    for d in gemv:
        result, error = reconstruct_gemv(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d s%d -> out %dx%d s%d  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[4],
                 dims[2], dims[3], dims[4], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(gemv)))

    buffer_output = [d for d in dispatches if buffer_output_shaped(d)]
    print("%d dispatches, %d buffer-output" % (len(dispatches), len(buffer_output)))
    exact = 0
    for d in buffer_output:
        result, error = reconstruct_buffer_output(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d s%d -> %d cols x %d rows, %d ch  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[2],
                 dims[3], dims[4], dims[5], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(buffer_output)))

    resample = [d for d in dispatches if resample_shaped(d)]
    print("%d dispatches, %d bilinear resample" % (len(dispatches), len(resample)))
    exact = 0
    for d in resample:
        result, error = reconstruct_resample(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d -> %dx%d, %d ch  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[2], dims[3],
                 dims[4], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(resample)))

    warp = [d for d in dispatches if warp_shaped(d)]
    print("%d dispatches, %d flow warp" % (len(dispatches), len(warp)))
    exact = 0
    for d in warp:
        result, error = reconstruct_warp(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d -> %dx%d, %d ch  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[2], dims[3],
                 dims[4], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(warp)))

    candidates = [d for d in dispatches if conv_shaped(d)]
    print("%d dispatches, %d conv-shaped" % (len(dispatches), len(candidates)))
    exact = 0
    for d in candidates:
        result, error = reconstruct(args.run_dir, d, params.get(d["key"], {}))
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, template, blocks, groups, stride, phase = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ" % (mismatch, total)
        print("d%04d %s  %s blocks=%d groups=%d stride=%d phase=(%d,%d)  "
              "in %dx%d -> out %dx%d  %s"
              % (d["dispatch"], d["key"], template, blocks, groups, stride,
                 phase[0], phase[1], dims[0], dims[1], dims[2], dims[3], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(candidates)))

    upsample = [d for d in dispatches if upsample_shaped(d)]
    print("%d dispatches, %d upsample-shaped" % (len(dispatches), len(upsample)))
    exact = 0
    for d in upsample:
        result, error = reconstruct_upsample(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d s%d -> out %dx%d s%d  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[4],
                 dims[2], dims[3], dims[5], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(upsample)))

    flow = [d for d in dispatches if flow_shaped(d)]
    print("%d dispatches, %d flow-shaped" % (len(dispatches), len(flow)))
    exact = 0
    for d in flow:
        result, error = reconstruct_flow(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d s%d -> out %dx%d s%d  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[4],
                 dims[2], dims[3], dims[5], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(flow)))

    splat = [d for d in dispatches if splat_shaped(d)]
    print("%d dispatches, %d motion-vector splat" % (len(dispatches), len(splat)))
    exact = 0
    for d in splat:
        result, error = reconstruct_splat(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d uints differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d -> out %dx%d  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[2], dims[3], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(splat)))

    downsample = [d for d in dispatches if downsample_shaped(d)]
    print("%d dispatches, %d int8 downsample" % (len(dispatches), len(downsample)))
    exact = 0
    for d in downsample:
        result, error = reconstruct_downsample(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note = result
        status = "EXACT" if mismatch == 0 else "%d/%d bytes differ%s" % (
            mismatch, total, " (" + note + ")" if note else "")
        print("d%04d %s  in %dx%d -> %dx%d and %dx%d, 2 ch  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[2], dims[3],
                 dims[4], dims[5], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(downsample)))

    warp_init = [d for d in dispatches if warp_init_shaped(d)]
    content = d25_bound.run_content(args.run_dir)
    print("%d dispatches, %d warp + flow init" % (len(dispatches), len(warp_init)))
    exact = 0
    passes = 0
    for d in warp_init:
        result, error = reconstruct_warp_init(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note, max_diff = result
        tolerated, reason = d25_bound.evaluate(
            mismatch, total, max_diff, width=dims[2], height=dims[3],
            content=content)
        status = "EXACT"
        if mismatch:
            status = ("%d/%d bytes differ (%s)%s" % (
                mismatch, total, note,
                "" if tolerated else "  OUTSIDE 1-LSB BOUND: " + reason))
        print("d%04d %s  %dx%d cells -> %dx%d colour + 16-lane u0  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[2], dims[3], status))
        exact += mismatch == 0
        passes += tolerated
        failures += not tolerated
    print("%d/%d reproduced byte-exactly; %d/%d within the 1-LSB bound"
          % (exact, len(warp_init), passes, len(warp_init)))

    final = [d for d in dispatches if final_composite_shaped(d)]
    print("%d dispatches, %d final composite" % (len(dispatches), len(final)))
    exact = 0
    for d in final:
        result, error = reconstruct_final_composite(args.run_dir, d)
        if error:
            print("d%04d %s  FAILED: %s" % (d["dispatch"], d["key"], error))
            failures += 1
            continue
        mismatch, total, dims, note, max_diff, pixels = result
        status = "EXACT"
        if mismatch:
            status = "%d/%d bytes differ (%s)" % (mismatch, total, note)
        print("d%04d %s  in %dx%d -> mode %d, 5 gates  %s"
              % (d["dispatch"], d["key"], dims[0], dims[1], dims[3], status))
        exact += mismatch == 0
        failures += mismatch != 0
    print("%d/%d reproduced byte-exactly" % (exact, len(final)))
    return 1 if failures else 0
