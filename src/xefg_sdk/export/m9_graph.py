#!/usr/bin/env python
"""M9: execute the complete 38-node XeFG graph.

All 38 dispatches are lowered to ONNX. In `host` mode every node runs as its
ONNX graph, including `d24` and `d25`; in `captured` mode the recorded `d25`
output is copied instead, which is the wiring proof. Every consumer input is
resolved from the trace's resource identity and fed from the producer's
in-memory output; captured tensors are the fallback for graph inputs.
Subresource views are translated before the hand-off.

    python -I trace/m9_graph.py <run> --mode captured --report m9-captured.json
    python -I trace/m9_graph.py <run> --mode host --report m9-host.json

`captured` is the byte-exact proof of the assembly (37 computed + 1 copied).
`host` runs the `d25` graph too and shows its known 1-LSB capture propagation.
"""
# Internal legacy research module; not a supported CLI entry point.
import numpy as np

from . import d25_bound, recon

# The ONNX runtime and the exporter are imported lazily so the pure helpers in
# this module (wiring/count/bound checks) stay importable without onnxruntime.


# `d25` is lowered to ONNX, but captured mode copies its recorded output so the
# wiring proof stays byte-exact; host mode runs the graph and carries the known
# 1-LSB texture-filter residual instead.
CUSTOM = {25: "warp + flow init"}
FAMILIES = (
    ("conv", recon.conv_shaped),
    ("flow conv", recon.flow_shaped),
    ("upsample", recon.upsample_shaped),
    ("GEMV", recon.gemv_shaped),
    ("buffer output", recon.buffer_output_shaped),
    ("resample", recon.resample_shaped),
    ("downsample", recon.downsample_shaped),
    ("warp", recon.warp_shaped),
    ("splat", recon.splat_shaped),
    ("final composite", recon.final_composite_shaped),
)

# Host-mode calibration per output size. d25's 1-LSB residual and its
# propagation through the flow pyramid scale with the allocation, so the
# expected (max LSB, mismatched bytes) pairs are pinned per resolution. A size
# that is not listed fails the gate instead of borrowing another size's numbers.
M9_CALIBRATION = {
    (640, 360): {
        "d25": d25_bound.WARP_INIT_CALIBRATION[(640, 360)]["aggregate"],
        "resolution": (640, 360),
        "d38": (0, 0),
        "nodes": {
            26: (0, 0), 27: (0, 0), 28: (0, 0), 29: (0, 0),
            30: (0, 0), 31: (0, 0), 32: (0, 0), 33: (0, 0),
            34: (0, 0), 35: (0, 0), 36: (0, 0), 37: (0, 0),
        },
    },
    (1280, 720): {
        "d25": d25_bound.WARP_INIT_CALIBRATION[(1280, 720)]["aggregate"],
        "resolution": (1280, 720),
        "d38": (0, 0),
        "nodes": {
            26: (0, 0), 27: (0, 0), 28: (0, 0), 29: (0, 0),
            30: (0, 0), 31: (0, 0), 32: (0, 0), 33: (0, 0),
            34: (0, 0), 35: (0, 0), 36: (0, 0), 37: (0, 0),
        },
    },
    (1920, 1080): {
        "d25": d25_bound.WARP_INIT_CALIBRATION[(1920, 1080)]["aggregate"],
        "resolution": (1920, 1080),
        "d38": (0, 0),
        "nodes": {
            26: (1, 6), 27: (1, 13), 28: (1, 16), 29: (1, 24),
            30: (1, 112), 31: (2, 1291), 32: (6, 3812),
            33: (4, 9601), 34: (17, 17659), 35: (9, 32460),
            36: (41, 76613), 37: (45, 194192),
        },
    },
    (2560, 1440): {
        "d25": d25_bound.WARP_INIT_CALIBRATION[(2560, 1440)]["aggregate"],
        "resolution": (2560, 1440),
        "d38": (0, 0),
        "nodes": {
            26: (1, 5), 27: (1, 13), 28: (1, 22), 29: (1, 27),
            30: (2, 124), 31: (2, 1390), 32: (7, 4176),
            33: (5, 11049), 34: (12, 17540), 35: (12, 31576),
            36: (37, 89328), 37: (43, 200384),
        },
    },
}
M9_DEFAULT_CALIBRATION = M9_CALIBRATION[(640, 360)]

# Motion content has its own per-node table. The 640x360 motion capture is the
# only motion run: d25's `u0` is exact after the empty-word branch, so every
# downstream node is expected byte-exact. The M9 motion run verifies these
# zeros; a non-zero node would mean something other than d25 differs.
M9_MOTION_CALIBRATION = {
    (640, 360): {
        "d25": d25_bound.WARP_INIT_MOTION_CALIBRATION[(640, 360)]["aggregate"],
        "resolution": (640, 360),
        # d38 is exact when its inputs are captured. In the assembled chain it
        # consumes the replayed d25 u1/u2, whose 1-LSB colour residual reaches
        # 12 final bytes at 1 LSB on motion (the static scene absorbs all of
        # it). The per-dispatch d38 replay stays byte-exact.
        "d38": (1, 12),
        "nodes": {
            26: (0, 0), 27: (0, 0), 28: (0, 0), 29: (0, 0),
            30: (0, 0), 31: (0, 0), 32: (0, 0), 33: (0, 0),
            34: (0, 0), 35: (0, 0), 36: (0, 0), 37: (0, 0),
        },
    },
    (1920, 1080): {
        "d25": d25_bound.WARP_INIT_MOTION_CALIBRATION[(1920, 1080)]["aggregate"],
        "resolution": (1920, 1080),
        # d25.u0 has one byte of error at 1080/1440 motion, so the flow
        # pyramid propagates it; these are the measured M9 host bounds.
        "d38": (27, 3805),
        "nodes": {
            26: (1, 2), 27: (1, 2), 28: (1, 2), 29: (1, 6),
            30: (1, 21), 31: (1, 216), 32: (4, 1114),
            33: (2, 3405), 34: (9, 7638), 35: (7, 12354),
            36: (16, 33388), 37: (25, 93901),
        },
    },
    (2560, 1440): {
        "d25": d25_bound.WARP_INIT_MOTION_CALIBRATION[(2560, 1440)]["aggregate"],
        "resolution": (2560, 1440),
        "d38": (23, 3545),
        "nodes": {
            26: (1, 17), 27: (1, 28), 28: (1, 58), 29: (1, 65),
            30: (2, 237), 31: (3, 2378), 32: (7, 7249),
            33: (4, 18732), 34: (16, 31777), 35: (10, 59996),
            36: (49, 152203), 37: (34, 369399),
        },
    },
}
M9_MOTION_DEFAULT_CALIBRATION = M9_MOTION_CALIBRATION[(640, 360)]

# Wiring pins. The 47 resource edges collapse to 58 slot-level hand-offs because
# a producer can feed more than one consumer slot. Both counts are asserted.
M9_RESOURCE_EDGES = 47
M9_SLOT_HANDOFFS = 58

# Structural gate for a resolution with no pinned M9_CALIBRATION entry. It is
# deliberately weaker than the calibrated per-node table but proves the same
# contract: d25 stays inside 1 LSB and the fraction envelope, every other
# difference is downstream of d25, and the final frame is byte-exact.
M9_STRUCTURAL_MAX_LSB = d25_bound.WARP_INIT_MAX_DIFF


def classify(dispatch, base=0):
    """The lowered family name, or the custom-node name.

    `base` is the dispatch id before this frame's first node, so the custom
    nodes are found by frame-relative id (24/25) on every frame.
    """
    node = dispatch["dispatch"] - base
    if node in CUSTOM:
        return CUSTOM[node]
    for family, shaped in FAMILIES:
        if shaped(dispatch):
            return family
    return None


def edge_index(dispatches):
    """(consumer dispatch, SRV slot) -> (producer dispatch, UAV slot)."""
    writers = {}
    for dispatch in dispatches:
        for view in dispatch.get("resources", []):
            if view.get("kind") == "uav":
                writers.setdefault(view["resource"], []).append(
                    (dispatch["dispatch"], view["slot"]))
    multiple = [resource for resource, rows in writers.items() if len(rows) > 1]
    if multiple:
        raise SystemExit("%d resources have multiple writers in frame %d"
                         % (len(multiple), dispatches[0]["frame"]))
    single = {resource: rows[0] for resource, rows in writers.items()}
    edges = {}
    for dispatch in dispatches:
        number = dispatch["dispatch"]
        for view in dispatch.get("resources", []):
            if view.get("kind") != "srv":
                continue
            producer = single.get(view["resource"])
            if producer is not None and producer[0] < number:
                edges[(number, view["slot"])] = producer
    return edges


def view_maps(dispatches):
    """The UAV and SRV descriptor used for each (dispatch, slot)."""
    writer_views = {}
    reader_views = {}
    for dispatch in dispatches:
        for view in dispatch.get("resources", []):
            key = (dispatch["dispatch"], view["slot"])
            if view.get("kind") == "uav":
                writer_views[key] = view
            elif view.get("kind") == "srv":
                reader_views[key] = view
    return writer_views, reader_views


def _view_shape(view):
    return (view.get("dimension"), view.get("width"), view.get("height"),
            view.get("array"), view.get("format"),
            int(view.get("first_slice") or 0),
            int(view.get("buffer_offset") or 0))


def _slice_bytes(view):
    size = recon.FORMAT_BYTES.get(view.get("format"))
    width, height = view.get("width"), view.get("height")
    if size is None or not width or not height:
        return None
    return int(width * height * size)


def view_bytes(raw, producer, consumer):
    """Translate a producer allocation to the consumer's bound view.

    The calibrated run has identical producer/consumer views, so this returns
    the raw bytes unchanged. The slice rules cover the documented subresource
    cases: a texture2darray consumer takes `array` slices starting at
    `first_slice`, and a buffer consumer takes `width` bytes at
    `buffer_offset`. Unknown shape changes fail instead of guessing.
    """
    raw = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    if _view_shape(producer) == _view_shape(consumer):
        return raw
    if producer.get("dimension") != consumer.get("dimension"):
        raise ValueError("view dimension %s -> %s"
                         % (producer.get("dimension"),
                            consumer.get("dimension")))
    if producer.get("dimension") == "texture2darray":
        if (producer.get("width"), producer.get("height")) != (
                consumer.get("width"), consumer.get("height")):
            raise ValueError("texture2darray crop is not supported")
        slice_bytes = _slice_bytes(producer)
        if slice_bytes is None:
            raise ValueError("no texel size for texture2darray")
        first = int(consumer.get("first_slice") or 0)
        count = int(consumer.get("array") or 1)
        start = first * slice_bytes
        end = start + count * slice_bytes
        if start < 0 or end > raw.size:
            raise ValueError("texture2darray slice outside the allocation")
        return raw[start:end]
    if producer.get("dimension") == "buffer":
        offset = int(consumer.get("buffer_offset") or 0)
        size = int(consumer.get("width") or 0) * int(consumer.get("height") or 1)
        if size <= 0:
            size = raw.size - offset
        if offset < 0 or offset + size > raw.size:
            raise ValueError("buffer view outside the allocation")
        return raw[offset:offset + size]
    raise ValueError("unsupported view translation for %s"
                     % producer.get("dimension"))


def raw_bytes(value):
    return np.ascontiguousarray(value).view(np.uint8).reshape(-1)


def diff_count(got, expected):
    """Mismatch/max in the output's native signedness, over whole bytes."""
    got = np.ascontiguousarray(got)
    expected = np.asarray(expected)
    if expected.dtype.kind in "iu" and expected.dtype.itemsize > 1:
        # Plans such as conv_plan keep the signed values in an int64 array.
        ref = expected.reshape(-1)
    elif got.dtype == np.int8:
        ref = (expected.reshape(-1) if expected.dtype == np.int8
               else expected.view(np.uint8).reshape(-1).view(np.int8))
    elif got.dtype == np.uint8:
        ref = (expected.reshape(-1) if expected.dtype == np.uint8
               else expected.view(np.uint8).reshape(-1))
    else:
        raise ValueError("unsupported output dtype %s" % got.dtype)
    left = got.reshape(-1).astype(np.int16)
    right = ref.astype(np.int16)
    if left.size != right.size:
        raise ValueError("output size %d != expected size %d"
                         % (left.size, right.size))
    delta = left - right
    mismatch = int((delta != 0).sum())
    max_diff = int(np.abs(delta).max()) if mismatch else 0
    return mismatch, max_diff


def edge_counts(edges):
    """(resource edges, slot-level hand-offs) implied by the wiring."""
    resources = {(producer[0], consumer)
                 for (consumer, _slot), producer in edges.items()}
    return len(resources), len(edges)


def downstream_of(number, edges):
    """Every dispatch reachable from a producer, at node granularity."""
    adjacency = {}
    for (consumer, _slot), (producer, _pslot) in edges.items():
        adjacency.setdefault(producer, set()).add(consumer)
    seen = set()
    frontier = [number]
    while frontier:
        current = frontier.pop()
        for consumer in adjacency.get(current, ()):
            if consumer not in seen:
                seen.add(consumer)
                frontier.append(consumer)
    return seen


def structural_problems(results, errors, edges, strict=False, base=0,
                        check_counts=True, content=d25_bound.STATIC):
    """The gate for a resolution with no pinned M9_CALIBRATION entry.

    It keeps the properties that do not depend on a size: `d24` exact, `d25`
    inside 1 LSB plus the fraction envelope, every other difference downstream
    of `d25`, and `d38` byte-exact. Per-node LSB numbers stay calibrated-only.
    """

    def node_of(record):
        return record.get("node", record["dispatch"] - base)

    problems = ["d%04d: %s" % (node_of(error), error["error"])
                for error in errors]
    if check_counts:
        resources, handoffs = edge_counts(edges)
        if resources != M9_RESOURCE_EDGES:
            problems.append("%d resource edges, expected %d"
                            % (resources, M9_RESOURCE_EDGES))
        if handoffs != M9_SLOT_HANDOFFS:
            problems.append("%d slot-level hand-offs, expected %d"
                            % (handoffs, M9_SLOT_HANDOFFS))
    if len(results) != 38:
        problems.append("%d nodes ran, expected 38" % len(results))
    by_node = {node_of(result): result for result in results}
    d24 = by_node.get(24)
    if d24 is None or d24.get("mismatch") != 0:
        problems.append("d24 is not byte-exact")
    d25 = by_node.get(25)
    if d25 is None:
        problems.append("d25 did not run")
    else:
        ok, reason = d25_bound.evaluate(
            d25.get("mismatch", 1 << 30), d25.get("bytes", 0),
            d25.get("max_diff", 1 << 30), content=content)
        if not ok:
            problems.append("d25 is outside its structural bound: " + reason)
    downstream = downstream_of(25 + base, edges)
    for result in results:
        number = node_of(result)
        if number == 25:
            continue
        if result.get("mismatch") and result["dispatch"] not in downstream:
            problems.append("d%04d differs but is not downstream of d25"
                            % number)
    d38 = by_node.get(38)
    if d38 is None:
        problems.append("d38 did not run")
    elif d38.get("mismatch") != 0:
        problems.append("d38 final frame is not byte-exact")
    return problems


def result_problems(results, errors, mode, edges, strict=False, base=0,
                    calibration=None, structural=False, check_counts=True,
                    content=d25_bound.STATIC):
    if structural:
        return structural_problems(results, errors, edges, strict, base,
                                   check_counts, content)
    calibration = calibration or M9_DEFAULT_CALIBRATION

    def node_of(record):
        return record.get("node", record["dispatch"] - base)

    problems = ["d%04d: %s" % (node_of(error), error["error"])
                for error in errors]
    if check_counts:
        resources, handoffs = edge_counts(edges)
        if resources != M9_RESOURCE_EDGES:
            problems.append("%d resource edges, expected %d"
                            % (resources, M9_RESOURCE_EDGES))
        if handoffs != M9_SLOT_HANDOFFS:
            problems.append("%d slot-level hand-offs, expected %d"
                            % (handoffs, M9_SLOT_HANDOFFS))
    if len(results) != 38:
        problems.append("%d nodes ran, expected 38" % len(results))
    by_node = {node_of(result): result for result in results}
    if mode == "captured":
        for result in results:
            if result.get("mismatch") != 0:
                problems.append("d%04d is not byte-exact in captured mode"
                                % node_of(result))
        return problems

    d24 = by_node.get(24)
    if d24 is None or d24.get("mismatch") != 0:
        problems.append("d24 is not byte-exact")
    d25 = by_node.get(25)
    if d25 is None:
        problems.append("d25 did not run")
    else:
        width, height = calibration.get("resolution", (640, 360))
        ok, reason = d25_bound.evaluate(
            d25.get("mismatch", 1 << 30), d25.get("bytes", 0),
            d25.get("max_diff", 1 << 30), width=width, height=height,
            require_calibration=True, content=content)
        if not ok:
            problems.append("d25 is outside its 1-LSB bound: " + reason)
    downstream = downstream_of(25 + base, edges)
    for result in results:
        number = node_of(result)
        if number in CUSTOM:
            continue
        if result.get("mismatch") and result["dispatch"] not in downstream:
            problems.append("d%04d differs but is not downstream of d25"
                            % number)
    for number, (bound, count) in calibration["nodes"].items():
        result = by_node.get(number)
        if result is None:
            problems.append("d%04d did not run" % number)
            continue
        if result.get("max_diff", 1 << 30) > bound:
            problems.append("d%04d max diff %s exceeds its %d-LSB bound"
                            % (number, result.get("max_diff"), bound))
        if result.get("mismatch", 1 << 30) > count:
            problems.append("d%04d mismatch %s exceeds its %d-byte bound"
                            % (number, result.get("mismatch"), count))
    d38 = by_node.get(38)
    if d38 is None:
        problems.append("d38 did not run")
    elif strict:
        if d38.get("mismatch") != 0:
            problems.append("d38 final frame is not byte-exact (strict)")
    else:
        max_diff, count = calibration["d38"]
        if not (d38.get("max_diff", 1 << 30) <= max_diff
                and d38.get("mismatch", 1 << 30) <= count):
            problems.append("d38 is outside the d25 propagation bound")
    return problems


def _summary(results, base=0):
    def node_of(record):
        return record.get("node", record["dispatch"] - base)

    computed = [result for result in results if not result.get("copied")]
    copied = [result for result in results if result.get("copied")]
    exact = sum(1 for result in computed if result.get("mismatch") == 0)
    bound = sum(1 for result in computed
                if result.get("mismatch") and node_of(result) == 25)
    diff = sum(1 for result in computed
               if result.get("mismatch") and node_of(result) != 25)
    return {"computed": len(computed), "copied": len(copied),
            "exact": exact, "bound": bound, "diff": diff,
            "errors": sum(1 for result in results if result.get("error"))}
