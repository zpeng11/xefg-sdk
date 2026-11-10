#!/usr/bin/env python
"""Single source of truth for the `d25` (warp + flow init) capture bound.

`d25` is the only dispatch that samples the scene through the texture unit's
bilinear filter, so it is the only replay with a non-zero capture residual.
The residual scales with the written allocation and cannot be represented by
one global number. Calibrated sizes therefore keep their measured aggregate and
per-plane counts; an uncalibrated size gets a fraction-of-bytes envelope and is
never allowed to borrow another size's numbers.

Calibration is keyed by `(width, height, content)`: the static scene and a
moving scene have different residuals at the same size (the motion-only
empty-word branch and the moving flow). A run's content comes from
`run_content()`; a size with no entry for that content uses the structural
envelope, never the other content's numbers.

Every gate (recon, export_onnx, verify, host_replay, m9_graph) imports this
module instead of carrying its own copy. `d25_bound.py` has no imports beyond
the standard library, so it can be used from any of them without a cycle.
"""
import json
import os

# `max_diff` is an LSB count; the fraction is only the fallback for a size with
# no calibration entry. Calibrated sizes use their measured aggregate.
WARP_INIT_MAX_DIFF = 1
WARP_INIT_MAX_FRACTION = 0.0005

# Aggregate byte count and per-plane counts for each captured output size.
# Measured with the `uv32exact` sampler model (float32 uv, exact float64
# `uv * size`, half-to-even subtexel weights). `u3` is exact on every capture;
# `u0` carries the flow-init features and `u1`/`u2` the two motion-warp colour
# planes, so the residual lives almost entirely on the warped colour pulls.
WARP_INIT_CALIBRATION = {
    (640, 360): {"aggregate": 647,
                 "planes": {"u0": 0, "u1": 287, "u2": 360, "u3": 0}},
    (1280, 720): {"aggregate": 791,
                  "planes": {"u0": 0, "u1": 293, "u2": 498, "u3": 0}},
    (1920, 1080): {"aggregate": 1010,
                   "planes": {"u0": 3, "u1": 414, "u2": 593, "u3": 0}},
    (2560, 1440): {"aggregate": 790,
                   "planes": {"u0": 2, "u1": 259, "u2": 529, "u3": 0}},
}

# Motion content has its own entry: the empty-t8-word branch and the moving
# flow make the residual larger than the static scene, but `u0` (the flow
# features) and `u3` are still exact after the missing branch was added.
WARP_INIT_MOTION_CALIBRATION = {
    (640, 360): {"aggregate": 955,
                 "planes": {"u0": 0, "u1": 529, "u2": 426, "u3": 0}},
    (1920, 1080): {"aggregate": 916,
                   "planes": {"u0": 1, "u1": 462, "u2": 453, "u3": 0}},
    (2560, 1440): {"aggregate": 930,
                   "planes": {"u0": 4, "u1": 440, "u2": 486, "u3": 0}},
}

MOTION = "motion"
STATIC = "static"


def calibration(width, height, content=STATIC):
    """The measured entry for an output size and content class, or None."""
    if width is None or height is None:
        return None
    table = (WARP_INIT_MOTION_CALIBRATION if content == MOTION
             else WARP_INIT_CALIBRATION)
    return table.get((int(width), int(height)))


def calibrated(width, height, content=STATIC):
    return calibration(width, height, content) is not None


def gate(width, height, content=STATIC):
    """`calibrated` when a pinned entry exists, else `structural`."""
    return "calibrated" if calibrated(width, height, content) else "structural"


def fraction_limit(total):
    """The structural envelope for a size with no calibration entry."""
    return int(int(total) * WARP_INIT_MAX_FRACTION)


def aggregate_limit(total, width=None, height=None, content=STATIC):
    """The allowed mismatched-byte count for this allocation.

    A calibrated size and content class uses its measured count exactly. Any
    other combination gets the documented fraction, which is the same envelope
    `recon.py` used before the per-size tables existed and is deliberately
    independent of the allocation's padding shape.
    """
    entry = calibration(width, height, content)
    if entry is not None:
        return entry["aggregate"]
    return fraction_limit(total)


def evaluate(mismatch, total, max_diff, width=None, height=None, planes=None,
             require_calibration=False, content=STATIC):
    """Check one `d25` result against the shared bound.

    `planes` is an optional mapping slot -> (mismatched bytes, max LSB) for the
    four written planes. Per-plane caps are only meaningful where a calibration
    entry exists; a caller that requires one asks for it explicitly.

    Returns `(ok, reason)`. `reason` is empty on success and a short phrase on
    failure, so each caller can keep its own status wording.
    """
    mismatch = int(mismatch)
    total = int(total)
    max_diff = int(max_diff)
    if max_diff > WARP_INIT_MAX_DIFF:
        return False, "%d LSB max exceeds %d" % (max_diff, WARP_INIT_MAX_DIFF)
    if mismatch == 0:
        return True, ""
    entry = calibration(width, height, content)
    if entry is None and require_calibration:
        return False, "no d25 calibration for %sx%s %s" % (
            width, height, content)
    limit = aggregate_limit(total, width, height, content)
    if mismatch > limit:
        source = "calibrated" if entry is not None else "fraction"
        return False, ("%d/%d bytes exceeds the %s limit %d"
                       % (mismatch, total, source, limit))
    if planes is not None and entry is not None:
        bad = []
        for slot in ("u0", "u1", "u2", "u3"):
            value = planes.get(slot)
            if value is None:
                bad.append("%s missing" % slot)
                continue
            count, plane_max = value
            if plane_max > WARP_INIT_MAX_DIFF:
                bad.append("%s max %d" % (slot, plane_max))
            elif count > entry["planes"][slot]:
                bad.append("%s %d > %d" % (slot, count,
                                           entry["planes"][slot]))
        if bad:
            return False, "per-plane " + ", ".join(bad)
    return True, ""


def run_resolution(run_dir):
    """The configured output size of a probe/oracle run, or (None, None)."""
    width, height, _content = run_config(run_dir)
    return width, height


def run_content(run_dir):
    """The content class of a probe/oracle run: `motion` or `static`.

    The probe records `static: false` for a moving scene. A run without the
    field (or without a manifest) is treated as static.
    """
    return run_config(run_dir)[2]


def run_config(run_dir):
    """(width, height, content) for a probe/oracle run."""
    for relative in ("trace/meta.json", "manifest.json"):
        path = os.path.join(run_dir, relative)
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        config = data.get("configuration") or data
        width, height = config.get("width"), config.get("height")
        content = STATIC if config.get("static", True) else MOTION
        if width and height:
            return int(width), int(height), content
    return None, None, STATIC
