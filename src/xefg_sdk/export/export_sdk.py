#!/usr/bin/env python
"""Export the captured XeFG M9 dynamic graph as a portable ONNX SDK package.

The exporter reuses `m9_dynamic.build_specs()` and `m9_dynamic.Assembler` to
rebuild the exact 38-dispatch graph for one retained frame, then folds the
verified external resource aliases into a small public input contract:

    previous_color  UINT8  [H, W, 4]      (BGRA8)
    current_color   UINT8  [H, W, 4]      (BGRA8)
    motion_vectors  FLOAT16[Hm, Wm, 2]    (RG16F, MV-grid pixels, current->prev,
                                           unjittered)
    depth           FLOAT  [Hm, Wm]       (raw, non-inverted)
    ui              UINT8  [H, W, 4]      (mode 3 only, premultiplied)
    backbuffer      UINT8  [H, W, 4]      (mode 4 only, current colour with UI)

Mode 1 uses the backbuffer colour pair; modes 3 and 4 use the hudless pair.
Composition uses current `ui` in mode 3 or current `backbuffer` in mode 4.

This is a *normalized* model input contract.  The caller converts its raw
engine/API conventions (camera transforms, jitter offsets, motion-vector units
and scale, depth inversion, UI compositing) into the values above before
submission.  The manifest `conventions` and profile `configuration` describe
the normalized tensors the model requires; they are not restrictions on the
application's engine or graphics-API state.

The 32x32 gate LUT is frozen into an initializer.  Every planner scalar, pad
and shape feed remains a graph input and is written to the matching profile.
No captured frame, feature or reference output is packaged.  The public model
has one output, `generated_frame`; the optional debug model keeps all 42 raw
assembled outputs with a separate mapping.

    xefg-export --runs /data/capture... --out /data/model_package \
        [--frame 1] [--debug]

The graph is rewired by resource identity only.  There is no numeric
optimization: the only additions are exact Cast/Reshape adapters, the frozen
LUT initializer and the final Identity rename.
"""
import hashlib
import json
import os
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import m9_dynamic


SCHEMA_VERSION = 1
PACKAGE_NAME = "xefg2x"
MODEL_OP_SET = 18
MODEL_IR_VERSION = 8
FINAL_OUTPUT = "n0038_u0"
PUBLIC_OUTPUT = "generated_frame"
LUT_INPUT = "n0025_x5"
LUT_SIZE = 1024
MAX_PLANNER_FEED_VALUES = 16

# Public semantic resource -> graph-internal inputs that carry the same bytes.
ALIASES = {
    1: {
        "previous_color": ("n0001_x0", "n0025_x1", "n0038_in_t11"),
        "current_color": ("n0001_x1", "n0025_x2", "n0038_in_t12"),
        "motion_vectors": ("n0024_x0", "n0025_x3"),
        "depth": ("n0024_x1", "n0025_x4"),
    },
    3: {
        "previous_color": ("n0001_x0", "n0025_x1", "n0038_in_t11"),
        "current_color": ("n0001_x1", "n0025_x2", "n0038_in_t12"),
        "motion_vectors": ("n0024_x0", "n0025_x3"),
        "depth": ("n0024_x1", "n0025_x4"),
        "ui": ("n0038_in_t9",),
    },
    4: {
        # d38.t9 is bound but unused; only t10 exposes the backbuffer.
        "previous_color": ("n0001_x0", "n0025_x1", "n0038_in_t11"),
        "current_color": ("n0001_x1", "n0025_x2", "n0038_in_t12",
                          "n0038_in_t8"),
        "motion_vectors": ("n0024_x0", "n0025_x3"),
        "depth": ("n0024_x1", "n0025_x4"),
        "backbuffer": ("n0038_in_t10",),
    },
}

PUBLIC_INPUTS = {
    1: (
        ("previous_color", TensorProto.UINT8, ["H", "W", 4]),
        ("current_color", TensorProto.UINT8, ["H", "W", 4]),
        ("motion_vectors", TensorProto.FLOAT16, ["Hm", "Wm", 2]),
        ("depth", TensorProto.FLOAT, ["Hm", "Wm"]),
    ),
    3: (
        ("previous_color", TensorProto.UINT8, ["H", "W", 4]),
        ("current_color", TensorProto.UINT8, ["H", "W", 4]),
        ("motion_vectors", TensorProto.FLOAT16, ["Hm", "Wm", 2]),
        ("depth", TensorProto.FLOAT, ["Hm", "Wm"]),
        ("ui", TensorProto.UINT8, ["H", "W", 4]),
    ),
    4: (
        ("previous_color", TensorProto.UINT8, ["H", "W", 4]),
        ("current_color", TensorProto.UINT8, ["H", "W", 4]),
        ("motion_vectors", TensorProto.FLOAT16, ["Hm", "Wm", 2]),
        ("depth", TensorProto.FLOAT, ["Hm", "Wm"]),
        ("backbuffer", TensorProto.UINT8, ["H", "W", 4]),
    ),
}

# Internal alias -> exact adapter needed before the dispatch consumers.
FLAT_ALIASES = {"n0025_x1", "n0025_x2", "n0025_x4"}
CAST_FLAT_ALIASES = {"n0025_x3"}

# Runtime conventions of the *normalized* model inputs.  The SDK package
# loader validates these exact values.  They describe the tensors the model
# requires, not raw engine/API conventions; callers normalize before
# submission.
CONVENTIONS = {
    "color_format": "BGRA8_UNORM",
    "motion_format": "RG16_FLOAT",
    "motion_units": "mv_pixels",
    "motion_direction": "current_to_previous",
    "depth_inverted": False,
    "jittered_motion": False,
    "ui_premultiplied": True,
}

# Explanatory, additive metadata.  It does not change schema_version 1 or the
# runtime-validated `conventions` values.
INPUT_CONTRACT = {
    "scope": "normalized model inputs",
    "description": (
        "`conventions` and profile `configuration` describe the normalized "
        "tensors the model requires.  Raw engine/API conventions (camera "
        "transforms, jitter offsets, motion-vector units/scale, depth "
        "inversion and UI compositing) must be converted by the caller; they "
        "are not constraints on application or engine configuration."
    ),
    "reference": "docs/IO_CONTRACT.md",
}

REFERENCES = [
    "MFG-Unlocker@3d67197:tools/xefg_oracle/DYNAMIC_SHAPE.md",
    "MFG-Unlocker@3d67197:tools/xefg_oracle/CONTENT_MATRIX.md",
    "https://github.com/intel/xess/blob/8fe81bdbbaf00b3c1b733fd0d830c333dc84e6f0/inc/xess_fg/xefg_swapchain.h",
    "https://github.com/intel/xess/blob/8fe81bdbbaf00b3c1b733fd0d830c333dc84e6f0/inc/xess_fg/xefg_swapchain_d3d12.h",
    "https://github.com/optiscaler/OptiScaler/blob/07aa9da670d6f0b9358acc392163b7c1cdc345fe/OptiScaler/inputs/FG/XeFG_Inputs_Dx12.cpp",
]

LIMITATIONS = [
    "The exporter supports only the normalized XeFG input contract: "
    "BGRA8_UNORM color, mode 3 premultiplied UI, mode 4 current backbuffer, "
    "RG16_FLOAT half-resolution MV-grid pixel-space MV "
    "(current_to_previous), raw non-inverted R32_FLOAT depth, full rect and "
    "2x midpoint interpolation.",
    "The model consumes normalized inputs: motion vectors already expressed "
    "in MV-grid pixels, current_to_previous, with jitter already removed, and "
    "raw non-inverted depth.  Raw engine/API conventions (camera transforms, "
    "jitter offsets, MV units/scale, depth inversion and UI compositing) must "
    "be converted by the caller.  The model does not require identity camera "
    "matrices or a particular engine-side MV scale.  The splat scale is baked "
    "into an initializer; the warp motion scale and temporal ratio are "
    "profile-fixed graph inputs.",
    "mode 1, 3 and 4 only; modes 2/5 and AUTO are not exported.  mode 3 "
    "requires premultiplied UI; mode 4 consumes the current backbuffer and "
    "does not take a separate UI texture.",
    "Only the profiles listed in manifest.json are supported.  Shape, phase, "
    "padding and allocation scalars are profile-fixed; shape feeds are "
    "profile-based.",
    "The package contains no captured frames, features, reference outputs or "
    "static/motion calibration tables.  The runtime must supply the "
    "application colors, MV, depth and the mode 3 UI or mode 4 backbuffer.",
    "The optional debug model exposes all 42 raw assembled outputs for "
    "investigation only and is not a stable deployment contract.",
    "Model assets (weights, quantisation constants and the gate LUT) are "
    "frozen from the capture.  An unknown external tensor input is rejected "
    "instead of being zero-filled.",
    "No vendor NPU/backend is implemented; the model is plain ONNX opset 18 "
    "and the runtime owns execution.",
]

_SEMANTIC_DESCRIPTION = {
    "previous_color": "Previous application/hudless colour, BGRA8",
    "current_color": "Current application/hudless colour, BGRA8",
    "motion_vectors": ("Current-frame RG16F motion vectors in MV-grid pixels, "
                       "current_to_previous, unjittered"),
    "depth": "Current-frame raw, non-inverted R32F depth on the MV grid",
    "ui": "Premultiplied BGRA8 UI for mode 3",
    "backbuffer": ("Current application backbuffer with UI already composed, "
                   "BGRA8; mode 4 only"),
}

_CONFIG_SEMANTIC = {
    "n0025_wi_motion_scale": (
        "configuration.warp_motion_scale",
        "Provider root constant c47 captured in the profile; inferred motion "
        "t0 plane scale, not a closed app-MV-ratio formula",
        "provider root constant c47; neutral warp motion scale"),
    "n0025_wi_ratio": (
        "configuration.temporal_interpolation_ratio",
        "Provider root constant c50/(1-c50), known temporal midpoint 1; "
        "profile-fixed feed, separate from the warp motion scale",
        "provider root constant c50/(1-c50); known temporal midpoint 1"),
}

_ELEM_NAMES = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
}


class ExportError(RuntimeError):
    """A capture or rewrite does not satisfy the SDK contract."""


class Capture(object):
    """One retained frame rebuilt by the dynamic assembler."""

    def __init__(self, run_dir, frame, meta, mode, specs, dispatches,
                 assembler, model, feeds):
        self.run_dir = run_dir
        self.frame = frame
        self.meta = meta
        self.mode = mode
        self.specs = specs
        self.dispatches = dispatches
        self.assembler = assembler
        self.model = model
        self.feeds = feeds
        self.output_size = None
        self.motion_size = None
        config = meta.get("configuration") or {}
        self.config = config

    @property
    def name(self):
        return os.path.basename(os.path.normpath(self.run_dir))


class Rewrite(object):
    """The public/debug graphs produced from one assembled capture."""

    def __init__(self, public_model, debug_model, info):
        self.public_model = public_model
        self.debug_model = debug_model
        self.info = info


class ProfileRegistry(object):
    """Collision-safe profile ids for one package."""

    def __init__(self):
        self.entries = {}

    @staticmethod
    def _canonical(profile):
        payload = {key: value for key, value in profile.items() if key != "id"}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def add(self, profile):
        canonical = self._canonical(profile)
        base = profile["id"]
        candidate = base
        if candidate in self.entries:
            if self.entries[candidate]["canonical"] == canonical:
                return self.entries[candidate]["profile"]
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
            candidate = base + "-" + digest
        while candidate in self.entries:
            if self.entries[candidate]["canonical"] == canonical:
                return self.entries[candidate]["profile"]
            raise ExportError(
                "profile id %s already holds different metadata" % candidate)
        stored = dict(profile)
        stored["id"] = candidate
        self.entries[candidate] = {"profile": stored, "canonical": canonical}
        return stored

    def profiles(self):
        return [self.entries[key]["profile"] for key in sorted(self.entries)]


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_array(array):
    return np.ascontiguousarray(np.asarray(array))


def _dtype_name(elem_type):
    if elem_type not in _ELEM_NAMES:
        raise ExportError("unsupported ONNX element type %d" % elem_type)
    return _ELEM_NAMES[elem_type]


def _array_dtype_name(array):
    return np.asarray(array).dtype.name


def _declared_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _feed_record(array):
    array = np.asarray(array)
    if array.dtype.kind in "iu":
        values = [int(value) for value in array.reshape(-1)]
    elif array.dtype.kind == "f":
        values = [float(value) for value in array.reshape(-1)]
    elif array.dtype.kind == "b":
        values = [bool(value) for value in array.reshape(-1)]
    else:
        raise ExportError("unsupported profile feed dtype %s" % array.dtype)
    return {
        "dtype": _array_dtype_name(array),
        "shape": [int(value) for value in array.shape],
        "values": values,
    }


def _initializer_digest(initializers):
    digest = hashlib.sha256()
    for tensor in sorted(initializers, key=lambda item: item.name):
        array = numpy_helper.to_array(tensor)
        digest.update(tensor.name.encode("utf-8"))
        digest.update(str(tensor.data_type).encode("ascii"))
        digest.update(str(list(tensor.dims)).encode("ascii"))
        digest.update(_canonical_array(array).tobytes())
    return digest.hexdigest()


def _output_value_info(model, name):
    for value in model.graph.output:
        if value.name == name:
            return value
    raise ExportError("assembled graph has no output %s" % name)


def _input_value_info(model, name):
    for value in model.graph.input:
        if value.name == name:
            return value
    return None


def _model_names(model):
    names = set()
    for value in model.graph.input:
        names.add(value.name)
    for value in model.graph.output:
        names.add(value.name)
    for value in model.graph.initializer:
        names.add(value.name)
    for node in model.graph.node:
        names.update(value for value in node.input if value)
        names.update(value for value in node.output if value)
    return names


def _same_bytes(left, right):
    return _canonical_array(left).tobytes() == _canonical_array(right).tobytes()


def _require(condition, message):
    if not condition:
        raise ExportError(message)


def _plan_mode(specs, run_dir):
    """The exported variant comes from the final composite plan."""
    plan_mode = specs[38]["plan"].get("mode")
    if plan_mode not in (1, 3, 4):
        raise ExportError(
            "%s: unsupported final composite mode %r (exports 1, 3 and 4)"
            % (run_dir, plan_mode))
    return int(plan_mode)


def build_capture(run_dir, frame=1):
    """Plan, assemble and feed one retained frame without running inference."""
    run_dir = os.path.abspath(run_dir)
    meta_path = os.path.join(run_dir, "trace", "meta.json")
    if not os.path.isfile(meta_path):
        raise ExportError("missing trace/meta.json in %s" % run_dir)
    with open(meta_path, encoding="utf-8") as handle:
        meta_bytes = handle.read()
    meta = json.loads(meta_bytes)
    config = meta.get("configuration") or {}

    specs = m9_dynamic.build_specs(run_dir, frame)
    dispatches = m9_dynamic.normalized_dispatches(run_dir, frame)
    assembler = m9_dynamic.Assembler(specs, dispatches)
    model = assembler.build()
    feeds = assembler.feeds_for(specs)

    missing = [name for name in (ALIASES[1]["previous_color"][0],
                                 ALIASES[1]["current_color"][0],
                                 "n0024_x0", "n0024_x1", LUT_INPUT)
               if name not in feeds]
    if missing:
        raise ExportError("capture is missing external inputs: %s"
                          % ", ".join(missing))

    capture = Capture(run_dir, frame, meta, None, specs, dispatches,
                      assembler, model, feeds)
    capture.meta_sha256 = _sha256(meta_bytes.encode("utf-8"))
    capture.output_size = (int(feeds["n0001_x0"].shape[0]),
                           int(feeds["n0001_x0"].shape[1]))
    capture.motion_size = (int(feeds["n0024_x0"].shape[0]),
                           int(feeds["n0024_x0"].shape[1]))
    capture.has_ui = "n0038_in_t9" in assembler.cut_inputs
    capture.has_backbuffer = "n0038_in_t10" in assembler.cut_inputs

    # The composite plan, not the metadata UI flag, defines the exported
    # variant.  Other UI modes (2/5) must not be relabelled as mode 1/3/4 even
    # when their t9/t10 presence happens to look similar.
    capture.mode = _plan_mode(specs, run_dir)
    if capture.mode == 3 and not capture.has_ui:
        raise ExportError(
            "%s: mode 3 composite has no UI input" % run_dir)
    if capture.mode == 4 and not capture.has_backbuffer:
        raise ExportError(
            "%s: mode 4 composite has no current backbuffer input"
            % run_dir)

    _validate_capture(capture)
    return capture


def _validate_capture(capture):
    feeds = capture.feeds
    mode = capture.mode
    height, width = capture.output_size
    motion_height, motion_width = capture.motion_size

    if width <= 0 or height <= 0 or motion_width <= 0 or motion_height <= 0:
        raise ExportError("%s: empty output/MV size" % capture.run_dir)
    config = capture.config
    for key, expected in (("width", width), ("height", height),
                          ("motion_width", motion_width),
                          ("motion_height", motion_height)):
        if key in config and int(config[key]) != expected:
            raise ExportError(
                "%s: meta %s=%s does not match the captured feed %s"
                % (capture.run_dir, key, config[key], expected))
    if config.get("flip_mv"):
        raise ExportError("%s: flip_mv is not supported"
                          % capture.run_dir)

    # The alias groups are the verified resource identities.  Byte/bit
    # equality is required; the SDK never assumes two slots are the same.
    for public, names in ALIASES[mode].items():
        for name in names:
            _require(name in feeds, "%s: missing %s for %s"
                     % (capture.run_dir, name, public))
        if public in ("previous_color", "current_color"):
            expected_size = width * height * 4
            for name in names:
                array = _canonical_array(feeds[name])
                _require(array.dtype == np.uint8,
                         "%s: %s is not uint8" % (capture.run_dir, name))
                _require(array.size == expected_size,
                         "%s: %s has %d values, expected %d"
                         % (capture.run_dir, name, array.size, expected_size))
                _require(_same_bytes(array, feeds[names[0]]),
                         "%s: %s bytes differ from %s"
                         % (capture.run_dir, name, names[0]))
        elif public == "motion_vectors":
            half = _canonical_array(feeds["n0024_x0"])
            flat = _canonical_array(feeds["n0025_x3"])
            _require(half.dtype == np.float16,
                     "%s: n0024_x0 is not float16" % capture.run_dir)
            _require(half.shape == (motion_height, motion_width, 2),
                     "%s: n0024_x0 shape %s, expected %s"
                     % (capture.run_dir, half.shape,
                        (motion_height, motion_width, 2)))
            _require(flat.dtype == np.float32,
                     "%s: n0025_x3 is not float32" % capture.run_dir)
            _require(flat.size == motion_width * motion_height * 2,
                     "%s: n0025_x3 has %d values, expected %d"
                     % (capture.run_dir, flat.size,
                        motion_width * motion_height * 2))
            _require(np.array_equal(
                half.astype(np.float32).reshape(-1), flat),
                "%s: MV aliases are not the same values" % capture.run_dir)
        elif public == "depth":
            depth = _canonical_array(feeds["n0024_x1"])
            flat = _canonical_array(feeds["n0025_x4"])
            _require(depth.dtype == np.float32,
                     "%s: n0024_x1 is not float32" % capture.run_dir)
            _require(depth.shape == (motion_height, motion_width),
                     "%s: n0024_x1 shape %s, expected %s"
                     % (capture.run_dir, depth.shape,
                        (motion_height, motion_width)))
            _require(flat.dtype == np.float32,
                     "%s: n0025_x4 is not float32" % capture.run_dir)
            _require(flat.size == motion_width * motion_height,
                     "%s: n0025_x4 has %d values, expected %d"
                     % (capture.run_dir, flat.size,
                        motion_width * motion_height))
            _require(np.array_equal(depth.reshape(-1), flat),
                     "%s: depth aliases are not the same values"
                     % capture.run_dir)
        elif public == "ui":
            ui = _canonical_array(feeds["n0038_in_t9"])
            _require(ui.dtype == np.uint8,
                     "%s: UI is not uint8" % capture.run_dir)
            _require(ui.shape == (height, width, 4),
                     "%s: UI shape %s, expected %s"
                     % (capture.run_dir, ui.shape, (height, width, 4)))
        elif public == "backbuffer":
            backbuffer = _canonical_array(feeds[names[0]])
            _require(backbuffer.dtype == np.uint8,
                     "%s: backbuffer is not uint8" % capture.run_dir)
            _require(backbuffer.shape == (height, width, 4),
                     "%s: backbuffer shape %s, expected %s"
                     % (capture.run_dir, backbuffer.shape,
                        (height, width, 4)))

    lut = _canonical_array(feeds[LUT_INPUT])
    _require(lut.dtype == np.float32,
             "%s: gate LUT is not float32" % capture.run_dir)
    _require(lut.size == LUT_SIZE,
             "%s: gate LUT has %d values, expected %d"
             % (capture.run_dir, lut.size, LUT_SIZE))

    _validate_canonical_configuration(capture)


def _validate_canonical_configuration(capture):
    """Check the export boundary of the captured canonical graph.

    This is an export-time check over the captured profile-fixed feeds, not a
    runtime input restriction.  It ensures the constants serialized into the
    profile (full rect, 2x midpoint and baked splat scale) faithfully match
    the captured graph.  It says nothing about raw engine camera, jitter or
    motion-vector-scale conventions, which the caller normalizes before
    submission.
    """
    feeds = capture.feeds

    def scalar(name):
        array = np.asarray(feeds[name])
        if array.size != 1:
            raise ExportError("%s: %s is not a scalar"
                              % (capture.run_dir, name))
        return float(array.reshape(-1)[0])

    motion_scale = scalar("n0025_wi_motion_scale")
    ratio = scalar("n0025_wi_ratio")
    if not np.isclose(motion_scale, 4.0, rtol=0.0, atol=1e-6):
        raise ExportError("%s: unsupported MV scale %r (expects 4.0)"
                          % (capture.run_dir, motion_scale))
    if not np.isclose(ratio, 1.0, rtol=0.0, atol=1e-6):
        raise ExportError("%s: unsupported interpolation ratio %r "
                          "(expects 1.0)" % (capture.run_dir, ratio))

    height, width = capture.output_size
    lo_x = scalar("n0025_wi_lo_ax")
    lo_y = scalar("n0025_wi_lo_ay")
    hi_x = scalar("n0025_wi_hi_ax")
    hi_y = scalar("n0025_wi_hi_ay")
    if not (np.isclose(lo_x, 0.5001, atol=1e-3)
            and np.isclose(lo_y, 0.5001, atol=1e-3)
            and np.isclose(hi_x, width - 0.5001, atol=1e-2)
            and np.isclose(hi_y, height - 0.5001, atol=1e-2)):
        raise ExportError("%s: only the full-rect configuration is supported"
                          % capture.run_dir)

    for name, expected in (("n0024_sp_scale_x", 0.5),
                           ("n0024_sp_scale_y", 0.5)):
        tensor = next((item for item in capture.model.graph.initializer
                       if item.name == name), None)
        if tensor is None:
            raise ExportError("%s: missing baked %s"
                              % (capture.run_dir, name))
        value = float(numpy_helper.to_array(tensor).reshape(-1)[0])
        if not np.isclose(value, expected, rtol=0.0, atol=1e-7):
            raise ExportError("%s: %s is %r, expected %r"
                              % (capture.run_dir, name, value, expected))


def _adapter_nodes(mode, names):
    """Exact Cast/Reshape adapters for the non-canonical alias layouts."""
    nodes = []
    initializers = []
    flat_shape = "sdk_flat_shape"
    if any(alias in FLAT_ALIASES or alias in CAST_FLAT_ALIASES
           for alias in names):
        initializers.append(numpy_helper.from_array(
            np.asarray([-1], dtype=np.int64), flat_shape))
    for alias in names:
        if alias in FLAT_ALIASES:
            public = None
            for semantic, group in ALIASES[mode].items():
                if alias in group:
                    public = semantic
                    break
            nodes.append(helper.make_node(
                "Reshape", [public, flat_shape], [alias],
                name="sdk_%s_flat" % public))
        elif alias in CAST_FLAT_ALIASES:
            public = "motion_vectors"
            cast_name = "sdk_motion_float32"
            nodes.append(helper.make_node(
                "Cast", [public], [cast_name], to=TensorProto.FLOAT,
                name="sdk_motion_float32"))
            nodes.append(helper.make_node(
                "Reshape", [cast_name, flat_shape], [alias],
                name="sdk_motion_flat"))
    return nodes, initializers


def rewrite_graph(model, feeds, mode, want_debug=False):
    """Fold verified aliases/LUT and expose the public SDK input/output names.

    `model` is the assembled dynamic graph, `feeds` is `Assembler.feeds_for()`
    for the same capture.  The returned public model has exactly one output,
    `generated_frame`; the debug model (when requested) keeps every original
    graph output under its original name.
    """
    if mode not in ALIASES:
        raise ExportError("unsupported mode %r (exports 1, 3 and 4)" % mode)
    aliases = ALIASES[mode]
    alias_names = set()
    for group in aliases.values():
        alias_names.update(group)

    input_names = [value.name for value in model.graph.input]
    missing = [name for name in input_names if name not in feeds]
    if missing:
        raise ExportError("no feed for graph input(s): %s"
                          % ", ".join(sorted(missing)))
    absent = [name for group in aliases.values() for name in group
              if name not in input_names]
    if absent:
        raise ExportError("mode %d is missing required external input(s): %s"
                          % (mode, ", ".join(sorted(absent))))

    # Reject an unknown external tensor instead of silently zero-filling it.
    # Every legitimate remaining input is a planner scalar/pad with at most a
    # handful of int64/float32 values.
    scalars = []
    for name in input_names:
        if name in alias_names or name == LUT_INPUT:
            continue
        array = _canonical_array(feeds[name])
        if array.size > MAX_PLANNER_FEED_VALUES:
            raise ExportError(
                "unsupported external input %s (%d values); the SDK does not "
                "zero-fill unknown tensors" % (name, array.size))
        if array.dtype not in (np.int64, np.float32, np.int32):
            raise ExportError(
                "unsupported external input %s (dtype %s); expected a planner "
                "scalar/pad" % (name, array.dtype))
        scalars.append(name)

    # Adapter needs are known from the verified alias layout.  A direct alias
    # must have exactly the public rank (the byte/bit check already ran).
    adapter_names = sorted(name for name in alias_names
                           if name in FLAT_ALIASES or name in CAST_FLAT_ALIASES)
    direct = {name: semantic
              for semantic, group in aliases.items()
              for name in group
              if name not in adapter_names}

    # Public input shapes are symbolic so the serialized graph is identical
    # across sizes/frames; the profile carries the concrete extents.
    public_inputs = []
    for name, elem, dims in PUBLIC_INPUTS[mode]:
        public_inputs.append(
            helper.make_tensor_value_info(name, elem, list(dims)))

    remaining_inputs = [
        value for value in model.graph.input
        if value.name not in alias_names and value.name != LUT_INPUT
    ]

    adapter_nodes, adapter_initializers = _adapter_nodes(
        mode, adapter_names)

    nodes = []
    for node in adapter_nodes:
        copy = onnx.NodeProto()
        copy.CopyFrom(node)
        nodes.append(copy)
    for node in model.graph.node:
        copy = onnx.NodeProto()
        copy.CopyFrom(node)
        copy.input[:] = [direct.get(value, value) if value else ""
                         for value in node.input]
        nodes.append(copy)

    final_info = _output_value_info(model, FINAL_OUTPUT)
    final_elem = final_info.type.tensor_type.elem_type
    nodes.append(helper.make_node(
        "Identity", [FINAL_OUTPUT], [PUBLIC_OUTPUT],
        name="sdk_generated_frame"))

    initializers = []
    for tensor in model.graph.initializer:
        copy = onnx.TensorProto()
        copy.CopyFrom(tensor)
        initializers.append(copy)
    lut = _canonical_array(feeds[LUT_INPUT]).reshape(LUT_SIZE)
    initializers.append(numpy_helper.from_array(lut, LUT_INPUT))
    initializers.extend(adapter_initializers)

    public_outputs = [helper.make_tensor_value_info(
        PUBLIC_OUTPUT, final_elem, ["H", "W", 4])]
    graph = helper.make_graph(
        nodes, "xefg2x_mode%d" % mode,
        public_inputs + remaining_inputs, public_outputs, initializers)
    public_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", MODEL_OP_SET)])
    _stamp_model(public_model)
    onnx.checker.check_model(public_model)

    debug_model = None
    if want_debug:
        debug_nodes = []
        for node in nodes:
            if node.name != "sdk_generated_frame":
                copy = onnx.NodeProto()
                copy.CopyFrom(node)
                debug_nodes.append(copy)
        debug_outputs = []
        for value in model.graph.output:
            copy = onnx.ValueInfoProto()
            copy.CopyFrom(value)
            debug_outputs.append(copy)
        debug_initializers = []
        for tensor in initializers:
            copy = onnx.TensorProto()
            copy.CopyFrom(tensor)
            debug_initializers.append(copy)
        debug_graph = helper.make_graph(
            debug_nodes, "xefg2x_mode%d_debug" % mode,
            public_inputs + remaining_inputs, debug_outputs,
            debug_initializers)
        debug_model = helper.make_model(
            debug_graph,
            opset_imports=[helper.make_opsetid("", MODEL_OP_SET)])
        _stamp_model(debug_model)
        onnx.checker.check_model(debug_model)

    info = {
        "mode": mode,
        "public_inputs": [entry[0] for entry in PUBLIC_INPUTS[mode]],
        "removed_inputs": sorted(alias_names | {LUT_INPUT}),
        "adapters": [{"output": name,
                      "transform": ("cast:float16_to_float32+reshape:flat"
                                    if name in CAST_FLAT_ALIASES
                                    else "reshape:flat")}
                     for name in adapter_names],
        "direct_aliases": sorted(direct),
        "lut_input": LUT_INPUT,
        "lut_sha256": _sha256(lut.tobytes()),
        "public_output": PUBLIC_OUTPUT,
        "final_output": FINAL_OUTPUT,
        "final_dtype": _ELEM_NAMES[final_elem],
        "original_inputs": len(model.graph.input),
        "public_graph_inputs": len(public_inputs) + len(remaining_inputs),
        "public_graph_outputs": len(public_outputs),
        "original_outputs": len(model.graph.output),
        "initializer_count": len(initializers),
        "initializer_digest": _initializer_digest(initializers),
        "public_model_sha256": _sha256(public_model.SerializeToString()),
        "scalar_inputs": sorted(scalars),
    }
    return Rewrite(public_model, debug_model, info)


def _stamp_model(model):
    model.ir_version = MODEL_IR_VERSION
    model.producer_name = "xefg_oracle.export_sdk"
    model.producer_version = "0.1"
    model.model_version = 1


def collect_profile_feeds(feeds, removed):
    """Typed planner feeds, excluding every real image input and the LUT."""
    removed = set(removed)
    profile = {}
    for name in sorted(feeds):
        if name in removed:
            continue
        array = _canonical_array(feeds[name])
        if array.size > MAX_PLANNER_FEED_VALUES:
            raise ExportError(
                "profile feed %s has %d values; image inputs must not enter a "
                "profile" % (name, array.size))
        profile[name] = _feed_record(array)
    return profile


def _configuration(mode):
    """Profile configuration for the normalized model input contract.

    Camera transforms, jitter offsets, raw motion-vector scale and other
    engine/API conventions are converted before submission and are not part of
    a profile.  Reference-capture properties are recorded under `source`.
    """
    return {
        "input_contract": "normalized",
        "color_format": "BGRA8_UNORM",
        "motion_format": "RG16_FLOAT",
        "motion_units": "mv_pixels",
        "motion_direction": "current_to_previous",
        "motion_vector_resolution": "half",
        "jittered_motion": False,
        "depth_format": "R32_FLOAT",
        "depth_inverted": False,
        "full_rect": True,
        "interpolation_factor": 2,
        "interpolation_position": 0.5,
        "ui_premultiplied": True,
        "ui_enabled": mode in (3, 4),
        "baked": {
            "splat_scale": [0.5, 0.5],
        },
        "profile_fixed": {
            "warp_motion_scale": 4.0,
            "temporal_interpolation_ratio": 1.0,
        },
    }


def _profile_id(capture):
    height, width = capture.output_size
    motion_height, motion_width = capture.motion_size
    return "mode%d-%dx%d-mv%dx%d" % (
        capture.mode, width, height, motion_width, motion_height)


def _build_profile(capture, rewrite):
    profile = {
        "schema_version": SCHEMA_VERSION,
        "id": _profile_id(capture),
        "variant": "mode%d" % capture.mode,
        "output_size": [capture.output_size[1], capture.output_size[0]],
        "motion_size": [capture.motion_size[1], capture.motion_size[0]],
        "feeds": collect_profile_feeds(capture.feeds,
                                       rewrite.info["removed_inputs"]),
        "configuration": _configuration(capture.mode),
    }
    return profile


def _source(captures, rewrites):
    entries = []
    for capture in captures:
        rewrite = rewrites[capture.run_dir]
        config = capture.config
        entries.append({
            "run": capture.run_dir,
            "name": capture.name,
            "frame": capture.frame,
            "variant": "mode%d" % capture.mode,
            "output_size": [capture.output_size[1], capture.output_size[0]],
            "motion_size": [capture.motion_size[1], capture.motion_size[0]],
            "meta_sha256": capture.meta_sha256,
            "meta": {
                "adapter": capture.meta.get("adapter"),
                "xefg_family": capture.meta.get("xefg_family"),
                "xefg_dll_sha256": capture.meta.get("xefg_dll_sha256"),
                "configuration": {
                    key: config.get(key) for key in (
                        "width", "height", "motion_width", "motion_height",
                        "ui", "ui_mode", "frames", "warmup", "velocity",
                        "flip_mv", "static") if key in config
                },
            },
            "graph": {
                "inputs": rewrite.info["original_inputs"],
                "outputs": rewrite.info["original_outputs"],
                "nodes": len(capture.model.graph.node),
                "initializers": len(capture.model.graph.initializer),
                "packaged_inputs": rewrite.info["public_graph_inputs"],
                "packaged_outputs": rewrite.info["public_graph_outputs"],
                "packaged_nodes": len(rewrite.public_model.graph.node),
                "packaged_initializers": rewrite.info["initializer_count"],
            },
            "initializer_digest": rewrite.info["initializer_digest"],
            "model_sha256": rewrite.info["public_model_sha256"],
        })
    entries.sort(key=lambda item: (item["variant"], item["run"],
                                   item["frame"]))
    return {
        "tool": "xefg_sdk.export.export_sdk",
        "generator": "xefg_sdk.export.m9_dynamic.Assembler",
        "captures": entries,
        "references": list(REFERENCES),
        # Provenance only: these describe how the retained reference captures
        # were produced.  They are not required of engines or applications.
        "reference_capture_conventions": {
            "scope": "probe/reference captures used by this export",
            "view_matrix": "identity",
            "projection_matrix": "identity",
            "motion_vector_scale": [1.0, 1.0],
            "jitter": [0.0, 0.0],
            "note": (
                "Properties of the reference capture harness, not a universal "
                "requirement.  Callers normalize raw engine inputs to MV-grid "
                "pixels, current_to_previous, unjittered, with raw "
                "non-inverted depth before submission."
            ),
        },
    }


def _public_input_contract(mode):
    entries = []
    for name, elem, dims in PUBLIC_INPUTS[mode]:
        entries.append({
            "name": name,
            "dtype": _dtype_name(elem),
            "shape": list(dims),
            "semantic": name,
            "description": _SEMANTIC_DESCRIPTION[name],
            "required": True,
        })
    return entries


def _input_slot(capture, node, input_name):
    """The dispatch SRV slot for a submodel input, e.g. x1 -> t2."""
    spec = capture.specs.get(node) or {}
    return spec.get("input_slots", {}).get(input_name)


def _classify_inputs(capture, rewrite):
    alias_by_name = {}
    for semantic, group in ALIASES[capture.mode].items():
        for name in group:
            alias_by_name[name] = semantic
    cut = capture.assembler.cut_inputs
    extra = capture.assembler.extra_feeds
    shape = capture.assembler.shape_feeds
    entries = []
    for value in capture.model.graph.input:
        name = value.name
        elem = value.type.tensor_type.elem_type
        feed = capture.feeds[name]
        base = {
            "graph_name": name,
            "node": _node_number(name),
            "slot": None,
            "submodel_input": None,
            "field_name": None,
            "dtype": _dtype_name(elem),
            "shape": _declared_shape(value),
            "profile_shape": [int(item) for item in np.asarray(feed).shape],
            "storage": "graph_input",
            "public_name": None,
            "frame_offset": 0,
        }
        if name in alias_by_name:
            semantic = alias_by_name[name]
            node, input_name = cut[name]
            base.update({
                "semantic": semantic,
                "slot": _input_slot(capture, node, input_name),
                "submodel_input": input_name,
                "category": "application_input",
                "frame_offset": -1 if semantic == "previous_color" else 0,
                "producer": "application.%s" % semantic,
                "transform": _alias_transform(capture.mode, name),
                "dependencies": [semantic],
                "storage": "public_input",
                "public_name": semantic,
                "evidence": "verified resource alias: %s"
                            % ", ".join(ALIASES[capture.mode][semantic]),
                "status": "supported",
                "description": _SEMANTIC_DESCRIPTION[semantic],
            })
        elif name == LUT_INPUT:
            node, input_name = cut[name]
            base.update({
                "semantic": "gate_lut",
                "slot": _input_slot(capture, node, input_name),
                "submodel_input": input_name,
                "category": "model_asset",
                "frame_offset": None,
                "producer": "package.assets.gate_lut",
                "transform": ["freeze:initializer"],
                "dependencies": [],
                "storage": "initializer",
                "evidence": "d25.t6 frozen %d-value float32 LUT" % LUT_SIZE,
                "status": "supported_frozen",
                "description": "32x32 gate LUT decoded to float32",
            })
        elif name in _CONFIG_SEMANTIC:
            semantic, description, evidence = _CONFIG_SEMANTIC[name]
            node, input_name = cut[name]
            base.update({
                "semantic": semantic,
                "submodel_input": input_name,
                "field_name": input_name,
                "category": "configuration",
                "frame_offset": 0,
                "producer": "planner.d%04d.%s" % (node, input_name),
                "transform": [],
                "dependencies": [],
                "evidence": evidence,
                "status": "supported_fixed",
                "description": description,
            })
        elif name in extra:
            node, key = extra[name]
            base.update({
                "semantic": "planner_allocation_parameter",
                "field_name": key,
                "category": "planner_allocation",
                "frame_offset": 0,
                "producer": "planner.d%04d.%s" % (node, key),
                "transform": [],
                "dependencies": [],
                "evidence": "assembler.extra_feeds",
                "status": "supported_profile_fixed",
                "description": "Allocation packing parameter",
            })
        elif name in shape:
            node, input_name = shape[name]
            base.update({
                "semantic": "shape_parameter",
                "field_name": "%s_shape" % input_name,
                "category": "shape_parameter",
                "frame_offset": 0,
                "producer": "planner.d%04d.%s" % (node, input_name),
                "transform": [],
                "dependencies": [],
                "evidence": "assembler.shape_feeds",
                "status": "supported_profile_fixed",
                "description": "Profile-fixed runtime shape feed",
            })
        elif name in cut:
            node, input_name = cut[name]
            is_pad = "pads" in input_name or "pad" in name
            base.update({
                "semantic": "planner_pad" if is_pad else "planner_scalar",
                "submodel_input": input_name,
                "field_name": input_name,
                "category": "planner_pad" if is_pad else "planner_scalar",
                "frame_offset": 0,
                "producer": "planner.d%04d.%s" % (node, input_name),
                "transform": [],
                "dependencies": [],
                "evidence": "assembler.cut_inputs",
                "status": "supported_profile_fixed",
                "description": ("Padding parameter" if is_pad
                                else "Planner scalar"),
            })
        else:
            raise ExportError("unclassified original input %s" % name)
        entries.append(base)
    return entries


def _alias_transform(mode, name):
    if name in FLAT_ALIASES:
        return ["reshape:HW4_or_HmWm_to_flat"]
    if name in CAST_FLAT_ALIASES:
        return ["cast:float16_to_float32", "reshape:flat"]
    return []


def _node_number(name):
    match = re.match(r"^n(\d{4})_", name)
    return int(match.group(1)) if match else None


def _classify_outputs(capture):
    slots = {}
    for number, slot, canonical, _elem in capture.assembler.output_specs:
        slots[canonical] = (number, slot)
    entries = []
    for value in capture.model.graph.output:
        name = value.name
        number, slot = slots.get(name, (_node_number(name), None))
        family = capture.specs[number]["family"] if number in capture.specs \
            else None
        final = name == FINAL_OUTPUT
        entry = {
            "graph_name": name,
            "node": number,
            "slot": slot,
            "dtype": _dtype_name(value.type.tensor_type.elem_type),
            "shape": _declared_shape(value),
            "semantic": "generated_frame" if final else family,
            "category": "public_output" if final else "debug_intermediate",
            "frame_offset": 0,
            "producer": "assembler.d%04d.%s" % (number, slot),
            "transform": ["identity:generated_frame"] if final else [],
            "dependencies": (list(ALIASES[capture.mode]) if final else []),
            "storage": "graph_output",
            "public_name": PUBLIC_OUTPUT if final else None,
            "evidence": ("assembled final composite" if final
                         else "assembled dispatch output"),
            "status": "supported" if final else "debug_only",
            "description": ("Public interpolated frame" if final
                            else "Raw assembled intermediate"),
        }
        if final:
            entry["public_shape"] = ["H", "W", 4]
        entries.append(entry)
    return entries


def _initializer_classification(name, tensor):
    array = numpy_helper.to_array(tensor)
    category = "model_asset"
    subcategory = "fixed_parameter"
    description = "Fixed algorithm/table constant"
    binding = "lowered_dispatch_initializer"
    if name == LUT_INPUT:
        category = "model_asset"
        subcategory = "gate_lut"
        description = "32x32 gate LUT decoded to float32"
        binding = "capture.d25.t6.folded_asset"
    elif name.startswith("n0024_") and "sp_scale" in name:
        category = "configuration"
        subcategory = "splat_motion_scale"
        description = ("d24 baked splat scale; configuration, not a network "
                       "weight")
        binding = "d24.splat.baked_configuration"
    elif "seed" in name.lower():
        subcategory = "seed"
        description = "GEMV algorithmic seed table; not temporal history"
        binding = "gemv.algorithmic_seed"
    else:
        lowered = name.lower()
        if "weight" in lowered or lowered.endswith("_w"):
            subcategory = "weight"
            description = "Convolution/GEMV weight"
        elif "bias" in lowered:
            subcategory = "bias"
            description = "Convolution/GEMV bias"
        elif "activation" in lowered:
            subcategory = "activation"
            description = "Activation table"
        elif "scale" in lowered or "scales" in lowered:
            subcategory = "scale"
            description = "Quantisation scale"
        elif any(token in lowered for token in
                 ("qlo", "qhi", "zero", "zp", "pos_", "neg_")):
            subcategory = "requant"
            description = "Quantisation/requantisation constant"
        origin = _node_number(name)
        binding = ("d%04d.lowered_model_initializer" % origin
                   if origin is not None else "assembler.initializer")
    if "_to_" in name or name.startswith("sdk_"):
        binding = "assembler.edge_adapter_constant"
    status = "baked_configuration" if category == "configuration" \
        else "frozen_asset"
    return {
        "name": name,
        "dtype": _dtype_name(tensor.data_type),
        "shape": [int(item) for item in tensor.dims],
        "count": int(array.size),
        "category": category,
        "subcategory": subcategory,
        "origin": ("d%04d" % _node_number(name)
                   if _node_number(name) is not None else "assembler"),
        "binding_origin": binding,
        "status": status,
        "description": description,
    }


def _classify_initializers(model):
    items = [_initializer_classification(tensor.name, tensor)
             for tensor in model.graph.initializer]
    items.sort(key=lambda item: item["name"])
    counts = {}
    for item in items:
        key = item["category"] + "/" + item["subcategory"]
        counts[key] = counts.get(key, 0) + 1
    return {
        "count": len(items),
        "set_sha256": _initializer_digest(model.graph.initializer),
        "by_category": counts,
        "items": items,
    }


def _io_contract(captures, rewrites, profiles_by_mode):
    source = _source(captures, rewrites)
    variants = {}
    for mode in sorted(set(capture.mode for capture in captures)):
        representative = next(capture for capture in captures
                              if capture.mode == mode)
        rewrite = rewrites[representative.run_dir]
        variant_profiles = sorted(profile["id"]
                                  for profile in profiles_by_mode[mode])
        variants["mode%d" % mode] = {
            "mode": mode,
            "graph": {
                "original_inputs": rewrite.info["original_inputs"],
                "original_outputs": rewrite.info["original_outputs"],
                "original_initializers":
                    len(representative.model.graph.initializer),
                "packaged_inputs": rewrite.info["public_graph_inputs"],
                "packaged_outputs": rewrite.info["public_graph_outputs"],
                "packaged_initializers": rewrite.info["initializer_count"],
            },
            "public_inputs": _public_input_contract(mode),
            "binding": {
                semantic: {
                    "public_name": semantic,
                    "aliases": list(group),
                    "frame_offset": -1 if semantic == "previous_color" else 0,
                }
                for semantic, group in ALIASES[mode].items()
            },
            "inputs": _classify_inputs(representative, rewrite),
            "outputs": _classify_outputs(representative),
            "initializers": _classify_initializers(rewrite.public_model),
            "profile_ids": variant_profiles,
            "unsupported_input_policy":
                "reject unknown external tensor inputs; never zero-fill",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "name": PACKAGE_NAME,
        "source": source,
        "input_contract": dict(INPUT_CONTRACT),
        "variants": variants,
        "limits": list(LIMITATIONS),
        "references": list(REFERENCES),
    }


def _debug_output_map(capture):
    return {
        "schema_version": SCHEMA_VERSION,
        "variant": "mode%d" % capture.mode,
        "model": "debug/mode%d.onnx" % capture.mode,
        "public_output": PUBLIC_OUTPUT,
        "outputs": _classify_outputs(capture),
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_out_dir(out_dir):
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise ExportError("output path is not a directory: %s" % out_dir)
        if os.listdir(out_dir):
            raise ExportError("output directory is not empty: %s" % out_dir)
    else:
        os.makedirs(out_dir)
    return out_dir


def export_package(run_dirs, out_dir, frame=1, debug=False):
    """Build and write the package.  Returns the manifest payload."""
    out_dir = _prepare_out_dir(out_dir)
    if not run_dirs:
        raise ExportError("no capture runs supplied")
    captures = [build_capture(path, frame) for path in run_dirs]

    rewrites = {}
    by_mode = {}
    for capture in captures:
        rewrite = rewrite_graph(capture.model, capture.feeds, capture.mode,
                                want_debug=debug)
        rewrites[capture.run_dir] = rewrite
        by_mode.setdefault(capture.mode, []).append(capture)

    # One model per variant.  Different sizes/frames must serialize to the
    # same graph and must carry the same assets; otherwise this is not a
    # portable profile family.
    for mode, group in by_mode.items():
        reference = rewrites[group[0].run_dir]
        reference_bytes = reference.public_model.SerializeToString()
        for capture in group[1:]:
            current = rewrites[capture.run_dir]
            if current.public_model.SerializeToString() != reference_bytes:
                raise ExportError(
                    "mode %d graph differs between %s and %s after rewrite"
                    % (mode, group[0].run_dir, capture.run_dir))
            if (current.info["initializer_digest"]
                    != reference.info["initializer_digest"]):
                raise ExportError(
                    "mode %d initializers differ between %s and %s"
                    % (mode, group[0].run_dir, capture.run_dir))

    models_dir = os.path.join(out_dir, "models")
    profiles_dir = os.path.join(out_dir, "profiles")
    os.makedirs(models_dir)
    os.makedirs(profiles_dir)

    profiles_by_mode = {}
    variants = {}
    for mode in sorted(by_mode):
        group = by_mode[mode]
        representative = group[0]
        rewrite = rewrites[representative.run_dir]
        model_path = os.path.join(out_dir, "models", "mode%d.onnx" % mode)
        onnx.save(rewrite.public_model, model_path)
        model_sha = _file_sha256(model_path)
        if model_sha != rewrite.info["public_model_sha256"]:
            raise ExportError("saved model hash changed for %s" % model_path)

        registry = ProfileRegistry()
        for capture in group:
            registry.add(_build_profile(capture, rewrites[capture.run_dir]))
        profiles = registry.profiles()
        for profile in profiles:
            _write_json(os.path.join(profiles_dir, profile["id"] + ".json"),
                        profile)
        profiles_by_mode[mode] = profiles

        variants["mode%d" % mode] = {
            "mode": mode,
            "model": "models/mode%d.onnx" % mode,
            "model_sha256": model_sha,
            "output": PUBLIC_OUTPUT,
            "inputs": [entry[0] for entry in PUBLIC_INPUTS[mode]],
            "profiles": ["profiles/%s.json" % profile["id"]
                         for profile in profiles],
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": PACKAGE_NAME,
        "source": _source(captures, rewrites),
        "conventions": dict(CONVENTIONS),
        "input_contract": dict(INPUT_CONTRACT),
        "variants": variants,
        "limitations": list(LIMITATIONS),
    }

    if debug:
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(debug_dir)
        debug_manifest = {}
        for mode in sorted(by_mode):
            representative = by_mode[mode][0]
            rewrite = rewrites[representative.run_dir]
            if rewrite.debug_model is None:
                raise ExportError("debug model was not built for mode %d"
                                  % mode)
            model_rel = "debug/mode%d.onnx" % mode
            onnx.save(rewrite.debug_model,
                      os.path.join(out_dir, model_rel))
            map_rel = "debug/mode%d.outputs.json" % mode
            _write_json(os.path.join(out_dir, map_rel),
                        _debug_output_map(representative))
            debug_manifest["mode%d" % mode] = {
                "model": model_rel,
                "outputs": map_rel,
                "public_output": PUBLIC_OUTPUT,
            }
        manifest["debug"] = debug_manifest

    _write_json(os.path.join(out_dir, "manifest.json"), manifest)
    _write_json(os.path.join(out_dir, "io_contract.json"),
                _io_contract(captures, rewrites, profiles_by_mode))
    return manifest


def main(argv=None):
    """Backward-compatible alias for the supported ``xefg_sdk.export`` CLI."""
    from .cli import main as cli_main
    return cli_main(argv)
