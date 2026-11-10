# XeFG 2x model package

A model package is the deployable, language-neutral input to
`Xefg2xRuntime.load`. It contains ONNX models, a manifest, and per-profile
metadata. It contains no application frames, capture feeds, expected outputs,
or debug trace tensors as inference dependencies.

This document describes API version `0.2.0`. The model-package schema is `1`.
Packages produced for version 0.1 remain compatible; they need no migration or
model rebuild. The exporter writes the manifest and profile files from retained
captures, and those generated files are authoritative.

**This repository ships source only. It does not contain ONNX models, weights,
quantisation constants, the gate LUT, or profile data for a particular capture.**
A package must be exported from captures you supply, or obtained separately.
See [Ready-to-use profiles](PROFILES.md) for the sizes the profiles describe and
[Capture guide](CAPTURE_GUIDE.md) to create captures using this checkout.
[Capture format](CAPTURE_FORMAT.md) defines the producer/exporter interface.

## 1. Directory layout

```text
xefg2x-package/
  manifest.json
  io_contract.json
  models/
    mode1.onnx
    mode3.onnx
  profiles/
    mode1-1280x720-mv640x360.json
    mode3-1280x720-mv640x360.json
  debug/                         # optional: --debug only
    mode1.onnx
    mode1.outputs.json
```

`io_contract.json` is written by the exporter as provenance and is ignored by
the runtime. `debug/` is optional and is never required to deploy.

All paths in `manifest.json` and profile files are relative to the package
directory. Paths that resolve outside the package are rejected.

## 2. Manifest

```json
{
  "schema_version": 1,
  "name": "xefg2x",
  "conventions": {
    "color_format": "BGRA8_UNORM",
    "motion_format": "RG16_FLOAT",
    "motion_units": "mv_pixels",
    "motion_direction": "current_to_previous",
    "depth_inverted": false,
    "jittered_motion": false,
    "ui_premultiplied": true
  },
  "variants": {
    "mode1": {
      "mode": 1,
      "model": "models/mode1.onnx",
      "model_sha256": "<64 hex characters>",
      "inputs": [
        "previous_color",
        "current_color",
        "motion_vectors",
        "depth"
      ],
      "output": "generated_frame",
      "profiles": [
        "profiles/mode1-1280x720-mv640x360.json"
      ]
    },
    "mode3": {
      "mode": 3,
      "model": "models/mode3.onnx",
      "model_sha256": "<64 hex characters>",
      "inputs": [
        "previous_color",
        "current_color",
        "motion_vectors",
        "depth",
        "ui"
      ],
      "output": "generated_frame",
      "profiles": [
        "profiles/mode3-1280x720-mv640x360.json"
      ]
    }
  }
}
```

The exporter also writes `source` and `limitations`; the runtime ignores them.
A variant may list several profiles for different output/motion sizes. The
exported variant IDs are `mode1` and `mode3`.

`conventions` describes the required **model-boundary** representation that any
caller or engine adapter must supply. It is not a description of raw engine
metadata and it is validated by the runtime:

- every convention present in the manifest must be a known key with exactly the
  value shown above;
- an unknown key or a different value raises `ValueError`;
- the runtime uses these values to reject a package whose boundary does not
  match the implementation.

The conventions are part of schema `1`. Version 0.1 packages that carry the
same values remain valid.

Validation performed by `ModelPackage`:

- `schema_version` must be `1`;
- every manifest convention must match the required model-boundary value;
- `variants` must be a non-empty object;
- every variant `mode` must be `1` or `3`;
- the model file exists and its SHA-256 matches `model_sha256`;
- every profile is readable, has `schema_version: 1`, and declares the same
  variant ID;
- a variant has at least one profile.

## 3. Public tensor inputs

The package exposes only these application resources to the model:

| Mode | UI mode | Public inputs |
| --- | --- | --- |
| 1 | `NONE` | `previous_color`, `current_color`, `motion_vectors`, `depth` |
| 3 | `HUDLESS_UITEXTURE` | `previous_color`, `current_color`, `motion_vectors`, `depth`, `ui` |

That is four tensors in mode 1 and five in mode 3. Colours and UI are
`BGRA8_UNORM` `uint8 (H, W, 4)`; motion vectors are `float16 (h, w, 2)` after
the SDK's single precision conversion; depth is raw fg78 non-inverted
`float32 (h, w)` (smaller is nearer).

The variant's `inputs` set must exactly equal the resources selected by the UI
mode. The SDK does not rename public resources to `dN.tN` slots. Internal
profile metadata is added by the package/binding layer and cannot replace a
public input.

Four or five is the **application-facing** input count, not the total serialized
graph input count. In the inspected packages the public graph has 311 inputs for
mode 1 and 312 for mode 3: the 4 or 5 public image inputs plus the 307
profile-fixed metadata feeds. Those 307 feeds break down as 285 `int64` and 22
`float32` scalars, pads, shapes, and allocation constants. They are supplied by
the profile, not by the application, and they are not capture tensors.

The package boundary is already normalized: motion vectors are complete
`current_to_previous` pixel displacements on the motion-vector grid, with
positive X right and positive Y down, including camera motion and already
jitter compensated. Depth is the raw fg78 non-inverted value; the runtime does
not invert, linearize, clamp, or rescale it.

## 4. Profile metadata

A profile describes one supported output/motion size and its internal metadata:

```json
{
  "schema_version": 1,
  "id": "mode1-1280x720-mv640x360",
  "variant": "mode1",
  "output_size": [1280, 720],
  "motion_size": [640, 360],
  "feeds": {
    "n0025_wi_motion_scale": {
      "dtype": "float32",
      "shape": [1],
      "values": [4.0]
    },
    "n0025_wi_ratio": {
      "dtype": "float32",
      "shape": [1],
      "values": [1.0]
    }
  }
}
```

- `output_size` is `[width, height]` of the generated frame.
- `motion_size` is `[width, height]` of the motion/depth grid.
- `id` is the profile selector exposed as `Config.profile_id`.
- `feeds` holds profile-fixed scalars and small shape/phase/padding/allocation
  values, each with `int64` or `float32` values. The planner reconstructs the
  array from `values` and `shape` and marks it read-only. The frozen 32x32 gate
  LUT is a model initializer, not a profile feed.

The exporter's inspected profiles carry 307 feeds, each with at most 8 values:
285 `int64` and 22 `float32`. Combined with the public image inputs, that gives
the 311-input mode-1 graph and 312-input mode-3 graph. The exact feed set is
package data, not an API promise.

Profiles include a `configuration` object describing normalized input and model
configuration, including `interpolation_factor`, `interpolation_position`,
`motion_vector_resolution`, `full_rect`, and baked/profile-fixed internal scales.
The planner uses `feeds` rather than interpreting this descriptive object. The
current exporter removes only raw engine fields (camera matrices, native MV
scale and jitter offsets) from `configuration`; reference capture settings are
recorded separately under the manifest's `source`. Older packages that retain
those raw source fields remain compatible; they do not impose camera or
coordinate-system restrictions on applications.

The planner resolves exactly one profile by:

1. variant `mode`;
2. `output_size` equal to the current colour size;
3. `motion_size` equal to the current motion/depth size;
4. optional `profile_id`.

Zero matches or multiple matches raise. This is deliberate: the SDK does not
guess a nearby profile or resample to an arbitrary resolution.

Profile metadata is imported from the exporter. Formula-generated profiles are
future work. Version 0.2.0 supports the known profiles present in the package
and makes no arbitrary-resolution claim.

## 5. Final output and optional debug outputs

A deployable variant declares one `output` tensor: the final generated frame.
The runtime requests only that name and checks that the executor returns
`uint8 (H, W, 4)` at the profile's output size.

The inspected graph has 42 raw outputs. With `--debug`, the exporter writes
`debug/modeN.onnx`, `debug/modeN.outputs.json`, and a matching `debug` object in
`manifest.json`. Those outputs are for validation and investigation only. They
are not part of the application contract and are not required for deployment. A
deployable package returns only `generated_frame`; the SDK does not add implicit
cross-frame feature history for debug outputs.

The exported public model is plain ONNX opset 18 and IR 8. ONNX validity does
not mean every backend or NPU can execute the mixed integer/float operator set.
The runtime does not import the exporter or any trace module.

## 6. What the package must not contain

The runtime package must not contain or depend on:

- application frame images, capture runs, or capture datasets;
- captured ONNX feeds used as runtime inputs;
- expected outputs or reference images;
- trace JSON, dispatch slot names (`dN.tN`), shader plans, or DXIL;
- provider-specific game fixes or unit guesses.

Regression fixtures and provenance files belong outside the deployable package
or in an optional diagnostic file that the runtime ignores.

## 7. Runtime use

```python
from xefg_sdk import Xefg2xRuntime

with Xefg2xRuntime.load("model_package") as runtime:
    result = runtime.interpolate_pair(previous, current, config)
```

- `Xefg2xRuntime.load(package_path, executor=None)` validates the package and
  uses the optional ONNX Runtime executor when no executor is supplied.
- `runtime.prepare(previous, current, config)` resolves the mode, profile, and
  internal feeds without loading an inference engine.
- `runtime.interpolate_pair(...)` loads the variant model once per runtime,
  binds public resources and profile metadata, and returns an
  `InterpolationResult`.
- `runtime.close()` releases executor handles. The runtime is also a context
  manager.

A frame pair carries only `FrameInputs`; there is no constants object. History
is owned by `Xefg2xSession.submit(frame)` and its per-frame `reset_history`
flag.

The executor protocol is small:

```python
class Executor:
    def load(self, model_path): ...
    def run(self, handle, inputs, output_names): ...
    def close(self, handle): ...
```

`OrtExecutor` is optional. The core SDK imports only NumPy; installing the
`ort` extra adds ONNX Runtime. See [Quickstart](QUICKSTART.md) for installation.
