# XeFG 2x SDK input/output contract

This document defines the application-facing data that the Python XeFG 2x SDK
consumes and returns. It describes normalized application frame inputs, not a
capture trace, an ONNX feed list, or a swapchain ABI.

The runtime API is in `src/xefg_sdk`; the examples in `examples/` read exactly
this contract. API version is `0.3.0`; the model-package schema is `1`. Existing
schema-1 packages remain compatible; a runtime older than 0.3.0 does not know
the mode-4 variant.

## 1. Scope

The caller supplies one frame at a time through `FrameInputs`:

- previous and current colour;
- current motion vectors;
- current depth;
- current UI for `HUDLESS_UITEXTURE`, or the current composed backbuffer for
  `BACKBUFFER_HUDLESS`;
- a per-frame `reset_history` flag.

The SDK supplies internally:

- model inputs derived from the application images;
- profile metadata from the model package;
- the generated frame.

These inputs are already normalized to the model boundary. A future engine
adapter, such as an OptiScaler integration, is responsible for producing them;
this SDK does not include an adapter implementation. In particular the adapter,
not the core, must complete motion vectors with camera motion, compensate
jitter, and adapt engine depth to the model depth convention.

The application never supplies `dN.tN` capture slots, ONNX tensor names, debug
outputs, or expected results. Those are not part of this contract. The public
runtime boundary has four application image inputs for mode 1, or five for mode
3 and mode 4. The serialized ONNX graph also has, in the inspected packages, 307
profile-filled metadata inputs (311 total inputs for mode 1, 312 for mode 3).
The mode-4 public model has 312 total inputs and 307 profile-filled feeds for
all five sizes. Those metadata feeds are package data and never application
inputs.

## 2. Sequence manifest

A sequence is a JSON file plus frame files. File paths are resolved relative to
the manifest directory. The manifest is application input; it is not a capture
trace manifest.

```json
{
  "schema_version": 1,
  "package": "model_package",
  "config": {
    "ui_mode": "AUTO",
    "profile_id": null
  },
  "frames": [
    { "path": "frames/frame_0000.npz", "reset_history": true },
    { "path": "frames/frame_0001.npz" }
  ]
}
```

Top-level keys:

| Key | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | Must be `1`. |
| `package` | unless `--package` | Model package directory. |
| `config` | no | Sequence configuration; see section 5. |
| `frames` | yes | Non-empty array of frame entries. |
| `description` | no | Free-form text; ignored by the SDK. |

There is no top-level `constants` object in version 0.3.0 (unchanged from
0.2.0). Per-frame history reset belongs on the frame entry; see section 3.

The examples read the keys below and ignore extra descriptive keys.

## 3. Frame entries

A frame entry is either an `.npz` path or an explicit field object. A frame
entry may also carry `reset_history`:

```json
{ "path": "frames/frame_0000.npz", "reset_history": true }
```

`reset_history` is optional and defaults to `false`. It becomes
`FrameInputs.reset_history` for that frame. A frame with `reset_history: true`
starts a new history in `Xefg2xSession`; it does not form a pair with the
previously submitted frame.

### 3.1 Simple `.npz` frames

The preferred form stores one natural image array per application field:

```json
{ "path": "frames/frame_0000.npz" }
```

The archive may contain:

| Array | Shape | Meaning |
| --- | --- | --- |
| `frame_id` | scalar | Pairing key. |
| `reset_history` | boolean scalar | Optional history reset; a JSON frame-entry value takes precedence. |
| `backbuffer_color` | `(H, W, 4)` | Application colour for mode `NONE`, or the current composed backbuffer for mode `BACKBUFFER_HUDLESS`. |
| `hudless_color` | `(H, W, 4)` | No-UI colour for mode `HUDLESS_UITEXTURE` or `BACKBUFFER_HUDLESS`. |
| `motion_vectors` | `(h, w, 2)` | Current-to-previous vectors on the MV grid. |
| `depth` | `(h, w)` | Raw fg78 non-inverted depth on the MV grid. |
| `ui` | `(H, W, 4)` | Premultiplied UI for mode `HUDLESS_UITEXTURE`. |

The selected mode requires its colour field (`backbuffer_color` for `NONE`;
`hudless_color` for `HUDLESS_UITEXTURE`; both previous and current
`hudless_color` plus current `backbuffer_color` for `BACKBUFFER_HUDLESS`), plus
`motion_vectors` and `depth`; `ui` is additionally required for
`HUDLESS_UITEXTURE` and is not used by `BACKBUFFER_HUDLESS`. `H`/`W` are the
output colour dimensions; `h`/`w` are the half-resolution motion/depth grid used
by the current profiles. All tensors are application images, not model inputs.

If the JSON entry also has `frame_id`, it must equal the archive scalar.

Formats are inferred for `.npz` arrays:

| Array | Inferred format |
| --- | --- |
| `backbuffer_color`, `hudless_color`, `ui` | `BGRA8_UNORM` for `uint8 (H, W, 4)` |
| `motion_vectors` | `RG16_FLOAT` for `float16 (h, w, 2)`, `RG32_FLOAT` for `float32` |
| `depth` | `R32_FLOAT` for `float32 (h, w)` or `(h, w, 1)` |

`BGRA8_UNORM` is the default because it is the common application colour
layout. If the array is already `RGBA8_UNORM`, say so explicitly:

```json
{
  "path": "frames/frame_0000.npz",
  "formats": { "backbuffer_color": "RGBA8_UNORM" }
}
```

Unsupported formats raise. A `formats` entry for an array that is not present
is ignored.

### 3.2 Explicit `.npy` and raw fields

Use an explicit field object when the application already has separate files or
raw buffers:

```json
{
  "frame_id": 1,
  "fields": {
    "backbuffer_color": { "file": "bb_0001.npy", "format": "BGRA8_UNORM" },
    "motion_vectors": {
      "file": "mv_0001.raw",
      "format": "RG16_FLOAT",
      "width": 640,
      "height": 360
    },
    "depth": {
      "file": "depth_0001.raw",
      "format": "R32_FLOAT",
      "width": 640,
      "height": 360
    }
  }
}
```

- `.npy` files take their shape from the array; `width`/`height` are ignored.
- Raw files require explicit `width`, `height`, and `format`. The element count
  must equal `width * height * channels`; a short or oversized file raises.
- Raw files are native-endian; on Windows x64 that is little-endian.
- `file` is relative to the manifest directory unless absolute.
- `reset_history` is a sibling key of `frame_id` and `fields`.

## 4. Image formats

| SDK format | NumPy representation | Use |
| --- | --- | --- |
| `BGRA8_UNORM` | `uint8 (H, W, 4)` | Backbuffer, hudless, UI |
| `RGBA8_UNORM` | `uint8 (H, W, 4)` | Backbuffer, hudless, UI |
| `RG16_FLOAT` | `float16 (h, w, 2)` | Motion vectors |
| `RG32_FLOAT` | `float32 (h, w, 2)` | Motion vectors |
| `R32_FLOAT` | `float32 (h, w)` | Depth |

The SDK copies accepted data to contiguous arrays. Version 0.3.0 supports
full-resource rectangles only; non-zero sub-rectangles raise instead of being
silently cropped. `ImageView.rect` may be omitted.

The core performs only two type conversions:

- `RGBA8_UNORM` colour and UI are reordered to `BGRA8_UNORM`;
- `RG32_FLOAT` motion vectors are rounded once to `RG16_FLOAT`.

It does not normalize coordinate conventions, rescale channels, or reinterpret
depth.

## 5. Configuration

```json
{
  "ui_mode": "AUTO",
  "profile_id": null
}
```

| Key | Default | Version 0.3.0 support |
| --- | --- | --- |
| `ui_mode` | `AUTO` | All enum names parse; `AUTO`, `NONE`, `HUDLESS_UITEXTURE`, and `BACKBUFFER_HUDLESS` supported. |
| `profile_id` | `null` | Optional exact profile selector. |

Version 0.3.0 has no motion-unit, motion-direction, depth-inversion,
jittered-motion, or UI-premultiplication switches. Those conventions are fixed
at the model boundary and are supplied already satisfied by the caller or
adapter; see section 6.

Unsupported values raise `ValueError`; they are never silently ignored. This
includes the public enum values `BACKBUFFER_UITEXTURE` and
`BACKBUFFER_HUDLESS_UITEXTURE`: the names exist for semantic alignment, but
version 0.3.0 has no model path for them.

`AUTO` resolves using the supplied resources and is unchanged from 0.2.0:

- both hudless colour and `ui` present -> `HUDLESS_UITEXTURE`;
- otherwise both backbuffer colour present -> `NONE`;
- neither -> `ValueError`.

`AUTO` never selects mode 4. To use `BACKBUFFER_HUDLESS`, set it explicitly:

```python
from xefg_sdk import Config, UiMode

Config(ui_mode=UiMode.BACKBUFFER_HUDLESS)
```

The resolved mode is reported on `InterpolationResult.ui_mode`.

## 6. Frame inputs and conventions

A `FrameInputs` object contains:

| Field | Meaning |
| --- | --- |
| `frame_id` | Integer pairing key. |
| `backbuffer_color` | `ImageView` for mode `NONE`, or the current composed backbuffer for `BACKBUFFER_HUDLESS`; absent otherwise. |
| `hudless_color` | `ImageView` for mode `HUDLESS_UITEXTURE` or `BACKBUFFER_HUDLESS`; absent otherwise. |
| `depth` | `R32_FLOAT` current-frame depth. |
| `motion_vectors` | `RG16_FLOAT` or `RG32_FLOAT` current-frame motion vectors. |
| `ui` | Premultiplied current-frame UI for `HUDLESS_UITEXTURE`, or absent. |
| `reset_history` | Boolean, default `false`; starts a new session history. |

The required conventions are:

- Colour and UI are `BGRA8_UNORM` or `RGBA8_UNORM`; `RGBA8_UNORM` is converted
  to BGRA by the core. UI is premultiplied.
- Mode 4 supplies the current presented backbuffer as `backbuffer_color`. It is
  the image after UI has been composited, not a separate UI texture. There is no
  previous-backbuffer input, and `ui` is not used.
- Motion vectors are `current_to_previous` displacements on the motion-vector
  grid, in pixels, with positive X to the right and positive Y down. They must
  include camera motion and must already be jitter compensated. The core does
  not add camera motion, remove jitter, flip signs, convert NDC, or apply a
  scale.
- Depth is the raw fg78 non-inverted model-bound depth: `R32_FLOAT`, where a
  smaller value is nearer. The core does not invert, linearize, clamp, or
  rescale it. Depth is not converted to metres and is not rescaled to `0..1`.
- A generated frame always represents the midpoint of the supplied
  previous/current pair: `interpolation_fraction` is fixed at `0.5`.

"Normalized inputs" in this contract means the conventions above. It does not
mean that every depth value is rescaled to `0..1`, nor that the core adapts an
engine's raw matrices, jitter, motion-vector units, or depth encoding. The model
package manifest asserts the same values in `conventions`; they describe this
model boundary, not raw engine metadata.

Depth detail: the captured fg78 graph derives its own depth key internally as
`clip(trunc((1 - d) * 32767), 0, 32767) << 16`, and the `d25` missing-splat
fallback uses `d - 1000`. Those are graph behaviours after the model boundary,
not core input transformations. The analytical probe scene contains only depth
values `1.0` and `100.0`, so this SDK does not claim that the full standard
device-depth range has been proven.

Version 0.3.0 does not accept frame render time, view or projection matrices,
jitter offsets, motion-vector scales, or NDC/inversion switches. They were
removed from the public API. Any such engine metadata must be resolved by the
adapter before `FrameInputs`.

## 7. Frame pairing

- `frame_id` is an integer pairing key and must increase between the previous
  and current frame in a pair.
- The direct pair API consumes previous colour and current colour.
- Motion vectors, depth, and UI are taken from the current frame.
- For mode 4, the previous and current colour fields are the hudless pair; the
  current `backbuffer_color` is taken from the current frame. No previous
  backbuffer is read.
- A `FrameInputs` object may carry motion/depth/UI on every frame; the SDK uses
  the current frame's values for the pair.
- The UI input is the current frame's UI. The retained test scene holds UI
  fixed, so a temporally varying UI has not yet been distinguished from the
  previous frame in direct evidence; the API deliberately selects the current
  UI.
- A sequence session snapshots each submitted frame before returning, so the
  application may reuse its resources immediately after `submit`.
- `reset_history=True` on the current frame invalidates a direct pair. Submit
  through `Xefg2xSession` so it can reset, or start a new pair.
- `Xefg2xSession.submit(frame)` accepts a single argument. The first frame after
  construction or reset reports `history_ready`; a frame with
  `reset_history=True` resets the session and also reports `history_ready`.

## 8. Output

`InterpolationResult` contains:

| Field | Meaning |
| --- | --- |
| `status` | `generated` or `history_ready`. |
| `generated_frame` | `ImageView` with `BGRA8_UNORM` data, or `None`. |
| `previous_frame_id` | Previous input ID, or `None` for history setup. |
| `current_frame_id` | Current input ID. |
| `interpolation_fraction` | Fixed `0.5` for a generated pair. |
| `ui_mode` | Resolved UI mode. |
| `profile_id` | Resolved package profile. |

`generated_frame.data` is a `uint8 (H, W, 4)` image at the profile's output
size. It is an ordinary image, not a swapchain buffer and not a model feed.

The examples write the image as `.npy` (BGRA order preserved). With `--png`,
they also write an RGBA PNG using only the Python standard library. PNG output
is optional; no debug reference image, second swapchain, or expected output is
required.

## 9. Errors and dimensions

The SDK raises `ValueError` for:

- missing required application fields for the resolved mode;
- mismatched colour dimensions between previous and current;
- motion and depth with different dimensions;
- UI dimensions different from colour;
- unsupported formats, config values, or UI modes;
- a profile that does not match the exact output and motion sizes;
- non-increasing frame IDs;
- non-full-resource rectangles;
- `reset_history` on the current frame of a direct pair.

Supported profiles are fixed data in the model package. Version 0.3.0 makes no
arbitrary-resolution claim; a frame sequence must match a profile shipped with
the package. Mode-4 profiles exist at all five sizes and are included in the
15-profile package. The exported ONNX graph and SDK runtime reproduce the
project's own research reconstruction exactly on the inspected frames at every
size; 640×360 used independent application images, and larger sizes used
application-shaped arrays recovered from capture resources. That exactness is
against the project's reconstruction, not the provider: official comparisons are
bounded, with the `d25` residual defined in
[Compatibility §5.4](COMPATIBILITY.md#54-d25-sampling-precision).

## 10. Non-goals

- No engine adapter implementation in this SDK; only the `FrameInputs`
  boundary that an adapter targets.
- No capture-trace reader, dataset layout, or debug tensor names in the runtime
  API.
- No `libxess_fg.dll` loading or Intel binary ABI.
- No swapchain creation, Present insertion, or presentation pacing.
- No NPU requirement or supported-NPU claim, and no performance guarantee.
- No claim of last-bit equality with any vendor implementation. Mode 4 is
  supported at the API/package boundary and checked at 640×360 with a small
  remaining provider residual.
