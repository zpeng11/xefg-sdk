# XeFG 2x SDK compatibility

The SDK follows Intel's public XeFG *semantics* for frame resources, UI modes,
motion vectors, depth, and frame parameters. It is a device-neutral ONNX
reference runtime, not a drop-in replacement for Intel's DLL, a DirectX 12 or
Vulkan swapchain, or OptiScaler's presentation path.

This document describes API version `0.3.0`. The model-package schema is `1`.
Packages produced for versions 0.1 and 0.2 remain compatible and need no
migration or model rebuild. A runtime older than 0.3.0 does not understand a
mode-4 variant and rejects it as an unsupported mode.

## 1. Pinned references

| Layer | Reference | Pin |
| --- | --- | --- |
| Intel public XeFG headers | `intel/xess` XeFG enums and resource types, e.g. `inc/xess_fg/xefg_swapchain.h` and `inc/xess_fg/xefg_swapchain_d3d12.h` | `8fe81bdbbaf00b3c1b733fd0d830c333dc84e6f0` |
| OptiScaler XeFG output backend | `OptiScaler/framegen/xefg/XeFG_Dx12.cpp` | `07aa9da670d6f0b9358acc392163b7c1cdc345fe` |
| OptiScaler XeFG input adapter | `OptiScaler/inputs/FG/XeFG_Inputs_Dx12.cpp` | `07aa9da670d6f0b9358acc392163b7c1cdc345fe` |

These pins fix the meaning of the names used here. They do not make this SDK
binary-compatible with either project, and later upstream changes do not
silently widen the supported subset.

## 2. Semantic compatibility, not ABI compatibility

The SDK keeps the public vocabulary:

- UI mode names (`AUTO`, `NONE`, `BACKBUFFER_UITEXTURE`,
  `HUDLESS_UITEXTURE`, `BACKBUFFER_HUDLESS`,
  `BACKBUFFER_HUDLESS_UITEXTURE`);
- resource roles: backbuffer colour, hudless colour, depth, motion vectors, UI;
- normalized frame inputs: colour, depth, motion vectors, UI, and a per-frame
  history reset;
- the notion of a previous/current pair and a generated interpolated frame.

It deliberately does **not** provide:

- Intel DLL exports, COM interfaces, struct layouts, or ABI-compatible
  descriptors;
- a D3D12/Vulkan swapchain, `GetSwapChainPtr`, Present insertion, or
  `framesPresented` state;
- descriptor heaps, resource states, or command-list recording;
- a replacement swapchain buffer.

`generated_frame` is an in-memory `BGRA8_UNORM` image returned to the caller.
The application keeps its original swapchain and decides how to present or copy
the generated image. A future OptiScaler or graphics-API adapter may hand the
image to a presentation chain, independently of the inference core.

## 3. Version history

### 3.1 Version 0.3.0: mode 4 (`BACKBUFFER_HUDLESS`)

Version 0.3.0 adds a mode-4 model path alongside the existing modes 1 and 3:

- mode 4 takes the previous/current **hudless** colour pair as `previous_color`
  and `current_color`, plus the current **backbuffer** as the public `backbuffer`
  input;
- the backbuffer is the current presented image with UI already composed; mode 4
  does not consume or require a separate UI texture;
- motion vectors and depth are still the current frame's values;
- `AUTO` is unchanged: it resolves only to `NONE` or `HUDLESS_UITEXTURE`. Mode 4
  is selected explicitly with `Config(ui_mode=UiMode.BACKBUFFER_HUDLESS)`;
- the model-package schema remains `1`. A 0.3.0 runtime loads older schema-1
  packages; a pre-0.3.0 runtime refuses a mode-4 variant.

The capture-side mapping is recorded in
[Capture format](CAPTURE_FORMAT.md#4-frames-and-dispatch-normalization): `t8` is
the current hudless colour and aliases `current_color`, `t10` is the current
backbuffer, and `t9` is a bound placeholder that mode 4 does not use.

### 3.2 Version 0.2.0 public API simplification

Version 0.2.0 keeps the core small and moves engine adaptation to the boundary:

- `FrameConstants` is deleted. `FrameInputs` carries only `frame_id`,
  `backbuffer_color`, `hudless_color`, `depth`, `motion_vectors`, `ui`, and
  `reset_history` (default `false`).
- `Config` carries only `ui_mode` and `profile_id`.
- `Xefg2xSession.submit(frame)` takes no constants argument.
- There is no frame render time, view/projection matrix, jitter offset,
  motion-vector scale, NDC switch, or depth-inversion switch in the core.
- The core still performs the `RGBA8_UNORM` to `BGRA8_UNORM` and
  `RG32_FLOAT` to `RG16_FLOAT` conversions. It does not normalize coordinate
  conventions.

These are source-level API changes only. The graph, the model-package schema,
and the profile data are unchanged. An existing version 0.1 package is still
accepted by `Xefg2xRuntime.load` when its manifest conventions and profiles are
the same.

## 4. Public enum and resource coverage

Version 0.3.0 exposes every `UiMode` name, but only these model paths exist:

| `UiMode` | Value | Version 0.3.0 |
| --- | --- | --- |
| `AUTO` | 0 | Resolves to `NONE` or `HUDLESS_UITEXTURE` from supplied resources; never mode 4. |
| `NONE` | 1 | Supported; interpolates `backbuffer_color`. |
| `BACKBUFFER_UITEXTURE` | 2 | Named only; raises. |
| `HUDLESS_UITEXTURE` | 3 | Supported; interpolates `hudless_color` and composites premultiplied `ui`. |
| `BACKBUFFER_HUDLESS` | 4 | Supported when selected explicitly; interpolates the hudless pair and consumes current `backbuffer_color`. |
| `BACKBUFFER_HUDLESS_UITEXTURE` | 5 | Named only; raises. |

The names 2 and 5 are present for semantic alignment. They are not faked with
zero textures, and they are not silently downgraded. Unsupported modes raise
`ValueError`.

`AUTO` is a runtime resolver, not an exported model variant, and its choices are
unchanged in 0.3.0: it resolves only `NONE` or `HUDLESS_UITEXTURE`. The exporter
can now write `mode4` packages, but only an explicit
`Config(ui_mode=UiMode.BACKBUFFER_HUDLESS)` selects them.

Supported resource formats are `BGRA8_UNORM` and `RGBA8_UNORM` for colour/UI,
`RG16_FLOAT` and `RG32_FLOAT` for motion vectors, and `R32_FLOAT` for depth.
The existence of an upstream enum does not imply support for every format.

## 5. Normalized input semantics

The caller or a future engine adapter supplies inputs that already satisfy the
model-boundary conventions:

- Colour and UI are `BGRA8_UNORM` (or `RGBA8_UNORM`, converted by the core);
  UI is premultiplied.
- Mode 4 additionally supplies the current presented backbuffer as
  `backbuffer_color`. It is the image with application UI already composed, not
  a UI texture, and there is no previous-backbuffer input.
- Motion vectors are `current_to_previous` pixel displacements on the
  motion-vector grid, with positive X right and positive Y down. They include
  camera motion and are already jitter compensated.
- Depth is raw fg78 non-inverted `R32_FLOAT`, where a smaller value is nearer.
- A generated pair is always the `0.5` fraction between its two frame IDs. The
  SDK does not generate an exact intermediate timestamp.
- Rectangles are full-resource only; non-zero sub-rectangles raise.
- The current profiles pair a full-resolution output with a half-resolution
  motion/depth grid.

The core does not accept or infer frame render time, camera matrices, jitter
offsets, motion-vector scales, NDC motion, previous-to-current motion, inverted
depth, or non-premultiplied UI. Those must be resolved before the boundary.
"Normalized" here means the conventions above; it does **not** mean that every
depth value is rescaled to `0..1` or that the core linearizes depth.

### 5.1 Depth contract

The existing model boundary is named **fg78 raw non-inverted** depth:

- the source convention is non-inverted: a smaller value is nearer;
- the model-bound tensor is `R32_FLOAT` and is passed through without core
  inversion, linearization, clamping, or unit conversion;
- the graph derives its own splat depth key internally as
  `clip(trunc((1 - d) * 32767), 0, 32767) << 16`;
- the `d25` missing-splat fallback uses `d - 1000` in the graph.

Those formulas are graph behaviour after the model boundary, not core input
transformations. The retained capture scene contains only depth `1.0` and
`100.0`, so the SDK does not claim that the full standard device-depth range,
a `0..1` range, or a metric-metres range has been proven. An adapter that
targets this contract must adapt its engine depth to the fg78 raw non-inverted
representation; the core does not do that adaptation.

### 5.2 Capture source versus application requirement

The historical reference capture used identity camera matrices and zero jitter.
Those are observations about the capture inputs, not restrictions imposed on
applications by version 0.3.0. The core has no matrix or jitter fields at all:
an application or adapter supplies already-normalized motion vectors and depth,
so non-identity camera transforms and non-zero jitter are handled before the
boundary rather than rejected by it.

### 5.3 Future OptiScaler adapter boundary

No adapter is implemented in this SDK. A future OptiScaler XeFG adapter would
sit between engine resources and `FrameInputs` and would be responsible for:

- selecting the current and previous colour resources and, for mode 3, the
  current UI, or, for mode 4, the current composed backbuffer;
- producing complete `current_to_previous` motion vectors, including camera
  motion, on the model motion-vector grid, in pixels, with positive X right and
  positive Y down, already jitter compensated;
- adapting depth to the fg78 raw non-inverted model contract;
- premultiplying UI for mode 3, or binding the current composed backbuffer for
  mode 4, and choosing `BGRA8_UNORM` or `RGBA8_UNORM` input;
- carrying the per-frame history reset flag into `FrameInputs.reset_history`.

Those responsibilities are an outline of the boundary, not an implementation
claim. The inference core remains package-driven and device-neutral.

## 6. Profile and gate behaviour

The current gate uses a fixed look-up table for the frame contents and profile
resolutions that were inspected. It is a seeded bias, not an accumulated
history: submitting more frames does not change the table. The table is stable
across the inspected frames, content variants, and resolutions.

Profiles are explicit model-package data: output size, motion size, and internal
metadata, with the model mode on the variant. They were imported from the
exporter and are not yet generated by formulas. Version 0.3.0 makes no
arbitrary-resolution claim; a sequence must match a shipped profile. Mode-4
profiles exist at all five sizes through 4K and are in the 15-profile package;
SDK outputs match the full reconstruction exactly on both frames at every size.

## 7. Proven semantic aliases

The following aliases are evidence that the internal graph consumes the
application resources with the stated roles. `dN` is a dispatch number and `tN`
a captured resource slot; they are not SDK input names and must not be supplied
by applications.

| Semantic role | Proven aliases |
| --- | --- |
| Previous colour | `d1.t0` = `d25.t2` = `d38.t11` |
| Current colour | `d1.t1` = `d25.t3` = `d38.t12`; mode 4 also `d38.t8` |
| Motion vectors | `d24.t0` = `d25.t4` |
| Depth | `d24.t1` = `d25.t5` |
| UI (mode 3) | `d38.t9` |
| Current backbuffer (mode 4) | `d38.t10` |
| Gate LUT | `d25.t6` |

For mode 4, `d38.t8` is the current hudless colour and aliases `current_color`;
`d38.t10` is the current backbuffer. `d38.t9` remains a bound placeholder in the
captured composite but is not consumed as UI by mode 4, so the mode-4 package
exposes no `ui` input.

The public SDK exposes only application resources and profile-filled internal
metadata. It never asks the caller for a `tN` slot.

## 8. Frame mapping status

The application-frame mapping is closed for the inspected profiles:

- `traceN` matches application frame `N` (frames 1..5) for 640-wide mode 3 and
  1080-wide mode 1 in the historical source evidence.
- Colour A is application frame `N-1`; colour B is application frame `N`.
- Motion vectors and depth are from application frame `N`.
- UI is the current frame's UI by API choice. The inspected scene holds UI
  fixed, so a temporally varying UI has not yet been distinguished from the
  previous frame in direct evidence.
- Mode 4 uses the same previous/current hudless pair for colour, the current
  frame's motion/depth, and the current frame's composed backbuffer. The mode-4
  `d38` mapping (`t8` = current hudless, `t10` = current backbuffer, `t9`
  placeholder) is confirmed by its 640×360 captures.

This mapping was confirmed in the upstream project with the repository's exact
`scene.cpp` input generator rather than a Python approximation or a
provider-side reimplementation. The validation artifacts remain upstream and
are not shipped here. If a later source investigation changes the mapping, this
document and the IO contract will be revised; the application-facing resource
roles stay the same.

## 9. Datasets are never a runtime prerequisite

Capture and validation datasets are used only to develop and check the model.
The SDK does not read them. It needs application frame inputs and a model
package, nothing else.

Historical source evidence, retained upstream and not shipped here:

- an installed wheel was run in a separate process with ordinary application
  BGRA/RG16F/R32F files; a file audit hook was used to reject capture, trace,
  shader-disassembly, and tensor-dump paths;
- mode 3 at 640x360 and mode 1 at 1920x1080 completed for two frames each;
- the SDK output matched the full research reconstruction with 0 differing
  bytes for those inspected frames;
- provider-capture comparisons remained nonzero, and no provider equality is
  claimed.

Those runs used the installed SDK, an exported package, and the exact
`scene.cpp` application files. They are historical source evidence, not a
current validation result for this checkout. See [Validation](VALIDATION.md) for
the current status.

## 10. Non-guarantees

- No performance, frame-rate, latency, or memory guarantee.
- No NPU or vendor GPU guarantee or supported-NPU claim. The reference executor
  is ONNX Runtime on CPU; another `Executor` may be supplied. ONNX validity
  alone does not make a backend or NPU able to execute the mixed
  integer/float graph.
- No last-bit precision promise against Intel, OptiScaler, or any captured
  reference. The SDK is a semantic reference runtime, not bit-identical vendor
  emulation. Historical equality was against the project's own full research
  reconstruction, not against the provider. Mode 4 has capture/export/runtime
  plumbing and a checked 640×360 capture with a small remaining provider
  residual; no mode-4 provider equality is claimed.
- No claim that every public XeFG enum, format, or mode is implemented. The
  implemented model paths are modes 1, 3, and 4.
- No claim of provider equality for mode 4. SDK outputs match the full research
  reconstruction at all five sizes; provider comparisons remain nonzero.
- No claim that the model-bound depth contract covers the full device-depth
  range; the retained scene exercises only the values `1.0` and `100.0`.
- No game-specific unit guesses, provider hacks, or per-title quirks. The
  application or its adapter supplies already-normalized resources; the core
  does not reinterpret engine conventions.
