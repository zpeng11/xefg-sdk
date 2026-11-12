# Render validation results

## Measured result

The supported matrix covers nine scenes, five output sizes, and UI modes 1, 3,
and 4: **135 combinations and 1065 generated frames**. Every combination
completed input checks, pairing, and same-input SDK replay. There are no
execution or input-coverage failures.

**1020/1065 frames satisfy the numeric gates.** 111 combinations pass; 24
combinations contain at least one frame outside a gate.

Gates, applied independently to the valid comparison region:

- RGB MAE <= 0.01;
- maximum channel difference <= 32;
- pixels with any RGB difference > 2: <= 0.0002 (0.02%).

| Output | Completed combinations | Numeric passing frames | Worst frame MAE | Max channel | Worst >2 fraction |
|---|---:|---:|---:|---:|---:|
| 640x360 | 27/27 | **213/213** | 0.000197187 | 1 | 0% |
| 1280x720 | 27/27 | **213/213** | 0.006686843 | 7 | 0.008708% |
| 1920x1080 | 27/27 | **180/213** | 0.010826603 | 28 | 0.089297% |
| 2560x1440 | 27/27 | **207/213** | 0.008984963 | 32 | 0.029730% |
| 3840x2160 | 27/27 | **207/213** | 0.006790328 | 25 | 0.068932% |
| Total | **135/135** | **1020/1065** | 0.010826603 | 32 | 0.089297% |

All frames satisfy the maximum-channel gate. Three frames exceed MAE and 42
exceed the >2-pixel fraction; the groups do not overlap.

Each combination measures source times 4..11, except source7 in `cut`, where
history resets. Output files and representative images are under the external
validation root:

- `results/matrix.md` and `results/matrix.json` — complete numeric results;
- `images/` — 225 official/SDK/difference triplets, selected by worst metric and
  failure cause per combination;
- `package/` — the model package used for these measurements;
- `bin/` — the frozen renderer executable and shaders.

## Remaining differences

Every row applies to each of Modes 1, 3, and 4. Frames are identified by scene
source time.

| Output | Scene | Source frames | Failed metric |
|---|---|---|---|
| 1920x1080 | translation | 7, 11 | >2 fraction |
| 1920x1080 | occlusion | 4, 5, 9 | >2 fraction |
| 1920x1080 | screen_cross | 4, 5, 8 | >2 fraction |
| 1920x1080 | nonrigid | 5, 7 | >2 fraction |
| 1920x1080 | nonrigid | 6 | MAE |
| 2560x1440 | translation | 7 | >2 fraction |
| 2560x1440 | depth_layers | 8 | >2 fraction |
| 3840x2160 | depth_layers | 8 | >2 fraction |
| 3840x2160 | screen_cross | 8 | >2 fraction |

These are small-area differences: the worst failing MAE is 0.010827 and the
worst failing >2 fraction is 0.000893. Transparency passes all 120 frames;
nonrigid passes 111/120, with its failures only at 1080p. No threshold is scaled
by resolution or content.

## Inputs and consistency

- All 75 combinations with explicit scene-distribution criteria pass coverage.
  The 60 rigid-scene combinations have basic input validation and no additional
  distribution gate.
- Mode3/4 input comparison: **2700/2700 resource hashes identical** across
  scenes, sizes, and source frames, including warmup.
- Mode4 UI comparison regions: **355/355 generated frames RGB-exact**.
- All 675 retained representative PNGs are present.
- Pairing artifacts are hash-checked before replay. Compact reports retain
  numeric results and fingerprints rather than raw inputs or every output array.

## Environment and models

- Windows x64, AMD Radeon RX 6800 XT, official XeSS SDK 3.0.2 (fg78).
- Provider SHA-256:
  `ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27`.
- CPU ONNX Runtime, one intra/inter-op thread; synchronous native readback.
  These are image comparisons, not performance measurements.
- Package: three model variants and fifteen profiles covering every mode/size
  combination.

| Model | SHA-256 |
|---|---|
| mode1 | `25670199901ef8f754095ce752902340f6fdff7d95d8f08db1fbdeb1168d1989` |
| mode3 | `0bee8d52f0cc98b82f470149d060f8a04e43bb50a92ea14bcd8484b930e666ad` |
| mode4 | `221c87b33c47e506630a997131911d082528f0cfc6839ac00b6b81a75e262832` |

The d25 sampler applies `round_ties_even(T/8)/8192` in byte units after the
integer four-tap sum, consistently in the NumPy reference and both ONNX
lowerings. Existing ONNX files are not changed by source edits; re-export to
apply algorithm changes. Numeric contracts are in
[compatibility](../../docs/COMPATIBILITY.md).

## Scope

The matrix measures procedural DX12 scenes, not game engines. It exercises
rigid and nonrigid motion, occlusion, depth layering, camera movement, screen
edges, transparency, history resets, and the supported fixed profiles. It does
not establish arbitrary-resolution behavior, HDR, every material pipeline,
provider byte-equality, or performance. Results apply to the scenes, inputs,
provider, and GPU used.

## Test status

- Render-validation tools: **226 tests, OK**.
- SDK suite: **75 tests, OK**; six opt-in capture integration classes skip when
  no fixture paths are configured. Self-contained sampler tests remain runnable.
- Checkpoint-only replay resumes from compact results without reopening raw
  source captures. Its exit status still reports the recorded numeric failures.
- Logs are under the validation root's `logs/` directory.

## Retention

The external validation root holds only the active result set: models and
profiles, numeric reports, representative images, frozen renderer, logs, and the
explicit cleanup manifest/outcome. Raw captures, per-frame arrays, tensor dumps,
superseded models, and investigation archives are removed. Cleanup uses an
inspected path manifest; the source repository, Python environment, SDK, and
toolchains are protected. No backup archive is kept.

Inference uses only the retained package. A future raw replay or model re-export
requires new captures.
