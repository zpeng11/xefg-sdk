# d24 splat rounding fix — 2026-11-11

## Root cause and correction

The old same-grid implementation used `ceil(displacement)` based on an incorrect
interpretation of DXIL `Round_ni(displacement + 0.5)`. The retained shader actually
computes **floor(float32(displacement + 0.5))**. Integer displacement captures
hid the mistake; fractional motion exposed wrong target-pixel selection at edges.

Evidence: retained shader
`D:/xefg-capture-standalone-6bngbttd/mode3/trace/disasm/3548711932777909681613-c0481f4f9956f357.ll`,
lines 146–154 (same grid) and 159–174 (rescaled grid).

Implemented in all three places:
- `src/xefg_sdk/export/recon.py`: NumPy reference.
- `src/xefg_sdk/export/export_onnx.py`: static ONNX builder.
- `src/xefg_sdk/export/dynamic_index.py`: dynamic ONNX builder.

Rules:
- Same grid: `tid + floor(float32(disp + 0.5))`; ties toward positive infinity.
- Different grids: `Round_ne(float32(float32((tid+0.5+disp)*dst)/src))`, ties to
  even, no subtraction of 0.5. Preserve multiply-then-divide order.
- Static rescale output is already an absolute coordinate: do not add `tid` again.

No production motion/depth clamping or integer-MV workaround was added. Weights,
runtime public input API and renderer shader are unchanged.

## Official GPU confirmation

`probe_splat_rounding.py` captured the actual d24 inputs and output for ten
constant displacements, each across two retained dispatches. At 640x360 output,
320x180 MV, velocity V produces displacement -V/4. RX 6800 XT, tested fg78 DLL.

| Displacement | Old Ceil mismatched words | floor(d+0.5) | ties-even rint(d) |
|---|---:|---:|---:|
| +0.25 | 612 | 0 | 0 |
| -0.25 | 0 | 0 | 0 |
| +0.50 | 0 | 0 | 612 |
| -0.50 | 0 | 0 | 0 |
| +0.75 | 0 | 0 | 0 |
| -0.75 | 612 | 0 | 0 |
| +1.50 | 0 | 0 | 0 |
| -1.50 | 0 | 0 | 916 |
| +1.00 | 0 | 0 | 0 |
| -1.00 | 0 | 0 | 0 |

The corrected formula is **byte-exact in 20/20 official d24 allocations**.
These are horizontal same-grid cases from retained warmup frames 1/2; they do
not establish native rescaled-grid coverage. Rescaled rules are supported by
shader instructions plus synthetic static/dynamic graph tests.

Artifacts: `D:/xefg-splat-rounding-20261111-01/`, `-02/`, `-03/` and
`D:/xefg-splat-rounding-recheck-20261111/`. The old `.ll` shader identity was
matched when reusing disassembly; only data tensors vary across cases.

## Regenerated package

**`D:/xefg-splat-fixed-package-20261111-final/`** contains mode 1/3/4 models,
640x360 profiles and debug models. Generated with the normal exporter, not
ONNX graph surgery. Original packages were not overwritten.

| Variant | SHA-256 |
|---|---|
| mode1 | `0cb89eb16b0aac2c87dd9c97a2ebb7818c0b48a05b44b0727b8df4342589bd8e` |
| mode3 | `f675738a95be84165c8b09d8cdc597df0c529b73cba5793bff4ed0e077c702bc` |
| mode4 | `08f77f3ad7fecfc6d422b88ba6c8566605bda1582808afd2c9d0afb0c91bb83e` |

The earlier mode3-only package `D:/xefg-splat-fixed-package-20261111-01` has the
same mode3 model hash and produced the replay evidence below. Existing packages
do not change when Python source is updated: re-export them to obtain this fix.
The full 15-profile distribution was not regenerated in this increment.

## Four-scene result

Stable replay now explicitly uses CPU ORT with one intra-op and one inter-op
thread. The SDK's default executor behavior is unchanged. All values below use
valid RGB pixels, excluding official debug marks and the frame-ID strip.

| Scene | Frames | Mean frame MAE | Worst frame MAE | Max channel | Worst fraction >2 |
|---|---:|---:|---:|---:|---:|
| translation | 8 | 0.000774080 | 0.00328699 | 8 | 0.005867% |
| occlusion | 8 | 0.000352003 | 0.00211364 | 3 | 0.000978% |
| camera | 8 | 0.00172375 | 0.00904940 | 6 | 0.012711% |
| cut | 7 | 0 | 0 | 0 | 0% |

The camera-cut reset frame emits no generated output and is excluded by verified
history, not silently dropped. Compared with the original model's sequence mean
MAE, reductions are about 98.9%, 99.4%, 98.6%, and 100%, respectively.

Provisional engineering gate for these four 640x360 mode3 cases:
- each frame valid RGB MAE <= 0.01 (byte units);
- maximum channel difference <= 8;
- fraction of pixels with any RGB channel difference >2 <= 0.0002 (0.02%).

**31/31 frames pass.** Gate report: `D:/xefg-splat-fixed-quality-20261111.json`.
These limits are explicit and adjustable via `check_quality.py`, not a claim of
perceptual equivalence or coverage of other resolutions/HDR/content.

Replay folders: `D:/xefg-splat-fixed-{translation,occlusion,camera,cut}-20261111-single/`.
Fresh four-scene capture/replay: `D:/xefg-splat-fixed-fresh-matrix-20261111-01/`.
All 31 official outputs in that fresh capture matched the original captures
byte-for-byte. No camera/UI/depth inputs were tuned to improve the result.

## Remaining residual and repeatability

A few frames retain small non-UI differences: worst camera frame has 26 valid
pixels differing by >2, worst translation frame 12. The exact downstream cause
has not been isolated. Changing d25 sampling-weight rounding to Floor/Ceil makes
that camera residual worse, so those speculative modifications were not applied.

A separate ORT repeatability check found default multithreaded CPU execution
could produce differing outputs across repeated identical inputs (up to 3482
bytes in one four-run sample). Single-thread repeats were byte-identical. This
is why the validation tools pin threading; it does not prove the underlying
parallel-operation source of nondeterminism or fix it in the public executor.

## Regression verification

- `tests/test_splat_rounding.py`: four tests with seven parameterized coordinate
  cases, positive/negative fractions and ties, float32 near-ties, integer/noninteger
  resize ratios, independent packed keys and UMax scatter. Static ONNX, dynamic
  ONNX and reconstruction all checked against independent expectations.
- Full SDK suite with all capture environment variables: **87/87, zero skips**.
  Includes modes 1/3/4, animated UI and cross-size graph equality (640/1080).
- Standalone render-validation suite: **61/61 passed**.
- Logs contain pre-existing DXIL parser ResourceWarnings (unclosed text readers);
  those warnings are not test failures and were not addressed in this numerical fix.

## Usage

```powershell
python -I -B tools/render-validation/validate.py replay `
  --run D:/xefg-raster-matrix-20261111-final/camera/capture `
  --pairing D:/xefg-raster-matrix-20261111-final/camera/pairing/pairing.json `
  --package D:/xefg-splat-fixed-package-20261111-final `
  --out D:/new-fixed-replay --all
python -I -B tools/render-validation/check_quality.py `
  --report D:/new-fixed-replay/report.json --out D:/new-fixed-quality.json
```
