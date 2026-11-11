# Edge difference investigation — 2026-11-11

## Status

The validated render pipeline was committed as **b58f2ca** before these experiments.
The initial investigation below used diagnostic copies only. The subsequent
fix now updates static/dynamic exporters and the NumPy reconstruction; see
[SPLAT_FIX.md](SPLAT_FIX.md) for official tensor evidence, exact tie rules,
regression tests and the regenerated package. The original model package and
default renderer shader remain unchanged. Work after b58f2ca is uncommitted.

**The dominant mismatch was d24 splat destination-coordinate rounding.** The
initial Round experiment below was a localization step, not the final fix.
Official same-size behavior is floor(float32(disp + 0.5)); ties-to-even Round
still leaves avoidable errors. The final implementation also corrects the
separate rescaled branch.

## 1. Spatial localization

`analyze_edges.py` compares the existing paired official/SDK outputs, excluding
debug overlays. Significant means any RGB channel differs by more than 2 byte
values. Motion/depth edges are detected on the geometry grid and dilated by eight
output pixels. This is a spatial correlation, not a dispatch correctness test.

| Scene | Significant pixels across sequence | Near motion/depth edge | Under UI |
|---|---:|---:|---:|
| translation | 12367 | 9783 (79.1%) | 1 |
| occlusion | 11592 | 9193 (79.3%) | 8 |
| camera | 18848 | 18269 (96.9%) | 0 |
| cut | 0 | 0 | 0 |

Evidence: `D:/xefg-edge-analysis-20261111.json`.

## 2. Matrix control

Only the matrices passed to official FG were replaced by identity; rendering and
input textures were unchanged. On the occlusion sequence:

- **60/60 input files byte-identical** (12 frames x five roles).
- **8/8 official output files byte-identical**, including the official tag.

Thus real-versus-identity FG matrices do not explain this sequence's error.
The native diagnostic option records `configuration.fg_matrix_control` in root
metadata; `scene.json` still describes the actual rendered transforms.

Evidence: `D:/xefg-edge-identity-20261111-01/`,
`D:/xefg-edge-identity-pair-20261111-01/`,
`D:/xefg-edge-identity-replay-20261111-01/`.

The suggested extra -0.5 motion correction was not applied: SV_Position and the
NDC-to-viewport mapping both use pixel-center positions here. Static cut frames
have maximum measured motion about 0.0037–0.0052 pixels, not a uniform half-pixel
bias. Cut reset remains required for the camera discontinuity even though its
reset frame's current/previous matrices yield near-zero motion.

## 3. Same-input ablations

Shader copies were created outside the checkout. Each variant was run through
both official FG and unchanged SDK, with complete pairing. Byte checks confirmed
that only the intended resource role changed: all 12 color/UI/backbuffer/depth
files stayed identical for motion variants; all motion files stayed identical
for the depth variant.

| Occlusion experiment | Highest frame valid RGB MAE | Max channel difference |
|---|---:|---:|
| Original fractional motion | 0.130396 | 117 in the original replay |
| Motion rounded to integer pixels | 0.001452 | 2 |
| Motion forced to zero | 0.001633 | 3 |
| Depth forced to 1 | 0.159239 | 117 |

These controls isolate sensitivity to fractional motion. They are deliberately
altered inputs, not a proposal to round real application motion or replace depth.
Evidence and shader copies: `D:/xefg-edge-controls-20261111/`, including
`control_summary.json` and each variant's capture/pairing/replay directories.

## 4. d25 sampling-weight rounding is not the dominant explanation

On occlusion frame 4, diagnostic ONNX copies changed only the 32 d25 texture-weight
Round nodes. Original model and feeds were untouched.

| d25 weight operation | Valid RGB MAE | Max channel |
|---|---:|---:|
| Round (original) | 0.130396 | 80 |
| Floor | 0.129333 | 80 |
| Ceil | 0.127691 | 78 |

This perturbation barely changes the dominant error. It does not rule out smaller
d25 texture-sampling residuals. Evidence: `D:/xefg-edge-sampling-20261111-01/`.

## 5. d24 target-coordinate rounding explains most of the error

Code location: `src/xefg_sdk/export/dynamic_index.py`, `build_dynamic_splat`,
currently uses Ceil for rescaled target coordinates and same-size displacement
coordinates (lines 179, 184, 192, 193 at the committed revision). Four matching
`n0024_*` nodes were changed in diagnostic model copies; no weights or other
stages were changed. Copies passed ONNX checker.

Average valid-region RGB MAE over each scene's paired frames:

| Scene | Original Ceil | Diagnostic Round | Diagnostic Floor | Round reduction |
|---|---:|---:|---:|---:|
| translation | 0.0721595 | 0.000774080 | 0.0837295 | 98.9% |
| occlusion | 0.0600195 | 0.00301524 | 0.0517122 | 95.0% |
| camera | 0.124319 | 0.0128964 | 0.142719 | 89.6% |
| cut | 0.0000288679 | 0 | 0.0000232806 | 100% |

Round residual maximum channel errors: translation 8, occlusion 26, camera 32,
cut 0. Some individual frames are already within one byte value; others retain
localized residuals. Original-model reruns can vary slightly in a few maxima
(e.g. occlusion sequence 116 vs 117); comparisons above use each diagnostic
experiment's own original Ceil control.

Evidence:
- `D:/xefg-edge-splat-summary-20261111.json`
- `D:/xefg-edge-splat-translation-20261111-all/`
- `D:/xefg-edge-splat-occlusion-20261111-all/`
- `D:/xefg-edge-splat-camera-20261111-all/`
- `D:/xefg-edge-splat-cut-20261111-all/`

## Next precise step

Do not globally replace Ceil yet. Determine the provider's target-coordinate rule
at positive/negative half-integers and under equal-size versus rescaled grids.
ONNX Round is ties-to-even, which is not interchangeable with floor(x+0.5).
Capture/compare the d24 output for the same problematic fractional-motion input,
then change static/dynamic/reconstruction implementations together and regenerate
packages with focused fractional-motion regression cases. Once d24 is exact,
re-evaluate the remaining d25/flow/composite residuals rather than attributing
all residuals to the previously known d25 tolerance.

## Reproduction helpers

```powershell
python -I -B tools/render-validation/analyze_edges.py `
  --matrix D:/xefg-raster-matrix-20261111-final --out D:/new-edge-report.json
python -I -B tools/render-validation/probe_sampling.py `
  --run D:/xefg-raster-matrix-20261111-final/occlusion/capture `
  --pairing D:/xefg-raster-matrix-20261111-final/occlusion/pairing/pairing.json `
  --package D:/xefg-capture-standalone-6bngbttd/package-f1 `
  --stage d24 --all --out D:/new-splat-probe
```

All output paths must be new. These commands analyze or create diagnostic copies;
they never rewrite the installed model package.
