# Render validation evidence — 2026-11-11

## Current acceptance gate: maximum channel 20

The accepted default gate is now MAE <=0.01, maximum channel difference <=20,
and fraction of pixels with any RGB difference >2 <=0.0002 (0.02%). Only the
maximum-channel limit changed. Values >8 up to the configured maximum produce
non-failing channel warnings; MAE/fraction failures remain failures independently.
The previous strict measurements below remain historical evidence, not current
acceptance counts. Existing captures, replay reports and matrix reports were not
modified, and per-case measured baselines remain available for regression comparison.

All 195 replay reports / 465 frames were re-evaluated without rerendering:
`D:/xefg-multires-quality-max20-20261111-01.json`. This report includes source
report SHA-256 hashes, individual checks, warnings and per-resolution counts.

| Output | Passing frames | Failing frames | Channel-warning frames |
|---|---:|---:|---:|
| 640x360 | 93 | 0 | 0 |
| 1280x720 | 93 | 0 | 3 |
| 1920x1080 | 81 | 12 | 15 |
| 2560x1440 | 90 | 3 | 18 |
| 3840x2160 | 93 | 0 | 21 |

**450/465 pass; 15 fail the unchanged >2-pixel-fraction limit.** The 57 channel
warnings can coexist with fraction failures and are not an additional failure
count. `--maximum 8` remains available for the old gate. New matrix runs inherit
the updated defaults automatically. Tool regression: **90/90 passed**, including
20/21 boundaries, warning-only success, fraction failure despite a channel
warning, and explicit strict-limit override. Log:
`D:/xefg-quality-max20-tests-20261111.log`.

## Multi-resolution validation

The renderer now accepts five fixed output sizes, retaining half-resolution
motion/depth and the same scene time/camera/object transforms. Native resources
are rasterized at their actual sizes. The centered crosshair retains fixed pixel
extent; its region remains in the comparison.

Re-exported package: `D:/xefg-multires-fixed-package-20261111-01/`.
All 15 retained source captures were available, so no tensor recapture was
needed. The normal exporter produced five profiles per mode; all three ONNX
hashes are identical to the corrected package recorded in SPLAT_FIX.md. All 15
mode/size combinations resolve through the SDK planner.

### 1080p tag-layout evidence

Captures: `D:/xefg-multires-layout-1080-mode{1,3,4}-20261111-01/` and
`D:/xefg-multires-layout-1080-mode4-notag-20261111-01/`.
All three modes show the same six-region union: 64x64 corner blocks, a 16-pixel
left bar at x=48..64, and a right bar at x=1856..1872. The full purple union is
46,848 pixels. Explicit coordinates are registered for this provider/1920x1080;
they were measured, not extrapolated from 640.

The tagged/no-tag mode4 control had **25/25 input files byte-identical**, zero
purple pixels in the untagged output, and **zero differing pixels outside the
measured tag union**. The frame marker stays at x96/y32, outside these tags.
Mode4 debug smoke: 0 errors, 0 corruption, 70 existing clear-value warnings.
First mode4 replay: valid RGB MAE 0.001173, numeric gate passed unchanged.

The parameterized renderer's 640 regression completed: **93/93 frames pass**,
and **93/93 SDK NPY outputs are byte-identical** to the prior three-mode matrix.
Evidence: `D:/xefg-multires-matrix-640-1080-20261111-01/640x360/`.
1080p has real numeric regressions under the unchanged 640 gates. Mode1/3
translation both exceed at source frames 7/9/11, occlusion at 4/9; this is not
specific to isolated-pair identity. The mode1 spatial diagnostic is retained at
`D:/xefg-multires-1080-translation-edges-20261111.json`; many >2 pixels lie away
from the eight-pixel motion/depth boundary band, so no model stage is assigned
from that correlation. Full new-size matrices and remaining-size layouts are
recorded after completion.

### Completed 1080p matrix

`D:/xefg-multires-matrix-640-1080-20261111-01/1920x1080/`: all 93 frames
were paired/replayed; **72 pass, 21 fail numeric limits**, with no run failures.
The two-size matrix exits 2 (`quality_failed`), not a capture/inference error.
Each mode has the same seven failing source frames: translation 7/9/11,
occlusion 4/9, camera 4/8; cut passes all seven. Worst metrics across modes:
MAE 0.004583880, maximum channel 18, fraction >2 0.000385735 (0.038574%).
No limits were changed. Mode3/4 input files are **240/240 byte-identical** and
all 31 mode4 dynamic UI regions have RGB maximum error 0.

### Other-size layout controls

`D:/xefg-multires-layout-{1280x720,2560x1440,3840x2160}-mode{1,3,4}-20261111-01/`,
plus matching `mode4-notag` controls. All three modes agree at each size.
Each tagged/no-tag control has 25/25 identical inputs, zero untagged purple
pixels and zero differing output pixels outside the measured tag union.
Summary: `D:/xefg-multires-layouts-confirmed-20261111.json`.

First 4K mode4 replay completed in 23.70 CPU seconds (including first load),
MAE 0.000419846, max channel 10; it fails only the unchanged max-channel gate.
The remaining-size matrix is `D:/xefg-multires-matrix-rest-20261111-01/`.
720p completed all 93 frames: **90 pass, 3 fail**, solely camera source frame 8
in each mode (max channel 10). All other 720p scenes pass.
1440p also completed all 93: **75 pass, 18 fail**. Worst MAE 0.003527870,
maximum channel 16, fraction >2 0.000297299 (0.029730%); all cut frames are exact.
4K completed all 93: **72 pass, 21 fail**. Worst MAE 0.002488224,
maximum channel 17, fraction >2 0.000151627 (0.015163%).

### Final five-resolution summary

All **60 mode/scene combinations / 465 generated frames** completed pairing and
replay, with **zero run failures**. Under unchanged gates, **402 frames pass and
63 fail numeric limits**. Both matrix processes exit 2 because their requested
sets include numerical failures. Full summary: `D:/xefg-multires-summary-20261111.json`.

| Output | Frames passing | Worst frame MAE | Max channel | Worst >2 fraction |
|---|---:|---:|---:|---:|
| 640x360 | 93/93 | 0.009049398 | 8 | 0.012711% |
| 1280x720 | 90/93 | 0.001346190 | 10 | 0.001696% |
| 1920x1080 | 72/93 | 0.004583880 | 18 | 0.038574% |
| 2560x1440 | 75/93 | 0.003527870 | 16 | 0.029730% |
| 3840x2160 | 72/93 | 0.002488224 | 17 | 0.015163% |

Every measured frame meets the MAE limit. Remaining failures are maximum-channel
and/or >2-pixel-fraction limits. No production numerical changes or threshold
relaxation were made to make new sizes pass; the residual cause remains open.

For each of the five sizes, Mode3/4 have 240/240 identical input files and all
31 Mode4 dynamic UI regions are RGB-exact. Additional-size cross-check:
`D:/xefg-multires-input-ui-check-20261111.json`. All camera-cut valid frames are
RGB-exact in all modes and sizes.

Verification: SDK **87/87** (all external captures enabled), validation tools
**87/87**, native WARP CTest **1/1** including 640/1080 raster sizes and all-five
size derivation tests. `git diff --check` passes. Logs:
`D:/xefg-multires-sdk-tests-20261111.log` and
`D:/xefg-multires-tools-tests-20261111-final.log`.
The frozen native executable/shaders used for all captures are under
`D:/xefg-multires-runner-20261111-01/`. A sampled 4K replay process used about
3.85 GB working set; this is one observation, not a measured suite-wide peak.

## Three-mode real-scene matrix

Final run: `D:/xefg-three-mode-matrix-20261111-02/` (`matrix.json` / `matrix.md`).
RX 6800 XT, the same fg78 provider hash recorded below, 640x360 output and
320x180 depth/motion. Package: `D:/xefg-splat-fixed-package-20261111-final/`.
The numerical fix was committed as `c9269e5` before this mode expansion.

**12/12 mode/scene cases and 93/93 generated frames pass** the unchanged limits:
per-frame valid RGB MAE <=0.01, maximum channel difference <=8, and fraction of
pixels with a channel difference >2 <=0.0002. Matrix process exit code: 0.

| Mode | Scene | Frames | Worst frame MAE | Max channel | Worst >2 fraction |
|---|---|---:|---:|---:|---:|
| 1 | translation | 8 | 0.003254402 | 8 | 0.005809% |
| 1 | occlusion | 8 | 0.002174979 | 3 | 0.000968% |
| 1 | camera | 8 | 0.008959689 | 6 | 0.012585% |
| 1 | cut | 7 | 0 | 0 | 0% |
| 3 | translation | 8 | 0.003286986 | 8 | 0.005867% |
| 3 | occlusion | 8 | 0.002113645 | 3 | 0.000978% |
| 3 | camera | 8 | 0.009049398 | 6 | 0.012711% |
| 3 | cut | 7 | 0 | 0 | 0% |
| 4 | translation | 8 | 0.003286986 | 8 | 0.005867% |
| 4 | occlusion | 8 | 0.002098978 | 3 | 0.000978% |
| 4 | camera | 8 | 0.009049398 | 6 | 0.012711% |
| 4 | cut | 7 | 0 | 0 | 0% |

Mode 1 uses 31 fresh two-submission contexts, with no UI. Its source frames
remain 4..11, with source frame 7 skipped for cut; local indices are 0/1. The
first submission resets history and the second generates the unique eligible
pair. Both submissions are capture-enabled. Mode 3/4 use four warmup plus eight
measured submissions, and the existing current-frame marker. All three modes
were checked against the complete provider-specific generated tag layout.
Mode 1 comparisons exclude the official tags only, not a nonexistent marker.
Thus valid-region pixel counts differ slightly across modes.

Additional checks:
- Mode 1: **62/62** input frames have zero UI and backbuffer byte-equal to color.
- Mode 3 vs mode 4: **240/240** input files byte-identical across the four scenes.
- Mode 3: **31/31** SDK NPY outputs byte-identical to the single-thread numerical-fix baseline.
- Mode 4: **31/31** current dynamic UI regions RGB-exact against official output.
  This region uses current UI alpha !=0 intersected with the normal valid mask;
  the moving crosshair was never excluded from the primary comparison.
- Mode 1 debug smoke: `D:/xefg-modes-mode1-smoke-20261111-02/`, zero errors/corruption.
- Mode 4 debug smoke: `D:/xefg-modes-mode4-smoke-20261111-01/`, zero errors/corruption.
  Existing clear-value performance warnings remain (28 and 70 respectively).
- Full SDK regression: **87/87**, zero skips; native WARP CTest: **1/1**.
- Standalone validation tools: **77/77**; log
  `D:/xefg-three-mode-tools-tests-20261111-final.log`.
- Final pairing checks revalidated all **39 captures / 93 generated frames** at
  `D:/xefg-three-mode-pairing-recheck-20261111-01/`. Tests cover skipped tags,
  duplicate/unknown outputs, failed reset/native statuses, local/source IDs,
  mode-aware replay inputs and matrix quality/run-failure separation.

The final matrix used an immutable executable/shader copy at
`D:/xefg-three-mode-runner-20261111-01/`. Executable SHA-256:
`05268fdffed64a6ab37e1bfc82a01e2847236c110613d21e09d663ed42c43538`.
The first matrix (`...-01/`) is retained as an interrupted run: after 18 passing
Mode 1 pairs, concurrent relinking temporarily removed the executable. That
run is not the acceptance result. The complete `...-02/` run uses the fixed copy.

This covers Mode 1 isolated pairs and Mode 3/4 continuous sequences, not Mode 1
long-running history, other resolutions, or performance.

## Latest numerical fix

The original-model measurements below are retained as the baseline. The d24
coordinate-rounding fix and regenerated mode1/3/4 package are recorded in
[SPLAT_FIX.md](SPLAT_FIX.md): official same-grid tensors match in 20/20 probes,
all 31 rendered mode3 frames meet the documented provisional numeric gate,
and the full SDK suite passes 87/87 with no skips.

## Real DX12 raster scenes

Final evidence: `D:/xefg-raster-matrix-20261111-final/`. The native renderer uses
indexed geometry, normal device depth, matrix-derived camera/object motion,
GPU-rendered UI and composition. Official integration uses public APIs only.
No decoder relay or intermediate tensor capture runs in this executable.

GPU: AMD Radeon RX 6800 XT. Provider hash is the same tested hash below. Model:
`D:/xefg-capture-standalone-6bngbttd/package-f1`. Each case has four warmup and
eight measured application frames, at 640x360 mode 3 with 320x180 depth/motion.

| Scene | Verified generated frames | Largest per-frame valid RGB MAE | Largest channel error |
|---|---:|---:|---:|
| translation | 8 | 0.107449 | 127 |
| occlusion | 8 | 0.130396 | 117 |
| camera | 8 | 0.155334 | 108 |
| cut | 7 | 0.000050519 | 1 |

MAE/channel values use byte units (0..255), excluding official purple debug
marks and the UI frame-ID strip. Full images and full-frame metrics are retained
but are not the primary comparison. The renderer also saves a native midpoint
reference; replay reports compare each implementation with it separately.

The cut scene resets at application frame 7. Official status is success for the
API calls, `frameGenResult=3`, zero frames presented; no native output exists for
that frame. Inventory returns `needs_attention` and exit one. Pairing accounts
for the reset, pairs the other seven frames, and the overall matrix succeeds.
All four cases passed geometry-input finite/range checks and complete replay.
Observed depth minima across the sequences span approximately 0.9698..0.9762,
with background depth 1.0. This expands coverage beyond the old 1/100 fixture,
not to the entire device-depth range.

Visual inspection of the occlusion outputs and masked difference image confirms
localized differences around the moving cube silhouette/occlusion edges. The
cause has not been assigned to a particular model stage. No input values were
adjusted to minimize these differences. A prior complete matrix at
`D:/xefg-raster-matrix-20261111-02/` measured similar but not identical numbers;
this is not a claim of cross-run byte repeatability.

### Native correctness gates

- Full Release build: passed.
- WARP renderer CTest: **1/1 passed**, including four scenes, matrix conventions,
  finite/ranged depth, motion bounds/sign, repeated renders, midpoint independence,
  current UI markers, cut reset and debug ERROR/CORRUPTION assertions.
- Hardware render-only debug: `D:/xefg-raster-hardware-debug-20261111-01/`.
- Hardware official-FG debug: `D:/xefg-raster-fg-debug-20261111-01/`.
- Both hardware debug runs: **0 errors, 0 corruption**, 84 clear-value performance
  warnings (IDs 820/821, omitted optimized clear value). Warnings are retained.
- WARP debug run after queue-state fix:
  `D:/xefg-raster-warp-gate-20261111-03/`, also zero errors/corruption.
- Standalone Python tests: **54/54 passed**.
- SDK default tests: unittest **66 tests, 6 skips, no failures**.
- Existing capture PowerShell checks: **12/12 passed**.
- Legacy analytical replay smoke: `D:/xefg-raster-20261111-legacy-smoke/`.

Issues caught during implementation and fixed before the final matrix:
1. Oversized root signature: frame data moved to a root CBV.
2. Row-vector translation/projection placement and RG16F readback byte size.
3. DIRECT-to-COPY queue handoff without COMMON: synchronous readback now uses a
   DIRECT queue. The original debug run had 36 errors and is retained at
   `D:/xefg-raster-warp-gate-20261111-02/`.
4. Missing XeLL context in the new official entry point: first matrix returned
   initialization error -15; context creation/binding is now explicit.

Implementation scope adjustment: the new executable has its own small manifest
writer rather than moving the old probe's serializers during this increment.
Their existing schemas remain compatible, while the old capture/export path
stays untouched by that refactor. Sources/dependencies: `native/SOURCES.md`.

## Calibration investigation

The first moving calibration run exposed a real capture bug that inventory alone
could not detect. In `D:/xefg-calibration-20261111-clean16`, observed application
labels were 4..11, but decoded pixel IDs were **4, 2, 6, 4, 8, 9, 7, 11**. Motion
reference matching independently agreed with those decoded IDs. The run contains
stale/duplicate images and is not a valid complete comparison sequence.

The tagged all-presents control in `D:/xefg-calibration-20261111-tag16` also
showed that Intel's purple debug boxes cover the original frame-ID marker at
x=32. The marker was moved to x=96, outside the measured debug overlay. The
motion ROI [80,80,560,300] excludes the observed debug boxes and vertical lines.

The old-queue zero-pacing control is retained at
`D:/xefg-calibration-20261111-oldqueue0`; pairing is incomplete there too.
These failed calibration runs are evidence, not baseline-quality results.

The SDK default suite was subsequently run from an isolated environment with
this checkout installed: unittest reported **66 tests, 6 skips, no failures**.
The skips are opt-in external capture tests (including class-level skips).

## Corrected calibration and SDK replay

The capture fix uses the native swapchain's actual present queue and copies its
backbuffer before native Present. Each completed native call updates its capture
record using a unique token, not the last record in a shared list.

Corrected RX 6800 XT runs (all mode 3, 640x360, four warmup frames):

| Local run directory under D:/ | Experiment | Pairing result |
|---|---|---|
| xefg-calibration-20261111-fixed16 | 8 measured frames, 16 ms pacing | 8 generated, complete |
| xefg-calibration-20261111-fixed0 | 8 measured frames, 0 ms pacing | 8 generated, complete |
| xefg-calibration-20261111-fixedtag | Official purple tag, all presents | 8 generated + 8 base, complete |
| xefg-calibration-20261111-fixedreset | Reset at application frame 7 | 7 generated, complete; reset frame emitted no native present |
| xefg-calibration-20261111-fixedstatic | 4 static frames | 4 ambiguous, intentionally no temporal pairing |

The all-presents and reset controls return inventory `needs_attention`; their
separate pairing reports explain the expected behavior. Static pairing returns
`incomplete` because static content cannot identify a source time.

`fixed16/replay-first`, `fixed16/replay-all`, and the final-code rerun
`fixed16/replay-final` (using `fixed16/pairing-final`) used the validated local package
`D:/xefg-capture-standalone-6bngbttd/package-f1` through the public SDK API.
Eight-pair results (RGB byte units, 0..255):

| Current frame | RGB MAE | Max channel error | Changed pixel fraction |
|---|---:|---:|---:|
| 4 | 0.000030382 | 1 | 0.006944% |
| 5 | 0.000137442 | 1 | 0.033854% |
| 6 | 0.000018808 | 1 | 0.004774% |
| 7 | 0.000160590 | 1 | 0.041667% |
| 8 | 0.000044850 | 1 | 0.010851% |
| 9 | 0.000040509 | 1 | 0.010417% |
| 10 | 0.000069444 | 1 | 0.016927% |
| 11 | 0.000089699 | 1 | 0.022135% |

Alpha matched exactly in all eight frames. These are measured differences, not
a configured quality gate. Raw/PNG outputs, difference images at fixed 8x gain,
model/profile hashes and runtime metadata are in the replay reports.

Final standalone Python tests: **38/38 passed**. Updated native scene CTest:
**1/1 passed**. Existing PowerShell checks: **12/12 passed**. The SDK default
suite has no failures, with external-capture skips as described above.

## Stage-1 historical result

## Automated checks

- Standard-library inventory tests: **14/14 passed**.
- Native `xefg_scene` CTest: **1/1 passed**.
- Existing capture PowerShell script tests: **12/12 passed**.
- MSVC Release build: passed. The initial full build reported the existing
  wchar-to-char conversion warning in `xefg_runtime.cpp:45`.
- `git diff --check`: passed.

The SDK Python suite was not rerun: the selected Miniconda environment does not
have `xefg_sdk` installed. This increment changes native capture and standalone
tooling, not the SDK runtime/exporter.

## GPU smoke

Two eight-frame runs completed on AMD Radeon RX 6800 XT, with the final code
rebuilt before the second run:

- Mode 3, 640x360 output, 320x180 motion/depth.
- Four warmup frames; twelve application input sets retained.
- Eight native present files, observed under application labels 4 through 11.
- Interpolation-only debug API result: success.
- All eight measured present/status-query/frame-generation results: success.
- No capture error. Tensor snapshots disabled.
- Inventory result: `inventory_complete`; semantic pairing: `unverified`.
- Provider family: `fg78`.
- Provider SHA-256:
  `ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27`.

Local evidence (not distributed with the repository):

- `D:/xefg-render-validation-20261111-01/`
- `D:/xefg-render-validation-20261111-02/` (final build)

Each directory contains the executable hash/command in `invocation.json`, native
log, raw input/output files, native manifests, and JSON/Markdown inventory reports.
These runs establish extraction and inventory, not pixel repeatability between
runs, source-frame pairing, image quality, or real-time performance.
