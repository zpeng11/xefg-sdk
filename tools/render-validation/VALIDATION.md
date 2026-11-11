# Render validation evidence — 2026-11-11

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
