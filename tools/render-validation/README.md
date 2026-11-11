# Render validation

Independent validation tools, outside the Python SDK package: capture official
XeSS-FG output, establish calibration frame pairing, replay the same inputs
through the public SDK, and measure pixel differences. Two inputs are supported:
the original analytical probe scene and an independent DX12 raster-scene path.

## Build and capture

Build the current probe (Windows, Visual Studio 2022, XeSS SDK 3.0.2):

```powershell
cmake -S tools/capture -B D:/xefg-render-validation-build `
  -G "Visual Studio 17 2022" -A x64 -DXESS_SDK_ROOT=C:/path/to/XeSS_SDK_3.0.2
cmake --build D:/xefg-render-validation-build --config Release --parallel

python -I -B tools/render-validation/validate.py capture `
  --probe D:/xefg-render-validation-build/Release/XefgProbe.exe `
  --xess-bin C:/path/to/XeSS_SDK_3.0.2/bin `
  --out D:/xefg-validation/run-001 --frames 8
```

The inventory/capture commands use only the standard library. Pairing needs
NumPy. Neither requires an installed `xefg_sdk`. Output must be a new directory;
it never replaces an earlier run.
The initial case is fixed at mode 3, 640x360, 320x180 motion/depth, four warmup
frames, and the probe's deterministic moving scene. `--adapter` selects a GPU.
Tensor snapshots and model-catalog dumps are off. The probe still records its
existing dispatch/program trace; this is not a timing benchmark or an
uninstrumented official sample.

```text
run-001/
  invocation.json                  command and probe executable SHA-256
  capture.log                      native stdout/stderr
  capture/
    presents.json                  native presents and official API statuses
    captures/                      tightly packed output pixels
    application_inputs/            inputs, including warmup, and manifest
    trace/                         configuration, adapter, provider hash, trace
    ...                            existing probe program/parameter artifacts
  report/
    report.json                    inventory, hashes, statuses, observed labels
    report.md                      short summary
```

Inventory an existing run produced by the updated probe:

```powershell
python -I -B tools/render-validation/validate.py inspect `
  --run D:/xefg-validation/run-001/capture `
  --out D:/xefg-validation/report-001
```

Old captures without `presents.json` must be recaptured. The tool does not infer
missing metadata from filenames.

## What the report means

`inventory_complete` means the inputs/presents exist with the expected byte
sizes, frame records are consistent, interpolation-only debug mode was enabled,
and measured official API statuses report success. Exit code is zero.
`needs_attention` preserves the report with the status issues and exits one.
Missing/corrupt files fail the command rather than producing a successful report.

**Inventory success is not image-quality success.** Each present remains
`classification: unassigned`, and `pairing` is `unverified`. An observed
application-frame label is sampled by the native capture code; it is not a
provider-confirmed source frame ID. Likewise, ordinal zero is not proof that an
image is generated rather than a base/fallback frame. Warmup status is recorded
but excluded from measured-status checks.

### Native present manifest

Current schema: `xefg.present_capture/2`; inventory also accepts v1.

- `only_interpolation_requested`, `only_interpolation_result`: debug request and
  its API result. When not requested, the result is zero but does not indicate
  that the feature was enabled.
- `capture_error`: native readback/file failure, or an empty string.
- `frame_status`: application frame, HRESULT of proxy Present, status-query
  result, `frames_presented`, and `frame_generation_result`.
- `presents`: contiguous `sequence`, `observed_application_frame`, per-label
  `ordinal`, dimensions, integer DXGI `format`, and run-relative pixel `file`.

The manifest is written after XeFG context destruction. Capture remains enabled
through shutdown so late native presents can be included. The cross-thread
capture-enable/frame-label fields are atomic. This removes data races, not the
need to establish semantic frame pairing.

## Calibration and pairing

Use `capture --calibration` for an opaque UI frame-ID strip and analytical
midpoint references. The strip encodes a 16-bit application ID followed by its
16-bit complement. It is outside the motion comparison ROI. Default captures
remain unchanged. Add `--tag-interpolated --all-presents` for an official debug
identity experiment; add `--reset-frame 7` for a history-reset control, or
`--static` for a static control. `--pacing-ms` sets application pacing.

```powershell
python -I -B tools/render-validation/validate.py capture `
  --probe D:/xefg-render-validation-build/Release/XefgProbe.exe `
  --xess-bin C:/path/to/XeSS_SDK_3.0.2/bin `
  --out D:/xefg-validation/calibration-001 --calibration
python -I -B tools/render-validation/validate.py pair `
  --run D:/xefg-validation/calibration-001/capture `
  --out D:/xefg-validation/calibration-001/pairing
```

Calibration adds `calibration/manifest.json` and one raw analytical midpoint
per input frame, including warmup. Pairing requires native present schema v2;
inventory still accepts v1. V2 also records official debug flags/results,
frame-generation enabled state, application/native QPC intervals and native
DXGI present counts with their API results.

Pairing decodes the UI strip, then ranks analytical midpoint and base-frame
references in the motion ROI. A temporal candidate must agree with the decoded
ID, stay within 4.0 reference MAE, and beat the runner-up by at least 0.5 MAE
and 20%. The official purple tag detector checks the solid 48x48 corner patch
observed for this provider. These are calibration-specific pairing criteria,
not image-quality thresholds. Static scenes, tied candidates, duplicate generated
outputs, and conflicting official status stay ambiguous. Observed application
labels and timestamps do not determine pair identity. The JSON retains evidence
and hashes; `paired` requires all expected measured generated frames and no
ambiguous outputs. `incomplete` returns exit code one and retains the report.
All-presents and reset experiments may produce an inventory `needs_attention`;
this does not prevent running the separate pairing experiment.

## Real DX12 scene path

The independent native target lives under `native/`. Unlike the probe it uses
only public XeSS-FG APIs: no decoder relay, dispatch trace or model extraction.
It rasterizes indexed geometry, normal-Z depth and matrix-derived motion vectors.
The same GPU resources tagged to FG are read back as application inputs.

```powershell
cmake -S tools/render-validation/native -B D:/xefg-render-validation-raster-build `
  -G "Visual Studio 17 2022" -A x64 -DXESS_SDK_ROOT=C:/path/to/XeSS_SDK_3.0.2
cmake --build D:/xefg-render-validation-raster-build --config Release --parallel
python -I -B tools/render-validation/validate.py render `
  --renderer D:/xefg-render-validation-raster-build/Release/XefgRenderValidation.exe `
  --xess-bin C:/path/to/XeSS_SDK_3.0.2/bin --scene translation `
  --out D:/xefg-validation/raster-001
python -I -B tools/render-validation/validate.py pair `
  --run D:/xefg-validation/raster-001/capture `
  --out D:/xefg-validation/raster-001/pairing
```

Scenes: `translation`, `occlusion`, `camera`, `cut`. Default output is 640x360
with 320x180 geometry buffers. `render --size WIDTHxHEIGHT` selects one of the
five standard sizes listed below; geometry buffers remain exactly half-size.
The scene is rasterized at the selected resolution, not upscaled from 640.
Select `--mode 1`, `--mode 3` (default), or
`--mode 4`. Mode 1 FG capture additionally requires `--pair-frame N`: a fresh
context receives source scene frames N-1 and N as local submission IDs 0 and 1.
It renders no UI, tags only depth/motion, and uses the actual backbuffer as color.
Mode 3 tags hudless color and UI; mode 4 tags hudless color but not UI, consuming
the composed backbuffer instead. Both retain the animated crosshair for comparison.
All modes dump the five original resource roles, including unused UI for inspection. `render --render-only --warp` exercises rasterization
without loading the FG runtime; hardware FG cannot run on WARP. Use `--debug`
for D3D12 validation and `--no-tag` only for input/tag-layout control experiments.
Validate dumped geometry resources without loading a model:

```powershell
python -I -B tools/render-validation/check_render_inputs.py `
  --run D:/xefg-validation/raster-001/capture `
  --out D:/xefg-validation/raster-001/input-check.json
```

This checks finite values, resource sizes and normal device-depth range. Native
math/raster tests separately check projection and motion conventions.

Raster runs use root `meta.json` and `scene.json` rather than fake trace metadata.
`scene.json` records input submissions, transforms, resets, conventions and native
midpoint references. The `pair` command selects `rendered_scene` from this manifest.
Pairing v2 requires the official generated-frame tag and the decoded current UI
ID, checks all input-submission API results and rejects duplicate/failed history.
No analytical-image or SDK-output candidate ranking is used for raster scenes.

Mode 1 uses a separate isolated-pair identity method rather than decoding an
interpolated marker. Each capture must contain exactly two local submissions,
first reset and second continuous, and exactly one successfully tagged generated
output. Its source scene pair is recorded separately from local IDs 0/1. All
native outputs are retained, including reset/base outputs; ambiguous or duplicate
generated outputs fail pairing. Source identity is never inferred from observed
application labels, ordinal, or SDK image similarity.

The debug-layout profile is bound to the measured provider hash and output
resolution. Every recorded rectangle must match, not just a few purple pixels.
A resolution without an explicitly measured layout is rejected; the 640 layout
is never automatically stretched to fit another size. For calibration,
`inspect_tag_layout.py --run <capture> --out <new.json>` records exact purple
pixel rectangles and hashes. This observation tool does not register a mask or
certify generated identity; compare tagged/untagged captures and source content
before adding a measured layout. Quality
comparison excludes those regions plus the frame-ID strip; complete images are
still retained. Native midpoint images are quality references only and use current
UI composition, not a simulated half-frame UI. They do not establish frame identity.

Normal device depth and camera/object motion are new coverage beyond the old
1/100-depth analytical scene. Neither depth nor motion is tuned to reduce output
differences: actual GPU values go to both implementations.

## SDK replay and comparison

Install the checkout and replay dependencies into an isolated environment:

```powershell
python -m pip install -e ".[test]" pillow
python -I -B tools/render-validation/validate.py replay `
  --run D:/xefg-validation/calibration-001/capture `
  --pairing D:/xefg-validation/calibration-001/pairing/pairing.json `
  --package D:/path/to/model-package `
  --out D:/xefg-validation/calibration-001/replay --all
```

Without `--all`, only the first verified generated pair is replayed. The tool
requires a complete pairing report and rechecks its artifact hashes. Analytical
calibration comparisons require untagged runs; rendered-scene v2 (mode 3/4) and
isolated-pair v3 (mode 1) comparisons require official tags and exclude their
recorded regions. It calls public `Xefg2xRuntime.prepare/interpolate_pair` with the captured explicit
mode: previous/current backbuffer for mode 1; hudless plus current UI for mode 3;
hudless plus current backbuffer for mode 4. Current motion/depth are used in all
three modes. Reset frames are
not interpolated. Warmup inputs remain available, but the stateless pair API does
not require replaying every warmup frame.

Each frame produces a BGRA NPY, official and SDK RGB PNGs, and an absolute RGB
difference PNG at fixed 8x gain. Raster runs also save `_valid_diff_x8.png`
with excluded debug-overlay pixels blacked out. `report.json` and `report.md` contain RGB MAE,
maximum error, RMSE/PSNR, changed-pixel fractions and separate alpha statistics.
Full-frame and `valid_region` metrics are both reported; the Markdown summary
uses the valid region. Modes 3/4 additionally report `ui_region` over current UI
nonzero-alpha pixels intersected with the valid region. This isolates the dynamic
crosshair for diagnosis without removing it from the primary comparison. Raster reports additionally compare SDK and official
output separately against native midpoint references. Identical RGB
uses `rgb_identical: true` and `rgb_psnr_db: null` (not nonstandard JSON Infinity).
The report records model/profile hashes, input pairing hash, dependency versions,
Git state and output hashes. `measured` means the comparison ran, not that a
quality threshold passed. CPU wall times include first model loading and are not
a real-time benchmark.

Native capture copies the current backbuffer **before** native Present on the
actual queue passed to the native swapchain factory. Using the application queue
or copying after Present produced stale images in calibration; those failures
and the corrected experiments are documented in [VALIDATION.md](VALIDATION.md).

## d24 numerical fix and regression limits

The fractional-motion splat-coordinate fix, official tensor experiments and new
model package are documented in [SPLAT_FIX.md](SPLAT_FIX.md). Old model packages
must be re-exported; updating Python source alone does not patch their ONNX graphs.
Replay and diagnostic sampling tools pin ORT CPU threading to one thread for
repeatable baselines; the SDK executor default is not changed.

`check_quality.py --report <replay/report.json> --out <new-check.json>` applies
explicit valid-region regression limits (MAE 0.01, maximum byte difference 20,
over-two-byte pixel fraction 0.02%). Maximum channel differences above 8 but
within the configured maximum are recorded in `warnings`, not failed solely
for that maximum. Other limits still apply independently. `--maximum 8` restores
the earlier strict gate. Existing reports and per-case measured baselines remain
unchanged; a relaxed acceptance gate does not mean the measured residual disappeared. The limits are CLI-configurable and scoped
to the rendered four-scene regression suite, not a universal quality claim. The
same limits are applied across modes and resolutions without automatic scaling;
see the evidence report for which combinations actually pass.
Multiple `--report` arguments check the full suite together.

## Resolution/mode/scene matrix

Once the native target passes its render-only tests, run all three modes:

```powershell
python -I -B tools/render-validation/run_matrix.py `
  --renderer D:/xefg-render-validation-raster-build/Release/XefgRenderValidation.exe `
  --xess-bin C:/path/to/XeSS_SDK_3.0.2/bin `
  --package D:/path/to/model-package --modes 1 3 4 --out D:/xefg-validation/matrix-001
```

Defaults remain `--modes 3 --sizes 640x360`, retaining the original scene directory
layout. Multi-mode 640 output uses `modeN/scene/`. For other or multiple sizes,
output uses `WIDTHxHEIGHT/modeN/scene/`. Add e.g. `--sizes 640x360 1920x1080`.
The five choices are 640x360, 1280x720, 1920x1080, 2560x1440 and 3840x2160;
MV/depth is always half width and height. Each case creates fresh fixed-size
resources; this does not exercise runtime resolution changes.

Before capturing anything, the matrix validates model hashes and resolves all
requested mode/output/MV combinations through the SDK planner. Missing or
ambiguous profiles stop the run before GPU work. Model/profile identities are
retained in the total report. Re-export from compatible retained captures to
obtain missing profiles; do not edit a profile's dimensions by hand. All cases run serially, not concurrently
on the GPU. Each case goes through render, input checks, pairing, full replay,
and `check_quality.py` with the same explicit numeric limits. Mode 1 uses one
fresh native process/context per source pair, stored under `pair_NNNNN/`; modes
3/4 use continuous sequences. The measured source times are 4..11, excluding
frame 7 in the cut scene because it resets history.

A failed stage stops its case; other cases still run. `matrix.json` schema v3
records each stage, source pair, reset skip, model/profile identities and
resolution/mode/scene aggregate. Its status
is `running` until all requested cases finish, then `pass`, `quality_failed`,
or `run_failed`. Exit codes are respectively 0, 2, and 1 (run failure takes
precedence). `completed` means replay and numeric checking completed, **not**
that quality passed. Reset-related inventory warnings may proceed to pairing,
which decides whether generated outputs are accounted for. No raw artifacts
are automatically deleted.

## Artifact policy

Keep runs and build products outside the checkout. Source, test fixtures made
in temporary directories, and scenario definitions belong in Git; raw pixels,
SDK binaries, models and run reports do not. The first stage keeps all public
inputs and presents so pairing can be investigated; tensor dumps stay off.
Hashes cover the report's input/output files and source manifests, not the
entire dispatch/program trace. Later regression suites can retain failure windows
instead of every raw frame.

## Tests

```powershell
python -I -B -m unittest discover -s tools/render-validation/tests -v
ctest --test-dir D:/xefg-render-validation-build -C Release --output-on-failure
```

## Next increments

1. Expand scene coverage and add versioned mesh/material assets when procedural
   geometry no longer exercises the behavior being investigated.
2. Extend mode/resolution coverage and depth conventions one dimension at a time.
3. Add performance measurements separately from synchronous validation readback.

No third-party engine, model assets or SDK runtime binaries are vendored.
