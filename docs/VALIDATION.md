# Validation

This document describes the testing layers, what each layer establishes, and how
to reproduce it. Measured render results are published in
[render-validation/VALIDATION.md](../tools/render-validation/VALIDATION.md); this
document describes how to produce them and does not duplicate result counts.

Validation artifacts are kept outside the source tree. No capture, report, model,
or run directory is a runtime dependency or part of the Python package.

## Testing layers

| Layer | Scope | Requires |
| --- | --- | --- |
| SDK unit/synthetic tests | Runtime API, profile planning, package loading, exporter graph construction, sampler parity, examples | Installed checkout; NumPy; ONNX/ORT for the export and smoke cases |
| SDK capture-backed integration tests | Exported package and runtime against supplied captures and the project's own reconstruction | Explicit capture directories, opt-in |
| Native capture producer | Probe build, scene generation, dispatch/tensor capture contracts | Windows, Visual Studio, user-supplied XeSS SDK; GPU only for real capture |
| Render-validation tools | Official-present inventory, frame pairing, same-input SDK replay, raster scenes, quality gates | Installed checkout, NumPy, ONNX Runtime, Windows/GPU for capture |
| Render matrix | Official vs SDK comparison across supported modes, sizes, and scenes | Frozen renderer, XeSS runtime, model package, external output directory |

Synthetic tests and the small real-ORT smoke test need no captures. Integration
and render-matrix results are evidence for the inputs actually exercised; they
are not a blanket accuracy, provider-equality, resolution, or backend guarantee.

## SDK tests

Install the checkout and run the default suite:

```sh
python -m pip install -e ".[test]"
python -I -B -m unittest discover -s tests -v
```

The default suite covers the public runtime, profile selection, package
validation, export logic, examples, and sampler precision. The sampler tests
compare the reference kernel with both ONNX lowerings at small and real
resolutions, including embedded hardware vectors; they do not need the external
capture directories.

Capture-backed integration tests are opt-in and use explicit fixture paths:

| Environment variable | Fixture |
| --- | --- |
| `XEFG_SDK_MODE1_RUN` | Mode 1, half-resolution MV/depth |
| `XEFG_SDK_MODE3_RUN` | Mode 3, 640x360 |
| `XEFG_SDK_MODE3_SIZE_RUN` | Mode 3 at another size, for cross-size comparison |
| `XEFG_SDK_MODE4_RUN` | Mode 4, 640x360 |
| `XEFG_SDK_MODE4_ANIMATED_RUN` | Mode 4, 640x360 with animated UI |
| `XEFG_MODE4_CAPTURE` | Mode 4 with `application_inputs/`, for provenance |

Unset variables skip the corresponding tests. Invalid configured paths fail
rather than silently skipping. Never commit capture directories to the
repository.

For exporter-side checks, install the export extra:

```sh
python -m pip install -e ".[export]"
```

Exports read a capture that satisfies [Capture format](CAPTURE_FORMAT.md). The
output directory must be absent or empty. `--debug` adds diagnostic model
outputs; `--frame` selects the retained frame.

## Native capture producer

The capture tool is optional and is not part of the Python build or wheel. With
a user-provided XeSS SDK (Windows, Visual Studio 2022):

```powershell
cmake -S tools/capture -B "$env:TEMP/xefg-sdk-capture-build" `
  -G "Visual Studio 17 2022" -A x64 "-DXESS_SDK_ROOT=$env:XESS_SDK_ROOT"
cmake --build "$env:TEMP/xefg-sdk-capture-build" --config Release --parallel
ctest --test-dir "$env:TEMP/xefg-sdk-capture-build" -C Release --output-on-failure
powershell -NoProfile -File tools/capture/script-tests/run-tests.ps1
```

The CTest scene/runtime tests check CPU data generation and capture contracts.
The PowerShell script tests use command stubs and do not build native code, run
XeFG, or require a GPU. Real capture requires the XeSS FG runtime and a supported
GPU. See [Capture guide](CAPTURE_GUIDE.md) for the capture/export sequence.

## Render-validation tools

The independent harness under `tools/render-validation/` captures official
generated frames, establishes frame pairing, replays the same application inputs
through the public SDK, and measures the difference. It also contains an
independent DX12 raster renderer for the analytical scene, depth/motion scenes,
and nonrigid/transparency content cases.

```sh
python -I -B -m unittest discover -s tools/render-validation/tests -v
ctest --test-dir <native-build-dir> -C Release --output-on-failure
```

Build and capture commands are documented in
[render-validation/README.md](../tools/render-validation/README.md). A matrix run
uses a frozen renderer, the provider runtime, a model package, and a new output
directory:

```powershell
python -I -B tools/render-validation/run_matrix.py `
  --renderer <renderer-executable> `
  --xess-bin <xess-runtime-bin> `
  --package <model-package> `
  --modes 1 3 4 --sizes 640x360 1280x720 1920x1080 2560x1440 3840x2160 `
  --out <new-output-directory>
```

Runs and build products stay outside the checkout. The tools do not fetch model
assets or silently repair a missing capture/profile.

Measured results, gates, coverage, per-mode/size status, and the scope of the
inputs actually exercised are in
[render-validation/VALIDATION.md](../tools/render-validation/VALIDATION.md).
The matrix is produced by `replay_matrix.py` or `run_matrix.py`. Report statuses
distinguish execution, input coverage, and numeric quality: `inventory_complete`,
`measured`, and `completed` mean the stage ran, not that the images passed a
quality gate or that another resolution, scene, GPU, or backend is covered.

## Numerical contracts and limitations

- The SDK reproduces the project's own research reconstruction for inspected
  frames; validation against official output is a separate comparison. Neither
  is a provider byte-equality claim.
- `d25` (warp + flow init) is the only stage that samples through the texture
  unit's bilinear filter. Its measured sampler rule is described in
  [Compatibility](COMPATIBILITY.md#54-d25-sampling-precision). A color-store
  residual of one LSB is allowed by `src/xefg_sdk/export/d25_bound.py`, which
  keeps calibrated per-size/content aggregates and a structural fraction
  fallback. `d25` is bounded, not byte-exact; matching downstream stages when fed
  captured `d25` outputs does not make the chained graph exact.
- Matching the project's reconstruction does not establish correctness against
  the provider. Mode 4 retains a small provider residual in the measured renders;
  no mode-4 provider equality is claimed.
- No performance, frame-rate, latency, memory, or NPU-support claim is made. ONNX
  validity and an ONNX Runtime CPU run do not prove that another backend can
  execute the mixed integer/float graph.
- Only the profiles shipped in a model package are supported. Validation does
  not imply arbitrary-resolution planning, automatic resampling, or a broader
  device-depth range than the inputs exercised. The analytical probe scene
  exercises only depth `1.0` and `100.0`; rendered scenes exercise additional
  normal-device-depth values but not the full device range.
- Capture-backed tests and render-matrix reports are not shipped. Source-level
  tests and capture-backed evidence are separate; external results depend on the
  inputs supplied for that run.
