# Render validation

Independent DX12 validation against official Intel XeSS-FG. The renderer saves
exactly the resources submitted to the provider; verified generated frames are
compared with public `Xefg2xRuntime` replay of those same inputs.

**Measured results:** [VALIDATION.md](VALIDATION.md). Local artifacts live in
`E:/xefg-validation-current/`: model package, reports, representative images and
frozen renderer. The workflow keeps the active result set rather than raw runs.

## Coverage

| Scenes | Purpose |
|---|---|
| `translation` | Rigid-object translation |
| `occlusion` | Occlusion and disocclusion |
| `camera` | Camera-driven motion |
| `cut` | Camera cut and history reset |
| `depth_layers` | Near/middle/far geometry and relative motion |
| `screen_cross` | Large bidirectional motion and screen-edge entry/exit |
| `camera_turn` | Camera rotation/translation and depth-dependent parallax |
| `nonrigid` | Subdivided surface deformation with current/previous vertex motion |
| `transparency` | Moving alpha-blended scene layers and opaque occlusion |

Modes: **1, 3, 4**. Output sizes: **640x360, 1280x720, 1920x1080, 2560x1440,
3840x2160**. Motion/depth width and height are always half the output dimensions.
The full matrix is **135 combinations / 1065 generated frames**: sources4..11,
except source7 in `cut`, where history resets. Each size is actually rasterized;
this is not upscaling a640 input or testing dynamic resolution changes.

Mode1 submits previous/current backbuffer and current motion/depth, without UI.
Mode3 uses hudless color plus current premultiplied UI. Mode4 uses hudless color
plus current composed backbuffer, without a separate UI input tensor. AUTO is
not part of this explicit-mode matrix.

The nonrigid case uses continuous analytic vertex deformation, including normals
and previous-time positions. The transparency case blends into scene color,
not UI; transparent objects are depth-tested without writing depth or motion.
Depth/MV therefore describe opaque surfaces. This does not cover refraction,
transparent-surface MV writes, cloth physics, HDR or every engine's material path.

## Build

Windows, VS2022 x64 and XeSS SDK3.0.2 are required for official capture. Build
products and the official SDK stay outside the checkout.

```powershell
cmake -S tools/render-validation/native -B C:/build/xefg-render `
  -G "Visual Studio 17 2022" -A x64 `
  -DXESS_SDK_ROOT=C:/SDK/XeSS_SDK_3.0.2
cmake --build C:/build/xefg-render --config Release
python -m pip install -e ".[test]" pillow
```

The normal renderer uses public FG APIs. Diagnostic tensor capture is separately
enabled by `-DXEFG_RENDER_VALIDATION_TENSOR_PROBE=ON`; the standalone hardware
sampler uses `-DXEFG_BUILD_SAMPLER_PROBE=ON`. Neither is enabled by default.

## Capture, pair, replay

```powershell
python -I -B tools/render-validation/validate.py render `
  --renderer E:/xefg-validation-current/bin/XefgRenderValidation.exe `
  --xess-bin C:/SDK/XeSS_SDK_3.0.2/bin `
  --scene nonrigid --mode 3 --size 1920x1080 --out E:/validation/new-case
python -I -B tools/render-validation/check_render_inputs.py `
  --run E:/validation/new-case/capture --out E:/validation/new-case/input-check.json
python -I -B tools/render-validation/validate.py pair `
  --run E:/validation/new-case/capture --out E:/validation/new-case/pairing
python -I -B tools/render-validation/validate.py replay `
  --run E:/validation/new-case/capture `
  --pairing E:/validation/new-case/pairing/pairing.json `
  --package E:/xefg-validation-current/package `
  --out E:/validation/new-case/replay --all
python -I -B tools/render-validation/check_quality.py `
  --report E:/validation/new-case/replay/report.json `
  --out E:/validation/new-case/quality.json
```

Use new output directories. Mode1 FG capture additionally requires
`--pair-frame N`: a fresh context submits sourceN-1 andN as local0/1, with only
the first submission resetting history. `--render-only --warp` tests rasterization
without FG; hardware FG does not run on WARP. `--debug` enables DX12 validation.
`--no-tag` is for dedicated layout controls, not ordinary generated-frame pairing.
Replay without `--all` selects the first verified pair only.

To render a complete new raw matrix, use `run_matrix.py` with explicit
`--modes 1 3 4`, all five `--sizes`, and all nine `--scenes`. Defaults deliberately
remain Mode3,640x360 and the original four scenes. This raw runner retains its
artifacts; use the compact replay workflow for a space-limited final deliverable.

## Identity and input checks

- Mode3/4 pairing uses the measured provider/resolution tag layout, current
  opaque UI marker, and successful submission/history statuses.
- Mode1 pairing uses the isolated two-submission context and a unique verified
  generated output. Observed labels, ordinals and SDK image similarity are not
  sufficient identity evidence.
- Official tag masks are explicitly registered for each supported size and
  provider hash; unknown layouts fail rather than scaling a mask heuristically.
- Native midpoint images are quality references, not pairing evidence.
- Input checks verify shape, byte counts, finite depth/MV and actual scene
  distributions. Nonrigid checks independently reconstruct interior MV/depth;
  transparency checks sampled blending and opaque-only depth/MV.
- Coverage aggregates measured source times across Mode1 isolated pairs; reset
  and warmup frames do not count as measured coverage.

## Quality policy

Metrics use byte-unit RGB differences. The primary valid region excludes only
recorded official tags and the frame marker; moving UI remains in the comparison.
UI and full-frame metrics are also reported separately.

All sizes and modes use the same inclusive limits:
- RGB MAE **<=0.011**;
- maximum channel difference **<=32**;
- pixels with any RGB difference >2: **<=0.001 (0.1%)**.

Maxima above8 but within32 are warnings. The three gates are independent; no
automatic resolution scaling or visual waiver is applied. A completed replay is
not necessarily a passing comparison. Coverage failures and execution failures
are separate from numeric failures. Identical-image PSNR is JSON null, not Infinity.

Validation uses CPU ORT with one intra/inter-op thread. Native readback is
synchronous; timings are not a real-time performance benchmark.

## Compact full replay and retention

`replay_matrix.py` maps compatible source-matrix captures when available and
renders any missing combinations. It verifies package/profile identities, source
pairing hashes and coverage, then checkpoints each combination. Its help
describes the source-matrix, output and resume arguments.

Final directory layout:

```text
xefg-validation-current/
  package/      corrected models, fifteen profiles, optional debug graphs
  bin/          renderer executable and shaders
  results/      matrix summary and complete per-frame numeric reports
  images/       representative official, SDK and difference images
  logs/         execution and test logs
  cleanup/      explicit deletion manifest and outcome
```

Temporary inputs and replay arrays are removed after the needed reports/images
are safely saved. The workflow retains the active result set: models, numeric
reports, representative images and logs. Raw captures, per-frame arrays, tensor
dumps and superseded model copies are not kept. `cleanup_runs.py` uses an
inspected, explicit path manifest and protects the result root, source
repository, Python environment, SDK and toolchains. Cleanup is a separate
explicit action, not an implicit side effect of ordinary capture commands.

**After raw/source-capture cleanup, another replay or model re-export requires
new captures.** The retained package remains usable for SDK inference. Source-code
changes do not update exported ONNX files: export a new package when algorithms
change. Model generation and numeric contracts are documented in
[../../docs/COMPATIBILITY.md](../../docs/COMPATIBILITY.md).

## Diagnostics and tests

- `compare_tensors.py`: compare selected official d24/d25 tensors with debug
  models; requires an explicit tensor capture and normal-replay control.
- `probe_hardware_sampler.py`: explicit captured texture/coordinate input for
  `XefgSamplerProbe`; it has no preset capture input.
- `analyze_edges.py`, `inspect_tag_layout.py`: diagnostic analysis of supplied runs.
- Analytical probe capture remains available via `validate.py capture`; its
  calibration identity rules differ from the real raster path.

```powershell
python -I -B -m unittest discover -s tools/render-validation/tests -v
python -I -B -m unittest discover -s tests -v
ctest --test-dir C:/build/xefg-render -C Release --output-on-failure
```

No third-party engine, model assets, or official SDK binaries are vendored.
