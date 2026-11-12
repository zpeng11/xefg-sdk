# XeFG 2x SDK

A device-neutral ONNX exporter and Python reference runtime for the reconstructed
XeFG 2x pipeline. This is an independent research SDK, not Intel's official XeSS
SDK and not a swapchain integration.

- **Capture (optional, Windows):** user-provided XeFG SDK → retained trace and tensors.
- **Exporter:** compatible captures → ONNX models + manifest + resolution profiles.
- **Runtime:** normalized application images + model package → generated frame.
- **Executor:** a small injectable interface; ONNX Runtime is an optional reference
  implementation. No vendor-specific NPU/GPU backend is implemented here.

This repository contains source code only. It does **not** include Intel SDK
binaries, extracted weights, ONNX models, captures, or application frames.

## Install from this checkout

Python 3.10 or newer:

```sh
python -m pip install .                # runtime core (NumPy only)
python -m pip install ".[ort]"         # optional ONNX Runtime executor
python -m pip install ".[export]"      # exporter (NumPy + ONNX, no ORT needed)
python -m pip install ".[export,ort]"  # both
```

The distribution is named `xefg-sdk`; the Python package is `xefg_sdk`.
These are local installation instructions, not a claim of a published PyPI release.
API version is **0.3.0**; model-package schema is **1**. See
[compatibility](docs/COMPATIBILITY.md) for supported modes and older packages.

## Capture source data (optional, Windows)

If you already have a compatible capture or model package, skip this step.
Otherwise, use the included native tool with your own XeSS SDK:

```powershell
powershell -NoProfile -File tools/capture/run.ps1 `
  -XessSdkRoot "C:/SDKs/XeSS_SDK_3.0.2" `
  -OutDir "$env:TEMP/my-xefg-mode3" -Mode 3 -Width 640 -Height 360
```

Replace the SDK path with your own and choose a new/empty output directory.
The runner builds, captures, and disassembles; export is a separate step.
Use `-Mode 1`, `3`, or `4` to select the UI mode described below. See the
[capture guide](docs/CAPTURE_GUIDE.md) for prerequisites and capture options.
This tool is built separately from the Python package and is not in the wheel.

## Export a model package

The exporter is self-contained: it does not require the `MFG-Unlocker` repository.
Use existing compatible captures, or create them with this repository's optional
[Windows capture tool](tools/capture/README.md). Only that native tool loads the
user-provided `libxess_fg.dll` and runs D3D12; exporting existing captures and
running exported models do not need the DLL, capture executable, or Windows SDK.

```sh
xefg-export --runs /data/capture-mode1,/data/capture-mode3 --out /data/model_package
# Equivalent:
python -m xefg_sdk.export --runs /data/capture-mode1,/data/capture-mode3 --out /data/model_package
```

The output directory must be absent or empty. Use `--frame 2` to select a retained
frame or `--debug` to include diagnostic graphs. One variant model serves all its
exported profiles; supplying additional compatible captures adds profiles.
See [capture format](docs/CAPTURE_FORMAT.md) and [model package](docs/MODEL_PACKAGE.md).

## Generate a frame

With a model package already exported, mode 1 accepts two backbuffer colours and
the current frame's motion/depth. The arrays below come from the application:

```python
from xefg_sdk import FrameInputs, ImageView, Xefg2xRuntime

previous = FrameInputs(
    frame_id=0,
    backbuffer_color=ImageView(previous_bgra, "BGRA8_UNORM"),
)
current = FrameInputs(
    frame_id=1,
    backbuffer_color=ImageView(current_bgra, "BGRA8_UNORM"),
    motion_vectors=ImageView(current_mv, "RG16_FLOAT"),
    depth=ImageView(current_depth, "R32_FLOAT"),
)
with Xefg2xRuntime.load("/data/model_package") as runtime:
    result = runtime.interpolate_pair(previous, current)
    generated_bgra = result.generated_frame.data
```

Colours are `uint8 (H, W, 4)`. Motion is `float16 (h, w, 2)` (float32 is also
accepted with `RG32_FLOAT`); depth is `float32 (h, w)`. Current profiles use
`h=H/2`, `w=W/2`. The dimensions must match a profile in the supplied package.

All modes use the current frame's motion/depth; their colour inputs differ:

| Mode | Previous/current colour pair | Additional current-frame input |
| --- | --- | --- |
| 1 — `NONE` | `backbuffer_color` | None; UI is interpolated with the image |
| 3 — `HUDLESS_UITEXTURE` | `hudless_color` | Premultiplied `ui` |
| 4 — `BACKBUFFER_HUDLESS` | `hudless_color` | `backbuffer_color` with UI already composed |

`AUTO` selects mode 3 when a hudless pair and UI are available, otherwise mode 1
from a backbuffer pair. Select mode 4 explicitly with
`Config(ui_mode=UiMode.BACKBUFFER_HUDLESS)`; it needs no separate UI texture.
`Xefg2xSession.submit(frame)` retains previous-frame snapshots and honours
`FrameInputs.reset_history`.

**The engine/adapter owns coordinate adaptation.** Motion must already contain
complete current-to-previous displacement in MV-grid pixels, including camera
motion and jitter compensation. Depth uses the raw fg78 non-inverted convention;
“normalized” does not mean rescaling it to `[0, 1]`. Core does not accept camera
matrices, native MV scales, or jitter offsets. See the full
[input/output contract](docs/IO_CONTRACT.md).

The [pair and sequence examples](examples/) read ordinary application `.npz`,
`.npy`, or raw image files through a JSON sequence manifest. No trace is used by
runtime. A custom executor can replace ORT through `load`, `run`, and `close`;
see [the executor protocol](docs/MODEL_PACKAGE.md#7-runtime-use).

## Scope

- 2x midpoint interpolation with UI modes 1, 3, and 4.
- Five output sizes in each mode: 640×360, 1280×720, 1920×1080, 2560×1440,
  and 3840×2160, with half-resolution motion/depth. These are
  [reproducible profiles](docs/PROFILES.md), not bundled model assets.
- Exact profile selection, not arbitrary-resolution planning.
- Standard ONNX opset 18 / IR 8. Backend support for the mixed integer/float
  operator set must be checked; ONNX alone does not guarantee NPU compatibility.
- No engine adapter, presentation scheduling, HDR path, or performance guarantee.
- Packaging preserves the research implementation, including its known
  numerical differences from original XeFG. See [validation](docs/VALIDATION.md)
  for measured results and their scope.

## Develop and test

Install the package before running tests; tests do not inject source paths:

```sh
python -m pip install -e ".[test]"
python -I -B -m unittest discover -s tests -v
```

Synthetic unit tests and the small real-ORT smoke test need no captures.
Capture integration tests are opt-in, with no machine-specific dataset defaults.
See [development](docs/DEVELOPMENT.md) for fixture paths and native-tool tests.
The independent [render-validation tool](tools/render-validation/README.md)
captures official presents, verifies frame pairing, and compares same-input SDK
replay against official output. Its DX12 renderer covers nine scenes: rigid and
nonrigid motion, layered depth, transparent content, occlusion, camera movement,
and history resets. The supported matrix is 135 mode/size/scene combinations and
1065 generated frames. Measured outcomes and gates are in
[render validation](tools/render-validation/VALIDATION.md). Validation artifacts
and model packages are kept outside the source checkout; they are not runtime
dependencies or part of the Python package.

```text
src/xefg_sdk/          public runtime API, package loader, planner, executors
src/xefg_sdk/export/   exporter and internal capture/graph reconstruction modules
tools/capture/        optional native capture source, CMake, PowerShell scripts
tools/render-validation/ official-present capture inventory and offline checks
examples/             application input examples
tests/                synthetic tests and opt-in capture integration
docs/                 contracts, compatibility, capture interface, validation
```

## Source and licensing

Extracted from `MFG-Unlocker` commit `3d67197` (2026-11-10), including the minimal
XefgProbe capture source. Game injection/unlocking integrations and the wider
reconstruction experiments remain upstream. No upstream checkout is required to
build the capture tool or use this SDK. The standalone runtime retains its
existing public API; internal export modules are not a stable API.

Source is under the [MIT License](LICENSE), with the original copyright retained.
Extracting a model does not place its Intel-derived weights or LUT under MIT.
Review applicable Intel terms before redistributing generated assets; see
[NOTICE.md](NOTICE.md). No model redistribution or affiliation is implied.
