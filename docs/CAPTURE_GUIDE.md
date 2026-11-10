# Capture → export → runtime

The optional Windows capture tool is included in this checkout under
[`tools/capture`](../tools/capture/README.md). You do not need an
`MFG-Unlocker` checkout. Existing captures remain valid; capturing again is only
necessary when you need new source data, such as another size/configuration.

## Separation of dependencies

| Stage | Inputs | Requirements |
| --- | --- | --- |
| Capture | User-provided XeSS SDK and reference scene configuration | Windows x64, MSVC, CMake, Windows SDK/DXC, a GPU capable of running XeFG |
| Export | Retained capture directory | Python, NumPy, ONNX (`.[export]`) |
| Runtime | Exported package and normalized application frames | Python, NumPy and an executor (`.[ort]` for the reference implementation) |

Only capture loads the original provider DLL and uses D3D12. Exporting existing
captures does not require Windows, a GPU, the DLL, or a C++ toolchain. Installing
the Python SDK does not build the capture executable. The wheel contains the
Python SDK; the source distribution also contains capture source and scripts.

The capture tool runs a deterministic reference scene, not a game. It observes
provider dispatches and captures resources using the inherited XefgProbe
instrumentation. It does not install a system driver, replace a game DLL, or
implement a deployment inference backend.

## 1. Prerequisites

- Windows x64 with an interactive desktop and a D3D12 GPU that can run XeFG.
  The retained baseline uses an AMD Radeon RX 6800 XT. WARP alone is not a
  replacement for the GPU needed by the provider.
- Visual Studio 2022 / Build Tools with the Desktop development with C++ workload.
- CMake 3.24 or newer (the Visual Studio bundled copy is supported).
- A Windows SDK with the x64 `dxc.exe` tool, used to disassemble captured DXIL.
- A user-provided XeSS SDK 3.0.2 directory containing:

  ```text
  inc/xess_fg/xefg_swapchain_d3d12.h
  inc/xell/xell_d3d12.h
  bin/libxess_fg.dll
  bin/libxell.dll
  ```

Provider internals are version-specific. The supported/export-validated family
is `fg78`; the retained provider SHA-256 is
`ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27`.
A matching public header version is not a promise that every provider build has
the same internal layout. Capture metadata records the actual DLL hash/family.
No SDK headers, binaries, or generated program/model data are bundled or fetched.

## 2. Capture both SDK modes

Run PowerShell from the checkout root. Choose a work directory outside the source
checkout with enough free disk space. Each capture output must be new or empty;
the build directory may be reused.

```powershell
$sdk = "C:/SDKs/XeSS_SDK_3.0.2"  # replace with your own SDK location
$work = Join-Path $env:TEMP ("xefg-sdk-work-" + [guid]::NewGuid().ToString("N"))

powershell -NoProfile -File tools/capture/run.ps1 `
  -XessSdkRoot $sdk -OutDir "$work/mode1" -BuildDir "$work/build" `
  -Mode 1 -Width 640 -Height 360

powershell -NoProfile -File tools/capture/run.ps1 `
  -XessSdkRoot $sdk -OutDir "$work/mode3" -BuildDir "$work/build" `
  -Mode 3 -Width 640 -Height 360
```

Use `-Adapter "6800"`, for example, to select a matching adapter description on a
multi-GPU machine; without an explicit selector the tool chooses an eligible
hardware adapter. `-XessSdkRoot` can be replaced by the `XESS_SDK_ROOT` environment
variable. If automatic tool discovery does not find your installation, supply
`-CMake <path-to-cmake.exe>` and `-Dxc <path-to-dxc.exe>`.

The runner:

1. Checks configuration and tool paths, then builds the local `XefgProbe` target.
2. Runs the provider with trace and tensor capture enabled.
3. Checks metadata/readback and retained tensor-frame completeness.
4. Runs DXC disassembly, failing if a required shader cannot be disassembled.

No Python research replay or old upstream script is invoked. The resulting
`trace/meta.json`, `trace/trace.jsonl`, tensor dumps, and disassembly are sufficient
for `xefg-export`; see [Capture format](CAPTURE_FORMAT.md).

### Defaults and supported configuration

- `-Mode 1`: UI off (`NONE`); `-Mode 3`: hudless colour + UI (`HUDLESS_UITEXTURE`).
- `-Width 640 -Height 360`; MV/depth dimensions are always half each axis.
  Output dimensions must be positive and even.
- `-Frames 2 -Warmup 4 -Velocity "12,4" -Scene standard`.
- `-TensorDispatches 76`: retain the first two complete 38-dispatch interpolation
  frames. Trace metadata can include later frames without tensor snapshots.
- `-Static` selects a stationary scene; `-Scene smooth` selects the inherited
  smooth-content scene. Neither changes the core SDK's application contract.
- `-TensorBudgetMB` can explicitly set the readback/dump budget. The automatic
  budget scales from 6144 MB at 4K, with a 512 MB floor and 6144 MB cap, for the
  default two-frame retention. Increase it explicitly when retaining more data.

`Frames` controls recorded presentation frames after warmup; it is not the same
as the number of tensor-complete trace frames. The provider produces interpolation
work during warmup as well. With defaults, export trace frames 1 or 2; do not
assume later metadata-only frames can be exported. Increasing tensor retention
requires enough provider frames and memory/disk budget; incomplete captures fail.

For another supported profile, change output dimensions (720p, 1080p, 1440p or
3840×2160). Larger captures consume substantial memory/disk and are not included
in the source repository. Keep each mode/size in its own output directory.

## 3. Export

Install the Python exporter once from this checkout:

```powershell
python -m pip install ".[export,ort]"
xefg-export --runs "$work/mode1,$work/mode3" --out "$work/package"
```

The package contains two variants and two profiles for this example. Add more
compatible captures to `--runs` to build a larger profile registry. Use `--frame 2`
to rebuild from the second retained tensor frame, or `--debug` for diagnostic
graph outputs. The package output directory must be absent or empty.

Export does not run the provider or call the capture tool automatically. A
successful export is not a promise of provider byte-exactness or arbitrary-size
support. See [Validation](VALIDATION.md) and [Profiles](PROFILES.md).

## 4. Runtime

Only the exported package is needed by an application. Capture data, original
DLLs and the C++ toolchain are not runtime dependencies. Use the
[quickstart](QUICKSTART.md) and [input contract](IO_CONTRACT.md) to provide ordinary
application colour/MV/depth/UI arrays. Engine-side coordinate adaptation remains
the caller's responsibility.

## Build and scene tests without capture

```powershell
cmake -S tools/capture -B "$work/build" -G "Visual Studio 17 2022" -A x64 `
  "-DXESS_SDK_ROOT=$sdk" -DBUILD_TESTING=ON
cmake --build "$work/build" --config Release --parallel
ctest --test-dir "$work/build" -C Release --output-on-failure
```

The scene test exercises the CPU reference scene generation, not provider
accuracy. Running it does not perform a GPU capture.

For existing raw captures that only lack disassembly:

```powershell
powershell -NoProfile -File tools/capture/disasm.ps1 -RunDir "$work/mode1"
```

## Licensing and data handling

Capture writes shader programs, parameter/tensor snapshots and metadata derived
from the supplied provider. These outputs, and exported model assets, are not
made MIT-licensed by this tool. Review applicable Intel terms before sharing
captures or models; see [NOTICE](../NOTICE.md). Keep generated data and builds
outside the checkout and do not include them in source releases.
