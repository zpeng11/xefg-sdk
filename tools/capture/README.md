# Optional XeFG capture tool

This directory contains the native XefgProbe producer for SDK captures. It builds
independently of the MFG-Unlocker checkout and uses a user-provided XeSS SDK.

See the [capture guide](../../docs/CAPTURE_GUIDE.md) for prerequisites, supported
configurations, and the capture-to-export sequence.

## Quick start

From the repository root:

```powershell
powershell -NoProfile -File tools/capture/run.ps1 `
  -XessSdkRoot "C:/SDKs/XeSS_SDK_3.0.2" `
  -OutDir "$env:TEMP/my-xefg-mode3" -Mode 3 -Width 640 -Height 360
```

Replace the SDK path with your own and choose a new or empty output directory.
The runner builds `XefgProbe`, records trace and tensors, and disassembles the
captured DXIL. Export is a separate step. By default, tensor capture retains the
first two complete interpolation frames (76 dispatches).

## Options

- `-Mode 1`: interpolate the backbuffer, including UI.
- `-Mode 3`: interpolate hudless colour and compose a separate UI texture.
- `-Mode 4`: interpolate hudless colour and extract UI from the backbuffer.
- `-AnimateUi`: move the reference scene's crosshair each frame; off by default.
- `-DumpInputs`: save raw application images and `application_inputs/manifest.json`
  for every frame, including warmup. This provenance manifest is distinct from
  the Python examples' sequence manifest.

See [profiles](../../docs/PROFILES.md) for supported sizes,
[validation](../../docs/VALIDATION.md) for test layers, and
[render results](../render-validation/VALIDATION.md) for measured official
comparisons.

## Layout

- `src/`: reference scene, provider loader, and trace/resource/tensor capture.
- `include/`: local fixture, hash utility, and provider-layout descriptions.
- `tests/`: CPU scene regression tests registered with CTest.
- `run.ps1`: build, capture, completion checks, and disassembly.
- `disasm.ps1`: DXC disassembly for an existing raw capture.
- `script-tests/`: PowerShell checks that run without a GPU capture.

The Python wheel excludes this directory and pip never builds it. The source
distribution includes source and scripts only. No provider DLL, SDK header,
captured program, model weight, tensor dump, or generated binary is included.

The native code was extracted from `MFG-Unlocker` commit `3d67197`, under the
repository's [LICENSE](../../LICENSE). Provider-derived capture/model outputs
are separate; see [NOTICE](../../NOTICE.md).
