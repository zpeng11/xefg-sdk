# Optional XeFG capture tool

This directory contains the minimum native XefgProbe producer required to create
captures for the SDK exporter. It is independent of the MFG-Unlocker checkout.

See [Capture guide](../../docs/CAPTURE_GUIDE.md) for prerequisites, configuration,
complete mode-1/mode-3 commands, export, and licensing notes.

```powershell
powershell -NoProfile -File tools/capture/run.ps1 `
  -XessSdkRoot "C:/SDKs/XeSS_SDK_3.0.2" `
  -OutDir "$env:TEMP/my-xefg-mode3" -Mode 3 -Width 640 -Height 360
```

Replace the SDK path with your own. `OutDir` must be new or empty. This command
builds the local `XefgProbe` target, runs the user-provided provider, records trace
and tensors, and disassembles captured DXIL. It does not invoke the exporter or
run the upstream research validation scripts. The default tensor retention is
the first two complete interpolation frames (76 dispatches).

- `src/`: reference scene, provider loader, trace/resource/tensor capture.
- `include/`: local fixture, hash utility, and provider-layout descriptions.
- `tests/`: CPU scene regression tests registered with CTest.
- `run.ps1`: build + capture + completion checks + disassembly.
- `disasm.ps1`: DXC disassembly for an existing raw capture.

The Python wheel excludes this directory and pip never builds it. The source
distribution includes source and scripts only. No provider DLL, SDK header,
captured program, model weight, tensor dump, or generated binary is included.

The native code is extracted from `MFG-Unlocker` commit `3d67197`, under the
repository's existing [LICENSE](../../LICENSE). Provider-derived capture/model
outputs are separate; see [NOTICE](../../NOTICE.md).
