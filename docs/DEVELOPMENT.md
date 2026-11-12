# Development

## Source separation

This repository is the standalone XeFG 2x SDK extracted from
`MFG-Unlocker` at commit `3d67197` (2026-11-10). The current API version is
`0.3.0`; the model-package schema is `1`.

| Area | Location |
| --- | --- |
| Public runtime (`Xefg2xRuntime`, `FrameInputs`, profiles, package loader, executors) | `src/xefg_sdk/` |
| Exporter and internal reconstruction modules | `src/xefg_sdk/export/` |
| Application-input examples | `examples/` |
| Tests | `tests/` |
| Contracts and specifications | `docs/` |
| Optional XefgProbe capture producer | `tools/capture/`, independently built with CMake |
| Independent render validation | `tools/render-validation/`, outside the Python package |
| Game injection/unlocking, wider experiments | external `MFG-Unlocker` project |

This repository contains source only. It does not ship captures, ONNX models,
weights, quantisation constants, the gate LUT, Intel DLLs, or native binaries.
The exporter consumes captures produced by the optional native tool or retained
compatible captures and writes a model package to an empty output directory.
The native tool is not built by pip and is not included in the Python wheel.
The source distribution includes its source, scripts, tests, and documentation.

The public runtime API is the supported interface. `xefg_sdk.export` modules
are implementation details of the exporter and are not a stable API.

## Install

Python 3.10 or newer:

```sh
python -m pip install -e .                 # runtime core (NumPy only)
python -m pip install -e ".[ort]"          # ONNX Runtime executor
python -m pip install -e ".[export]"       # exporter (ONNX, no ORT needed)
python -m pip install -e ".[export,ort]"   # exporter plus executor
python -m pip install -e ".[test]"         # test dependencies
```

The distribution name is `xefg-sdk`; the import package is `xefg_sdk`. The
console entry point is `xefg-export`.

## Commands

Run tests:

```sh
python -I -B -m unittest discover -s tests -v
```

Export a package from compatible captures:

```sh
xefg-export --runs <capture-run-a>,<capture-run-b> --out <package-directory>
```

Use `--frame N` to select a retained frame (default 1) and `--debug` for
diagnostic outputs. The output directory must be absent or empty.

Run one pair or a sequence after installing the SDK and exporting a package:

```sh
python -I examples/pair.py --manifest work/sequence/sequence.json --package work/model_package --output work/out
python -I examples/sequence.py --manifest work/sequence/sequence.json --package work/model_package --output work/out
```

Synthetic tests need no captures. Integration tests use explicit fixture paths:

| Environment variable | Capture fixture |
| --- | --- |
| `XEFG_SDK_MODE1_RUN` | Mode 1, half-resolution MV/depth |
| `XEFG_SDK_MODE3_RUN` | Mode 3, 640x360 |
| `XEFG_SDK_MODE3_SIZE_RUN` | Mode 3 at another size, for cross-size comparison |
| `XEFG_SDK_MODE4_RUN` | Mode 4, 640x360 |
| `XEFG_SDK_MODE4_ANIMATED_RUN` | Mode 4, 640x360 with animated UI |
| `XEFG_MODE4_CAPTURE` | Mode 4 with `application_inputs/`, for provenance |

Unset variables skip the corresponding tests; invalid configured paths fail.
See [Validation](VALIDATION.md) for the test layers, commands, and the link to
the measured render matrix.

## Optional native tool tests

The capture tool is not part of the Python build. With a user-provided XeSS SDK:

```powershell
cmake -S tools/capture -B "$env:TEMP/xefg-sdk-capture-build" `
  -G "Visual Studio 17 2022" -A x64 "-DXESS_SDK_ROOT=$env:XESS_SDK_ROOT"
cmake --build "$env:TEMP/xefg-sdk-capture-build" --config Release --parallel
ctest --test-dir "$env:TEMP/xefg-sdk-capture-build" -C Release --output-on-failure
powershell -NoProfile -File tools/capture/script-tests/run-tests.ps1
```

The script tests use Windows PowerShell and command stubs; they do not build
native code, run XeFG, or require a GPU. The scene test tests CPU data generation.
See [Capture guide](CAPTURE_GUIDE.md) for the real capture/export sequence.

## Independent render-validation tests

The harness under `tools/render-validation/` is separate from the Python package
and from the capture producer. It captures official presents, checks frame
pairing, replays the same inputs through the SDK, and runs the native raster
scenes. Build and capture commands are in the
[render-validation README](../tools/render-validation/README.md).

```sh
python -I -B -m unittest discover -s tools/render-validation/tests -v
ctest --test-dir <native-build-dir> -C Release --output-on-failure
```

A real matrix also needs the provider runtime, a frozen renderer, and a model
package. Measured results are kept in the render-validation result set, not in
this guide.

## Documentation map

- [Quickstart](QUICKSTART.md): install, create application inputs, run examples.
- [IO contract](IO_CONTRACT.md): `FrameInputs`, formats, profiles, errors.
- [Model package](MODEL_PACKAGE.md): package layout, manifest, graph inputs.
- [Profiles](PROFILES.md): supported mode/size combinations, including mode-4 sizes.
- [Capture guide](CAPTURE_GUIDE.md): optional Windows build, capture and export.
- [Capture format](CAPTURE_FORMAT.md): capture/export contract revision 1.
- [Compatibility](COMPATIBILITY.md): semantic scope, version history, sampler
  precision, and non-guarantees.
- [Validation](VALIDATION.md): testing layers, commands, and the measured matrix
  link.

## Licensing

The project source is MIT licensed. Intel-derived program data, extracted model
assets, and generated ONNX packages are outside that grant and remain subject to
Intel's XeSS terms. See [NOTICE](../NOTICE.md).
