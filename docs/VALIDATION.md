# Validation status

## Standalone validation — 2026-11-10

This repository was split from `MFG-Unlocker` at commit `3d67197`. Its wheel was
built offline and installed non-editably into a separate virtual environment.
The environment reused installed third-party dependencies: Python 3.13.5,
NumPy 2.4.3, ONNX 1.23.2, and ONNX Runtime 1.31.1 on Windows x64.

| Check | Result |
| --- | --- |
| Tests without capture environment variables | 39 passed; 3 capture test classes skipped |
| Complete suite with all three explicit capture paths | 48 passed; no skips |
| Ten-profile export with ORT imports blocked | Passed |
| Export reads from original repository blocked | Passed |
| Exported model SHA-256 vs retained package | Both variants identical |
| Exported profile JSON vs retained package | All 10 equal, including all 307 feeds per profile |
| `onnx.checker` on exported models | Both passed |
| Runtime source vs upstream source | All 12 files byte-identical |
| Independent application-input inference, mode 3 / 640x360 | Two generated frames match retained hashes |
| Pair and sequence examples | Three NPY outputs match; three PNGs written |

The runtime process blocked capture and original-repository reads, and blocked
imports of ONNX and the exporter. It loaded the installed SDK, exported model
package, ordinary application image files, and the optional ORT executor only.
The export check also loaded the installed SDK rather than the source checkout.

Deployment model SHA-256:

- mode 1: `5afc74d967df8c3069deebf61e392307f074d4cdb4556ff5ec805a201704583c`
- mode 3: `91152b622504ccdadf3cc4e051704a0a8f28c9534a45475cc920c10138438626`

Generated frame pixel-byte SHA-256 (mode 3, 640x360):

- frame 1: `ef0885d8f1d16412a45ed6ff2aea1ad13a0c5ffc3fe69b94ca87c5323fef563b`
- frame 2: `c88e52bd01f168d3e91352ccdc238d9bf34a39efdb999b012aa56fab40b26d86`

These checks establish preservation across repository extraction, not a new
provider comparison or fresh native capture. All ten profiles were re-exported;
application inference was rerun at 640x360, not at every profile size. No NPU run
or real-time performance claim is made. Artifacts and logs remain outside this
source tree and are not runtime dependencies.

The first pair-example invocation used `--limit 1` and was correctly rejected;
rerunning with `--limit 2` passed. The completed installed-package suites had no
failures.

## Included capture-tool validation — 2026-11-10

The native producer was subsequently migrated into `tools/capture/` and tested
end-to-end without building or invoking the upstream repository:

- VS 2022 / MSVC 14.44.35207, Windows SDK 10.0.26100.0, Windows PowerShell 5.1,
  SDK-provided DXC, and the user-supplied XeSS SDK 3.0.2.
- Built `XefgProbe` and `XefgSceneTests` from the standalone CMake project.
  CTest passed **1/1**. The inherited `xefg_runtime.cpp` emits one C4244
  wchar-to-char conversion warning; the build passed.
- **11 PowerShell script checks passed**, including output protection and DXC
  error handling. No additional PowerShell test dependency is required.
- Captured standard moving content at 640x360 in modes 1 and 3 on an AMD Radeon
  RX 6800 XT, with 320x180 MV/depth, `Frames=2`, `Warmup=4`, velocity `12,4`, and
  76 tensor dispatches. Both runs use the `fg78` provider hash documented in the
  [capture guide](CAPTURE_GUIDE.md).
- Each run retained **420 tensor files / 91,257,216 bytes**, 46 shader
  disassemblies, and complete tensor frames 1 and 2. `tensor_wait` was true and
  `trace_error` was empty. Later frames have trace metadata only.
- Only the new runner and DXC postprocessing were used. No upstream
  extract/ops/recon/verify script was invoked; no model-catalog dump was needed.
- Exported both captures independently from frames 1 and 2, with ORT imports and
  upstream-source reads blocked. All four model exports matched the deployment
  hashes above; both profile JSON objects (307 feeds each) matched the retained
  profiles for each frame. `onnx.checker` passed.
- The full Python suite passed **48/48** using the fresh 640 mode-1/mode-3
  captures and a retained 1080 mode-3 capture for cross-size comparison. Without
  capture paths, **39** tests passed and 3 integration classes were skipped.
- Inference using the fresh exported package and retained ordinary application
  inputs reproduced both mode-3 pixel hashes above. Runtime capture/source reads
  and exporter/ONNX imports remained blocked.

The 22 native implementation files under `tools/capture/src/` are byte-identical
to their upstream counterparts. Only the build layout and shared fixture were
adapted; the scene and capture arithmetic were not changed. This new capture
validation covers 640x360 in both modes, not a fresh capture of all ten profiles.
It establishes capture/export/runtime continuity, not new provider-equality or
NPU-performance evidence. Generated assets remain outside the source checkout.

## Historical source evidence

The source implementation recorded these results before extraction.

### SDK tests

- Version 0.2.0 recorded **42 SDK/exporter/provenance/example tests passed**.
- The wheel was installed in an independent directory and run with generated
  application inputs, not captured ONNX feeds.
- A later source regression also recorded 42 SDK tests and 153 complete oracle
  tests. Those oracle tests depend on retained captures and are not part of this
  source-only repository.
- Both deployment ONNX models passed `onnx.checker`.

### Independent application-input runs

A separate process loaded only the installed SDK, the model package, and
ordinary application BGRA/RG16F/R32F files. A file audit hook rejected reads
from capture, trace, shader-disassembly, and tensor-dump paths. No `recon`,
`m9_dynamic`, `m9_graph`, or `export_onnx` module was loaded by the SDK process.

| Mode / resolution | Generated frame | SDK vs full research reconstruction | SDK vs provider capture: bytes / max delta |
| --- | ---: | ---: | ---: |
| mode 3 / 640x360 | 1 | 0 | 12 / 1 |
| mode 3 / 640x360 | 2 | 0 | 9 / 1 |
| mode 1 / 1920x1080 | 1 | 0 | 3805 / 27 |
| mode 1 / 1920x1080 | 2 | 0 | 29 / 1 |

The comparison against the research graph was performed afterward in a separate
offline verification process. All four outputs were byte-identical to that
graph. This establishes that the SDK packaging and input preparation preserved
the existing implementation for those inspected frames; it does **not** claim
equality with the provider or with Intel XeFG.

### Profile expansion evidence

The 4K, 1440p, 720p, and earlier size expansions recorded additional evidence:

- 4K mode 1 and mode 3 frames 1 and 2 were complete (420 tensor files,
  2,983,089,600 bytes) and matched the retained full research reconstruction
  with 0 differing bytes.
- 1280x720 and 2560x1440 mode-1 frames matched the retained reconstruction with
  0 differing bytes.
- In each case the provider-capture comparison remained nonzero; for example,
  the 4K frames recorded 26/1 and 2282/25 for mode 1 and 27/1 and 2283/25 for
  mode 3.
- All 307 profile metadata feeds matched the corresponding captured frame plans.

These are historical source results. They are not provider-equality claims and
have not been re-run for this standalone checkout.

### Known numerical cautions

- The SDK reproduces the project's full research reconstruction for inspected
  frames. That is not the same as reproducing the provider byte for byte.
- `d25` (warp + flow init) is the one individually non-byte-exact dispatch in
  the inspected captures. Small residuals there propagate through later flow
  stages and can produce larger final-image differences. Research calibration
  and structural checks are not universal numerical error guarantees.
- No NPU support, performance, latency, or memory result is claimed. ONNX
  validity and an ONNX Runtime CPU run do not establish that an NPU backend can
  execute the mixed integer/float graph.

## Reproduce tests in this checkout

```sh
python -m pip install -e ".[test]"
python -I -B -m unittest discover -s tests -v
```

Capture integration tests are opt-in through `XEFG_SDK_MODE3_RUN`,
`XEFG_SDK_MODE1_RUN`, and `XEFG_SDK_MODE3_SIZE_RUN`. They are skipped without
those paths. Synthetic tests and the small ONNX Runtime smoke test need no
captures.

For exporter-side checks, install `.[export]` and run the export tests with a
capture that satisfies [Capture format](CAPTURE_FORMAT.md).

## Interpretation

The historical evidence and standalone preservation checks support this claim:
for the inspected frames and profiles, the SDK package and input preparation
preserve the project's existing full reconstruction. They do not support claims
of provider byte-exactness, arbitrary-resolution coverage, or NPU support.
