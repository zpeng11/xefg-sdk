# Validation status

## Version 0.3.0 mode-4 validation — 2026-11-10

The mode-4 (`BACKBUFFER_HUDLESS`) path is now captured at 640×360, 1280×720,
1920×1080, 2560×1440, and 3840×2160. Each run retained **420 tensor files** and
46 shader disassemblies, with complete tensor frames 1 and 2, `tensor_wait` true,
and an empty `trace_error`. The 640×360 runs used 320×180 MV/depth and covered
both the standard scene and `-AnimateUi`.

The captured `d38` input mapping is confirmed: `t8` is the current hudless colour
and aliases `current_color`, `t10` is the current backbuffer, and `t9` remains a
bound placeholder that mode 4 does not consume. No separate UI texture is bound.

Two previously unvalidated mode-4 issues were corrected: the ONNX distance sum
referenced a nonexistent `cp_dsum_0`, and the final blend used intermediate `m2`
instead of the shader's full blend `m`. Captured DXIL and numerical replay
established the latter correction. The corrected graph matches captured `d38`
individually with **0 differing bytes**
on frames 1 and 2 for both standard and animated UI. The corrected full chained
graph, compared with the provider capture, records:

| Scene | Frame | Differing bytes | Max delta |
| --- | ---: | ---: | ---: |
| standard | 1 | 12 | 1 |
| standard | 2 | 8 | 1 |
| animated UI | 1 | 12 | 1 |
| animated UI | 2 | 8 | 1 |

That is a small residual, not provider equality; differences originating at
`d25` propagate through the downstream graph. Mode-1 and mode-3 deployment model
hashes were unchanged by the mode-4 fix.

At 640×360, four runtime outputs (standard and animated UI, frames 1 and 2)
matched the full research reconstruction with 0 differing bytes. The 15-profile
export is confirmed: all five mode-4 sizes use the same ONNX model SHA-256
`b42f93a76a97a32bc03636b86206a06449fea787a12a35e8979217cbf815bca0`,
and all frame-2 mode-4 profiles are identical to their frame-1 profiles. The
mode-1 and mode-3 model and profile objects are unchanged. The full Python suite
passed **83/83** with all capture environment variables enabled.

The raw mode-4 graph has 320 serialized inputs and 6039 nodes before public
packaging. The packaged model has 6045 nodes, 312 serialized inputs (five
application resources plus 307 profile feeds), and one output.

The remaining mode-4 sizes were also executed through both the SDK and full
reconstruction for frames 1 and 2:

| Output | Frame | SDK vs full reconstruction: bytes | SDK vs provider: bytes / max delta |
| --- | ---: | ---: | ---: |
| 1280x720 | 1 | 0 | 47 / 1 |
| 1280x720 | 2 | 0 | 24 / 1 |
| 1920x1080 | 1 | 0 | 3805 / 27 |
| 1920x1080 | 2 | 0 | 29 / 1 |
| 2560x1440 | 1 | 0 | 3545 / 23 |
| 2560x1440 | 2 | 0 | 44 / 1 |
| 3840x2160 | 1 | 0 | 26 / 1 |
| 3840x2160 | 2 | 0 | 2282 / 25 |

For these larger-size comparisons, application-shaped arrays were reconstructed
from captured external resources. They validate packaging and execution, but
are not independent application-input provenance tests. The separate 640x360
runs use ordinary application-side images dumped directly by `GenerateScene`.
Capture access and exporter imports were blocked in those runtime processes.
The default scene's application bytes for frames 0..2 match the prior baseline.

Initial failed composite measurements were retained before the formula fix;
no retries or output patches were used to hide differences. Provenance reports
remain `partial` for complete traces because frames 3..5 have no tensor dumps;
the retained frames 1/2 establish the current backbuffer/hudless identities.

## Installed v0.3.0 wheel validation — 2026-11-10

A non-editable v0.3.0 wheel was installed into a separate environment and
checked:

| Check | Result |
| --- | --- |
| Unit tests | 65 passed; 5 classes and 1 test skipped |
| Integration tests | 83/83 passed; zero skips |
| Capture tool | CTest 1/1; 12 script checks passed |
| Installed pure export, ORT imports blocked | All 15 profiles passed |
| Installed application runtime | Four generated frames exact |
| Pair and sequence examples | Three NPY outputs match; three PNGs written |

ORT imports were blocked for the export check. The application runtime used the
installed wheel and application inputs; all four generated frames matched the
full reconstruction.

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
of provider byte-exactness, arbitrary-resolution coverage, or NPU support. The
mode-4 addition is validated at 640×360 in both standard and animated-UI scenes;
its full-chain provider comparison has a small residual. All five mode-4 sizes
are captured and packaged and match the full research reconstruction exactly on
both frames at every size (640×360 with independent application images; larger
sizes with application-shaped arrays recovered from capture resources).
