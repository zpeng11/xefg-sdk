# Ready-to-use SDK profiles

The SDK supports five output sizes in UI modes 1 and 3, each with a
half-resolution motion/depth grid. That is ten profiles served by two graph
variants: one model for all mode-1 sizes and one for all mode-3 sizes. Version
0.3.0 adds an optional mode-4 family (`BACKBUFFER_HUDLESS`); all five sizes
through 4K are supported and included in the confirmed 15-profile package.

**Profile metadata is not model data.** This repository does not ship the ONNX
models, weights, quantisation constants, gate LUT, or profile files for these
sizes. Export them from retained captures, or obtain a package separately, as
described in [Model package](MODEL_PACKAGE.md).

## Available combinations

| Output | MV/depth | Mode 1: no separate UI composition | Mode 3: hudless + premultiplied UI | Mode 4: hudless + current backbuffer |
|---|---|---|---|---|
| 640x360 | 320x180 | yes | yes | yes |
| 1280x720 | 640x360 | yes | yes | yes |
| 1920x1080 | 960x540 | yes | yes | yes |
| 2560x1440 | 1280x720 | yes | yes | yes |
| 3840x2160 | 1920x1080 | yes | yes | yes |

All five mode-4 sizes are captured and included in the 15-profile package.
SDK outputs match the full reconstruction for both retained frames at every
size. The 640×360 checks use independent application images; larger-size checks
use application-shaped arrays recovered from capture resources. Provider
comparisons remain nonzero; see [Validation](VALIDATION.md).

These are layout/configuration profiles, not scene-specific models or quality
presets. Static and moving content do not select different profiles. "2K" means
2560x1440 here, and "4K" means 3840x2160. All motion/depth sizes are exactly half
the output width and height.

The runtime selects by mode, output size, and MV/depth size. It does not rescale
the inputs or choose the closest profile:

```python
from xefg_sdk import Config, UiMode, Xefg2xRuntime

with Xefg2xRuntime.load("model_package") as runtime:
    result = runtime.interpolate_pair(
        previous, current,
        Config(ui_mode=UiMode.NONE),
    )
```

For mode 1 provide backbuffer colour; for mode 3 provide hudless colour plus
premultiplied UI; for mode 4 provide the previous/current hudless pair plus the
current composed backbuffer. `AUTO` follows the supplied resources for modes 1
and 3 only; mode 4 must be requested explicitly with
`Config(ui_mode=UiMode.BACKBUFFER_HUDLESS)`.

## Mode 1/3 historical source evidence

The sizes above were added in stages and checked against retained captures in
the upstream `MFG-Unlocker` project at commit `3d67197`. This is historical
source evidence, not a fresh validation result for this standalone checkout.

- 4K mode 3 and mode 1 were captured at 3840x2160 with 1920x1080 MV/depth.
  Complete tensor frames 1 and 2 contained 420 files and 2,983,089,600 bytes,
  with fenced readback, no missing tensors, and no trace error. Frames 3..5 had
  metadata only and were not replayed.
- For every inspected 4K frame, the SDK/exported package output matched the
  retained full research reconstruction with 0 differing bytes. Comparisons
  against the provider capture remained nonzero; that is expected and is not an
  equality claim.
- The new 4K mode-3 and mode-1 profiles had all 307 profile metadata feeds
  matching both frame plans. The deployment models were unchanged from the
  earlier sizes, and both passed the ONNX checker.
- The 1280x720 and 2560x1440 mode-1 captures likewise matched the retained
  reconstruction with 0 differing bytes for the inspected frames. Their
  provider comparisons remained nonzero.
- Earlier 640x360 mode-1 and the added mode-3 sizes also matched the retained
  reconstruction with 0 differing bytes.

The known `d25` capture residual and the motion-content difference are separate
from this profile evidence; see [Capture format](CAPTURE_FORMAT.md#6-d25-is-bounded-not-byte-exact)
and [Validation](VALIDATION.md).

## Mode 4 evidence (five sizes validated, 15-profile package confirmed)

The mode-4 `BACKBUFFER_HUDLESS` path is a version 0.3.0 addition. Captures now
exist at 640×360, 1280×720, 1920×1080, 2560×1440, and 3840×2160; each retained
420 tensor files and 46 shader disassemblies, with `tensor_wait` true and an
empty `trace_error`. The 640×360 runs used 320×180 MV/depth and covered both the
standard scene and `-AnimateUi`.

For 640×360, the corrected mode-4 composite matches the captured `d38`
individually with 0 differing bytes on frames 1 and 2 for both scenes. Four
runtime outputs (standard and animated UI, two frames) matched the full research
reconstruction with 0 differing bytes. The full chained graph comparison against
the provider capture recorded 12 differing bytes with maximum delta 1 on frame 1
and 8 differing bytes with maximum delta 1 on frame 2, for both scenes. The
mode-1/mode-3 model hashes were unchanged by this correction. This is a small
residual, not provider equality, and the downstream `d25` propagation remains
part of the final report.

The 15-profile export is confirmed. All five mode-4 sizes use the same ONNX model
SHA-256 `b42f93a76a97a32bc03636b86206a06449fea787a12a35e8979217cbf815bca0`,
all frame-2 mode-4 profiles are identical to their frame-1 profiles, and the
entire mode-1/mode-3 model and profile objects are unchanged. All additional
sizes also match the full research reconstruction exactly on both frames. The
640×360 checks use independent application images; larger-size checks use
application-shaped arrays recovered from capture resources.

## Re-export the package

From a checkout with the export extra installed:

```sh
xefg-export --runs <run-640-mode1>,<run-640-mode3>,<run-640-mode4>,<run-720-mode1>,<run-720-mode3>,<run-720-mode4>,<run-1080-mode1>,<run-1080-mode3>,<run-1080-mode4>,<run-1440-mode1>,<run-1440-mode3>,<run-1440-mode4>,<run-4k-mode1>,<run-4k-mode3>,<run-4k-mode4> --out <new-package-directory> --debug
```

This is the 15-profile layout (five sizes times three modes). The mode-4 runs
for all five sizes are available, the resulting package has been confirmed, and
all five sizes match the full reconstruction exactly on both frames (640×360
with independent application images; larger sizes with application-shaped arrays
recovered from capture resources).

- Supply one compatible capture run per profile. The exporter reads the mode
  from the captured `d38` composite plan and the sizes from the captured feeds.
- The output directory must be absent or empty.
- `--debug` is optional; without it the package omits diagnostic model files.
- Do not include a 1280x720 output with 320x180 MV. That quarter-size MV
  configuration is not the current half-resolution model contract.
- Model assets are derived from the supplied captures. The exporter does not
  fetch, bundle, or reconstruct a missing capture.

The resulting package has `manifest.json`, `models/mode1.onnx`,
`models/mode3.onnx`, optionally `models/mode4.onnx`, and one profile JSON per
supplied mode/size. If a size or mode is missing from the package, the runtime
raises instead of guessing a nearby profile.
