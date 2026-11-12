# Ready-to-use SDK profiles

The SDK supports five output sizes in UI modes 1 and 3, each with a
half-resolution motion/depth grid. That is ten profiles served by two graph
variants: one model for all mode-1 sizes and one for all mode-3 sizes. Version
0.3.0 includes an optional mode-4 family (`BACKBUFFER_HUDLESS`); all five sizes
through 4K are supported and included in the 15-profile package.

**Profile metadata is not model data.** This repository does not ship the ONNX
models, weights, quantisation constants, gate LUT, or profile files for these
sizes. Export them from compatible captures, or obtain a package separately, as
described in [Model package](MODEL_PACKAGE.md).

## Available combinations

| Output | MV/depth | Mode 1: no separate UI composition | Mode 3: hudless + premultiplied UI | Mode 4: hudless + current backbuffer |
|---|---|---|---|---|
| 640x360 | 320x180 | yes | yes | yes |
| 1280x720 | 640x360 | yes | yes | yes |
| 1920x1080 | 960x540 | yes | yes | yes |
| 2560x1440 | 1280x720 | yes | yes | yes |
| 3840x2160 | 1920x1080 | yes | yes | yes |

All five mode-4 sizes are supported and included in the 15-profile package. The
SDK/package output is checked against the project's own reconstruction for the
inspected frames; provider comparisons remain bounded. `d25` is bounded,
not byte-exact. See [Validation](VALIDATION.md) and the measured matrix it links
to.

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

## Validation status

The matrix above is configuration data, not a quality preset. All fifteen
mode/size combinations (five sizes times three modes) can be exported from
compatible captures. The resulting package contains one profile per supplied
combination and up to three model variants (mode 1, mode 3, and optional mode 4).

Measured evidence, including modes, sizes, scenes, numeric gates, and the scope
of the inputs actually exercised, is maintained in
[render-validation/VALIDATION.md](../tools/render-validation/VALIDATION.md).
External results are not shipped with this repository.

The comparison claims are bounded:

- SDK/package output is checked against the project's own research
  reconstruction for inspected frames. That is not equality with Intel XeFG or
  any provider.
- `d25` (warp + flow init) is bounded, not byte-exact. Its measured sampler rule
  and the allowed one-LSB color-store residual are documented in
  [Compatibility §5.4](COMPATIBILITY.md#54-d25-sampling-precision) and enforced
  by `src/xefg_sdk/export/d25_bound.py`.
- 640×360 checks use independent application images. Larger-size checks use
  application-shaped arrays recovered from capture resources, so they validate
  packaging and execution rather than independent application-input provenance.
- Mode 4 uses the previous/current hudless pair plus the current composed
  backbuffer. Its captures and `d38` mapping are checked; provider comparisons
  retain a small residual, and no provider equality is claimed.

## Re-export the package

From a checkout with the export extra installed:

```sh
xefg-export --runs <run-640-mode1>,<run-640-mode3>,<run-640-mode4>,<run-720-mode1>,<run-720-mode3>,<run-720-mode4>,<run-1080-mode1>,<run-1080-mode3>,<run-1080-mode4>,<run-1440-mode1>,<run-1440-mode3>,<run-1440-mode4>,<run-4k-mode1>,<run-4k-mode3>,<run-4k-mode4> --out <new-package-directory> --debug
```

This is the 15-profile layout (five sizes times three modes). The mode-4 runs
for all five sizes are available, and the resulting package is validated as
described above. Changing Python source does not patch existing ONNX files;
re-export to pick up exporter changes.

- Supply one compatible capture run per profile. The exporter reads the mode
  from the captured `d38` composite plan and the sizes from the captured feeds.
- The output directory must be absent or empty.
- `--debug` is optional; without it the package omits diagnostic model files.
- Do not include a 1280x720 output with 320x180 MV. That quarter-size MV
  configuration is not the half-resolution model contract.
- Model assets are derived from the supplied captures. The exporter does not
  fetch, bundle, or reconstruct a missing capture.

The resulting package has `manifest.json`, `models/mode1.onnx`,
`models/mode3.onnx`, optionally `models/mode4.onnx`, and one profile JSON per
supplied mode/size. If a size or mode is missing from the package, the runtime
raises instead of guessing a nearby profile.
