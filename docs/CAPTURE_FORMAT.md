# Capture trace format (revision 1)

`xefg-export` consumes retained capture-run directories produced by XefgProbe.
The optional native producer is included as source in
[`tools/capture`](../tools/capture/README.md), separate from the Python package.
It was extracted from `MFG-Unlocker`'s `tools/xefg_oracle` without changing the
trace format; existing compatible captures remain usable. See the
[capture guide](CAPTURE_GUIDE.md) to create new captures.

This document describes **contract revision 1** of that trace as consumed by the
exporter. It is descriptive, not an embedded schema. There is no schema file in
this repository, and the field lists below are not a license to invent new
events. If the producer changes an event or file name, this contract
must be revised before the exporter can be expected to read it.

The exporter reads capture directories only. It does not load
`libxess_fg.dll`, does not record D3D12 commands, and does not modify the capture.
It writes a model package only to an empty output directory.

## 1. Run directory layout

```text
<run>/
  trace/
    meta.json
    trace.jsonl
    shaders/<key>-<sha256-prefix>.dxil
    params/<key>-weights.bin
    params/<key>-bias.bin
    params/<key>-activation.bin
    tensors/d<dispatch>_<slot>_in.bin
    tensors/d<dispatch>_<slot>_out.bin
    disasm/<key>-<sha256-prefix>.ll
  application_inputs/              # optional: producer --dump-inputs only
    manifest.json
    color_<frame>.bgra
    ui_<frame>.bgra
    backbuffer_<frame>.bgra
    depth_<frame>.r32f
    mv_<frame>.rg16f
```

- `trace/meta.json` and `trace/trace.jsonl` are required.
- `trace/tensors/` is required for a frame that will be exported.
- `trace/disasm/` is required by the replay and graph builders that inspect the
  captured programs.
- `trace/shaders/` and `trace/params/` are producer side files. They are part of
  the retained capture, but export decisions use the trace rows and tensor files.
- `application_inputs/` is an optional provenance dump written with
  `--dump-inputs`. It is not consumed by package export and is not the examples'
  sequence manifest; see section 8.
- `graph.json`, `layers.json`, `ops.json`, and other offline products may exist
  beside the required files. The exporter ignores them.
- Resource `file` names resolve under `trace/tensors/`; shader and parameter
  files live under `trace/shaders/` and `trace/params/`, respectively.

The exact file-name rules are:

| File | Rule |
| --- | --- |
| Tensor dump | `d%04u_<slot>_<in\|out>.bin`, for example `d0025_t2_in.bin`, `d0038_u0_out.bin`, or `d0025_b1_in.bin`. |
| Disassembly | `<key>-<first 16 hex characters of sha256>.ll`. |
| Shader container | `<key>-<first 16 hex characters of sha256>.dxil`. |
| Layer parameter span | `<key>-weights.bin`, `<key>-bias.bin`, `<key>-activation.bin`. |

`<slot>` is the shader register name resolved by the producer (`tN`, `uN`, or a
constant-buffer window such as `bN`). `in` dumps are taken before the dispatch;
`out` dumps are taken after it. The `d25` capture also stores `b1` to `b4`
constant-buffer windows.

## 2. `trace/meta.json`

`meta.json` is one JSON object. The exporter uses:

| Field | Use |
| --- | --- |
| `configuration.width`, `configuration.height` | Must match the captured output colour size. |
| `configuration.motion_width`, `configuration.motion_height` | Must match the captured motion/depth grid. |
| `configuration.static` | Selects static or motion calibration for the `d25` bound. Missing means static. |
| `configuration.flip_mv` | Must be absent or `false`. |
| `xefg_family` | Provenance and compatibility signal. The profiled family is `fg78`. |
| `xefg_dll_sha256`, `adapter`, `vendor_id`, `device_id` | Provenance only. |

The producer also records `scene`, `scene_revision`, `ui`, `ui_mode`,
`ui_mode_requested`, `ui_mode_explicit`, `ui_texture_tagged`,
`initialization_fallback`, `animate_ui`, `dump_inputs`, `application_inputs`,
`max_interpolated_frames`, `frames`, `warmup`, `pacing_ms`,
`only_interpolation`, `velocity`, and `tensors` under `configuration`, plus
counts such as `dispatches`, `tensor_files`, `tensor_bytes`, `provider_queues`,
`tensor_wait`, and `trace_error`. The exporter does not interpret the UI flag or
these counts when selecting a variant; the `d38` composite plan does that. A
`frames` mismatch is a provenance warning, not a graph input. `ui_texture_tagged`
is false for mode 4 because that mode does not bind a separate UI texture.

`meta.json` is not a substitute for the trace. Dispatch rows and tensor files are
still read directly.

## 3. `trace/trace.jsonl`

`trace.jsonl` is line-delimited JSON. Each line has a global `seq` plus a `type`.
The producer emits these event types:

| `type` | Important fields |
| --- | --- |
| `program` | `key`, `sha256`, `bytes`, `file` |
| `pipeline` | `pso`, `sha256`, `bytes`, `root_signature` |
| `root_signature` | `root_signature`, `sha256`, `version`, `parameters`, `static_samplers` |
| `layer_parameters` | `key`, `parameters`, span sizes/hashes/files, quantisation fields |
| `dispatch` | `dispatch`, `frame`, `present`, `pso`, `sha256`, `key`, `root_signature`, `group`, `constants`, `resources` |

The exporter consumes the `dispatch` rows and the `layer_parameters` rows. The
other events are producer provenance and are not required to reconstruct the
graph.

### 3.1 `dispatch`

A dispatch row has these fields:

```json
{
  "type": "dispatch",
  "dispatch": 25,
  "frame": 1,
  "present": 1,
  "pso": "0x...",
  "sha256": "<program sha256>",
  "key": "d25",
  "root_signature": "0x...",
  "group": [40, 23, 1],
  "constants": [
    { "parameter": 0, "values": [0, 0, 0, 0] }
  ],
  "resources": [
    {
      "parameter": 0,
      "kind": "srv",
      "slot": "t2",
      "index": 0,
      "resource": "0x...",
      "format": 87,
      "dimension": "texture2darray",
      "width": 640,
      "height": 360,
      "array": 1,
      "mips": 1,
      "first_mip": 0,
      "first_slice": 0,
      "bytes": 0,
      "buffer_offset": 0,
      "file": "d0025_t2_in.bin"
    }
  ]
}
```

Consumed dispatch fields:

- `dispatch`: global capture dispatch id. It is also the number in the tensor
  file name.
- `frame`: application frame the row belongs to. A retained frame contains
  dispatches `1..38`; the next frame continues at `39..76`.
- `sha256` and `key`: locate the disassembly as
  `trace/disasm/<key>-<sha256[:16]>.ll`.
- `group`: dispatch group count `[x, y, z]`; combined with the program's thread
  count to derive the written extent.
- `constants`: root 32-bit constants, as one or more
  `{"parameter": N, "values": [...]}` entries. `recon.py` folds these into a
  16-value constant array.
- `resources`: descriptor bindings. The fields used are:
  - `kind` (`srv` or `uav`) for producer/consumer direction;
  - `slot` for the register name;
  - `resource` for resource identity. This pointer value joins a consumer SRV to
    a producer UAV;
  - `format` for the producer's numeric DXGI format code;
  - `dimension`, `width`, `height`, `array`, `first_slice`, and
    `buffer_offset` for view shape and slicing;
  - `file` for the tensor snapshot relative to `trace/tensors/`.

`present`, `index`, `mips`, `first_mip`, `bytes`, and `parameter` are carried by
the producer but are not used to decide the graph topology. They remain useful
provenance.

### 3.2 `layer_parameters`

A `layer_parameters` row records the pipeline-builder parameter vector:

```json
{
  "type": "layer_parameters",
  "key": "d4",
  "parameters": [["name", "value"], ["name", "value"]],
  "weights_file": "d4-weights.bin",
  "bias_file": "d4-bias.bin",
  "activation_file": "d4-activation.bin",
  "positive_scale": 0,
  "positive_shift": 0,
  "negative_shift": 0
}
```

`key` and `parameters` are read into a per-program map. The optional
`weights_*`, `bias_*`, and `activation_*` files, `positive_scale`,
`positive_shift`, `negative_shift`, and `output_channels` are producer side
metadata for the pipeline builder. They are not runtime model inputs.

## 4. Frames and dispatch normalization

The capture stores absolute dispatch ids that continue across frames:

| Application frame | Capture dispatch ids | Normalized node ids |
| --- | --- | --- |
| 1 | 1..38 | 1..38 |
| 2 | 39..76 | 1..38 |
| 3 | 77..114 | 1..38 |

`xefg-export --frame N` selects one retained application frame, sorts its rows by
capture id, and remaps the ids to frame-relative nodes `1..N` (38 for the
profiled graph). This normalization is only a planning view. Tensor reads,
tensor file names, disassembly names, and provenance keep the original capture
dispatch id.

The graph is expected to contain 38 dispatches per frame. A frame with missing
rows, missing tensor snapshots, or incomplete disassembly is rejected rather
than partially exported. `--tensor-dispatches 76` is the usual way to retain
both complete frame 1 and frame 2 snapshots.

The final composite is node `d38`:

- its variant mode is read from the captured composite plan (the `d38` constant
  `c[14]`), not from the `meta.json` UI flag;
- mode 1 folds previous colour `t11` and current colour `t12` into the graph
  that consumes the captured motion/depth-derived resources;
- mode 3 additionally consumes UI `t9`, which must be present;
- mode 4 uses the hudless previous/current pair (`t11`/`t12`) and the current
  backbuffer `t10`; `t8` is the current hudless colour and aliases
  `current_color`. `t9` is a bound placeholder and is not a separate UI input.
  No UI texture is required for mode 4;
- the composite's `u0` output is the final generated frame.

Modes 2 and 5 are rejected even if their resource population looks similar.
Mode 4 is an explicit capture/export path; it is not inferred from `AUTO`.

## 5. Programs and family classification

The assembler uses the reconstructed 38-dispatch topology. Captured resource
identity establishes aliases and producer/consumer relationships: consumer SRVs
and producer UAVs can refer to the same `resource` pointer. This is not an
arbitrary-shader graph compiler. Each dispatch is classified into one of the
reconstructed families from its captured shape and program:

- `conv`
- `flow conv`
- `upsample`
- `GEMV`
- `buffer output`
- `resample`
- `downsample`
- `warp`
- `splat`
- `final composite`
- `warp + flow init` (`d25`)

The supported capture family is `fg78`; the slot map and reconstruction templates
were derived for that family. The `xefg_family` metadata field is recorded as
provenance, not enforced as a separate family whitelist. Other families are not
supported or validated even if their metadata can be read.

## 6. `d25` is bounded, not byte-exact

`d25` (warp + flow init) includes texture sampling and has small residuals
against the inspected hardware captures. It is the individually non-byte-exact
stage; its errors can propagate through later flow stages and produce larger
final-image differences. Matching downstream dispatches when fed captured
inputs does not establish an exact chained graph.

The inherited research helpers contain static/motion calibrations and structural
checks. Those checks are not universal accuracy guarantees, and successful model
export does not certify provider equality. The SDK does not add scene-specific
runtime error thresholds. See [Validation](VALIDATION.md) for the actual scope of
numerical evidence. Mode 4 has its own final-composite branch; its 640×360
capture and `d38` mapping are checked, and the full-chained comparison retains a
small provider residual rather than equality.

## 7. Export-time limits

The exporter checks the following while reading a revision 1 capture:

- output and motion sizes must be positive and must match
  `meta.json` when those fields are present;
- `flip_mv` must be false;
- mode must be 1, 3, or 4; mode 3 must have a UI `t9` input; mode 4 must have
  the hudless `t8` alias and the current backbuffer `t10`, and ignores `t9`;
- colour is `uint8 (H, W, 4)`, motion is `float16 (h, w, 2)` and equals its
  float32 alias, depth is `float32 (h, w)` and equals its flat alias;
- the gate LUT has 1024 `float32` values;
- the full-rect configuration, MV scale `4.0`, interpolation ratio `1.0`, and
  baked splat scale `0.5` are the supported profile values;
- every remaining non-image graph input must be a small planner scalar or pad
  (at most 16 values at the export boundary, `int64`/`float32`/`int32`). The
  exporter rejects unknown external tensors instead of zero-filling them;
- inspected profiles carry 307 planner feeds, each with at most 8 values
  (285 `int64` and 22 `float32`), in addition to the 4 or 5 public application
  inputs. The serialized public graph therefore has 311 inputs for mode 1 and
  312 for mode 3. The mode-4 public model also has five public application
  inputs, 307 planner feeds, and 312 serialized inputs across all five sizes.
  The debug graph has 42 raw outputs. The deployable graph has one output,
  `generated_frame`.

These are export checks over retained data. They do not impose engine camera,
jitter, or motion-vector-scale requirements on an application.

## 8. `application_inputs/` provenance dump

`run.ps1 -DumpInputs` (native `--dump-inputs`) writes raw application inputs
beside the trace. The dump covers **all produced application frames, including
warmup**, not only the traced presentation frames:

```text
application_inputs/
  manifest.json
  color_00000.bgra
  ui_00000.bgra
  backbuffer_00000.bgra
  depth_00000.r32f
  mv_00000.rg16f
```

The producer writes one `color` (hudless colour), `ui`, `backbuffer`, `depth`,
and `mv` file per frame. `backbuffer` is the hudless colour with the
premultiplied UI already composited; it is the current backbuffer used by mode 4.
The index is the application frame index including warmup.

`manifest.json` declares schema `xefg.application_inputs/1` and records:

| Field | Meaning |
| --- | --- |
| `width`, `height` | Output colour dimensions. |
| `motion_width`, `motion_height` | Half-resolution motion/depth grid. |
| `frame_count`, `first_frame`, `last_frame` | Dumped range; currently `0..frame_count-1`. |
| `includes_warmup` | `true`; the dump is not limited to traced frames. |
| `warmup`, `traced_frames` | The `-Warmup` / `-Frames` values used for the run. |
| `formats` | `color`/`ui`/`backbuffer` are `BGRA8`; `depth` is `R32_FLOAT`; `motion` is `R16G16_FLOAT`. |
| `frames` | Per-frame `index` and relative `inputs` file names. |

All files are raw, tightly packed, little-endian on Windows x64, and relative to
`application_inputs/`. The manifest is a provenance description of the producer's
scene, **not** the examples' sequence manifest (`schema_version: 1` with
`frames[].path` or `frames[].fields`). Converting a dump for `examples/pair.py`
or `examples/sequence.py` requires building a separate sequence manifest and
mapping these files to the example field names.

The package exporter does not consume `application_inputs/`; the separate
input-provenance report can read it. It is useful for independent
application-input inference, provenance, and re-checking outputs. Copy it with
the rest of the run as read-only evidence; see section 9.

## 9. Copy captures unchanged

Copy an entire run directory as a unit. Do not:

- rename or renumber dispatch ids or `frame` values;
- rename tensor files or change `trace/` subdirectory names;
- rewrite `trace.jsonl`, `meta.json`, shaders, params, or disassembly;
- merge rows from different runs into one trace;
- copy only `tensors/` without the matching trace rows and disassembly.

The exporter joins rows by `frame`, `dispatch`, `slot`, `resource`, and exact
file basenames. Reformatting JSON, changing line endings, or rebuilding an
archive in a way that alters file bytes can break resource joins or the recorded
provenance hashes. Treat a retained capture as read-only evidence.

Extraction, if needed, should preserve the directory tree and binary file
contents exactly. The exporter writes only to the `--out` package directory; the
output directory must be absent or empty.
