# Native renderer source and dependencies

This target is part of the repository's MIT-licensed source. The renderer uses
Windows SDK D3D12/DXGI/D3DCompiler and small procedural indexed meshes; it does
not vendor a third-party engine, asset loader, mesh pack or texture pack.

The public integration structure was checked against Intel XeSS SDK 3.0.2's
`samples/basic_sample_frame_generation/basic_sample.cpp` and
`shader_xess_fg.hlsl` (Microsoft/Intel MIT notices on sample source), and the
SDK's frame-generation developer guide. This target's CMake, scene shaders and
renderer are local implementations, not a copy of the SDK sample distribution.
SDK headers and runtime DLLs remain external dependencies supplied through
`XESS_SDK_ROOT` and `--xess-dir`. Their license is separate from this repository.

Shared local components:
- `tools/capture/include/common.h`: window/device/queue fixture.
- `tools/capture/src/swapchain_capture.h`: native swapchain readback wrapper.
- `tools/capture/src/xefg_runtime.cpp`: DLL/API loading and file hash utility.

The executable does not call the provider-family inspection routine, decoder
relay, dispatch trace, model dump or exporter. A DLL hash is recorded as run
provenance; Python's debug-overlay layout profile is bound to the tested hash.

Geometry, camera motion and shader source are versioned with this checkout.
Each run records executable/shader hashes and per-frame scene transforms. No
model weights, Intel DLLs or captured output are distributed in this directory.
