#!/usr/bin/env python3
"""Interpolate a sequence of application frames with the XeFG 2x SDK.

The input is a JSON manifest of ordinary application frames; see
``docs/xefg_sdk/IO_CONTRACT.md``. This example submits frames in order through
``Xefg2xSession``, which keeps the previous frame, handles ``reset_history``,
and reports the first frame as ``history_ready`` without inventing a generated
frame.

    python examples/sequence.py --manifest sequence.json --output out --png

Install the SDK first with ``python -m pip install ".[ort]"`` from the
repository root (the default executor uses ONNX Runtime). This script adds no
repository or capture-trace paths to ``sys.path``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_common():
    path = Path(__file__).resolve().with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_xefg_example_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load example helper: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def main(argv=None):
    common = _load_common()
    parser = argparse.ArgumentParser(
        description="Interpolate an application frame sequence with the XeFG 2x SDK.",
        epilog=("Reads the application-input manifest only. No capture trace, "
                "dataset, debug tensor list, or original swapchain is used."),
    )
    common.add_arguments(parser)
    args = parser.parse_args(argv)

    defaults, entries, package = common.open_sequence(args)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        entries = entries[:args.limit]

    from xefg_sdk import Xefg2xRuntime, Xefg2xSession

    written = []
    with Xefg2xRuntime.load(package) as runtime:
        session = Xefg2xSession(runtime, defaults.config)
        for entry in entries:
            frame = common.load_frame(entry, defaults)
            if frame.reset_history:
                print("frame %d: reset history" % frame.frame_id)
            result = session.submit(frame)
            if result.status == "history_ready":
                print("frame %d: history ready" % frame.frame_id)
                continue
            print("%s -> %s: %s (interpolation_fraction %.3f)"
                  % (result.previous_frame_id, result.current_frame_id,
                     result.status, result.interpolation_fraction))
            written.extend(common.save_result(
                result, args.output, common.result_stem(result), args.png))

    for path in written:
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
