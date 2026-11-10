#!/usr/bin/env python3
"""Interpolate one application frame pair with the XeFG 2x SDK.

The input is a JSON manifest of ordinary application frames; see
``docs/xefg_sdk/IO_CONTRACT.md``. This example runs a single previous/current
pair through ``Xefg2xRuntime.interpolate_pair`` and writes the generated frame.

    python examples/pair.py --manifest sequence.json --output out --png

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
        description="Interpolate one application frame pair with the XeFG 2x SDK.",
        epilog=("Reads the application-input manifest only. No capture trace, "
                "dataset, debug tensor list, or original swapchain is used."),
    )
    common.add_arguments(parser)
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="zero-based index of the previous frame (default: 0)",
    )
    args = parser.parse_args(argv)

    defaults, entries, package = common.open_sequence(args)
    if args.limit is not None:
        if args.limit < 2:
            parser.error("--limit must be at least 2 for a pair")
        entries = entries[:args.limit]
    index = args.pair_index
    if index < 0 or index + 1 >= len(entries):
        parser.error(
            "pair index %d needs frames %d and %d; manifest has %d frames"
            % (index, index, index + 1, len(entries)))

    previous = common.load_frame(entries[index], defaults)
    current = common.load_frame(entries[index + 1], defaults)

    from xefg_sdk import Xefg2xRuntime

    with Xefg2xRuntime.load(package) as runtime:
        result = runtime.interpolate_pair(previous, current, defaults.config)

    written = common.save_result(
        result, args.output, common.result_stem(result), args.png)
    print("%s -> %s: %s (interpolation_fraction %.3f)"
          % (result.previous_frame_id, result.current_frame_id, result.status,
             result.interpolation_fraction))
    for path in written:
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
