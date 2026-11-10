"""Supported command-line entry point for the XeFG SDK exporter."""
import argparse
import os
import sys

def build_parser():
    parser = argparse.ArgumentParser(
        prog="xefg-export",
        description="Export the captured XeFG 2x graph as an ONNX SDK package")
    parser.add_argument("--runs", required=True,
                        help="comma-separated capture run directories")
    parser.add_argument("--out", required=True,
                        help="new output package directory")
    parser.add_argument("--frame", type=int, default=1,
                        help="retained capture frame (default 1)")
    parser.add_argument("--debug", action="store_true",
                        help="also write the 42-output debug model")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from .export_sdk import ExportError, export_package
    except ModuleNotFoundError as exc:
        if exc.name != "onnx":
            raise
        parser.error('export requires ONNX; install "xefg-sdk[export]"')
    run_dirs = [value for value in args.runs.split(",") if value]
    try:
        manifest = export_package(run_dirs, args.out, frame=args.frame,
                                  debug=args.debug)
    except ExportError as exc:
        print("%s: %s" % (parser.prog, exc), file=sys.stderr)
        return 1
    for name in sorted(manifest["variants"]):
        variant = manifest["variants"][name]
        print("%s -> %s (%d profile%s) sha256 %s"
              % (name, variant["model"], len(variant["profiles"]),
                 "" if len(variant["profiles"]) == 1 else "s",
                 variant["model_sha256"][:16]))
    print("package %s" % os.path.abspath(args.out))
    return 0
