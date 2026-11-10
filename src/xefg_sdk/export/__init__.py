"""ONNX export/build toolkit for captured XeFG 2x frames.

The supported command is ``xefg-export`` (or ``python -m xefg_sdk.export``).
The legacy research pilots remain importable for tests and internal reuse, but
they are not supported command-line entry points.

Heavy dependencies are loaded on first use so importing this package does not
require ONNX or ONNX Runtime.  In particular, cross-capture provenance helpers
remain usable with only the standard library.
"""
import importlib

from . import input_provenance

__all__ = [
    "Assembler",
    "ExportError",
    "build_capture",
    "build_specs",
    "export_package",
    "export_sdk",
    "input_provenance",
    "m9_dynamic",
    "main",
    "normalized_dispatches",
    "rewrite_graph",
]

# name -> (module, attribute); a None attribute exposes the module itself.
_LAZY = {
    "export_sdk": ("export_sdk", None),
    "m9_dynamic": ("m9_dynamic", None),
    "ExportError": ("export_sdk", "ExportError"),
    "build_capture": ("export_sdk", "build_capture"),
    "rewrite_graph": ("export_sdk", "rewrite_graph"),
    "export_package": ("export_sdk", "export_package"),
    "main": ("export_sdk", "main"),
    "Assembler": ("m9_dynamic", "Assembler"),
    "build_specs": ("m9_dynamic", "build_specs"),
    "normalized_dispatches": ("m9_dynamic", "normalized_dispatches"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    module = importlib.import_module("." + module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(__all__))
