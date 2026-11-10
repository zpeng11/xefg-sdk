"""Small shared helpers for the XeFG 2x SDK examples.

The examples load this file by path from the standalone ``examples/`` directory,
so ``python examples/pair.py`` works from an installed ``xefg_sdk`` package
without adding repository paths to ``sys.path``. Input is a JSON sequence
manifest plus one ``.npz`` per frame, or explicit ``.npy``/raw fields. No trace
is read.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FIELDS = ("backbuffer_color", "hudless_color", "motion_vectors", "depth", "ui")
# NumPy dtype and channel count for the formats the runtime supports.
FORMATS = {
    "BGRA8_UNORM": ("uint8", 4),
    "RGBA8_UNORM": ("uint8", 4),
    "RG16_FLOAT": ("float16", 2),
    "RG32_FLOAT": ("float32", 2),
    "R32_FLOAT": ("float32", None),
}
UI_MODES = (
    "AUTO", "NONE", "BACKBUFFER_UITEXTURE", "HUDLESS_UITEXTURE",
    "BACKBUFFER_HUDLESS", "BACKBUFFER_HUDLESS_UITEXTURE",
)


def add_arguments(parser):
    parser.add_argument("--manifest", required=True,
                        help="application-input JSON sequence manifest")
    parser.add_argument("--package", default=None,
                        help="model package directory (overrides manifest 'package')")
    parser.add_argument("--output", default="xefg_output",
                        help="directory for generated .npy files")
    parser.add_argument("--png", action="store_true",
                        help="also write an RGBA PNG (standard library only)")
    parser.add_argument("--ui-mode", choices=UI_MODES, default=None,
                        help="override config.ui_mode")
    parser.add_argument("--profile", default=None,
                        help="override config.profile_id")
    parser.add_argument("--limit", type=int, default=None,
                        help="read at most this many frames")


def open_sequence(args):
    """Return (defaults, frame entries, package path) for a manifest."""
    from xefg_sdk import Config, UiMode

    path = Path(args.manifest).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    frames = raw.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest 'frames' must be a non-empty array")

    config = dict(raw.get("config") or {})
    mode = args.ui_mode if args.ui_mode is not None else config.get("ui_mode", "AUTO")
    config["ui_mode"] = UiMode[mode] if isinstance(mode, str) else UiMode(mode)
    if args.profile is not None:
        config["profile_id"] = args.profile

    if "constants" in raw:
        raise ValueError("SDK 0.2 takes normalized inputs, not engine constants; "
                         "put reset_history on individual frame entries")
    defaults = SimpleNamespace(config=Config(**config), directory=path.parent)

    entries = []
    for item in frames:
        if isinstance(item, str):
            item = {"path": item}
        entry = dict(item)
        if ("path" in entry) == ("fields" in entry):
            raise ValueError("each frame needs exactly one of 'path' or 'fields'")
        if "path" in entry:
            entry["path"] = _resolve(path.parent, entry["path"])
        entries.append(entry)

    package = args.package or raw.get("package")
    if not package:
        raise ValueError("no model package: pass --package or set 'package' in the manifest")
    if not args.package:
        package = _resolve(path.parent, package)
    return defaults, entries, Path(package).resolve()


def load_frame(entry, defaults):
    """Build one FrameInputs from an application frame entry."""
    from xefg_sdk import FrameInputs, ImageView

    if "constants" in entry:
        raise ValueError("engine constants belong to the adapter; use frame.reset_history")
    values = {}
    frame_id = entry.get("frame_id")
    reset_history = entry.get("reset_history")
    if "path" in entry:
        archive = np.load(entry["path"], allow_pickle=False)
        if not hasattr(archive, "files"):
            raise ValueError("'path' must name an .npz archive")
        with archive:
            if "frame_id" in archive.files:
                archive_id = int(np.asarray(archive["frame_id"]).reshape(()))
                if frame_id is not None and frame_id != archive_id:
                    raise ValueError("JSON and .npz frame_id differ in %s" % entry["path"])
                frame_id = archive_id
            if "reset_history" in archive.files and reset_history is None:
                reset_history = bool(np.asarray(archive["reset_history"]).reshape(()))
            formats = entry.get("formats") or {}
            for name in FIELDS:
                if name not in archive.files:
                    continue
                array = np.array(archive[name], copy=True)
                fmt = formats.get(name) or _infer(name, array)
                values[name] = ImageView(_checked(array, fmt), fmt)
    else:
        if frame_id is None:
            raise ValueError("frame_id is required with explicit 'fields'")
        for name, field in entry["fields"].items():
            path = _resolve(defaults.directory, field["file"])
            fmt = field["format"]
            values[name] = ImageView(_checked(_read_field(path, field), fmt), fmt)

    if frame_id is None:
        raise ValueError(".npz frame is missing frame_id")

    return FrameInputs(frame_id=int(frame_id), reset_history=bool(reset_history), **values)


def save_result(result, directory, stem, png=False):
    """Write generated_frame as .npy, and optionally as an RGBA PNG."""
    if result.status != "generated" or result.generated_frame is None:
        return []
    image = np.asarray(result.generated_frame.data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npy_path = directory / (stem + ".npy")
    np.save(npy_path, image)
    written = [npy_path]
    if png:
        png_path = directory / (stem + ".png")
        png_path.write_bytes(_png(image))
        written.append(png_path)
    return written


def result_stem(result):
    return "xefg_%s_%06d" % (
        "none" if result.previous_frame_id is None else "%06d" % result.previous_frame_id,
        result.current_frame_id)


def _read_field(path, field):
    dtype, channels = FORMATS[field["format"]]
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    width, height = int(field["width"]), int(field["height"])
    shape = (height, width) if channels is None else (height, width, channels)
    count = int(np.prod(shape))
    array = np.fromfile(path, dtype=np.dtype(dtype))
    if array.size != count:
        raise ValueError("%s has %d elements, expected %d" % (path, array.size, count))
    return array.reshape(shape)


def _infer(name, array):
    if name in ("backbuffer_color", "hudless_color", "ui"):
        if array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 4:
            return "BGRA8_UNORM"
    elif name == "motion_vectors" and array.ndim == 3 and array.shape[2] == 2:
        if array.dtype == np.float16:
            return "RG16_FLOAT"
        if array.dtype == np.float32:
            return "RG32_FLOAT"
    elif name == "depth":
        if array.dtype == np.float32 and (
                array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 1)):
            return "R32_FLOAT"
    raise ValueError("add a 'formats' entry for %s %s %s"
                     % (name, array.dtype, array.shape))


def _checked(array, fmt):
    if fmt not in FORMATS:
        raise ValueError("unsupported format %s" % fmt)
    dtype, channels = FORMATS[fmt]
    if array.dtype != np.dtype(dtype):
        raise ValueError("%s requires %s, got %s" % (fmt, np.dtype(dtype), array.dtype))
    if channels is None:
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError("%s requires HW data" % fmt)
    elif array.ndim != 3 or array.shape[2] != channels:
        raise ValueError("%s requires HWC data with %d channels" % (fmt, channels))
    return np.ascontiguousarray(array)


def _png(image):
    """Encode a BGRA8 HWC image as an RGBA PNG using zlib and struct."""
    height, width, _ = image.shape
    rgba = image[..., [2, 1, 0, 3]]
    rows = b"".join(b"\x00" + rgba[y].tobytes() for y in range(height))

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def _resolve(base, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(base) / path).resolve()
