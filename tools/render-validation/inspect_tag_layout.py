"""Describe exact purple regions in captured outputs; never register a pairing mask."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rectangles(mask):
    """Merge identical horizontal runs on consecutive rows into rectangles."""
    result, active = [], {}
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False].astype(np.int8)))
        spans = set(zip(edges[::2].tolist(), edges[1::2].tolist()))
        for span in list(active):
            if span not in spans:
                result.append(active.pop(span))
        for x0, x1 in sorted(spans):
            if (x0, x1) in active:
                active[(x0, x1)][3] = y + 1
            else:
                active[(x0, x1)] = [x0, y, x1, y + 1]
    result.extend(active.values())
    return sorted(result, key=lambda rect: (rect[1], rect[0]))


def inspect(run):
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    native = json.loads((run / "presents.json").read_text(encoding="utf-8"))
    config = meta["configuration"]
    width, height = config["width"], config["height"]
    rows = []
    for present in native["presents"]:
        data = (run / present["file"]).read_bytes()
        image = np.frombuffer(data, np.uint8).reshape(height, width, 4)
        purple = np.all(image[:, :, :3] == [255, 0, 255], axis=2)
        rows.append({"sequence": present["sequence"], "file": present["file"],
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "purple_pixels": int(purple.sum()), "rectangles": rectangles(purple)})
    return {"schema": "xefg.tag_layout_observation/1", "capture_directory": str(run),
            "provider": meta["xefg_dll_sha256"], "mode": config["ui_mode"],
            "output_size": [width, height], "presents": rows,
            "note": "Exact purple pixels only. Review against source content/control captures before registering a layout; this does not establish pairing."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.run.resolve())
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"mode": report["mode"], "output_size": report["output_size"],
                      "presents": report["presents"]}, indent=2))
