"""Probe the official d24 splat destination-coordinate rounding rule.

Independent diagnostic: capture real XeFG d24 tensors with a controlled constant
velocity, then replay only the coordinate selection with candidate rules and
compare the whole captured u0 allocation byte-by-byte.  Production export code
and the model package are never modified.

The 640x360 mode 3 profile stores motion at 320x180, so the probe's
"object displacement per frame in output pixels" V becomes a motion-vector value
of V * 320/640 = V/2, and d24 applies its captured scale of 0.5:

    disp = -V / 4  (current-to-previous motion)

The same-size d24 branch (src == dst == 320x180) computes
``tid + Round_ni(disp + 0.5)``; the rescaled branch computes
``Round_ne((tid + 0.5 + disp) * dst / src)`` (multiply before divide, no -0.5).

Candidates applied to the coordinate argument d:
  ceil                 old SDK reconstruction (ceil(d))
  floor_d_plus_half    floor(d + 0.5)   (round half toward +inf)
  rint                 rint(d)          (ties to even)
  rint_d_plus_half     rint(d + 0.5)    (DXIL Round_ne on d+0.5)
  round_half_away      round-half-away-from-zero(d + 0.5) (DXIL Round_ni guess)

The first three are the requested comparison set; the last two only disambiguate
the tie rule when a capture lands exactly on one.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from xefg_sdk.export import recon

DEFAULT_DISPS = [0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.5, -1.5, 1.0, -1.0]
SPLAT_OUTPUT_SCALE = 0.5  # profile baked splat scale


def signed_tag(value):
    text = ("m" if value < 0 else "p") + ("%.6g" % abs(value))
    return text.replace(".", "p").replace("+", "")


def find_dxc(explicit):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit("DXC not found: %s" % path)
        return str(path)
    found = []
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if not root:
            continue
        base = Path(root) / "Windows Kits" / "10" / "bin"
        if base.is_dir():
            found.extend(base.glob("*/x64/dxc.exe"))
    if found:
        return str(sorted(found)[-1])
    for name in ("dxc.exe", "dxc"):
        try:
            result = subprocess.run(["where" if os.name == "nt" else "which", name],
                                    capture_output=True, text=True)
        except OSError:
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    return None


def ensure_disasm(run_dir, dxc, log):
    """Reuse an existing .ll for the same shader, otherwise run DXC -dumpbin."""
    shaders = sorted((run_dir / "trace" / "shaders").glob("*.dxil"))
    target_dir = run_dir / "trace" / "disasm"
    target_dir.mkdir(parents=True, exist_ok=True)
    generated = reused = 0
    for shader in shaders:
        target = target_dir / (shader.stem + ".ll")
        if target.is_file() and target.stat().st_size > 0:
            reused += 1
            continue
        if not dxc:
            raise SystemExit("DXC is required for trace/disasm but was not found")
        result = subprocess.run([dxc, "-dumpbin", str(shader)],
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise SystemExit("DXC failed for %s: %s" % (shader.name, result.stderr.strip()))
        text = result.stdout
        if "target triple" not in text and "\ndefine " not in text:
            raise SystemExit("DXC output for %s is not LLVM/DXIL" % shader.name)
        target.write_text(text.rstrip("\r\n") + "\n", encoding="utf-8")
        generated += 1
    log.append({"shaders": len(shaders), "generated": generated, "reused": reused})


def capture_one(probe, xess_bin, run_dir, velocity, args):
    capture_dir = run_dir / "capture"
    if capture_dir.exists():
        raise SystemExit("output already exists: %s" % capture_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        probe, "--xess-dir", xess_bin, "--out", str(capture_dir),
        "--width", "640", "--height", "360",
        "--mv-width", "320", "--mv-height", "180",
        "--frames", str(args.frames), "--warmup", str(args.warmup),
        "--ui", "on", "--ui-mode", "hudless-uitexture",
        "--velocity", "%g,0" % velocity,
        "--tensors", "--tensor-dispatches", str(args.tensor_dispatches),
        "--tensor-budget-mb", str(args.tensor_budget_mb),
        "--no-model-dump",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    (run_dir / "capture.log").write_text(
        " ".join(cmd) + "\n\n" + (result.stdout or "") + (result.stderr or ""),
        encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit("probe failed (%d), see %s" % (
            result.returncode, run_dir / "capture.log"))
    return capture_dir


def frame_rows(run_dir):
    """Capture rows grouped by application frame, keeping capture order."""
    rows = recon.read_events(run_dir)
    frames = {}
    for row in rows:
        if row.get("type") == "dispatch":
            frames.setdefault(row.get("frame"), []).append(row)
    for frame in frames:
        frames[frame].sort(key=lambda r: r["dispatch"])
    return frames


def coordinate_candidates(plan):
    """Candidate destination coordinates for the captured splat plan."""
    src_w, src_h = plan["src_w"], plan["src_h"]
    dst_w, dst_h = plan["dst_w"], plan["dst_h"]
    disp_x, disp_y = plan["disp_x"], plan["disp_y"]
    same_size = src_w == dst_w and src_h == dst_h
    if same_size:
        tid_x = np.arange(src_w, dtype=np.float32)[None, :]
        tid_y = np.arange(src_h, dtype=np.float32)[:, None]
        base_x = np.broadcast_to(tid_x, disp_x.shape)
        base_y = np.broadcast_to(tid_y, disp_y.shape)
        expr_x, expr_y = disp_x, disp_y
        offset_x = base_x
        offset_y = base_y
    else:
        tid_x = np.arange(src_w, dtype=np.float32)[None, :]
        tid_y = np.arange(src_h, dtype=np.float32)[:, None]
        expr_x = (tid_x + np.float32(0.5) + disp_x) * np.float32(dst_w) / np.float32(src_w)
        expr_y = (tid_y + np.float32(0.5) + disp_y) * np.float32(dst_h) / np.float32(src_h)
        offset_x = np.float32(0.0)
        offset_y = np.float32(0.0)

    def half_away(values):
        return np.copysign(np.floor(np.abs(values) + 0.5), values)

    rules = {
        "ceil": lambda d: np.ceil(d),
        "floor_d_plus_half": lambda d: np.floor(d + np.float32(0.5)),
        "rint": lambda d: np.rint(d),
        "rint_d_plus_half": lambda d: np.rint(d + np.float32(0.5)),
        "round_half_away": lambda d: half_away(d + np.float32(0.5)),
    }
    candidates = {}
    for name, rule in rules.items():
        if same_size:
            dx = np.broadcast_to(tid_x, expr_x.shape) + rule(expr_x)
            dy = np.broadcast_to(tid_y, expr_y.shape) + rule(expr_y)
        else:
            dx = rule(expr_x)
            dy = rule(expr_y)
        candidates[name] = (dx.astype(np.int64), dy.astype(np.int64))
    if not same_size:
        candidates["legacy_ceil_expr_minus_half"] = (
            np.ceil(expr_x - np.float32(0.5)).astype(np.int64),
            np.ceil(expr_y - np.float32(0.5)).astype(np.int64))
    return same_size, candidates, expr_x


def scatter(plan, dst_x, dst_y):
    """Global atomic-max scatter, identical to recon.reconstruct_splat."""
    dst_w, dst_h = plan["dst_w"], plan["dst_h"]
    plane = dst_w * dst_h
    got = np.zeros(plan["allocation_words"], dtype=np.uint32)
    for ox in (0, 1):
        for oy in (0, 1):
            xx = dst_x + ox
            yy = dst_y + oy
            ok = (plan["finite"] & (xx >= 0) & (xx < dst_w)
                  & (yy >= 0) & (yy < dst_h))
            np.maximum.at(got, yy[ok] * dst_w + xx[ok], plan["key_x"][ok])
            np.maximum.at(got, plane + yy[ok] * dst_w + xx[ok], plan["key_y"][ok])
    return got.view(np.uint8)


def analyze_frame(run_dir, frame, row, velocity, disp_target):
    if not recon.splat_shaped(row):
        return {"frame": frame, "capture_dispatch": row.get("dispatch"),
                "error": "dispatch is not the d24 splat shape"}
    plan, error = recon.splat_plan(run_dir, row)
    if error:
        return {"frame": frame, "capture_dispatch": row.get("dispatch"),
                "error": error}

    same_size, candidates, expr_x = coordinate_candidates(plan)
    expected = plan["expected"]
    disp_x = plan["disp_x"]
    finite = plan["finite"]
    nonzero = disp_x[finite & (np.abs(disp_x) > 0)]
    unique_disp = sorted({float(np.float32(v)) for v in nonzero})
    scales = (float(plan["scale_x"]), float(plan["scale_y"]))
    measured = (unique_disp[0] if len(unique_disp) == 1 else None)

    reports = []
    base_x = candidates["floor_d_plus_half"][0]
    for name, (dx, dy) in candidates.items():
        got = scatter(plan, dx, dy)
        mismatch_bytes = int(np.count_nonzero(got != expected))
        mismatch_words = int(np.count_nonzero(
            got.view(np.uint32) != expected.view(np.uint32)))
        reports.append({
            "rule": name,
            "mismatch_bytes": mismatch_bytes,
            "mismatch_words": mismatch_words,
            "total_bytes": int(expected.size),
            "coord_diff_x_vs_floor": int(np.count_nonzero(dx != base_x)),
        })

    # Tie bookkeeping on the same-size Round_ni(disp + 0.5) expression.
    shifted = np.float32(disp_x + np.float32(0.5))
    tie_shifted = int(np.count_nonzero(
        (np.floor(shifted) + np.float32(0.5) == shifted) & finite))
    tie_direct = int(np.count_nonzero(
        (np.floor(disp_x) + np.float32(0.5) == disp_x) & finite & (disp_x != 0)))

    return {
        "frame": frame,
        "capture_dispatch": row.get("dispatch"),
        "shader_key": row.get("key"),
        "path": "same_size" if same_size else "rescaled",
        "source": [plan["src_w"], plan["src_h"]],
        "destination": [plan["dst_w"], plan["dst_h"]],
        "scale": list(scales),
        "velocity": velocity,
        "disp_target": disp_target,
        "disp_values": unique_disp[:16],
        "disp_unique_count": len(unique_disp),
        "disp_matches_target": (None if disp_target is None else
                                measured is not None
                                and abs(measured - disp_target) < 1e-6),
        "tie_count_shifted_half_integer": tie_shifted,
        "tie_count_integer_disp": tie_direct,
        "allocation_words": plan["allocation_words"],
        "candidates": reports,
        "exact_rules": [r["rule"] for r in reports if r["mismatch_bytes"] == 0],
    }


def analyze_run(run_dir, velocity, disp_target):
    capture = run_dir / "capture" if (run_dir / "capture" / "trace").is_dir() else run_dir
    frames = frame_rows(capture)
    results = []
    skipped = []
    for frame in sorted(frames):
        dispatches = frames[frame]
        if len(dispatches) < 24:
            continue
        row = dispatches[23]  # frame-relative d24
        slots = {r.get("slot"): r.get("file") for r in row.get("resources", [])}
        if not slots.get("u0"):
            skipped.append({"frame": frame, "dispatch": row.get("dispatch"),
                            "reason": "no tensor snapshot in this frame"})
            continue
        results.append(analyze_frame(capture, frame, row, velocity, disp_target))
    return {
        "run": str(run_dir),
        "capture": str(capture),
        "velocity": velocity,
        "frames": results,
        "skipped_frames": skipped,
        "has_d24_tensor": bool(results) and any("error" not in r for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", help="XefgProbe executable (required for capture)")
    parser.add_argument("--xess-bin", help="SDK runtime directory (required for capture)")
    parser.add_argument("--dxc", default="")
    parser.add_argument("--out", type=Path, required=True, help="new output root")
    parser.add_argument("--disp", default="",
                        help="comma-separated target displacements; default all")
    parser.add_argument("--velocity", default="",
                        help="comma-separated velocities (overrides --disp)")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--tensor-dispatches", type=int, default=76)
    parser.add_argument("--tensor-budget-mb", type=int, default=128)
    parser.add_argument("--analyze-only", action="append", default=[],
                        metavar="RUN", help="analyze an existing run; repeatable")
    parser.add_argument("--disp-of-run", type=float, default=None,
                        help="target displacement of the first --analyze-only run")
    args = parser.parse_args()

    if args.analyze_only:
        out = args.out
        out.mkdir(parents=True, exist_ok=False)
        report = {"schema": "xefg.splat_rounding/1", "created": datetime.now().isoformat(),
                  "mode": "analyze-only", "runs": []}
        for text in args.analyze_only:
            run = Path(text)
            velocity = None
            target = args.disp_of_run
            result = analyze_run(run, velocity, target)
            report["runs"].append(result)
        _write_and_summarize(out, report)
        return

    if args.velocity:
        velocities = [float(v) for v in args.velocity.replace(" ", "").split(",") if v]
        # The probe stores current->previous motion, so positive V yields -disp.
        targets = {v: -v * 0.25 for v in velocities}
    else:
        text = args.disp or ",".join("%g" % d for d in DEFAULT_DISPS)
        disps = [float(v) for v in text.replace(" ", "").split(",") if v]
        # V output-pixel velocity -> |disp| = V * (mv_w/width) * splat_scale;
        # the negative sign comes from current->previous motion convention.
        targets = {round(-d / (0.5 * SPLAT_OUTPUT_SCALE), 6): d for d in disps}

    if not args.probe or not args.xess_bin:
        parser.error("capture requires --probe and --xess-bin")
    probe = str(Path(args.probe))
    xess = str(Path(args.xess_bin))
    if not Path(probe).is_file():
        raise SystemExit("probe not found: %s" % probe)
    if not Path(xess).is_dir():
        raise SystemExit("XeSS bin not found: %s" % xess)
    dxc = find_dxc(args.dxc)

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "xefg.splat_rounding/1", "created": datetime.now().isoformat(),
              "mode": "capture", "probe": probe, "xess_bin": xess,
              "dxc": dxc, "runs": []}

    print("out=%s  dxc=%s" % (out, dxc), flush=True)
    for velocity, target in targets.items():
        tag = "disp_%s_v%s" % (signed_tag(target), "%g" % velocity)
        run_dir = out / tag
        print("[%s] capture velocity=%g disp=%+g" % (tag, velocity, target), flush=True)
        entry = {"tag": tag, "velocity": velocity, "disp_target": target,
                 "run": str(run_dir)}
        try:
            capture_one(probe, xess, run_dir, velocity, args)
            log = []
            ensure_disasm(run_dir / "capture", dxc, log)
            entry["disasm"] = log
            entry.update(analyze_run(run_dir, velocity, target))
        except SystemExit as exc:
            entry["error"] = str(exc)
            print("  ERROR: %s" % exc, flush=True)
        report["runs"].append(entry)
        for frame in entry.get("frames", []):
            if "error" in frame:
                print("  frame %s d%s ERROR %s" % (
                    frame.get("frame"), frame.get("capture_dispatch"), frame["error"]),
                    flush=True)
                continue
            summary = ", ".join("%s=%d" % (c["rule"], c["mismatch_bytes"])
                                for c in frame["candidates"])
            print("  frame %s d%s %s disp=%s exact=%s | %s" % (
                frame["frame"], frame["capture_dispatch"], frame["path"],
                frame["disp_values"], frame["exact_rules"], summary), flush=True)

    _write_and_summarize(out, report)


def _write_and_summarize(out, report):
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    exact = {}
    for run in report["runs"]:
        for frame in run.get("frames", []):
            for rule in frame.get("exact_rules", []):
                exact.setdefault(rule, 0)
                exact[rule] += 1
    report["exact_rule_counts"] = exact
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("report=%s" % path, flush=True)
    print("exact rule counts: %s" % exact, flush=True)


if __name__ == "__main__":
    main()
