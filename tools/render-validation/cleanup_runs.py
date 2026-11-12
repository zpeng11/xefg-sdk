"""Inventory old XeFG run artifacts and delete only an explicitly approved manifest.

The default ``inventory`` command is read-only.  It scans only top-level
``xefg-*`` entries on the configured run roots, top-level ``xefg*`` entries on
the user temp root, and the two named stray compiler artifacts at the repository
root.  It records exact paths, ownership evidence, and stable file identity in a
manifest below ``<current>/cleanup/``.  It never creates the current validation
root.

The ``apply`` command consumes that manifest.  Before deleting anything it
requires a complete results matrix (135 combinations / 1065 generated frames by
default) with no ``run_failed`` rows and a current model package whose hashes
match the matrix.  It then revalidates every path, reparse-point status and
identity, and deletes only entries that were classified ``delete``.  There is no
backup or recycle-bin step, and symlinks/junctions are never followed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
import uuid


MANIFEST_SCHEMA = "xefg.cleanup_manifest/1"
APPLY_SCHEMA = "xefg.cleanup_apply/1"
EXPECTED_MATRIX_ROWS = 135
EXPECTED_GENERATED_FRAMES = 1065
ALLOWED_MATRIX_STATUSES = frozenset({
    "pass", "quality_failed", "coverage_failed", "complete", "completed", "ok",
})
BAD_MATRIX_STATUSES = frozenset({
    "run_failed", "running", "pending", "incomplete", "error", "failed", "aborted",
})
REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT_ROOT = Path("E:/xefg-validation-current")
DEFAULT_TEMP_ROOT = Path(os.environ.get("TEMP") or os.environ.get("TMP") or "C:/Users/Eleven/AppData/Local/Temp")
DEFAULT_EXACT_PATHS = (REPO_ROOT / "ptrcheck.obj", REPO_ROOT / "vc140.pdb")

# Kept completely outside cleanup scope.  The two exact repository artifacts
# above are the sole exception to the repository-root protection.
DEFAULT_PROTECTED = (
    REPO_ROOT,
    REPO_ROOT / ".git",
    Path("D:/xefg-render-validation-env"),
    Path("D:/miniconda3"),
    DEFAULT_TEMP_ROOT / "xess-sdk-dl",
    Path("E:/xefg-validation-current"),
    Path("C:/Program Files/Microsoft Visual Studio"),
    Path("C:/Program Files (x86)/Microsoft Visual Studio"),
    Path("C:/Program Files/Intel"),
    Path("C:/Program Files (x86)/Intel"),
)

MARKER_NAMES = frozenset({
    "manifest.json", "matrix.json", "matrix.md", "report.json", "report.md",
    "invocation.json", "presents.json", "meta.json", "scene.json", "pairing.json",
    "input-check.json", "quality.json", "capture.log", "xefg_log.txt",
    "model_metadata.json", "onnx_report.json", "io_contract.json",
    "CMakeCache.txt", "README.md", "MANIFEST.sha256", "NOT_ARCHIVED.txt",
    "content_matrix_report.json",
})
MARKER_DIRS = frozenset({
    "capture", "replay", "pairing", "results", "matrix", "images", "logs",
    "bin", "package", "application_inputs", "debug", "models", "profiles",
    "trace", "inventory", "quality", "dump", "verify", "recon", "probe-opsonly",
})
TEXT_SUFFIXES = frozenset({
    ".json", ".log", ".md", ".txt", ".py", ".ps1", ".jsonl", ".bat", ".cmd",
    ".hlsl", ".asm",
})
BUILD_SUFFIXES = frozenset({
    ".obj", ".exe", ".dll", ".lib", ".pdb", ".ilk", ".exp", ".vcxproj",
    ".sln", ".vcproj", ".cmake", ".bat", ".cmd",
})
BUILD_NAMES = frozenset({"CMakeCache.txt", "build.bat", "build.cmd", "CMakeLists.txt"})
MAX_EVIDENCE = 12
MAX_HEAD_BYTES = 4096
MAX_HASH_EVIDENCE_BYTES = 1024 * 1024
EXACT_MAGIC = {
    "ptrcheck.obj": (b"\x64\x86", "Intel amd64 COFF object"),
    "vc140.pdb": (b"Microsoft C/C++ MSF 7.00", "MSVC program database"),
}
_SCHEMA_RE = re.compile(r'"schema"\s*:\s*"([^"]+)"')


class CleanupError(RuntimeError):
    """A safety precondition failed; no deletion should be attempted."""


@dataclass(frozen=True)
class ScopeRule:
    root: str
    pattern: str
    kind: str  # "run", "temp", or "exact"

    def to_json(self):
        return {"root": self.root, "pattern": self.pattern, "kind": self.kind}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def path_key(value):
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(value))))


def display_path(value):
    return os.path.normpath(os.path.abspath(os.fspath(value)))


def is_within(child, parent):
    child_key = path_key(child)
    parent_key = path_key(parent)
    if child_key == parent_key:
        return True
    try:
        return os.path.commonpath([child_key, parent_key]) == parent_key
    except ValueError:
        return False


def is_local_path(value):
    text = os.fspath(value)
    if not os.path.isabs(text):
        return False
    if text.startswith("\\\\") or text.startswith("//"):
        return False
    if os.name == "nt":
        drive = os.path.splitdrive(os.path.abspath(text))[0]
        if not drive:
            return False
        try:
            import ctypes

            # DRIVE_REMOTE and DRIVE_UNKNOWN are rejected.  Fixed/removable/ram
            # drives are local; no network traversal or mount discovery is done.
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")
            if drive_type in (0, 4):
                return False
        except Exception:
            # If the platform call is unavailable, the UNC check above remains
            # the hard guard for this local-only tool.
            pass
    return True


def lstat_path(value):
    st = os.lstat(value)
    attrs = getattr(st, "st_file_attributes", 0)
    reparse = stat.S_ISLNK(st.st_mode) or bool(attrs & REPARSE_ATTRIBUTE)
    return st, reparse


def sha256_file(path, limit=None):
    digest = hashlib.sha256()
    remaining = limit
    with open(path, "rb") as stream:
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            block = stream.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def read_head(path, limit=MAX_HEAD_BYTES):
    try:
        with open(path, "rb") as stream:
            return stream.read(limit)
    except OSError:
        return b""


def _schema_from_head(head):
    text = head.decode("utf-8", "replace")
    match = _SCHEMA_RE.search(text)
    return match.group(1) if match else None


def _evidence(path, relative, st=None):
    if st is None:
        st = os.lstat(path)
    base = os.path.basename(path).lower()
    suffix = os.path.splitext(base)[1]
    head = b""
    if st.st_size <= MAX_HEAD_BYTES or suffix in TEXT_SUFFIXES or base in MARKER_NAMES:
        head = read_head(path)
    text = head.decode("utf-8", "replace")
    lower_text = text.lower()
    schema = _schema_from_head(head)
    contains_project = any(
        token in lower_text for token in ("xefg", "xess", "input signature:"))
    item = {
        "relative": relative,
        "name": base,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "schema": schema,
        "contains_xefg": contains_project,
        "contains_project": contains_project,
        "marker": False,
    }
    if st.st_size <= MAX_HASH_EVIDENCE_BYTES:
        item["sha256"] = sha256_file(path)
    if (base in MARKER_NAMES or base in BUILD_NAMES or base.startswith("xefg")
            or suffix in BUILD_SUFFIXES
            or (schema or "").lower().startswith("xefg")
            or contains_project):
        item["marker"] = True
    return item


def _normalize_link_target(path, target):
    raw = target
    if target.startswith("\\\\?\\UNC\\"):
        return target, False
    if target.startswith("\\\\?\\"):
        target = target[4:]
    if target.startswith("\\\\") or target.startswith("//"):
        return target, False
    if not os.path.isabs(target):
        target = os.path.abspath(os.path.join(os.path.dirname(path), target))
    target = os.path.normpath(target)
    return target, is_local_path(target)


def _target_in_owned_run(target):
    drive, rest = os.path.splitdrive(target)
    if not drive:
        return False
    parts = [part for part in rest.replace("/", "\\").split("\\") if part]
    return any(part.lower().startswith("xefg") for part in parts)


def _reparse_info(path, st, relative, root):
    kind = ("directory" if stat.S_ISDIR(st.st_mode)
            else "file" if stat.S_ISREG(st.st_mode) else "other")
    try:
        raw_target = os.readlink(path)
    except OSError as exc:
        raw_target = None
        read_error = str(exc)
    else:
        read_error = None
    target = None
    target_local = False
    if raw_target is not None:
        target, target_local = _normalize_link_target(path, raw_target)
    try:
        is_junction = bool(os.path.isjunction(path)) if hasattr(os.path, "isjunction") else None
    except OSError:
        is_junction = None
    target_owned = bool(target) and target_local and (
        is_within(target, root) or _target_in_owned_run(target))
    return {
        "relative": relative,
        "kind": kind,
        "raw_target": raw_target,
        "target": target,
        "target_local": target_local,
        "target_inside_entry": bool(target) and target_local and is_within(target, root),
        "target_owned": target_owned,
        "is_junction": is_junction,
        "attributes": int(getattr(st, "st_file_attributes", 0)),
        "link_bytes": 0,
        "read_error": read_error,
        "link_only_safe": (kind in ("directory", "file") and raw_target is not None
                           and target_local and target_owned),
    }


def walk_tree(root):
    """Measure a tree without following links/reparse points and collect evidence."""
    root = os.fspath(root)
    total_bytes = 0
    file_count = 0
    dir_count = 0
    reparse = []
    reparse_details = []
    errors = []
    top_names = []
    evidence = []
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name.lower())
        except OSError as exc:
            errors.append(f"{directory}: {exc}")
            continue
        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append(f"{entry.path}: {exc}")
                continue
            relative = os.path.relpath(entry.path, root)
            if depth == 0:
                top_names.append(entry.name)
            attrs = getattr(st, "st_file_attributes", 0)
            if stat.S_ISLNK(st.st_mode) or (attrs & REPARSE_ATTRIBUTE):
                reparse.append(relative)
                reparse_details.append(_reparse_info(entry.path, st, relative, root))
                continue
            if stat.S_ISDIR(st.st_mode):
                dir_count += 1
                stack.append((entry.path, depth + 1))
            elif stat.S_ISREG(st.st_mode):
                file_count += 1
                total_bytes += st.st_size
                base = entry.name.lower()
                suffix = os.path.splitext(base)[1]
                if len(evidence) < MAX_EVIDENCE and (
                        base in MARKER_NAMES or base in BUILD_NAMES
                        or base.startswith("xefg")
                        or suffix in BUILD_SUFFIXES
                        or (depth <= 2 and suffix in TEXT_SUFFIXES)
                        or (depth <= 3 and suffix == ".json")):
                    evidence.append(_evidence(entry.path, relative, st))
            else:
                # Sockets, devices and other non-regular entries are not cleaned.
                errors.append(f"{entry.path}: unsupported entry type")
    return {
        "bytes": int(total_bytes),
        "files": file_count,
        "dirs": dir_count,
        "reparse": reparse,
        "reparse_details": reparse_details,
        "errors": errors,
        "top": top_names[:30],
        "evidence": evidence,
    }


def _top_file_evidence(path, st):
    item = _evidence(path, "", st)
    item["head_hex"] = read_head(path, 64).hex()
    return item


def _identity(st, kind, aggregate=None):
    identity = {
        "kind": kind,
        "st_dev": int(st.st_dev),
        "st_ino": int(st.st_ino),
        "st_mtime_ns": int(st.st_mtime_ns),
        "st_mode": int(st.st_mode),
    }
    if stat.S_ISREG(st.st_mode):
        identity["st_size"] = int(st.st_size)
    if aggregate is not None:
        identity.update({
            "bytes": aggregate["bytes"],
            "files": aggregate["files"],
            "dirs": aggregate["dirs"],
        })
    return identity


def _protected_conflict(path, protection_roots, exact_allowed):
    key = path_key(path)
    if key in exact_allowed:
        return None
    for protected in protection_roots:
        if is_within(path, protected):
            return f"inside protected path {display_path(protected)}"
        if is_within(protected, path):
            return f"contains protected path {display_path(protected)}"
    return None


def _matches_pattern(name, pattern):
    if os.name == "nt":
        return fnmatch.fnmatchcase(name.lower(), pattern.lower())
    return fnmatch.fnmatchcase(name, pattern)


def _entry_reason(marker_names):
    if not marker_names:
        return "xefg-named entry with recognised project content"
    shown = ", ".join(marker_names[:5])
    if len(marker_names) > 5:
        shown += ", ..."
    return f"recognised XeFG ownership markers: {shown}"


def classify_candidate(path, rule, protection_roots, exact_allowed):
    path = display_path(path)
    if not is_local_path(path):
        return {"path": path, "classification": "skip", "reason": "not a local path"}
    conflict = _protected_conflict(path, protection_roots, exact_allowed)
    if conflict:
        entry = {"path": path, "classification": "keep",
                 "reason": "protected: " + conflict}
        # Never require identity for a kept path, but report a useful size when
        # it can be read without following links.
        try:
            st, reparse = lstat_path(path)
            if not reparse:
                if stat.S_ISDIR(st.st_mode):
                    aggregate = walk_tree(path)
                    entry.update({"kind": "directory", "bytes": aggregate["bytes"],
                                  "files": aggregate["files"], "dirs": aggregate["dirs"]})
                elif stat.S_ISREG(st.st_mode):
                    entry.update({"kind": "file", "bytes": int(st.st_size),
                                  "files": 1, "dirs": 0})
        except OSError:
            pass
        return entry
    try:
        st, reparse = lstat_path(path)
    except OSError as exc:
        return {"path": path, "classification": "skip", "reason": f"cannot inspect: {exc}"}
    if reparse:
        return {"path": path, "classification": "skip", "reason": "reparse point/symlink; not followed"}

    if stat.S_ISREG(st.st_mode):
        exact_name = os.path.basename(path).lower()
        if rule.kind == "exact":
            magic = EXACT_MAGIC.get(exact_name)
            head = read_head(path, 64)
            if magic is None:
                return {"path": path, "classification": "skip", "reason": "unrecognised exact artifact name"}
            if not head.startswith(magic[0]):
                return {"path": path, "classification": "skip",
                        "reason": f"content does not match {magic[1]}"}
            evidence = [_top_file_evidence(path, st)]
            return {"path": path, "classification": "delete", "kind": "file",
                    "bytes": int(st.st_size), "files": 1, "dirs": 0,
                    "identity": _identity(st, "file"), "evidence": evidence,
                    "markers": [exact_name],
                    "reason": f"stray {magic[1]} at repository root"}
        evidence = [_top_file_evidence(path, st)]
        known = any(item["marker"] for item in evidence)
        if not known and not os.path.basename(path).lower().startswith("xefg"):
            return {"path": path, "classification": "skip",
                    "reason": "no XeFG ownership evidence; unknown ownership"}
        markers = [item["name"] for item in evidence if item["marker"]] or [os.path.basename(path)]
        return {"path": path, "classification": "delete", "kind": "file",
                "bytes": int(st.st_size), "files": 1, "dirs": 0,
                "identity": _identity(st, "file"), "evidence": evidence,
                "markers": markers, "reason": _entry_reason(markers)}

    if not stat.S_ISDIR(st.st_mode):
        return {"path": path, "classification": "skip", "reason": "unsupported filesystem entry"}

    aggregate = walk_tree(path)
    if aggregate["errors"]:
        return {"path": path, "classification": "skip",
                "reason": "tree is not fully readable; ownership uncertain: " + aggregate["errors"][0],
                "bytes": aggregate["bytes"], "files": aggregate["files"], "dirs": aggregate["dirs"]}
    if aggregate["files"] == 0 and aggregate["dirs"] == 0 and not aggregate["reparse"]:
        return {"path": path, "classification": "delete", "kind": "directory",
                "bytes": 0, "files": 0, "dirs": 0,
                "identity": _identity(st, "directory", aggregate), "evidence": [],
                "markers": [], "reason": "empty xefg-named directory"}

    marker_items = [item for item in aggregate["evidence"] if item["marker"]]
    markers = [item["name"] for item in marker_items] or [
        name for name in aggregate["top"] if name.lower() in MARKER_DIRS]
    known = bool(marker_items) or bool(markers)

    if aggregate["reparse"]:
        links = aggregate["reparse_details"]
        safe_links = bool(links) and all(link["link_only_safe"] for link in links)
        if known and safe_links:
            return {"path": path, "classification": "delete", "kind": "directory",
                    "bytes": aggregate["bytes"], "files": aggregate["files"],
                    "dirs": aggregate["dirs"],
                    "identity": _identity(st, "directory", aggregate),
                    "evidence": aggregate["evidence"], "markers": markers,
                    "top": aggregate["top"],
                    "reparse_policy": "delete_links_only",
                    "reparse_links": links,
                    "reason": (f"{_entry_reason(markers)}; {len(links)} local owned "
                               f"junction/symlink link(s) are removed without following "
                               f"or deleting their targets")}
        detail = "; ".join(
            f"{link['relative']} -> {link['target'] or link['raw_target']!r} "
            f"({link['kind']}, local={link['target_local']}, owned={link['target_owned']})"
            for link in links[:4])
        return {"path": path, "classification": "skip",
                "reason": ("reparse point/symlink not followed and not safely "
                           "owned/link-only: " + (detail or "unreadable link")),
                "bytes": aggregate["bytes"], "files": aggregate["files"],
                "dirs": aggregate["dirs"], "reparse_links": links}

    if not known:
        return {"path": path, "classification": "skip",
                "reason": "no XeFG ownership markers found; unknown ownership",
                "bytes": aggregate["bytes"], "files": aggregate["files"], "dirs": aggregate["dirs"]}
    return {"path": path, "classification": "delete", "kind": "directory",
            "bytes": aggregate["bytes"], "files": aggregate["files"], "dirs": aggregate["dirs"],
            "identity": _identity(st, "directory", aggregate),
            "evidence": aggregate["evidence"], "markers": markers,
            "top": aggregate["top"], "reason": _entry_reason(markers)}


def collect_candidates(scope):
    candidates = []
    warnings = []
    seen = set()
    for rule in scope:
        root = Path(rule.root)
        if rule.kind == "exact":
            path = display_path(root)
            if path_key(path) not in seen:
                seen.add(path_key(path))
                candidates.append((path, rule))
            if not os.path.exists(path):
                warnings.append(f"exact candidate is missing: {path}")
            continue
        if not is_local_path(root):
            warnings.append(f"run root is not local: {root}")
            continue
        try:
            st, reparse = lstat_path(root)
        except OSError as exc:
            warnings.append(f"cannot read run root {root}: {exc}")
            continue
        if reparse:
            warnings.append(f"run root is a reparse point; not followed: {root}")
            continue
        if not stat.S_ISDIR(st.st_mode):
            warnings.append(f"run root is not a directory: {root}")
            continue
        try:
            with os.scandir(root) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name.lower())
        except OSError as exc:
            warnings.append(f"cannot list run root {root}: {exc}")
            continue
        for entry in entries:
            if not _matches_pattern(entry.name, rule.pattern):
                continue
            path = display_path(entry.path)
            key = path_key(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((path, rule))
    return candidates, warnings


def _discover_package(current_root):
    candidates = [
        Path(current_root) / "package",
        Path(current_root) / "model-package",
        Path(current_root) / "package-v2",
    ]
    existing = []
    seen = set()
    for path in candidates:
        if path.exists():
            key = path_key(path)
            if key not in seen:
                seen.add(key)
                existing.append(path)
    return existing[0] if len(existing) == 1 else None


def build_manifest(current_root, scope, exact_paths, protection_roots,
                   expected_rows=EXPECTED_MATRIX_ROWS,
                   expected_frames=EXPECTED_GENERATED_FRAMES):
    scope_all = list(scope)
    known_exact = {path_key(rule.root) for rule in scope_all if rule.kind == "exact"}
    for path in exact_paths:
        key = path_key(path)
        if key not in known_exact:
            scope_all.append(ScopeRule(display_path(path), "", "exact"))
            known_exact.add(key)
    candidates, warnings = collect_candidates(scope_all)
    exact_allowed = {path_key(path) for path in exact_paths}
    entries = []
    for path, rule in candidates:
        entries.append(classify_candidate(path, rule, protection_roots, exact_allowed))
    entries.sort(key=lambda item: (item["classification"] != "delete",
                                   path_key(item["path"])))
    counts = Counter(item["classification"] for item in entries)
    delete_bytes = sum(int(item.get("bytes") or 0) for item in entries
                       if item["classification"] == "delete")
    keep_bytes = sum(int(item.get("bytes") or 0) for item in entries
                     if item["classification"] == "keep")
    skip_bytes = sum(int(item.get("bytes") or 0) for item in entries
                     if item["classification"] == "skip")
    current_package = _discover_package(current_root)
    return {
        "schema": MANIFEST_SCHEMA,
        "inventory_id": uuid.uuid4().hex,
        "created_utc": utc_now(),
        "current_root": display_path(current_root),
        "current_package": display_path(current_package) if current_package else None,
        "scope": [rule.to_json() for rule in scope_all],
        "exact_paths": [display_path(path) for path in exact_paths],
        "protected": [display_path(path) for path in protection_roots],
        "expected": {"matrix_rows": int(expected_rows),
                     "generated_frames": int(expected_frames)},
        "summary": {
            "candidate_count": len(entries),
            "delete_count": counts.get("delete", 0),
            "delete_bytes": delete_bytes,
            "delete_files": sum(int(item.get("files") or 0) for item in entries
                                if item["classification"] == "delete"),
            "keep_count": counts.get("keep", 0),
            "keep_bytes": keep_bytes,
            "skip_count": counts.get("skip", 0),
            "skip_bytes": skip_bytes,
        },
        "warnings": warnings,
        "entries": entries,
    }


def _write_json_exclusive(path, data, force=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if force:
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    return path


def wait_for_current(current_root, wait_seconds=0.0, poll_seconds=2.0):
    current_root = Path(current_root)
    if not is_local_path(current_root):
        raise CleanupError(f"current validation root is not a local path: {current_root}")
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while True:
        try:
            st, reparse = lstat_path(current_root)
        except OSError:
            st = None
            reparse = False
        if st is not None and stat.S_ISDIR(st.st_mode) and not reparse:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CleanupError(
                f"current validation root does not exist (not creating it): {current_root}")
        time.sleep(min(max(0.05, float(poll_seconds)), remaining))


def run_inventory(current_root, scope, exact_paths, protection_roots,
                  output=None, force=False, wait_seconds=0.0, poll_seconds=2.0,
                  expected_rows=EXPECTED_MATRIX_ROWS,
                  expected_frames=EXPECTED_GENERATED_FRAMES):
    current_root = Path(current_root)
    wait_for_current(current_root, wait_seconds, poll_seconds)
    if output is None:
        output = current_root / "cleanup" / "proposed-cleanup.json"
    output = Path(output)
    if not is_within(output, current_root / "cleanup"):
        raise CleanupError(f"manifest must be written below <current>/cleanup: {output}")
    manifest = build_manifest(current_root, scope, exact_paths, protection_roots,
                              expected_rows, expected_frames)
    # Recheck immediately before the only write.  The current root is never
    # created by this tool, even if it disappeared during inventory.
    wait_for_current(current_root, 0.0)
    manifest["manifest_path"] = display_path(output)
    _write_json_exclusive(output, manifest, force=force)
    return manifest, output


def _check_finite(value, label, problems, depth=0):
    if depth > 24 or len(problems) >= 20:
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(f"{label} is non-finite")
    elif isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{label}.{key}", problems, depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, f"{label}[{index}]", problems, depth + 1)


def validate_matrix(matrix_path, expected_rows=EXPECTED_MATRIX_ROWS,
                    expected_frames=EXPECTED_GENERATED_FRAMES):
    try:
        matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CleanupError(f"cannot read results matrix {matrix_path}: {exc}") from exc
    schema = matrix.get("schema")
    if not isinstance(schema, str):
        raise CleanupError(f"unexpected results matrix schema: {schema!r}")
    if schema.startswith("xefg.replay_matrix/"):
        kind = "replay"
        rows = matrix.get("combos")
        config = matrix.get("config") or {}
        if not isinstance(config, dict):
            raise CleanupError("results matrix config is not an object")
        if config.get("expected_combos") is not None and int(config["expected_combos"]) != int(expected_rows):
            raise CleanupError(
                f"results matrix expects {config['expected_combos']} combinations; expected {expected_rows}")
        if config.get("expected_frames") is not None and int(config["expected_frames"]) != int(expected_frames):
            raise CleanupError(
                f"results matrix expects {config['expected_frames']} frames; expected {expected_frames}")
    elif schema.startswith("xefg.render_matrix/"):
        kind = "render"
        rows = matrix.get("scenes")
        config = {}
    else:
        raise CleanupError(f"unexpected results matrix schema: {schema!r}")
    if not isinstance(rows, list):
        raise CleanupError("results matrix has no combination/scene list")
    if len(rows) != int(expected_rows):
        raise CleanupError(
            f"results matrix has {len(rows)} rows; expected {expected_rows}")
    problems = []
    statuses = Counter()
    frames = 0
    expected_frame_sum = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index} is not an object")
            continue
        status = row.get("status")
        statuses[str(status)] += 1
        status_key = status.lower() if isinstance(status, str) else None
        if status_key in BAD_MATRIX_STATUSES:
            problems.append(f"row {index} status is {status!r} (run failure/incomplete)")
        elif status_key not in ALLOWED_MATRIX_STATUSES:
            problems.append(f"row {index} has unrecognised status {status!r}")
        if kind == "render":
            if row.get("completed") is not True:
                problems.append(f"row {index} is not completed")
        elif "completed" in row and row["completed"] is not True:
            problems.append(f"row {index} is not completed")
        generated = row.get("generated_frames")
        if generated is None:
            generated = len(row.get("frames", [])) if isinstance(row.get("frames"), list) else None
        if not isinstance(generated, int) or isinstance(generated, bool) or generated < 0:
            problems.append(f"row {index} generated_frames is {generated!r}")
        else:
            frames += generated
        expected_here = row.get("expected_frames")
        if isinstance(expected_here, int) and not isinstance(expected_here, bool):
            expected_frame_sum += expected_here
        base = f"row {index}"
        for key in ("frames", "coverage", "quality"):
            if key in row:
                _check_finite(row[key], f"{base} {key}", problems)
    if problems:
        shown = "; ".join(problems[:10])
        if len(problems) > 10:
            shown += f"; ... {len(problems) - 10} more"
        raise CleanupError("results matrix is not complete: " + shown)
    if frames != int(expected_frames):
        raise CleanupError(
            f"results matrix reports {frames} generated frames; expected {expected_frames}")
    if expected_frame_sum and expected_frame_sum != int(expected_frames):
        raise CleanupError(
            f"results matrix per-combination expected_frames totals {expected_frame_sum}; "
            f"expected {expected_frames}")
    top_status = matrix.get("status")
    if isinstance(top_status, str) and top_status.lower() in BAD_MATRIX_STATUSES:
        raise CleanupError(f"results matrix top-level status is {top_status!r}")
    summary = matrix.get("summary")
    if isinstance(summary, dict):
        if summary.get("combos") is not None and int(summary["combos"]) != int(expected_rows):
            raise CleanupError(
                f"results matrix summary reports {summary['combos']} combinations; "
                f"expected {expected_rows}")
        if summary.get("frames") is not None and int(summary["frames"]) != int(expected_frames):
            raise CleanupError(
                f"results matrix summary reports {summary['frames']} frames; "
                f"expected {expected_frames}")
        summary_counts = summary.get("status_counts")
        if isinstance(summary_counts, dict) and int(summary_counts.get("run_failed") or 0) != 0:
            raise CleanupError("results matrix summary reports run_failed combinations")

    status_file = None
    status_path = Path(matrix_path).with_name("status.json")
    if status_path.exists():
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CleanupError(f"cannot read results status file {status_path}: {exc}") from exc
        status_schema = status_data.get("schema")
        if isinstance(status_schema, str) and status_schema.startswith("xefg.replay_matrix_status/"):
            if status_data.get("expected_combos") is not None and \
                    int(status_data["expected_combos"]) != int(expected_rows):
                raise CleanupError("status file expected_combos does not match")
            if status_data.get("expected_frames") is not None and \
                    int(status_data["expected_frames"]) != int(expected_frames):
                raise CleanupError("status file expected_frames does not match")
            completed_combos = status_data.get("completed_combos")
            completed_frames = status_data.get("completed_frames")
            if completed_frames is None:
                completed_frames = status_data.get("reported_frames")
            if completed_combos is not None and int(completed_combos) != int(expected_rows):
                raise CleanupError(
                    f"status file reports {completed_combos} completed combinations; "
                    f"expected {expected_rows}")
            if completed_frames is not None and int(completed_frames) != int(expected_frames):
                raise CleanupError(
                    f"status file reports {completed_frames} completed frames; "
                    f"expected {expected_frames}")
            if isinstance(status_data.get("status"), str) and \
                    status_data["status"].lower() in BAD_MATRIX_STATUSES:
                raise CleanupError(f"status file reports {status_data['status']!r}")
            counts = status_data.get("counts")
            if isinstance(counts, dict) and int(counts.get("run_failed") or 0) != 0:
                raise CleanupError("status file reports run_failed combinations")
            status_file = {
                "path": display_path(status_path),
                "schema": status_schema,
                "completed_combos": completed_combos,
                "completed_frames": completed_frames,
                "status": status_data.get("status"),
                "counts": counts,
            }
    return {
        "path": display_path(matrix_path),
        "schema": schema,
        "kind": kind,
        "rows": len(rows),
        "generated_frames": frames,
        "status_counts": dict(sorted(statuses.items())),
        "top_status": top_status,
        "status_file": status_file,
        "config": config,
    }


def _package_path(package, relative):
    if os.path.isabs(relative):
        raise CleanupError(f"model package path must be relative: {relative}")
    candidate = Path(package) / relative
    if not is_within(candidate, package):
        raise CleanupError(f"model package path escapes package: {relative}")
    resolved = Path(os.path.realpath(candidate))
    if not is_within(resolved, os.path.realpath(package)):
        raise CleanupError(f"model package path resolves outside package: {relative}")
    try:
        reparse = lstat_path(candidate)[1]
    except OSError as exc:
        raise CleanupError(f"model package path cannot be inspected: {relative}: {exc}") from exc
    if reparse:
        raise CleanupError(f"model package path is a reparse point: {relative}")
    return candidate


def validate_package(package_path, matrix):
    package = Path(package_path)
    manifest_path = package / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CleanupError(f"cannot read model package manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise CleanupError("unsupported model package schema_version")
    variants = manifest.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise CleanupError("model package contains no variants")
    model_hashes = {}
    profile_hashes = {}
    file_hashes = {}
    for variant_id, variant in variants.items():
        if not isinstance(variant, dict):
            raise CleanupError(f"variant is not an object: {variant_id}")
        model_relative = variant.get("model")
        expected_model = variant.get("model_sha256")
        if not isinstance(model_relative, str) or not isinstance(expected_model, str):
            raise CleanupError(f"variant lacks model hash identity: {variant_id}")
        model_path = _package_path(package, model_relative)
        if not model_path.is_file():
            raise CleanupError(f"model file is missing: {model_relative}")
        actual_model = sha256_file(model_path)
        if actual_model != expected_model:
            raise CleanupError(f"model hash mismatch: {variant_id}")
        model_hashes[variant_id] = actual_model
        file_hashes[str(model_relative)] = actual_model
        for relative in variant.get("profiles", []):
            if not isinstance(relative, str):
                raise CleanupError(f"non-string profile path in variant {variant_id}")
            profile_path = _package_path(package, relative)
            if not profile_path.is_file():
                raise CleanupError(f"profile file is missing: {relative}")
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise CleanupError(f"invalid profile JSON {relative}: {exc}") from exc
            if profile.get("schema_version") != 1 or profile.get("variant") != variant_id:
                raise CleanupError(f"profile schema or variant mismatch: {relative}")
            profile_hashes[str(relative)] = sha256_file(profile_path)
            file_hashes[str(relative)] = profile_hashes[str(relative)]

    manifest_sha256 = sha256_file(manifest_path)
    matrix_profiles = None
    matrix_package = None
    if isinstance(matrix, dict):
        matrix_profiles = matrix.get("package_profiles")
        matrix_package = matrix.get("package")
        config = matrix.get("config")
        if isinstance(config, dict):
            if matrix_profiles is None:
                matrix_profiles = config.get("package_profiles")
            if matrix_package is None:
                matrix_package = config.get("package")
    linked_profiles = 0
    if isinstance(matrix_profiles, list) and matrix_profiles:
        for index, row in enumerate(matrix_profiles):
            if not isinstance(row, dict):
                raise CleanupError(f"matrix package_profiles[{index}] is not an object")
            relative = row.get("profile_file")
            expected_profile = row.get("profile_sha256")
            expected_model = row.get("model_sha256")
            if not all(isinstance(value, str) for value in (relative, expected_profile, expected_model)):
                raise CleanupError(f"matrix package_profiles[{index}] lacks hashes")
            if relative not in profile_hashes:
                profile_path = _package_path(package, relative)
                if not profile_path.is_file():
                    raise CleanupError(f"matrix references a missing profile: {relative}")
                profile_hashes[relative] = sha256_file(profile_path)
            if profile_hashes[relative] != expected_profile:
                raise CleanupError(f"profile hash does not match matrix: {relative}")
            if expected_model not in model_hashes.values():
                raise CleanupError(f"matrix model hash does not match package: {relative}")
            linked_profiles += 1
    expected_manifest = None
    if isinstance(matrix_package, dict):
        expected_manifest = matrix_package.get("manifest_sha256")
    if expected_manifest is None and isinstance(matrix, dict):
        expected_manifest = matrix.get("package_manifest_sha256")
        config = matrix.get("config")
        if expected_manifest is None and isinstance(config, dict):
            expected_manifest = config.get("package_manifest_sha256")
    if expected_manifest is not None and expected_manifest != manifest_sha256:
        raise CleanupError("model package manifest hash does not match the results matrix")
    matrix_files = matrix_package.get("files") if isinstance(matrix_package, dict) else None
    if isinstance(matrix_files, dict):
        for relative, info in matrix_files.items():
            if not isinstance(relative, str) or not isinstance(info, dict):
                raise CleanupError("results matrix package file identity is malformed")
            if relative not in file_hashes:
                profile_path = _package_path(package, relative)
                if not profile_path.is_file():
                    raise CleanupError(f"matrix package file is missing: {relative}")
                file_hashes[relative] = sha256_file(profile_path)
            expected_sha = info.get("sha256")
            if expected_sha is None or file_hashes[relative] != expected_sha:
                raise CleanupError(f"matrix package hash does not match current package: {relative}")
            expected_size = info.get("bytes")
            if expected_size is not None and int(expected_size) != _package_path(package, relative).stat().st_size:
                raise CleanupError(f"matrix package size does not match current package: {relative}")
        extra = sorted(set(file_hashes) - set(matrix_files))
        if extra:
            raise CleanupError(
                "current package has files absent from the results matrix identity: "
                + ", ".join(extra[:5]))
    return {
        "path": display_path(package),
        "manifest_sha256": manifest_sha256,
        "variants": {key: value for key, value in sorted(model_hashes.items())},
        "profile_count": len(profile_hashes),
        "matrix_profile_rows": linked_profiles,
        "matrix_profile_link": linked_profiles > 0 or expected_manifest is not None,
    }


def _find_single(candidates, label):
    existing = [Path(path) for path in candidates if Path(path).exists()]
    unique = []
    seen = set()
    for path in existing:
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise CleanupError(f"could not locate {label}; pass it explicitly")
    if len(unique) > 1:
        raise CleanupError(
            f"multiple {label} candidates found ({', '.join(map(str, unique))}); pass one explicitly")
    return unique[0]


def find_matrix(current_root):
    current_root = Path(current_root)
    candidates = [
        current_root / "results" / "matrix.json",
        current_root / "matrix.json",
        current_root / "results" / "matrix" / "matrix.json",
    ]
    return _find_single(candidates, "results matrix")


def find_package(current_root):
    current_root = Path(current_root)
    candidates = [
        current_root / "package",
        current_root / "model-package",
        current_root / "package-v2",
    ]
    return _find_single(candidates, "current model package")


def _in_scope(path, scope):
    key = path_key(path)
    for rule in scope:
        if rule.kind == "exact":
            if key == path_key(rule.root):
                return True
            continue
        if path_key(os.path.dirname(os.path.abspath(path))) != path_key(rule.root):
            continue
        if _matches_pattern(os.path.basename(path), rule.pattern):
            return True
    return False


def _verify_evidence(path, evidence):
    for item in evidence:
        relative = item.get("relative")
        if relative is None or not isinstance(relative, str):
            raise CleanupError(f"invalid evidence record for {path}")
        target = path if relative == "" else os.path.join(path, relative)
        if not is_within(target, path):
            raise CleanupError(f"evidence path escapes entry: {relative}")
        try:
            st, reparse = lstat_path(target)
        except OSError as exc:
            raise CleanupError(f"evidence file is missing: {relative}: {exc}") from exc
        if reparse:
            raise CleanupError(f"evidence file became a reparse point: {relative}")
        if int(st.st_size) != int(item.get("size", -1)):
            raise CleanupError(f"evidence file size changed: {relative}")
        if int(st.st_mtime_ns) != int(item.get("mtime_ns", -1)):
            raise CleanupError(f"evidence file mtime changed: {relative}")
        if "sha256" in item:
            actual = sha256_file(target)
            if actual != item["sha256"]:
                raise CleanupError(f"evidence file hash changed: {relative}")


def _verify_reparse_links(path, expected_links, actual_links):
    expected = {link["relative"]: link for link in expected_links}
    actual = {link["relative"]: link for link in actual_links}
    if set(expected) != set(actual):
        raise CleanupError(f"reparse link set changed since inventory: {path}")
    for relative, expected_link in expected.items():
        actual_link = actual[relative]
        if not actual_link.get("link_only_safe"):
            raise CleanupError(f"reparse link is no longer safe to unlink: {relative}")
        for key in ("kind", "raw_target", "target", "target_local", "target_owned"):
            if expected_link.get(key) != actual_link.get(key):
                raise CleanupError(f"reparse link {key} changed since inventory: {relative}")


def _preflight_entry(entry, protection_roots, exact_allowed, scope):
    path = display_path(entry["path"])
    if not is_local_path(path):
        raise CleanupError(f"entry is not a local path: {path}")
    if not _in_scope(path, scope):
        raise CleanupError(f"entry is outside the approved inventory scope: {path}")
    conflict = _protected_conflict(path, protection_roots, exact_allowed)
    if conflict:
        raise CleanupError(f"entry is protected ({conflict}): {path}")
    if path_key(path) in exact_allowed:
        magic = EXACT_MAGIC.get(os.path.basename(path).lower())
        if magic is None or not read_head(path, 64).startswith(magic[0]):
            raise CleanupError(f"exact artifact content changed or is unrecognised: {path}")
    try:
        st, reparse = lstat_path(path)
    except OSError as exc:
        raise CleanupError(f"cannot inspect entry {path}: {exc}") from exc
    if reparse:
        raise CleanupError(f"entry became a reparse point; refusing: {path}")
    identity = entry.get("identity") or {}
    for key, current in (("st_dev", st.st_dev), ("st_ino", st.st_ino),
                         ("st_mtime_ns", st.st_mtime_ns)):
        if int(identity.get(key, -1)) != int(current):
            raise CleanupError(f"identity changed ({key}) since inventory: {path}")
    if entry.get("kind") == "file":
        if not stat.S_ISREG(st.st_mode):
            raise CleanupError(f"entry is no longer a regular file: {path}")
        if int(identity.get("st_size", -1)) != int(st.st_size):
            raise CleanupError(f"file size changed since inventory: {path}")
        _verify_evidence(path, entry.get("evidence", []))
    elif entry.get("kind") == "directory":
        if not stat.S_ISDIR(st.st_mode):
            raise CleanupError(f"entry is no longer a directory: {path}")
        aggregate = walk_tree(path)
        if aggregate["errors"]:
            raise CleanupError(f"tree is not readable: {path}: {aggregate['errors'][0]}")
        if aggregate["reparse"]:
            if entry.get("reparse_policy") != "delete_links_only":
                raise CleanupError(f"tree now contains an unexpected reparse point: {path}")
            _verify_reparse_links(path, entry.get("reparse_links") or [],
                                  aggregate["reparse_details"])
        elif entry.get("reparse_policy") == "delete_links_only":
            raise CleanupError(f"expected reparse links disappeared since inventory: {path}")
        for key in ("bytes", "files", "dirs"):
            if int(identity.get(key, -1)) != int(aggregate[key]):
                raise CleanupError(f"tree {key} changed since inventory: {path}")
        _verify_evidence(path, entry.get("evidence", []))
    else:
        raise CleanupError(f"entry has unsupported kind: {entry.get('kind')}")
    return int(entry.get("bytes") or 0)


def _unlink_file(path):
    try:
        os.unlink(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)


def _remove_reparse_link(path, st):
    """Remove the link/junction itself; never traverse or delete its target."""
    if stat.S_ISDIR(st.st_mode):
        # os.rmdir on a Windows junction/directory symlink removes the link
        # entry only.  It does not recurse into the target.
        os.rmdir(path)
    elif stat.S_ISREG(st.st_mode):
        _unlink_file(path)
    else:
        raise CleanupError(f"unsupported reparse entry type: {path}")


def remove_path_no_follow(path):
    """Remove one path recursively without following links or reparse points."""
    st, reparse = lstat_path(path)
    if reparse:
        _remove_reparse_link(path, st)
        return
    if stat.S_ISREG(st.st_mode):
        _unlink_file(path)
        return
    if not stat.S_ISDIR(st.st_mode):
        raise CleanupError(f"unsupported filesystem entry: {path}")
    try:
        with os.scandir(path) as iterator:
            entries = list(iterator)
    except OSError as exc:
        raise CleanupError(f"cannot list directory {path}: {exc}") from exc
    for entry in entries:
        child_st, child_reparse = lstat_path(entry.path)
        if child_reparse:
            _remove_reparse_link(entry.path, child_st)
            continue
        if stat.S_ISDIR(child_st.st_mode):
            remove_path_no_follow(entry.path)
        elif stat.S_ISREG(child_st.st_mode):
            _unlink_file(entry.path)
        else:
            raise CleanupError(f"unsupported filesystem entry: {entry.path}")
    try:
        os.rmdir(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        os.rmdir(path)


def _disk_snapshot(paths):
    snapshot = {}
    for value in paths:
        drive = os.path.splitdrive(os.path.abspath(os.fspath(value)))[0]
        root = (drive + os.sep) if drive else os.sep
        key = path_key(root)
        if key in snapshot:
            continue
        try:
            usage = shutil.disk_usage(root)
            snapshot[key] = {"root": root, "total": int(usage.total),
                             "used": int(usage.used), "free": int(usage.free)}
        except OSError as exc:
            snapshot[key] = {"root": root, "error": str(exc)}
    return snapshot


def run_apply(manifest_path, current_root, matrix_path=None, package_path=None,
              expected_rows=EXPECTED_MATRIX_ROWS,
              expected_frames=EXPECTED_GENERATED_FRAMES,
              dry_run=False, report_path=None):
    manifest_path = Path(manifest_path)
    current_root = Path(current_root)
    wait_for_current(current_root, 0.0)
    if not is_within(manifest_path, current_root / "cleanup"):
        raise CleanupError(f"manifest must be below <current>/cleanup: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CleanupError(f"cannot read cleanup manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CleanupError(f"unexpected cleanup manifest schema: {manifest.get('schema')!r}")
    if path_key(manifest.get("current_root", "")) != path_key(current_root):
        raise CleanupError("manifest current_root does not match --current-root")
    if int(manifest.get("expected", {}).get("matrix_rows", -1)) != int(expected_rows):
        raise CleanupError("manifest matrix row expectation does not match this invocation")
    if int(manifest.get("expected", {}).get("generated_frames", -1)) != int(expected_frames):
        raise CleanupError("manifest generated-frame expectation does not match this invocation")

    matrix_path = Path(matrix_path) if matrix_path else find_matrix(current_root)
    if package_path:
        package_path = Path(package_path)
    elif manifest.get("current_package"):
        package_path = Path(manifest["current_package"])
    else:
        package_path = find_package(current_root)
    if not is_within(matrix_path, current_root):
        raise CleanupError("results matrix must be below the current validation root")
    if not is_within(package_path, current_root):
        raise CleanupError("model package must be below the current validation root")
    matrix_validation = validate_matrix(matrix_path, expected_rows, expected_frames)
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    package_validation = validate_package(package_path, matrix)

    scope = [ScopeRule(rule["root"], rule["pattern"], rule["kind"])
             for rule in manifest.get("scope", [])]
    if not scope:
        raise CleanupError("manifest has no approved scope")
    protected = list(DEFAULT_PROTECTED) + [Path(value) for value in manifest.get("protected", [])]
    protected.append(current_root)
    exact_allowed = {path_key(value) for value in manifest.get("exact_paths", [])
                     if os.path.basename(os.fspath(value)).lower() in EXACT_MAGIC}

    delete_entries = [entry for entry in manifest.get("entries", [])
                      if entry.get("classification") == "delete"]
    if not delete_entries:
        raise CleanupError("manifest contains no delete entries")
    projected_bytes = sum(int(entry.get("bytes") or 0) for entry in delete_entries)
    # Validate every entry before touching any of them.
    seen_paths = set()
    for entry in delete_entries:
        key = path_key(entry.get("path", ""))
        if key in seen_paths:
            raise CleanupError(f"duplicate delete entry in manifest: {entry.get('path')}")
        seen_paths.add(key)
        _preflight_entry(entry, protected, exact_allowed, scope)

    disk_before = _disk_snapshot([entry["path"] for entry in delete_entries])
    results = []
    actual_bytes = 0
    if not dry_run:
        for entry in delete_entries:
            path = display_path(entry["path"])
            try:
                # Cheap last-moment identity check before the recursive delete.
                st, reparse = lstat_path(path)
                identity = entry.get("identity") or {}
                if reparse:
                    raise CleanupError("entry became a reparse point")
                if (int(identity.get("st_dev", -1)) != int(st.st_dev)
                        or int(identity.get("st_ino", -1)) != int(st.st_ino)
                        or int(identity.get("st_mtime_ns", -1)) != int(st.st_mtime_ns)):
                    raise CleanupError("identity changed immediately before deletion")
                remove_path_no_follow(path)
                actual_bytes += int(entry.get("bytes") or 0)
                result = {"path": path, "status": "deleted",
                          "bytes": int(entry.get("bytes") or 0)}
                if entry.get("reparse_policy"):
                    result["reparse_policy"] = entry["reparse_policy"]
                    result["reparse_links"] = entry.get("reparse_links", [])
                results.append(result)
            except Exception as exc:  # keep deleting other explicitly approved entries
                result = {"path": path, "status": "failed", "error": str(exc)}
                if entry.get("reparse_policy"):
                    result["reparse_policy"] = entry["reparse_policy"]
                    result["reparse_links"] = entry.get("reparse_links", [])
                results.append(result)
    else:
        results = []
        for entry in delete_entries:
            result = {"path": display_path(entry["path"]), "status": "would_delete",
                      "bytes": int(entry.get("bytes") or 0)}
            if entry.get("reparse_policy"):
                result["reparse_policy"] = entry["reparse_policy"]
                result["reparse_links"] = entry.get("reparse_links", [])
            results.append(result)
    disk_after = _disk_snapshot([entry["path"] for entry in delete_entries])

    report = {
        "schema": APPLY_SCHEMA,
        "applied_utc": utc_now(),
        "dry_run": bool(dry_run),
        "manifest": {"path": display_path(manifest_path),
                     "sha256": sha256_file(manifest_path),
                     "inventory_id": manifest.get("inventory_id")},
        "validation": {"matrix": matrix_validation, "package": package_validation},
        "projected": {"entries": len(delete_entries), "bytes": projected_bytes},
        "actual": {"deleted_entries": sum(1 for row in results if row["status"] == "deleted"),
                   "deleted_bytes": actual_bytes,
                   "failed_entries": sum(1 for row in results if row["status"] == "failed")},
        "disk_before": disk_before,
        "disk_after": disk_after,
        "results": results,
    }
    if report_path is None:
        stem = "apply-dry-run" if dry_run else "apply-report"
        report_path = current_root / "cleanup" / f"{stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    report_path = Path(report_path)
    if not is_within(report_path, current_root):
        raise CleanupError("apply report must be below the current validation root")
    _write_json_exclusive(report_path, report, force=True)
    report["report_path"] = display_path(report_path)
    return report


def _scope_from_args(args):
    run_roots = args.run_root or []
    temp_roots = args.temp_root or []
    if not run_roots and not temp_roots:
        run_roots = [Path("D:/"), Path("E:/")]
        temp_roots = [DEFAULT_TEMP_ROOT]
    scope = []
    for root in run_roots:
        scope.append(ScopeRule(display_path(root), "xefg-*", "run"))
    for root in temp_roots:
        scope.append(ScopeRule(display_path(root), "xefg*", "temp"))
    return scope


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="read-only inventory; writes proposed manifest")
    inventory.add_argument("--current-root", type=Path, default=DEFAULT_CURRENT_ROOT)
    inventory.add_argument("--run-root", type=Path, action="append",
                           help="scan top-level xefg-* entries (default D:/ and E:/)")
    inventory.add_argument("--temp-root", type=Path, action="append",
                           help="scan top-level xefg* entries (default user TEMP)")
    inventory.add_argument("--exact-path", type=Path, action="append",
                           help="add an exact file candidate")
    inventory.add_argument("--no-default-exact", action="store_true",
                           help="do not add the repository ptrcheck.obj/vc140.pdb candidates")
    inventory.add_argument("--protect", type=Path, action="append", default=[],
                           help="additional protected path")
    inventory.add_argument("--output", type=Path,
                           help="manifest path (must be below <current>/cleanup unless it is the default)")
    inventory.add_argument("--wait-seconds", type=float, default=0.0,
                           help="wait this long for the current root to exist; never create it")
    inventory.add_argument("--poll-seconds", type=float, default=2.0)
    inventory.add_argument("--force", action="store_true", help="replace an existing manifest")
    inventory.add_argument("--expected-rows", type=int, default=EXPECTED_MATRIX_ROWS)
    inventory.add_argument("--expected-frames", type=int, default=EXPECTED_GENERATED_FRAMES)

    apply = commands.add_parser("apply", help="validate gates and delete an approved manifest")
    apply.add_argument("manifest", type=Path)
    apply.add_argument("--current-root", type=Path, default=DEFAULT_CURRENT_ROOT)
    apply.add_argument("--matrix", type=Path, help="complete results matrix (default: auto-detect)")
    apply.add_argument("--package", type=Path, help="current model package (default: auto-detect)")
    apply.add_argument("--expected-rows", type=int, default=EXPECTED_MATRIX_ROWS)
    apply.add_argument("--expected-frames", type=int, default=EXPECTED_GENERATED_FRAMES)
    apply.add_argument("--dry-run", action="store_true", help="validate without deleting")
    apply.add_argument("--report", type=Path, help="apply report path below current root")

    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            scope = _scope_from_args(args)
            exact_paths = args.exact_path or ([] if args.no_default_exact else list(DEFAULT_EXACT_PATHS))
            protected = list(DEFAULT_PROTECTED) + list(args.protect) + [args.current_root]
            manifest, output = run_inventory(
                args.current_root, scope, exact_paths, protected,
                output=args.output, force=args.force,
                wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds,
                expected_rows=args.expected_rows, expected_frames=args.expected_frames)
            summary = manifest["summary"]
            gib = summary["delete_bytes"] / (1024 ** 3)
            print(f"inventory: {summary['delete_count']} delete entries, "
                  f"{gib:.2f} GiB; keep {summary['keep_count']}; skip {summary['skip_count']}")
            print(f"manifest: {display_path(output)}")
            if manifest["warnings"]:
                for warning in manifest["warnings"]:
                    print(f"warning: {warning}")
            return 0
        report = run_apply(
            args.manifest, args.current_root, args.matrix, args.package,
            expected_rows=args.expected_rows, expected_frames=args.expected_frames,
            dry_run=args.dry_run, report_path=args.report)
        print(json.dumps({
            "dry_run": report["dry_run"],
            "projected_entries": report["projected"]["entries"],
            "projected_bytes": report["projected"]["bytes"],
            "actual_deleted_entries": report["actual"]["deleted_entries"],
            "actual_deleted_bytes": report["actual"]["deleted_bytes"],
            "failed_entries": report["actual"]["failed_entries"],
            "status_counts": report["validation"]["matrix"]["status_counts"],
            "report": report["report_path"],
        }, indent=2))
        return 0 if report["actual"]["failed_entries"] == 0 else 1
    except CleanupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
