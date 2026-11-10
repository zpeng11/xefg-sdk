"""Installed-package CLI and dependency-isolation checks.

Each check runs in a fresh subprocess with ``-I`` so it observes exactly what
an installed ``xefg_sdk`` exposes.  No repository source path is injected.

* Importing the core runtime must not import ONNX, ONNX Runtime, or the
  exporter package.
* Importing the exporter must work with ONNX Runtime deliberately blocked.
* The supported exporter CLI must answer ``--help`` without a trace or a
  package.
"""
import subprocess
import sys
import unittest

BLOCKER = """\
import importlib.abc
import sys

BLOCKED = {%(blocked)s}


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError("blocked import: " + fullname)
        return None


sys.meta_path.insert(0, Blocker())
%(body)s
"""


def _run(code, timeout=120):
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True, text=True, timeout=timeout)


def _probe(blocked, body):
    names = ", ".join("%r" % name for name in blocked)
    return _run(BLOCKER % {"blocked": names, "body": body})


class DependencyIsolationTests(unittest.TestCase):
    def assert_succeeded(self, completed):
        self.assertEqual(
            completed.returncode, 0,
            "stdout:\n%s\nstderr:\n%s" % (completed.stdout, completed.stderr))

    def test_core_runtime_does_not_import_onnx_or_exporter(self):
        completed = _probe(("onnx", "onnxruntime"), """
import xefg_sdk

loaded = sorted(name for name in sys.modules
                if name.split(".")[0] in BLOCKED)
assert not loaded, loaded
assert "xefg_sdk.export" not in sys.modules, "core imported xefg_sdk.export"
assert "xefg_sdk.export.export_sdk" not in sys.modules
assert xefg_sdk.__all__
print(xefg_sdk.__version__)
""")
        self.assert_succeeded(completed)

    def test_exporter_imports_with_onnxruntime_blocked(self):
        completed = _probe(("onnxruntime",), """
import onnx
from xefg_sdk.export import export_sdk, input_provenance, m9_dynamic

assert "onnxruntime" not in sys.modules, sorted(
    name for name in sys.modules if name.startswith("onnxruntime"))
assert callable(export_sdk.main)
assert callable(input_provenance.build_report)
assert callable(m9_dynamic.build_specs)
print("ok")
""")
        self.assert_succeeded(completed)


class CliHelpTests(unittest.TestCase):
    def test_help_without_export_dependencies(self):
        completed = _probe(("onnx", "onnxruntime"), """
from xefg_sdk.export.cli import main
main(["--help"])
""")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--runs", completed.stdout)

    def test_missing_onnx_explains_export_extra(self):
        completed = _run("""
import importlib.abc
import sys

class MissingOnnx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "onnx":
            raise ModuleNotFoundError("No module named 'onnx'", name="onnx")

sys.meta_path.insert(0, MissingOnnx())
from xefg_sdk.export.cli import main
main(["--runs", "unused", "--out", "unused"])
""")
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("xefg-sdk[export]", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_export_cli_help(self):
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "xefg_sdk.export", "--help"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(
            completed.returncode, 0,
            "stdout:\n%s\nstderr:\n%s" % (completed.stdout, completed.stderr))
        self.assertNotIn("ModuleNotFoundError", completed.stderr)
        text = completed.stdout.lower()
        self.assertIn("usage:", text)
        self.assertIn("xefg-export", text)
        self.assertIn("--runs", text)
        self.assertIn("--out", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
