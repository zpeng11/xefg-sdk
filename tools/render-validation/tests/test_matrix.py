import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("matrix", Path(__file__).parents[1] / "run_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class MatrixTests(unittest.TestCase):
    def test_failed_scene_does_not_hide_other_scene_results(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "matrix"
            argv = ["run_matrix.py","--renderer","renderer.exe","--xess-bin","sdk",
                    "--package","package","--out",str(output)]
            with patch.object(matrix.sys,"argv",argv), patch.object(matrix.subprocess,"run",return_value=SimpleNamespace(returncode=2)) as run:
                self.assertEqual(matrix.main(),1)
            self.assertEqual(run.call_count,4)
            report = json.loads((output / "matrix.json").read_text())
            self.assertEqual(len(report["scenes"]),4)
            self.assertTrue(all(not row["completed"] for row in report["scenes"]))
            self.assertTrue(all(row["stages"][0]["exit_code"] == 2 for row in report["scenes"]))

    def test_new_output_only(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = ["run_matrix.py","--renderer","renderer.exe","--xess-bin","sdk",
                    "--package","package","--out",temp]
            with patch.object(matrix.sys,"argv",argv), patch.object(matrix.subprocess,"run") as run:
                with self.assertRaises(FileExistsError):
                    matrix.main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
