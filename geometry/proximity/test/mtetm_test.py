import os
from pathlib import Path
import subprocess
import unittest

from python import runfiles


class TetMTetM(unittest.TestCase):

    def test_mtetm(self):
        manifest = runfiles.Create()
        mtetm = manifest.Rlocation(
            "drake/geometry/proximity/mtetm")
        surface_file = manifest.Rlocation(
            "drake/geometry/test/non_convex_mesh.obj")
        tmpdir = Path(os.environ["TEST_TMPDIR"])

        result_file = tmpdir / "non_convex_mesh.vtk"
        subprocess.run(
            [mtetm, surface_file, result_file], check=True)

        self.assertTrue(result_file.exists())
