#
# Copyright 2025 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the ir_diff_main binary."""

import pathlib
import subprocess

from absl.testing import absltest
from xls.common import runfiles
from xls.eco import ir_patch_pb2

_IR_DIFF_MAIN_PATH = runfiles.get_path("xls/eco/ir_diff_main")
_RISCV_SIMPLE_UNOPT_IR_PATH = runfiles.get_path("xls/examples/riscv_simple.ir")
_RISCV_SIMPLE_OPT_IR_PATH = runfiles.get_path(
    "xls/examples/riscv_simple.opt.ir"
)


class IrDiffMainTest(absltest.TestCase):

  def test_riscv_unopt_vs_opt_optimal(self):
    res = subprocess.run(
        [
            _IR_DIFF_MAIN_PATH,
            "--before_ir",
            _RISCV_SIMPLE_UNOPT_IR_PATH,
            "--after_ir",
            _RISCV_SIMPLE_OPT_IR_PATH,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    diff_output = res.stdout.decode("utf-8")
    self.recordProperty("stderr", res.stderr)
    self.assertIn("Found the optimal edit paths:", diff_output)
    self.assertIn("path cost:", diff_output)
    self.assertIn("Delete edge", diff_output)
    self.assertIn("Delete node", diff_output)
    self.assertIn("Change node", diff_output)
    self.assertIn("Insert node", diff_output)
    self.assertIn("Insert edge", diff_output)

  def test_riscv_unopt_vs_opt_with_timeout(self):
    res = subprocess.run(
        [
            _IR_DIFF_MAIN_PATH,
            "--before_ir",
            _RISCV_SIMPLE_UNOPT_IR_PATH,
            "--after_ir",
            _RISCV_SIMPLE_OPT_IR_PATH,
            "--t",
            "600",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    diff_output = res.stdout.decode("utf-8")
    self.recordProperty("stderr", res.stderr)
    self.assertIn("Found 1 edit paths", diff_output)
    self.assertIn("path cost:", diff_output)
    self.assertIn("Delete edge", diff_output)
    self.assertIn("Delete node", diff_output)
    self.assertIn("Change node", diff_output)
    self.assertIn("Insert node", diff_output)
    self.assertIn("Insert edge", diff_output)

  # TODO(allight): Ideally we'd test that if edit-path checking times out we
  # exit cleanly but its not clear there's a good way to do this.

  def test_riscv_unopt_vs_opt_writes_proto(self):
    tempdir = self.create_tempdir()
    res = subprocess.run(
        [
            _IR_DIFF_MAIN_PATH,
            "--before_ir",
            _RISCV_SIMPLE_UNOPT_IR_PATH,
            "--after_ir",
            _RISCV_SIMPLE_OPT_IR_PATH,
            "--o",
            tempdir.full_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    diff_output = res.stdout.decode("utf-8")
    self.recordProperty("stderr", res.stderr)
    self.assertIn("path cost:", diff_output)
    self.assertIn("Wrote proto:", diff_output)

    with open(
        pathlib.Path(tempdir.full_path) / "patch.bin", "rb"
    ) as proto_file:
      proto = ir_patch_pb2.IrPatchProto.FromString(proto_file.read())
    self.assertNotEmpty(proto.edit_paths)


if __name__ == "__main__":
  absltest.main()
