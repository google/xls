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

import io
import pathlib
from unittest import mock

from absl import flags
from absl.testing import flagsaver

from absl.testing import absltest
from xls.common import runfiles
from xls.eco import ir_diff
from xls.eco import ir_diff_main
from xls.eco import ir_patch_pb2

_IR_DIFF_MAIN_PATH = runfiles.get_path("xls/eco/ir_diff_main")
_RISCV_SIMPLE_UNOPT_IR_PATH = runfiles.get_path("xls/examples/riscv_simple.ir")
_RISCV_SIMPLE_OPT_IR_PATH = runfiles.get_path(
    "xls/examples/riscv_simple.opt.ir"
)


class IrDiffMainTest(absltest.TestCase):

  @flagsaver.flagsaver(
      before_ir=_RISCV_SIMPLE_UNOPT_IR_PATH,
      after_ir=_RISCV_SIMPLE_OPT_IR_PATH,
  )
  def test_riscv_unopt_vs_opt_optimal(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      ir_diff_main.main([])

    diff_output = mock_stdout.getvalue()
    self.assertIn("Found the optimal edit paths:", diff_output)
    self.assertIn("path cost:", diff_output)
    self.assertIn("Delete edge", diff_output)
    self.assertIn("Delete node", diff_output)
    self.assertIn("Change node", diff_output)
    self.assertIn("Insert node", diff_output)
    self.assertIn("Insert edge", diff_output)

  @flagsaver.flagsaver(
      before_ir=_RISCV_SIMPLE_UNOPT_IR_PATH,
      after_ir=_RISCV_SIMPLE_OPT_IR_PATH,
      # We test that setting a timeout works.
      t=600,
  )
  def test_riscv_unopt_vs_opt_with_timeout(self):
    mock_stdout = io.StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      ir_diff_main.main([])

    diff_output = mock_stdout.getvalue()
    self.assertIn("Found 1 edit paths", diff_output)
    self.assertIn("path cost:", diff_output)
    self.assertIn("Delete edge", diff_output)
    self.assertIn("Delete node", diff_output)
    self.assertIn("Change node", diff_output)
    self.assertIn("Insert node", diff_output)
    self.assertIn("Insert edge", diff_output)

  @flagsaver.flagsaver(
      before_ir=_RISCV_SIMPLE_UNOPT_IR_PATH,
      after_ir=_RISCV_SIMPLE_OPT_IR_PATH,
      # We set a timeout here, but the value doesn't matter. We'll mock the
      # underlying find_optimized_edit_paths call to yield no paths.
      t=1,
  )
  @mock.patch.object(ir_diff, "find_optimized_edit_paths")
  def test_riscv_unopt_vs_opt_with_timeout_returning_nothing(
      self, mock_find_optimized_edit_paths
  ):
    # We mock the underlying find_optimized_edit_paths call to yield no paths.
    # This checks that the timeout codepath handles that case correctly.
    def _mock_find_optimized_edit_paths(
        before_graph, after_graph, timeout_limit
    ):
      del before_graph, after_graph, timeout_limit
      yield from []

    mock_find_optimized_edit_paths.side_effect = _mock_find_optimized_edit_paths
    with self.assertRaisesRegex(ValueError, "edit paths was not set"):
      ir_diff_main.main([])

  @flagsaver.flagsaver(
      before_ir=_RISCV_SIMPLE_UNOPT_IR_PATH,
      after_ir=_RISCV_SIMPLE_OPT_IR_PATH,
  )
  def test_riscv_unopt_vs_opt_writes_proto(self):
    tempdir = self.create_tempdir()
    mock_stdout = io.StringIO()
    with flagsaver.flagsaver(o=tempdir.full_path), mock.patch(
        "sys.stdout", mock_stdout
    ):
      ir_diff_main.main([])

    diff_output = mock_stdout.getvalue()
    self.assertIn("path cost:", diff_output)
    self.assertIn("Wrote proto:", diff_output)

    with open(
        pathlib.Path(tempdir.full_path) / "patch.bin", "rb"
    ) as proto_file:
      proto = ir_patch_pb2.IrPatchProto.FromString(proto_file.read())
    self.assertNotEmpty(proto.edit_paths)


if __name__ == "__main__":
  # Use flagholders to set required flags to some unusable default.
  flags.set_default(ir_diff_main._BEFORE_IR_PATH, "replace_me")
  flags.set_default(ir_diff_main._AFTER_IR_PATH, "replace_me")

  absltest.main()
