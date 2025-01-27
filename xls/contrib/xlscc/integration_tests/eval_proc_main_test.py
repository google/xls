#
# Copyright 2020 The XLS Authors
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

import subprocess
import textwrap

from absl import logging

from absl.testing import absltest
from absl.testing import parameterized
from xls.common import runfiles


EVAL_PROC_MAIN_PATH = runfiles.get_path("xls/tools/eval_proc_main")

BLOCK_MEMORY_IR_PATH = runfiles.get_path(
    "xls/contrib/xlscc/integration_tests/testdata/eval_proc_main_test.test_memory.block.ir"
)
BLOCK_MEMORY_SIGNATURE_PATH = runfiles.get_path(
    "xls/contrib/xlscc/integration_tests/testdata/eval_proc_main_test.test_memory.sig.textproto"
)
BLOCK_MEMORY_REWRITES_PATH = runfiles.get_path(
    "xls/contrib/xlscc/integration_tests/testdata/eval_proc_main_test.ram_rewrites.textproto"
)
PROC_ABSTRACT_MEMORY_IR_PATH = runfiles.get_path(
    "xls/contrib/xlscc/integration_tests/testdata/eval_proc_main_test.test_memory.ir"
)
PROC_REWRITTEN_MEMORY_IR_PATH = runfiles.get_path(
    "xls/contrib/xlscc/integration_tests/testdata/eval_proc_main_test.test_memory.opt.ir"
)


def parameterized_block_backends(func):
  return parameterized.named_parameters(
      ("block_jit", ["--backend", "block_jit"]),
      ("block_interpreter", ["--backend", "block_interpreter"]),
  )(func)


def parameterized_proc_backends(func):
  return parameterized.named_parameters(
      ("serial_jit", ["--backend", "serial_jit"]),
      ("ir_interpreter", ["--backend", "ir_interpreter"]),
  )(func)


def run_command(args):
  """Runs the command described by args and returns the completion object."""
  # Don't use check=True because we want to print stderr/stdout on failure for a
  # better error message.
  # pylint: disable=subprocess-run-check

  comp = subprocess.run(
      args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
  )
  if comp.returncode != 0:
    logging.error("Failed to run: %s", repr(args))
    logging.error("stderr: %s", comp.stderr)
    logging.error("stdout: %s", comp.stdout)
  comp.check_returncode()
  return comp


class EvalProcTest(parameterized.TestCase):

  @parameterized_block_backends
  def test_block_memory(self, backend):
    ir_file = BLOCK_MEMORY_IR_PATH
    ram_rewrites_file = BLOCK_MEMORY_REWRITES_PATH
    signature_file = BLOCK_MEMORY_SIGNATURE_PATH
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in : {
            bits[32]:42
            bits[32]:101
            bits[32]:50
            bits[32]:11
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out : {
            bits[32]:126
            bits[32]:303
            bits[32]:150
            bits[32]:33
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--ticks",
        "17",  # Need double the ticks for the memory model to run
        "-v=1",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
        "--block_signature_proto",
        signature_file,
        "--ram_rewrites_textproto",
        ram_rewrites_file,
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Memory Model", output.stderr)

  @parameterized_proc_backends
  def test_proc_abstract_memory(self, backend):
    ir_file = PROC_ABSTRACT_MEMORY_IR_PATH
    ram_rewrites_file = BLOCK_MEMORY_REWRITES_PATH
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in : {
            bits[32]:42
            bits[32]:101
            bits[32]:50
            bits[32]:11
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out : {
            bits[32]:126
            bits[32]:303
            bits[32]:150
            bits[32]:33
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--ticks",
        "17",  # Need double the ticks for the memory model to run
        "-v=1",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
        "--ram_rewrites_textproto",
        ram_rewrites_file,
        "--abstract_ram_model",
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Proc Test_proc", output.stderr)

  @parameterized_proc_backends
  def test_proc_rewritten_memory(self, backend):
    ir_file = PROC_REWRITTEN_MEMORY_IR_PATH
    ram_rewrites_file = BLOCK_MEMORY_REWRITES_PATH
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in : {
            bits[32]:42
            bits[32]:101
            bits[32]:50
            bits[32]:11
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out : {
            bits[32]:126
            bits[32]:303
            bits[32]:150
            bits[32]:33
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--ticks",
        "17",  # Need double the ticks for the memory model to run
        "-v=1",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
        "--ram_rewrites_textproto",
        ram_rewrites_file,
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Proc Test_proc", output.stderr)


if __name__ == "__main__":
  absltest.main()
