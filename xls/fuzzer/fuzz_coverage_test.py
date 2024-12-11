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
"""Coverage tests for xls.fuzzer.run_fuzz."""

import os
import subprocess

from absl import logging

from google.protobuf import text_format
from absl.testing import absltest
from xls.common import runfiles
from xls.common import test_base
from xls.fuzzer import sample_summary_pb2
from xls.ir import op_pb2 as ir_op_pb2

RUN_FUZZ_MULTIPROCESS_PATH = runfiles.get_path(
    'xls/fuzzer/run_fuzz_multiprocess'
)


def _op_to_string(op: ir_op_pb2.OpProto) -> str:
  name = ir_op_pb2.OpProto.Name(op)
  if name.startswith('OP_'):
    return name[len('OP_') :].lower()
  else:
    return name.lower()


class FuzzSummary:
  """Class encapsulating summary statistics from the fuzzer."""

  def __init__(self, summaries: sample_summary_pb2.SampleSummariesProto):
    self.summaries = summaries
    self._ops = {}
    self._ops_types = {}
    self.max_bits_type_width = 0
    self.max_aggregate_type_width = 0

    # Gather aggregate information about the number of nodes of each type and
    # the maximum bit widths.
    for summary in self.summaries.samples:
      # Only count the ops before optimization because we are testing the fuzzer
      # coverage not the optimizer.
      for node in summary.unoptimized_nodes:
        self._ops[node.op] = self._ops.get(node.op, 0) + 1
        self._ops_types[(node.op, node.type)] = (
            self._ops_types.get((node.op, node.type), 0) + 1
        )
        if node.type == 'bits':
          self.max_bits_type_width = max(self.max_bits_type_width, node.width)
        else:
          self.max_aggregate_type_width = max(
              self.max_aggregate_type_width, node.width
          )

    # Compute aggregate timing information for each field in SampleTimingProto.
    self.aggregate_timing = sample_summary_pb2.SampleTimingProto()
    for summary in self.summaries.samples:
      for field_desc, value in summary.timing.ListFields():
        setattr(
            self.aggregate_timing,
            field_desc.name,
            getattr(self.aggregate_timing, field_desc.name) + value,
        )

  def get_op_count(self, op: str, type_str=None) -> int:
    """Returns the number of generated ops of the given opcode and type."""
    if type_str is None:
      return self._ops.get(op, 0)
    else:
      return self._ops_types.get((op, type_str), 0)


class FuzzCoverageTest(test_base.TestCase):

  def _create_tempdir(self) -> str:
    # Don't cleanup temporary directory if test fails.
    return self.create_tempdir(
        cleanup=test_base.TempFileCleanup.SUCCESS
    ).full_path

  def _read_summaries(self, summary_dir: str) -> FuzzSummary:
    summaries = sample_summary_pb2.SampleSummariesProto()
    for filename in os.listdir(summary_dir):
      if filename.endswith('pb'):
        with open(os.path.join(summary_dir, filename), 'rb') as f:
          # Read in the binary proto file.
          tmp = sample_summary_pb2.SampleSummariesProto().FromString(f.read())
          # Then merge into existing one using text_format.Parse which appends
          # to the message.
          summaries = text_format.Parse(str(tmp), summaries)
    return FuzzSummary(summaries)

  def test_timing(self):
    # Verify the elapsed time for the various operations performed by the
    # fuzzer are non-zero.
    crasher_path = self._create_tempdir()
    summaries_path = self._create_tempdir()
    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH,
        '--seed=42',
        '--crash_path=' + crasher_path,
        '--sample_count=10',
        '--summary_path=' + summaries_path,
        '--calls_per_sample=1',
        '--worker_count=4',
        '--max_width_bits_types=256',
        '--max_width_aggregate_types=1024',
    ])
    summary = self._read_summaries(summaries_path)

    expect_nonzero = (
        'total_ns generate_sample_ns interpret_dslx_ns '
        'convert_ir_ns unoptimized_interpret_ir_ns unoptimized_jit_ns '
        'optimize_ns optimized_jit_ns optimized_interpret_ir_ns'
    ).split()
    for field in expect_nonzero:
      self.assertGreater(
          getattr(summary.aggregate_timing, field),
          0,
          msg=f'Expected non-zero value in timing field {field}',
      )

    expect_zero = ('codegen_ns simulate_ns').split()
    for field in expect_zero:
      self.assertEqual(
          getattr(summary.aggregate_timing, field),
          0,
          msg=f'Expected zero value in timing field {field}',
      )

  def test_width_coverage_wide(self):
    crasher_path = self._create_tempdir()
    summaries_path = self._create_tempdir()
    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH,
        '--seed=42',
        '--crash_path=' + crasher_path,
        '--sample_count=1000',
        '--summary_path=' + summaries_path,
        '--calls_per_sample=1',
        '--worker_count=4',
        '--max_width_bits_types=256',
        '--max_width_aggregate_types=1024',
    ])
    summary = self._read_summaries(summaries_path)
    # Verify the type widths are within limits and some large types are
    # generated.
    self.assertLessEqual(summary.max_bits_type_width, 256)
    self.assertGreater(summary.max_bits_type_width, 64)

    self.assertLessEqual(summary.max_aggregate_type_width, 1024)
    self.assertGreater(summary.max_aggregate_type_width, 512)

  def test_op_coverage_function(self):
    # This is a probabilistic test which verifies that all expected op codes are
    # covered. To ensure coverage with near certain probability this test runs
    # for a long time.
    crasher_path = self._create_tempdir()
    summaries_path = self._create_tempdir()

    # Test coverage of all ops.
    expect_seen = [
        ir_op_pb2.OP_ADD,
        ir_op_pb2.OP_AND,
        ir_op_pb2.OP_AND_REDUCE,
        ir_op_pb2.OP_ARRAY,
        ir_op_pb2.OP_ARRAY_CONCAT,
        ir_op_pb2.OP_ARRAY_INDEX,
        ir_op_pb2.OP_ARRAY_SLICE,
        ir_op_pb2.OP_ARRAY_UPDATE,
        ir_op_pb2.OP_BIT_SLICE,
        ir_op_pb2.OP_BIT_SLICE_UPDATE,
        ir_op_pb2.OP_CONCAT,
        ir_op_pb2.OP_COUNTED_FOR,
        ir_op_pb2.OP_DECODE,
        ir_op_pb2.OP_DYNAMIC_BIT_SLICE,
        ir_op_pb2.OP_ENCODE,
        ir_op_pb2.OP_EQ,
        ir_op_pb2.OP_GATE,
        ir_op_pb2.OP_INVOKE,
        ir_op_pb2.OP_LITERAL,
        ir_op_pb2.OP_MAP,
        ir_op_pb2.OP_NE,
        ir_op_pb2.OP_NEG,
        ir_op_pb2.OP_NOT,
        ir_op_pb2.OP_ONE_HOT,
        ir_op_pb2.OP_ONE_HOT_SEL,
        ir_op_pb2.OP_OR,
        ir_op_pb2.OP_OR_REDUCE,
        ir_op_pb2.OP_PARAM,
        ir_op_pb2.OP_PRIORITY_SEL,
        ir_op_pb2.OP_REVERSE,
        ir_op_pb2.OP_SDIV,
        ir_op_pb2.OP_SEL,
        ir_op_pb2.OP_SGE,
        ir_op_pb2.OP_SGT,
        ir_op_pb2.OP_SHLL,
        ir_op_pb2.OP_SHRA,
        ir_op_pb2.OP_SHRL,
        ir_op_pb2.OP_SIGN_EXT,
        ir_op_pb2.OP_SLE,
        ir_op_pb2.OP_SLT,
        ir_op_pb2.OP_SMOD,
        ir_op_pb2.OP_SMUL,
        ir_op_pb2.OP_SMULP,
        ir_op_pb2.OP_SUB,
        ir_op_pb2.OP_TUPLE,
        ir_op_pb2.OP_TUPLE_INDEX,
        ir_op_pb2.OP_UDIV,
        ir_op_pb2.OP_UGE,
        ir_op_pb2.OP_UGT,
        ir_op_pb2.OP_ULE,
        ir_op_pb2.OP_ULT,
        ir_op_pb2.OP_UMUL,
        ir_op_pb2.OP_UMULP,
        ir_op_pb2.OP_XOR,
        ir_op_pb2.OP_XOR_REDUCE,
        ir_op_pb2.OP_ZERO_EXT,
        ir_op_pb2.OP_UMOD,
    ]
    expect_not_seen = [
        ir_op_pb2.OP_AFTER_ALL,
        ir_op_pb2.OP_ASSERT,
        ir_op_pb2.OP_COVER,
        ir_op_pb2.OP_DYNAMIC_COUNTED_FOR,
        ir_op_pb2.OP_IDENTITY,
        ir_op_pb2.OP_INPUT_PORT,
        ir_op_pb2.OP_INSTANTIATION_INPUT,
        ir_op_pb2.OP_INSTANTIATION_OUTPUT,
        ir_op_pb2.OP_INVALID,
        ir_op_pb2.OP_MIN_DELAY,
        ir_op_pb2.OP_NAND,
        ir_op_pb2.OP_NEXT_VALUE,
        ir_op_pb2.OP_NOR,
        ir_op_pb2.OP_OUTPUT_PORT,
        ir_op_pb2.OP_RECEIVE,
        ir_op_pb2.OP_REGISTER_READ,
        ir_op_pb2.OP_REGISTER_WRITE,
        ir_op_pb2.OP_SEND,
        ir_op_pb2.OP_STATE_READ,
        ir_op_pb2.OP_TRACE,
    ]
    # The set of expected to be seen and expected to be not seen should cover
    # all ops.
    self.assertLen(
        ir_op_pb2.OpProto.values(), len(expect_seen) + len(expect_not_seen)
    )

    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH,
        '--seed=42',
        '--crash_path=' + crasher_path,
        '--sample_count=5000',
        '--summary_path=' + summaries_path,
        '--calls_per_sample=1',
        '--worker_count=4',
    ])

    summary = self._read_summaries(summaries_path)

    # Test that we generate interesting parameters.
    self.assertGreater(summary.get_op_count('param'), 0)
    self.assertGreater(summary.get_op_count('param', type_str='tuple'), 0)
    self.assertGreater(summary.get_op_count('param', type_str='array'), 0)

    for op in expect_seen:
      logging.vlog(
          1,
          'seen op %s: %d',
          ir_op_pb2.OpProto.Name(op),
          summary.get_op_count(_op_to_string(op)),
      )
    for op in expect_not_seen:
      logging.vlog(
          1,
          'not seen op %s: %d',
          _op_to_string(op),
          summary.get_op_count(_op_to_string(op)),
      )

    for op in expect_seen:
      self.assertGreater(
          summary.get_op_count(_op_to_string(op)),
          0,
          msg=f'Expected fuzzer to generate op "{_op_to_string(op)}"',
      )

    for op in expect_not_seen:
      self.assertEqual(
          summary.get_op_count(_op_to_string(op)),
          0,
          msg=f'Expected fuzzer to not generate op "{_op_to_string(op)}"',
      )

  def test_op_coverage_proc(self):
    # This is a probabilistic test which verifies that all expected op codes are
    # covered. To ensure coverage with near certain probability this test runs
    # for a long time.
    crasher_path = self._create_tempdir()
    summaries_path = self._create_tempdir()

    # Test coverage of all ops.
    expect_seen = [
        ir_op_pb2.OP_ADD,
        ir_op_pb2.OP_AFTER_ALL,
        ir_op_pb2.OP_AND,
        ir_op_pb2.OP_AND_REDUCE,
        ir_op_pb2.OP_ARRAY,
        ir_op_pb2.OP_ARRAY_CONCAT,
        ir_op_pb2.OP_ARRAY_INDEX,
        ir_op_pb2.OP_ARRAY_SLICE,
        ir_op_pb2.OP_ARRAY_UPDATE,
        ir_op_pb2.OP_BIT_SLICE,
        ir_op_pb2.OP_BIT_SLICE_UPDATE,
        ir_op_pb2.OP_CONCAT,
        ir_op_pb2.OP_COUNTED_FOR,
        ir_op_pb2.OP_DECODE,
        ir_op_pb2.OP_DYNAMIC_BIT_SLICE,
        ir_op_pb2.OP_ENCODE,
        ir_op_pb2.OP_EQ,
        ir_op_pb2.OP_GATE,
        ir_op_pb2.OP_INVOKE,
        ir_op_pb2.OP_LITERAL,
        ir_op_pb2.OP_MAP,
        ir_op_pb2.OP_NE,
        ir_op_pb2.OP_NEG,
        ir_op_pb2.OP_NOT,
        ir_op_pb2.OP_ONE_HOT,
        ir_op_pb2.OP_ONE_HOT_SEL,
        ir_op_pb2.OP_OR,
        ir_op_pb2.OP_OR_REDUCE,
        ir_op_pb2.OP_PARAM,
        ir_op_pb2.OP_PRIORITY_SEL,
        ir_op_pb2.OP_RECEIVE,
        ir_op_pb2.OP_REVERSE,
        ir_op_pb2.OP_SEND,
        ir_op_pb2.OP_SDIV,
        ir_op_pb2.OP_SEL,
        ir_op_pb2.OP_SGE,
        ir_op_pb2.OP_SGT,
        ir_op_pb2.OP_SHLL,
        ir_op_pb2.OP_SHRA,
        ir_op_pb2.OP_SHRL,
        ir_op_pb2.OP_SIGN_EXT,
        ir_op_pb2.OP_SLE,
        ir_op_pb2.OP_SLT,
        ir_op_pb2.OP_SMOD,
        ir_op_pb2.OP_SMUL,
        ir_op_pb2.OP_SMULP,
        ir_op_pb2.OP_STATE_READ,
        ir_op_pb2.OP_SUB,
        ir_op_pb2.OP_TUPLE,
        ir_op_pb2.OP_TUPLE_INDEX,
        ir_op_pb2.OP_UDIV,
        ir_op_pb2.OP_UGE,
        ir_op_pb2.OP_UGT,
        ir_op_pb2.OP_ULE,
        ir_op_pb2.OP_ULT,
        ir_op_pb2.OP_UMOD,
        ir_op_pb2.OP_UMUL,
        ir_op_pb2.OP_UMULP,
        ir_op_pb2.OP_XOR,
        ir_op_pb2.OP_XOR_REDUCE,
        ir_op_pb2.OP_ZERO_EXT,
    ]
    expect_not_seen = [
        ir_op_pb2.OP_ASSERT,
        ir_op_pb2.OP_COVER,
        ir_op_pb2.OP_DYNAMIC_COUNTED_FOR,
        ir_op_pb2.OP_IDENTITY,
        ir_op_pb2.OP_INPUT_PORT,
        ir_op_pb2.OP_INSTANTIATION_INPUT,
        ir_op_pb2.OP_INSTANTIATION_OUTPUT,
        ir_op_pb2.OP_INVALID,
        ir_op_pb2.OP_MIN_DELAY,
        ir_op_pb2.OP_NAND,
        ir_op_pb2.OP_NEXT_VALUE,
        ir_op_pb2.OP_NOR,
        ir_op_pb2.OP_OUTPUT_PORT,
        ir_op_pb2.OP_REGISTER_READ,
        ir_op_pb2.OP_REGISTER_WRITE,
        ir_op_pb2.OP_TRACE,
    ]
    # The set of expected to be seen and expected to be not seen should cover
    # all ops.
    self.assertEqual(
        len(expect_seen) + len(expect_not_seen), len(ir_op_pb2.OpProto.values())
    )

    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH,
        '--seed=42',
        '--crash_path=' + crasher_path,
        '--sample_count=5000',
        '--summary_path=' + summaries_path,
        '--proc_ticks=1',
        '--generate_proc',
        '--worker_count=4',
    ])

    summary = self._read_summaries(summaries_path)

    # Test that we generate interesting state elements.
    self.assertGreater(summary.get_op_count('state_read'), 0)
    self.assertGreater(summary.get_op_count('state_read', type_str='tuple'), 0)
    self.assertGreater(summary.get_op_count('state_read', type_str='array'), 0)

    for op in expect_seen:
      logging.vlog(
          1,
          'seen op %s: %d',
          _op_to_string(op),
          summary.get_op_count(_op_to_string(op)),
      )
    for op in expect_not_seen:
      logging.vlog(
          1,
          'not seen op %s: %d',
          _op_to_string(op),
          summary.get_op_count(_op_to_string(op)),
      )

    for op in expect_seen:
      self.assertGreater(
          summary.get_op_count(_op_to_string(op)),
          0,
          msg=f'Expected fuzzer to generate op "{_op_to_string(op)}"',
      )

    for op in expect_not_seen:
      self.assertEqual(
          summary.get_op_count(_op_to_string(op)),
          0,
          msg=f'Expected fuzzer to not generate op "{_op_to_string(op)}"',
      )


if __name__ == '__main__':
  absltest.main()
