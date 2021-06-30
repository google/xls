# Lint as: python3
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

from google.protobuf import text_format
from absl.testing import absltest
from xls.common import runfiles
from xls.common import test_base
from xls.fuzzer import sample_summary_pb2

RUN_FUZZ_MULTIPROCESS_PATH = runfiles.get_path(
    'xls/fuzzer/run_fuzz_multiprocess')


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
        self._ops_types[(node.op, node.type)] = self._ops_types.get(
            (node.op, node.type), 0) + 1
        if node.type == 'bits':
          self.max_bits_type_width = max(self.max_bits_type_width, node.width)
        else:
          self.max_aggregate_type_width = max(self.max_aggregate_type_width,
                                              node.width)

    # Compute aggregate timing information for each field in SampleTimingProto.
    self.aggregate_timing = sample_summary_pb2.SampleTimingProto()
    for summary in self.summaries.samples:
      for field_desc, value in summary.timing.ListFields():
        setattr(self.aggregate_timing, field_desc.name,
                getattr(self.aggregate_timing, field_desc.name) + value)

  def get_op_count(self, op: str, type_str=None) -> int:
    """Returns the number of generated ops of the given opcode and type."""
    if type_str is None:
      return self._ops.get(op, 0)
    else:
      return self._ops_types.get((op, type_str), 0)


class FuzzCoverageTest(test_base.TestCase):

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
    # Verify the the elapsed time for the various operations performed by the
    # fuzzer are non-zero.
    crasher_path = self.create_tempdir().full_path
    summaries_path = self.create_tempdir().full_path
    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH, '--seed=42', '--crash_path=' + crasher_path,
        '--sample_count=10', '--summary_path=' + summaries_path,
        '--calls_per_sample=1', '--worker_count=4',
        '--max_width_bits_types=256', '--max_width_aggregate_types=1024'
    ])
    summary = self._read_summaries(summaries_path)

    expect_nonzero = (
        'total_ns generate_sample_ns interpret_dslx_ns '
        'convert_ir_ns unoptimized_interpret_ir_ns unoptimized_jit_ns '
        'optimize_ns optimized_jit_ns optimized_interpret_ir_ns').split()
    for field in expect_nonzero:
      self.assertGreater(
          getattr(summary.aggregate_timing, field),
          0,
          msg=f'Expected non-zero value in timing field {field}')

    expect_zero = ('codegen_ns simulate_ns').split()
    for field in expect_zero:
      self.assertEqual(
          getattr(summary.aggregate_timing, field),
          0,
          msg=f'Expected zero value in timing field {field}')

  def test_width_coverage_wide(self):
    crasher_path = self.create_tempdir().full_path
    summaries_path = self.create_tempdir().full_path
    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH, '--seed=42', '--crash_path=' + crasher_path,
        '--sample_count=100', '--summary_path=' + summaries_path,
        '--calls_per_sample=1', '--worker_count=4',
        '--max_width_bits_types=256', '--max_width_aggregate_types=1024'
    ])
    summary = self._read_summaries(summaries_path)
    # Verify that at least some bits types are greater than 64.
    self.assertGreater(summary.max_bits_type_width, 64)

    # TODO(meheff): The aggregate limit doesn't work perhaps because map can
    # generate map generates wide arrays. Enable these assertions when fixed.
    # self.assertLessEqual(summary.max_aggregate_type_width, 10240)
    # self.assertGreater(summary.max_aggregate_type_width, 5120)

  def test_op_coverage(self):
    crasher_path = self.create_tempdir().full_path
    summaries_path = self.create_tempdir().full_path

    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH, '--seed=42', '--crash_path=' + crasher_path,
        '--sample_count=500', '--summary_path=' + summaries_path,
        '--calls_per_sample=1', '--worker_count=4'
    ])

    summary = self._read_summaries(summaries_path)

    # Test that we generate interesting parameters.
    self.assertGreater(summary.get_op_count('param'), 0)
    self.assertGreater(summary.get_op_count('param', type_str='tuple'), 0)
    self.assertGreater(summary.get_op_count('param', type_str='array'), 0)

    # Test coverage of all other  ops.
    expect_seen = (
        'add and and_reduce array array_index array_update array_concat '
        'bit_slice bit_slice_update concat counted_for dynamic_bit_slice '
        'encode eq literal map ne neg not one_hot one_hot_sel or '
        'or_reduce reverse sel sge sgt shll shra shrl sign_ext sle slt smul '
        'sub tuple tuple_index uge ugt ule ult umul xor xor_reduce zero_ext'
    ).split()
    for op in expect_seen:
      self.assertGreater(
          summary.get_op_count(op),
          0,
          msg=f'Expected fuzzer to generate op "{op}"')

    # These ops are not yet supported by the fuzzer.
    expect_not_seen = ('after_all assert decode identity '
                       'dynamic_counted_for invoke nand nor receive '
                       'sdiv send smod udiv umod').split()
    for op in expect_not_seen:
      self.assertEqual(
          summary.get_op_count(op),
          0,
          msg=f'Expected fuzzer to not generate op "{op}"')


if __name__ == '__main__':
  absltest.main()
