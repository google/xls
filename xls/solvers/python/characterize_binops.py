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
"""Runs characterization tests for XLS IR binary operations.

A more-or-less simple driver for LecCharacterizer.Run; creates test case protos
and feeds them (and other config info) into a LecCharacterizer object.
"""

from absl import app
from absl import flags

from google.protobuf import text_format
from xls.common import gfile
from xls.ir import op_pb2
from xls.ir import xls_type_pb2
from xls.solvers.python import lec_characterizer as lc_mod
from xls.solvers.python import lec_characterizer_pb2 as lc_pb2
from xls.solvers.python import z3_lec

flags.DEFINE_string(
    'cell_library_textproto_path', None,
    'Path to the cell library [as a text-format proto] '
    'to use to define the netlist.')
flags.DEFINE_integer(
    'runs_per_type', 8, 'The number of tests to run per set of input types. '
    'If a results file already has at least this many samples '
    'for a set of input types, then no more will be run.')
flags.DEFINE_string('op', None, 'The operator to characterize.')
flags.DEFINE_string(
    'results_path', None,
    'Path at which to write the results. If this file already '
    'exists, new results will be appended there (as '
    'text-format protos.')
flags.DEFINE_string(
    'synthesis_server_address', None,
    'hostname:port on which a synthesis server is running '
    '(to convert generated IR into a netlist).')
flags.DEFINE_list(
    'widths', [], 'Comma-separated list of bit widths to test. Each width will '
    'be applied to the input and output args from each operator.')

flags.mark_flag_as_required('cell_library_textproto_path')
flags.mark_flag_as_required('op')
flags.mark_flag_as_required('results_path')
flags.mark_flag_as_required('synthesis_server_address')
FLAGS = flags.FLAGS


def _save_results(results: lc_pb2.LecTiming) -> None:
  """Callback to save the results proto after every LEC."""
  with gfile.open(FLAGS.results_path, 'w') as fd:
    fd.write(text_format.MessageToString(results))


def main(argv):
  if len(argv) > 3:
    raise app.UsageError('Too many command-line arguments.')

  # Read in the results file to see what configs to test.
  results = lc_pb2.LecTiming()
  if FLAGS.results_path and gfile.exists(FLAGS.results_path):
    with gfile.open(FLAGS.results_path, 'r') as fd:
      results = text_format.ParseLines(fd, lc_pb2.LecTiming())

  with gfile.open(FLAGS.cell_library_textproto_path, 'r') as fd:
    cell_library_textproto = fd.read()

  lc = lc_mod.LecCharacterizer(FLAGS.synthesis_server_address)

  for width in FLAGS.widths:
    bits_type = xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.BITS, bit_count=int(width))

    function_type = xls_type_pb2.FunctionTypeProto()
    function_type.parameters.add().CopyFrom(bits_type)
    function_type.parameters.add().CopyFrom(bits_type)
    function_type.return_type.CopyFrom(bits_type)

    test_case = None
    for result_case in results.test_cases:
      # Find or create a matching test case for this function type.
      if result_case.function_type == function_type:
        test_case = result_case

    if test_case is None:
      test_case = results.test_cases.add()
      test_case.function_type.CopyFrom(function_type)

    runs_left = FLAGS.runs_per_type - len(test_case.exec_times_us)
    if runs_left > 0:
      lc.run(results, op_pb2.OpProto.Value(FLAGS.op), function_type,
             int(runs_left), cell_library_textproto, z3_lec.run, _save_results)


if __name__ == '__main__':
  app.run(main)
