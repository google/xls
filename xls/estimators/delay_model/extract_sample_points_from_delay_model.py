#
# Copyright 2023 The XLS Authors
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

r"""Delay model utility.

Reads a "delay_model" textproto and outputs the sample points
(bitwidth tuples) from the data points in "OpSamplesList" format.

Usage:
  extract_sample_points_from_delay_model \
      --input=/path/to/delay_model.textproto \
      --output=/path/to/op_samples.textproto
"""

from absl import app
from absl import flags

from google.protobuf import text_format
from xls.common import gfile
from xls.estimators import estimator_model_pb2


_INPUT = flags.DEFINE_string(
    'input',
    None,
    'The file path/name to write the input delay_model textproto.',
)
flags.mark_flag_as_required('input')

_OUTPUT = flags.DEFINE_string(
    'output',
    None,
    'The file path/name to write the output op_samples textproto.',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dm = estimator_model_pb2.EstimatorModel()
  with gfile.open(_INPUT.value, 'r') as f:
    dm = text_format.Parse(f.read(), dm)

  oss = estimator_model_pb2.OpSamplesList()

  this_op_samples = None

  for dp in dm.data_points:

    if (
        (this_op_samples is None)
        or (this_op_samples.op != dp.operation.op)
        or (this_op_samples.specialization != dp.operation.specialization)
    ):
      this_op_samples = oss.op_samples.add(op=dp.operation.op)
      if dp.operation.specialization:
        this_op_samples.specialization = dp.operation.specialization

    this_op_samples.samples.add(
        result_width=dp.operation.bit_count,
        operand_widths=[opnd.bit_count for opnd in dp.operation.operands],
    )

  print('# proto-file: xls/estimators/estimator_model.proto')
  print('# proto-message: xls.estimator_model.OpSamples')
  print(oss)

  if _OUTPUT.value:
    with gfile.open(_OUTPUT.value, 'w') as f:
      f.write('# proto-file: xls/estimators/estimator_model.proto\n')
      f.write('# proto-message: xls.estimator_model.OpSamples\n')
      f.write(text_format.MessageToString(oss))


if __name__ == '__main__':
  app.run(main)
