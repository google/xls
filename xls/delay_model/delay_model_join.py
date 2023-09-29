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

Joins an "op_models" textproto and a "data_points" textproto
into a single "delay_model" textproto.  Checks are performed
that ops with regression estimators (and only those ops)
have data points.

Usage:
  delay_model_join --op_models=/path/to/op_models.textproto \
      --data_points=/path/to/data_points.textproto \
      --output=/path/to/delay_model.textproto
"""

import sys

from absl import app
from absl import flags

from google.protobuf import text_format
from xls.common import gfile
from xls.delay_model import delay_model_pb2


_OP_MODELS = flags.DEFINE_string(
    'op_models', None,
    'The file path/name location of the input op_models textproto .')
flags.mark_flag_as_required('op_models')

_DATA_POINTS = flags.DEFINE_string(
    'data_points', None,
    'The file path/name location of the input data_points textproto .')
flags.mark_flag_as_required('data_points')

_OUTPUT = flags.DEFINE_string(
    'output', None,
    'The file path/name to write the output delay_model textproto.')


def sync_check(
    oms: delay_model_pb2.OpModels,
    dps: delay_model_pb2.DataPoints) -> None:
  """Compare the op types in the source protos to make sure they're in sync.

  Every op that has a regression estimator should have data point(s).
  Any op that doesn't have a regresssion estimator should not have
  any data points.

  Args:
    oms: delay_model_pb2.OpModels     OpModels proto
    dps: delay_model_pb2.DataPoints   DataPoints proto
  """
  regression_ops = set([x.op for x in oms.op_models if
                        x.estimator.WhichOneof('estimator') == 'regression'])
  data_point_ops = set([x.operation.op for x in dps.data_points])

  errors = []
  for op in regression_ops:
    if op not in data_point_ops:
      errors.append(
          f'# {op} has a regression estimator but no data points.')
  for op in data_point_ops:
    if op not in regression_ops:
      errors.append(
          f'# {op} has data points but doesn\'t have a regression estimator.')
  if errors:
    error_msg = '#\n# ERROR: ISSUES FOUND:\n' + '\n'.join(errors) + '\n#'
    print(error_msg, file=sys.stderr)
    exit(1)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  oms = delay_model_pb2.OpModels()
  with gfile.open(_OP_MODELS.value, 'r') as f:
    oms = text_format.Parse(f.read(), oms)

  dps = delay_model_pb2.DataPoints()
  with gfile.open(_DATA_POINTS.value, 'r') as f:
    dps = text_format.Parse(f.read(), dps)

  dm = delay_model_pb2.DelayModel()

  for om in oms.op_models:
    dm.op_models.append(om)

  for dp in dps.data_points:
    dm.data_points.append(dp)

  print('# proto-file: xls/delay_model/delay_model.proto')
  print('# proto-message: xls.delay_model.DelayModel')
  print(dm, end='')

  if _OUTPUT.value:
    with gfile.open(_OUTPUT.value, 'w') as f:
      f.write('# proto-file: xls/delay_model/delay_model.proto\n')
      f.write('# proto-message: xls.delay_model.DelayModel\n')
      f.write(text_format.MessageToString(dm))

  sync_check(oms, dps)


if __name__ == '__main__':
  app.run(main)
