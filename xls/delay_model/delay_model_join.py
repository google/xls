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

import collections
import enum
from typing import MutableMapping, Sequence

from absl import app
from absl import flags

from google.protobuf import text_format
from xls.common import gfile
from xls.delay_model import delay_model_pb2
from xls.delay_model import delay_model_utils


@enum.unique
class UpdateMode(enum.Enum):
  """Options for delay model update semantics."""

  # Replace the whole delay model.
  NONE = 'none'
  # Replace the whole set of points for all ops having any updated/added point.
  UPDATE_WHOLE_OPS = 'update_whole_ops'
  # Update/add data points, and keep all existing points having no updated
  # counterparts.
  UPDATE_POINTS = 'update_points'


_OP_MODELS = flags.DEFINE_string(
    'op_models',
    None,
    'The file path/name location of the input op_models textproto.',
)
flags.mark_flag_as_required('op_models')

_DATA_POINTS = flags.DEFINE_string(
    'data_points',
    None,
    'The file path/name location of the input data_points textproto.',
)
flags.mark_flag_as_required('data_points')

_OUTPUT = flags.DEFINE_string(
    'output',
    None,
    'The file path/name to write the output delay_model textproto.',
)

_UPDATE_MODE = flags.DEFINE_enum_class(
    'update_mode',
    UpdateMode.NONE,
    UpdateMode,
    'Whether and how to apply update semantics, where `data_points` may contain'
    ' only some new points to incorporate into an existing model. Can be'
    ' `none`, `update_whole_ops`, or `update_points`. `update_whole_ops`'
    ' discards any existing points for ops represented in data_points, while'
    ' `update_points` only discards existing points that have exact'
    ' counterparts.',
)


def sync_check(
    oms: delay_model_pb2.OpModels,
    dps: delay_model_pb2.DataPoints,
    update_mode: UpdateMode,
) -> None:
  """Compares the op types in the source protos to make sure they're in sync.

  Every op that has a regression estimator should have data point(s).
  Any op that doesn't have a regresssion estimator should not have
  any data points.

  Args:
    oms: delay_model_pb2.OpModels     OpModels proto
    dps: delay_model_pb2.DataPoints   DataPoints proto
    update_mode: UpdateMode           The update semantics to apply
  """
  regression_ops = {
      x.op
      for x in oms.op_models
      if x.estimator.WhichOneof('estimator') == 'regression'
  }
  data_point_ops = set([x.operation.op for x in dps.data_points])

  errors = []
  for op in regression_ops:
    if op not in data_point_ops and update_mode == UpdateMode.NONE:
      errors.append(f'# {op} has a regression estimator but no data points.')
  for op in data_point_ops:
    if op not in regression_ops:
      errors.append(
          f"# {op} has data points but doesn't have a regression estimator."
      )
  if errors:
    raise app.UsageError('Issues found:\n' + '\n'.join(errors))


def create_op_to_points_mapping(
    dps: Sequence[delay_model_pb2.DataPoint],
) -> MutableMapping[str, Sequence[delay_model_pb2.DataPoint]]:
  """Returns a dict of sublists of the given data points, separated by op."""
  result = collections.defaultdict(list)
  for point in dps:
    result[point.operation.op].append(point)
  return result


def update_op_data_points(
    existing_dps: Sequence[delay_model_pb2.DataPoint],
    new_dps: Sequence[delay_model_pb2.DataPoint],
) -> Sequence[delay_model_pb2.DataPoint]:
  """Returns a combined sequence of existing_dps + new_dps that are for one op.

  The new_dps data is preferred where a sample spec is represented in both.

  Args:
    existing_dps: The existing data points.
    new_dps: The updated and/or additional data points.
  """
  existing_order = [
      delay_model_utils.get_data_point_key(point) for point in existing_dps
  ]
  existing_points = delay_model_utils.map_data_points_by_key(existing_dps)
  new_points = delay_model_utils.map_data_points_by_key(new_dps)
  return [
      new_points[key] if key in new_points else existing_points[key]
      for key in existing_order
  ] + [
      point
      for point in new_dps
      if delay_model_utils.get_data_point_key(point) not in existing_points
  ]


def update_data_points(
    output_file: str,
    new_dps: delay_model_pb2.DataPoints,
    update_mode: UpdateMode,
) -> delay_model_pb2.DataPoints:
  """Creates a proto of the data points in output_file updated with new_dps."""
  result_dps = delay_model_pb2.DataPoints()
  new_points_by_op = create_op_to_points_mapping(new_dps.data_points)
  ops_with_finished_updates = set()
  with gfile.open(output_file, 'r') as f:
    existing_dm = text_format.ParseLines(f, delay_model_pb2.DelayModel())
  old_points_by_op = create_op_to_points_mapping(existing_dm.data_points)
  for point in existing_dm.data_points:
    op = point.operation.op
    if op in new_points_by_op:
      result_dps.data_points.extend(
          update_op_data_points(old_points_by_op[op], new_points_by_op[op])
          if update_mode == UpdateMode.UPDATE_POINTS
          else new_points_by_op[op]
      )
      del new_points_by_op[op]
      ops_with_finished_updates.add(op)
    elif op not in ops_with_finished_updates:
      result_dps.data_points.append(point)
  for op in new_points_by_op:
    result_dps.data_points.extend(new_points_by_op[op])

  return result_dps


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

  if _UPDATE_MODE.value != UpdateMode.NONE:
    dm.data_points.extend(
        update_data_points(_OUTPUT.value, dps, _UPDATE_MODE.value).data_points
    )
  else:
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

  sync_check(oms, dps, _UPDATE_MODE.value)


if __name__ == '__main__':
  app.run(main)
