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

r"""Calculate error statistics for a delay model and write to CSV spreadsheet.

Summarizes the difference between the regression-fitted curve and the
actual data points.

Write one row per XLS op type, containing the op's max percent error,
mean absolute percent error, mean percent error (bias),
and root mean squared error.

Sample usage:
  delay_model_stats \
    [--output_csv stats.csv] \
    xls/estimators/delay_model/models/sky130.textproto
"""

from typing import Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np

from google.protobuf import text_format
from xls.estimators import estimator_model_pb2
from xls.estimators.delay_model import delay_model

_CSV = flags.DEFINE_string(
    'output_csv', None, 'The file to write statistics into.'
)


def stats_for_op_model(
    csv_handle,
    op_model: delay_model.OpModel,
    specialization_kind: Optional[str] = None,
    specialization_details: Optional[delay_model.SpecializationDetails] = None,
) -> None:
  """Write one row of a CSV spreadsheet with this op's curve fitting error.

  Summarizes the difference between the regression-fitted curve and the
  actual data points.

  Writes one row containing the op's max percent error,
  mean absolute percent error, mean percent error (bias),
  and root mean squared error.

  Args:
    csv_handle: file handle for output CSV (can be None).
    op_model: OpModel to summarize.
    specialization_kind: Optional kind of specialization.
    specialization_details: Optional details about the specified specialization.
  """

  if not isinstance(
      op_model, delay_model.RegressionEstimator
  ) and not isinstance(op_model, delay_model.BoundingBoxEstimator):
    return

  if specialization_details and not specialization_kind:
    raise ValueError(
        'specialization_kind must be specified when using '
        'specialization_details'
    )

  def delay_f(*args):
    return op_model.raw_delay(args)

  title = op_model.op
  if specialization_kind:
    title += ' ' + specialization_kind

  if specialization_details:
    title += ' ' + str(specialization_details)

  for dp in op_model.raw_data_points:
    x_actual = dp.delay_factors
    y_actual = dp.delay_ps

    y_est = delay_f(*x_actual)
    y_delta = y_actual - y_est
    if y_actual > 0:
      y_delta_pct = round(y_delta * 100.0 / y_actual, 2)
    else:
      y_delta_pct = 'NaN'
    logging.vlog(
        1,
        f'x: {x_actual},  actual delay: {y_actual},  '
        f'estimated delay: {round(y_est,2)},  '
        f'delta: {round(y_delta,2)}, '
        f'{y_delta_pct}%',
    )

  delays = [dp.delay_ps for dp in op_model.raw_data_points]
  if min(delays) > 0:
    max_pct_error = max([
        abs((dp.delay_ps - delay_f(*dp.delay_factors)) * 100.0 / dp.delay_ps)
        for dp in op_model.raw_data_points
    ])
    mean_abs_pct_error = np.mean([
        abs((dp.delay_ps - delay_f(*dp.delay_factors)) * 100.0 / dp.delay_ps)
        for dp in op_model.raw_data_points
    ])
    mean_pct_error = np.mean([
        ((dp.delay_ps - delay_f(*dp.delay_factors)) * 100.0 / dp.delay_ps)
        for dp in op_model.raw_data_points
    ])
    mean_square_error = np.mean([
        ((dp.delay_ps - delay_f(*dp.delay_factors)) ** 2)
        for dp in op_model.raw_data_points
    ])

    # Round and add 0. to switch -0.0 values to 0.0.
    max_pct_error = round(max_pct_error, 2) + 0.0
    mean_abs_pct_error = round(mean_abs_pct_error, 3) + 0.0
    mean_pct_error = round(mean_pct_error, 3) + 0.0
    mean_square_error = round(mean_square_error**0.5, 2) + 0.0

    csv_line = (
        f'{title}, {max_pct_error}, {mean_abs_pct_error}, {mean_pct_error}, '
        f'{mean_square_error}'
    )
    print(csv_line)
    if csv_handle:
      csv_handle.write(csv_line + '\n')
  else:
    logging.vlog(1, f'Zero delay detected for {op_model.op}.')


def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')

  with open(argv[1], 'rb') as f:
    contents = f.read()

  if _CSV.value:
    csv_handle = open(_CSV.value, 'wt')
  else:
    csv_handle = None

  dm = delay_model.DelayModel(
      text_format.Parse(contents, estimator_model_pb2.EstimatorModel())
  )

  # Print column headers.  Add a blank row to make sorting easier.
  print(
      'Operation type, max % error, mean abs % error, '
      'mean % error (bias), root mean sq error (ps)\n'
  )
  if csv_handle:
    csv_handle.write(
        'Operation type, max % error, mean abs % error, '
        'mean % error (bias), root mean sq error (ps)\n\n'
    )

  for op in dm.ops():
    op_model = dm.op_model(op)
    stats_for_op_model(csv_handle, op_model.estimator)

    for (
        specialization_kind,
        specialization_details,
    ), estimator in op_model.specializations.items():
      stats_for_op_model(
          csv_handle,
          estimator,
          estimator_model_pb2.SpecializationKind.Name(specialization_kind),
          specialization_details,
      )

  if csv_handle:
    csv_handle.close()


if __name__ == '__main__':
  app.run(main)
