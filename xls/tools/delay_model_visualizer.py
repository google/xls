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

r"""Delay model visualization tool.

Dumps a graph (as an image) of each XLS op delay model in a specified directory.

Usage:
  delay_model_visualizer --output_dir=/tmp/images \
    xls/estimators/delay_model/models/unit.textproto
"""

import os.path

from typing import Text, Optional

from absl import app
from absl import flags

from matplotlib import pyplot
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
import numpy as np

from google.protobuf import text_format
from xls.estimators.delay_model import delay_model
from xls.estimators.delay_model import delay_model_pb2

flags.DEFINE_string(
    'output_dir', None, 'The directory to write image files into.'
)
flags.mark_flag_as_required('output_dir')

FLAGS = flags.FLAGS

# Choosing figsizes that do not cutoff graph details
_2D_GRAPH_FIGSIZE = (8.0, 5.0)
_3D_GRAPH_FIGSIZE = (8.0, 7.0)


def get_r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.float64:
  residual_sum_of_squares = np.sum(np.square(y_pred - y_true))
  total_sum_of_squares = np.sum(np.square(y_true - np.mean(y_true)))
  if total_sum_of_squares == 0:
    return np.float64(1.0)
  return 1.0 - residual_sum_of_squares / total_sum_of_squares


def maybe_plot_estimator_and_data_points(
    estimator: delay_model.Estimator,
    specialization_kind: Optional[Text] = None,
    specialization_details: Optional[delay_model.SpecializationDetails] = None,
):
  """Plots the given delay model and writes the figure to a file.

  Only plots one-factor (2D plot) and two-factor (3D plot) regression and
  bounding box models.

  Args:
    estimator: Estimator to plot. Estimator class also contains data points.
    specialization_kind: Optional kind of specialization. Used in plot title and
      file name.
    specialization_details: Optional specialization details. Used in plot title.
  """
  if not isinstance(
      estimator, delay_model.RegressionEstimator
  ) and not isinstance(estimator, delay_model.BoundingBoxEstimator):
    return

  def delay_f(*args):
    try:
      return estimator.raw_delay(args)
    except delay_model.Error:
      return 0

  title = estimator.op
  if specialization_kind:
    title += ' ' + specialization_kind
  if specialization_details:
    title += ' ' + str(specialization_details)

  coeffs = estimator.params

  if len(estimator.delay_expressions) == 1:
    fig, ax = pyplot.subplots(figsize=_2D_GRAPH_FIGSIZE)

    # Plot the real data points as circles.
    x_actual = [dp.delay_factors[0] for dp in estimator.raw_data_points]
    y_actual = [dp.delay_ps for dp in estimator.raw_data_points]
    ax.plot(x_actual, y_actual, 'o')

    # Compute the R^2 score and add the score to the graph
    r2_score = get_r2_score(np.vectorize(delay_f)(x_actual), y_actual)
    title += f', $R^2$ = {r2_score:.2f}'

    # Describe the equation
    estimator_description = (
        f'f(x) = {coeffs[0]:.2f} + {coeffs[1]:.2f} * x + {coeffs[2]:.2f} *'
        ' log(x)\nwhere x ='
        f' {delay_model.delay_expression_description(estimator.delay_expressions[0])}'
    )
    title += '\n' + estimator_description

    # Plot a curve for the delay model.
    x_range = np.linspace(1, max(x_actual), num=100)
    y_est = np.vectorize(delay_f)(x_range)
    ax.plot(x_range, y_est)

    pyplot.title(title)
    ax.set_xlabel(
        delay_model.delay_expression_description(estimator.delay_expressions[0])
    )
    ax.set_ylabel('delay (ps)')
    pyplot.ylim(bottom=0)
    pyplot.xlim(left=1)

  elif len(estimator.delay_expressions) == 2:
    x_actual = [dp.delay_factors[0] for dp in estimator.raw_data_points]
    y_actual = [dp.delay_factors[1] for dp in estimator.raw_data_points]
    z_actual = [dp.delay_ps for dp in estimator.raw_data_points]
    fig = pyplot.figure(figsize=_3D_GRAPH_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Compute the R^2 score and add the score to the graph
    r2_score = get_r2_score(np.vectorize(delay_f)(x_actual, y_actual), z_actual)
    title += f', $R^2$ = {r2_score:.2f}'

    # Describe the equation
    estimator_description = (
        f'f(x,y) = {coeffs[0]:.2f} + {coeffs[1]:.2f} * x + {coeffs[2]:.2f} *'
        f' log(x) + {coeffs[3]:.2f} * y + {coeffs[4]:.2f} * log(y)\nwhere x ='
        f' {delay_model.delay_expression_description(estimator.delay_expressions[0])}\ny'
        f' = {delay_model.delay_expression_description(estimator.delay_expressions[1])}'
    )
    title += '\n' + estimator_description

    # Plot the surface of the delay estimate.
    x_range, y_range = np.meshgrid(
        np.arange(1, max(x_actual), 1), np.arange(1, max(y_actual), 1)
    )
    z_est = np.vectorize(delay_f)(x_range, y_range)
    surf = ax.plot_surface(
        x_range,
        y_range,
        z_est,
        rstride=10,
        cstride=1,
        cmap=pyplot.get_cmap('coolwarm'),
        linewidth=0,
        antialiased=False,
        alpha=0.25,
    )
    ax.set_zlim(min(0, min(z_actual)), max(z_actual))
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # Plot the actual data points as circles with a line extending from the
    # model estimate.
    for rdp in estimator.raw_data_points:
      x_i, y_i = rdp.delay_factors[0:2]
      z_i = rdp.delay_ps
      z_est_i = delay_f(x_i, y_i)
      ax.scatter(x_i, y_i, z_i, marker='o', c='black')
      ax.plot([x_i, x_i], [y_i, y_i], [z_est_i, z_i], color='black', marker='_')

    pyplot.title(title)
    ax.set_xlabel(
        delay_model.delay_expression_description(estimator.delay_expressions[0])
    )
    ax.set_ylabel(
        delay_model.delay_expression_description(estimator.delay_expressions[1])
    )
    ax.set_zlabel('delay (ps)')

  else:
    # More than two delay expressions not supported.
    return

  if specialization_kind:
    filename = f'{estimator.op}_{specialization_kind}.png'
  else:
    filename = f'{estimator.op}.png'

  fig.savefig(os.path.join(FLAGS.output_dir, filename))
  pyplot.close(fig)


def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')

  with open(argv[1], 'rb') as f:
    contents = f.read()

  dm = delay_model.DelayModel(
      text_format.Parse(contents, delay_model_pb2.DelayModel())
  )

  for op in dm.ops():
    op_model = dm.op_model(op)
    maybe_plot_estimator_and_data_points(op_model.estimator)

    for (
        specialization_kind,
        specialization_details,
    ), estimator in op_model.specializations.items():
      maybe_plot_estimator_and_data_points(
          estimator,
          delay_model_pb2.SpecializationKind.Name(specialization_kind),
          specialization_details,
      )


if __name__ == '__main__':
  app.run(main)
