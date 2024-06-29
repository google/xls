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
    xls/delay_model/models/unit.textproto
"""

import os.path

from typing import Text, Optional

from absl import app
from absl import flags

from matplotlib import pyplot
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
import numpy as np

from google.protobuf import text_format
from xls.delay_model import delay_model
from xls.delay_model import delay_model_pb2

flags.DEFINE_string(
    'output_dir', None, 'The directory to write image files into.'
)
flags.mark_flag_as_required('output_dir')

FLAGS = flags.FLAGS


def maybe_plot_op_model(
    op_model: delay_model.OpModel, specialization_kind: Optional[Text] = None
):
  """Plots the given delay model and writes the figure to a file.

  Only plots one-factor (2D plot) and two-factor (3D plot) regression and
  bounding box models.

  Args:
    op_model: OpModel to plot.
    specialization_kind: Optional kind of specialization. Used in plot title and
      file name.
  """
  if not isinstance(
      op_model, delay_model.RegressionEstimator
  ) and not isinstance(op_model, delay_model.BoundingBoxEstimator):
    return

  def delay_f(*args):
    try:
      return op_model.raw_delay(args)
    except delay_model.Error:
      return 0

  title = op_model.op
  if specialization_kind:
    title += ' ' + specialization_kind

  if len(op_model.delay_expressions) == 1:
    fig, ax = pyplot.subplots()

    # Plot the real data points as circles.
    x_actual = [dp.delay_factors[0] for dp in op_model.raw_data_points]
    y_actual = [dp.delay_ps for dp in op_model.raw_data_points]
    ax.plot(x_actual, y_actual, 'o')

    # Plot a curve for the delay model.
    x_range = np.linspace(1, max(x_actual), num=100)
    y_est = np.vectorize(delay_f)(x_range)
    ax.plot(x_range, y_est)

    pyplot.title(title)
    ax.set_xlabel(
        delay_model.delay_expression_description(op_model.delay_expressions[0])
    )
    ax.set_ylabel('delay (ps)')
    pyplot.ylim(bottom=0)
    pyplot.xlim(left=1)

  elif len(op_model.delay_expressions) == 2:
    x_actual = [dp.delay_factors[0] for dp in op_model.raw_data_points]
    y_actual = [dp.delay_factors[1] for dp in op_model.raw_data_points]
    z_actual = [dp.delay_ps for dp in op_model.raw_data_points]
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

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
    for rdp in op_model.raw_data_points:
      x_i, y_i = rdp.delay_factors[0:2]
      z_i = rdp.delay_ps
      z_est_i = delay_f(x_i, y_i)
      ax.scatter(x_i, y_i, z_i, marker='o', c='black')
      ax.plot([x_i, x_i], [y_i, y_i], [z_est_i, z_i], color='black', marker='_')

    pyplot.title(title)
    ax.set_xlabel(
        delay_model.delay_expression_description(op_model.delay_expressions[0])
    )
    ax.set_ylabel(
        delay_model.delay_expression_description(op_model.delay_expressions[1])
    )
    ax.set_zlabel('delay (ps)')

  else:
    # More than two delay expressions not supported.
    return

  if specialization_kind:
    filename = '%s_%s.png' % (op_model.op, specialization_kind)
  else:
    filename = '%s.png' % op_model.op

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
    maybe_plot_op_model(op_model.estimator)

    for specialization_kind, estimator in op_model.specializations.items():
      maybe_plot_op_model(
          estimator,
          delay_model_pb2.SpecializationKind.Name(specialization_kind),
      )


if __name__ == '__main__':
  app.run(main)
