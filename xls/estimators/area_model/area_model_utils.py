# Copyright 2024 The XLS Authors
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
"""Utility functions used in area model generation."""

from xls.estimators import estimator_model


def get_one_bit_register_area(
    area_model: estimator_model.EstimatorModel,
) -> float:
  """Getting one-bit register area from an area model.

  If kIdentity has some data points (thus, not a fixed estimator), then,
      one_bit_register_area = sequential_area /
                                   (total_input_bits + total_output_bits)
  If kIdentity is a fixed estimator, we think that the area model is a unit area
  model, and one_bit_register_area = kIdentity.estimator.constant.

  Args:
    area_model: The area model.

  Returns:
    The area of one-bit register.

  Raises:
    ValueError: if the area model is not a unit area model or does not have
    any data points for kIdentity.
  """
  if area_model.get_metric() != estimator_model.Metric.AREA_METRIC:
    raise ValueError("Getting register area from non-area estimator!")

  one_bit_register_area = None
  # We can use any op data point sequential area to derive one-bit register area
  # by:
  # one_bit_register_area = sequential_area /
  #                                   (total_input_bits + total_output_bits)
  # However, for consistency, we always use kIdentity data point to derive the
  # one_bit_register_area. This is analogous to the use kIdentity in delay
  # models, which use kIdentity to derive delay_offset.
  for op_model in area_model.op_models.values():
    if op_model.op == "kIdentity":
      if isinstance(op_model.estimator, estimator_model.FixedEstimator):
        one_bit_register_area = op_model.estimator.constant
        break
      if not isinstance(
          op_model.estimator,
          (
              estimator_model.AreaRegressionEstimator,
              estimator_model.RegressionEstimator,
          ),
      ):
        break
      data_point = op_model.estimator.data_points[0]
      total_output_bits = data_point.operation.bit_count
      total_input_bits = 0
      for operand in data_point.operation.operands:
        if operand.element_count == 0:  # non-array input
          total_input_bits += operand.bit_count
        else:  # array input
          total_input_bits += operand.bit_count * operand.element_count
      one_bit_register_area = data_point.sequential_area / (
          total_input_bits + total_output_bits
      )
      break
  if not one_bit_register_area:
    raise ValueError(
        "Cannot find kIdentity data point to derive one-bit register area. To"
        " fix this, please add one kIdentity data point textproto."
    )
  return one_bit_register_area
