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
"""Utility functions used in delay model generation."""

from collections.abc import Mapping
import dataclasses
import math
from typing import Sequence

from xls.estimators.delay_model import delay_model_pb2


# A `Parameterization` proto message along with the containing `OpSamples`
# message. This represents a request for the collection of one data point.
@dataclasses.dataclass
class SampleSpec(object):
  op_samples: delay_model_pb2.OpSamples
  point: delay_model_pb2.Parameterization


def map_data_points_by_key(
    points: Sequence[delay_model_pb2.DataPoint],
) -> Mapping[str, delay_model_pb2.DataPoint]:
  """Converts a bare sequence of data point protos to a mapping by data point key."""
  return {get_data_point_key(point): point for point in points}


def get_data_point_key(data_point: delay_model_pb2.DataPoint) -> str:
  """Returns a key that can be used to represent a data point in a dict."""
  return get_sample_spec_key(data_point_to_sample_spec(data_point))


def get_sample_spec_key(spec: SampleSpec) -> str:
  """Returns a key that can be used to represent a sample spec in a dict."""
  # This dictionary stores the element counts for array operands. For non-array
  # operands, the element count is 0.
  array_operand_element_counts = {
      e.operand_number: math.prod(e.element_counts)
      for e in spec.point.operand_element_counts
  }
  bit_count_strs = []
  for operand_idx, operand_width in enumerate(spec.point.operand_widths):
    operand = delay_model_pb2.Operation.Operand(
        bit_count=operand_width,
        element_count=array_operand_element_counts.get(operand_idx, 0),
    )
    bit_count_strs.append(str(operand))
  key = (
      spec.op_samples.op
      + ': '
      + ', '.join([str(spec.point.result_width)] + bit_count_strs)
  )
  if spec.op_samples.specialization:
    key = key + ' ' + str(spec.op_samples.specialization)
  return key


def data_point_to_sample_spec(
    data_point: delay_model_pb2.DataPoint,
) -> SampleSpec:
  """Converts a data point to the sample spec describing it."""
  op_samples = delay_model_pb2.OpSamples()
  op_samples.op = data_point.operation.op
  if data_point.operation.specialization:
    op_samples.specialization = data_point.operation.specialization
  point = delay_model_pb2.Parameterization()
  point.result_width = data_point.operation.bit_count
  point.operand_widths.extend(
      [operand.bit_count for operand in data_point.operation.operands]
  )
  for operand_idx, operand in enumerate(data_point.operation.operands):
    if operand.element_count:
      point.operand_element_counts.append(
          delay_model_pb2.OperandElementCounts(
              operand_number=operand_idx,
              element_counts=[operand.element_count],
          )
      )
  return SampleSpec(op_samples, point)
