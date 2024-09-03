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

from absl.testing import parameterized
from xls.common import test_base
from xls.estimators.delay_model import delay_model_pb2
from xls.estimators.delay_model import delay_model_utils


def _create_sample_spec(
    op_name: str,
    result_width: int,
    operand_widths: list[int],
    specialization: delay_model_pb2.SpecializationKind = delay_model_pb2.SpecializationKind.NO_SPECIALIZATION,
) -> delay_model_utils.SampleSpec:
  """Convenience function for tests to create SampleSpec protos."""
  op_samples = delay_model_pb2.OpSamples()
  op_samples.op = op_name
  if specialization != delay_model_pb2.SpecializationKind.NO_SPECIALIZATION:
    op_samples.specialization = specialization
  point = delay_model_pb2.Parameterization()
  point.result_width = result_width
  point.operand_widths.extend(operand_widths)
  return delay_model_utils.SampleSpec(op_samples, point)


def _create_data_point(
    op_name: str,
    result_width: int,
    operand_widths: list[int],
    specialization: delay_model_pb2.SpecializationKind = delay_model_pb2.SpecializationKind.NO_SPECIALIZATION,
) -> delay_model_pb2.DataPoint:
  """Convenience function for tests to create DataPoint protos."""
  dp = delay_model_pb2.DataPoint()
  dp.operation.op = op_name
  if specialization != delay_model_pb2.SpecializationKind.NO_SPECIALIZATION:
    dp.operation.specialization = specialization
  dp.operation.bit_count = result_width
  for width in operand_widths:
    operand = delay_model_pb2.Operation.Operand()
    operand.bit_count = width
    dp.operation.operands.append(operand)
  return dp


class DelayModelUtilsTest(parameterized.TestCase):

  def test_sample_spec_key_equal_for_same_sample(self):
    self.assertEqual(
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 16, [8, 8])
        ),
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 16, [8, 8])
        ),
    )

  def test_sample_spec_key_differs_for_op(self):
    self.assertNotEqual(
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 16, [8, 8])
        ),
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kOr', 16, [8, 8])
        ),
    )

  def test_sample_spec_key_differs_for_result_width(self):
    self.assertNotEqual(
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 16, [8, 8])
        ),
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 8, [8, 8])
        ),
    )

  @parameterized.named_parameters(
      ('different_operand_counts', [8, 8, 8], [8, 8]),
      ('different_widths_same_count', [8, 8], [8, 4]),
  )
  def test_sample_spec_key_differs_for_operand_widths(
      self, left_operand_widths, right_operand_widths
  ):
    self.assertNotEqual(
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 8, left_operand_widths)
        ),
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 8, right_operand_widths)
        ),
    )

  @parameterized.named_parameters(
      (
          'no_specialization_vs_specialization',
          delay_model_pb2.SpecializationKind.NO_SPECIALIZATION,
          delay_model_pb2.SpecializationKind.OPERANDS_IDENTICAL,
      ),
      (
          'different_specializations',
          delay_model_pb2.SpecializationKind.OPERANDS_IDENTICAL,
          delay_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND,
      ),
  )
  def test_sample_spec_key_differs_for_specialization(
      self, left_specialization, right_specialization
  ):
    self.assertNotEqual(
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 4, [4], left_specialization)
        ),
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec('kAnd', 4, [4], right_specialization)
        ),
    )

  @parameterized.named_parameters(
      ('no_operand_widths', 'kOr', 32, []),
      ('no_specialization', 'kOr', 32, [16, 16]),
      (
          'specialization',
          'kOr',
          32,
          [16, 16],
          delay_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND,
      ),
  )
  def test_data_point_to_sample_spec(
      self,
      op_name,
      result_width,
      operand_widths,
      specialization=delay_model_pb2.SpecializationKind.NO_SPECIALIZATION,
  ):
    self.assertEqual(
        delay_model_utils.data_point_to_sample_spec(
            _create_data_point(
                op_name, result_width, operand_widths, specialization
            )
        ),
        _create_sample_spec(
            op_name, result_width, operand_widths, specialization
        ),
    )

  def test_get_data_point_key(self):
    self.assertEqual(
        delay_model_utils.get_data_point_key(
            _create_data_point(
                'kOr',
                32,
                [16, 16],
                delay_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND,
            )
        ),
        delay_model_utils.get_sample_spec_key(
            _create_sample_spec(
                'kOr',
                32,
                [16, 16],
                delay_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND,
            )
        ),
    )

  def test_map_data_points_by_key(self):
    point1 = _create_data_point('kOr', 32, [])
    point2 = _create_data_point('kOr', 32, [16, 16])
    point3 = _create_data_point(
        'kAdd',
        32,
        [16, 16],
        delay_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND,
    )
    mapping = delay_model_utils.map_data_points_by_key([point1, point2, point3])
    self.assertLen(mapping, 3)
    self.assertEqual(
        mapping[delay_model_utils.get_data_point_key(point1)], point1
    )
    self.assertEqual(
        mapping[delay_model_utils.get_data_point_key(point2)], point2
    )
    self.assertEqual(
        mapping[delay_model_utils.get_data_point_key(point3)], point3
    )


if __name__ == '__main__':
  test_base.main()
