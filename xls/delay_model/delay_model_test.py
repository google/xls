# Copyright 2020 Google LLC
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
#
# Lint as: python3

"""Tests for xls.delay_model.models.delay_model."""

import re
from typing import Text

from google.protobuf import text_format
from xls.delay_model import delay_model
from xls.delay_model import delay_model_pb2
from absl.testing import absltest


def _parse_data_point(s: Text) -> delay_model_pb2.DataPoint:
  """Parses a text proto representation of a DataPoint."""
  return text_format.Parse(s, delay_model_pb2.DataPoint())


def _parse_operation(s: Text) -> delay_model_pb2.Operation:
  """Parses a text proto representation of an Operation."""
  return text_format.Parse(s, delay_model_pb2.Operation())


class DelayModelTest(absltest.TestCase):

  def assertEqualIgnoringWhitespace(self, a: Text, b: Text):
    """Asserts the two strings are equal ignoring all whitespace."""

    def _strip_ws(s: Text) -> Text:
      return re.sub(r'\s+', '', s.strip())

    self.assertEqual(_strip_ws(a), _strip_ws(b))

  def assertEqualIgnoringWhitespaceAndFloats(self, a: Text, b: Text):
    """Asserts the two strings are equal ignoring whitespace and floats.

    Floats are replaced with 0.0.

    Args:
      a: Input string.
      b: Input string.
    """

    def _floats_to_zero(s: Text) -> Text:
      return re.sub(r'-?[0-9]+\.[0-9e+-]+', '0.0', s)

    self.assertEqualIgnoringWhitespace(_floats_to_zero(a), _floats_to_zero(b))

  def test_fixed_estimator(self):
    foo = delay_model.FixedEstimator('kFoo', 42)
    self.assertEqual(
        foo.operation_delay(_parse_operation('op: "kBar" bit_count: 1')), 42)
    self.assertEqual(
        foo.operation_delay(_parse_operation('op: "kBar" bit_count: 123')), 42)
    self.assertEqual(foo.cpp_delay_code('node'), 'return 42;')

  def test_alias_estimator(self):
    foo = delay_model.AliasEstimator('kFoo', 'kBar')
    self.assertEqual(foo.cpp_delay_code('node'), """return BarDelay(node);""")

  def test_bounding_box_estimator(self):
    data_points_str = [
        'operation { op: "kBar" bit_count: 3 operands { bit_count: 7 } } ' +
        'delay: 33 delay_offset: 10',
        'operation { op: "kBar" bit_count: 12 operands { bit_count: 42 } }' +
        'delay: 100 delay_offset: 0',
        'operation { op: "kBar" bit_count: 32 operands { bit_count: 10 } }' +
        'delay: 123 delay_offset: 1',
        'operation { op: "kBar" bit_count: 64 operands { bit_count: 64 } }' +
        'delay: 1234 delay_offset: 0',
    ]
    result_bit_count = delay_model_pb2.DelayFactor()
    result_bit_count.source = delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    operand0_bit_count = delay_model_pb2.DelayFactor()
    operand0_bit_count.source = delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    operand0_bit_count.operand_number = 0
    bar = delay_model.BoundingBoxEstimator(
        'kBar', (result_bit_count, operand0_bit_count),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 1 operands { bit_count: 2 }')), 23)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 10 operands { bit_count: 32 }')), 100)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 32 operands { bit_count: 9 }')), 122)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 32 operands { bit_count: 33 }')), 1234)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 64 operands { bit_count: 64 }')), 1234)
    self.assertEqualIgnoringWhitespace(
        bar.cpp_delay_code('node'), """
          if (node->GetType()->GetFlatBitCount() <= 3 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 7) {
            return 23;
          }
          if (node->GetType()->GetFlatBitCount() <= 12 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 42) {
            return 100;
          }
          if (node->GetType()->GetFlatBitCount() <= 32 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 10) {
            return 122;
          }
          if (node->GetType()->GetFlatBitCount() <= 64 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 64) {
            return 1234;
          }
          return absl::UnimplementedError(
              "Unhandled node for delay estimation: " +
              node->ToStringWithOperandTypes());
        """)
    with self.assertRaises(delay_model.Error) as e:
      bar.operation_delay(
          _parse_operation(
              'op: "kBar" bit_count: 65 operands { bit_count: 64 }'))
    self.assertIn('Operation outside bounding box', str(e.exception))

  def test_one_factor_regression_estimator(self):
    data_points_str = [
        'operation { op: "kFoo" bit_count: 2 } delay: 210 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 4 } delay: 410 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 6 } delay: 610 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 8 } delay: 810 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 10 } delay: 1010 delay_offset: 10',
    ]
    result_bit_count = delay_model_pb2.DelayFactor()
    result_bit_count.source = delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    foo = delay_model.RegressionEstimator(
        'kFoo', (result_bit_count,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 2')),
        200,
        delta=2)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 3')),
        300,
        delta=2)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 5')),
        500,
        delta=2)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 42')),
        4200,
        delta=2)
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * node->GetType()->GetFlatBitCount() +
              0.0 * std::log2(node->GetType()->GetFlatBitCount()));
        """)

  def test_one_regression_estimator_operand_count(self):

    def gen_operation(operand_count):
      operands_str = 'operands { bit_count: 42 }'
      return 'op: "kFoo" bit_count: 42 %s' % ' '.join(
          [operands_str] * operand_count)

    data_points_str = [
        'operation { %s } delay: 10 delay_offset: 0' % gen_operation(1),
        'operation { %s } delay: 11 delay_offset: 0' % gen_operation(2),
        'operation { %s } delay: 12 delay_offset: 0' % gen_operation(4),
        'operation { %s } delay: 13 delay_offset: 0' % gen_operation(8),
    ]
    result_bit_count = delay_model_pb2.DelayFactor()
    result_bit_count.source = delay_model_pb2.DelayFactor.Source.OPERAND_COUNT
    foo = delay_model.RegressionEstimator(
        'kFoo', (result_bit_count,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(1))), 10, delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(4))), 12, delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(256))), 18, delta=1)
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * node->operand_count() +
              0.0 * std::log2(node->operand_count()));
        """)

  def test_two_factor_regression_estimator(self):

    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 100 delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 125 delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 150 delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 175 delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 200 delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 400 delay_offset: 0' % gen_operation(30, 15),
    ]
    result_bit_count = delay_model_pb2.DelayFactor()
    result_bit_count.source = delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    operand_bit_count = delay_model_pb2.DelayFactor()
    operand_bit_count.source = delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    operand_bit_count.operand_number = 1
    foo = delay_model.RegressionEstimator(
        'kFoo', (result_bit_count, operand_bit_count),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(1, 2))),
        100,
        delta=10)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(10, 12))),
        200,
        delta=10)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(8, 8))),
        200,
        delta=50)
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * node->GetType()->GetFlatBitCount() +
              0.0 * std::log2(node->GetType()->GetFlatBitCount()) +
              0.0 * node->operand(1)->GetType()->GetFlatBitCount() +
              0.0 * std::log2(node->operand(1)->GetType()->GetFlatBitCount()));
        """)

  def test_fixed_op_model(self):
    op_model = delay_model.OpModel(
        text_format.Parse('op: "kFoo" estimator { fixed: 42 }',
                          delay_model_pb2.OpModel()), ())
    self.assertEqual(op_model.op, 'kFoo')
    self.assertEqual(
        op_model.estimator.operation_delay(
            _parse_operation('op: "kBar" bit_count: 123')), 42)
    self.assertEqualIgnoringWhitespace(
        op_model.cpp_delay_function(),
        """absl::StatusOr<int64> FooDelay(Node* node) {
             return 42;
           }""")

  def test_fixed_op_model_with_specialization(self):
    op_model = delay_model.OpModel(
        text_format.Parse(
            'op: "kFoo" estimator { fixed: 42 } '
            'specializations { kind: OPERANDS_IDENTICAL estimator { fixed: 123 } }',
            delay_model_pb2.OpModel()), ())
    self.assertEqual(op_model.op, 'kFoo')
    self.assertEqual(
        op_model.estimator.operation_delay(
            _parse_operation('op: "kBar" bit_count: 123')), 42)
    self.assertEqual(
        op_model.specializations[
            delay_model_pb2.SpecializationKind.OPERANDS_IDENTICAL]
        .operation_delay(_parse_operation('op: "kBar" bit_count: 123')), 123)
    self.assertEqualIgnoringWhitespace(
        op_model.cpp_delay_function(), """
          absl::StatusOr<int64> FooDelay(Node* node) {
            if (std::all_of(node->operands().begin(), node->operands().end(),
                [&](Node* n) { return n == node->operand(0); })) {
              return 123;
            }
            return 42;
          }
        """)

  def test_regression_op_model_with_bounding_box_specialization(self):

    def gen_data_point(bit_count, delay, specialization=''):
      return _parse_data_point(
          'operation { op: "kFoo" bit_count: %d %s} delay: %d delay_offset: 0' %
          (bit_count, specialization, delay))

    op_model = delay_model.OpModel(
        text_format.Parse(
            'op: "kFoo" estimator { regression { factors { source: RESULT_BIT_COUNT } } }'
            'specializations { kind: OPERANDS_IDENTICAL '
            'estimator { bounding_box { factors { source: RESULT_BIT_COUNT } } } }',
            delay_model_pb2.OpModel()),
        [gen_data_point(bc, 10 * bc) for bc in range(1, 10)] + [
            gen_data_point(bc, 2 * bc, 'specialization: OPERANDS_IDENTICAL')
            for bc in range(1, 3)
        ])
    self.assertEqual(op_model.op, 'kFoo')
    self.assertEqualIgnoringWhitespaceAndFloats(
        op_model.cpp_delay_function(), """
          absl::StatusOr<int64> FooDelay(Node* node) {
            if (std::all_of(node->operands().begin(), node->operands().end(),
                [&](Node* n) { return n == node->operand(0); })) {
              if (node->GetType()->GetFlatBitCount() <= 1) { return 2; }
              if (node->GetType()->GetFlatBitCount() <= 2) { return 4; }
              return absl::UnimplementedError(
                "Unhandled node for delay estimation: " +
                node->ToStringWithOperandTypes());
            }
            return std::round(
                0.0 + 0.0 * node->GetType()->GetFlatBitCount() +
                0.0 * std::log2(node->GetType()->GetFlatBitCount()));
          }
        """)


if __name__ == '__main__':
  absltest.main()
