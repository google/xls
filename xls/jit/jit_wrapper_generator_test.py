# Copyright 2026 The XLS Authors
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

from absl.testing import absltest
from xls.ir import xls_type_pb2 as type_pb2
from xls.jit import jit_wrapper_generator


class JitWrapperGeneratorIsFloatTest(absltest.TestCase):

  def test_double_tuple(self):
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    u11 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=11)
    u52 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=52)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u1, u11, u52]
    )
    self.assertTrue(jit_wrapper_generator.is_double_tuple(tup))

  def test_double_tuple_fail(self):
    u11 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=11)
    u52 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=52)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u11, u52]
    )
    self.assertFalse(jit_wrapper_generator.is_double_tuple(tup))

  def test_float_tuple(self):
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u23 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=23)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u1, u8, u23]
    )
    self.assertTrue(jit_wrapper_generator.is_float_tuple(tup))

  def test_float_tuple_fail(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u23 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=23)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u23]
    )
    self.assertFalse(jit_wrapper_generator.is_float_tuple(tup))


class JitWrapperGeneratorToCTypeTest(absltest.TestCase):

  def test_bits(self):
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    self.assertEqual(jit_wrapper_generator.to_c_type(u1), 'uint8_t')
    u2 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=2)
    self.assertEqual(jit_wrapper_generator.to_c_type(u2), 'uint8_t')
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    self.assertEqual(jit_wrapper_generator.to_c_type(u8), 'uint8_t')
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    self.assertEqual(jit_wrapper_generator.to_c_type(u32), 'uint32_t')
    u64 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=64)
    self.assertEqual(jit_wrapper_generator.to_c_type(u64), 'uint64_t')

  def test_float_tuple(self):
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    u11 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=11)
    u52 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=52)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u1, u11, u52]
    )
    # It doesn't interpret it as a float.
    self.assertEqual(
        jit_wrapper_generator.to_c_type(tup),
        'std::tuple<uint8_t, uint16_t, uint64_t>',
    )

  def test_array(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    a8 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=4, array_element=u8
    )
    self.assertEqual(
        jit_wrapper_generator.to_c_type(a8), 'std::array<uint8_t, 4>'
    )

  def test_tuple(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u32]
    )
    self.assertEqual(
        jit_wrapper_generator.to_c_type(tup), 'std::tuple<uint8_t, uint32_t>'
    )

  def test_array_of_tuples(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u32]
    )
    arr = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=3, array_element=tup
    )
    self.assertEqual(
        jit_wrapper_generator.to_c_type(arr),
        'std::array<std::tuple<uint8_t, uint32_t>, 3>',
    )

  def test_nested_tuple(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    inner_tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u32]
    )
    u64 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=64)
    outer_tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[inner_tup, u64]
    )
    self.assertEqual(
        jit_wrapper_generator.to_c_type(outer_tup),
        'std::tuple<std::tuple<uint8_t, uint32_t>, uint64_t>',
    )


if __name__ == '__main__':
  absltest.main()
