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


class JitWrapperGeneratorConvertToFuzztestParamTest(absltest.TestCase):

  def test_bits(self):
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    param1 = jit_wrapper_generator.convert_to_fuzztest_param('u1', u1)
    self.assertEqual(param1.name, 'u1')
    self.assertEqual(param1.value_name, 'u1_value')
    self.assertEqual(param1.value_type, 'BITS')
    self.assertEqual(param1.c_type, 'uint8_t')
    self.assertEqual(param1.domain, 'fuzztest::InRange(0, 1)')

    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    param8 = jit_wrapper_generator.convert_to_fuzztest_param('u8', u8)
    self.assertEqual(param8.name, 'u8')
    self.assertEqual(param8.c_type, 'uint8_t')
    self.assertEqual(param8.domain, 'fuzztest::Arbitrary<uint8_t>()')

    u11 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=11)
    param11 = jit_wrapper_generator.convert_to_fuzztest_param('u11', u11)
    self.assertEqual(param11.name, 'u11')
    self.assertEqual(param11.c_type, 'uint16_t')
    self.assertEqual(param11.domain, 'fuzztest::InRange(0, 2047)')

    u64 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=64)
    param64 = jit_wrapper_generator.convert_to_fuzztest_param('u64', u64)
    self.assertEqual(param64.name, 'u64')
    self.assertEqual(param64.c_type, 'uint64_t')
    self.assertEqual(param64.domain, 'fuzztest::Arbitrary<uint64_t>()')

  def test_array_of_bits(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    a8 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=4, array_element=u8
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param('a8', a8)
    self.assertEqual(param.name, 'a8')
    self.assertEqual(param.value_name, 'a8_value')
    self.assertEqual(param.value_type, 'ARRAY')
    self.assertEqual(param.c_type, 'std::array<uint8_t, 4>')
    self.assertEqual(
        param.domain,
        'fuzztest::ArrayOf<4>(fuzztest::Arbitrary<uint8_t>())',
    )
    self.assertLen(param.children, 1)
    child = param.children[0]
    self.assertEqual(child.name, '_a8_element')
    self.assertEqual(child.value_name, '_a8_element_value')
    self.assertEqual(child.value_type, 'BITS')
    self.assertEqual(child.c_type, 'uint8_t')
    self.assertEqual(child.domain, 'fuzztest::Arbitrary<uint8_t>()')

  def test_array_of_5_bits(self):
    u5 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=5)
    a5 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=3, array_element=u5
    )
    param5 = jit_wrapper_generator.convert_to_fuzztest_param('a5', a5)
    self.assertEqual(param5.name, 'a5')
    self.assertEqual(param5.value_name, 'a5_value')
    self.assertEqual(param5.value_type, 'ARRAY')
    self.assertEqual(param5.c_type, 'std::array<uint8_t, 3>')
    self.assertEqual(
        param5.domain,
        'fuzztest::ArrayOf<3>(fuzztest::InRange(0, 31))',
    )
    self.assertLen(param5.children, 1)
    child5 = param5.children[0]
    self.assertEqual(child5.name, '_a5_element')
    self.assertEqual(child5.num_bits, 5)
    self.assertEqual(child5.c_type, 'uint8_t')

  def test_array_of_tuples(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u32]
    )
    arr = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=3, array_element=tup
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param('arr', arr)
    self.assertEqual(param.name, 'arr')
    self.assertEqual(param.value_name, 'arr_value')
    self.assertEqual(param.value_type, 'ARRAY')
    self.assertEqual(
        param.c_type, 'std::array<std::tuple<uint8_t, uint32_t>, 3>'
    )
    self.assertEqual(
        param.domain,
        'fuzztest::ArrayOf<3>(fuzztest::TupleOf('
        'fuzztest::Arbitrary<uint8_t>(), fuzztest::Arbitrary<uint32_t>()))',
    )
    self.assertLen(param.children, 1)
    child = param.children[0]
    self.assertEqual(child.name, '_arr_element')
    self.assertEqual(child.value_name, '_arr_element_value')
    self.assertEqual(child.value_type, 'TUPLE')
    self.assertEqual(child.c_type, 'std::tuple<uint8_t, uint32_t>')
    self.assertEqual(
        child.domain,
        'fuzztest::TupleOf(fuzztest::Arbitrary<uint8_t>(),'
        ' fuzztest::Arbitrary<uint32_t>())',
    )
    self.assertLen(child.children, 2)
    c0 = child.children[0]
    self.assertEqual(c0.name, '__arr_element_0')
    self.assertEqual(c0.value_type, 'BITS')
    self.assertEqual(c0.c_type, 'uint8_t')
    self.assertEqual(c0.value_type, 'BITS')
    self.assertEqual(c0.c_type, 'uint8_t')
    self.assertEqual(c0.domain, 'fuzztest::Arbitrary<uint8_t>()')
    self.assertEqual(c0.parent, '_arr_element')
    self.assertEqual(c0.tuple_index, 0)
    c1 = child.children[1]
    self.assertEqual(c1.name, '__arr_element_1')
    self.assertEqual(c1.value_type, 'BITS')
    self.assertEqual(c1.c_type, 'uint32_t')
    self.assertEqual(c1.value_type, 'BITS')
    self.assertEqual(c1.c_type, 'uint32_t')
    self.assertEqual(c1.domain, 'fuzztest::Arbitrary<uint32_t>()')
    self.assertEqual(c1.parent, '_arr_element')
    self.assertEqual(c1.tuple_index, 1)

  def test_array_of_arrays(self):
    u16 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=16)
    inner_arr = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=2, array_element=u16
    )
    outer_arr = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY,
        array_size=3,
        array_element=inner_arr,
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param('outer', outer_arr)
    self.assertEqual(param.name, 'outer')
    self.assertEqual(param.value_type, 'ARRAY')
    self.assertEqual(param.c_type, 'std::array<std::array<uint16_t, 2>, 3>')
    self.assertEqual(
        param.domain,
        'fuzztest::ArrayOf<3>(fuzztest::ArrayOf<2>('
        'fuzztest::Arbitrary<uint16_t>()))',
    )
    self.assertLen(param.children, 1)
    child = param.children[0]
    self.assertEqual(child.name, '_outer_element')
    self.assertEqual(child.value_type, 'ARRAY')
    self.assertEqual(child.c_type, 'std::array<uint16_t, 2>')
    self.assertEqual(
        child.domain,
        'fuzztest::ArrayOf<2>(fuzztest::Arbitrary<uint16_t>())',
    )
    self.assertLen(child.children, 1)
    grandchild = child.children[0]
    self.assertEqual(grandchild.name, '__outer_element_element')
    self.assertEqual(grandchild.value_type, 'BITS')
    self.assertEqual(grandchild.c_type, 'uint16_t')
    self.assertEqual(grandchild.domain, 'fuzztest::Arbitrary<uint16_t>()')

  def test_tuple_of_bits(self):
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u1, u8, u32]
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param('tup', tup)
    self.assertEqual(param.name, 'tup')
    self.assertEqual(param.value_type, 'TUPLE')
    self.assertEqual(param.c_type, 'std::tuple<uint8_t, uint8_t, uint32_t>')
    self.assertEqual(
        param.domain,
        'fuzztest::TupleOf(fuzztest::InRange(0, 1),'
        ' fuzztest::Arbitrary<uint8_t>(), fuzztest::Arbitrary<uint32_t>())',
    )
    self.assertLen(param.children, 3)
    self.assertEqual(param.children[0].name, '_tup_0')
    self.assertEqual(param.children[0].parent, 'tup')
    self.assertEqual(param.children[0].tuple_index, 0)
    self.assertEqual(param.children[1].name, '_tup_1')
    self.assertEqual(param.children[1].parent, 'tup')
    self.assertEqual(param.children[1].tuple_index, 1)
    self.assertEqual(param.children[2].name, '_tup_2')
    self.assertEqual(param.children[2].parent, 'tup')
    self.assertEqual(param.children[2].tuple_index, 2)

  def test_tuple_of_tuples(self):
    u5 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=5)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    inner1 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u5, u5]
    )
    inner2 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u32]
    )
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[inner1, inner2]
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param('tup', tup)
    self.assertEqual(param.name, 'tup')
    self.assertEqual(param.value_type, 'TUPLE')
    self.assertEqual(
        param.c_type,
        'std::tuple<std::tuple<uint8_t, uint8_t>, std::tuple<uint32_t>>',
    )
    self.assertEqual(
        param.domain,
        'fuzztest::TupleOf(fuzztest::TupleOf(fuzztest::InRange(0, 31),'
        ' fuzztest::InRange(0, 31)),'
        ' fuzztest::TupleOf(fuzztest::Arbitrary<uint32_t>()))',
    )
    self.assertLen(param.children, 2)
    self.assertEqual(param.children[0].name, '_tup_0')
    self.assertEqual(param.children[0].value_type, 'TUPLE')
    self.assertEqual(param.children[0].parent, 'tup')
    self.assertEqual(param.children[0].tuple_index, 0)
    self.assertLen(param.children[0].children, 2)
    self.assertEqual(param.children[0].children[0].parent, '_tup_0')
    self.assertEqual(param.children[0].children[0].tuple_index, 0)
    self.assertEqual(param.children[0].children[1].parent, '_tup_0')
    self.assertEqual(param.children[0].children[1].tuple_index, 1)
    self.assertEqual(param.children[1].name, '_tup_1')
    self.assertEqual(param.children[1].value_type, 'TUPLE')
    self.assertEqual(param.children[1].parent, 'tup')
    self.assertEqual(param.children[1].tuple_index, 1)
    self.assertLen(param.children[1].children, 1)
    self.assertEqual(param.children[1].children[0].parent, '_tup_1')
    self.assertEqual(param.children[1].children[0].tuple_index, 0)

  def test_tuple_of_arrays(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    a8 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=4, array_element=u8
    )
    u16 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=16)
    a16 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=2, array_element=u16
    )
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[a8, a16]
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param('tup', tup)
    self.assertEqual(param.name, 'tup')
    self.assertEqual(param.value_type, 'TUPLE')
    self.assertEqual(
        param.c_type,
        'std::tuple<std::array<uint8_t, 4>, std::array<uint16_t, 2>>',
    )
    self.assertEqual(
        param.domain,
        'fuzztest::TupleOf(fuzztest::ArrayOf<4>(fuzztest::Arbitrary<uint8_t>()),'
        ' fuzztest::ArrayOf<2>(fuzztest::Arbitrary<uint16_t>()))',
    )
    self.assertLen(param.children, 2)
    self.assertEqual(param.children[0].name, '_tup_0')
    self.assertEqual(param.children[0].value_type, 'ARRAY')
    self.assertEqual(param.children[0].parent, 'tup')
    self.assertEqual(param.children[0].tuple_index, 0)
    self.assertEqual(param.children[1].name, '_tup_1')
    self.assertEqual(param.children[1].value_type, 'ARRAY')
    self.assertEqual(param.children[1].parent, 'tup')
    self.assertEqual(param.children[1].tuple_index, 1)

  def test_tuple_mixed(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    inner_tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u8]
    )
    a32 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=3, array_element=u32
    )
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE,
        tuple_elements=[inner_tup, a32, u1],
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param('tup', tup)
    self.assertEqual(param.name, 'tup')
    self.assertEqual(param.value_type, 'TUPLE')
    self.assertEqual(
        param.c_type,
        'std::tuple<std::tuple<uint8_t, uint8_t>, std::array<uint32_t, 3>,'
        ' uint8_t>',
    )
    self.assertEqual(
        param.domain,
        'fuzztest::TupleOf(fuzztest::TupleOf(fuzztest::Arbitrary<uint8_t>(),'
        ' fuzztest::Arbitrary<uint8_t>()),'
        ' fuzztest::ArrayOf<3>(fuzztest::Arbitrary<uint32_t>()),'
        ' fuzztest::InRange(0, 1))',
    )
    self.assertLen(param.children, 3)
    self.assertEqual(param.children[0].name, '_tup_0')
    self.assertEqual(param.children[0].value_type, 'TUPLE')
    self.assertEqual(param.children[0].parent, 'tup')
    self.assertEqual(param.children[0].tuple_index, 0)
    self.assertEqual(param.children[0].children[0].parent, '_tup_0')
    self.assertEqual(param.children[0].children[0].tuple_index, 0)
    self.assertEqual(param.children[0].children[1].parent, '_tup_0')
    self.assertEqual(param.children[0].children[1].tuple_index, 1)
    self.assertEqual(param.children[1].name, '_tup_1')
    self.assertEqual(param.children[1].value_type, 'ARRAY')
    self.assertEqual(param.children[1].parent, 'tup')
    self.assertEqual(param.children[1].tuple_index, 1)
    self.assertEqual(param.children[2].name, '_tup_2')
    self.assertEqual(param.children[2].value_type, 'BITS')
    self.assertEqual(param.children[2].parent, 'tup')
    self.assertEqual(param.children[2].tuple_index, 2)

    self.assertEqual(param.children[2].tuple_index, 2)

  def test_tuple_top_level(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8]
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param(
        'tup', tup, is_top_level=True
    )
    self.assertEqual(param.name, 'tup')
    self.assertEqual(param.value_type, 'TUPLE')
    self.assertEqual(param.c_type, 'std::tuple<uint8_t>')
    self.assertEqual(
        param.domain,
        'fuzztest::TupleOf(fuzztest::TupleOf(fuzztest::Arbitrary<uint8_t>()))',
    )

  def test_tuple_top_level_mixed(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    inner_tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u8]
    )
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u32, inner_tup]
    )
    param = jit_wrapper_generator.convert_to_fuzztest_param(
        'tup', tup, is_top_level=True
    )
    self.assertEqual(param.name, 'tup')
    self.assertEqual(param.value_type, 'TUPLE')
    self.assertEqual(
        param.c_type, 'std::tuple<uint32_t, std::tuple<uint8_t, uint8_t>>'
    )
    self.assertEqual(
        param.domain,
        'fuzztest::TupleOf(fuzztest::TupleOf(fuzztest::Arbitrary<uint32_t>(),'
        ' fuzztest::TupleOf(fuzztest::Arbitrary<uint8_t>(),'
        ' fuzztest::Arbitrary<uint8_t>())))',
    )


class JitWrapperGeneratorWrappedToFuzztestTest(absltest.TestCase):

  def test_function_bits_params(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='my_func',
        class_name='MyFuncJit',
        header_guard='HEADER_GUARD',
        header_filename='my_func_jit.h',
        namespace='xls::test',
        aot_entrypoint=None,
        params=[
            jit_wrapper_generator.XlsNamedValue(
                name='a',
                type_proto=u8,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
            jit_wrapper_generator.XlsNamedValue(
                name='b',
                type_proto=u32,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
        ],
        result=jit_wrapper_generator.XlsNamedValue(
            name='res',
            type_proto=u8,
            packed_type='',
            unpacked_type='',
            specialized_type=None,
        ),
    )
    prop_func = jit_wrapper_generator.wrapped_to_fuzztest(wrapped_ir)
    self.assertEqual(prop_func.fuzztest_name, 'my_func_fuzztest')
    self.assertEqual(prop_func.property_function_name, 'my_func')
    self.assertEqual(prop_func.jit_classname, 'xls::test::MyFuncJit')
    self.assertEqual(prop_func.jit_class_header_filename, 'my_func_jit.h')
    self.assertEqual(prop_func.namespace, 'xls::test')
    self.assertTrue(prop_func.return_type)
    self.assertLen(prop_func.params, 2)
    self.assertEqual(prop_func.params[0].name, 'a')
    self.assertEqual(prop_func.params[0].c_type, 'uint8_t')
    self.assertEqual(prop_func.params[1].name, 'b')
    self.assertEqual(prop_func.params[1].c_type, 'uint32_t')

  def test_function_array_param(self):
    u16 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=16)
    a16 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=4, array_element=u16
    )
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='array_func',
        class_name='ArrayFuncJit',
        header_guard='HEADER_GUARD',
        header_filename='array_func_jit.h',
        namespace='xls',
        aot_entrypoint=None,
        params=[
            jit_wrapper_generator.XlsNamedValue(
                name='x',
                type_proto=a16,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
        ],
        result=jit_wrapper_generator.XlsNamedValue(
            name='res',
            type_proto=u16,
            packed_type='',
            unpacked_type='',
            specialized_type=None,
        ),
    )
    prop_func = jit_wrapper_generator.wrapped_to_fuzztest(wrapped_ir)
    self.assertEqual(prop_func.fuzztest_name, 'array_func_fuzztest')
    self.assertEqual(prop_func.jit_classname, 'xls::ArrayFuncJit')
    self.assertTrue(prop_func.return_type)
    self.assertLen(prop_func.params, 1)
    self.assertEqual(prop_func.params[0].name, 'x')
    self.assertEqual(prop_func.params[0].c_type, 'std::array<uint16_t, 4>')
    self.assertEqual(
        prop_func.params[0].domain,
        'fuzztest::ArrayOf<4>(fuzztest::Arbitrary<uint16_t>())',
    )

  def test_function_array_of_tuples_param(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u32 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=32)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u32]
    )
    arr = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=3, array_element=tup
    )
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='array_tuple_func',
        class_name='ArrayTupleFuncJit',
        header_guard='HEADER_GUARD',
        header_filename='array_tuple_func_jit.h',
        namespace='xls',
        aot_entrypoint=None,
        params=[
            jit_wrapper_generator.XlsNamedValue(
                name='x',
                type_proto=arr,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
        ],
        result=jit_wrapper_generator.XlsNamedValue(
            name='res',
            type_proto=u8,
            packed_type='',
            unpacked_type='',
            specialized_type=None,
        ),
    )
    prop_func = jit_wrapper_generator.wrapped_to_fuzztest(wrapped_ir)
    self.assertLen(prop_func.params, 1)
    param = prop_func.params[0]
    self.assertEqual(param.name, 'x')
    self.assertEqual(
        param.c_type, 'std::array<std::tuple<uint8_t, uint32_t>, 3>'
    )
    self.assertEqual(
        param.domain,
        'fuzztest::ArrayOf<3>(fuzztest::TupleOf(fuzztest::Arbitrary<uint8_t>(),'
        ' fuzztest::Arbitrary<uint32_t>()))',
    )

  def test_function_tuple_param(self):
    u1 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=1)
    u64 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=64)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u1, u64]
    )
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='tuple_func',
        class_name='TupleFuncJit',
        header_guard='HEADER_GUARD',
        header_filename='tuple_func_jit.h',
        namespace='xls',
        aot_entrypoint=None,
        params=[
            jit_wrapper_generator.XlsNamedValue(
                name='t',
                type_proto=tup,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
        ],
        result=jit_wrapper_generator.XlsNamedValue(
            name='res',
            type_proto=u1,
            packed_type='',
            unpacked_type='',
            specialized_type=None,
        ),
    )
    prop_func = jit_wrapper_generator.wrapped_to_fuzztest(wrapped_ir)
    self.assertEqual(prop_func.fuzztest_name, 'tuple_func_fuzztest')
    self.assertTrue(prop_func.return_type)
    self.assertLen(prop_func.params, 1)
    self.assertEqual(prop_func.params[0].name, 't')
    self.assertEqual(
        prop_func.params[0].c_type, 'std::tuple<uint8_t, uint64_t>'
    )
    self.assertEqual(
        prop_func.params[0].domain,
        'fuzztest::TupleOf(fuzztest::TupleOf(fuzztest::InRange(0, 1),'
        ' fuzztest::Arbitrary<uint64_t>()))',
    )

  def test_function_no_result(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='no_res',
        class_name='NoResJit',
        header_guard='HEADER_GUARD',
        header_filename='no_res_jit.h',
        namespace='xls',
        aot_entrypoint=None,
        params=[
            jit_wrapper_generator.XlsNamedValue(
                name='a',
                type_proto=u8,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
        ],
        result=None,
    )
    prop_func = jit_wrapper_generator.wrapped_to_fuzztest(wrapped_ir)
    self.assertEqual(prop_func.fuzztest_name, 'no_res_fuzztest')
    self.assertFalse(prop_func.return_type)
    self.assertLen(prop_func.params, 1)
    self.assertEqual(prop_func.params[0].name, 'a')


if __name__ == '__main__':
  absltest.main()
