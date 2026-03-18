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

import jinja2

from absl.testing import absltest
from xls.common import runfiles
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
    self.assertEqual(prop_func.params[0].index, 0)
    self.assertEqual(prop_func.params[1].name, 'b')
    self.assertEqual(prop_func.params[1].index, 1)

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
    self.assertEqual(prop_func.params[0].index, 0)

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
    self.assertEqual(param.index, 0)

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
    self.assertEqual(prop_func.params[0].index, 0)

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


class JitWrapperGeneratorRenderFuzztestTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = jinja2.Environment(
        undefined=jinja2.StrictUndefined, lstrip_blocks=True, trim_blocks=True
    )
    self.template = runfiles.get_contents_as_text('xls/jit/fuzztest_cc.tmpl')

  def test_render_fuzztest_basic(self):
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
    rendered_code = jit_wrapper_generator.render_fuzztest(
        wrapped_ir, self.env, self.template
    )
    self.assertIn('void my_func(xls::Value a, xls::Value b)', rendered_code)
    self.assertIn(
        'XLS_ASSERT_OK_AND_ASSIGN(xls::Value result, f->Run(', rendered_code
    )
    self.assertIn('FUZZ_TEST(my_func_fuzztest, my_func)', rendered_code)
    self.assertIn('xls::ArbitraryValue(', rendered_code)
    self.assertIn(
        'xls::test::MyFuncJit::GetParamType(0).value()', rendered_code
    )
    self.assertIn(
        'xls::test::MyFuncJit::GetParamType(1).value()', rendered_code
    )

  def test_render_fuzztest_array_of_int(self):
    u16 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=16)
    a16 = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.ARRAY, array_size=4, array_element=u16
    )
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='array_int_func',
        class_name='ArrayIntFuncJit',
        header_guard='HEADER_GUARD',
        header_filename='array_int_func_jit.h',
        namespace='xls::test',
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
        result=None,
    )
    rendered_code = jit_wrapper_generator.render_fuzztest(
        wrapped_ir, self.env, self.template
    )
    self.assertIn('void array_int_func(xls::Value x)', rendered_code)
    self.assertIn(
        'FUZZ_TEST(array_int_func_fuzztest, array_int_func)', rendered_code
    )
    self.assertIn('xls::test::ArrayIntFuncJit::GetParamType(0)', rendered_code)

  def test_render_fuzztest_array_of_tuple(self):
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
        namespace='xls::test',
        aot_entrypoint=None,
        params=[
            jit_wrapper_generator.XlsNamedValue(
                name='y',
                type_proto=arr,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
        ],
        result=None,
    )
    rendered_code = jit_wrapper_generator.render_fuzztest(
        wrapped_ir, self.env, self.template
    )
    self.assertIn('void array_tuple_func(xls::Value y)', rendered_code)
    self.assertIn(
        'FUZZ_TEST(array_tuple_func_fuzztest, array_tuple_func)', rendered_code
    )
    self.assertIn(
        'xls::test::ArrayTupleFuncJit::GetParamType(0).value()', rendered_code
    )

  def test_render_fuzztest_tuple_of_int(self):
    u16 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=16)
    u64 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=64)
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u16, u64]
    )
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='tuple_int_func',
        class_name='TupleIntFuncJit',
        header_guard='HEADER_GUARD',
        header_filename='tuple_int_func_jit.h',
        namespace='xls::test',
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
        result=None,
    )
    rendered_code = jit_wrapper_generator.render_fuzztest(
        wrapped_ir, self.env, self.template
    )
    self.assertIn('void tuple_int_func(xls::Value t)', rendered_code)
    self.assertIn(
        'FUZZ_TEST(tuple_int_func_fuzztest, tuple_int_func)', rendered_code
    )
    self.assertIn(
        'xls::test::TupleIntFuncJit::GetParamType(0).value()', rendered_code
    )

  def test_render_fuzztest_tuple_mixed(self):
    u8 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=8)
    u16 = type_pb2.TypeProto(type_enum=type_pb2.TypeProto.BITS, bit_count=16)
    inner_tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE, tuple_elements=[u8, u16]
    )
    tup = type_pb2.TypeProto(
        type_enum=type_pb2.TypeProto.TUPLE,
        tuple_elements=[u8, inner_tup],
    )
    wrapped_ir = jit_wrapper_generator.WrappedIr(
        jit_type=jit_wrapper_generator.JitType.FUNCTION,
        ir_text='',
        function_name='tuple_mixed_func',
        class_name='TupleMixedFuncJit',
        header_guard='HEADER_GUARD',
        header_filename='tuple_mixed_func_jit.h',
        namespace='xls::test',
        aot_entrypoint=None,
        params=[
            jit_wrapper_generator.XlsNamedValue(
                name='m',
                type_proto=tup,
                packed_type='',
                unpacked_type='',
                specialized_type=None,
            ),
        ],
        result=None,
    )
    rendered_code = jit_wrapper_generator.render_fuzztest(
        wrapped_ir, self.env, self.template
    )
    self.assertIn('void tuple_mixed_func(xls::Value m)', rendered_code)
    self.assertIn(
        'FUZZ_TEST(tuple_mixed_func_fuzztest, tuple_mixed_func)', rendered_code
    )
    self.assertIn(
        'xls::test::TupleMixedFuncJit::GetParamType(0).value()', rendered_code
    )


if __name__ == '__main__':
  absltest.main()
