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

"""Tests for xls.ir.python.function_builder."""

from xls.ir.python import bits as bits_mod
from xls.ir.python import fileno as fileno_mod
from xls.ir.python import function_builder
from xls.ir.python import lsb_or_msb
from xls.ir.python import package as ir_package
from xls.ir.python import source_location
from xls.ir.python import value as ir_value
from absl.testing import absltest


class FunctionBuilderTest(absltest.TestCase):

  def test_simple_build_and_dump_package(self):
    p = ir_package.Package('test_package')
    fileno = p.get_or_create_fileno('my_file.x')
    fb = function_builder.FunctionBuilder('test_function', p)
    t = p.get_bits_type(32)
    x = fb.add_param('x', t)

    lineno = fileno_mod.Lineno(42)
    colno = fileno_mod.Colno(64)
    loc = source_location.SourceInfo(
        [source_location.SourceLocation(fileno, lineno, colno)])
    fb.add_or(x, x, loc=loc, name='my_or')
    fb.add_not(x, loc=loc, name='why_not')

    f = fb.build()
    self.assertEqual(f.name, 'test_function')
    self.assertEqual(
        f.dump_ir(), """\
fn test_function(x: bits[32]) -> bits[32] {
  my_or: bits[32] = or(x, x, id=2, pos=[(0,42,64)])
  ret why_not: bits[32] = not(x, id=3, pos=[(0,42,64)])
}
""")

    self.assertMultiLineEqual(
        p.dump_ir(), """\
package test_package

file_number 0 "my_file.x"

fn test_function(x: bits[32]) -> bits[32] {
  my_or: bits[32] = or(x, x, id=2, pos=[(0,42,64)])
  ret why_not: bits[32] = not(x, id=3, pos=[(0,42,64)])
}
""")

  def test_invoke_adder_2_plus_3_eq_5(self):
    p = ir_package.Package('test_package')
    fb = function_builder.FunctionBuilder('add_wrapper', p)
    t = p.get_bits_type(32)
    x = fb.add_param('x', t)
    y = fb.add_param('y', t)
    fb.add_add(x, y)
    add_wrapper = fb.build()

    main_fb = function_builder.FunctionBuilder('main', p)
    two = main_fb.add_literal_bits(bits_mod.UBits(value=2, bit_count=32))
    three = main_fb.add_literal_bits(bits_mod.UBits(value=3, bit_count=32))
    observed = main_fb.add_invoke([two, three], add_wrapper)
    main_fb.add_eq(
        observed,
        main_fb.add_literal_bits(bits_mod.UBits(value=5, bit_count=32)))
    main_fb.build()
    self.assertMultiLineEqual(
        p.dump_ir(), """\
package test_package

fn add_wrapper(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.3: bits[32] = add(x, y, id=3)
}

fn main() -> bits[1] {
  literal.4: bits[32] = literal(value=2, id=4)
  literal.5: bits[32] = literal(value=3, id=5)
  invoke.6: bits[32] = invoke(literal.4, literal.5, to_apply=add_wrapper, id=6)
  literal.7: bits[32] = literal(value=5, id=7)
  ret eq.8: bits[1] = eq(invoke.6, literal.7, id=8)
}
""")

  def test_literal_array(self):
    p = ir_package.Package('test_package')
    fb = function_builder.FunctionBuilder('f', p)
    fb.add_literal_value(
        ir_value.Value.make_array([
            ir_value.Value(bits_mod.UBits(value=5, bit_count=32)),
            ir_value.Value(bits_mod.UBits(value=6, bit_count=32)),
        ]))
    fb.build()
    self.assertMultiLineEqual(
        p.dump_ir(), """\
package test_package

fn f() -> bits[32][2] {
  ret literal.1: bits[32][2] = literal(value=[5, 6], id=1)
}
""")

  def test_match_true(self):
    p = ir_package.Package('test_package')
    fb = function_builder.FunctionBuilder('f', p)
    pred_t = p.get_bits_type(1)
    expr_t = p.get_bits_type(32)
    pred_x = fb.add_param('pred_x', pred_t)
    x = fb.add_param('x', expr_t)
    pred_y = fb.add_param('pred_y', pred_t)
    y = fb.add_param('y', expr_t)
    default = fb.add_param('default', expr_t)
    fb.add_match_true([pred_x, pred_y], [x, y], default)
    fb.build()
    self.assertMultiLineEqual(
        p.dump_ir(), """\
package test_package

fn f(pred_x: bits[1], x: bits[32], pred_y: bits[1], y: bits[32], default: bits[32]) -> bits[32] {
  concat.6: bits[2] = concat(pred_y, pred_x, id=6)
  one_hot.7: bits[3] = one_hot(concat.6, lsb_prio=true, id=7)
  ret one_hot_sel.8: bits[32] = one_hot_sel(one_hot.7, cases=[x, y, default], id=8)
}
""")

  def test_bvalue_methods(self):
    # This test is mainly about checking that pybind11 is able to map parameter
    # and return types properly. Because of this it's not necessary to check
    # the result at the end; that methods don't throw when called is enough.
    p = ir_package.Package('test_package')
    fb = function_builder.FunctionBuilder('test_function', p)
    x = fb.add_param('param_name', p.get_bits_type(32))

    self.assertIn('param_name', str(x))
    self.assertEqual(32, x.get_type().get_bit_count())

  def test_all_add_methods(self):
    # This test is mainly about checking that pybind11 is able to map parameter
    # and return types properly. Because of this it's not necessary to check
    # the result at the end; that methods don't throw when called is enough.
    p = ir_package.Package('test_package')
    fileno = p.get_or_create_fileno('my_file.x')
    lineno = fileno_mod.Lineno(42)
    colno = fileno_mod.Colno(64)
    loc = source_location.SourceInfo(
        [source_location.SourceLocation(fileno, lineno, colno)])
    fb = function_builder.FunctionBuilder('test_function', p)

    single_zero_bit = fb.add_literal_value(
        ir_value.Value(bits_mod.UBits(value=0, bit_count=1)))
    t = p.get_bits_type(32)
    x = fb.add_param('x', t)
    s = fb.add_param('s', p.get_bits_type(1))

    fb.add_shra(x, x, loc=loc)
    fb.add_shra(x, x, loc=loc)
    fb.add_shrl(x, x, loc=loc)
    fb.add_shll(x, x, loc=loc)
    fb.add_or(x, x, loc=loc)
    fb.add_nary_or([x], loc=loc)
    fb.add_xor(x, x, loc=loc)
    fb.add_and(x, x, loc=loc)
    fb.add_smul(x, x, loc=loc)
    fb.add_umul(x, x, loc=loc)
    fb.add_udiv(x, x, loc=loc)
    fb.add_sub(x, x, loc=loc)
    fb.add_add(x, x, loc=loc)

    fb.add_concat([x], loc=loc)

    fb.add_ule(x, x, loc=loc)
    fb.add_ult(x, x, loc=loc)
    fb.add_uge(x, x, loc=loc)
    fb.add_ugt(x, x, loc=loc)

    fb.add_sle(x, x, loc=loc)
    fb.add_slt(x, x, loc=loc)
    fb.add_sge(x, x, loc=loc)
    fb.add_sgt(x, x, loc=loc)

    fb.add_eq(x, x, loc=loc)
    fb.add_ne(x, x, loc=loc)

    fb.add_neg(x, loc=loc)
    fb.add_not(x, loc=loc)
    fb.add_clz(x, loc=loc)

    fb.add_one_hot(x, lsb_or_msb.LsbOrMsb.LSB, loc=loc)
    fb.add_one_hot_sel(s, [x], loc=loc)
    fb.add_priority_sel(s, [x], loc=loc)

    fb.add_literal_bits(bits_mod.UBits(value=2, bit_count=32), loc=loc)
    fb.add_literal_value(
        ir_value.Value(bits_mod.UBits(value=5, bit_count=32)), loc=loc)

    fb.add_sel(s, x, x, loc=loc)
    fb.add_sel_multi(s, [x], x, loc=loc)
    fb.add_match_true([single_zero_bit], [x], x, loc=loc)

    tuple_node = fb.add_tuple([x], loc=loc)
    fb.add_array([x], t, loc=loc)

    fb.add_tuple_index(tuple_node, 0, loc=loc)

    for_function_builder = function_builder.FunctionBuilder('for_f', p)
    for_function_builder.add_param('i', t)
    for_function_builder.add_param('x', t)
    for_function_builder.add_param('s', t)
    for_function = for_function_builder.build()

    fb.add_counted_for(x, 1, 1, for_function, [x], loc=loc)

    map_function_builder = function_builder.FunctionBuilder('map_f', p)
    map_function_builder.add_param('x', t)
    map_function = map_function_builder.build()
    fb.add_map(fb.add_array([x], t, loc=loc), map_function, loc=loc)

    input_function_builder = function_builder.FunctionBuilder('input_f', p)
    input_function_builder.add_param('x', t)
    input_function = input_function_builder.build()
    fb.add_invoke([x], input_function, loc=loc)

    fb.add_array_index(fb.add_array([x], t, loc=loc), [x], loc=loc)
    fb.add_reverse(x, loc=loc)
    fb.add_identity(x, loc=loc)
    fb.add_signext(x, 100, loc=loc)
    fb.add_zeroext(x, 100, loc=loc)
    fb.add_bit_slice(x, 4, 2, loc=loc)

    fb.build()


if __name__ == '__main__':
  absltest.main()
