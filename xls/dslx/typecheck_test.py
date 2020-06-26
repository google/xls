# Lint as: python3
#
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

"""Tests for xls.dslx.typecheck."""

import textwrap
from typing import Text, Optional

from absl import logging

from absl.testing import absltest
from xls.common.xls_error import XlsError
from xls.dslx import fakefs_util
from xls.dslx import parser_helpers
from xls.dslx import span
from xls.dslx import typecheck
from xls.dslx.xls_type_error import ArgCountMismatchError
from xls.dslx.xls_type_error import TypeInferenceError
from xls.dslx.xls_type_error import XlsTypeError


class TypecheckTest(absltest.TestCase):

  def _typecheck(self,
                 text: Text,
                 error: Optional[Text] = None,
                 error_type=XlsTypeError):
    """Checks the first function in "text" for type errors.

    Args:
      text: Text to parse.
      error: Whether it is expected that the text will cause a type error.
      error_type: Type of error to check for, if "error" is given.
    """
    filename = '/fake/test_module.x'
    with fakefs_util.scoped_fakefs(filename, text):
      m = parser_helpers.parse_text(
          text, 'test_module', print_on_error=True, filename=filename)
      if error:
        with self.assertRaises(error_type) as cm:
          typecheck.check_module(m, f_import=None)
        self.assertIn(error, str(cm.exception))
      else:
        try:
          typecheck.check_module(m, f_import=None)
        except span.PositionalError as e:
          parser_helpers.pprint_positional_error(e)
          raise

  def test_typecheck_identity(self):
    self._typecheck('fn f(x: u32) -> u32 { x }')
    self._typecheck('fn f(x: bits[3], y: bits[4]) -> bits[3] { x }')
    self._typecheck('fn f(x: bits[3], y: bits[4]) -> bits[4] { y }')
    self._typecheck(
        'fn f(x: bits[3], y: bits[4]) -> bits[5] { y }', error='uN[4] vs uN[5]')

  def test_typecheck_nil(self):
    self._typecheck('fn f(x: u32) -> () { () }')
    self._typecheck('fn f(x: u32) { () }')

  def test_typecheck_arithmetic(self):
    self._typecheck('fn f(x: u32, y: u32) -> u32 { x + y }')
    self._typecheck('fn f(x: u32, y: u32) { x + y }', error='<none> vs uN[32]')
    self._typecheck(
        'fn f(x: u32, y: bits[4]) { x + y }', error='uN[32] vs uN[4]')
    self._typecheck(
        'fn [N: u32] f(x: bits[N], y: bits[N]) -> bits[N] { x + y }')

  def test_typecheck_unary(self):
    self._typecheck('fn f(x: u32) -> u32 { !x }')
    self._typecheck('fn f(x: u32) -> u32 { -x }')

  def test_typecheck_let(self):
    self._typecheck('fn f() -> u32 { let x: u32 = u32:2 in x }')
    self._typecheck(
        """fn f() -> u32 {
          let x: u32 = u32:2 in
          let y: bits[4] = bits[4]:3 in
          y
        }
        """,
        error='uN[4] vs uN[32]')
    self._typecheck(
        'fn f() -> u32 { let (x, y): (u32, bits[4]) = (u32:2, bits[4]:3) in x }'
    )

  def test_typecheck_let_bad_rhs(self):
    self._typecheck(
        """fn f() -> bits[2] {
          let (x, (y, (z,))): (u32, (bits[4], (bits[2],))) = (u32:2, bits[4]:3) in
          z
        }
        """,
        error='did not match inferred type of right hand side')

  def test_typecheck_parametric_invocation(self):
    self._typecheck("""
    fn [N: u32] p(x: bits[N]) -> bits[N] { x + bits[N]:1 }
    fn f() -> u32 { p(u32:3) }
    """)

  def test_typecheck_parametric_invocation_with_tuple(self):
    self._typecheck("""
    fn [N: u32] p(x: bits[N]) -> (bits[N], bits[N]) { (x, x) }
    fn f() -> (u32, u32) { p(u32:3) }
    """)

  def test_typecheck_double_parametric_invocation(self):
    self._typecheck("""
    fn [N: u32] p(x: bits[N]) -> bits[N] { x + bits[N]:1 }
    fn [M: u32] o(x: bits[M]) -> bits[M] { p(x) }
    fn f() -> u32 { o(u32:3) }
    """)

  def test_typecheck_parametric_invocation_conflicting_args(self):
    self._typecheck(
        """
    fn [N: u32] id(x: bits[N], y: bits[N]) -> bits[N] { x }
    fn f() -> u32 { id(u8:3, u32:5) }
    """,
        error='saw: 8; then: 32')

  def test_typecheck_parametric_wrong_kind(self):
    self._typecheck(
        """
    fn [N: u32] id(x: bits[N]) -> bits[N] { x }
    fn f() -> u32 { id((u8:3,)) }
    """,
        error='different kinds')

  def test_typecheck_parametric_wrong_argc(self):
    self._typecheck(
        """
    fn [N: u32] id(x: bits[N]) -> bits[N] { x }
    fn f() -> u32 { id(u8:3, u8:4) }
    """,
        error='Expected 1 parameter(s) but got 2 argument(s)',
        error_type=ArgCountMismatchError)

  def test_typecheck_parametric_wrong_number_of_dims(self):
    self._typecheck(
        textwrap.dedent("""\
        fn [N: u32, M: u32] id(x: bits[N][M]) -> bits[N][M] { x }
        fn f() -> u32 { id(u32:42) }
        """),
        error='types are different kinds (array vs ubits)')

  def test_typecheck_recursion_causes_error(self):
    self._typecheck(
        """
    fn f(x: u32) -> u32 { f(x) }
    """,
        error='Recursion detected while typechecking',
        error_type=XlsError)

  def test_invoke_wrong_arg(self):
    self._typecheck(
        """
    fn id_u32(x: u32) -> u32 { x }
    fn f(x: u8) -> u8 { id_u32(x) }
    """,
        error='Mismatch between parameter and argument types')

  def test_bad_tuple_type(self):
    self._typecheck(
        """\
fn f() -> u32 {
  let (a, b, c): (u32, u32) = (u32:1, u32:2, u32:3) in
  a
}
""",
        error='Annotated type did not match inferred type',
        error_type=XlsTypeError)

  def test_logical_and_of_comparisons(self):
    self._typecheck("""
    fn f(a: u8, b: u8) -> bool { a == b }
    """)
    self._typecheck("""
    fn f(a: u8, b: u8, c: u32, d: u32) -> bool { a == b && c == d }
    """)

  def test_typedef(self):
    self._typecheck("""\
  type MyTypeDef = (u32, u8);
  fn id(x: MyTypeDef) -> MyTypeDef { x }
  fn f() -> MyTypeDef { id((u32:42, u8:127)) }
  """)

  def test_for(self):
    self._typecheck("""\
fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, u32:3) {
    let new_accum: u32 = accum + i in
    new_accum
  }(u32:0)
}
""")

  def test_for_builtin_in_body(self):
    self._typecheck("""\
fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, u32:3) {
    trace(accum)
  }(u32:0)
}
""")

  def test_for_nested_bindings(self):
    self._typecheck("""\
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, (u32, u8)) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
}
""")

  def test_for_with_bad_type_tree(self):
    self._typecheck(
        """\
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, u8) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
}
""",
        error='Expected a tuple type for these names, but got uN[8].')

  def test_parametric_derived_expr_type_mismatch(self):
    self._typecheck(
        """
    fn [X: u32, Y: bits[4] = X+X] p(x: bits[X]) -> bits[X] { x }
    fn f() -> u32 { p(u32:3) }
    """,
        error='Annotated type of derived parametric value did not match')

  def test_parametric_instantiation_vs_arg_OK(self):
    self._typecheck(
        """
    fn [X: u32 = u32: 5] foo(x: bits[X]) -> bits[X] { x }
    fn bar() -> bits[5] { foo(u5: 1) }
        """)

  def test_parametric_instantiation_vs_arg_error(self):
    self._typecheck(
        """
    fn [X: u32 = u32: 5] foo(x: bits[X]) -> bits[X] { x }
    fn bar() -> bits[10] { foo(u5: 1) + foo(u10: 1) }
        """,
          error='Parametric constraint violated')

  def test_parametric_instantiation_vs_body_OK(self):
    self._typecheck(
        """
    fn [X: u32 = u32: 5] foo() -> bits[5] { bits[X]: 1 + bits[5]: 1 }
    fn bar() -> bits[5] { foo() }
        """)

  def test_parametric_instantiation_vs_body_error(self):
    self._typecheck(
        """
    fn [X: u32 = u32: 5] foo() -> bits[10] { bits[X]: 1 + bits[10]: 1 }
    fn bar() -> bits[10] { foo() }
        """,
          error='Types are not compatible: uN[5] vs uN[10]')

  def test_parametric_instantiation_vs_return_OK(self):
    self._typecheck(
        """
    fn [X: u32 = u32: 5] foo() -> bits[5] { bits[X]: 1 }
    fn bar() -> bits[5] { foo() }
        """)

  def test_parametric_instantiation_vs_return_error(self):
    self._typecheck(
        """
    fn [X: u32 = u32: 5] foo() -> bits[10] { bits[X]: 1 }
    fn bar() -> bits[10] { foo() }
        """,
          error='Return type of function body for "foo" did not match')

  def test_parametric_indirect_instantiation_vs_arg_OK(self):
    self._typecheck(
        """
    fn [X: u32] foo(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
    fn [Y: u32] fazz(y: bits[Y]) -> bits[Y] { foo(y, y + bits[Y]: 1) }
    fn bar() -> bits[10] { fazz(u10: 1) }
        """)


  def test_parametric_indirect_instantiation_vs_arg_error(self):
    self._typecheck(
        """
    fn [X: u32] foo(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
    fn [Y: u32] fazz(y: bits[Y]) -> bits[Y] { foo(y, y++y) }
    fn bar() -> bits[10] { fazz(u10: 1) }
        """,
          error='Parametric value X was bound to different values')

  def test_parametric_indirect_instantiation_vs_body_OK(self):
    self._typecheck(
        """
    fn [X: u32] foo(x: bits[X]) -> bits[X + X] {
      let a = bits[X + X]: 5 in
      x++x + a
    }
    fn [Y: u32] fazz(y: bits[Y]) -> bits[Y + Y] { foo(y) }
    fn bar() -> bits[10] { fazz(u5: 1) }
        """)

  def test_parametric_indirect_instantiation_vs_body_error(self):
    self._typecheck(
        """
    fn [X: u32] foo(x: bits[X]) -> bits[X] {
      let a = bits[X + X]: 5 in
      x + a
    }
    fn [Y: u32] fazz(y: bits[Y]) -> bits[Y] { foo(y) }
    fn bar() -> bits[5] { fazz(u5: 1) }
        """,
          error='Types are not compatible: uN[5] vs uN[10]')

  def test_parametric_indirect_instantiation_vs_return_OK(self):
    self._typecheck(
        """
    fn [X: u32] foo(x: bits[X]) -> bits[X + X] { x++x }
    fn [Y: u32] fazz(y: bits[Y]) -> bits[Y + Y] { foo(y) }
    fn bar() -> bits[10] { fazz(u5: 1) }
        """)

  def test_parametric_indirect_instantiation_vs_return_error(self):
    self._typecheck(
        """
    fn [X: u32] foo(x: bits[X]) -> bits[X + X] { x * x }
    fn [Y: u32] fazz(y: bits[Y]) -> bits[Y + Y] { foo(y) }
    fn bar() -> bits[10] { fazz(u5: 1) }
        """,
          error='Return type of function body for "foo" did not match')

  def test_parametric_derived_instantiation_vs_arg_OK(self):
    self._typecheck(
        """
    fn [X: u32, Y: u32 = X + X] foo(x: bits[X], y: bits[Y]) -> bits[X] { x }
    fn bar() -> bits[5] { foo(u5: 1, u10: 2) }
        """)

  def test_parametric_derived_instantiation_vs_arg_error(self):
    self._typecheck(
        """
    fn [X: u32, Y: u32 = X + X] foo(x: bits[X], y: bits[Y]) -> bits[X] { x }
    fn bar() -> bits[5] { foo(u5: 1, u11: 2) }
        """,
          error='Parametric constraint violated')

  def test_parametric_derived_instantiation_vs_body_OK(self):
    self._typecheck(
        """
    fn [W: u32, Z: u32 = W + W] foo(w: bits[W]) -> bits[1] {
        let val: bits[Z] = w++w + bits[Z]: 5 in
        and_reduce(val)
    }
    fn bar() -> bits[1] { foo(u5: 5) + foo(u10: 10) }
        """)

  def test_parametric_derived_instantiation_vs_body_error(self):
    self._typecheck(
        """
    fn [W: u32, Z: u32 = W + W] foo(w: bits[W]) -> bits[1] {
        let val: bits[Z] = w + w in
        and_reduce(val)
    }
    fn bar() -> bits[1] { foo(u5: 5) }
        """,
        error='Types are not compatible: uN[10] vs uN[5]')

  def test_parametric_derived_instantiation_vs_return_OK(self):
    self._typecheck(
        """
    fn [X: u32, Y: u32 = X + X] double(x: bits[X]) -> bits[Y] { x++x }
    fn [W: u32, Z: u32 = W + W] foo(w: bits[W]) -> bits[Z] { double(w) }
    fn bar() -> bits[10] { foo(u5: 1) }
        """)

  def test_parametric_derived_instantiation_vs_return_error(self):
    self._typecheck(
        """
    fn [X: u32, Y: u32 = X + X] double(x: bits[X]) -> bits[Y] { x + x }
    fn [W: u32, Z: u32 = W + W] foo(w: bits[W]) -> bits[Z] { double(w) }
    fn bar() -> bits[10] { foo(u5: 1) }
        """,
          error='Return type of function body for "double" did not match')

  def test_parametric_fn_not_always_polymorphic(self):
    self._typecheck(
        """
    fn [X: u32] foo(x: bits[X]) -> u1 {
        let non_polymorphic  =  x + u5: 1 in
        u1: 0
    }
    fn bar() -> bits[1] {
        foo(u5: 5) ^ foo(u10: 5)
    }
        """,
          error='Types are not compatible: uN[10] vs uN[5]')

  def test_let_binding_inferred_does_not_match_annotation(self):
    self._typecheck(
        """
    fn f() -> u32 {
      let x: u32 = bits[4]:7 in
      x
    }
    """,
        error='Annotated type did not match inferred type of right hand side')

  def test_update_builtin(self):
    self._typecheck("""\
fn f() -> u32[3] {
  let x: u32[3] = u32[3]:[0, 1, 2] in
  update(x, u32:1, u32:3)
}
""")

  def test_slice_builtin(self):
    self._typecheck("""\
fn f() -> u32[3] {
  let x: u32[2] = u32[2]:[0, 1] in
  slice(x, u32:0, u32[3]:[0, 0, 0])
}
""")

  def test_select_builtin(self):
    self._typecheck("""\
fn f(x: bool) -> u32 {
  select(x, u32:1, u32:0)
}
""")

  def test_select_builtin_different_on_false(self):
    self._typecheck(
        """\
fn f(x: bool) -> u32 {
  select(x, u32:1, u8:0)
}
""",
        error='Want arguments 1 and 2 to be of the same type')

  def test_ternary_non_boolean_test(self):
    self._typecheck(
        """\
fn f(x: u32) -> u32 {
  u32:42 if x else u32:64
}
""",
        error='Test type for conditional expression is not "bool"')

  def test_add_with_carry_builtin(self):
    self._typecheck("""\
fn f(x: u32) -> (u1, u32) {
  add_with_carry(x, x)
}
""")

  def test_update_incompatible_value(self):
    self._typecheck(
        textwrap.dedent("""\
        fn f(x: u32[5]) -> u32[5] {
          update(x, u32:1, u8:0)
        }
        """),
        error='uN[32] to match argument 2 type uN[8]')

  def test_update_multidim_index(self):
    self._typecheck(
        textwrap.dedent("""\
        fn f(x: u32[6][5], i: u32[2]) -> u32[6][5] {
          update(x, i, u32[6]:0)
        }
        """),
        error='Want argument 1 to be unsigned bits')

  def test_typecheck_index(self):
    self._typecheck('fn f(x: u32[5], i: u8) -> u32 { x[i] }')
    self._typecheck(
        'fn f(x: u32, i: u8) -> u32 { x[i] }',
        error='not an array',
        error_type=TypeInferenceError)
    self._typecheck(
        'fn f(x: u32[5], i: u8[5]) -> u32 { x[i] }', error='not scalar bits')

  def test_out_of_range_number(self):
    self._typecheck('fn f() -> u8 { u8:255 }')  # In range, no error.
    self._typecheck('fn f() -> u8 { u8:-1 }')  # In range, no error.
    self._typecheck('fn f() -> u8 { u8:-2 }')  # In range, no error.
    self._typecheck(
        'fn f() -> u8 { u8:256 }',
        error="value '256' does not fit in the bitwidth of a uN[8]",
        error_type=TypeInferenceError)

  def test_out_of_range_number_in_constant_array(self):
    self._typecheck(
        'fn f() -> u8[3] { u8[3]:[1, 2, 256] }',
        error="value '256' does not fit in the bitwidth of a uN[8]",
        error_type=TypeInferenceError)

  def test_missing_annotation(self):
    self._typecheck(
        textwrap.dedent("""\
        fn f() -> u32 {
          let x = u32:2 in
          x+x
        }"""))

  def test_match_arm_mismatch(self):
    self._typecheck(
        'fn f(x: u8) -> u8 { match x { u8:0 => u8:3; _ => u3:3 } }',
        error='match arm did not have the same type')

  def test_unsupported_parametric_expression(self):
    self._typecheck("""\
        fn [N: u32, M: u32] f(x: u8) -> bits[N-M] { x }
        fn main () -> u8 { f(u8: 5) }
        """,
        error_type=TypeInferenceError,
        error='Could not concretize type with dimension: (N) - (M)')

  def test_array_inconsistency(self):
    self._typecheck(
        """
type Foo = (u8, u32);
fn f() -> Foo {
  let xs = Foo[2]:[(u8:0, u32:1), u32:2] in
  xs[u32:1]
}""",
        error='vs uN[32]: Array member did not have same type as other members.'
    )

  def test_array_of_consts(self):
    self._typecheck("""
fn f() -> u4 {
  let a: u4 = u4:1 in
  let my_array = [a] in
  a
}
""")

  def test_one_hot_sel_of_signed(self):
    self._typecheck("""
fn f() -> s4 {
  let a: s4 = s4:1 in
  let b: s4 = s4:2 in
  let s: u2 = u2:0b01 in
  one_hot_sel(s, [a, b])
}
""")

  def test_overlarge_enum_value(self):
    self._typecheck(
        """
enum Foo : u1 {
  A = 0,
  B = 1,
  C = 2,
}
""",
        error="value '2' does not fit in the bitwidth of a uN[1]",
        error_type=TypeInferenceError)

  def test_cannot_add_enums(self):
    self._typecheck(
        """
enum Foo : u2 {
  A = 0,
  B = 1,
}
fn f() -> Foo {
  Foo::A + Foo::B
}
""",
        error="Cannot use '+' on values with enum type Foo @ /fake/test_module.x:7:10"
    )

  def test_width_slices(self):
    self._typecheck('fn f(x: u32) -> bits[0] { x[0+:bits[0]] }')
    self._typecheck(
        'fn f(x: u32) -> u33 { x[0+:u33] }',
        error='Slice type must have <= original number of bits; attempted slice from 32 to 33 bits.'
    )
    self._typecheck('fn f(x: u32) -> u2 { x[32+:u2] }')
    self._typecheck('fn f(x: u32) -> u1 { x[31+:u1] }')
    # Start literal is treated as unsigned.
    self._typecheck('fn f(x: u32) -> u1 { x[-1+:u1] }')
    self._typecheck('fn f(x: u32) -> u2 { x[-1+:u2] }')
    self._typecheck('fn f(x: u32) -> u3 { x[-2+:u3] }')
    # We reject signed start literals.
    self._typecheck(
        'fn f(start: s32, x: u32) -> u3 { x[start+:u3] }',
        error='Start index for width-based slice must be unsigned.',
        error_type=TypeInferenceError)

  def _typecheck_si(self, s: Text, *args, **kwargs):
    program = """
    struct Point {
      x: s8,
      y: u32,
    }
    """ + s
    logging.info('typechecking: %s', program)
    self._typecheck(program, *args, **kwargs)

  def test_struct_instance(self):

    # Wrong type.
    self._typecheck_si(
        'fn f() -> Point { Point { y: u8:42, x: s8:255 } }',
        error='Member type for \'y\' (uN[32]) does not match expression type uN[8].'
    )
    # Out of order, this is OK.
    self._typecheck_si('fn f() -> Point { Point { y: u32:42, x: s8:255 } }')
    # Missing x.
    self._typecheck_si(
        'fn f() -> Point { Point { y: u32:42 } }',
        error='Struct instance is missing member(s): \'x\'',
        error_type=TypeInferenceError)
    # Missing y.
    self._typecheck_si(
        'fn f() -> Point { Point { x: s8: 255 } }',
        error='Struct instance is missing member(s): \'y\'',
        error_type=TypeInferenceError)
    # One extra.
    self._typecheck_si(
        'fn f() -> Point { Point { x: s8:255, y: u32:42, z: u32:1024 } }',
        error='Struct \'Point\' has no member \'z\', but it was provided by this instance.',
        error_type=TypeInferenceError)
    # Duplicate.
    self._typecheck_si(
        'fn f() -> Point { Point { x: s8:255, y: u32:42, y: u32:1024 } }',
        error='Duplicate value seen for \'y\' in this \'Point\' struct instance.',
        error_type=TypeInferenceError)
    # Struct not compatible with its tuple equivalent.
    self._typecheck_si(
        """
        fn f(x: (s8, u32)) -> (s8, u32) { x }
        fn g() -> (s8, u32) {
          let p = Point { x: s8:255, y: u32:1024 } in
          f(p)
        }
        """,
        error='argument type name: \'Point\'')

  def test_nominal_typing(self):
    # Nominal typing not structural, e.g. OtherPoint cannot be passed where we
    # want a Point, even though their members are the same.
    self._typecheck(
        """
        struct Point {
          x: s8,
          y: u32,
        }
        struct OtherPoint {
          x: s8,
          y: u32
        }
        fn f(x: Point) -> Point { x }
        fn g() -> Point {
          let shp = OtherPoint { x: s8:255, y: u32:1024 } in
          f(shp)
        }
        """,
        error='parameter type name: \'Point\'; argument type name: \'OtherPoint\''
    )

  def test_bad_enum_ref(self):
    program = """enum MyEnum : u1 {
      A = 0,
      B = 1,
    }

    fn f() -> MyEnum {
      MyEnum::C
    }
    """
    self._typecheck(
        program,
        error_type=TypeInferenceError,
        error="Name 'C' is not defined by the enum MyEnum")


if __name__ == '__main__':
  absltest.main()
