# Lint as: python3
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

"""Tests for xls.dslx.typecheck."""

import textwrap
from typing import Text, Optional

from absl import logging

from xls.common import test_base
from xls.dslx import fakefs_test_util
from xls.dslx import parser_helpers
from xls.dslx.python import cpp_typecheck
from xls.dslx.python.cpp_deduce import TypeInferenceError
from xls.dslx.python.cpp_deduce import XlsTypeError
from xls.dslx.python.import_routines import ImportCache


class TypecheckTest(test_base.TestCase):

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
    with fakefs_test_util.scoped_fakefs(filename, text):
      m = parser_helpers.parse_text(
          text, 'test_module', print_on_error=True, filename=filename)

      import_cache = ImportCache()
      additional_search_paths = ()

      if error:
        with self.assertRaises(error_type) as cm:
          cpp_typecheck.check_module(m, import_cache, additional_search_paths)
        self.assertIn(error, str(cm.exception))
      else:
        cpp_typecheck.check_module(m, import_cache, additional_search_paths)

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
    self._typecheck('fn f(x: u32, y: u32) { x + y }', error='uN[32] vs ()')
    self._typecheck(
        """
      fn f<N: u32>(x: bits[N], y: bits[N]) { x + y }
      fn g() -> u64 { f(u64: 5, u64: 5) }
        """,
        error='uN[64] vs ()')
    self._typecheck(
        'fn f(x: u32, y: bits[4]) { x + y }', error='uN[32] vs uN[4]')
    self._typecheck('fn f<N: u32>(x: bits[N], y: bits[N]) -> bits[N] { x + y }')

  def test_typecheck_unary(self):
    self._typecheck('fn f(x: u32) -> u32 { !x }')
    self._typecheck('fn f(x: u32) -> u32 { -x }')

  def test_typecheck_let(self):
    self._typecheck('fn f() -> u32 { let x: u32 = u32:2; x }')
    self._typecheck(
        """fn f() -> u32 {
          let x: u32 = u32:2;
          let y: bits[4] = bits[4]:3;
          y
        }
        """,
        error='uN[4] vs uN[32]')
    self._typecheck(
        'fn f() -> u32 { let (x, y): (u32, bits[4]) = (u32:2, bits[4]:3); x }')

  def test_typecheck_let_bad_rhs(self):
    self._typecheck(
        """fn f() -> bits[2] {
          let (x, (y, (z,))): (u32, (bits[4], (bits[2],))) = (u32:2, bits[4]:3);
          z
        }
        """,
        error='did not match inferred type of right hand side')

  def test_typecheck_parametric_invocation(self):
    self._typecheck("""
    fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
    fn f() -> u32 { p(u32:3) }
    """)

  def test_typecheck_parametric_invocation_with_tuple(self):
    self._typecheck("""
    fn p<N: u32>(x: bits[N]) -> (bits[N], bits[N]) { (x, x) }
    fn f() -> (u32, u32) { p(u32:3) }
    """)

  def test_typecheck_double_parametric_invocation(self):
    self._typecheck("""
    fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
    fn o<M: u32>(x: bits[M]) -> bits[M] { p(x) }
    fn f() -> u32 { o(u32:3) }
    """)

  def test_typecheck_parametric_invocation_conflicting_args(self):
    self._typecheck(
        """
    fn id<N: u32>(x: bits[N], y: bits[N]) -> bits[N] { x }
    fn f() -> u32 { id(u8:3, u32:5) }
    """,
        error='saw: 8; then: 32')

  def test_typecheck_parametric_wrong_kind(self):
    self._typecheck(
        """
    fn id<N: u32>(x: bits[N]) -> bits[N] { x }
    fn f() -> u32 { id((u8:3,)) }
    """,
        error='different kinds')

  def test_typecheck_parametric_wrong_number_of_dims(self):
    self._typecheck(
        textwrap.dedent("""\
        fn id<N: u32, M: u32>(x: bits[N][M]) -> bits[N][M] { x }
        fn f() -> u32 { id(u32:42) }
        """),
        error='types are different kinds (array vs ubits)')

  def test_typecheck_recursion_causes_error(self):
    self._typecheck(
        """
    fn f(x: u32) -> u32 { f(x) }
    """,
        error='Recursion detected while typechecking',
        error_type=TypeInferenceError)

  def test_typecheck_parametric_recursion_causes_error(self):
    self._typecheck(
        """
    fn f<X: u32>(x: bits[X]) -> u32 { f(x) }
    fn g() -> u32 { f(u32: 5) }
    """,
        error='Recursion detected while typechecking',
        error_type=TypeInferenceError)

  def test_typecheck_higher_order_recursion_causes_error(self):
    self._typecheck(
        """
    fn h<Y: u32>(y: bits[Y]) -> bits[Y] { h(y) }
    fn g() -> u32[3] {
        let x0 = u32[3]:[0, 1, 2];
        map(x0, h)
    }
    """,
        error='Recursion detected while typechecking',
        error_type=TypeInferenceError)

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
  let (a, b, c): (u32, u32) = (u32:1, u32:2, u32:3);
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
    let new_accum: u32 = accum + i;
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
        error='Expected a tuple type for these names, but got uN[8].',
        error_type=TypeInferenceError)

  def test_parametric_derived_expr_type_mismatch(self):
    self._typecheck(
        """
    fn p<X: u32, Y: bits[4] = X+X>(x: bits[X]) -> bits[X] { x }
    fn f() -> u32 { p(u32:3) }
    """,
        error='Annotated type of derived parametric value did not match')

  def test_parametric_instantiation_vs_arg_ok(self):
    self._typecheck("""
    fn parametric<X: u32 = u32:5> (x: bits[X]) -> bits[X] { x }
    fn main() -> bits[5] { parametric(u5:1) }
        """)

  def test_parametric_instantiation_vs_arg_error(self):
    self._typecheck(
        """
    fn foo<X: u32 = u32: 5>(x: bits[X]) -> bits[X] { x }
    fn bar() -> bits[10] { foo(u5: 1) + foo(u10: 1) }
        """,
        error='Parametric constraint violated')

  def test_parametric_instantiation_vs_body_ok(self):
    self._typecheck("""
    fn parametric<X: u32 = u32:5>() -> bits[5] { bits[X]:1 + bits[5]:1 }
    fn main() -> bits[5] { parametric() }
        """)

  def test_parametric_instantiation_vs_body_error(self):
    self._typecheck(
        """
    fn foo<X: u32 = u32: 5>() -> bits[10] { bits[X]: 1 + bits[10]: 1 }
    fn bar() -> bits[10] { foo() }
        """,
        error='Types are not compatible: uN[5] vs uN[10]')

  def test_parametric_instantiation_vs_return_ok(self):
    self._typecheck("""
    fn parametric<X: u32 = u32: 5>() -> bits[5] { bits[X]: 1 }
    fn main() -> bits[5] { parametric() }
        """)

  def test_parametric_instantiation_vs_return_error(self):
    self._typecheck(
        """
    fn foo<X: u32 = u32: 5>() -> bits[10] { bits[X]: 1 }
    fn bar() -> bits[10] { foo() }
        """,
        error="Return type of function body for 'foo' did not match")

  def test_parametric_indirect_instantiation_vs_arg_ok(self):
    self._typecheck("""
    fn foo<X: u32>(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
    fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y, y + bits[Y]: 1) }
    fn bar() -> bits[10] { fazz(u10: 1) }
        """)

  def test_parametric_indirect_instantiation_vs_arg_error(self):
    self._typecheck(
        """
    fn foo<X: u32>(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
    fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y, y++y) }
    fn bar() -> bits[10] { fazz(u10: 1) }
        """,
        error='Parametric value X was bound to different values')

  def test_parametric_indirect_instantiation_vs_body_ok(self):
    self._typecheck("""
    fn foo<X: u32, R: u32 = X + X>(x: bits[X]) -> bits[R] {
      let a = bits[R]: 5;
      x++x + a
    }
    fn fazz<Y: u32, T: u32 = Y + Y>(y: bits[Y]) -> bits[T] { foo(y) }
    fn bar() -> bits[10] { fazz(u5: 1) }
        """)

  def test_parametric_indirect_instantiation_vs_body_error(self):
    self._typecheck(
        """
    fn foo<X: u32, D: u32 = X + X>(x: bits[X]) -> bits[X] {
      let a = bits[D]: 5;
      x + a
    }
    fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y) }
    fn bar() -> bits[5] { fazz(u5: 1) }
        """,
        error='Types are not compatible: uN[5] vs uN[10]')

  def test_parametric_indirect_instantiation_vs_return_ok(self):
    self._typecheck("""
    fn foo<X: u32, R: u32 = X + X>(x: bits[X]) -> bits[R] { x++x }
    fn fazz<Y: u32, T: u32 = Y + Y>(y: bits[Y]) -> bits[T] { foo(y) }
    fn bar() -> bits[10] { fazz(u5: 1) }
        """)

  def test_parametric_indirect_instantiation_vs_return_error(self):
    self._typecheck(
        """
    fn foo<X: u32, R: u32 = X + X>(x: bits[X]) -> bits[R] { x * x }
    fn fazz<Y: u32, T: u32 = Y + Y>(y: bits[Y]) -> bits[T] { foo(y) }
    fn bar() -> bits[10] { fazz(u5: 1) }
        """,
        error="Return type of function body for 'foo' did not match")

  def test_parametric_derived_instantiation_vs_arg_ok(self):
    self._typecheck("""
    fn foo<X: u32, Y: u32 = X + X>(x: bits[X], y: bits[Y]) -> bits[X] { x }
    fn bar() -> bits[5] { foo(u5: 1, u10: 2) }
        """)

  def test_parametric_derived_instantiation_vs_arg_error(self):
    self._typecheck(
        """
    fn foo<X: u32, Y: u32 = X + X>(x: bits[X], y: bits[Y]) -> bits[X] { x }
    fn bar() -> bits[5] { foo(u5: 1, u11: 2) }
        """,
        error='Parametric constraint violated')

  def test_parametric_derived_instantiation_vs_body_ok(self):
    self._typecheck("""
    fn foo<W: u32, Z: u32 = W + W>(w: bits[W]) -> bits[1] {
        let val: bits[Z] = w++w + bits[Z]: 5;
        and_reduce(val)
    }
    fn bar() -> bits[1] { foo(u5: 5) + foo(u10: 10) }
        """)

  def test_parametric_derived_instantiation_vs_body_error(self):
    self._typecheck(
        """
    fn foo<W: u32, Z: u32 = W + W>(w: bits[W]) -> bits[1] {
        let val: bits[Z] = w + w;
        and_reduce(val)
    }
    fn bar() -> bits[1] { foo(u5: 5) }
        """,
        error='Types are not compatible: uN[10] vs uN[5]')

  def test_parametric_derived_instantiation_vs_return_ok(self):
    self._typecheck("""
    fn double<X: u32, Y: u32 = X + X> (x: bits[X]) -> bits[Y] { x++x }
    fn foo<W: u32, Z: u32 = W + W> (w: bits[W]) -> bits[Z] { double(w) }
    fn bar() -> bits[10] { foo(u5: 1) }
        """)

  def test_parametric_derived_instantiation_vs_return_error(self):
    self._typecheck(
        """
    fn double<X: u32, Y: u32 = X + X>(x: bits[X]) -> bits[Y] { x + x }
    fn foo<W: u32, Z: u32 = W + W>(w: bits[W]) -> bits[Z] { double(w) }
    fn bar() -> bits[10] { foo(u5: 1) }
        """,
        error="Return type of function body for 'double' did not match")

  def test_parametric_derived_instantiation_via_fn_call(self):
    self._typecheck("""
    fn double(n: u32) -> u32 { n * u32: 2 }
    fn foo<W: u32, Z: u32 = double(W)>(w: bits[W]) -> bits[Z] { w++w }
    fn bar() -> bits[10] { foo(u5: 1) }
        """)

  def test_parametric_fn_not_always_polymorphic(self):
    self._typecheck(
        """
    fn foo<X: u32>(x: bits[X]) -> u1 {
        let non_polymorphic  =  x + u5: 1;
        u1: 0
    }
    fn bar() -> bits[1] {
        foo(u5: 5) ^ foo(u10: 5)
    }
        """,
        error='Types are not compatible: uN[10] vs uN[5]')

  def test_parametric_width_slice_start_error(self):
    self._typecheck(
        """
    fn make_u1<N: u32>(x: bits[N]) -> u1 {
        x[4 +: bits[1]]
    }
    fn bar() -> bits[1] {
        make_u1(u10: 5) ^ make_u1(u2: 1)
    }
        """,
        error='Cannot fit slice start 4 in 2 bits',
        error_type=TypeInferenceError)

  def test_bit_slice_on_parametric_width(self):
    self._typecheck("""
    fn get_middle_bits<N: u32, R: u32 = N - u32:2>(x: bits[N]) -> bits[R] {
      x[1:-1]
    }

    fn caller() {
      let x1: u2 = get_middle_bits(u4:15);
      let x2: u4 = get_middle_bits(u6:63);
      ()
    }""")

  def test_parametric_map_non_polymorphic(self):
    self._typecheck(
        """
        fn add_one<N: u32>(x: bits[N]) -> bits[N] { x + bits[5]:1 }

        fn main() {
            let arr = [u5:1, u5:2, u5:3];
            let mapped_arr = map(arr, add_one);
            let type_error = add_one(u6:1);
            ()
        }""",
        error='Types are not compatible: uN[6] vs uN[5]')

  def test_let_binding_inferred_does_not_match_annotation(self):
    self._typecheck(
        """
        fn f() -> u32 {
          let x: u32 = bits[4]:7;
          x
        }
        """,
        error='Annotated type did not match inferred type of right hand side')

  def test_update_builtin(self):
    self._typecheck("""\
fn f() -> u32[3] {
  let x: u32[3] = u32[3]:[0, 1, 2];
  update(x, u32:1, u32:3)
}
""")

  def test_slice_builtin(self):
    self._typecheck("""\
fn f() -> u32[3] {
  let x: u32[2] = u32[2]:[0, 1];
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
        error="Want arguments 1 and 2 to 'select' to be of the same type; got uN[32] vs uN[8]"
    )

  def test_enumerate_builtin(self):
    self._typecheck("""
        type MyTup = (u32, u2);
        fn f(x: u2[7]) -> MyTup[7] {
          enumerate(x)
        }
        """)

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
        error='Want argument 1 to be unsigned bits',
        error_type=TypeInferenceError)

  def test_typecheck_index(self):
    self._typecheck('fn f(x: u32[5], i: u8) -> u32 { x[i] }')
    self._typecheck(
        'fn f(x: u32, i: u8) -> u32 { x[i] }',
        error='not an array',
        error_type=TypeInferenceError)
    self._typecheck(
        'fn f(x: u32[5], i: u8[5]) -> u32 { x[i] }',
        error='not (scalar) bits',
        error_type=TypeInferenceError)

  def test_out_of_range_number(self):
    self._typecheck('fn f() -> u8 { u8:255 }')  # In range, no error.
    self._typecheck('fn f() -> u8 { u8:-1 }')  # In range, no error.
    self._typecheck('fn f() -> u8 { u8:-2 }')  # In range, no error.
    self._typecheck(
        'fn f() -> u8 { u8:256 }',
        error="Value '256' does not fit in the bitwidth of a uN[8]",
        error_type=TypeInferenceError)

  def test_out_of_range_number_in_constant_array(self):
    self._typecheck(
        'fn f() -> u8[3] { u8[3]:[1, 2, 256] }',
        error="Value '256' does not fit in the bitwidth of a uN[8]",
        error_type=TypeInferenceError)

  def test_missing_annotation(self):
    self._typecheck(
        textwrap.dedent("""\
        fn f() -> u32 {
          let x = u32:2;
          x+x
        }"""))

  def test_match_arm_mismatch(self):
    self._typecheck(
        'fn f(x: u8) -> u8 { match x { u8:0 => u8:3, _ => u3:3 } }',
        error='match arm did not have the same type')

  def test_array_inconsistency(self):
    self._typecheck(
        """
type Foo = (u8, u32);
fn f() -> Foo {
  let xs = Foo[2]:[(u8:0, u32:1), u32:2];
  xs[u32:1]
}""",
        error='vs uN[32]: Array member did not have same type as other members.'
    )

  def test_array_of_consts(self):
    self._typecheck("""
fn f() -> u4 {
  let a: u4 = u4:1;
  let my_array = [a];
  a
}
""")

  def test_one_hot_bad_prio_type(self):
    self._typecheck(
        """
    fn f(x: u7, prio: u2) -> u8 {
      one_hot(x, prio)
    }""",
        error="Expected argument 1 to 'one_hot' to be a u1; got uN[2]",
        error_type=TypeInferenceError)

  def test_one_hot_sel_of_signed(self):
    self._typecheck("""
fn f() -> s4 {
  let a: s4 = s4:1;
  let b: s4 = s4:2;
  let s: u2 = u2:0b01;
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
        error="Value '2' does not fit in the bitwidth of a uN[1]",
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
        error="Cannot use '+' on values with enum type Foo",
        error_type=TypeInferenceError,
    )

  def test_width_slices(self):
    self._typecheck('fn f(x: u32) -> bits[0] { x[0+:bits[0]] }')
    self._typecheck('fn f(x: u32) -> u2 { x[32+:u2] }')
    self._typecheck('fn f(x: u32) -> u1 { x[31+:u1] }')

  def test_width_slice_negative_start_number(self):
    # Start literal is treated as unsigned.
    self._typecheck('fn f(x: u32) -> u1 { x[-1+:u1] }')
    self._typecheck('fn f(x: u32) -> u2 { x[-1+:u2] }')
    self._typecheck('fn f(x: u32) -> u3 { x[-2+:u3] }')

  def test_width_slice_signed_start(self):
    # We reject signed start literals.
    self._typecheck(
        'fn f(start: s32, x: u32) -> u3 { x[start+:u3] }',
        error='Start index for width-based slice must be unsigned.',
        error_type=TypeInferenceError)

  def test_width_slice_tuple_start(self):
    # We reject signed start literals.
    self._typecheck(
        'fn f(start: (s32), x: u32) -> u3 { x[start+:u3] }',
        error='Start expression for width slice must be bits typed',
        error_type=TypeInferenceError)

  def test_width_slice_tuple_subject(self):
    # We reject signed start literals.
    self._typecheck(
        'fn f(start: s32, x: (u32)) -> u3 { x[start+:u3] }',
        error="Value to slice is not of 'bits' type",
        error_type=TypeInferenceError)

  def test_overlarge_width_slice(self):
    self._typecheck(
        'fn f(x: u32) -> u33 { x[0+:u33] }',
        error='Slice type must have <= original number of bits; attempted slice from 32 to 33 bits.'
    )

  def _typecheck_si(self, s: Text, *args, **kwargs):
    """Shorthand for 'typecheck struct instance'."""
    program = """
    struct Point {
      x: s8,
      y: u32,
    }
    """ + s
    logging.info('typechecking: %s', program)
    self._typecheck(program, *args, **kwargs)

  def test_access_missing_member(self):
    self._typecheck_si(
        'fn f(p: Point) -> () { p.z }',
        error="Could not infer type: Struct 'Point' does not have a member with name 'z'",
        error_type=TypeInferenceError)

  def test_struct_instance(self):
    # Wrong type.
    self._typecheck_si(
        'fn f() -> Point { Point { y: u8:42, x: s8:255 } }',
        error='Types are not compatible: uN[32] vs uN[8]')
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
          let p = Point { x: s8:255, y: u32:1024 };
          f(p)
        }
        """,
        error='argument type name: \'Point\'')

  def test_splat_struct_instance(self):
    self._typecheck_si(
        'fn f(p: Point) -> Point { Point { x: s8:42, x: s8:64, ..p } }',
        error='Duplicate value seen for \'x\'',
        error_type=TypeInferenceError)
    self._typecheck_si(
        'fn f(p: Point) -> Point { Point { q: u32:42, ..p } }',
        error='Struct \'Point\' has no member \'q\'',
        error_type=TypeInferenceError)

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
          let shp = OtherPoint { x: s8:255, y: u32:1024 };
          f(shp)
        }
        """,
        error='parameter type name: \'Point\'; argument type name: \'OtherPoint\''
    )

  def _typecheck_parametric_si(self, s: Text, *args, **kwargs):
    program = """
    struct Point<N: u32, M: u32 = N + N> {
      x: bits[N],
      y: bits[M]
    }
    """ + s
    logging.info('typechecking: %s', program)
    self._typecheck(program, *args, **kwargs)

  def test_parametric_struct_wrong_derived_type(self):
    self._typecheck_parametric_si(
        'fn f() -> Point<32, 63> { Point { x: u32:5, y: u63:255 } }',
        error='Types are not compatible: uN[64] vs uN[63]')

  def test_parametric_struct_too_many_parametric_args(self):
    self._typecheck_parametric_si(
        'fn f() -> Point<u32:5, u32:10, u32:15> { Point { x: u5:5, y: u10:255 } }',
        error="Expected 2 parametric arguments for 'Point'; got 3",
        error_type=TypeInferenceError)

  def test_parametric_struct_instance(self):
    # Out of order, this is OK.
    self._typecheck_parametric_si("""
      fn f() -> Point<32, 64> { Point { y: u64:42, x: u32:255 } }
    """)

  def test_parametric_struct_ok(self):
    # OK struct type-parametric instantiation in parametric function.
    self._typecheck_parametric_si("""
      fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
        Point { x, y }
      }

      fn main() {
        let _ = f(u5:1, u10:2);
        let _ = f(u14:1, u28:2);
        ()
      }""")

  def test_parametric_struct_bad_return_type(self):
    # Bad return type.
    self._typecheck_parametric_si(
        'fn f() -> Point<5, 10> { Point { x: u32:5, y: u64:255 } }',
        error='(x: uN[32], y: uN[64]) vs (x: uN[5], y: uN[10])')

  def test_parametric_struct_bad_struct_type_parametric_instantiation(self):
    # Bad struct type-parametric instantiation in parametric function.
    self._typecheck_parametric_si(
        """
      fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
        Point { x, y }
      }

      fn main() {
        let _ = f(u5:1, u10:2);
        let _ = f(u14:1, u15:2);
        ()
      }""",
        error='Types are not compatible: uN[28] vs uN[15]')

  def test_parametric_struct_bad_struct_type_parametric_splat_instantiation(
      self):
    self._typecheck_parametric_si(
        """
      fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
        let p = Point { x, y };
        Point { x: (x++x), ..p }
      }

      fn main() {
        let _ = f(u5:1, u10:2);
        ()
      }""",
        error='Types are not compatible: uN[20] vs uN[10]')

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

  def test_parametric_with_constant_array_ellipsis(self):
    program = """
    fn p<N: u32>(_: bits[N]) -> u8[2] {
      u8[2]:[0, ...]
    }

    fn main() -> u8[2] {
      p(false)
    }
    """
    self._typecheck(program)

  def test_char_literal_array(self):
    program = """
    fn main() -> u8[3] {
      u8[3]:['X', 'L', 'S']
    }
    """
    self._typecheck(program)

  def test_bad_array_literal_type(self):
    program = """
    fn main() -> s32[2] {
      s32:[1, 2]
    }
    """
    self._typecheck(
        program,
        error_type=TypeInferenceError,
        error='Annotated type for array literal must be an array type; got sbits s32'
    )

  def test_bad_quickcheck_function_ret(self):
    program = """
    #![quickcheck]
    fn f() -> u5 {
      u5:1
    }
    """
    self._typecheck(program, error='must return a bool')

  def test_bad_attribute_access_on_bits(self):
    program = """
    fn main() -> () {
      let x = u32:42;
      x.a
    }
    """
    self._typecheck(
        program,
        error='Expected a struct for attribute access',
        error_type=TypeInferenceError)

  def test_bad_attribute_access_on_tuple(self):
    program = """
    fn main() -> () {
      let x: (u32,) = (u32:42,);
      x.a
    }
    """
    self._typecheck(
        program,
        error='Expected a struct for attribute access',
        error_type=TypeInferenceError)

  def test_bad_quickcheck_function_parametrics(self):
    program = """
    #![quickcheck]
    fn f<N: u32>() -> bool {
      true
    }
    """
    self._typecheck(
        program,
        error_type=TypeInferenceError,
        error='Quickchecking parametric functions is unsupported')

  def test_array_ellipsis(self):
    self._typecheck("""
    fn main() -> u8[2] {
      u8[2]:[0, ...]
    }
    """)

  def test_bad_array_addition(self):
    program = """
    fn f(a: bits[32][4], b: bits[32][4]) -> bits[32][4] {
      a + b
    }
    """
    self._typecheck(
        program,
        error_type=TypeInferenceError,
        error='Binary operations can only be applied')

  def test_index(self):
    self._typecheck("""\
    fn f(x: uN[32][4]) -> u32 {
      x[u32:0]
    }""")


if __name__ == '__main__':
  test_base.main()
