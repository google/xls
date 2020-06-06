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
"""Tests for xls.dslx.parser."""

import io
import textwrap
from typing import Text, Optional, cast, Callable, TypeVar, Sequence, Any

from xls.dslx import ast
from xls.dslx import fakefs_util
from xls.dslx import parser
from xls.dslx import parser_helpers
from xls.dslx import scanner
from xls.dslx.scanner import Keyword
from xls.dslx.scanner import TokenKind
from xls.dslx.span import Pos
from xls.dslx.span import Span
from absl.testing import absltest


class ParserTest(absltest.TestCase):

  fake_filename = '/fake/fake.x'

  def _parse_internal(
      self, program: Text, bindings: Optional[parser.Bindings],
      fparse: Callable[[parser.Parser, parser.Bindings],
                       TypeVar('T')]
  ) -> TypeVar('T'):
    with fakefs_util.scoped_fakefs(self.fake_filename, program):
      s = scanner.Scanner(self.fake_filename, program)
      b = bindings or parser.Bindings(None)
      try:
        e = fparse(parser.Parser(s), b)
      except parser.ParseError as e:
        parser_helpers.pprint_positional_error(e)
        raise
      self.assertTrue(s.at_eof())
      return e

  def assertLen(self, sequence: Sequence[Any], target: int) -> None:
    assert len(sequence) == target, 'Got sequence of length %d, want %d' % (
        len(sequence), target)

  def assertEmpty(self, sequence: Sequence[Any]) -> None:
    assert not sequence, 'Got sequence of length %d, expected empty' % (
        len(sequence),)

  def parse_function(self, program, bindings=None):
    fparse = lambda p, b: p.parse_function(public=False, outer_bindings=b)
    return self._parse_internal(program, bindings, fparse)

  def parse_proc(self, program, bindings=None):
    fparse = lambda p, b: p.parse_proc(public=False, outer_bindings=b)
    return self._parse_internal(program, bindings, fparse)

  def parse_expression(self, program, bindings=None):
    fparse = lambda p, b: p.parse_expression(b)
    return self._parse_internal(program, bindings, fparse)

  def parse_module(self, program: Text):
    fparse = lambda p, _bindings: p.parse_module('test_module')
    return self._parse_internal(program, bindings=None, fparse=fparse)

  def test_let_expression(self):
    e = self.parse_expression('let x: u32 = 2 in x')
    self.assertIsInstance(e, ast.Let)
    self.assertEqual(e.name_def_tree.tree.identifier, 'x')
    self.assertIsInstance(e.type_, ast.TypeAnnotation)
    self.assertEqual(e.type_.primitive.value, scanner.Keyword.U32)
    self.assertIsInstance(e.rhs, ast.Number)
    self.assertEqual(e.rhs.value, '2')
    self.assertIsInstance(e.body, ast.NameRef)
    self.assertEqual(e.body.identifier, 'x')

  def test_identity_function(self):
    f = self.parse_function('fn ident(x: bits) { x }')
    self.assertIsInstance(f, ast.Function)
    self.assertIsInstance(f.body, ast.NameRef)
    self.assertEqual(f.body.identifier, 'x')

  def test_simple_proc(self):
    f = self.parse_proc("""
proc simple(addend: u32) {
  next(x: u32) {
    next(x+addend)
  }
}""")
    self.assertIsInstance(f, ast.Proc)
    self.assertIsInstance(f.name_def, ast.NameDef)
    self.assertEqual(f.name_def.identifier, 'simple')

  def test_concat_function(self):
    f = self.parse_function('fn concat(x: bits, y: bits) { x ++ y }')
    self.assertIsInstance(f, ast.Function)
    self.assertIsInstance(f.body, ast.Binop)
    self.assertIsInstance(f.body.lhs, ast.NameRef)
    self.assertIs(f.body.lhs.name_def, f.params[0].name)
    self.assertIs(f.body.rhs.name_def, f.params[1].name)
    self.assertIsInstance(f.body.rhs, ast.NameRef)
    self.assertEqual(f.body.lhs.identifier, 'x')
    self.assertEqual(f.body.rhs.identifier, 'y')

  def test_shra_function(self):
    f = self.parse_function('fn concat(x: bits, y: bits) { x >>> y }')
    self.assertIsInstance(f, ast.Function)
    self.assertIsInstance(f.body, ast.Binop)
    self.assertTrue(f.body.operator.is_kind(TokenKind.TRIPLE_CANGLE))

  def test_trailing_param_comma(self):
    program = textwrap.dedent("""\
    fn concat(
      x: bits,
      y: bits,
    ) {
      x ++ y
    }""")
    f = self.parse_function(program)
    self.assertLen(f.params, 2)

  def test_derived_parametric(self):
    program = textwrap.dedent("""\
    fn [X: u32, Y: u32 = X+X] parametric() -> (u32, u32) {
      (X, Y)
    }""")
    f = self.parse_function(program)
    self.assertLen(f.parametric_bindings, 2)
    self.assertIs(f.parametric_bindings[0].expr, None)
    self.assertEqual(f.parametric_bindings[0].name.identifier, 'X')
    self.assertIsInstance(f.parametric_bindings[1].expr, ast.Binop)
    self.assertEqual(f.parametric_bindings[1].name.identifier, 'Y')

  def test_let_destructure_flat(self):
    e = self.parse_expression('let (x, y, z): (u32,u32,u32) = (1, 2, 3) in y')
    self.assertIsInstance(e.rhs, ast.XlsTuple)
    self.assertLen(e.rhs.members, 3)

  def test_let_destructure_wildcard(self):
    e = self.parse_expression('let (x, y, _): (u32,u32,u32) = (1, 2, 3) in y')
    self.assertIsInstance(e.rhs, ast.XlsTuple)
    self.assertLen(e.rhs.members, 3)
    self.assertIsInstance(e.name_def_tree.tree[2].get_leaf(),
                          ast.WildcardPattern)

  def test_let_destructure_nested(self):
    e = self.parse_expression(
        'let (w, (x, (y)), z): (u32,(u32,(u32)),u32) = (1, (2, (3,)), 4) in y')
    self.assertIsInstance(e.rhs, ast.XlsTuple)
    # Three top-level members.
    self.assertLen(e.rhs.members, 3)
    # The middle one has two members.
    self.assertLen(e.rhs.members[1], 2)
    # The second one of those has one member.
    self.assertLen(e.rhs.members[1].members[1], 1)

    self.assertEqual(
        e.name_def_tree.span,
        Span(Pos('/fake/fake.x', 0, 4), Pos('/fake/fake.x', 0, 20)))
    self.assertEqual(
        e.name_def_tree.tree[1].span,
        Span(Pos('/fake/fake.x', 0, 8), Pos('/fake/fake.x', 0, 16)))

  def test_pprint_parse_error(self):
    output = io.StringIO()
    filename = '/fake/test_file.x'
    text = 'oh\nwhoops\nI did an\nerror somewhere\nthat is bad'
    with fakefs_util.scoped_fakefs(filename, text):
      pos = scanner.Pos(filename, lineno=2, colno=0)
      span = Span(pos, pos.bump_col())
      error = parser.ParseError(span, 'This is bad')
      parser_helpers.pprint_positional_error(
          error,
          output=cast(io.IOBase, output),
          color=False,
          error_context_line_count=3)

    expected = textwrap.dedent("""\
    /fake/test_file.x:2-4
      0002: whoops
    * 0003: I did an
            ^^ This is bad @ /fake/test_file.x:3:1
      0004: error somewhere
    """)
    self.assertMultiLineEqual(expected, output.getvalue())

  def test_for(self):
    program = textwrap.dedent("""
    let accum: u32 = 0 in
    let accum: u32 = for (i, accum): (u32, u32) in range(4) {
      let new_accum: u32 = accum + i in
      new_accum
    }(accum) in
    accum
    """)
    b = parser.Bindings(None)
    b.add('range', ast.BuiltinNameDef('range'))
    e = self.parse_expression(program, bindings=b)
    self.assertIsInstance(e, ast.Let)
    self.assertIsInstance(e.body, ast.Let)
    self.assertIsInstance(e.body.rhs, ast.For)
    for_ = e.body.rhs
    self.assertIsInstance(for_.init, ast.NameRef)
    self.assertIs(for_.init.name_def, e.name_def_tree.get_leaf())
    self.assertIsNot(for_.init.name_def, for_.names.tree[1].get_leaf())

  def test_for_freevars(self):
    program = """for (i, accum): (u32, u32) in range(4) {
      let new_accum: u32 = accum + i + j in
      new_accum
    }(u32:0)"""
    b = parser.Bindings(None)
    b.add('range', ast.BuiltinNameDef('range'))
    b.add('j', ast.BuiltinNameDef('j'))
    e = self.parse_expression(program, bindings=b)
    self.assertIsInstance(e, ast.For)
    self.assertEqual(e.span.start, Pos(self.fake_filename, lineno=0, colno=3))
    freevars = e.get_free_variables(e.span.start)
    self.assertLen(freevars, 1)

  def test_typedef(self):
    program = 'type MyType = u32;'
    m = self.parse_module(program)
    self.assertIsInstance(m, ast.Module)
    self.assertIsInstance(m.top[0], ast.TypeDef)
    self.assertEqual(m.top[0].name.identifier, 'MyType')

  def test_typedef_tuple_with_typedef_array(self):
    program = """
    type MyType = u32;
    type MyTupleType = (
      MyType[2],
    );
    """
    m = self.parse_module(program)
    self.assertIsInstance(m, ast.Module)
    self.assertIsInstance(m.top[0], ast.TypeDef)
    self.assertEqual(m.top[0].name.identifier, 'MyType')
    self.assertIsInstance(m.top[1], ast.TypeDef)
    self.assertEqual(m.top[1].name.identifier, 'MyTupleType')

  def test_typedef_tuple_with_const_sized_array(self):
    program = """
    const HOW_MANY_THINGS = u32:42;
    type MyTupleType = (
      u32[HOW_MANY_THINGS]
    );
    fn get_things(x: MyTupleType) -> u32[HOW_MANY_THINGS] {
      x[u32:0]
    }
    """
    m = self.parse_module(program)
    self.assertIsInstance(m, ast.Module)
    self.assertIsInstance(m.top[1], ast.TypeDef)
    self.assertEqual(m.top[1].name.identifier, 'MyTupleType')

  def test_enum(self):
    program = """enum MyEnum : u2 {
      A = 0,
      B = 1,
      C = 2,
      D = 3,
    }"""
    m = self.parse_module(program)
    self.assertIsInstance(m, ast.Module)
    self.assertIsInstance(m.top[0], ast.Enum)
    enum = m.top[0]
    self.assertEqual(enum.name.identifier, 'MyEnum')
    self.assertEqual([t[0].identifier for t in enum.values],
                     ['A', 'B', 'C', 'D'])

  def test_logical_equality(self):
    b = parser.Bindings(None)
    b.add('a', ast.BuiltinNameDef('a'))
    b.add('b', ast.BuiltinNameDef('b'))
    b.add('f', ast.BuiltinNameDef('f'))
    e = self.parse_expression('a ^ !b == f()', bindings=b)
    # This should group as:
    #   ((a) ^ (!b)) == (f())
    self.assertTrue(e.operator.is_kind(TokenKind.DOUBLE_EQUALS), msg=e.operator)
    self.assertTrue(e.lhs.operator.is_kind(TokenKind.HAT))
    self.assertTrue(e.lhs.rhs.operator.is_kind(TokenKind.BANG))
    self.assertIsInstance(e.rhs, ast.Invocation)
    self.assertIsInstance(e.rhs.callee, ast.NameRef)
    self.assertEqual(e.rhs.callee.identifier, 'f')

  def test_double_negation(self):
    b = parser.Bindings(None)
    b.add('x', ast.BuiltinNameDef('x'))
    e = self.parse_expression('!!x', bindings=b)
    self.assertIsInstance(e, ast.Unop)
    self.assertIsInstance(e.operand, ast.Unop)
    self.assertIsInstance(e.operand.operand, ast.NameRef)
    self.assertEqual(e.operand.operand.identifier, 'x')

  def test_logical_operator_binding(self):
    b = parser.Bindings(None)
    b.add('a', ast.BuiltinNameDef('a'))
    b.add('b', ast.BuiltinNameDef('b'))
    b.add('c', ast.BuiltinNameDef('c'))
    e = self.parse_expression('!a || !b && c', bindings=b)
    # This should group as:
    #   ((!a) || ((!b) && c))
    self.assertTrue(e.operator.is_kind(TokenKind.DOUBLE_BAR), msg=e.operator)
    self.assertTrue(e.lhs.operator.is_kind(TokenKind.BANG))
    self.assertTrue(e.rhs.operator.is_kind(TokenKind.DOUBLE_AMPERSAND))
    self.assertTrue(e.rhs.lhs.operator.is_kind(TokenKind.BANG))
    self.assertIsInstance(e.rhs.rhs, ast.NameRef)
    self.assertEqual(e.rhs.rhs.identifier, 'c')

  def test_cast(self):
    b = parser.Bindings(None)
    b.add('foo', ast.BuiltinNameDef('foo'))
    e = self.parse_expression('foo() as u32', bindings=b)
    self.assertIsInstance(e, ast.Cast)
    self.assertIsInstance(e.expr, ast.Invocation)
    self.assertIsInstance(e.type_, ast.TypeAnnotation)

  def test_module(self):
    program = textwrap.dedent("""\
    fn id(x: u32) -> u32 { x }
    test id_4 {
      assert_eq(u32:4, id(u32:4))
    }
    """)
    m = self.parse_module(program)
    self.assertEqual(['id_4'], m.get_test_names())
    id_4 = m.get_test('id_4')
    self.assertIsInstance(id_4.body, ast.Invocation)
    self.assertIsInstance(id_4.body.args[1], ast.Invocation)

  def test_module_with_semis(self):
    program = textwrap.dedent("""\
    #![cfg(let_terminator_is_semi = true)]
    fn f() -> s32 {
      let x: s32 = 42;
      x
    }
    """)
    self.parse_module(program)

  def test_empty_tuple(self):
    e = self.parse_expression('()')
    self.assertIsInstance(e, ast.XlsTuple)
    self.assertEmpty(e.members)

  def test_array(self):
    b = parser.Bindings(None)
    for identifier in 'a b c d'.split():
      b.add(identifier, ast.BuiltinNameDef(identifier))
    e = self.parse_expression('[a, b, c, d]', bindings=b)
    self.assertIsInstance(e, ast.Array)
    a = e
    self.assertLen(a.members, 4)
    self.assertIsInstance(a.members[0], ast.NameRef)
    self.assertIsInstance(a.members[1], ast.NameRef)
    self.assertIsInstance(a.members[2], ast.NameRef)
    self.assertIsInstance(a.members[3], ast.NameRef)

  def test_tuple_array_and_int(self):
    e = self.parse_expression('(u8[4]:[1, 2, 3, 4], 7)')
    self.assertIsInstance(e, ast.XlsTuple)
    self.assertIsInstance(e.members[0], ast.ConstantArray)
    self.assertIsInstance(e.members[1], ast.Number)

  def test_invalid_constructs(self):
    # TODO(rhundt): Over time these tests will fail at parse time or
    # runtime. At this point, we will have to check the relevant
    # errors and handle them properly.
    program = """
    // A handful of language constructs that could be considered harmful, borderline
    // illegal, or definitely illegal. They could be allowed, warned about, or
    // be reported as errors. This file collects all such cases and it will be
    // used in testing, once the semantics have been clarified. As new cases will
    // emerge, they should be added to this file as well.

    // Parameter re-definition.
    //
    // This is certainly allowed in imperative languages, and even Haskell allows
    // the following expression:
    //   f x = let x = 3 in x * x
    // No matter what you pass to this function, it will always return 9.
    //
    // The question is whether or not to warn on this, perhaps we should warn
    // optionally?
    //
    fn param_redefine(x: u32) -> u32 {
      let x: u32 = 3 in
        x * x
    }

    // Dead code
    //
    // The following code is legal, but is probably not what the user intended.
    // We should emit a warning, perhaps optionally.
    //
    fn dead_code(x: u32) -> u32 {
      let y: u32 = 3 in
      let y: u32 = 4 in
      x * y
    }

    // Unused code
    //
    // The following code is legal, but contains an unused expression.
    // Languages like Go warn on unused variables.
    fn unused_code(x: u32) -> u32 {
      let y: u32 = 3 in
      let z: u32 = 4 in
        x * y
    }

    // Unclear semantics in tuple assignments
    //
    // This code is probably legal, but it is not clear which value
    // will be assigned to i in the end.
    //

    // Note that types have to be defined globally (we should probably change that
    // and allow function-scoped types. Also, it is weird that 'type' needs a
    // semicolon.
    //
    type Tuple2 = (u32, u32);

    fn tuple_assign(x: u32, y: u32) -> (u32) {
      let (i, i): Tuple2 = (x, y) in
        i
    }

    // Invalid init expression size
    //
    // It is not clear whether or not this should be caught in the front-end,
    // or whether some other semantic applies, eg., non-initialized variables
    // are auto-initialized to the zero element of a given type.
    fn invalid_init_size() {
      const K: u32[2] = [
         0x428a2f98
      ] in K[1]
    }

    // Invalid init expression type
    //
    // This should probably be an error
    fn invalid_init_type() {
      const K: u32[2] = [
         'a', 'b'
      ] in K[1]
    }

    // Unused parameter
    //
    // Parameter is not used, this could point to a code problem.
    //
    fn unused_parm(x: u32, y: u32) {
      x + 1
    }

    // Double defined parameter
    //
    // Two parameters have the same name. This could point to a code problem.
    //
    fn double_defined_parm(x: u32, x: u32) {
      x + 1
    }

    // Invalid index
    //
    // Static or dynamic indices can be out of range. The front-end could find
    // static bounds check violations, but only a limited subset. The middle-end
    // can find a wider class of violations via copy propatation. The run-time
    // can find dynamic violations. This test should help to clarify what the limits
    // are.
    //
    fn invalid_index(x: u32) -> u32 {
      let K: u32[10] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) in
         let v: u32 = 10 in
           K[10]  // static violation with constant. Should be an error.
           + K[v] // static violation. Should be found via copy prop.
           + K[x] // dynamic violation. Can only be found at runtime or via
                  // inter-procedural copy propagation.
    }

    // Tail recursion
    //
    // Tail recursion can be transformed into a loop, but unless we have
    // the corresponding pass, this type of code must be considered
    // illegal.
    //
    fn tail_recursion(x: u32) -> u32 {
       select(x==0, 0, x + tail_recursion(x - 1))
    }

    // Regular recursion
    //
    // Is also illegal.
    //
    fn regular_recursion(x: u32) -> u32 {
       let y: u32 = 7 in
          let z: u32 = regular_recursion(x - 1) in
             y + z
    }
    """
    with fakefs_util.scoped_fakefs(self.fake_filename, program):
      parser_helpers.parse_text(
          program,
          name=self.fake_filename,
          print_on_error=True,
          filename=self.fake_filename)

  def test_bindings_stack(self):
    top = parser.Bindings(None)
    leaf0 = parser.Bindings(top)
    leaf1 = parser.Bindings(top)
    a = ast.BuiltinNameDef('a')
    b = ast.BuiltinNameDef('b')
    c = ast.BuiltinNameDef('c')
    top.add('a', a)
    leaf0.add('b', b)
    leaf1.add('c', c)
    pos = scanner.Pos(self.fake_filename, lineno=0, colno=0)
    span = Span(pos, pos)
    self.assertEqual(leaf0.resolve('a', span), a)
    self.assertEqual(leaf1.resolve('a', span), a)
    self.assertEqual(top.resolve('a', span), a)
    with self.assertRaises(parser.ParseError):
      top.resolve('b', span)
    with self.assertRaises(parser.ParseError):
      leaf1.resolve('b', span)
    with self.assertRaises(parser.ParseError):
      leaf0.resolve('c', span)
    self.assertEqual(leaf0.resolve('b', span), b)
    self.assertEqual(leaf1.resolve('c', span), c)

  def test_parser_errors(self):
    with self.assertRaises(parser.ParseError):
      program = """
      type Tuple2 = (u32, u32);

      fn tuple_assign(x: u32, y: u32) -> (u32) {
        // We don't have compound expressions yet, so the use of
        // curly braces should result in an error.
        let (i, i): Tuple2 = (x, y) in {
             i
        }
      }"""
      with fakefs_util.scoped_fakefs(self.fake_filename, program):
        parser_helpers.parse_text(
            program,
            name=self.fake_filename,
            print_on_error=True,
            filename=self.fake_filename)

  def test_co_recursion(self):
    with self.assertRaises(parser.ParseError) as cm:
      program = """
        // Co-recursion
        //
        // Also illegal. Can be found by finding cycles in the call graph.
        //
        fn foo(x: u32) -> u32 {
          let y: u32 = 7 in
             bar(x - 1, y)
        }

        fn bar(x: u32, y: u32) -> u32 {
          let z: u32 = 11 in
             foo(x - 1 + y + z)
        }
        """
      with fakefs_util.scoped_fakefs(self.fake_filename, program):
        parser_helpers.parse_text(
            program,
            name=self.fake_filename,
            print_on_error=True,
            filename=self.fake_filename)
    self.assertIn("Cannot find a definition for name: 'bar'", str(cm.exception))

  def test_match(self):
    b = parser.Bindings(None)
    b.add('x', ast.BuiltinNameDef('x'))
    e = self.parse_expression(
        'match x { u32:42 => u32:64; _ => u32:42 }', bindings=b)
    self.assertIsInstance(e, ast.Match)

  def test_match_with_const_pattern(self):
    m = self.parse_module("""
        const FOO = u32:64;
        fn f(x: u32) {
          match x {
            FOO => u32:64;
            _ => u32:42
          }
        }
        """)
    match = m.get_function_by_name()['f'].body
    self.assertIsInstance(match, ast.Match)
    self.assertIsInstance(match.arms[0], ast.MatchArm)
    self.assertLen(match.arms[0].patterns, 1)
    matcher = match.arms[0].patterns[0]
    self.assertIsInstance(matcher, ast.NameDefTree)
    self.assertTrue(matcher.is_leaf())
    self.assertIsInstance(matcher.get_leaf(), ast.NameRef)

  def test_match_multi_pattern_with_bindings(self):
    program = """
        fn f(x: u32) {
          match x {
            y | z => u32:64;
            _ => u32:42
          }
        }
        """
    with self.assertRaises(parser.ParseError) as cm:
      self.parse_function(program)
    self.assertIn('Cannot have multiple patterns that bind names.',
                  str(cm.exception))

  def test_ternary(self):
    b = parser.Bindings(None)
    e = self.parse_expression('u32:42 if true else u32:24', bindings=b)
    self.assertIsInstance(e, ast.Ternary)
    self.assertIsInstance(e.consequent, ast.Number)
    self.assertEqual(e.consequent.value, '42')
    self.assertIsInstance(e.alternate, ast.Number)
    self.assertEqual(e.alternate.value, '24')
    self.assertIsInstance(e.test, ast.Number)
    self.assertEqual(e.test.value.value, 'true')

  def test_constant_array(self):
    b = parser.Bindings(None)
    e = self.parse_expression('u32[2]:[u32:0, u32:1]', bindings=b)
    self.assertIsInstance(e, ast.ConstantArray)

  def test_bad_annotation(self):
    program = """
    fn foo(x: x: u32) -> u32 {
      x
    }
    """
    with self.assertRaises(parser.ParseError) as cm:
      self.parse_function(program)
    self.assertIn("identifier 'x' doesn't resolve to a type", str(cm.exception))

  def test_double_define_top_level_function(self):
    program = """
    fn foo(x: u32) -> u32 {
      x
    }
    fn foo(x: u32) -> u32 {
      x+u32:1
    }
    """
    with self.assertRaises(parser.ParseError) as cm:
      self.parse_module(program)
    self.assertIn('defined in this package multiple times', str(cm.exception))

  def test_parse_name_def_tree(self):
    text = '(a, (b, (c, d), e), f)'
    fparse = lambda p, b: p._parse_name_def_tree(b)
    bindings = parser.Bindings()
    ndt = self._parse_internal(text, bindings, fparse)
    self.assertIsInstance(ndt, ast.NameDefTree)
    self.assertLen(ndt.tree, 3)
    self.assertIsInstance(ndt.tree[0], ast.NameDefTree)
    self.assertTrue(ndt.tree[0].is_leaf())
    self.assertIsInstance(ndt.tree[2], ast.NameDefTree)
    self.assertTrue(ndt.tree[2].is_leaf())
    self.assertEqual(
        ndt.tree[0].span,
        Span(Pos(self.fake_filename, 0, 1), Pos(self.fake_filename, 0, 2)))
    self.assertEqual(
        ndt.tree[2].span,
        Span(Pos(self.fake_filename, 0, 20), Pos(self.fake_filename, 0, 21)))
    self.assertNotEqual(ndt.tree[2].span, ndt.tree[0].span)

  def test_match_freevars(self):
    text = """match x {
      y => z
    }"""
    b = parser.Bindings(None)
    for identifier in ('x', 'y', 'z'):
      b.add(identifier, ast.BuiltinNameDef(identifier))
    n = self.parse_expression(text, bindings=b)
    freevars = n.get_free_variables(n.span.start)
    self.assertEqual(freevars.keys(), {'x', 'y', 'z'})

  def test_multiple_module_level_const_bindings_is_error(self):
    program = textwrap.dedent("""\
    const FOO = u32:42;
    const FOO = u3:42;
    """)
    with self.assertRaises(parser.ParseError) as cm:
      self.parse_module(program)
    self.assertIn(
        f'Constant definition is shadowing an existing definition from {self.fake_filename}:1:7-1:10 @ {self.fake_filename}:2:7-2:10',
        str(cm.exception),
    )

  def test_cast_of_cast(self):
    program = """
    fn f(x: s8) -> u32 {
      x as s32 as u32
    }
    """
    m = self.parse_module(program)
    f = m.get_function('f')
    self.assertIsInstance(f.body, ast.Cast)
    self.assertIsInstance(f.body.expr, ast.Cast)
    self.assertIsInstance(f.body.expr.expr, ast.NameRef)

  def test_cast_of_cast_enum(self):
    program = """
    enum MyEnum: u3 {
      SOME_VALUE = 0,
    }
    fn f(x: u8) -> MyEnum {
      (x as u3) as MyEnum
    }
    """
    m = self.parse_module(program)
    f = m.get_function('f')
    self.assertIsInstance(f.body, ast.Cast)
    self.assertIsInstance(f.body.expr, ast.Cast)
    self.assertIsInstance(f.body.expr.expr, ast.NameRef)

  def test_bit_slice(self):
    program = """
    fn f(x: u32) -> u8 {
      x[0:8]
    }
    """
    m = self.parse_module(program)
    f = m.get_function('f')
    self.assertIsInstance(f.body, ast.Index)
    self.assertIsInstance(f.body.index, ast.Slice)
    self.assertIsInstance(f.body.index.start, ast.Number)
    self.assertIsInstance(f.body.index.limit, ast.Number)

  def test_bit_slice_of_call(self):
    program = """
    fn id(x: u32) -> u32 { x }
    fn f(x: u32) -> u8 {
      id(x)[0:8]
    }
    """
    m = self.parse_module(program)
    f = m.get_function('f')
    self.assertIsInstance(f.body, ast.Index)
    self.assertIsInstance(f.body.lhs, ast.Invocation)
    self.assertIsInstance(f.body.index, ast.Slice)
    self.assertIsInstance(f.body.index.start, ast.Number)
    self.assertIsInstance(f.body.index.limit, ast.Number)

  def test_bit_slice_of_bit_slice(self):
    program = """
    fn f(x: u32) -> u4 {
      x[0:8][4:]
    }
    """
    m = self.parse_module(program)
    f = m.get_function('f')
    outer_index = f.body
    self.assertIsInstance(outer_index, ast.Index)
    inner_index = f.body.lhs
    self.assertIsInstance(inner_index, ast.Index)

    self.assertIsInstance(inner_index.index, ast.Slice)
    self.assertIsInstance(inner_index.index.start, ast.Number)
    self.assertIsInstance(inner_index.index.limit, ast.Number)

    self.assertIsInstance(outer_index.index, ast.Slice)
    self.assertIsInstance(outer_index.index.start, ast.Number)
    self.assertIsNone(outer_index.index.limit)

  def test_bit_slice_with_width(self):
    program = """
    fn f(x: u32) -> u8 {
      x[1+:u8]
    }
    """
    m = self.parse_module(program)
    f = m.get_function('f')
    self.assertIsInstance(f.body, ast.Index)
    self.assertIsInstance(f.body.index, ast.WidthSlice)
    self.assertIsInstance(f.body.index.start, ast.Number)
    self.assertIsInstance(f.body.index.width, ast.TypeAnnotation)

  def test_module_const_with_enum(self):
    program = """
    enum MyEnum: u2 {
      FOO = 0,
      BAR = 1,
    }
    const MY_TUPLE = (MyEnum, MyEnum):(MyEnum::FOO, MyEnum::BAR);
    """
    m = self.parse_module(program)
    c = m.get_constant_by_name()['MY_TUPLE']
    self.assertIsInstance(c, ast.Constant)
    self.assertIsInstance(c.value, ast.XlsTuple)
    t = c.value
    self.assertIsInstance(t.members[0], ast.EnumRef)
    self.assertIsInstance(t.members[1], ast.EnumRef)

  def test_module_const_array_of_const_refs(self):
    program = """
    const MOL = u32:42;
    const ZERO = u32:0;
    const ARR = u32[2]:[MOL, ZERO];
    """
    m = self.parse_module(program)
    c = m.get_constant_by_name()['ARR']
    self.assertIsInstance(c, ast.Constant)
    self.assertIsInstance(c.value, ast.ConstantArray)
    self.assertLen(c.value.members, 2)
    self.assertIsInstance(c.value.members[0], ast.ConstRef)
    self.assertIsInstance(c.value.members[1], ast.ConstRef)

  def test_module_const_array_of_const_refs_elipsis(self):
    program = """
    const MOL = u32:42;
    const ZERO = u32:0;
    const ARR = u32[2]:[MOL, ZERO, ...];
    """
    m = self.parse_module(program)
    c = m.get_constant_by_name()['ARR']
    self.assertIsInstance(c, ast.Constant)
    self.assertIsInstance(c.value, ast.ConstantArray)
    self.assertLen(c.value.members, 2)
    self.assertIsInstance(c.value.members[0], ast.ConstRef)
    self.assertIsInstance(c.value.members[1], ast.ConstRef)
    self.assertTrue(c.value.has_ellipsis)

  def test_cast_to_typedef(self):
    program = """
    type u128 = bits[128];
    fn f(x: u32) -> u128 { x as u128 }
    """
    m = self.parse_module(program)
    body = m.get_function_by_name()['f'].body
    self.assertIsInstance(body, ast.Cast)

  def test_const_array_of_enum_refs(self):
    program = """
    enum MyEnum : u3 {
        FOO = 1,
        BAR = 2,
    }
    const A = MyEnum[2]:[MyEnum::FOO, MyEnum::BAR];
    """
    m = self.parse_module(program)
    c = m.get_constant_by_name()['A']
    self.assertIsInstance(c, ast.Constant)
    self.assertIsInstance(c.value, ast.ConstantArray)
    self.assertLen(c.value.members, 2)
    self.assertIsInstance(c.value.members[0], ast.EnumRef)
    self.assertIsInstance(c.value.members[1], ast.EnumRef)

  def test_struct(self):
    program = """
    struct Point {
      x: u32,
      y: u32,
    }
    """
    m = self.parse_module(program)
    c = m.get_typedef_by_name()['Point']
    self.assertIsInstance(c, ast.Struct)

  def test_struct_with_access_fn(self):
    program = """
    struct Point {
      x: u32,
      y: u32,
    }
    fn f(p: Point) -> u32 {
      p.x
    }
    fn g(xy: u32) -> Point {
      Point { x: xy, y: xy }
    }
    """
    m = self.parse_module(program)
    c = m.get_typedef_by_name()['Point']
    self.assertIsInstance(c, ast.Struct)
    attr = m.get_function_by_name()['f'].body
    self.assertIsInstance(attr, ast.Attr)
    self.assertIsInstance(attr.lhs, ast.NameRef)
    self.assertIsInstance(attr.attr, ast.NameDef)
    self.assertEqual(attr.attr.identifier, 'x')

  def test_enum_with_type_on_value(self):
    program = """
    enum MyEnum : u2 {
      FOO = u2:2
    }
    """
    with self.assertRaises(parser.ParseError) as cm:
      self.parse_module(program)
    self.assertIn('Type is annotated in enum value, but enum defines a type',
                  str(cm.exception))

  def test_import(self):
    program = """
    import thing
    """
    bindings = parser.Bindings(None)
    fparse = lambda p, bindings: p.parse_module('test_module', bindings)
    m = self._parse_internal(program, bindings, fparse)
    self.assertIsInstance(m.top[0], ast.Import)
    fake_pos = Pos(self.fake_filename, 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    self.assertIsInstance(bindings.resolve_node('thing', fake_span), ast.Import)

  def test_import_as(self):
    program = """
    import thing as other
    """
    bindings = parser.Bindings(None)
    fparse = lambda p, bindings: p.parse_module('test_module', bindings)
    m = self._parse_internal(program, bindings, fparse)
    self.assertIsInstance(m.top[0], ast.Import)
    fake_pos = Pos(self.fake_filename, 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    self.assertIsInstance(bindings.resolve_node('other', fake_span), ast.Import)
    self.assertIsNone(bindings.resolve_node_or_none('thing'), None)


if __name__ == '__main__':
  absltest.main()
