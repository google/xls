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

"""Module for converting AST to IR text dumps."""

import pprint
from typing import Text, List, Optional, Tuple, Callable, Union

from absl import logging

from xls.common.xls_error import XlsError
from xls.dslx import ast_helpers
from xls.dslx import bit_helpers
from xls.dslx import extract_conversion_order
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_ast_visitor
from xls.dslx.python import cpp_dslx_builtins
from xls.dslx.python import cpp_ir_converter
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python.cpp_ast_visitor import visit
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import ConcreteTypeDim
from xls.dslx.python.cpp_concrete_type import EnumType
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_pos import Span
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.span import PositionalError
from xls.ir.python import bits as bits_mod
from xls.ir.python import fileno as fileno_mod
from xls.ir.python import function as ir_function
from xls.ir.python import function_builder
from xls.ir.python import number_parser
from xls.ir.python import package as ir_package
from xls.ir.python import source_location
from xls.ir.python import type as type_mod
from xls.ir.python import verifier as verifier_mod
from xls.ir.python.function_builder import BValue


class ParametricConversionError(XlsError):
  """Raised when we attempt to IR convert a parametric function."""


class ConversionError(PositionalError):
  """Raised on an issue converting to IR at a particular file position."""


def _int_to_bits(value: int, bit_count: int) -> bits_mod.Bits:
  """Converts a Python arbitrary precision int to a Bits type."""
  assert isinstance(bit_count, int), bit_count
  if bit_count <= 64:
    return bits_mod.UBits(value, bit_count) if value >= 0 else bits_mod.SBits(
        value, bit_count)
  return number_parser.bits_from_string(
      bit_helpers.to_hex_string(value, bit_count), bit_count=bit_count)


class _IrConverterFb(cpp_ast_visitor.AstVisitor):
  """An AST visitor that converts AST nodes into IR.

  Note that ASTs can only be converted to IR once they have been fully
  concretized; there is no parametric-function support in the IR text!

  Attributes:
    module: Module that we're converting IR for.
    node_to_ir: Mapping from AstNode to the IR that was used to emit it (as a
      FunctionBuilder BValue).
    symbolic_bindings: Mapping from parametric binding name (e.g. "N" in
      `fn [N: u32] id(x: bits[N]) -> bits[N]`) to its value in this conversion.
    type_info: Type information for the AstNodes determined during the type
      checking phase (that must precede IR conversion).
    constant_deps: Externally-noted constant dependencies that this function
      has (as free variables); noted via add_constant_dep().
    emit_positions: Whether or not we should emit position data based on the AST
      node source positions.
  """

  def __init__(self, package: ir_package.Package, module: ast.Module,
               type_info: type_info_mod.TypeInfo, emit_positions: bool):
    self.state = cpp_ir_converter.IrConverter(package, module, type_info,
                                              emit_positions)

  @property
  def fb(self):
    return self.state.function_builder

  @property
  def fileno(self):
    return self.state.fileno

  @property
  def module(self):
    return self.state.module

  @property
  def type_info(self):
    return self.state.type_info

  @property
  def package(self):
    return self.state.package

  @property
  def emit_positions(self):
    return self.state.emit_positions

  @property
  def last_expression(self):
    return self.state.last_expression

  @last_expression.setter
  def set_last_expression(self, value):
    self.state.last_expression = value

  def _extract_module_level_constants(self, m: ast.Module):
    """Populates `self.symbolic_bindings` with module-level constant values."""
    for constant in m.get_constants():
      if isinstance(constant.value, ast.Number):
        value = ast_helpers.get_value_as_int(constant.value)
        logging.vlog(
            3, 'Found module-level constant %s for symbolic bindings: %s',
            constant.name.identifier, value)
        self.state.set_symbolic_binding(constant.name.identifier, value)

  def add_constant_dep(self, constant: ast.Constant) -> None:
    self.state.add_constant_dep(constant)

  def _type_to_ir(self, concrete_type: ConcreteType) -> type_mod.Type:
    """Converts a concrete type to its corresponding IR representation."""
    assert isinstance(concrete_type, ConcreteType), concrete_type
    logging.vlog(4, 'Converting concrete type to IR: %s', concrete_type)
    if isinstance(concrete_type, ArrayType):
      element_type = self._type_to_ir(concrete_type.get_element_type())
      element_count = concrete_type.size.value
      if not isinstance(element_count, int):
        raise ValueError(
            'Expect array element count to be integer; got {!r}'.format(
                element_count))
      result = self.package.get_array_type(element_count, element_type)
      logging.vlog(
          4, 'Converted type to IR; concrete type: %s ir: %s element_count: %s',
          concrete_type, result, concrete_type.size)
      return result
    elif isinstance(concrete_type, BitsType) or isinstance(
        concrete_type, EnumType):
      return self.package.get_bits_type(
          concrete_type.get_total_bit_count().value)
    else:
      if not isinstance(concrete_type, TupleType):
        raise ValueError(
            'Expect type to be bits/enum, array, or tuple; got: '
            f'{concrete_type} ({concrete_type!r})')
      members = tuple(
          self._type_to_ir(m) for m in concrete_type.get_unnamed_members())
      return self.package.get_tuple_type(members)

  def _resolve_type_to_ir(self, node: ast.AstNode) -> type_mod.Type:
    concrete_type = self._resolve_type(node)
    assert isinstance(concrete_type, ConcreteType), concrete_type
    try:
      return self._type_to_ir(concrete_type)
    except ValueError as e:
      if 'Expect type to be' in str(e):
        raise ConversionError(f'Could not resolve type: {e}', node.span)
      if 'Expect array element count to be' in str(e):
        raise ConversionError(f'Could not resolve type: {e}', node.span)
      raise

  def _def(self, node: ast.AstNode, ir_func: Callable[..., BValue], *args,
           **kwargs) -> BValue:
    # TODO(leary): 2019-07-19 When everything is Python 3 we can switch to use a
    # single kwarg "span" after vararg with a normal pytype annotation.
    span = kwargs.pop('span', None)
    assert not kwargs
    assert isinstance(span,
                      (type(None), Span)), 'Expect span kwarg as Span or None.'

    if span is None and hasattr(node, 'span') and isinstance(node.span, Span):
      span = node.span
    loc = None
    if span is not None and self.emit_positions:
      start_pos = span.start
      lineno = fileno_mod.Lineno(start_pos.lineno)
      colno = fileno_mod.Colno(start_pos.colno)
      loc = source_location.SourceLocation(self.fileno, lineno, colno)

    try:
      ir = ir_func(*args, loc=loc)
    except TypeError:
      logging.error('Failed to call IR-creating function %s with args: %s',
                    ir_func, args)
      raise
    assert isinstance(
        ir, BValue), 'Expect ir_func to return a BValue; got {!r}'.format(ir)
    logging.vlog(4, 'Define node "%s" (%s) to be %s @ %s',
                 node, node.__class__.__name__, ir,
                 ast_helpers.get_span_or_fake(node))
    self.state.set_node_to_ir(node, ir)
    return ir

  def _def_const(self, node: ast.AstNode, value: int, bit_count: int) -> BValue:
    bits = _int_to_bits(value, bit_count)
    ir = self._def(node, self.fb.add_literal_bits, bits)
    self.state.set_node_to_ir(node, (bits_mod.from_long(value, bit_count), ir))
    return ir

  def _is_const(self, node: ast.AstNode) -> bool:
    """Returns whether "node" corresponds to a known-constant (int) value."""
    record = self.state.get_node_to_ir(node)
    return isinstance(record, tuple)

  def _get_const_bits(self, node: ast.AstNode) -> bits_mod.Bits:
    return self.state.get_const_bits(node)

  def _get_const(self, node: ast.AstNode, *, signed: bool) -> int:
    """Retrieves the known-constant (int) value associated with "node".

    Args:
      node: Node to retrieve the constant value for.
      signed: Whether or not the constant value should be converted as a signed
        number (to Python int type). That is, if all bits are set, should it be
        -1 or a positive value?

    Returns:
      A Python integer representing the resulting value. If signed is False,
      this value will always be >= 0.

    Raises:
      ConversionError: If the node was not registered as a constant during IR
        conversion.
    """
    bits = self._get_const_bits(node)
    result = bits.to_int() if signed else bits.to_uint()
    if not signed:
      assert result >= 0
    return result

  def _use(self, node: ast.AstNode) -> BValue:
    return self.state.use(node)

  def _def_alias(self, from_: ast.AstNode, to: ast.AstNode) -> BValue:
    return self.state.def_alias(from_, to)

  def _resolve_dim(self, dim: ConcreteTypeDim) -> ConcreteTypeDim:
    return self.state.resolve_dim(dim)

  def _resolve_type(self, node: ast.AstNode) -> ConcreteType:
    return self.state.resolve_type(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_TypeRef(self, node: ast.TypeRef) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_TypeRefTypeAnnotation(self,
                                  node: ast.TypeRefTypeAnnotation) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_ArrayTypeAnnotation(self, node: ast.ArrayTypeAnnotation) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_BuiltinTypeAnnotation(self,
                                  node: ast.BuiltinTypeAnnotation) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_TupleTypeAnnotation(self, node: ast.TupleTypeAnnotation) -> None:
    pass

  def visit_Ternary(self, node: ast.Ternary):
    self.state.handle_ternary(node)

  def visit_Binop(self, node: ast.Binop):
    self.state.handle_binop(node)

  def _next_counted_for_ordinal(self) -> int:
    return self.state.get_and_bump_counted_for_count()

  def _visit_matcher(self, matcher: ast.NameDefTree, index: Tuple[int, ...],
                     matched_value: BValue,
                     matched_type: ConcreteType) -> BValue:
    assert isinstance(matched_value, BValue), matched_value
    if matcher.is_leaf():
      leaf = matcher.get_leaf()
      logging.vlog(5, 'Matcher is leaf: %s (%s)', leaf, leaf.__class__.__name__)
      if isinstance(leaf, ast.WildcardPattern):
        return self._def(matcher, self.fb.add_literal_bits,
                         bits_mod.UBits(1, 1))
      elif isinstance(leaf, (ast.Number, ast.ColonRef)):
        visit(leaf, self)
        return self._def(matcher, self.fb.add_eq, self._use(leaf),
                         matched_value)
      elif isinstance(leaf, ast.NameRef):
        result = self._def(matcher, self.fb.add_eq, self._use(leaf.name_def),
                           matched_value)
        self._def_alias(leaf.name_def, to=leaf)
        return result
      else:
        assert isinstance(
            leaf, ast.NameDef
        ), 'Expected leaf to be wildcard, number, or name; got: {!r}'.format(
            leaf)
        ok = self._def(leaf, self.fb.add_literal_bits, bits_mod.UBits(1, 1))
        self.state.set_node_to_ir(matcher, matched_value)
        self.state.set_node_to_ir(leaf, matched_value)
        return ok
    else:
      ok = self.fb.add_literal_bits(bits_mod.UBits(value=1, bit_count=1))
      for i, (element, element_type) in enumerate(
          zip(matcher.tree, matched_type.get_unnamed_members())):  # pytype: disable=attribute-error
        # Extract the element.
        member = self.fb.add_tuple_index(matched_value, i)
        cond = self._visit_matcher(element, index + (i,), member, element_type)
        ok = self.fb.add_and(ok, cond)
      return ok

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Match(self, node: ast.Match):
    if (not node.arms or not node.arms[-1].patterns[0].is_irrefutable()):
      raise ConversionError(
          'Only matches with trailing irrefutable patterns are currently handled.',
          node.span)

    visit(node.matched, self)
    matched = self._use(node.matched)
    matched_type = self._resolve_type(node.matched)

    default_arm = node.arms[-1]
    assert len(default_arm.patterns) == 1, (
        'Multiple patterns in default arm is not yet implemented for IR '
        'conversion.')
    self._visit_matcher(default_arm.patterns[0], (len(node.arms) - 1,), matched,
                        matched_type)
    visit(default_arm.expr, self)

    arm_selectors = []
    arm_values = []
    for i, arm in enumerate(node.arms[:-1]):
      # Visit all the match patterns.
      this_arm_selectors = []
      for pattern in arm.patterns:
        selector = self._visit_matcher(pattern, (i,), matched, matched_type)
        this_arm_selectors.append(selector)

      # "Or" together the patterns, if necessary, to determine if the arm is
      # selected.
      if len(this_arm_selectors) > 1:
        arm_selectors.append(self.fb.add_nary_or(this_arm_selectors))
      else:
        arm_selectors.append(this_arm_selectors[0])
      visit(arm.expr, self)
      arm_values.append(self._use(arm.expr))

    # So now we have the following representation of the match arms:
    #   match x {
    #     42  => blah
    #     64  => snarf
    #     128 => yep
    #     _   => burp
    #   }
    #
    #   selectors:     [x==42, x==64, x==128]
    #   values:        [blah,  snarf,    yep]
    #   default_value: burp
    self.state.set_node_to_ir(
        node,
        self.fb.add_match_true(arm_selectors, arm_values,
                               self._use(default_arm.expr)))
    self.state.last_expression = node

  def visit_Unop(self, node: ast.Unop):
    self.state.handle_unop(node)

  def _visit_width_slice(self, node: ast.Index, width_slice: ast.WidthSlice,
                         lhs_type: ConcreteType) -> None:
    visit(width_slice.start, self)
    self._def(node, self.fb.add_dynamic_bit_slice, self._use(node.lhs),
              self._use(width_slice.start),
              self._resolve_type(node).get_total_bit_count().value)

  def visit_Attr(self, node: ast.Attr) -> None:
    self.state.handle_attr(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Index(self, node: ast.Index) -> None:
    visit(node.lhs, self)
    lhs_type = self.type_info.get_type(node.lhs)
    if isinstance(lhs_type, TupleType):
      visit(node.index, self)
      self._def(node, self.fb.add_tuple_index, self._use(node.lhs),
                self._get_const(node.index, signed=False))
    elif isinstance(lhs_type, BitsType):
      index_slice = node.index
      if isinstance(index_slice, ast.WidthSlice):
        return self._visit_width_slice(node, index_slice, lhs_type)
      assert isinstance(index_slice, ast.Slice), index_slice

      start, width = self.type_info.get_slice_start_width(
          index_slice, self._get_symbolic_bindings_tuple())
      self._def(node, self.fb.add_bit_slice, self._use(node.lhs), start, width)
    else:
      visit(node.index, self)
      self._def(node, self.fb.add_array_index, self._use(node.lhs),
                [self._use(node.index)])

  def visit_Number(self, node: ast.Number):
    type_ = self._resolve_type(node)
    bit_count = type_.get_total_bit_count().value
    assert isinstance(bit_count, int), bit_count
    self._def_const(node, ast_helpers.get_value_as_int(node), bit_count)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Constant(self, node: ast.Constant) -> None:
    visit(node.value, self)
    logging.vlog(5, 'Aliasing NameDef for constant: %r', node.name)
    self._def_alias(node.value, to=node.name)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Array(self, node: ast.Array) -> None:
    array_type = self._resolve_type(node)
    members = []
    for member in node.members:
      visit(member, self)
      members.append(self._use(member))
    if node.has_ellipsis:
      while len(members) < array_type.size.value:
        members.append(members[-1])
    self._def(node, self.fb.add_array, members, members[0].get_type())

  # Note: need to traverse to define constants for members.
  def visit_ConstantArray(self, node: ast.ConstantArray) -> None:
    self.state.handle_constant_array(node)

  def _cast_to_array(self, node: ast.Cast, output_type: ConcreteType) -> None:
    bits = self._use(node.expr)
    slices = []
    assert isinstance(output_type, ArrayType), output_type
    element_bit_count = output_type.get_element_type().get_total_bit_count(
    ).value
    # MSb becomes lowest-indexed array element.
    for i in range(0,
                   output_type.get_total_bit_count().value, element_bit_count):
      slices.append(self.fb.add_bit_slice(bits, i, element_bit_count))
    slices.reverse()
    element_type = self.package.get_bits_type(element_bit_count)
    self._def(node, self.fb.add_array, slices, element_type)

  def _cast_from_array(self, node: ast.Cast, output_type: ConcreteType) -> None:
    array = self._use(node.expr)
    array_type = self._resolve_type_to_ir(node.expr)
    pieces = []
    for i in range(array_type.get_size()):
      pieces.append(
          self.fb.add_array_index(
              array, [self.fb.add_literal_bits(bits_mod.UBits(i, 32))]))
    self._def(node, self.fb.add_concat, pieces)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Cast(self, node: ast.Cast) -> None:
    visit(node.expr, self)
    output_type = self._resolve_type(node)
    if isinstance(output_type, ArrayType):
      return self._cast_to_array(node, output_type)
    if not (isinstance(output_type, BitsType) or
            isinstance(output_type, EnumType)):
      raise NotImplementedError(
          'Cast can only handle bits output types; got: '
          f'{output_type} @ {node.span} ({output_type!r})')
    input_type = self._resolve_type(node.expr)
    if isinstance(input_type, ArrayType):
      return self._cast_from_array(node, output_type)
    new_bit_count = output_type.get_total_bit_count().value
    input_type = self._resolve_type(node.expr)
    if new_bit_count < input_type.get_total_bit_count().value:
      self._def(node, self.fb.add_bit_slice, self._use(node.expr), 0,
                new_bit_count)
    else:
      signed_input = input_type.signed
      assert signed_input is not None, input_type
      f = self.fb.add_signext if signed_input else self.fb.add_zeroext
      self._def(node, f, self._use(node.expr), new_bit_count)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_XlsTuple(self, node: ast.XlsTuple) -> None:
    for o in node.members:
      visit(o, self)
    operands = tuple(self._use(o) for o in node.members)
    self._def(node, self.fb.add_tuple, operands)

  def _deref_struct_or_enum(
      self, node: Union[ast.StructDef, ast.TypeDef, ast.EnumDef]
  ) -> Union[ast.StructDef, ast.EnumDef]:
    while isinstance(node, ast.TypeDef):
      annotation = node.type_
      if not isinstance(annotation, ast.TypeRefTypeAnnotation):
        raise NotImplementedError(
            'Unhandled typedef for resolving to struct-or-enum: %s' %
            annotation)
      node = annotation.type_ref.type_def

    if isinstance(node, (ast.StructDef, ast.EnumDef)):
      return node

    if isinstance(node, ast.NameRef):
      logging.vlog(3, 'Resolving NameRef %s to struct or enum', node)
      definer = node.name_def.definer
      assert isinstance(
          definer,
          (ast.StructDef, ast.TypeDef, ast.EnumDef)), (definer, type(definer))
      return self._deref_struct_or_enum(definer)

    if not isinstance(node, ast.ColonRef):
      raise TypeError(f'Cannot resolve enum for node: {node!r}')

    import_node = node.subject.name_def.definer
    imported_mod, _ = dict(self.type_info.get_imports())[import_node]
    td = imported_mod.get_type_definition_by_name()[node.attr]
    # Recurse to resolve the typedef within the imported module.
    td = self._deref_struct_or_enum(td)
    assert isinstance(td, (ast.StructDef, ast.EnumDef)), td
    return td

  def _deref_struct(self, node: Union[ast.StructDef,
                                      ast.ColonRef]) -> ast.StructDef:
    result = self._deref_struct_or_enum(node)
    assert isinstance(result, ast.StructDef), result
    return result

  def _deref_enum(self, node: Union[ast.EnumDef, ast.ColonRef]) -> ast.EnumDef:
    result = self._deref_struct_or_enum(node)
    assert isinstance(result, ast.EnumDef), result
    return result

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_SplatStructInstance(self, node: ast.SplatStructInstance) -> None:
    visit(node.splatted, self)
    orig = self._use(node.splatted)
    updates = {}
    for k, e in node.members:
      visit(e, self)
      updates[k] = self._use(e)
    struct = self._deref_struct(node.struct)

    members = []
    for i, k in enumerate(struct.member_names):
      if k in updates:
        members.append(updates[k])
      else:
        members.append(self.fb.add_tuple_index(orig, i))
    self._def(node, self.fb.add_tuple, members)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_StructInstance(self, node: ast.StructInstance) -> None:
    operands = []
    struct = self._deref_struct(node.struct)
    for _, m in node.get_ordered_members(struct):
      visit(m, self)
      operands.append(self._use(m))
    operands = tuple(operands)
    self._def(node, self.fb.add_tuple, operands)

  def _is_constant_zero(self, node: ast.AstNode) -> bool:
    return isinstance(node,
                      ast.Number) and ast_helpers.get_value_as_int(node) == 0

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_For(self, node: ast.For) -> None:
    visit(node.init, self)

    def query_const_range_call() -> int:
      """Returns trip count if this is a `for ... in range(CONST)` construct."""
      range_callee = (
          isinstance(node.iterable, ast.Invocation) and
          isinstance(node.iterable.callee, ast.NameRef) and
          node.iterable.callee.identifier == 'range')
      if not range_callee:
        raise ConversionError(
            'For-loop is of an unsupported form for IR conversion; only a '
            "'range(0, const)' call is supported, found non-range callee.",
            node.span)
      if len(node.iterable.args) != 2:
        raise ConversionError(
            'For-loop is of an unsupported form for IR conversion; only a '
            "'range(0, const)' call is supported, found inappropriate number "
            'of arguments.', node.span)
      if not self._is_constant_zero(node.iterable.args[0]):
        raise ConversionError(
            'For-loop is of an unsupported form for IR conversion; only a '
            "'range(0, const)' call is supported, found inappropriate number "
            'of arguments.', node.span)
      arg = node.iterable.args[1]
      visit(arg, self)
      if not self._is_const(arg):
        raise ConversionError(
            'For-loop is of an unsupported form for IR conversion; only a '
            "'range(const)' call is supported, did not find a const value "
            f'for {arg} ({arg!r}).', node.span)
      return self._get_const(arg, signed=False)

    # TODO(leary): We currently only support counted loops of the form:
    #
    #   for (i, ...): (u32, ...) in range(N) {
    #      ...
    #   }
    trip_count = query_const_range_call()

    logging.vlog(3, 'Converting for-loop @ %s', node.span)
    body_converter = _IrConverterFb(
        self.package,
        self.module,
        self.type_info,
        emit_positions=self.emit_positions)
    body_converter.state.set_symbolic_bindings(
        self.state.get_symbolic_bindings())
    body_fn_name = ('__' + self.fb.name + '_counted_for_{}_body').format(
        self._next_counted_for_ordinal()).replace('.', '_')
    body_converter.state.instantiate_function_builder(body_fn_name)
    flat = node.names.flatten1()
    assert len(
        flat
    ) == 2, 'Expect an induction binding and loop carry binding; got {!r}'.format(
        flat)

    # Add the induction value.
    assert isinstance(
        flat[0], ast.NameDef
    ), 'Induction variable was not a NameDef: {0} ({0!r})'.format(flat[0])
    body_converter.state.set_node_to_ir(
        flat[0],
        body_converter.fb.add_param(flat[0].identifier,
                                    self._resolve_type_to_ir(flat[0])))

    # Add the loop carry value.
    if isinstance(flat[1], ast.NameDef):
      body_converter.state.set_node_to_ir(
          flat[1],
          body_converter.fb.add_param(flat[1].identifier,
                                      self._resolve_type_to_ir(flat[1])))
    else:
      # For tuple loop carries we have to destructure names on entry.
      carry_type = self._resolve_type_to_ir(flat[1])
      carry = body_converter.fb.add_param('__loop_carry', carry_type)
      body_converter.state.set_node_to_ir(flat[1], carry)
      body_converter._visit_matcher(  # pylint: disable=protected-access
          flat[1], (), carry, self._resolve_type(flat[1]))

    # Free variables are suffixes on the function parameters.
    freevars = node.body.get_free_variables(node.span.start)
    freevars = freevars.drop_builtin_defs()
    relevant_name_defs = []
    for name_def in freevars.get_name_defs(self.module):
      try:
        type_ = self.type_info.get_type(name_def)
      except type_info_mod.TypeMissingError:
        continue
      if isinstance(type_, FunctionType):
        continue
      if isinstance(name_def.definer, ast.EnumDef):
        continue
      if isinstance(name_def.definer, ast.TypeDef):
        continue
      relevant_name_defs.append(name_def)
      logging.vlog(3, 'Converting freevar name: %s', name_def)
      body_converter.state.set_node_to_ir(
          name_def,
          body_converter.fb.add_param(name_def.identifier,
                                      self._resolve_type_to_ir(name_def)))

    visit(node.body, body_converter)
    body_function = body_converter.fb.build()
    logging.vlog(3, 'Converted body function: %s', body_function.name)

    stride = 1
    invariant_args = tuple(
        self._use(name_def)
        for name_def in relevant_name_defs
        if not isinstance(self.type_info.get_type(name_def), FunctionType))
    self._def(node, self.fb.add_counted_for, self._use(node.init), trip_count,
              stride, body_function, invariant_args)

  def _get_symbolic_bindings_tuple(self) -> SymbolicBindings:
    # We only consider function symbolic bindings for invocations.
    # The typechecker doesn't care about module-level constants.
    module_level_constants = {
        c.name.identifier
        for c in self.module.get_constants()
        if isinstance(c.value, ast.Number)
    }
    return SymbolicBindings(
        tuple((k, v)
              for k, v in self.state.get_symbolic_bindings_items()
              if k not in module_level_constants))

  def _get_invocation_bindings(self,
                               invocation: ast.Invocation) -> SymbolicBindings:
    """Returns the symbolic bindings of the invocation.

    We must provide the current evaluation context (module name, function name,
    symbolic bindings) in order to retrieve the correct symbolic bindings to use
    in the invocation.

    Args:
      invocation: Invocation that the bindings are being retrieved for.

    Returns:
      The symbolic bindings for the given invocation.
    """

    key = self._get_symbolic_bindings_tuple()
    return self.type_info.get_invocation_symbolic_bindings(invocation, key)

  def _get_callee_identifier(self, node: ast.Invocation) -> Text:
    logging.vlog(3, 'Getting callee identifier for invocation: %s', node)
    if isinstance(node.callee, ast.NameRef):
      callee_name = node.callee.identifier
      m = self.module
    elif isinstance(node.callee, ast.ColonRef):
      m = dict(
          self.type_info.get_imports())[node.callee.subject.name_def.definer][0]
      callee_name = node.callee.attr
    else:
      raise NotImplementedError('Callee not currently supported @ {}'.format(
          node.span))
    try:
      function = m.get_function(callee_name)
    except KeyError:
      # For e.g. builtins that are not in the module we just provide the name
      # directly.
      return callee_name
    if not function.is_parametric():
      return cpp_ir_converter.mangle_dslx_name(
          function.name.identifier, function.get_free_parametric_keys(), m,
          None)
    resolved_symbolic_bindings = self._get_invocation_bindings(node)
    logging.vlog(2, 'Node %s @ %s symbolic bindings %r', node, node.span,
                 resolved_symbolic_bindings)
    assert resolved_symbolic_bindings, node
    return cpp_ir_converter.mangle_dslx_name(
        function.name.identifier, function.get_free_parametric_keys(), m,
        resolved_symbolic_bindings)

  def _def_map_with_builtin(self, parent_node: ast.Invocation,
                            node: ast.NameRef, arg: ast.AstNode,
                            symbolic_bindings: SymbolicBindings) -> BValue:
    """Makes the specified builtin available to the package."""
    assert isinstance(symbolic_bindings, SymbolicBindings), symbolic_bindings
    mangled_name = cpp_ir_converter.mangle_dslx_name(node.name_def.identifier,
                                                     set(), self.module,
                                                     symbolic_bindings)

    arg = self._use(arg)
    if mangled_name not in self.package.get_function_names():
      fb = function_builder.FunctionBuilder(mangled_name, self.package)
      param = fb.add_param('arg', arg.get_type().get_element_type())
      builtin_name = node.name_def.identifier
      assert builtin_name in cpp_dslx_builtins.UNARY_PARAMETRIC_BUILTIN_NAMES, cpp_dslx_builtins.UNARY_PARAMETRIC_BUILTIN_NAMES
      fbuilds = {'clz': fb.add_clz, 'ctz': fb.add_ctz}
      assert set(fbuilds.keys()
                ) == cpp_dslx_builtins.UNARY_PARAMETRIC_BUILTIN_NAMES, set(
                    fbuilds.keys())
      fbuilds[builtin_name](param)
      fb.build()
    return self._def(parent_node, self.fb.add_map, arg,
                     self.package.get_function(mangled_name))

  def _visit_map(self, node: ast.Invocation) -> BValue:
    for arg in node.args[:-1]:
      visit(arg, self)
    arg = self._use(node.args[0])
    fn_node = node.args[1]

    if isinstance(fn_node, ast.NameRef):
      map_fn_name = fn_node.name_def.identifier
      if map_fn_name in cpp_dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
        return self._def_map_with_builtin(node, fn_node, node.args[0],
                                          self._get_invocation_bindings(node))
      else:
        lookup_module = self.module
        fn = lookup_module.get_function(map_fn_name)
    elif isinstance(fn_node, ast.ColonRef):
      map_fn_name = fn_node.attr
      imports = dict(self.type_info.get_imports())
      import_node = fn_node.subject.name_def.definer
      lookup_module, _ = imports[import_node]
      fn = lookup_module.get_function(map_fn_name)
    else:
      raise NotImplementedError(
          'Unhandled function mapping: {!r}'.format(fn_node))

    node_sym_bindings = self._get_invocation_bindings(node)
    mangled_name = cpp_ir_converter.mangle_dslx_name(
        fn.name.identifier, fn.get_free_parametric_keys(), lookup_module,
        node_sym_bindings)

    return self._def(node, self.fb.add_map, arg,
                     self.package.get_function(mangled_name))

  def _visit_scmp(self, node: ast.Invocation, args: Tuple[BValue, ...],
                  which: Text) -> BValue:
    lhs, rhs = args
    return self._def(node, getattr(self.fb, 'add_{}'.format(which)), lhs, rhs)

  def _visit_trace(self, node: ast.Invocation, args: Tuple[BValue,
                                                           ...]) -> BValue:
    return self._def(node, self.fb.add_identity, args[0])

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Invocation(self, node: ast.Invocation):
    called_name = self._get_callee_identifier(node)

    def accept_args() -> Tuple[BValue, ...]:
      for arg in node.args:
        visit(arg, self)
      return tuple(self._use(arg) for arg in node.args)

    if called_name == 'fail!':
      args = accept_args()
      assert len(node.args) == 1, ('Expect fail! builtin to only accept a '
                                   'single argument; got: {!r}'.format(args))
      self._def(node, self.fb.add_identity, args[0])
    elif called_name == 'update':
      accept_args()
      self.state.handle_builtin_update(node)
    elif called_name == 'signex':
      accept_args()
      self.state.handle_builtin_signex(node)
    elif called_name == 'clz':
      accept_args()
      self.state.handle_builtin_clz(node)
    elif called_name == 'ctz':
      accept_args()
      self.state.handle_builtin_ctz(node)
    elif called_name == 'map':
      # Map needs special arg handling, so we handle that inside.
      self._visit_map(node)
    elif called_name == 'one_hot':
      accept_args()
      self.state.handle_builtin_one_hot(node)
    elif called_name == 'one_hot_sel':
      accept_args()
      self.state.handle_builtin_one_hot_sel(node)
    elif called_name == 'bit_slice':
      accept_args()
      self.state.handle_builtin_bit_slice(node)
    elif called_name == 'rev':
      accept_args()
      self.state.handle_builtin_rev(node)
    elif called_name in ('sgt', 'sge', 'slt', 'sle'):
      self._visit_scmp(node, accept_args(), called_name)
    elif called_name == 'and_reduce':
      accept_args()
      self.state.handle_builtin_and_reduce(node)
    elif called_name == 'or_reduce':
      accept_args()
      self.state.handle_builtin_or_reduce(node)
    elif called_name == 'xor_reduce':
      accept_args()
      self.state.handle_builtin_xor_reduce(node)
    elif called_name == 'trace':
      self._visit_trace(node, accept_args())
    else:
      try:
        f = self.package.get_function(called_name)
      except Exception as e:
        # TODO(leary): Switch to new pybind11 more-specific exception.
        raise ConversionError(
            'Failed to get function from invocation: {}'.format(e), node.span)
      self._def(node, self.fb.add_invoke, accept_args(), f)

  def visit_ConstRef(self, node: ast.ConstRef) -> None:
    self._def_alias(node.name_def, to=node)

  def visit_NameRef(self, node: ast.NameRef) -> None:
    self._def_alias(node.name_def, node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_ColonRef(self, node: ast.ColonRef) -> None:
    # Implementation note: ColonRef *invocations* are handled in Invocation (by
    # resolving the mangled callee name, which should have been IR converted in
    # dependency order).

    if isinstance(node.subject, ast.NameRef) and isinstance(
        node.subject.name_def.definer, ast.Import):
      definer = node.subject.name_def.definer
      imported_mod, _ = dict(self.type_info.get_imports())[definer]
      constant_def = imported_mod.get_constant_by_name()[node.attr]
      assert isinstance(constant_def, ast.Constant), repr(constant_def)
      self.visit_Constant(constant_def)
      self._def_alias(constant_def.name, to=node)
      return

    try:
      enum = self._deref_enum(node.subject)
    except TypeError as e:
      logging.vlog(3, 'ColonRef was not an enum ref: %s @ %s', node, node.span)
      if 'Cannot resolve enum for node' in str(e):
        return
      raise
    value = enum.get_value(node.attr)
    visit(value, self)
    self._def_alias(from_=value, to=node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Let(self, node: ast.Let):
    visit(node.rhs, self)
    if node.name_def_tree.is_leaf():
      self._def_alias(node.rhs, to=node.name_def_tree.get_leaf())
      visit(node.body, self)
      self._def_alias(node.body, node)
    else:
      # Walk the tree performing tuple_index operations to get to the binding
      # levels desired.

      names = [self._use(node.rhs)]  # List[BValue]

      def walk(x: ast.NameDefTree, level: int, index: int) -> None:
        """Invoked at each level of the name def tree.

        Binds the name in the name def tree to the corresponding value being
        pattern matched.

        Args:
          x: The current level of the NameDefTree.
          level: Level in the NameDefTree (root is 0).
          index: Index of node in the current tree level (e.g. leftmost is 0).
        """
        del names[level:]
        names.append(
            self._def(
                x,
                self.fb.add_tuple_index,
                names[-1],
                index,
                span=(x.get_leaf().span if x.is_leaf() else x.span)))
        if x.is_leaf():
          self._def_alias(x, x.get_leaf())

      ast_helpers.do_preorder(node.name_def_tree, walk)
      visit(node.body, self)
      self._def_alias(node.body, to=node)

    if self.state.last_expression is None:
      self.state.last_expression = node.body

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Param(self, node: ast.Param):
    self._def(node.name, self.fb.add_param, node.name.identifier,
              self._resolve_type_to_ir(node.type_))

  def _visit_Function(
      self, node: ast.Function,
      symbolic_bindings: Optional[SymbolicBindings]) -> ir_function.Function:
    self.state.set_symbolic_bindings(symbolic_bindings)
    self._extract_module_level_constants(self.module)
    # We use a function builder for the duration of converting this
    # ast.Function. When it's done being built, we drop the reference to it (by
    # setting self.fb to None).
    mangled_name = cpp_ir_converter.mangle_dslx_name(
        node.name.identifier, node.get_free_parametric_keys(), self.module,
        symbolic_bindings)
    self.state.instantiate_function_builder(mangled_name)

    for param in node.params:
      visit(param, self)

    for parametric_binding in node.parametric_bindings:
      logging.vlog(4, 'Resolving parametric binding %s', parametric_binding)

      sb_value = self.state.get_symbolic_binding(
          parametric_binding.name.identifier)
      value = self._resolve_dim(ConcreteTypeDim(sb_value))
      assert isinstance(value.value, int), \
          'Expect integral parametric binding; got {!r}'.format(value)
      self._def_const(
          parametric_binding, value.value,
          self._resolve_type(
              parametric_binding.type_).get_total_bit_count().value)
      self._def_alias(parametric_binding, to=parametric_binding.name)

    for dep in self.state.constant_deps:
      visit(dep, self)
    self.state.clear_constant_deps()

    visit(node.body, self)

    last_expression = self.state.last_expression or node.body
    if isinstance(last_expression, ast.NameRef):
      self._def(last_expression, self.fb.add_identity,
                self._use(last_expression))
    f = self.fb.build()
    logging.vlog(3, 'Built function: %s', f.name)
    verifier_mod.verify_function(f)
    return f

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Function(
      self,
      node: ast.Function,
      symbolic_bindings: Optional[SymbolicBindings] = None,
      module: Optional[ast.Module] = None) -> ir_function.Function:
    return self._visit_Function(node, symbolic_bindings)

  def get_text(self) -> Text:
    return self.package.dump_ir()


def _convert_one_function(package: ir_package.Package,
                          module: ast.Module,
                          function: ast.Function,
                          type_info: type_info_mod.TypeInfo,
                          symbolic_bindings: Optional[SymbolicBindings] = None,
                          emit_positions: bool = True) -> Text:
  """Converts a single function into its emitted text form.

  Args:
    package: IR package we're converting the function into.
    module: Module we're converting a function within.
    function: Function we're converting.
    type_info: Type information about module from the typechecking phase.
    symbolic_bindings: Parametric bindings to use during conversion, if this
      function is parametric.
    emit_positions: Whether to emit position information into the IR based on
      the AST's source positions.

  Returns:
    The converted IR function text.
  """
  function_by_name = module.get_function_by_name()
  constant_by_name = module.get_constant_by_name()
  type_definition_by_name = module.get_type_definition_by_name()
  import_by_name = module.get_import_by_name()
  converter = _IrConverterFb(
      package, module, type_info, emit_positions=emit_positions)

  freevars = function.body.get_free_variables(
      function.span.start).get_name_def_tups(module)
  logging.vlog(2, 'Unfiltered free variables for function %s: %s',
               function.identifier, freevars)
  logging.vlog(3, 'Type definition by name: %r', type_definition_by_name)
  for identifier, name_def in freevars:
    if (identifier in function_by_name or
        identifier in type_definition_by_name or identifier in import_by_name or
        isinstance(name_def, ast.BuiltinNameDef)):
      pass
    elif identifier in constant_by_name:
      converter.add_constant_dep(constant_by_name[identifier])
    else:
      raise NotImplementedError(
          f'Cannot convert free variable: {identifier}; not a function nor constant'
      )

  symbolic_binding_keys = set(b.identifier for b in symbolic_bindings or ())
  f_parametric_keys = function.get_free_parametric_keys()
  if f_parametric_keys > symbolic_binding_keys:
    raise ValueError(
        'Not enough symbolic bindings to convert function {!r}; need {!r} got {!r}'
        .format(function.name.identifier, f_parametric_keys,
                symbolic_binding_keys))

  logging.vlog(3, 'Converting function: %s; symbolic bindings: %s', function,
               symbolic_bindings)
  f = converter.visit_Function(function, symbolic_bindings)
  return f.dump_ir(recursive=False)


def convert_module_to_package(
    module: ast.Module,
    type_info: type_info_mod.TypeInfo,
    emit_positions: bool = True,
    traverse_tests: bool = False) -> ir_package.Package:
  """Converts the contents of a module to IR form.

  Args:
    module: Module to convert.
    type_info: Concrete type information used in conversion.
    emit_positions: Whether to emit positional metadata into the output IR.
    traverse_tests: Whether to convert functions called in DSLX test constructs.
      Note that this does NOT convert the test constructs themselves.

  Returns:
    The IR package that corresponds to this module.
  """
  emitted = []  # type: List[Text]
  package = ir_package.Package(module.name)
  order = extract_conversion_order.get_order(module, type_info,
                                             dict(type_info.get_imports()),
                                             traverse_tests)
  logging.vlog(3, 'Convert order: %s', pprint.pformat(order))
  for record in order:
    logging.vlog(1, 'Converting to IR: %r', record)
    emitted.append(
        _convert_one_function(
            package,
            record.m,
            record.f,
            record.type_info,
            symbolic_bindings=record.bindings,
            emit_positions=emit_positions))

  verifier_mod.verify_package(package)
  return package


def convert_module(module: ast.Module,
                   type_info: type_info_mod.TypeInfo,
                   emit_positions: bool = True) -> Text:
  """Same as convert_module_to_package, but converts to IR text."""
  return convert_module_to_package(module, type_info, emit_positions).dump_ir()


def convert_one_function(module: ast.Module,
                         entry_function_name: Text,
                         type_info: type_info_mod.TypeInfo,
                         emit_positions: bool = True) -> Text:
  """Returns function named entry_function_name in module as IR text."""
  logging.vlog(1, 'IR-converting entry function: %r', entry_function_name)
  package = ir_package.Package(module.name)
  _convert_one_function(
      package,
      module,
      module.get_function(entry_function_name),
      type_info,
      emit_positions=emit_positions)
  return package.dump_ir()
