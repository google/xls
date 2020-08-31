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

"""Module for converting AST to IR text dumps."""

import pprint
from typing import Text, List, Dict, Optional, Tuple, Callable, Union

from absl import logging

from xls.common.xls_error import XlsError
from xls.dslx import ast
from xls.dslx import bit_helpers
from xls.dslx import deduce
from xls.dslx import dslx_builtins
from xls.dslx import extract_conversion_order
from xls.dslx import type_info as type_info_mod
from xls.dslx.concrete_type import ArrayType
from xls.dslx.concrete_type import BitsType
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.concrete_type import EnumType
from xls.dslx.concrete_type import FunctionType
from xls.dslx.concrete_type import TupleType
from xls.dslx.ir_name_mangler import mangle_dslx_name
from xls.dslx.parametric_expression import ParametricExpression
from xls.dslx.parametric_instantiator import SymbolicBindings
from xls.dslx.span import PositionalError
from xls.dslx.span import Span
from xls.ir.python import bits as bits_mod
from xls.ir.python import fileno as fileno_mod
from xls.ir.python import function as ir_function
from xls.ir.python import function_builder
from xls.ir.python import lsb_or_msb
from xls.ir.python import number_parser
from xls.ir.python import package as ir_package
from xls.ir.python import source_location
from xls.ir.python import type as type_mod
from xls.ir.python import value as ir_value
from xls.ir.python import verifier as verifier_mod
from xls.ir.python.function_builder import BValue


class ParametricConversionError(XlsError):
  """Raised when we attempt to IR convert a parametric function."""


class ConversionError(PositionalError):
  """Raised on an issue converting to IR at a particular file position."""


def _int_to_bits(value: int, bit_count: int) -> bits_mod.Bits:
  """Converts a Python arbitrary precision int to a Bits type."""
  if bit_count <= 64:
    return bits_mod.UBits(value, bit_count) if value >= 0 else bits_mod.SBits(
        value, bit_count)
  return number_parser.bits_from_string(
      bit_helpers.to_hex_string(value, bit_count), bit_count=bit_count)


class _IrConverterFb(ast.AstVisitor):
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
    _constant_deps: Externally-noted constant dependencies that this function
      has (as free variables); noted via add_constant_dep().
    emit_positions: Whether or not we should emit position data based on the AST
      node source positions.
  """

  def __init__(self, package: ir_package.Package, module: ast.Module,
               type_info: type_info_mod.TypeInfo, emit_positions: bool):
    self.module = module
    self.type_info = type_info
    self.emit_positions = emit_positions
    self.package = package
    self.node_to_ir = {
    }  # type: Dict[ast.AstNode, Union[BValue, Tuple[int, BValue]]]
    self.symbolic_bindings = {}  # type: Dict[Text, int]
    self._constant_deps = []
    self.fb = None  # Optional[function_builder.FunctionBuilder]
    # TODO(leary): 2019-07-19 Create a way to get the file path from the module.
    self.fileno = self.package.get_or_create_fileno('fake_file.x')
    # Number of "counted for" nodes we've observed in this function.
    self.counted_for_count = 0
    self.last_expression = None  # Optional[ast.Expr]

  def _extract_module_level_constants(self, m: ast.Module):
    """Populates `self.symbolic_bindings` with module-level constant values."""
    for constant in m.get_constants():
      if isinstance(constant.value, ast.Number):
        value = constant.value.get_value_as_int()
        logging.vlog(
            3, 'Found module-level constant %s for symbolic bindings: %s',
            constant.name.identifier, value)
        self.symbolic_bindings[constant.name.identifier] = value
    logging.vlog(2, 'Symbolic bindings now: %s', self.symbolic_bindings)

  def add_constant_dep(self, constant: ast.Constant) -> None:
    logging.vlog(2, 'Adding constant dep: %s', constant)
    self._constant_deps.append(constant)

  def _type_to_ir(self, concrete_type: ConcreteType) -> type_mod.Type:
    """Converts a concrete type to its corresponding IR representation."""
    assert isinstance(concrete_type, ConcreteType), concrete_type
    logging.vlog(4, 'Converting concrete type to IR: %s', concrete_type)
    if isinstance(concrete_type, ArrayType):
      element_type = self._type_to_ir(concrete_type.get_element_type())
      element_count = concrete_type.size
      if not isinstance(element_count, int):
        raise ValueError(
            'Expect array element count to be integer; got {!r}'.format(
                element_count))
      result = self.package.get_array_type(element_count, element_type)
      logging.vlog(
          4, 'Converted type to IR; concrete type: %s ir: %s element_count: %d',
          concrete_type, result, concrete_type.size)
      return result
    elif isinstance(concrete_type, BitsType) or isinstance(
        concrete_type, EnumType):
      return self.package.get_bits_type(concrete_type.get_total_bit_count())
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
        raise ConversionError('Could not resolve type: {}'.format(e), node.span)
      if 'Expect array element count to be' in str(e):
        raise ConversionError('Could not resolve type: {}'.format(e), node.span)
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
    logging.vlog(4, 'Define node "%s" to be %s', node, ir)
    self.node_to_ir[node] = ir
    return ir

  def _def_alias(self, from_: ast.AstNode, to: ast.AstNode) -> BValue:
    self.node_to_ir[to] = self.node_to_ir[from_]
    logging.vlog(4, 'Alias node "%s" to be same as %s', to, from_)
    return self._use(to)

  def _def_const(self, node: ast.AstNode, value: int, bit_count: int) -> BValue:
    bits = _int_to_bits(value, bit_count)
    ir = self._def(node, self.fb.add_literal_bits, bits)
    self.node_to_ir[node] = (value, ir)
    return ir

  def _is_const(self, node: ast.AstNode) -> bool:
    """Returns whether "node" corresponds to a known-constant (int) value."""
    record = self.node_to_ir[node]
    return isinstance(record, tuple)

  def _get_const(self, node: ast.AstNode) -> int:
    """Retrieves the known-constant (int) value associated with "node"."""
    record = self.node_to_ir[node]
    if not isinstance(record, tuple):
      raise ConversionError(
          'Expected a constant, but value does not appear constant.',
          node.get_span_or_fake())
    return record[0]

  def _use(self, node: ast.AstNode) -> BValue:
    record = self.node_to_ir[node]
    if isinstance(record, tuple):
      return record[1]
    else:
      return record

  def _resolve_dim(self, dim):
    while isinstance(dim, ParametricExpression):
      try:
        orig = dim
        dim = dim.evaluate(self.symbolic_bindings)
        logging.vlog(4, 'Evaluated dim %s to %s via %s', orig, dim,
                     self.symbolic_bindings)
      except KeyError:
        logging.vlog(4, 'Could not resolve %s dim %s via symbolic bindings %r',
                     dim, self.symbolic_bindings)
        return dim
    return dim

  def _resolve_type(self, node: ast.AstNode) -> ConcreteType:
    try:
      concrete_type = self.type_info[node]
    except deduce.TypeMissingError:
      raise ConversionError(
          f'Failed to convert to IR because type was missing for AST node: '
          f'{node} :: {node!r}', node.get_span_or_fake())

    assert isinstance(concrete_type, ConcreteType), concrete_type
    result = concrete_type.map_size(self._resolve_dim)
    logging.vlog(4, 'Resolved concrete type from %s to %s via %s',
                 concrete_type, result, self.symbolic_bindings)
    return result

  @ast.AstVisitor.no_auto_traverse
  def visit_ArrayTypeAnnotation(self, node: ast.ArrayTypeAnnotation) -> None:
    pass

  @ast.AstVisitor.no_auto_traverse
  def visit_TupleTypeAnnotation(self, node: ast.TupleTypeAnnotation) -> None:
    pass

  def visit_Ternary(self, node: ast.Ternary):
    self._def(node, self.fb.add_sel, self._use(node.test),
              self._use(node.consequent), self._use(node.alternate))

  def _visit_concat(self, node: ast.Binop):
    output_type = self._resolve_type(node)
    lhs, rhs = self._use(node.lhs), self._use(node.rhs)
    if isinstance(output_type, BitsType):
      self._def(node, self.fb.add_concat, (lhs, rhs))
      return

    # Array concat case: since we don't currently have an array_concat
    # operation (see https://github.com/google/xls/issues/72) in the IR we
    # gather up all the lhs and rhs elements and form an array from them.
    assert isinstance(output_type, ArrayType), output_type
    element_type = output_type.get_element_type()
    lhs_type = self._resolve_type(node.lhs)
    rhs_type = self._resolve_type(node.rhs)
    vals = []
    for i in range(lhs_type.size):
      vals.append(
          self.fb.add_array_index(
              lhs,
              self.fb.add_literal_bits(bits_mod.UBits(value=i, bit_count=32))))
    for i in range(rhs_type.size):
      vals.append(
          self.fb.add_array_index(
              rhs,
              self.fb.add_literal_bits(bits_mod.UBits(value=i, bit_count=32))))
    self._def(node, self.fb.add_array, vals, self._type_to_ir(element_type))

  def visit_Binop(self, node: ast.Binop):
    lhs_type = self.type_info[node.lhs]
    signed_input = isinstance(lhs_type, BitsType) and lhs_type.signed

    # Concat is handled specially since the array-typed operation has no
    # directly corresponding IR operation.
    # See https://github.com/google/xls/issues/72
    if node.kind == ast.BinopKind.CONCAT:
      return self._visit_concat(node)

    f = {
        # Arithmetic.
        ast.BinopKind.ADD:
            self.fb.add_add,
        ast.BinopKind.SUB:
            self.fb.add_sub,
        ast.BinopKind.MUL:
            self.fb.add_smul if signed_input else self.fb.add_umul,
        ast.BinopKind.DIV:
            self.fb.add_udiv,
        # Comparisons.
        ast.BinopKind.EQ:
            self.fb.add_eq,
        ast.BinopKind.NE:
            self.fb.add_ne,
        ast.BinopKind.GE:
            self.fb.add_sge if signed_input else self.fb.add_uge,
        ast.BinopKind.GT:
            self.fb.add_sgt if signed_input else self.fb.add_ugt,
        ast.BinopKind.LE:
            self.fb.add_sle if signed_input else self.fb.add_ule,
        ast.BinopKind.LT:
            self.fb.add_slt if signed_input else self.fb.add_ult,
        # Shifts.
        ast.BinopKind.SHRL:
            self.fb.add_shrl,
        ast.BinopKind.SHLL:
            self.fb.add_shll,
        ast.BinopKind.SHRA:
            self.fb.add_shra,
        # Bitwise.
        ast.BinopKind.XOR:
            self.fb.add_xor,
        ast.BinopKind.AND:
            self.fb.add_and,
        ast.BinopKind.OR:
            self.fb.add_or,
        # Logical.
        ast.BinopKind.LOGICAL_AND:
            self.fb.add_and,
        ast.BinopKind.LOGICAL_OR:
            self.fb.add_or,
    }[node.kind]

    self._def(node, f, self._use(node.lhs), self._use(node.rhs))

  def _next_counted_for_ordinal(self) -> int:
    result = self.counted_for_count
    self.counted_for_count += 1
    return result

  def _visit_matcher(self, matcher: ast.NameDefTree, index: Tuple[int, ...],
                     matched_value: BValue,
                     matched_type: ConcreteType) -> BValue:
    if matcher.is_leaf():
      leaf = matcher.get_leaf()
      if isinstance(leaf, ast.WildcardPattern):
        return self._def(matcher, self.fb.add_literal_bits,
                         bits_mod.UBits(1, 1))
      elif isinstance(leaf, (ast.Number, ast.EnumRef)):
        leaf.accept(self)
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
        self.node_to_ir[matcher] = self.node_to_ir[leaf] = matched_value
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

  @ast.AstVisitor.no_auto_traverse
  def visit_Match(self, node: ast.Match):
    if (not node.arms or not node.arms[-1].patterns[0].is_irrefutable()):
      raise ConversionError(
          'Only matches with trailing irrefutable patterns are currently handled.',
          node.span)

    node.matched.accept(self)
    matched = self._use(node.matched)
    matched_type = self._resolve_type(node.matched)

    default_arm = node.arms[-1]
    assert len(default_arm.patterns) == 1, (
        'Multiple patterns in default arm is not yet implemented for IR '
        'conversion.')
    self._visit_matcher(default_arm.patterns[0], (len(node.arms) - 1,), matched,
                        matched_type)
    default_arm.expr.accept(self)

    arm_selectors = []
    arm_values = []
    for i, arm in enumerate(node.arms[:-1]):
      this_arm_selectors = []
      for pattern in arm.patterns:
        selector = self._visit_matcher(pattern, (i,), matched, matched_type)
        this_arm_selectors.append(selector)
      if len(this_arm_selectors) > 1:
        arm_selectors.append(self.fb.add_nary_or(this_arm_selectors))
      else:
        arm_selectors.append(this_arm_selectors[0])
      arm.expr.accept(self)
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
    self.node_to_ir[node] = self.fb.add_match_true(arm_selectors, arm_values,
                                                   self._use(default_arm.expr))
    self.last_expression = node

  def visit_Unop(self, node: ast.Unop):
    if node.kind == ast.UnopKind.NEG:
      self._def(node, self.fb.add_neg, self._use(node.operand))
    elif node.kind == ast.UnopKind.INV:
      self._def(node, self.fb.add_not, self._use(node.operand))
    else:
      raise NotImplementedError(node.kind)

  def _visit_width_slice(self, node: ast.Index, width_slice: ast.WidthSlice,
                         lhs_type: ConcreteType) -> None:
    width_slice.start.accept(self)
    self._def(node, self.fb.add_dynamic_bit_slice, self._use(node.lhs),
              self._use(width_slice.start),
              self._resolve_type(node).get_total_bit_count())

  def visit_Attr(self, node: ast.Attr) -> None:
    lhs_type = self.type_info[node.lhs]
    index = lhs_type.tuple_names.index(node.attr.identifier)  # pytype: disable=attribute-error
    self._def(node, self.fb.add_tuple_index, self._use(node.lhs), index)

  def visit_Index(self, node: ast.Index) -> None:
    lhs_type = self.type_info[node.lhs]
    if isinstance(lhs_type, TupleType):
      self._def(node, self.fb.add_tuple_index, self._use(node.lhs),
                self._get_const(node.index))
    elif isinstance(lhs_type, BitsType) and not isinstance(lhs_type, ArrayType):
      index_slice = node.index
      if isinstance(index_slice, ast.WidthSlice):
        return self._visit_width_slice(node, index_slice, lhs_type)
      assert isinstance(index_slice, ast.Slice), index_slice

      start, width = node.index.bindings_to_start_width[
          self._get_symbolic_bindings_tuple()]
      assert isinstance(start, int)
      assert isinstance(width, int)

      self._def(node, self.fb.add_bit_slice, self._use(node.lhs), start, width)
    else:
      self._def(node, self.fb.add_array_index, self._use(node.lhs),
                self._use(node.index))

  def visit_Number(self, node: ast.Number):
    type_ = self._resolve_type(node)
    self._def_const(node, node.get_value_as_int(), type_.get_total_bit_count())

  @ast.AstVisitor.no_auto_traverse
  def visit_Constant(self, node: ast.Constant) -> None:
    node.value.accept(self)
    self._def_alias(node.value, to=node.name)

  @ast.AstVisitor.no_auto_traverse
  def visit_Array(self, node: ast.Array) -> None:
    array_type = self._resolve_type(node)
    members = []
    for member in node.members:
      member.accept(self)
      members.append(self._use(member))
    if node.has_ellipsis:
      while len(members) < array_type.size:  # pytype: disable=attribute-error
        members.append(members[-1])
    self._def(node, self.fb.add_array, members, members[0].get_type())

  # Note: need to traverse to define constants for members.
  def visit_ConstantArray(self, node: ast.ConstantArray) -> None:
    array_type = self._resolve_type(node)
    e_type = array_type.get_element_type()  # pytype: disable=attribute-error
    values = []
    for n in node.members:
      e = self._get_const(n)
      values.append(
          ir_value.Value(_int_to_bits(e, e_type.get_total_bit_count())))
    if node.has_ellipsis:
      while len(values) < array_type.size:  # pytype: disable=attribute-error
        values.append(values[-1])
    self._def(node, self.fb.add_literal_value,
              ir_value.Value.make_array(values))

  def _cast_to_array(self, node: ast.Cast, output_type: ConcreteType) -> None:
    bits = self._use(node.expr)
    slices = []
    element_bit_count = output_type.get_element_type().get_total_bit_count()  # pytype: disable=attribute-error
    # MSb becomes lowest-indexed array element.
    for i in range(0, output_type.get_total_bit_count(), element_bit_count):
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
              array, self.fb.add_literal_bits(bits_mod.UBits(i, 32))))
    self._def(node, self.fb.add_concat, pieces)

  @ast.AstVisitor.no_auto_traverse
  def visit_Cast(self, node: ast.Cast) -> None:
    node.expr.accept(self)
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
    new_bit_count = output_type.get_total_bit_count()
    input_type = self._resolve_type(node.expr)
    if new_bit_count < input_type.get_total_bit_count():
      self._def(node, self.fb.add_bit_slice, self._use(node.expr), 0,
                new_bit_count)
    else:
      signed_input = input_type.get_signedness()
      f = self.fb.add_signext if signed_input else self.fb.add_zeroext
      self._def(node, f, self._use(node.expr), new_bit_count)

  @ast.AstVisitor.no_auto_traverse
  def visit_XlsTuple(self, node: ast.XlsTuple) -> None:
    for o in node.members:
      o.accept(self)
    operands = tuple(self._use(o) for o in node.members)
    self._def(node, self.fb.add_tuple, operands)

  def _deref_struct_or_enum(
      self, node: Union[ast.Struct, ast.TypeDef, ast.Enum, ast.ModRef]
  ) -> Union[ast.Struct, ast.Enum]:
    while isinstance(node, ast.TypeDef):
      annotation = node.type_
      if not isinstance(annotation, ast.TypeRefTypeAnnotation):
        raise NotImplementedError(
            'Unhandled typedef for resolving to struct-or-enum: %s' %
            annotation)
      node = annotation.type_ref.type_def

    if isinstance(node, (ast.Struct, ast.Enum)):
      return node

    assert isinstance(node, ast.ModRef), node
    imported_mod, _ = self.type_info.get_imports()[node.mod]
    td = imported_mod.get_typedef_by_name()[node.value]
    # Recurse to resolve the typedef within the imported module.
    td = self._deref_struct_or_enum(td)
    assert isinstance(td, (ast.Struct, ast.Enum)), td
    return td

  def _deref_struct(self, node: Union[ast.Struct, ast.ModRef]) -> ast.Struct:
    result = self._deref_struct_or_enum(node)
    assert isinstance(result, ast.Struct), result
    return result

  def _deref_enum(self, node: Union[ast.Enum, ast.ModRef]) -> ast.Enum:
    result = self._deref_struct_or_enum(node)
    assert isinstance(result, ast.Enum), result
    return result

  @ast.AstVisitor.no_auto_traverse
  def visit_SplatStructInstance(self, node: ast.SplatStructInstance) -> None:
    node.splatted.accept(self)
    orig = self._use(node.splatted)
    updates = {}
    for k, e in node.members:
      e.accept(self)
      updates[k] = self._use(e)
    struct = self._deref_struct(node.struct)

    members = []
    for i, k in enumerate(struct.member_names):
      if k in updates:
        members.append(updates[k])
      else:
        members.append(self.fb.add_tuple_index(orig, i))
    self._def(node, self.fb.add_tuple, members)

  @ast.AstVisitor.no_auto_traverse
  def visit_StructInstance(self, node: ast.StructInstance) -> None:
    operands = []
    struct = self._deref_struct(node.struct)
    for _, m in node.get_ordered_members(struct):
      m.accept(self)
      operands.append(self._use(m))
    operands = tuple(operands)
    self._def(node, self.fb.add_tuple, operands)

  def _is_constant_zero(self, node: ast.AstNode) -> bool:
    return isinstance(node, ast.Number) and node.get_value_as_int() == 0

  @ast.AstVisitor.no_auto_traverse
  def visit_For(self, node: ast.For) -> None:
    node.init.accept(self)

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
      arg.accept(self)
      if not self._is_const(arg):
        raise ConversionError(
            'For-loop is of an unsupported form for IR conversion; only a '
            "'range(const)' call is supported, did not find a const value "
            f'for {arg} ({arg!r}).', node.span)
      return self._get_const(arg)

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
    body_converter.symbolic_bindings = dict(self.symbolic_bindings)
    body_fn_name = ('__' + self.fb.name + '_counted_for_{}_body').format(
        self._next_counted_for_ordinal()).replace('.', '_')
    body_converter.fb = function_builder.FunctionBuilder(
        body_fn_name, self.package)
    flat = node.names.flatten1()
    assert len(
        flat
    ) == 2, 'Expect an induction binding and loop carry binding; got {!r}'.format(
        flat)

    # Add the induction value.
    assert isinstance(
        flat[0], ast.NameDef
    ), 'Induction variable was not a NameDef: {0} ({0!r})'.format(flat[0])
    body_converter.node_to_ir[flat[0]] = body_converter.fb.add_param(
        flat[0].identifier.encode('utf-8'), self._resolve_type_to_ir(flat[0]))

    # Add the loop carry value.
    if isinstance(flat[1], ast.NameDef):
      body_converter.node_to_ir[flat[1]] = body_converter.fb.add_param(
          flat[1].identifier.encode('utf-8'), self._resolve_type_to_ir(flat[1]))
    else:
      # For tuple loop carries we have to destructure names on entry.
      carry_type = self._resolve_type_to_ir(flat[1])
      carry = body_converter.node_to_ir[flat[1]] = body_converter.fb.add_param(
          '__loop_carry', carry_type)
      body_converter._visit_matcher(  # pylint: disable=protected-access
          flat[1], (), carry, self._resolve_type(flat[1]))

    # Free variables are suffixes on the function parameters.
    freevars = node.body.get_free_variables(node.span.start)
    freevars = freevars.drop_defs(lambda x: isinstance(x, ast.BuiltinNameDef))
    for name_def in freevars.get_name_defs():
      type_ = self.type_info[name_def]
      if isinstance(type_, FunctionType):
        continue
      logging.vlog(3, 'Converting freevar name: %s', name_def)
      body_converter.node_to_ir[name_def] = body_converter.fb.add_param(
          name_def.identifier.encode('utf-8'),
          self._resolve_type_to_ir(name_def))

    node.body.accept(body_converter)
    body_function = body_converter.fb.build()
    logging.vlog(3, 'Converted body function: %s', body_function.name)

    stride = 1
    invariant_args = tuple(
        self._use(name_def)
        for name_def in freevars.get_name_defs()
        if not isinstance(self.type_info[name_def], FunctionType))
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
    return tuple((k, v)
                 for k, v in self.symbolic_bindings.items()
                 if k not in module_level_constants)

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
    elif isinstance(node.callee, ast.ModRef):
      m = self.type_info.get_imports()[node.callee.mod][0]
      callee_name = node.callee.value
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
      return mangle_dslx_name(function.name.identifier,
                              function.get_free_parametric_keys(), m, None)
    resolved_symbolic_bindings = self._get_invocation_bindings(node)
    logging.vlog(2, 'Node %s @ %s symbolic bindings %r', node, node.span,
                 resolved_symbolic_bindings)
    assert resolved_symbolic_bindings, node
    return mangle_dslx_name(function.name.identifier,
                            function.get_free_parametric_keys(), m,
                            resolved_symbolic_bindings)

  def _def_map_with_builtin(self, parent_node: ast.Invocation,
                            node: ast.NameRef, arg: ast.AstNode,
                            symbolic_bindings: SymbolicBindings) -> BValue:
    """Makes the specified builtin available to the package."""
    mangled_name = mangle_dslx_name(node.name_def.identifier, set(),
                                    self.module, symbolic_bindings)

    arg = self._use(arg)
    if mangled_name not in self.package.get_function_names():
      fb = function_builder.FunctionBuilder(mangled_name, self.package)
      param = fb.add_param('arg', arg.get_type().get_element_type())
      builtin_name = node.name_def.identifier
      assert builtin_name in dslx_builtins.UNARY_BUILTIN_NAMES, dslx_builtins.UNARY_BUILTIN_NAMES
      fbuilds = {'clz': fb.add_clz, 'ctz': fb.add_ctz}
      assert set(fbuilds.keys()) == dslx_builtins.UNARY_BUILTIN_NAMES, set(
          fbuilds.keys())
      fbuilds[builtin_name](param)
      fb.build()
    return self._def(parent_node, self.fb.add_map, arg,
                     self.package.get_function(mangled_name))

  def _visit_map(self, node: ast.Invocation) -> BValue:
    for arg in node.args[:-1]:
      arg.accept(self)
    arg = self._use(node.args[0])
    fn_node = node.args[1]

    if isinstance(fn_node, ast.NameRef):
      map_fn_name = fn_node.name_def.identifier
      if map_fn_name in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
        return self._def_map_with_builtin(node, fn_node, node.args[0],
                                          self._get_invocation_bindings(node))
      else:
        lookup_module = self.module
        fn = lookup_module.get_function(map_fn_name)
    elif isinstance(fn_node, ast.ModRef):
      map_fn_name = fn_node.value
      imports = self.type_info.get_imports()
      lookup_module, _ = imports[fn_node.mod]
      fn = lookup_module.get_function(map_fn_name)
    else:
      raise NotImplementedError(
          'Unhandled function mapping: {!r}'.format(fn_node))

    node_sym_bindings = self._get_invocation_bindings(node)
    mangled_name = mangle_dslx_name(fn.name, fn.get_free_parametric_keys(),
                                    lookup_module, node_sym_bindings)

    return self._def(node, self.fb.add_map, arg,
                     self.package.get_function(mangled_name))

  def _visit_update(self, node: ast.Invocation, args: Tuple[BValue,
                                                            ...]) -> BValue:
    array, target_index, update_value = args
    return self._def(node, self.fb.add_array_update, array, target_index,
                     update_value)

  def _visit_bitwise_reduction(self, node: ast.Invocation, args: Tuple[BValue,
                                                                       ...],
                               called_name: Text) -> BValue:
    add_func = None
    if called_name == 'and_reduce':
      add_func = self.fb.add_and_reduce
    elif called_name == 'or_reduce':
      add_func = self.fb.add_or_reduce
    elif called_name == 'xor_reduce':
      add_func = self.fb.add_xor_reduce
    else:
      raise NotImplementedError(
          'Unknown bitwise reduction: {}'.format(called_name))
    return self._def(node, add_func, args[0])

  def _visit_bit_slice(self, node: ast.Invocation, args: Tuple[BValue,
                                                               ...]) -> BValue:
    lhs, _, width = args
    width_type = width.get_type()
    return self._def(node, self.fb.add_bit_slice, lhs,
                     self._get_const(node.args[1]), width_type.get_bit_count())

  def _visit_rev(self, node: ast.Invocation, args: Tuple[BValue,
                                                         ...]) -> BValue:
    arg, = args
    return self._def(node, self.fb.add_reverse, arg)

  def _visit_signex(self, node: ast.Invocation, args: Tuple[BValue,
                                                            ...]) -> BValue:
    lhs, rhs = args
    rhs_type = rhs.get_type()
    return self._def(node, self.fb.add_signext, lhs, rhs_type.get_bit_count())

  def _visit_clz(self, node: ast.Invocation, args: Tuple[BValue,
                                                         ...]) -> BValue:
    lhs, = args
    return self._def(node, self.fb.add_clz, lhs)

  def _visit_ctz(self, node: ast.Invocation, args: Tuple[BValue,
                                                         ...]) -> BValue:
    lhs, = args
    return self._def(node, self.fb.add_ctz, lhs)

  def _visit_one_hot(self, node: ast.Invocation, args: Tuple[BValue,
                                                             ...]) -> BValue:
    lhs, _ = args
    lsb_prio = self._get_const(node.args[1])
    return self._def(
        node, self.fb.add_one_hot, lhs,
        lsb_or_msb.LsbOrMsb.LSB if lsb_prio else lsb_or_msb.LsbOrMsb.MSB)

  def _visit_one_hot_sel(self, node: ast.Invocation,
                         args: Tuple[BValue, ...]) -> BValue:
    lhs, array = args
    array_type = array.get_type()
    rhs_elements = []
    for i in range(array_type.get_size()):
      rhs_elements.append(
          self.fb.add_array_index(
              array, self.fb.add_literal_bits(bits_mod.UBits(i, 32))))
    return self._def(node, self.fb.add_one_hot_sel, lhs, rhs_elements)

  def _visit_scmp(self, node: ast.Invocation, args: Tuple[BValue, ...],
                  which: Text) -> BValue:
    lhs, rhs = args
    return self._def(node, getattr(self.fb, 'add_{}'.format(which)), lhs, rhs)

  def _visit_trace(self, node: ast.Invocation, args: Tuple[BValue,
                                                           ...]) -> BValue:
    return self._def(node, self.fb.add_identity, args[0])

  @ast.AstVisitor.no_auto_traverse
  def visit_Invocation(self, node: ast.Invocation):
    called_name = self._get_callee_identifier(node)

    def accept_args() -> Tuple[BValue, ...]:
      for arg in node.args:
        arg.accept(self)
      return tuple(self._use(arg) for arg in node.args)

    if called_name == 'fail!':
      args = accept_args()
      assert len(node.args) == 1, ('Expect fail! builtin to only accept a '
                                   'single argument; got: {!r}'.format(args))
      self._def(node, self.fb.add_identity, args[0])
    elif called_name == 'update':
      self._visit_update(node, accept_args())
    elif called_name == 'signex':
      self._visit_signex(node, accept_args())
    elif called_name == 'clz':
      self._visit_clz(node, accept_args())
    elif called_name == 'ctz':
      self._visit_ctz(node, accept_args())
    elif called_name == 'map':
      # Map needs special arg handling, so we handle that inside.
      self._visit_map(node)
    elif called_name == 'one_hot':
      self._visit_one_hot(node, accept_args())
    elif called_name == 'one_hot_sel':
      self._visit_one_hot_sel(node, accept_args())
    elif called_name == 'bit_slice':
      self._visit_bit_slice(node, accept_args())
    elif called_name == 'rev':
      self._visit_rev(node, accept_args())
    elif called_name in ('sgt', 'sge', 'slt', 'sle'):
      self._visit_scmp(node, accept_args(), called_name)
    elif called_name in ('and_reduce', 'or_reduce', 'xor_reduce'):
      self._visit_bitwise_reduction(node, accept_args(), called_name)
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

  def visit_EnumRef(self, node: ast.EnumRef) -> None:
    enum = self._deref_enum(node.enum)
    value = enum.get_value(node.value)
    value.accept(self)
    self._def_alias(from_=value, to=node)

  @ast.AstVisitor.no_auto_traverse
  def visit_Let(self, node: ast.Let):
    node.rhs.accept(self)
    if node.name_def_tree.is_leaf():
      self._def_alias(node.rhs, to=node.name_def_tree.get_leaf())
      node.body.accept(self)
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

      node.name_def_tree.do_preorder(walk)
      node.body.accept(self)
      self._def_alias(node.body, to=node)

    if self.last_expression is None:
      self.last_expression = node.body

  @ast.AstVisitor.no_auto_traverse
  def visit_Param(self, node: ast.Param):
    self._def(node.name, self.fb.add_param, node.name.identifier,
              self._resolve_type_to_ir(node.type_))

  def _visit_Function(
      self, node: ast.Function,
      symbolic_bindings: Optional[SymbolicBindings]) -> ir_function.Function:
    self.symbolic_bindings = {} if symbolic_bindings is None else dict(
        symbolic_bindings)
    self._extract_module_level_constants(self.module)
    # We use a function builder for the duration of converting this
    # ast.Function. When it's done being built, we drop the reference to it (by
    # setting self.fb to None).
    self.fb = function_builder.FunctionBuilder(
        mangle_dslx_name(node.name.identifier, node.get_free_parametric_keys(),
                         self.module, symbolic_bindings), self.package)
    try:
      for param in node.params:
        param.accept(self)

      for parametric_binding in node.parametric_bindings:
        logging.vlog(4, 'Resolving parametric binding %s', parametric_binding)

        sb_value = self.symbolic_bindings[parametric_binding.name.identifier]
        value = self._resolve_dim(sb_value)
        assert isinstance(value, int), \
            'Expect integral parametric binding; got {!r}'.format(value)
        self._def_const(
            parametric_binding, value,
            self._resolve_type(parametric_binding.type_).get_total_bit_count())
        self._def_alias(parametric_binding, to=parametric_binding.name)

      for dep in self._constant_deps:
        dep.accept(self)
      del self._constant_deps[:]

      node.body.accept(self)

      last_expression = self.last_expression or node.body
      if isinstance(last_expression, ast.NameRef):
        self._def(last_expression, self.fb.add_identity,
                  self._use(last_expression))
      f = self.fb.build()
      logging.vlog(3, 'Built function: %s', f.name)
      verifier_mod.verify_function(f)
      return f
    finally:
      self.fb = None

  @ast.AstVisitor.no_auto_traverse
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
  converter = _IrConverterFb(
      package, module, type_info, emit_positions=emit_positions)

  freevars = function.body.get_free_variables(
      function.span.start).get_name_def_tups()
  logging.vlog(3, 'Free variables for function: %s', freevars)
  for identifier, name_def in freevars:
    if identifier in function_by_name or isinstance(name_def,
                                                    ast.BuiltinNameDef):
      pass
    elif identifier in constant_by_name:
      converter.add_constant_dep(constant_by_name[identifier])
    else:
      raise NotImplementedError(identifier)

  symbolic_binding_keys = set(k for k, _ in symbolic_bindings or ())
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
                                             type_info.get_imports(),
                                             traverse_tests)
  logging.vlog(3, 'Convert order: %s', pprint.pformat(order))
  for record in order:
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
  package = ir_package.Package(module.name)
  _convert_one_function(
      package,
      module,
      module.get_function(entry_function_name),
      type_info,
      emit_positions=emit_positions)
  return package.dump_ir()
