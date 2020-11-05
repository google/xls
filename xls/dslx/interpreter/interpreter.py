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

"""Interpreter for the AST data structure.

Used as a reference for evaluating modules to a value from its syntactic form.

This is a complement to other execution modes that can help sanity check more
optimized forms of execution.
"""

import contextlib
import functools
import sys
from typing import Text, Optional, Dict, Tuple, Callable, Sequence, Union

from absl import logging
import termcolor

from xls.dslx import ast_helpers
from xls.dslx import import_fn
from xls.dslx import ir_name_mangler
from xls.dslx.concrete_type_helpers import map_size
from xls.dslx.interpreter import jit_comparison
from xls.dslx.interpreter.concrete_type_helpers import concrete_type_accepts_value
from xls.dslx.interpreter.concrete_type_helpers import concrete_type_convert_value
from xls.dslx.interpreter.concrete_type_helpers import concrete_type_from_value
from xls.dslx.interpreter.errors import EvaluateError
from xls.dslx.parametric_instantiator import SymbolicBindings
from xls.dslx.python import builtins
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_evaluate
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import ConcreteTypeDim
from xls.dslx.python.cpp_concrete_type import EnumType
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_parametric_expression import ParametricAdd
from xls.dslx.python.cpp_parametric_expression import ParametricSymbol
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span
from xls.dslx.python.interp_bindings import Bindings
from xls.dslx.python.interp_bindings import FnCtx
from xls.dslx.python.interp_value import Builtin
from xls.dslx.python.interp_value import Tag
from xls.dslx.python.interp_value import Value
from xls.ir.python import bits as ir_bits
from xls.ir.python import package as ir_package_mod
from xls.jit.python import ir_jit


class _WipSentinel(object):
  """Marker to show that something is the in process of being evaluated."""


ImportSubject = Tuple[Text, ...]
ImportInfo = Tuple[ast.Module, type_info_mod.TypeInfo]


class Interpreter(object):
  """Object that interprets an AST of expressions to evaluate it to a value."""

  def __init__(self,
               module: ast.Module,
               type_info: type_info_mod.TypeInfo,
               f_import: Optional[Callable[[ImportSubject], ImportInfo]],
               trace_all: bool = False,
               ir_package: Optional[ir_package_mod.Package] = None):
    self._module = module
    self._type_info = type_info
    self._wip = {}  # Work-in-progress constant evaluation annotations.
    self._started_top_level_index = None
    self._f_import = f_import
    self._trace_all = trace_all
    self._ir_package = ir_package

  def _evaluate_EnumRef(  # pylint: disable=invalid-name
      self,
      expr: ast.EnumRef,
      bindings: Bindings,  # pylint: disable=unused-argument
      _type_context: Optional[ConcreteType]) -> Value:
    """Evaluates a reference to an enum value."""
    enum = self._evaluate_to_enum(expr.enum, bindings)
    value_node = enum.get_value(expr.value)
    fresh_bindings = self._make_top_level_bindings(expr.module)
    raw_value = self._evaluate(
        value_node, fresh_bindings,
        self._evaluate_TypeAnnotation(enum.type_, fresh_bindings))
    return Value.make_enum(raw_value.get_bits(), enum)

  def _get_enum_values(self, type_: ConcreteType,
                       bindings: Bindings) -> Optional[Tuple[Value, ...]]:
    """Retrieves the flat/evaluated members of enum if type_ is an EnumType."""
    if not isinstance(type_, EnumType):
      return None

    enum = type_.get_nominal_type(self._module)
    result = []
    for member in enum.values:
      _, value = member.get_name_value(enum)
      result.append(self._evaluate(value, bindings))
    return tuple(result)

  def _evaluate_Cast(  # pylint: disable=invalid-name
      self, expr: ast.Cast, bindings: Bindings,
      _: Optional[ConcreteType]) -> Value:
    """Evaluates a 'Cast' AST node to a value."""
    type_ = self._evaluate_TypeAnnotation(expr.type_, bindings)
    logging.vlog(3, 'Cast to type: %s @ %s', type_, expr.span)
    value = self._evaluate(expr.expr, bindings, type_)
    type_accepts_value = concrete_type_accepts_value(self._module, type_, value)
    logging.vlog(3, 'Type %s accepts value %s? %s', type_, value,
                 type_accepts_value)
    return concrete_type_convert_value(
        self._module, type_, value, expr.span,
        self._get_enum_values(type_, bindings),
        value.get_type().get_signedness() if value.get_type() else None)

  def _evaluate_index_widthslice(self, expr: ast.Index, bindings: Bindings,
                                 bits: ir_bits.Bits) -> Value:
    """Evaluates a WidthSlice expression on a bits value."""
    index_slice = expr.index
    assert isinstance(index_slice, ast.WidthSlice), index_slice
    start = self._evaluate(index_slice.start, bindings,
                           BitsType(signed=False, size=bits.bit_count()))
    width_type = self._evaluate_TypeAnnotation(index_slice.width, bindings)
    start_index = start.get_bit_value_uint64()
    width_value = width_type.get_total_bit_count().value
    if start_index >= bits.bit_count():
      # Return early to potentially avoid an unreasonably long zero extend (e.g.
      # if the start index was a large negative number).
      return Value.make_ubits(value=0, bit_count=width_value)
    # Slicing off the end zero-fills, so we zext.
    if start_index + width_value > bits.bit_count():
      bits = bits.zero_ext(start_index + width_value)
    result = bits.slice(start_index, width_value)
    tag = Tag.SBITS if width_type.get_signedness() else Tag.UBITS
    return Value.make_bits(tag, result)

  def _evaluate_Index(  # pylint: disable=invalid-name
      self, expr: ast.Index, bindings: Bindings,
      _: Optional[ConcreteType]) -> Value:
    """Evaluates an index expression; e.g. `lhs[index]`."""
    lhs = self._evaluate(expr.lhs, bindings)
    if lhs.is_bits() and isinstance(expr.index, ast.Slice):
      return cpp_evaluate.evaluate_index_bitslice(self._type_info, expr,
                                                  bindings, lhs.get_bits())
    if lhs.is_bits() and isinstance(expr.index, ast.WidthSlice):
      return self._evaluate_index_widthslice(expr, bindings, lhs.get_bits())
    assert lhs.is_tuple() or lhs.is_array(), lhs
    index = self._evaluate(expr.index, bindings)
    if index.get_bit_value_uint64() >= len(lhs):
      raise EvaluateError(
          expr.span, 'Indexing out of bounds; {} > {}'.format(index, len(lhs)))
    result = lhs.index(index)
    return result

  def _evaluate_Let(  # pylint: disable=invalid-name
      self, expr: ast.Let, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    """Evaluates a let expression to its body value."""
    if expr.type_ is None:
      type_ = None
    else:
      type_ = self._evaluate_TypeAnnotation(expr.type_, bindings)

    to_bind = self._evaluate(expr.rhs, bindings, type_)
    assert to_bind is not None
    if type_ and not concrete_type_accepts_value(self._module, type_, to_bind):
      raise EvaluateError(
          expr.type_.span, 'Type error found at interpreter runtime! '
          'Let-expression right hand side did not conform to annotated type'
          '\n\twant: {}\n\tgot:  {}\n\tvalue: {}'.format(
              type_, concrete_type_from_value(to_bind), to_bind))
    new_bindings = bindings.clone_with(expr.name_def_tree, to_bind)
    result = self._evaluate(expr.body, new_bindings, type_context)
    return result

  def _evaluate_matcher(self, pattern: ast.NameDefTree, matched: Value,
                        bindings: Bindings) -> bool:
    """Returns whether this matcher pattern is accepted.

    Note that refutable patterns don't always match; e.g. in

      match (u32:3, u32:4) {
        (u32:2, y) => y;
        (x, _) => x;
      }

    The first pattern will not match, and so this method would return false for
    that match arm.

    Args:
      pattern: Decribes the pattern to attempt to match.
      matched: The value being matched against.
      bindings: The bindings to populate if the pattern has bindings associated
        with it.
    """
    if pattern.is_leaf():
      leaf = pattern.get_leaf()
      if isinstance(leaf, ast.NameDef):
        bindings.add_value(leaf.identifier, matched)
        return True
      elif isinstance(leaf, ast.WildcardPattern):
        return True
      elif isinstance(leaf, (ast.Number, ast.EnumRef)):
        return self._evaluate(leaf, bindings) == matched
      else:
        assert isinstance(leaf, ast.NameRef), repr(leaf)
        return self._evaluate(leaf, bindings) == matched
    else:
      assert isinstance(pattern.tree, tuple)
      for subtree, member in zip(pattern.tree, matched.get_elements()):
        if not self._evaluate_matcher(subtree, member, bindings):
          return False
      return True

  def _evaluate_Match(  # pylint: disable=invalid-name
      self, expr: ast.Match, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    """Resolves expr into a value via expression evaluation."""
    matched = self._evaluate(expr.matched, bindings)
    for arm in expr.arms:
      for pattern in arm.patterns:
        if isinstance(pattern, ast.WildcardPattern):
          return self._evaluate(arm.expr, bindings, type_context)

        arm_bindings = Bindings(bindings)
        if self._evaluate_matcher(pattern, matched, arm_bindings):
          return self._evaluate(arm.expr, arm_bindings, type_context)
    builtins.throw_fail_error(
        expr.span, 'The program being interpreted failed with an '
        'incomplete match! value: {}'.format(matched))

  def _get_imported_module_via_bindings(
      self, import_: ast.Import,
      bindings: Bindings) -> Tuple[ast.Module, Bindings]:
    """Uses bindings to retrieve the corresponding module of a ModRef."""
    imported_module = bindings.resolve_mod(import_.identifier)
    logging.vlog(3, 'Resolved import %s to %r', import_, imported_module)
    return imported_module, self._make_top_level_bindings(imported_module)

  def _deref_typeref(
      self, type_ref: ast.TypeRef, bindings: Bindings
  ) -> Union[ast.TypeAnnotation, ast.EnumDef, ast.StructDef]:
    """Resolves the typeref to a type using its identifier via bindings."""
    logging.vlog(5, 'Dereferencing TypeRef; type_def: %r', type_ref.type_def)
    if isinstance(type_ref.type_def, ast.ModRef):
      return ast_helpers.evaluate_to_struct_or_enum_or_annotation(
          type_ref.type_def, self._get_imported_module_via_bindings, bindings)

    result = bindings.resolve_type_definition(type_ref.text, type_ref.module)
    assert isinstance(result,
                      (ast.TypeAnnotation, ast.EnumDef, ast.StructDef)), result
    return result

  def _bindings_with_struct_parametrics(self, struct: ast.StructDef,
                                        parametrics: Tuple[
                                            ast.ParametricBinding, ...],
                                        bindings: Bindings) -> Bindings:
    """Returns new (derived) Bindings populated with `parametrics`.

    For example, if we have a struct defined as `struct [N: u32, M: u32] Foo`,
    and provided parametrics with values [A, 16], we'll create a new set of
    Bindings out of `bindings` and add (N, A) and (M, 16) to that.

    Args:
      struct: The struct that may have parametric bindings.
      parametrics: The parametric bindings that correspond to those on the
        struct.
      bindings: Bindings to use as the parent.
    """
    nested_bindings = Bindings(parent=bindings)
    for p, d in zip(struct.parametric_bindings, parametrics):
      if isinstance(d, ast.Number):
        nested_bindings.add_value(p.name.identifier,
                                  Value.make_ubits(p.type_.bits, int(d.value)))
      else:
        assert isinstance(p, ast.ParametricBinding), p
        nested_bindings.add_value(
            p.name.identifier,
            nested_bindings.resolve_value_from_identifier(d.identifier))

    return nested_bindings

  def _concretize(self, type_: Union[ast.TypeAnnotation, ast.EnumDef,
                                     ast.StructDef],
                  bindings: Bindings) -> ConcreteType:
    """Resolves type_ into a concrete type via expression evaluation."""
    assert isinstance(type_,
                      (ast.EnumDef, ast.TypeAnnotation, ast.StructDef)), type_
    if isinstance(type_, ast.EnumDef):
      return self._concretize(type_.type_, bindings)
    elif isinstance(type_, ast.StructDef):
      members = tuple((k.identifier, self._concretize(t, bindings))
                      for k, t in type_.members)
      return TupleType(members, type_)
    elif isinstance(type_, ast.TypeRefTypeAnnotation):
      logging.vlog(5, 'Concretizing typeref: %s', type_)
      deref = self._deref_typeref(type_.type_ref, bindings)
      if type_.parametrics:
        bindings = self._bindings_with_struct_parametrics(
            deref, type_.parametrics, bindings)
      type_def = type_.type_ref.type_def
      logging.vlog(5, 'Type definition referred to: %s', type_def)
      if isinstance(type_def, ast.EnumDef):
        enum = type_def
        bit_count = self._concretize(enum.type_, bindings).get_total_bit_count()
        return EnumType(enum, bit_count)
      return self._concretize(
          self._deref_typeref(type_.type_ref, bindings), bindings)
    elif isinstance(type_, ast.TupleTypeAnnotation):
      return TupleType(
          tuple(self._concretize(m, bindings) for m in type_.members))
    elif isinstance(type_, ast.ArrayTypeAnnotation):
      dim = self._resolve_dim(type_.dim, bindings)
      if (isinstance(type_.element_type, ast.BuiltinTypeAnnotation) and
          type_.element_type.bits == 0):
        return BitsType(type_.element_type.signedness, dim)
      elem_type = self._concretize(type_.element_type, bindings)
      return ArrayType(elem_type, dim)
    elif isinstance(type_, ast.BuiltinTypeAnnotation):
      signed, bits = type_.signedness_and_bits
      return BitsType(signed, bits)
    else:
      raise NotImplementedError('Unknown type for concretization: %r' % type_)

  def _resolve_dim(self, dim: Union[ConcreteTypeDim, ast.Number, ast.ConstRef,
                                    ast.NameRef, int],
                   bindings: Bindings) -> int:
    """Resolves (parametric) dim from deduction vs current bindings."""
    if isinstance(dim, int):
      return dim
    if isinstance(dim, ast.Number):
      return ast_helpers.get_value_as_int(dim)
    if isinstance(dim, (ast.ConstRef, ast.NameRef)):
      identifier = dim.identifier
      return bindings.resolve_value_from_identifier(
          identifier).get_bit_value_int64()
    assert isinstance(dim, ConcreteTypeDim), repr(dim)
    value = dim.value
    if isinstance(value, int):
      return value
    if isinstance(value, ParametricSymbol):
      identifier = value.identifier
      return bindings.resolve_value_from_identifier(
          identifier).get_bit_value_int64()
    if isinstance(value, ParametricAdd):
      return (self._resolve_dim(value.lhs, bindings) +
              self._resolve_dim(value.rhs, bindings))
    raise NotImplementedError(f'Unhandled dim for resolution: {dim!r}')

  def _evaluate_TypeAnnotation(  # pylint: disable=invalid-name
      self, type_: ast.TypeAnnotation, bindings: Bindings) -> ConcreteType:
    """Evaluates TypeAnnotation to a concrete type with dimensions resolved."""
    result = self._concretize(type_, bindings)
    logging.vlog(3, 'Type annotation %s concretized to %s', type_, result)

    # Check deduced type is consistent with what we've interpreted.
    #
    # TODO(leary): 2019-12-03 We don't have a way to check enum compatibility
    # with the corresponding bits-type value -- we should be using enum-based
    # ConcreteTypes in the interpreter instead of their bits equivalents.
    deduced = self._type_info[type_]
    deduced = map_size(deduced, self._module,
                       functools.partial(self._resolve_dim, bindings=bindings))
    if not deduced.has_enum():
      assert deduced.compatible_with(result), \
          (f'Deduced type {deduced} incompatible with '
           f'interp-determined type {result} ({deduced!r} vs {result!r})')

    logging.vlog(3, 'Type annotation %s evaluated to %s', type_, result)
    return result

  def _evaluate_Number(  # pylint: disable=invalid-name
      self, expr: ast.Number, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    """Evaluates a Number AST node to a value.

    Args:
      expr: Number AST node.
      bindings: Name bindings for this evaluation.
      type_context: Type context for evaluating this number; since numbers
        literals are agnostic of their bit width this allows us to create the
        proper-width value.

    Returns:
      The resulting interpreter value.

    Raises:
      EvaluateError: If the type context is missing or inappropriate (e.g. a
        tuple cannot be the type for a number).
    """
    logging.vlog(4, 'number: %s @ %s', expr, expr.span)
    if not type_context and expr.kind == ast.NumberKind.CHARACTER:
      type_context = ConcreteType.U8
    if not type_context and expr.kind == ast.NumberKind.BOOL:
      type_context = ConcreteType.U1  # Boolean.
    if not type_context and expr.type_ is None:
      raise EvaluateError(
          expr.span,
          'Internal error: no type context for expression, should be caught '
          'by type inference!')
    type_context = type_context or self._evaluate_TypeAnnotation(
        expr.type_, bindings)
    if type_context is None:
      raise EvaluateError(
          expr.span, 'Missing type context for number @ {}'.format(expr.span))
    elif isinstance(type_context, TupleType):
      raise EvaluateError(
          expr.span, 'Type context for number is a tuple type {} @ {}'.format(
              type_context, expr.span))
    assert isinstance(type_context, BitsType), type_context
    bit_count = type_context.get_total_bit_count().value
    signed = type_context.signed
    tag = Tag.SBITS if signed else Tag.UBITS
    return Value.make_bits(tag, ast_helpers.get_value_as_bits(expr, bit_count))

  def _evaluate_to_struct_or_enum_or_annotation(
      self, node: Union[ast.TypeDef, ast.ModRef,
                        ast.StructDef], bindings: Bindings
  ) -> Union[ast.StructDef, ast.EnumDef, ast.TypeAnnotation]:
    """Returns the node dereferenced into a Struct or Enum or TypeAnnotation.

    Will produce TypeAnnotation in the case we bottom out in a tuple, for
    example.

    Args:
      node: Node to resolve to a struct/enum/annotation.
      bindings: Current bindings for evaluating the node.
    """
    while isinstance(node, ast.TypeDef):
      annotation = node.type_
      if not isinstance(annotation, ast.TypeRefTypeAnnotation):
        return annotation
      node = annotation.type_ref.type_def

    if isinstance(node, (ast.StructDef, ast.EnumDef)):
      return node

    assert isinstance(node, ast.ModRef)
    imported_module = bindings.resolve_mod(node.mod.identifier)
    td = imported_module.get_typedef(node.value)
    # Recurse to dereference it if it's a typedef in the imported module.
    td = self._evaluate_to_struct_or_enum_or_annotation(
        td, self._make_top_level_bindings(imported_module))
    assert isinstance(td, (ast.StructDef, ast.EnumDef, ast.TypeAnnotation)), td
    return td

  def _evaluate_to_enum(self, node: Union[ast.TypeDef, ast.EnumDef],
                        bindings: Bindings) -> ast.EnumDef:
    type_definition = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
        node, self._get_imported_module_via_bindings, bindings)
    assert isinstance(type_definition, ast.EnumDef), type_definition
    return type_definition

  def _evaluate_to_struct(self, node: Union[ast.ModRef, ast.StructDef],
                          bindings: Bindings) -> ast.StructDef:
    """Evaluates potential module-reference-to-struct to a struct."""
    type_definition = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
        node, self._get_imported_module_via_bindings, bindings)
    assert isinstance(type_definition, ast.StructDef), type_definition
    return type_definition

  def _evaluate_StructInstance(  # pylint: disable=invalid-name
      self,
      expr: ast.StructInstance,
      bindings: Bindings,
      type_context: Optional[ConcreteType]  # pylint: disable=unused-argument
  ) -> Value:
    """Evaluates a struct instance AST node to a value."""
    struct = self._evaluate_to_struct(expr.struct, bindings)
    result = Value.make_tuple(
        tuple(
            self._evaluate(e, bindings)
            for _, e in expr.get_ordered_members(struct)))
    return result

  def _evaluate_SplatStructInstance(  # pylint: disable=invalid-name
      self,
      expr: ast.SplatStructInstance,
      bindings: Bindings,
      type_context: Optional[ConcreteType]  # pylint: disable=unused-argument
  ) -> Value:
    """Evaluates a 'splat' struct instance AST node to a value."""
    named_tuple = self._evaluate(expr.splatted, bindings)
    struct = self._evaluate_to_struct(expr.struct, bindings)
    for k, v in expr.members:
      new_value = self._evaluate(v, bindings)
      i = struct.member_names.index(k)
      named_tuple = named_tuple.update(i, new_value)

    assert isinstance(named_tuple, Value), named_tuple
    return named_tuple

  def _evaluate_Attr(  # pylint: disable=invalid-name
      self,
      expr: ast.Attr,
      bindings: Bindings,
      type_context: Optional[ConcreteType]  # pylint: disable=unused-argument
  ) -> Value:
    """Evaluates an attribute-accessing AST node to a value."""
    lhs_value = self._evaluate(expr.lhs, bindings)
    index = next(
        i for i, name in enumerate(self._type_info[expr.lhs].tuple_names)  # pytype: disable=attribute-error
        if name == expr.attr.identifier)
    return lhs_value.get_elements()[index]

  def _evaluate_XlsTuple(  # pylint: disable=invalid-name
      self, expr: ast.XlsTuple, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    """Evaluates an XlsTuple expression AST node to a value."""

    def get_type_context(i: int) -> Optional[ConcreteType]:
      """Retrieves the type context for a tuple member.

      Args:
        i: Which tuple member.

      Returns:
        The type context for the ith tuple member, if a type context is
        available at all.
      """
      if type_context is None:
        return None
      assert isinstance(type_context, TupleType), type_context
      return type_context.get_tuple_member(i)

    result = Value.make_tuple(
        tuple(
            self._evaluate(e, bindings, get_type_context(i))
            for i, e in enumerate(expr.members)))
    logging.vlog(3, 'tuple: %s', result)
    return result

  def _evaluate_Ternary(  # pylint: disable=invalid-name
      self, expr: ast.Ternary, bindings: Bindings,
      _: Optional[ConcreteType]) -> Value:
    test_value = self._evaluate(expr.test, bindings)
    if test_value.is_true():
      return self._evaluate(expr.consequent, bindings)
    else:
      return self._evaluate(expr.alternate, bindings)

  def _evaluate_Unop(  # pylint: disable=invalid-name
      self, expr: ast.Unop, bindings: Bindings,
      _: Optional[ConcreteType]) -> Value:
    """Evaluates a 'Unop' AST node to a Value."""
    operand_value = self._evaluate(expr.operand, bindings)
    if expr.kind == ast.UnopKind.INV:
      return operand_value.bitwise_negate()
    if expr.kind == ast.UnopKind.NEG:
      return operand_value.arithmetic_negate()
    raise NotImplementedError('Unimplemented unop.', expr.kind)

  def _evaluate_Binop(  # pylint: disable=invalid-name
      self, expr: ast.Binop, bindings: Bindings,
      _: Optional[ConcreteType]) -> Value:
    """Evaluates a 'Binop' AST node to a value."""
    lhs_value = self._evaluate(expr.lhs, bindings)
    rhs_value = self._evaluate(expr.rhs, bindings)
    if expr.kind == ast.BinopKind.ADD:
      result = lhs_value.add(rhs_value)
    elif expr.kind == ast.BinopKind.SUB:
      result = lhs_value.sub(rhs_value)
    elif expr.kind == ast.BinopKind.CONCAT:
      result = lhs_value.concat(rhs_value)
    elif expr.kind == ast.BinopKind.MUL:
      result = lhs_value.mul(rhs_value)
    elif expr.kind == ast.BinopKind.DIV:
      result = lhs_value.floordiv(rhs_value)
    elif expr.kind in (ast.BinopKind.OR, ast.BinopKind.LOGICAL_OR):
      result = lhs_value.bitwise_or(rhs_value)
    elif expr.kind in (ast.BinopKind.AND, ast.BinopKind.LOGICAL_AND):
      result = lhs_value.bitwise_and(rhs_value)
    elif expr.kind == ast.BinopKind.XOR:
      result = lhs_value.bitwise_xor(rhs_value)
    elif expr.kind == ast.BinopKind.SHLL:  # <<
      result = lhs_value.shll(rhs_value)
    elif expr.kind == ast.BinopKind.SHRL:  # >>
      result = lhs_value.shrl(rhs_value)
    elif expr.kind == ast.BinopKind.SHRA:  # >>>
      result = lhs_value.shra(rhs_value)
    elif expr.kind == ast.BinopKind.EQ:  # ==
      result = lhs_value.eq(rhs_value)
    elif expr.kind == ast.BinopKind.NE:  # !=
      result = lhs_value.ne(rhs_value)
    elif expr.kind == ast.BinopKind.GT:  # >
      result = lhs_value.gt(rhs_value)
    elif expr.kind == ast.BinopKind.LT:  # <
      result = lhs_value.lt(rhs_value)
    elif expr.kind == ast.BinopKind.LE:  # <=
      result = lhs_value.le(rhs_value)
    elif expr.kind == ast.BinopKind.GE:  # >=
      result = lhs_value.ge(rhs_value)
    else:
      raise NotImplementedError('Unimplemented binop', expr.kind)
    return result

  def _evaluate_For(  # pylint: disable=invalid-name
      self, expr: ast.For, bindings: Bindings,
      _: Optional[ConcreteType]) -> Value:
    """Evaluates a 'For' AST node to a value."""
    iterable = self._evaluate(expr.iterable, bindings)
    concrete_iteration_type = self._concretize(expr.type_, bindings)
    carry = self._evaluate(expr.init, bindings)
    for i, x in enumerate(iterable.get_elements()):
      iteration = Value.make_tuple((x, carry))
      if not concrete_type_accepts_value(self._module, concrete_iteration_type,
                                         iteration):
        raise EvaluateError(
            expr.type_.span,
            'type error found at interpreter runtime! iteration value does not conform to type annotation '
            'at top of iteration {}:\n  got value: {}\n  type: {};\n  want: {}'
            .format(i, iteration, concrete_type_from_value(iteration),
                    concrete_iteration_type))
      new_bindings = bindings.clone_with(expr.names, iteration)
      carry = self._evaluate(expr.body, new_bindings)
    return carry

  # This function signature conforms to an abstract interface.
  # pylint: disable=unused-argument
  def _evaluate_Carry(  # pylint: disable=invalid-name
      self, expr: ast.Carry, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    assert isinstance(expr, ast.Carry), expr
    return bindings.resolve_value_from_identifier('carry')

  def _evaluate_While(  # pylint: disable=invalid-name
      self, expr: ast.While, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    carry = self._evaluate(expr.init, bindings)
    new_bindings = Bindings(bindings)
    new_bindings.add_value('carry', carry)
    while self._evaluate(expr.test, new_bindings).is_true():
      carry = self._evaluate(expr.body, new_bindings)
      new_bindings.add_value('carry', carry)
    return carry

  def _evaluate_Array(  # pylint: disable=invalid-name
      self, expr: ast.Array, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    """Evaluates an 'Array' AST node to a value."""
    element_type = None
    if type_context is None and expr.type_:
      type_context = self._evaluate_TypeAnnotation(expr.type_, bindings)
    if type_context is not None:
      assert isinstance(type_context, ArrayType), type_context
      element_type = type_context.get_element_type()
      logging.vlog(3, 'element type for array members: %s @ %s', element_type,
                   expr.span)
    elements = tuple(
        self._evaluate(e, bindings, element_type) for e in expr.members)
    if expr.has_ellipsis:
      assert type_context is not None, type_context
      elements = elements + elements[-1:] * (
          type_context.size.value - len(elements))
    return Value.make_array(elements)

  def _evaluate_ConstantArray(  # pylint: disable=invalid-name
      self, expr: ast.ConstantArray, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    """Evaluates a 'ConstantArray' AST node to a value."""
    return self._evaluate_Array(expr, bindings, type_context)

  def _evaluate_ModRef(  # pylint: disable=invalid-name
      self, expr: ast.ModRef, bindings: Bindings,
      _: Optional[ConcreteType]) -> Value:
    """Evaluates a 'ModRef' AST node to a value."""
    logging.vlog(3, 'Evaluating ModRef %s @ %s; bindings: %s', expr, expr.span,
                 bindings.keys())
    mod = bindings.resolve_mod(expr.mod.identifier)
    f = mod.get_function(expr.value)
    assert f.get_containing_module() == mod, (f, f.get_containing_module(), mod)
    return Value.make_function(mod, f)

  def _call_builtin_fn(
      self,
      builtin: Builtin,
      args: Sequence[Value],
      span: Span,
      invocation: Optional[ast.Invocation] = None,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Calls a builtin function identified via a 'builtin' enum value."""
    name = builtin.to_name()
    if hasattr(builtins, name):
      f = getattr(builtins, name)
    elif name in ('slt', 'sgt', 'sge', 'sle'):
      f = self._builtin_scmp(method=name)
    else:
      f = getattr(self, f'_builtin_{name}')
    result = f(args, span, invocation, symbolic_bindings)
    assert isinstance(result, Value), (result, f)
    return result

  def _call_fn_value(
      self,
      fv: Value,
      args: Sequence[Value],
      span: Span,
      invocation: Optional[ast.Invocation] = None,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Calls function values, either a builtin or a user defined function."""
    if fv.is_builtin_function():
      return self._call_builtin_fn(fv.get_builtin_fn(), args, span, invocation,
                                   symbolic_bindings)
    else:
      _, f = fv.get_user_fn_data()

    return self._evaluate_fn(f, args, span, invocation, symbolic_bindings)

  def _evaluate_Invocation(  # pylint: disable=invalid-name
      self, expr: ast.Invocation, bindings: Bindings,
      _: Optional[ConcreteType]) -> Optional[Value]:
    """Evaluates an 'Invocation' AST node to a value."""
    if self._trace_all and isinstance(
        expr.callee,
        ast.NameRef) and expr.callee.name_def.identifier == 'trace':
      # Safe to skip this and return nothing if this is a trace invocation;
      # trace isn't an input to any downstream expressions.
      return Value.make_nil()
    arg_values = [self._evaluate(arg, bindings) for arg in expr.args]
    callee_value = self._evaluate(expr.callee, bindings)
    if not callee_value.is_function():
      raise EvaluateError(
          expr.callee.span,
          'Callee value is not a function (should have been determined during type inference); got: {}'
          .format(callee_value))
    fn_symbolic_bindings = ()
    if bindings.fn_ctx:
      # The symbolic bindings of this invocation were already computed during
      # typechecking.
      fn_symbolic_bindings = self._type_info.get_invocation_symbolic_bindings(
          expr, bindings.fn_ctx.sym_bindings)
    return self._call_fn_value(
        callee_value,
        arg_values,
        expr.span,
        expr,
        symbolic_bindings=fn_symbolic_bindings)

  def _perform_trace(self, lhs: Text, span: Span, value: Value) -> None:
    """Actually writes the tracing output to stderr."""
    leader = 'trace of {} @ {}:'.format(lhs, span)
    if sys.stderr.isatty():
      print(termcolor.colored(leader, color='blue'), value, file=sys.stderr)
    else:
      print(leader, value, file=sys.stderr)

  def _optional_trace(self, expr: ast.Expr, result: Value) -> None:
    """Traces the current experession if "trace all" mode is active.

    Args:
      expr: The expression to trace.
      result: The result of evaluating the given expression.
    """
    # We don't need to trace trace (obv), or Lets - we just want to see the
    # non-Let bodies.
    # NameRefs and ModRefs also add a lot of noise w/o a lot of value.
    is_trace_instance = isinstance(expr, ast.Invocation) and isinstance(
        expr.callee, ast.NameRef) and expr.callee.name_def.identifier == 'trace'
    is_let_instance = isinstance(expr, ast.Let)

    if (not is_trace_instance and not is_let_instance and
        not result.is_function()):
      self._perform_trace(str(expr), expr.span, result)

  def _evaluate(self,
                expr: ast.Expr,
                bindings: Bindings,
                type_context: Optional[ConcreteType] = None) -> Value:
    """Entry point for evaluating an expression to a value.

    Args:
      expr: Expression AST node to evaluate.
      bindings: Current bindings for this evaluation (i.e. ident: value map).
      type_context: If a type is deduced from surrounding context, it is
        provided via this argument.

    Returns:
      The value that the AST node evaluates to.

    Raises:
      EvaluateError: If an error occurs during evaluation. This also attempts to
        print a rough expression-stack-trace for determining the provenance of
        an error to stderr.
    """
    clsname = expr.__class__.__name__
    logging.vlog(3, 'Evaluating %s: %s', clsname, expr)
    handler = getattr(cpp_evaluate, f'evaluate_{clsname}', None)
    if handler is None:
      handler = getattr(self, '_evaluate_{}'.format(clsname))
    try:
      result = handler(expr, bindings, type_context)
      logging.vlog(3, 'Evaluated %s: %s => %s', clsname, expr, result)
      if self._trace_all and result is not None:
        self._optional_trace(expr, result)
      assert isinstance(result, Value), (result, handler)
      return result
    except (AssertionError, EvaluateError, TypeError) as e:
      # Give some more helpful traceback context in expression evaluation for
      # where errors come from.
      if isinstance(e, AssertionError):
        kind = 'assertion'
      elif isinstance(e, TypeError):
        kind = 'python type'
      else:
        kind = 'evaluation'
      print('{} error @ {}: {}'.format(kind, expr.span, e), file=sys.stderr)
      raise

  def evaluate_literal(self, expr: ast.Expr) -> Value:
    return self._evaluate(expr, Bindings(None))

  def evaluate_expr(self, expr: ast.Expr, bindings: Bindings) -> Value:
    """Evaluates a stand-alone expression with the given bindings."""
    return self._evaluate(expr, bindings)

  def _builtin_map(
      self,
      args: Sequence[Value],
      span: Span,
      expr: ast.Invocation,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Implements the 'map' builtin."""
    if len(args) != 2:
      raise EvaluateError(
          span,
          'Invalid number of arguments to map; got {} want 2'.format(len(args)))
    inputs, map_fn = args
    outputs = []
    for input_ in inputs.get_elements():
      ret = self._call_fn_value(map_fn, [input_], span, expr, symbolic_bindings)
      outputs.append(ret)

    return Value.make_array(tuple(outputs))

  def _builtin_trace(
      self,
      args: Sequence[Value],
      span: Span,
      expr: ast.Invocation,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Implements the 'trace' builtin."""
    if len(args) != 1:
      raise ValueError(
          'Invalid number of arguments to trace; got {} want 1'.format(
              len(args)))

    self._perform_trace(expr.format_args(), span, args[0])
    assert isinstance(args[0], Value), args[0]
    return args[0]

  def _builtin_enumerate(
      self,
      args: Sequence[Value],
      span: Span,
      expr: ast.Invocation,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Implements 'enumerate' builtin; decorates array with range of indices."""
    if len(args) != 1:
      raise EvaluateError(
          span,
          'Invalid number of arguments to enumerate; got {} want 1'.format(
              len(args)))
    if not args[0].is_array():
      raise EvaluateError(
          span,
          'Invalid argument to enumerate; want array, got {}'.format(args[0]))
    array = args[0].get_elements()
    elements = []
    for i, v in enumerate(array):
      elements.append(
          Value.make_tuple((Value.make_ubits(bit_count=32, value=i), v)))
    return Value.make_array(tuple(elements))

  def _builtin_range(
      self,
      args: Sequence[Value],
      span: Span,
      expr: ast.Invocation,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Implements 'range' builtin; populates an array with a range of values."""
    if len(args) == 1:
      rhs, = args
      lhs = Value.make_ubits(bit_count=rhs.get_bit_count(), value=0)
    elif len(args) != 2:
      raise EvaluateError(
          span, 'Invalid number of arguments to range; got {} want 2'.format(
              len(args)))
    else:
      lhs, rhs = args
    elements = []
    while lhs.lt(rhs).is_true():
      elements.append(lhs)
      lhs = lhs.add(Value.make_ubits(bit_count=rhs.get_bit_count(), value=1))
    return Value.make_array(tuple(elements))

  def _builtin_update(
      self,
      args: Sequence[Value],
      span: Span,
      expr: ast.Invocation,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Implements 'update' builtin."""
    if len(args) != 3:
      raise EvaluateError(
          span, 'Invalid number of arguments to update; got {} want 3'.format(
              len(args)))
    original, index, value = args
    return original.update(index, value)

  def _builtin_scmp(
      self, method: Text
  ) -> Callable[
      [Sequence[Value], Span, ast.Invocation, Optional[SymbolicBindings]],
      Value]:
    """Returns a signed-comparison function for use as a builtin."""

    def scmp(args: Sequence[Value],
             span: Span,
             expr: ast.Invocation,
             symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
      if len(args) != 2:
        raise EvaluateError(
            span, 'Invalid number of arguments to {}; got {} want 2'.format(
                method, len(args)))
      lhs, rhs = args
      return lhs.scmp(rhs, method)

    return scmp

  def _evaluate_derived_parametrics(self, fn: ast.Function, bindings: Bindings,
                                    bound_dims: Dict[Text, int]):
    """Evaluates the parametric values derived from other parametric values.

    Populates the "bindings" mapping with results computed by the typechecker.

    For example, in:

      fn [X: u32, Y: u32 = X+X] f(x: bits[X]) { ... }

    Args:
      fn: Function to evaluate parametric bindings for.
      bindings: Bindings mapping to populate with newly evaluated parametric
        binding names.
      bound_dims: Parametric bindings computed by the typechecker.
    """
    # All symbolic bindings should have been computed by the typechecker
    assert len(bound_dims) == len(fn.parametric_bindings)
    for parametric in fn.parametric_bindings:
      if parametric.name.identifier in bindings.keys():
        # Already bound.
        continue
      type_ = self._evaluate_TypeAnnotation(parametric.type_, bindings)
      # We already computed derived parametrics in parametric_instantiator.py
      # All that's left is to add it to the current Bindings
      raw_value = bound_dims[parametric.name.identifier]
      wrapped_value = Value.make_ubits(type_.get_total_bit_count().value,
                                       raw_value)
      bindings.add_value(parametric.name.identifier, wrapped_value)

  def _evaluate_fn_with_interpreter(
      self, fn: ast.Function, args: Sequence[Value], span: Span,
      expr: Optional[ast.Invocation],
      symbolic_bindings: Optional[SymbolicBindings]) -> Value:
    """Evaluates the user defined function fn as an invocation against args.

    Args:
      fn: The user-defined function to evaluate.
      args: The argument with which the user-defined function is being invoked.
      span: The source span of the invocation.
      expr: The invocation node
      symbolic_bindings: Tuple containing the symbolic bindings to use in
        the evaluation of this function body (computed by the typechecker)

    Returns:
      The value that results from evaluating the function on the arguments.

    Raises:
      EvaluateError: If the types annotated on either parameters or the return
        type do not match with the values presented as arguments / the value
        resulting from the function evaluation.
    """
    assert isinstance(args, (list, tuple)), repr(args)
    if len(args) != len(fn.params):
      raise EvaluateError(
          span,
          'Argument arity mismatch for invocation; want: {} got: {}'.format(
              len(fn.params), len(args)))

    m = fn.get_containing_module()
    bindings = self._make_top_level_bindings(m)
    logging.vlog(3, 'Module %r was containing function %r; bindings: %s', m, fn,
                 bindings.keys())
    assert '__top_level_bindings_' + m.name in bindings.keys()

    # Bind all args to the parameter identifiers.
    bound_dims = {} if not symbolic_bindings else dict(
        symbolic_bindings)  # type: Dict[Text, int]
    self._evaluate_derived_parametrics(fn, bindings, bound_dims)
    bindings.fn_ctx = FnCtx(m.name, fn.name.identifier, symbolic_bindings)
    for param, arg in zip(fn.params, args):
      bindings.add_value(param.name.identifier, arg)

    result = self._evaluate(fn.body, bindings)
    return result

  def _evaluate_fn(
      self,
      fn: ast.Function,
      args: Sequence[Value],
      span: Span,
      expr: Optional[ast.Invocation] = None,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Wraps _eval_fn_with_interpreter() to compare with JIT execution.

    Unless this Interpreter was created with an ir_package, this does nothing
    more than call _eval_fn_with_interpreter(). Otherwise, fn is executed with
    the LLVM IR JIT and its return value is compared against the interpreted
    value as a consistency check.

    TODO(hjmontero): 2020-8-4 This OK because there are no side effects. We
    should investigate what happens when there are side effects (e.g. DSLX fatal
    errors).

    Args:
      fn: Function to evaluate.
      args: Actual arguments used to invoke the function.
      span: Span of the invocation causing this evaluation.
      expr: Invocation AST node causing this evaluation.
      symbolic_bindings: Symbolic bindings to be used for this function
        evaluation present (if the function is parameteric).

    Returns:
      The value that results from DSL interpretation.
    """
    has_child_type_info = expr and self._type_info.has_instantiation(
        expr, symbolic_bindings)
    invocation_type_info = (
        self._type_info.get_instantiation(expr, symbolic_bindings)
        if has_child_type_info else self._type_info)

    @contextlib.contextmanager
    def ntt_swap(new_ntt):
      old_ntt = self._type_info
      self._type_info = new_ntt
      yield
      self._type_info = old_ntt

    with ntt_swap(invocation_type_info):
      interpreter_value = self._evaluate_fn_with_interpreter(
          fn, args, span, expr, symbolic_bindings)

    ir_name = ir_name_mangler.mangle_dslx_name(fn.name.identifier,
                                               fn.get_free_parametric_keys(),
                                               fn.get_containing_module(),
                                               symbolic_bindings)

    if self._ir_package:
      # TODO(hjmontero): 2020-07-28 Cache JIT function so we don't have to
      # create it every time. This requires us to figure out how to wrap
      # IrJit::Create().
      ir_function = self._ir_package.get_function(ir_name)
      try:
        ir_args = jit_comparison.convert_args_to_ir(args)

        jit_value = ir_jit.ir_jit_run(ir_function, ir_args)
        jit_comparison.compare_values(interpreter_value, jit_value)
      except (jit_comparison.UnsupportedJitConversionError,
              jit_comparison.JitMiscompareError) as e:
        builtins.throw_fail_error(expr.span if expr else fn.span, str(e))

    return interpreter_value

  def _do_import(self, subject: import_fn.ImportTokens,
                 span: Span) -> ast.Module:
    """Handles an import as specified by a top level module statement."""
    if self._f_import is None:
      raise EvaluateError(span,
                          'Cannot import, no import capability was provided.')
    imported_module, imported_type_info = self._f_import(subject)
    self._type_info.update(imported_type_info)
    return imported_module

  def _make_top_level_bindings(self, m: ast.Module) -> Bindings:
    """Creates a fresh set of bindings for use in module-level evaluation.

    Args:
      m: The module that the top level bindings are being made for, used to
        populate constants / enums.

    Returns:
      Bindings containing builtins and function identifiers at the top level of
      the module.
    """

    def is_wip(m: ast.Module, c: ast.Constant) -> Optional[Value]:
      """Returns whether the constant is in the process of being computed."""
      status = self._wip.get((m, c))
      logging.vlog(3, 'Constant eval status %r %r: %r', m, c, status)
      return status is _WipSentinel

    def note_wip(m: ast.Module, c: ast.Constant,
                 v: Optional[Value]) -> Optional[Value]:
      assert isinstance(m, ast.Module), repr(m)
      assert isinstance(c, ast.Constant), repr(c)
      assert v is None or isinstance(v, Value), repr(v)

      if v is None:  # Starting evaluation, attempting to mark as WIP.
        current = self._wip.get((m, c))
        if current is not None and current is not _WipSentinel:
          assert isinstance(current, Value), repr(current)
          return current  # Already computed.
        logging.vlog(3, 'Noting WIP constant eval: %r %r', m, c)
        self._wip[(m, c)] = _WipSentinel
        return None

      logging.vlog(3, 'Noting complete constant eval: %r %r => %r', m, c, v)
      self._wip[(m, c)] = v
      return v

    # Hack to avoid circular dependency on the importer definition.
    typecheck = getattr(self._f_import, '_typecheck', None)
    cache = getattr(self._f_import, '_cache', None)

    result = cpp_evaluate.make_top_level_bindings(m, typecheck, self._evaluate,
                                                  is_wip, note_wip, cache)
    assert '__top_level_bindings_' + m.name in result.keys(), (m, result.keys())
    return result

  def run_quickcheck(self, quickcheck: ast.QuickCheck, seed: int) -> None:
    """Runs a quickcheck AST node (via the LLVM JIT)."""
    assert self._ir_package
    fn = quickcheck.f
    ir_name = ir_name_mangler.mangle_dslx_name(fn.name.identifier,
                                               fn.get_free_parametric_keys(),
                                               self._module, ())

    ir_function = self._ir_package.get_function(ir_name)
    argsets, results = ir_jit.quickcheck_jit(ir_function, seed,
                                             quickcheck.test_count)
    last_result = results[-1].get_bits().to_uint()
    if not last_result:
      last_argset = argsets[-1]
      fn_type = self._type_info[fn]
      assert isinstance(fn_type, FunctionType), fn_type
      fn_param_types = fn_type.params
      dslx_argset = [
          str(jit_comparison.ir_value_to_interpreter_value(arg, arg_type))
          for arg, arg_type in zip(last_argset, fn_param_types)
      ]
      builtins.throw_fail_error(
          fn.span, f'Found falsifying example after '
          f'{len(results)} tests: {dslx_argset}')

  def run_test(self, name: Text) -> None:
    bindings = self._make_top_level_bindings(self._module)
    test = self._module.get_test(name)
    bindings.fn_ctx = FnCtx(self._module.name, '{}_test'.format(name), ())
    result = self._evaluate(test.body, bindings)
    if not result.is_nil_tuple():
      raise EvaluateError(
          test.body.span,
          'Want test to return nil tuple, but got {}'.format(result))

  def run_function(self, name: Text, args: Sequence[Value]) -> Value:
    f = self._module.get_function(name)
    assert not f.is_parametric()
    fake_pos = Pos('<fake>', 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    return self._evaluate_fn(f, args, fake_span, symbolic_bindings=())
