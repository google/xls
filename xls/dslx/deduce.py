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

# pylint: disable=invalid-name

"""Type system deduction rules for AST nodes."""

import typing
from typing import Union, Callable, Type, Tuple, Set

from absl import logging

from xls.dslx import ast_helpers
from xls.dslx import bit_helpers
from xls.dslx import concrete_type_helpers
from xls.dslx import dslx_builtins
from xls.dslx import parametric_instantiator
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_deduce
from xls.dslx.python import cpp_scanner as scanner
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import ConcreteTypeDim
from xls.dslx.python.cpp_concrete_type import EnumType
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_deduce import DeduceCtx
from xls.dslx.python.cpp_deduce import type_inference_error as TypeInferenceError
from xls.dslx.python.cpp_deduce import xls_type_error as XlsTypeError
from xls.dslx.python.cpp_parametric_expression import ParametricAdd
from xls.dslx.python.cpp_parametric_expression import ParametricExpression
from xls.dslx.python.cpp_parametric_expression import ParametricSymbol
from xls.dslx.python.cpp_pos import Span
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.python.cpp_type_info import TypeInfo
from xls.dslx.python.cpp_type_info import TypeMissingError


# Dictionary used as registry for rule dispatch based on AST node class.
RULES = {
    ast.Binop: cpp_deduce.deduce_Binop,
    ast.Cast: cpp_deduce.deduce_Cast,
    ast.Constant: cpp_deduce.deduce_ConstantDef,
    ast.EnumDef: cpp_deduce.deduce_EnumDef,
    ast.For: cpp_deduce.deduce_For,
    ast.Let: cpp_deduce.deduce_Let,
    ast.Number: cpp_deduce.deduce_Number,
    ast.Param: cpp_deduce.deduce_Param,
    ast.StructDef: cpp_deduce.deduce_StructDef,
    ast.Ternary: cpp_deduce.deduce_Ternary,
    ast.TypeDef: cpp_deduce.deduce_TypeDef,
    ast.TypeRef: cpp_deduce.deduce_TypeRef,
    ast.Unop: cpp_deduce.deduce_Unop,
    ast.XlsTuple: cpp_deduce.deduce_XlsTuple,
}


RuleFunction = Callable[[ast.AstNode, DeduceCtx], ConcreteType]


def _rule(cls: Type[ast.AstNode]):
  """Decorator for a type inference rule that pertains to class 'cls'."""

  def register(f):
    # Register the checked function as the rule.
    RULES[cls] = f
    return f

  return register


def _resolve_colon_ref_to_fn(ref: ast.ColonRef, ctx: DeduceCtx) -> ast.Function:
  """Resolves ref to an AST function."""
  assert isinstance(ref.subject, ast.NameRef), ref.subject
  definer = ref.subject.name_def.definer
  assert isinstance(definer, ast.Import), definer
  imported_module, _ = ctx.type_info.get_imported(definer)
  return imported_module.get_function(ref.attr)


@_rule(ast.ConstantArray)
def _deduce_ConstantArray(
    self: ast.ConstantArray, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a ConstantArray AST node."""
  # We permit constant arrays to drop annotations for numbers as a convenience
  # (before we have unifying type inference) by allowing constant arrays to have
  # a leading type annotation. If they don't have a leading type annotation,
  # just fall back to normal array type inference, if we encounter a number
  # without a type annotation we'll flag an error per usual.
  if self.type_ is None:
    return _deduce_Array(self, ctx)

  # Determine the element type that corresponds to the annotation and go mark
  # any un-typed numbers in the constant array as having that type.
  concrete_type = deduce(self.type_, ctx)
  if not isinstance(concrete_type, ArrayType):
    raise TypeInferenceError(
        self.type_.span, concrete_type,
        f'Annotated type for array literal must be an array type; got {concrete_type.get_debug_type_name()} {self.type_}'
    )
  element_type = concrete_type.get_element_type()
  for member in self.members:
    assert ast.is_constant(member)
    if isinstance(member, ast.Number) and not member.type_:
      ctx.type_info[member] = element_type
      cpp_deduce.check_bitwidth(member, element_type)
  # Use the base class to check all members are compatible.
  _deduce_Array(self, ctx)
  return concrete_type


def _create_element_invocation(owner: ast.AstNodeOwner, span_: Span,
                               callee: ast.NameRef,
                               arg_array: ast.Expr) -> ast.Invocation:
  """Creates a function invocation on the first element of the given array.

  We need to create a fake invocation to deduce the type of a function
  in the case where map is called with a builtin as the map function. Normally,
  map functions (including parametric ones) have their types deduced when their
  ast.Function nodes are encountered (where a similar fake ast.Invocation node
  is created).

  Builtins don't have ast.Function nodes, so that inference can't occur, so we
  essentually perform that synthesis and deduction here.

  Args:
    owner: AST node owner.
    span_: The location in the code where analysis is occurring.
    callee: The function to be invoked.
    arg_array: The array of arguments (at least one) to the function.

  Returns:
    An invocation node for the given function when called with an element in the
    argument array.
  """
  assert isinstance(callee, ast.NameRef), callee
  annotation = ast_helpers.make_builtin_type_annotation(
      owner, span_, scanner.Token(span_, scanner.Keyword.U32), ())
  index_number = ast.Number(owner, span_, '32', ast.NumberKind.OTHER,
                            annotation)
  index = ast.Index(owner, span_, arg_array, index_number)
  return ast.Invocation(owner, span_, callee, (index,))


def _check_parametric_invocation(parametric_fn: ast.Function,
                                 invocation: ast.Invocation,
                                 symbolic_bindings: SymbolicBindings,
                                 ctx: DeduceCtx):
  """Checks the parametric fn body using the invocation's symbolic bindings."""
  if isinstance(invocation.callee, ast.ColonRef):
    # We need to typecheck this function with respect to its own module.
    # Let's use typecheck._check_function_or_test_in_module() to do this
    # in case we run into more dependencies in that module.
    if ctx.type_info.has_instantiation(invocation, symbolic_bindings):
      # We've already typechecked this imported parametric function using
      # these symbolic bindings.
      return

    import_node = invocation.callee.subject.name_def.definer
    assert isinstance(import_node, ast.Import)

    imported_module, imported_type_info = ctx.type_info.get_imported(
        import_node)
    invocation_imported_type_info = TypeInfo(
        imported_module, parent=imported_type_info)
    imported_ctx = ctx.make_ctx(invocation_imported_type_info, imported_module)
    imported_ctx.add_fn_stack_entry(parametric_fn.name.identifier,
                                    symbolic_bindings)
    ctx.typecheck_function(parametric_fn, imported_ctx)

    ctx.type_info.add_instantiation(invocation, symbolic_bindings,
                                    invocation_imported_type_info)
    return

  assert isinstance(invocation.callee, ast.NameRef), invocation.callee
  has_instantiation = ctx.type_info.has_instantiation(invocation,
                                                      symbolic_bindings)
  # We need to typecheck this function with respect to its own module
  # Let's take advantage of the existing try-catch mechanism in
  # typecheck._check_function_or_test_in_module().

  try:
    # See if the body is present in the type_info mapping (we do this just
    # to observe if it raises an exception).
    ctx.type_info[parametric_fn.body]
  except TypeMissingError as e:
    # If we've already typechecked the parametric function with the
    # current symbolic bindings, no need to do it again.
    if not has_instantiation:
      # Let's typecheck this parametric function using the symbolic bindings
      # we just derived to make sure they check out ok.
      cpp_deduce.type_missing_error_set_node(e, invocation.callee.name_def)
      ctx.add_fn_stack_entry(parametric_fn.name.identifier, symbolic_bindings)
      ctx.add_derived_type_info()
      raise

  if not has_instantiation:
    # If we haven't yet stored a type_info for these symbolic bindings
    # and we're at this point, it means that we just finished typechecking
    # the parametric function. Let's store the results.
    ctx.type_info.parent.add_instantiation(invocation, symbolic_bindings,
                                           ctx.type_info)
    ctx.pop_derived_type_info()


@_rule(ast.Invocation)
def _deduce_Invocation(self: ast.Invocation, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an Invocation AST node."""
  logging.vlog(5, 'Deducing type for invocation: %s', self)
  arg_types = []
  fn_symbolic_bindings = ctx.peek_fn_stack().symbolic_bindings
  for arg in self.args:
    try:
      arg_types.append(cpp_deduce.resolve(deduce(arg, ctx), ctx))
    except TypeMissingError as e:
      # These nodes could be ColonRefs or NameRefs.
      callee_is_map = isinstance(
          self.callee, ast.NameRef) and self.callee.name_def.identifier == 'map'
      arg_is_builtin = isinstance(
          arg, ast.NameRef
      ) and arg.name_def.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES
      if callee_is_map and arg_is_builtin:
        invocation = _create_element_invocation(ctx.module, self.span, arg,
                                                self.args[0])
        arg_types.append(cpp_deduce.resolve(deduce(invocation, ctx), ctx))
      else:
        raise

  try:
    # This will get us the type signature of the function.
    # If the function is parametric, we won't check its body
    # until after we have symbolic bindings for it
    callee_type = deduce(self.callee, ctx)
  except TypeMissingError as e:
    cpp_deduce.type_missing_error_set_span(e, self.span)
    cpp_deduce.type_missing_error_set_user(e, self)
    raise

  if not isinstance(callee_type, FunctionType):
    raise XlsTypeError(self.callee.span, callee_type, None,
                       'Callee does not have a function type.')

  if isinstance(self.callee, ast.ColonRef):
    callee_fn = _resolve_colon_ref_to_fn(self.callee, ctx)
    callee_name = callee_fn.identifier
  else:
    assert isinstance(self.callee, ast.NameRef), self.callee
    callee_name = self.callee.identifier
    callee_fn = ctx.module.get_function(callee_name)

  # We need to deduce the type of all Invocation parametrics so they're in the
  # type cache.
  for parametric in self.parametrics:
    deduce(parametric, ctx)

  # Create new parametric bindings that capture the constraints from
  # the specified parametrics.
  new_bindings = []
  for binding, value in zip(callee_fn.parametric_bindings, self.parametrics):
    assert isinstance(value, ast.Expr)
    binding_type = deduce(binding.type_, ctx)
    value_type = deduce(value, ctx)
    if binding_type != value_type:
      raise XlsTypeError(self.callee.span, binding.type_, value.type_,
                         'Explicit parametric type did not match its binding.')
    new_binding = binding.clone(value)
    new_bindings.append(new_binding)

  for remaining_binding in callee_fn.parametric_bindings[len(self.parametrics
                                                            ):]:
    new_bindings.append(remaining_binding)

  caller_sym_bindings = fn_symbolic_bindings
  csb_dict = caller_sym_bindings.to_dict()

  # Map resolved parametrics from the caller's context onto the corresponding
  # symbols in the callee's.
  explicit_bindings = {}
  for new_binding in new_bindings:
    if isinstance(new_binding.expr,
                  ast.NameRef) and new_binding.expr.identifier in csb_dict:
      explicit_bindings[new_binding.name.identifier] = csb_dict[
          new_binding.expr.identifier]

  self_type, callee_sym_bindings = parametric_instantiator.instantiate_function(
      self.span, callee_type, tuple(arg_types), ctx, tuple(new_bindings),
      explicit_bindings)

  ctx.type_info.add_invocation_symbolic_bindings(self, caller_sym_bindings,
                                                 callee_sym_bindings)

  if callee_fn.is_parametric():
    # Now that we have callee_sym_bindings, let's use them to typecheck the body
    # of callee_fn to make sure these values actually work
    _check_parametric_invocation(callee_fn, self, callee_sym_bindings, ctx)

  return self_type


def _deduce_slice_type(self: ast.Index, ctx: DeduceCtx,
                       lhs_type: ConcreteType) -> ConcreteType:
  """Deduces the concrete type of an Index AST node with a slice spec."""
  index_slice = self.index
  assert isinstance(index_slice, (ast.Slice, ast.WidthSlice)), index_slice

  # TODO(leary): 2019-10-28 Only slicing bits types for now, and only with
  # number ast nodes, generalize to arrays and constant expressions.
  if not isinstance(lhs_type, BitsType):
    raise XlsTypeError(self.span, lhs_type, None,
                       'Value to slice is not of "bits" type.')

  bit_count = lhs_type.get_total_bit_count().value

  if isinstance(index_slice, ast.WidthSlice):
    start = index_slice.start
    if isinstance(start, ast.Number) and start.type_ is None:
      start_type = lhs_type.to_ubits()
      resolved_start_type = cpp_deduce.resolve(start_type, ctx)
      start_int = ast_helpers.get_value_as_int(start)
      if not bit_helpers.fits_in_bits(
          start_int,
          resolved_start_type.get_total_bit_count().value):
        raise TypeInferenceError(
            start.span, resolved_start_type,
            'Cannot fit {} in {} bits (inferred from bits to slice).'.format(
                start_int,
                resolved_start_type.get_total_bit_count().value))
      ctx.type_info[start] = start_type
    else:
      start_type = deduce(start, ctx)

    # Check the start is unsigned.
    if start_type.signed:
      raise TypeInferenceError(
          start.span, start_type,
          'Start index for width-based slice must be unsigned.')

    width_type = deduce(index_slice.width, ctx)
    if isinstance(width_type.get_total_bit_count().value, int) and isinstance(
        lhs_type.get_total_bit_count().value,
        int) and width_type.get_total_bit_count(
        ).value > lhs_type.get_total_bit_count().value:
      raise XlsTypeError(
          start.span, lhs_type, width_type,
          'Slice type must have <= original number of bits; attempted slice from {} to {} bits.'
          .format(lhs_type.get_total_bit_count().value,
                  width_type.get_total_bit_count().value))

    # Check the width type is bits-based (no enums, since value could be out
    # of range of the enum values).
    if not isinstance(width_type, BitsType):
      raise TypeInferenceError(
          self.span, width_type,
          'Require a bits-based type for width-based slice.')

    # The width type is the thing returned from the width-slice.
    return width_type

  assert isinstance(index_slice, ast.Slice), index_slice
  limit = ast_helpers.get_value_as_int(
      index_slice.limit) if index_slice.limit else None
  # PyType has trouble figuring out that start is definitely an Number at this
  # point.
  start = index_slice.start
  assert isinstance(start, (ast.Number, type(None)))
  start = ast_helpers.get_value_as_int(start) if start else None

  fn_symbolic_bindings = ctx.peek_fn_stack().symbolic_bindings
  if isinstance(bit_count, ParametricExpression):
    bit_count = bit_count.evaluate(fn_symbolic_bindings.to_dict())
  start, width = bit_helpers.resolve_bit_slice_indices(bit_count, start, limit)
  ctx.type_info.add_slice_start_width(index_slice, fn_symbolic_bindings,
                                      (start, width))
  return BitsType(signed=False, size=width)


def _deduce_tuple_index(self: ast.Index, ctx: DeduceCtx,
                        lhs_type: TupleType) -> ConcreteType:
  """Deduces the resulting type for a tuple indexing operation."""
  index = self.index

  # TODO(leary): 2020-11-09 When we add unifying type inference this will also
  # be able to be a ConstRef.
  if isinstance(index, ast.Number):
    if index.type_:
      # If the number has an annotated type, flag it as unnecessary.
      deduce(index, ctx)
      logging.warning(
          'Warning: type annotation for tuple index is unnecessary @ %s: %s',
          self.span, self)
    else:
      ctx.type_info[index] = ConcreteType.U32
    index_value = ast_helpers.get_value_as_int(index)
  else:
    raise TypeInferenceError(
        index.span, lhs_type,
        'Tuple index is not a literal number or named constant.')

  assert isinstance(index_value, int), index_value
  if index_value < 0 or index_value >= lhs_type.get_tuple_length():
    raise XlsTypeError(
        index.span, lhs_type, None,
        'Tuple index {} is out of range for this tuple type.'.format(
            index_value))
  return lhs_type.get_unnamed_members()[index_value]


@_rule(ast.Index)
def _deduce_Index(self: ast.Index, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an Index AST node."""
  lhs_type = deduce(self.lhs, ctx)

  # Check whether this is a slice-based indexing operations.
  if isinstance(self.index, (ast.Slice, ast.WidthSlice)):
    return _deduce_slice_type(self, ctx, lhs_type)

  if isinstance(lhs_type, TupleType):
    return _deduce_tuple_index(self, ctx, lhs_type)

  if not isinstance(lhs_type, ArrayType):
    raise TypeInferenceError(self.lhs.span, lhs_type,
                             'Value to index is not an array.')

  index_type = deduce(self.index, ctx)
  index_ok = isinstance(index_type,
                        BitsType) and not isinstance(index_type, ArrayType)
  if not index_ok:
    raise XlsTypeError(self.index.span, index_type, None,
                       'Index type is not scalar bits.')
  return lhs_type.get_element_type()


def _unify_WildcardPattern(_self: ast.WildcardPattern, _type: ConcreteType,
                           _ctx: DeduceCtx) -> None:
  pass  # Wildcard matches any type.


def _unify_NameDefTree(self: ast.NameDefTree, type_: ConcreteType,
                       ctx: DeduceCtx) -> None:
  """Unifies the NameDefTree AST node with the observed RHS type type_."""
  resolved_rhs_type = cpp_deduce.resolve(type_, ctx)
  if self.is_leaf():
    leaf = self.get_leaf()
    if isinstance(leaf, ast.NameDef):
      ctx.type_info[leaf] = resolved_rhs_type
    elif isinstance(leaf, ast.WildcardPattern):
      pass
    elif isinstance(leaf, (ast.Number, ast.ColonRef)):
      resolved_leaf_type = cpp_deduce.resolve(deduce(leaf, ctx), ctx)
      if resolved_leaf_type != resolved_rhs_type:
        raise TypeInferenceError(
            self.span, resolved_rhs_type,
            'Conflicting types; pattern expects {} but got {} from value'
            .format(resolved_rhs_type, resolved_leaf_type))
    else:
      assert isinstance(leaf, ast.NameRef), repr(leaf)
      ref_type = ctx.type_info[leaf.name_def]
      resolved_ref_type = cpp_deduce.resolve(ref_type, ctx)
      if resolved_ref_type != resolved_rhs_type:
        raise TypeInferenceError(
            self.span, resolved_rhs_type,
            'Conflicting types; pattern expects {} but got {} from reference'
            .format(resolved_rhs_type, resolved_ref_type))
  else:
    assert isinstance(self.tree, tuple)
    if isinstance(type_, TupleType) and type_.get_tuple_length() == len(
        self.tree):
      for subtype, subtree in zip(type_.get_unnamed_members(), self.tree):
        _unify(subtree, subtype, ctx)


def _unify(n: ast.AstNode, other: ConcreteType, ctx: DeduceCtx) -> None:
  f = globals()['_unify_{}'.format(n.__class__.__name__)]
  f(n, other, ctx)


@_rule(ast.Match)
def _deduce_Match(self: ast.Match, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Match AST node."""
  matched = deduce(self.matched, ctx)

  for arm in self.arms:
    for pattern in arm.patterns:
      _unify(pattern, matched, ctx)

  arm_types = tuple(deduce(arm, ctx) for arm in self.arms)

  resolved_arm0_type = cpp_deduce.resolve(arm_types[0], ctx)

  for i, arm_type in enumerate(arm_types[1:], 1):
    resolved_arm_type = cpp_deduce.resolve(arm_type, ctx)
    if resolved_arm_type != resolved_arm0_type:
      raise XlsTypeError(
          self.arms[i].span, resolved_arm_type, resolved_arm0_type,
          'This match arm did not have the same type as preceding match arms.')
  return resolved_arm0_type


@_rule(ast.MatchArm)
def _deduce_MatchArm(self: ast.MatchArm, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  return deduce(self.expr, ctx)


@_rule(ast.While)
def _deduce_While(self: ast.While, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a While AST node."""
  init_type = deduce(self.init, ctx)
  test_type = deduce(self.test, ctx)

  resolved_init_type = cpp_deduce.resolve(init_type, ctx)
  resolved_test_type = cpp_deduce.resolve(test_type, ctx)

  if resolved_test_type != ConcreteType.U1:
    raise XlsTypeError(self.test.span, test_type, ConcreteType.U1,
                       'Expect while-loop test to be a bool value.')

  body_type = deduce(self.body, ctx)
  resolved_body_type = cpp_deduce.resolve(body_type, ctx)

  if resolved_init_type != resolved_body_type:
    raise XlsTypeError(
        self.span, init_type, body_type,
        "While-loop init value type did not match while-loop body's "
        'result type.')
  return resolved_init_type


@_rule(ast.Carry)
def _deduce_Carry(self: ast.Carry, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  return deduce(self.loop.init, ctx)


@_rule(ast.Array)
def _deduce_Array(self: ast.Array, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an Array AST node."""
  member_types = [deduce(m, ctx) for m in self.members]
  resolved_type0 = cpp_deduce.resolve(member_types[0], ctx)
  for i, x in enumerate(member_types[1:], 1):
    resolved_x = cpp_deduce.resolve(x, ctx)
    logging.vlog(5, 'array member type %d: %s', i, resolved_x)
    if resolved_x != resolved_type0:
      raise XlsTypeError(
          self.members[i].span, resolved_type0, resolved_x,
          'Array member did not have same type as other members.')

  inferred = ArrayType(resolved_type0, len(member_types))

  if not self.type_:
    return inferred

  annotated = deduce(self.type_, ctx)
  if not isinstance(annotated, ArrayType):
    raise XlsTypeError(self.span, annotated, None,
                       'Array was not annotated with an array type.')
  resolved_element_type = cpp_deduce.resolve(annotated.get_element_type(), ctx)
  if resolved_element_type != resolved_type0:
    raise XlsTypeError(
        self.span, resolved_element_type, resolved_type0,
        'Annotated element type did not match inferred element type.')

  if self.has_ellipsis:
    # Since there are ellipsis, we determine the size from the annotated type.
    # We've already checked the element types lined up.
    return annotated
  else:
    if annotated.size != len(member_types):
      raise XlsTypeError(
          self.span, annotated, inferred,
          'Annotated array size {!r} does not match inferred array size {!r}.'
          .format(annotated.size, len(member_types)))
    return inferred


@_rule(ast.ConstRef)
@_rule(ast.NameRef)
def _deduce_NameRef(self: ast.NameRef, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a NameDef AST node."""
  try:
    result = ctx.type_info[self.name_def]
  except TypeMissingError as e:
    logging.vlog(3, 'Could not resolve name def: %s', self.name_def)
    cpp_deduce.type_missing_error_set_span(e, self.span)
    cpp_deduce.type_missing_error_set_user(e, self)
    raise
  return result


def _deduce_enum_ref_internal(span: Span, enum_type: EnumType,
                              attr: str) -> ConcreteType:
  """Checks that attr is available on the enum being colon-referenced."""
  # Check the name we're accessing is actually defined on the enum.
  enum = enum_type.get_nominal_type()
  assert isinstance(enum, ast.EnumDef), enum
  if not enum.has_value(attr):
    raise TypeInferenceError(
        span, None,
        f'Name {attr!r} is not defined by the enum {enum.identifier}')
  return enum_type


@_rule(ast.ColonRef)
def _deduce_ColonRef(self: ast.ColonRef, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a ColonRef AST node."""
  if isinstance(self.subject, ast.NameRef) and isinstance(
      self.subject.name_def.definer, ast.Import):
    # Importing from an (imported) module.
    import_node: ast.Import = self.subject.name_def.definer
    imported_module, imported_type_info = ctx.type_info.get_imported(
        import_node)
    elem = imported_module.find_member_with_name(self.attr)
    logging.vlog(
        3, 'Resolving type info for module element: %s referred to by %s', elem,
        self)

    if elem is None:
      raise TypeInferenceError(
          self.span, None,
          f'Attempted to refer to module {imported_module.name} member {self.attr!r} which does not exist.'
      )

    if not elem.public:
      raise TypeInferenceError(
          self.span, None,
          f'Attempted to refer to module type {elem} that is not public.')

    if isinstance(elem, ast.Function) and elem.name not in imported_type_info:
      logging.vlog(
          2, 'Function name not in imported_type_info; must be parametric: %r',
          elem.name)
      assert elem.is_parametric()
      # We don't type check parametric functions until invocations.
      # Let's typecheck this imported parametric function with respect to its
      # module (this will only get the type signature, body gets typechecked
      # after parametric instantiation).
      imported_ctx = ctx.make_ctx(imported_type_info, imported_module)
      peek_entry = ctx.peek_fn_stack()
      imported_ctx.add_fn_stack_entry(peek_entry.name,
                                      peek_entry.symbolic_bindings)
      ctx.typecheck_function(elem, imported_ctx)
      ctx.type_info.update(imported_ctx.type_info)
      imported_type_info = imported_ctx.type_info

    return imported_type_info[elem]

  try:
    subject_type = deduce(self.subject, ctx)
  except TypeMissingError as e:
    logging.vlog(3, 'Could not resolve ColonRef subject to type: %s @ %s',
                 self.subject, self.span)
    cpp_deduce.type_missing_error_set_span(e, self.span)
    cpp_deduce.type_missing_error_set_user(e, self)
    raise

  if isinstance(subject_type, EnumType):
    return _deduce_enum_ref_internal(self.span, subject_type, self.attr)
  raise NotImplementedError(self, subject_type)


def _dim_to_parametric(self: ast.TypeAnnotation,
                       expr: ast.Expr) -> ParametricExpression:
  """Converts a dimension expression to a 'parametric' AST node."""
  assert not isinstance(expr, ast.ConstRef), expr
  if isinstance(expr, ast.NameRef):
    return ParametricSymbol(expr.name_def.identifier, expr.span)
  if isinstance(expr, ast.Binop):
    if expr.kind == ast.BinopKind.ADD:
      return ParametricAdd(
          _dim_to_parametric(self, expr.lhs),
          _dim_to_parametric(self, expr.rhs))
  msg = 'Could not concretize type with dimension: {}.'.format(expr)
  raise TypeInferenceError(self.span, self, suffix=msg)


def _dim_to_parametric_or_int(
    self: ast.TypeAnnotation, expr: ast.Expr,
    ctx: DeduceCtx) -> Union[int, ParametricExpression]:
  """Converts dimension expression within an annotation to int or parametric."""
  if isinstance(expr, ast.Number):
    ctx.type_info[expr] = ConcreteType.U32
    return ast_helpers.get_value_as_int(expr)
  if isinstance(expr, ast.ConstRef):
    n = ctx.type_info.get_const_int(expr.name_def)
    if not isinstance(n, ast.Number):
      raise TypeInferenceError(
          span=expr.span,
          type_=None,
          suffix=f'Expected a constant integral value with the name {expr.name_def}; got {n}'
      )
    return ast_helpers.get_value_as_int(n)
  return _dim_to_parametric(self, expr)


@_rule(ast.TypeRefTypeAnnotation)
def _deduce_TypeRefTypeAnnotation(self: ast.TypeRefTypeAnnotation,
                                  ctx: DeduceCtx) -> ConcreteType:
  """Dedeuces the concrete type of a TypeRef type annotation."""
  base_type = deduce(self.type_ref, ctx)
  maybe_struct = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
      self.type_ref.type_def, _get_imported_module_via_type_info, ctx.type_info)
  if (isinstance(maybe_struct, ast.StructDef) and
      maybe_struct.is_parametric() and self.parametrics):
    base_type = _concretize_struct_annotation(ctx.module, self, maybe_struct,
                                              base_type)
  return base_type


@_rule(ast.BuiltinTypeAnnotation)
def _deduce_BuiltinTypeAnnotation(
    self: ast.BuiltinTypeAnnotation,
    ctx: DeduceCtx,  # pylint: disable=unused-argument
) -> ConcreteType:
  signedness, bits = self.signedness_and_bits
  return BitsType(signedness, bits)


@_rule(ast.TupleTypeAnnotation)
def _deduce_TupleTypeAnnotation(self: ast.TupleTypeAnnotation,
                                ctx: DeduceCtx) -> ConcreteType:
  members = []
  for member in self.members:
    members.append(deduce(member, ctx))
  return TupleType(tuple(members))


@_rule(ast.ArrayTypeAnnotation)
def _deduce_ArrayTypeAnnotation(self: ast.ArrayTypeAnnotation,
                                ctx: DeduceCtx) -> ConcreteType:
  """Deduces the concrete type of an Array type annotation."""
  dim = _dim_to_parametric_or_int(self, self.dim, ctx)
  if (isinstance(self.element_type, ast.BuiltinTypeAnnotation) and
      self.element_type.bits == 0):
    # No-volume builtin types like bits, uN, and sN.
    return BitsType(self.element_type.signedness, dim)
  element_type = deduce(self.element_type, ctx)
  result = ArrayType(element_type, dim)
  logging.vlog(4, 'array type annotation: %s => %s', self, result)
  return result


def _validate_struct_members_subset(
    members: ast_helpers.StructInstanceMembers, struct_type: ConcreteType,
    struct_text: str, ctx: DeduceCtx
) -> Tuple[Set[str], Tuple[ConcreteType], Tuple[ConcreteType]]:
  """Validates a struct instantiation is a subset of members with no dups.

  Args:
    members: Sequence of members used in instantiation. Note this may be a
      subset; e.g. in the case of splat instantiation.
    struct_type: The deduced type for the struct (instantiation).
    struct_text: Display name to use for the struct in case of an error.
    ctx: Wrapper containing node to type mapping context.

  Returns:
    A tuple containing the set of struct member names that were instantiated,
    the ConcreteTypes of the provided arguments, and the ConcreteTypes of the
    corresponding struct member definition.
  """
  assert isinstance(struct_type, TupleType), struct_type
  seen_names = set()
  arg_types = []
  member_types = []
  for k, v in members:
    if k in seen_names:
      raise TypeInferenceError(
          v.span, None,
          'Duplicate value seen for {!r} in this {!r} struct instance.'.format(
              k, struct_text))
    seen_names.add(k)
    expr_type = cpp_deduce.resolve(deduce(v, ctx), ctx)
    arg_types.append(expr_type)
    try:
      member_type = struct_type.get_member_type_by_name(k)
      member_types.append(member_type)
    except KeyError:
      raise TypeInferenceError(
          v.span, None, f'Struct {struct_text!r} has no member {k!r}, '
          'but it was provided by this instance.')

  return seen_names, tuple(arg_types), tuple(member_types)


@_rule(ast.StructInstance)
def _deduce_StructInstance(
    self: ast.StructInstance, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of the struct instantiation expression and its members."""
  logging.vlog(5, 'Deducing type for struct instance: %s', self)
  struct_type = deduce(self.struct, ctx)
  assert isinstance(struct_type, TupleType), struct_type
  assert struct_type.named, struct_type
  expected_names = set(struct_type.tuple_names)
  seen_names, arg_types, member_types = _validate_struct_members_subset(
      self.unordered_members, struct_type, self.struct_text, ctx)
  if seen_names != expected_names:
    missing = ', '.join(
        repr(s) for s in sorted(list(expected_names - seen_names)))
    raise TypeInferenceError(
        self.span, None,
        'Struct instance is missing member(s): {}'.format(missing))

  struct_def = self.struct
  if not isinstance(struct_def, ast.StructDef):
    # Traverse TypeDefs and ColonRefs until we get the struct AST node.
    struct_def = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
        struct_def, _get_imported_module_via_type_info, ctx.type_info)
  assert isinstance(struct_def, ast.StructDef), struct_def

  resolved_struct_type, _ = parametric_instantiator.instantiate_struct(
      self.span, struct_type, arg_types, member_types, ctx,
      struct_def.parametric_bindings)

  return resolved_struct_type


def _concretize_struct_annotation(module: ast.Module,
                                  type_annotation: ast.TypeRefTypeAnnotation,
                                  struct: ast.StructDef,
                                  base_type: ConcreteType) -> ConcreteType:
  """Returns concretized struct type using the provided bindings.

  For example, if we have a struct defined as `struct [N: u32, M: u32] Foo`,
  the default TupleType will be (N, M). If a type annotation provides bindings,
  (e.g. Foo[A, 16]), we will replace N, M with those values. In the case above,
  we will return (A, 16) instead.

  Args:
    module: Owning AST module for the nodes.
    type_annotation: The provided type annotation for this parametric struct.
    struct: The corresponding struct AST node.
    base_type: The TupleType of the struct, based only on the struct definition.
  """
  assert len(struct.parametric_bindings) == len(type_annotation.parametrics)
  defined_to_annotated = {}
  for defined_parametric, annotated_parametric in zip(
      struct.parametric_bindings, type_annotation.parametrics):
    assert isinstance(defined_parametric,
                      ast.ParametricBinding), defined_parametric
    if isinstance(annotated_parametric, ast.Cast):
      # Casts are "X as <type_annot>"; X can be a symbol or a number.
      expr = annotated_parametric.expr
      value = None
      if isinstance(expr, ast.Number):
        value = ast_helpers.get_value_as_int(expr.value)
      elif isinstance(expr, ast.NameRef):
        value = ParametricSymbol(expr.identifier, annotated_parametric.span)
      defined_to_annotated[defined_parametric.name.identifier] = value
    elif isinstance(annotated_parametric, ast.Number):
      defined_to_annotated[defined_parametric.name.identifier] = \
          int(annotated_parametric.value)
    else:
      assert isinstance(annotated_parametric,
                        ast.NameRef), repr(annotated_parametric)
      defined_to_annotated[defined_parametric.name.identifier] = \
          ParametricSymbol(annotated_parametric.identifier,
                           annotated_parametric.span)

  def resolver(dim: ConcreteTypeDim) -> ConcreteTypeDim:
    if isinstance(dim.value, ParametricExpression):
      return ConcreteTypeDim(dim.value.evaluate(defined_to_annotated))
    return dim

  return concrete_type_helpers.map_size(base_type, module, resolver)


def _get_imported_module_via_type_info(
    import_: ast.Import, type_info: TypeInfo) -> Tuple[ast.Module, TypeInfo]:
  """Uses type_info to retrieve the corresponding module of a ColonRef."""
  return type_info.get_imported(import_)


@_rule(ast.SplatStructInstance)
def _deduce_SplatStructInstance(
    self: ast.SplatStructInstance, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of the struct instantiation expression and its members."""
  struct_type = deduce(self.struct, ctx)
  splatted_type = deduce(self.splatted, ctx)

  assert isinstance(struct_type, TupleType), struct_type
  assert isinstance(splatted_type, TupleType), splatted_type

  # We will make sure this splat typechecks during instantiation. Let's just
  # ensure the same number of elements for now.
  assert len(struct_type.tuple_names) == len(splatted_type.tuple_names)

  (seen_names, seen_arg_types,
   seen_member_types) = _validate_struct_members_subset(self.members,
                                                        struct_type,
                                                        self.struct_text, ctx)

  arg_types = list(seen_arg_types)
  member_types = list(seen_member_types)
  for m in struct_type.tuple_names:
    if m not in seen_names:
      splatted_member_type = splatted_type.get_member_type_by_name(m)
      struct_member_type = struct_type.get_member_type_by_name(m)

      arg_types.append(splatted_member_type)
      member_types.append(struct_member_type)

  # At this point, we should have the same number of args compared to the
  # number of members defined in the struct.
  assert len(arg_types) == len(member_types)

  struct_def = self.struct
  if not isinstance(struct_def, ast.StructDef):
    # Traverse TypeDefs and ColonRefs until we get the struct AST node.
    struct_def = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
        struct_def, _get_imported_module_via_type_info, ctx.type_info)

  assert isinstance(struct_def, ast.StructDef), struct_def

  resolved_struct_type, _ = parametric_instantiator.instantiate_struct(
      self.span, struct_type, tuple(arg_types), tuple(member_types), ctx,
      struct_def.parametric_bindings)

  return resolved_struct_type


@_rule(ast.Attr)
def _deduce_Attr(self: ast.Attr, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of a struct attribute access expression."""
  struct = deduce(self.lhs, ctx)
  assert isinstance(struct, TupleType), struct
  if not struct.has_named_member(self.attr.identifier):
    raise TypeInferenceError(
        self.span, None,
        'Struct does not have a member with name {!r}.'.format(self.attr))

  return struct.get_member_type_by_name(self.attr.identifier)


def _deduce(n: ast.AstNode, ctx: DeduceCtx) -> ConcreteType:
  f = RULES[n.__class__]
  f = typing.cast(Callable[[ast.AstNode, DeduceCtx], ConcreteType], f)
  result = f(n, ctx)
  ctx.type_info[n] = result
  return result


def deduce(n: ast.AstNode, ctx: DeduceCtx) -> ConcreteType:
  """Deduces and returns the type of value produced by this expr.

  Also adds n to ctx.type_info memoization dictionary.

  Args:
    n: The AST node to deduce the type for.
    ctx: Wraps a type_info, a dictionary mapping nodes to their types.

  Returns:
    The type of this expression.

  As a side effect the type_info mapping is filled with all the deductions
  that were necessary to determine (deduce) the resulting type of n.
  """
  assert isinstance(n, ast.AstNode), n
  if n in ctx.type_info:
    result = ctx.type_info[n]
    assert isinstance(result, ConcreteType), result
  else:
    result = ctx.type_info[n] = _deduce(n, ctx)
    logging.vlog(5, 'Deduced type of %s => %s', n, result)
    assert isinstance(result, ConcreteType), \
        '_deduce did not return a ConcreteType; got: {!r}'.format(result)
  return result
