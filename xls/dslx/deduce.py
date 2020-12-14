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
from typing import Callable, Type

from absl import logging

from xls.dslx import ast_helpers
from xls.dslx import dslx_builtins
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_deduce
from xls.dslx.python import cpp_parametric_instantiator as parametric_instantiator
from xls.dslx.python import cpp_scanner as scanner
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_deduce import DeduceCtx
from xls.dslx.python.cpp_deduce import xls_type_error as XlsTypeError
from xls.dslx.python.cpp_pos import Span
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.python.cpp_type_info import TypeInfo
from xls.dslx.python.cpp_type_info import TypeMissingError


# Dictionary used as registry for rule dispatch based on AST node class.
RULES = {
    ast.Array:
        cpp_deduce.deduce_Array,
    ast.Attr:
        cpp_deduce.deduce_Attr,
    ast.Binop:
        cpp_deduce.deduce_Binop,
    ast.Carry:
        cpp_deduce.deduce_Carry,
    ast.Cast:
        cpp_deduce.deduce_Cast,
    ast.ColonRef:
        cpp_deduce.deduce_ColonRef,
    ast.Constant:
        cpp_deduce.deduce_ConstantDef,
    ast.ConstantArray:
        cpp_deduce.deduce_ConstantArray,
    ast.EnumDef:
        cpp_deduce.deduce_EnumDef,
    ast.For:
        cpp_deduce.deduce_For,
    ast.Index:
        cpp_deduce.deduce_Index,
    ast.Let:
        cpp_deduce.deduce_Let,
    ast.Match:
        cpp_deduce.deduce_Match,
    ast.MatchArm:
        cpp_deduce.deduce_MatchArm,
    ast.Number:
        cpp_deduce.deduce_Number,
    ast.Param:
        cpp_deduce.deduce_Param,
    ast.StructDef:
        cpp_deduce.deduce_StructDef,
    ast.StructInstance:
        cpp_deduce.deduce_StructInstance,
    ast.SplatStructInstance:
        cpp_deduce.deduce_SplatStructInstance,
    ast.Ternary:
        cpp_deduce.deduce_Ternary,
    ast.TypeDef:
        cpp_deduce.deduce_TypeDef,
    ast.TypeRef:
        cpp_deduce.deduce_TypeRef,
    ast.Unop:
        cpp_deduce.deduce_Unop,
    ast.While:
        cpp_deduce.deduce_While,
    ast.XlsTuple:
        cpp_deduce.deduce_XlsTuple,

    # Various type annotations.
    ast.ArrayTypeAnnotation:
        cpp_deduce.deduce_ArrayTypeAnnotation,
    ast.BuiltinTypeAnnotation:
        cpp_deduce.deduce_BuiltinTypeAnnotation,
    ast.TupleTypeAnnotation:
        cpp_deduce.deduce_TupleTypeAnnotation,
    ast.TypeRefTypeAnnotation:
        cpp_deduce.deduce_TypeRefTypeAnnotation,
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
