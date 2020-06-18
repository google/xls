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

"""Implementation of type checking functionality on a parsed AST object."""

from typing import Dict, Optional, Text, Tuple, Union, Callable

from absl import logging

from xls.common.xls_error import XlsError
from xls.dslx import ast
from xls.dslx import deduce
from xls.dslx import dslx_builtins
from xls.dslx.ast import Function
from xls.dslx.ast import Module
from xls.dslx.interpreter import interpreter_helpers
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.concrete_type import FunctionType
from xls.dslx.xls_type_error import XlsTypeError


def _check_function(f: Function, ctx: deduce.DeduceCtx):
  """Validates type annotations on parameters/return type of f are consistent.

  Args:
    f: The function to type check.
    ctx: Wraps a node_to_type, a mapping of AST node to its deduced type;
      (free-variable) references are resolved via this dictionary.

  Raises:
    XlsTypeError: When the return type deduced is inconsistent with the return
      type annotation on "f".
  """
  logging.vlog(1, 'Type-checking function: %s', f)

  for parametric in f.parametric_bindings:
    parametric_binding_type = deduce.deduce(parametric.type_, ctx)
    assert isinstance(parametric_binding_type, ConcreteType)
    if parametric.expr:
      expr_type = deduce.deduce(parametric.expr, ctx)
      if expr_type != parametric_binding_type:
        raise XlsTypeError(
            parametric.span,
            parametric_binding_type,
            expr_type,
            suffix='Annotated type of derived parametric '
            'value did not match inferred type.')
    ctx.node_to_type[parametric.name] = parametric_binding_type

  param_types = []
  for param in f.params:
    logging.vlog(2, 'Checking param: %s', param)
    param_type = deduce.deduce(param, ctx)
    assert isinstance(param_type, ConcreteType), param_type
    param_types.append(param_type)
    ctx.node_to_type[param.name] = param_type

  body_return_type = deduce.deduce(f.body, ctx)
  if f.return_type is None:
    if body_return_type.is_nil():
      # When body return type is nil and no return type is annotated, everything
      # is ok.
      pass
    else:
      # Otherwise there's a mismatch.
      raise XlsTypeError(
          f.span,
          None,
          body_return_type,
          suffix='No return type was annotated, but a non-nil return type was '
          'found.')
  else:
    annotated_return_type = deduce.deduce(f.return_type, ctx)
    if body_return_type != annotated_return_type:
      raise XlsTypeError(
          f.body.span,
          body_return_type,
          annotated_return_type,
          suffix='Return type of function body for "{}" did not match the '
          'annotated return type.'.format(f.name.identifier))

  ctx.node_to_type[f.name] = ctx.node_to_type[f] = FunctionType(
      tuple(param_types), body_return_type)


def check_test(t: ast.Test, ctx: deduce.DeduceCtx) -> None:
  """Typechecks a test (body) within a module."""
  while True:
    try:
      body_return_type = deduce.deduce(t.body, ctx)
    except deduce.TypeMissingError as e:
      if (isinstance(e.node, ast.BuiltinNameDef) and
          e.node.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES):
        if isinstance(e.user, ast.Invocation) and _instantiate(
            e.node, e.user, ctx):
          continue
      raise
    else:
      nil = ConcreteType.NIL
      if body_return_type != nil:
        raise XlsTypeError(
            t.body.span,
            body_return_type,
            nil,
            suffix='Return type of test body for "{}" did not match the '
            'expected test return type (nil).'.format(t.name.identifier))
      return  # Ok!


def _instantiate(builtin_name: ast.BuiltinNameDef, invocation: ast.Invocation,
                 ctx: deduce.DeduceCtx) -> bool:
  """Instantiates a builtin parametric invocation; e.g. 'update'."""
  arg_types = tuple(ctx.node_to_type[arg] for arg in invocation.args)
  if builtin_name.identifier not in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
    return False

  #print(ctx.node_to_type.module.get_function(builtin_name.identifier))
  fsignature = dslx_builtins.get_fsignature(builtin_name.identifier)
  fn_type, symbolic_bindings = fsignature(arg_types, builtin_name.identifier,
                                          invocation.span)
  invocation.symbolic_bindings = symbolic_bindings
  ctx.node_to_type[invocation.callee] = fn_type
  ctx.node_to_type[invocation] = fn_type.return_type
  return True


def _check_function_or_test_in_module(f: Union[Function, ast.Test],
                                      ctx: deduce.DeduceCtx):
  """Type-checks function f in the given module.

  Args:
    f: Function to type-check.
    ctx: Wraps a node_to_type, a mapping being populated with the
      inferred type for AST nodes. Also contains a module.

  Raises:
    TypeMissingError: When we attempt to resolve an AST node to a type that a)
      cannot be resolved via the node_to_type mapping and b) the AST node
      missing a type does not refer to a top-level function in the module
      (determined via function_map).
    XlsTypeError: When there is a type check failure.
  """
  # {name: (function, wip)}
  seen = {
      (f.name.identifier, isinstance(f, ast.Test)): (f, True)
  }  # type: Dict[Tuple[Text, bool], Tuple[Union[Function, ast.Test], bool]]
  stack = [(f.name.identifier, isinstance(f, ast.Test))]

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  while stack:
    try:
      f = seen[stack[-1]][0]
      if isinstance(f, ast.Function):
        _check_function(f, ctx)
        assert isinstance(f.name, ast.NameDef) and f.name in ctx.node_to_type
      else:
        assert isinstance(f, ast.Test)
        check_test(f, ctx)
      seen[(f.name.identifier, isinstance(f, ast.Test))] = (f, False
                                                           )  # Mark as done.
      stack.pop()
    except deduce.TypeMissingError as e:
      if isinstance(e.node, ast.NameDef) and e.node.identifier in function_map:
        # If it's seen and not-done, we're recursing.
        if seen.get((e.node.identifier, False), (None, False))[1]:
          raise XlsError(
              'Recursion detected while typechecking; name: {}'.format(
                  e.node.identifier))
        callee = function_map[e.node.identifier]
        assert isinstance(callee, ast.Function), callee
        seen[(e.node.identifier, False)] = (callee, True)
        stack.append((e.node.identifier, False))
        continue
      if (isinstance(e.node, ast.BuiltinNameDef) and
          e.node.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES):
        logging.vlog(2, 'node: %r; identifier: %r, exception user: %r', e.node,
                     e.node.identifier, e.user)
        if isinstance(e.user, ast.Invocation) and _instantiate(
            e.node, e.user, ctx):
          continue
      raise


ImportFn = Callable[[Tuple[Text, ...]], Tuple[ast.Module, deduce.NodeToType]]


def check_module(
    module: Module,
    f_import: Optional[ImportFn],
) -> deduce.NodeToType:
  """Validates type annotations on all functions within "module".

  Args:
    module: The module to type check functions for.
    f_import: Callback to import a module (a la a import statement). This may be
      None e.g. in unit testing situations where it's guaranteed there will be
      no import statements.

  Returns:
    Mapping from AST node to its deduced/checked type.

  Raises:
    XlsTypeError: If any of the function in f have typecheck errors.
  """
  node_to_type = deduce.NodeToType()
  interp_callback = interpreter_helpers.interpret_expr
  ctx = deduce.DeduceCtx(node_to_type, module, interp_callback)

  # First populate node_to_type with constants, enums, and resolved imports.
  for member in ctx.module.top:
    if isinstance(member, ast.Import):
      imported_module, imported_node_to_type = f_import(member.name)
      ctx.node_to_type.add_import(member, (imported_module, imported_node_to_type))
    elif isinstance(member, (ast.Constant, ast.Enum, ast.Struct, ast.TypeDef)):
      deduce.deduce(member, ctx)
    else:
      assert isinstance(member, (ast.Function, ast.Test)), member

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  for f in function_map.values():
    assert isinstance(f, ast.Function), f
    logging.vlog(2, 'Typechecking function: %s', f)
    _check_function_or_test_in_module(f, ctx)
    logging.vlog(2, 'Finished typechecking function: %s', f)

  test_map = {t.name.identifier: t for t in ctx.module.get_tests()}
  for t in test_map.values():
    assert isinstance(t, ast.Test), t
    logging.vlog(2, 'Typechecking test: %s', t)
    _check_function_or_test_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking test: %s', t)

  return ctx.node_to_type
