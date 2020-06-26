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
import copy

from absl import logging

from xls.common.xls_error import XlsError
from xls.dslx import ast
from xls.dslx import deduce
from xls.dslx import dslx_builtins
from xls.dslx.ast import Function
from xls.dslx.ast import Module
from xls.dslx.interpreter import interpreter_helpers
from xls.dslx.concrete_type import ConcreteType, TupleType
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
  if f.parametric_bindings and ctx.fn_name == f.name.identifier:
    annotated_return_type = ctx.node_to_type[f].return_type
    param_types = list(ctx.node_to_type[f].params)
  else:
    logging.vlog(1, 'Type-checking sig for function: %s', f)

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

    if f.parametric_bindings:
      annotated_return_type = deduce.deduce(f.return_type, ctx) if f.return_type else TupleType(None)
      ctx.node_to_type[f.name] = ctx.node_to_type[f] = FunctionType(
           tuple(param_types), annotated_return_type)
      return

  logging.vlog(1, 'Type-checking body for function: %s', f)

  if f.parametric_bindings:
    if f in ctx.parametric_fn_cache:
      cached_types = ctx.parametric_fn_cache[f]
      without_f_dict = { node : ctx.node_to_type[node] for node in ctx.node_to_type._dict if not node in cached_types }
      ctx.node_to_type._dict = without_f_dict
      body_return_type = deduce.deduce(f.body, ctx)
    else:
      og_dict = copy.copy(ctx.node_to_type._dict)
      body_return_type = deduce.deduce(f.body, ctx)
      diff = { node : ctx.node_to_type[node] for node in set(ctx.node_to_type._dict) - set(og_dict) }
      ctx.parametric_fn_cache[f] = diff
  else:
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

    resolver = deduce.mk_resolver(ctx.fn_symbolic_bindings) # No symbolic bindings
    resolved_return_type =  annotated_return_type.map_size(resolver)
    resolved_body_type = body_return_type.map_size(resolver)

    if resolved_return_type != resolved_body_type:
      raise XlsTypeError(
          f.body.span,
          resolved_body_type,
          resolved_return_type,
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
  resolver = deduce.mk_resolver(ctx.fn_symbolic_bindings)
  arg_types = tuple(ctx.node_to_type[arg].map_size(resolver) for arg in invocation.args)

  if builtin_name.identifier not in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
    return False

  fsignature = dslx_builtins.get_fsignature(builtin_name.identifier)
  fn_type, symbolic_bindings = fsignature(arg_types, builtin_name.identifier,
                                          invocation.span)
  invocation.symbolic_bindings = symbolic_bindings
  ctx.node_to_type[invocation.callee] = fn_type
  ctx.node_to_type[invocation] = fn_type.return_type


  if builtin_name.identifier == "map":
    map_fn = invocation.args[1]
    if isinstance(map_fn, ast.ModRef):
      imported_module, imported_node_to_type  = ctx.node_to_type.get_imported(map_fn.mod)
      ident = map_fn.value_tok.value
      function_def = imported_module.get_function(ident)

      importedCtx = deduce.DeduceCtx(imported_node_to_type, imported_module, ctx.interp_callback,
                             ctx.typecheck_callback, dict(symbolic_bindings), f_import=ctx.f_import)
      body_return_type = deduce.deduce(function_def.body, importedCtx)
      ctx.node_to_type.update(importedCtx.node_to_type)
    else:
      if (map_fn.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES):
        return True

      ident = map_fn.tok.value
      function_def = ctx.module.get_function(ident)

      old_sb = ctx.fn_symbolic_bindings
      ctx.fn_symbolic_bindings = dict(symbolic_bindings)
      body_return_type = deduce.deduce(function_def.body, ctx)
      ctx.fn_symbolic_bindings = old_sb

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
    print("#####", ctx.fn_name,"###########", stack, "#######################")
    try:
      f = seen[stack[-1]][0]
      if isinstance(f, ast.Function):
        """
        if not f.parametric_bindings:
          ret_type = _check_function_sig(f, ctx)
          _check_function_body(f, ctx, fn_type)
          assert isinstance(f.name, ast.NameDef) and f.name in ctx.node_to_type
        else:
          _check_function_sig(f, ctx)
        """
        _check_function(f, ctx)
        """
        if f.parametric_bindings:
          if f.name.identifier != ctx.fn_name:
            seen[(f.name.identifier, isinstance(f, ast.Test))] = (f, False
                                                           )  # Mark as done.
            stack.pop()
          continue
        """
      else:
        assert isinstance(f, ast.Test)
        check_test(f, ctx)
      seen[(f.name.identifier, isinstance(f, ast.Test))] = (f, False
                                                           )  # Mark as done.
      stack.pop()

      if len(ctx.sym_stack):
        old = ctx.sym_stack.pop()
        ctx.fn_symbolic_bindings = old[1]
        ctx.fn_name = old[0]

    except deduce.TypeMissingError as e:
      #print("##### picked up {}, currently in {}  #######".format(e.node, ctx.fn_name))
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
  ctx = deduce.DeduceCtx(node_to_type, module, interp_callback,
                         _check_function_or_test_in_module,
                         f_import=f_import)

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
    if f.parametric_bindings:
      # Let's typecheck parametric functions per instantiation
      continue

    logging.vlog(2, 'Typechecking function: %s', f)
    ctx.fn_name = f.name.identifier
    _check_function_or_test_in_module(f, ctx)
    logging.vlog(2, 'Finished typechecking function: %s', f)

  test_map = {t.name.identifier: t for t in ctx.module.get_tests()}
  ctx.fn_name = "test_"
  for t in test_map.values():
    assert isinstance(t, ast.Test), t
    logging.vlog(2, 'Typechecking test: %s', t)
    _check_function_or_test_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking test: %s', t)

  return ctx.node_to_type
