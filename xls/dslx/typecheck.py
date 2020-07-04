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
import functools

from absl import logging

from xls.common.xls_error import XlsError
from xls.dslx import ast
from xls.dslx import deduce
from xls.dslx import dslx_builtins
from xls.dslx.ast import Function
from xls.dslx.ast import Module
from xls.dslx.interpreter.interpreter_helpers import interpret_expr
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
  fn_name, fn_symbolic_bindings = ctx.fn_stack[-1]
  if f.is_parametric() and f.name.identifier == fn_name:
    # Parametric functions are evaluated per invocation. If we're currently
    # inside of this function, it must mean that we already have the type
    # signature and now we just need to evaluate the body.
    assert f in ctx.node_to_type
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

    if f.is_parametric():
      # We just needed the type signature so that we can instantiate this
      # invocation. Let's return this for now and typecheck the body once we
      # have symbolic bindings.
      annotated_return_type = deduce.deduce(f.return_type, ctx) \
                              if f.return_type else ConcreteType.NIL
      ctx.node_to_type[f.name] = ctx.node_to_type[f] = FunctionType(
           tuple(param_types), annotated_return_type)
      return

  logging.vlog(1, 'Type-checking body for function: %s', f)

  body_return_type = deduce.deduce(f.body, ctx)
  resolved_body_type = deduce.resolve(body_return_type, ctx)

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
          resolved_body_type,
          suffix='No return type was annotated, but a non-nil return type was '
          'found.')
  else:
    annotated_return_type = deduce.deduce(f.return_type, ctx)
    resolved_return_type =  deduce.resolve(annotated_return_type, ctx)

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
                 ctx: deduce.DeduceCtx) -> Tuple[bool, Optional[ast.NameDef]]:
  """Instantiates a builtin parametric invocation; e.g. 'update'."""
  arg_types = tuple(deduce.resolve(
                        ctx.node_to_type[arg], ctx) for arg in invocation.args)

  if builtin_name.identifier not in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
    return (False, None)

  fsignature = dslx_builtins.get_fsignature(builtin_name.identifier)
  fn_type, symbolic_bindings = fsignature(arg_types, builtin_name.identifier,
                                          invocation.span)

  fn_name, fn_symbolic_bindings = ctx.fn_stack[-1]
  invocation.symbolic_bindings[(fn_name,
                      tuple(fn_symbolic_bindings.items()))] = symbolic_bindings
  ctx.node_to_type[invocation.callee] = fn_type
  ctx.node_to_type[invocation] = fn_type.return_type


  if builtin_name.identifier == "map":
    map_fn_ref = invocation.args[1]
    # If the higher order function is parametric, we need to typecheck its body
    # with the symbolic bindings we just computed.
    if isinstance(map_fn_ref, ast.ModRef):
      imported_module, imported_node_to_type = \
                                  ctx.node_to_type.get_imported(map_fn_ref.mod)
      map_fn_name = map_fn_ref.value_tok.value
      map_fn = imported_module.get_function(map_fn_name)
      if not map_fn.is_parametric():
        return (True, None)

      importedCtx = deduce.DeduceCtx(imported_node_to_type, imported_module,
                                     ctx.interp_callback,
                                    ctx.typecheck_callback,
                                    parametric_fn_cache=ctx.parametric_fn_cache)
      importedCtx.fn_stack.append((map_fn_name, dict(symbolic_bindings)))
      # We need to typecheck this imported function with respect to its module
      ctx.typecheck_callback(map_fn, importedCtx)
      ctx.node_to_type.update(importedCtx.node_to_type)
    else:
      if (map_fn_ref.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES):
        # Already typechecked when we were typechecking the arguments of this
        # invocation
        return (True, None)

      map_fn_name = map_fn_ref.tok.value
      map_fn = ctx.module.get_function(map_fn_name)
      if not map_fn.is_parametric():
        return (True, None)

      ctx.fn_stack.append((map_fn_name, dict(symbolic_bindings)))
      return (True, map_fn_ref.name_def)

  return (True, None)

def _make_record(f: Union[Function, ast.Test], ctx: deduce.DeduceCtx) \
      -> Tuple[Text, bool, Optional[Dict[ast.AstNode, ConcreteType]]]:
  """Creates a tuple with information for typechecking functions/tests

  The third item in the tuple will optionally carry the contents of
  ctx.node_to_type._dict at the time of record creation. If this record is
  for typechecking the body of a parametric function for the first time,
  we'll use the before-version of ctx.node_to_type._dict to determine the
  dependencies of the parametric function when this record is popped.
  """
  fn_name, fn_symbolic_bindings = ctx.fn_stack[-1]
  if isinstance(f, ast.Function):
    if f.is_parametric() and fn_name == f.name.identifier:
      if (ctx.module.name, f) not in ctx.parametric_fn_cache:
        # This is our first time evaluating the body of this parametric fn.
        # Let's store what ctx.node_to_type looked like before so we know
        # what this function's dependencies are.
        before_ntt_dict = copy.copy(ctx.node_to_type._dict)
        rec = (f.name.identifier, False, before_ntt_dict)
      else:
        # We've previously evaluated the body of this parametric fn.
        # Let's remove the types we found so that they are reevaluated
        # with the current symbolic bindings.
        cached_types = ctx.parametric_fn_cache[(ctx.module.name, f)]
        without_deps_dict = { n: ctx.node_to_type[n] for n in \
                              ctx.node_to_type._dict if not n in cached_types } 
        # Assert that the body will be reevaluated
        assert not f.body in without_deps_dict
        ctx.node_to_type._dict = without_deps_dict
        rec = (f.name.identifier, False, None)
    else:
      rec = (f.name.identifier, False, None)
  else:
    rec = (f.name.identifier, True, None)

  return rec

def check_function_or_test_in_module(f: Union[Function, ast.Test],
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

  stack = [_make_record(f, ctx)]

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  while stack:
    # print("#####", ctx.fn_stack[-1][0],"###########", [r[:2] for r in stack], "#######################")
    try:
      f = seen[stack[-1][:2]][0]
      if isinstance(f, ast.Function):
        _check_function(f, ctx)
      else:
        assert isinstance(f, ast.Test)
        check_test(f, ctx)
      seen[(f.name.identifier, isinstance(f, ast.Test))] = (f, False
                                                           )  # Mark as done.
      rec = stack.pop()
      fn_name, _ = ctx.fn_stack[-1]
      if isinstance(f, ast.Function) and f.is_parametric() \
          and fn_name == f.name.identifier and \
                            (ctx.module.name, f) not in ctx.parametric_fn_cache:
        # We just evaluated the body of a parametric function for the first
        # time. Let's compute its dependencies so that we know which nodes to
        # recheck if we see another invocation of this parametric function.
        before_ntt_dict = rec[2]
        assert before_ntt_dict
        deps = { n : ctx.node_to_type[n] for n in \
                 set(ctx.node_to_type._dict) - set(before_ntt_dict) }
        ctx.parametric_fn_cache[(ctx.module.name, f)] = deps

      if rec[0] == fn_name:
        # ie. we just finished typechecking the body of the function we're
        # currently inside of
        ctx.fn_stack.pop()

    except deduce.TypeMissingError as e:
      # print("##### caught {}".format(e))
      while True:
        fn_name, fn_symbolic_bindings = ctx.fn_stack[-1]
        if isinstance(e.node, ast.NameDef) and \
                e.node.identifier in function_map:
          # If it's seen and not-done, we're recursing.
          if seen.get((e.node.identifier, False), (None, False))[1]:
            raise XlsError(
                'Recursion detected while typechecking; name: {}'.format(
                    e.node.identifier))
          callee = function_map[e.node.identifier]
          assert isinstance(callee, ast.Function), callee
          seen[(e.node.identifier, False)] = (callee, True)
          stack.append(_make_record(callee, ctx))
          break
        if (isinstance(e.node, ast.BuiltinNameDef) and
            e.node.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES):
          logging.vlog(2, 'node: %r; identifier: %r, exception user: %r',
                       e.node, e.node.identifier, e.user)

          if isinstance(e.user, ast.Invocation):
            ok_inst, func = _instantiate(e.node, e.user, ctx)
            if ok_inst:
              if func:
                # We need to figure out what to do with this higher order
                # parametric function
                e.node = func
                continue
              else:
                break

        raise



ImportFn = Callable[[Tuple[Text, ...]], Tuple[ast.Module, deduce.NodeToType]]


def check_module(
    module: Module,
    f_import: Optional[ImportFn], is_import: bool = False
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
  interpreter_callback = functools.partial(interpret_expr, f_import=f_import)
  ctx = deduce.DeduceCtx(node_to_type, module, interpreter_callback,
                         check_function_or_test_in_module)

  # First populate node_to_type with constants, enums, and resolved imports.
  ctx.fn_stack.append(('top', dict()))  # No sym bindings in the global scope
  for member in ctx.module.top:
    if isinstance(member, ast.Import):
      imported_module, imported_node_to_type = f_import(member.name)
      ctx.node_to_type.add_import(member,
                                  (imported_module, imported_node_to_type))
    elif isinstance(member, (ast.Constant, ast.Enum, ast.Struct, ast.TypeDef)):
      deduce.deduce(member, ctx)
    else:
      assert isinstance(member, (ast.Function, ast.Test)), member
  ctx.fn_stack.pop()

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  for f in function_map.values():
    assert isinstance(f, ast.Function), f
    if f.is_parametric():
      # Let's typecheck parametric functions per invocation
      continue

    logging.vlog(2, 'Typechecking function: %s', f)
    ctx.fn_stack.append((f.name.identifier, dict())) # No symbolic bindings
    check_function_or_test_in_module(f, ctx)
    logging.vlog(2, 'Finished typechecking function: %s', f)

  test_map = {t.name.identifier: t for t in ctx.module.get_tests()}
  for t in test_map.values():
    assert isinstance(t, ast.Test), t
    # No symbolic bindings inside of a test construct
    ctx.fn_stack.append(("{}_test".format(t.name.identifier), dict()))
    logging.vlog(2, 'Typechecking test: %s', t)
    check_function_or_test_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking test: %s', t)

  if is_import:
    # If we just typechecked an imported module, lets discard all
    # of the parametric fns' dependencies so that they are rechecked on
    # invocation in the main module
    for mod_name, f in ctx.parametric_fn_cache:
      if ctx.module.name == mod_name:
        deps = ctx.parametric_fn_cache[(ctx.module.name, f)]
        without_deps_dict = { n: ctx.node_to_type[n] for n in \
                              ctx.node_to_type._dict if not n in deps }
        ctx.node_to_type._dict = without_deps_dict
  else:
    # Add back the bodies of parametric fns for completeness, as they are
    # removed in deduce._deduce_Invocation()
    for mod_name, f in ctx.parametric_fn_cache:
      ctx.node_to_type[f.body] = ctx.parametric_fn_cache[(mod_name, f)][f.body]

  return ctx.node_to_type
