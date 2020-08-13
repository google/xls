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

import copy
import functools
from typing import Dict, Optional, Text, Tuple, Union, Callable, List

from absl import logging
import dataclasses

from xls.common.xls_error import XlsError
from xls.dslx import ast
from xls.dslx import deduce
from xls.dslx import dslx_builtins
from xls.dslx.ast import Function
from xls.dslx.ast import Module
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.concrete_type import FunctionType
from xls.dslx.interpreter.interpreter_helpers import interpret_expr
from xls.dslx.span import PositionalError
from xls.dslx.xls_type_error import XlsTypeError


def _check_function_params(f: Function,
                           ctx: deduce.DeduceCtx) -> List[ConcreteType]:
  """Checks the function's parametrics' and arguments' types."""
  for parametric in f.parametric_bindings:
    parametric_binding_type = deduce.deduce(parametric.type_, ctx)
    assert isinstance(parametric_binding_type, ConcreteType)
    if parametric.expr:
      # TODO(hans): 2020-07-06 Either throw a more descriptive error if
      # there's a parametric fn in this expr or add support for it
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

  return param_types


def _check_function(f: Function, ctx: deduce.DeduceCtx) -> None:
  """Validates type annotations on parameters/return type of f are consistent.

  Args:
    f: The function to type check.
    ctx: Wraps a node_to_type, a mapping of AST node to its deduced type;
      (free-variable) references are resolved via this dictionary.

  Raises:
    XlsTypeError: When the return type deduced is inconsistent with the return
      type annotation on "f".
  """
  fn_name, _ = ctx.fn_stack[-1]
  # First, get the types of the function's parametrics, args, and return type
  if f.is_parametric() and f.name.identifier == fn_name:
    # Parametric functions are evaluated per invocation. If we're currently
    # inside of this function, it must mean that we already have the type
    # signature and now we just need to evaluate the body.
    assert f in ctx.node_to_type, f
    annotated_return_type = ctx.node_to_type[f].return_type  # pytype: disable=attribute-error
    param_types = list(ctx.node_to_type[f].params)  # pytype: disable=attribute-error
  else:
    logging.vlog(1, 'Type-checking sig for function: %s', f)
    param_types = _check_function_params(f, ctx)
    if f.is_parametric():
      # We just needed the type signature so that we can instantiate this
      # invocation. Let's return this for now and typecheck the body once we
      # have symbolic bindings.
      annotated_return_type = (
          deduce.deduce(f.return_type, ctx)
          if f.return_type else ConcreteType.NIL)
      ctx.node_to_type[f.name] = ctx.node_to_type[f] = FunctionType(
          tuple(param_types), annotated_return_type)
      return

  logging.vlog(1, 'Type-checking body for function: %s', f)

  # Second, typecheck the body of the function
  body_return_type = deduce.deduce(f.body, ctx)
  resolved_body_type = deduce.resolve(body_return_type, ctx)

  # Third, assert that the annotated return type matches the body return type
  # (after resolving any parametric symbols)
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
    resolved_return_type = deduce.resolve(annotated_return_type, ctx)

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
  body_return_type = deduce.deduce(t.body, ctx)
  nil = ConcreteType.NIL
  if body_return_type != nil:
    raise XlsTypeError(
        t.body.span,
        body_return_type,
        nil,
        suffix='Return type of test body for "{}" did not match the '
        'expected test return type (nil).'.format(t.name.identifier))


def _instantiate(builtin_name: ast.BuiltinNameDef, invocation: ast.Invocation,
                 ctx: deduce.DeduceCtx) -> Optional[ast.NameDef]:
  """Instantiates a builtin parametric invocation; e.g. 'update'."""
  arg_types = tuple(
      deduce.resolve(ctx.node_to_type[arg], ctx) for arg in invocation.args)

  higher_order_parametric_bindings = None
  map_fn_name = None
  if builtin_name.identifier == 'map':
    map_fn_ref = invocation.args[1]
    if isinstance(map_fn_ref, ast.ModRef):
      imported_module, imported_node_to_type = ctx.node_to_type.get_imported(
          map_fn_ref.mod)
      map_fn_name = map_fn_ref.value_tok.value
      map_fn = imported_module.get_function(map_fn_name)
      higher_order_parametric_bindings = map_fn.parametric_bindings
    else:
      assert isinstance(map_fn_ref, ast.NameRef), map_fn_ref
      map_fn_name = map_fn_ref.tok.value
      if map_fn_ref.identifier not in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
        map_fn = ctx.module.get_function(map_fn_name)
        higher_order_parametric_bindings = map_fn.parametric_bindings

  fsignature = dslx_builtins.get_fsignature(builtin_name.identifier)
  fn_type, symbolic_bindings = fsignature(arg_types, builtin_name.identifier,
                                          invocation.span, ctx,
                                          higher_order_parametric_bindings)

  fn_name, fn_symbolic_bindings = ctx.fn_stack[-1]
  invocation.symbolic_bindings[(
      ctx.module.name, fn_name,
      tuple(fn_symbolic_bindings.items()))] = symbolic_bindings
  ctx.node_to_type[invocation.callee] = fn_type
  ctx.node_to_type[invocation] = fn_type.return_type  # pytype: disable=attribute-error

  if builtin_name.identifier == 'map':
    assert isinstance(map_fn_name, str), map_fn_name
    if (map_fn_name in dslx_builtins.PARAMETRIC_BUILTIN_NAMES or
        not map_fn.is_parametric()):
      # A builtin higher-order parametric fn would've been typechecked when we
      # were going through the arguments of this invocation.
      # If the function wasn't parametric, then we're good to go.
      # invocation
      return None

    # If the higher order function is parametric, we need to typecheck its body
    # with the symbolic bindings we just computed.
    if isinstance(map_fn_ref, ast.ModRef):
      invocation_imported_node_to_type = deduce.NodeToType(parent=imported_node_to_type)
      imported_ctx = deduce.DeduceCtx(invocation_imported_node_to_type, imported_module,
                                      ctx.interpret_expr,
                                      ctx.check_function_in_module)
      imported_ctx.fn_stack.append((map_fn_name, dict(symbolic_bindings)))
      # We need to typecheck this imported function with respect to its module
      ctx.check_function_in_module(map_fn, imported_ctx)

      invocation_node_to_type = deduce.NodeToType(parent=ctx.node_to_type)
      invocation_node_to_type.update(imported_ctx.node_to_type)
      invocation.bindings_to_ntt[symbolic_bindings] = invocation_node_to_type
    else:
      # If the higher-order parametric fn is in this module, let's try to push
      # it onto the typechecking stack
      ctx.fn_stack.append((map_fn_name, dict(symbolic_bindings)))
      invocation_node_to_type = deduce.NodeToType(parent=ctx.node_to_type)
      invocation.bindings_to_ntt[symbolic_bindings] = invocation_node_to_type
      ctx.node_to_type = invocation_node_to_type
      return map_fn_ref.name_def

  return None


@dataclasses.dataclass
class _TypecheckStackRecord:
  """A wrapper over information used to typecheck a test/function.

  Attributes:
    name: The name of this test/function.
    is_test: Flag indicating whether or not this record is for a test.
    user: The node in this module that needs 'name' to be typechecked. Used to
      detect the typechecking of the higher order function in map invocations.
  """
  name: Text
  is_test: bool
  user: Optional[ast.AstNode] = None

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

  stack = [_TypecheckStackRecord(f.name.identifier, isinstance(f, ast.Test))]  # type: List[_TypecheckStackRecord]

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  while stack:
    parent_count = 0
    p = ctx.node_to_type._parent
    while p:
      parent_count += 1
      p = p._parent
    try:
      f = seen[(stack[-1].name, stack[-1].is_test)][0]
      if isinstance(f, ast.Function):
        _check_function(f, ctx)
      else:
        assert isinstance(f, ast.Test)
        check_test(f, ctx)
      seen[(f.name.identifier, isinstance(f, ast.Test))] = (f, False
                                                           )  # Mark as done.
      stack_record = stack.pop()
      fn_name, _ = ctx.fn_stack[-1]

      def is_callee_map(n: Optional[ast.AstNode]) -> bool:
        return (n and isinstance(n, ast.Invocation) and
                isinstance(n.callee, ast.NameRef) and
                n.callee.tok.value == 'map')

      if is_callee_map(stack_record.user):
        # We need to remove the body of this higher order parametric function
        # in case we need to recheck it later in this module. See
        # deduce._check_parametric_invocation() for more information.
        assert isinstance(f, ast.Function) and f.is_parametric()
        # ctx.node_to_type._dict.pop(f.body)  # pylint: disable=protected-access
        ctx.node_to_type = ctx.node_to_type._parent

      if stack_record.name == fn_name:
        # i.e. we just finished typechecking the body of the function we're
        # currently inside of.
        ctx.fn_stack.pop()

    except deduce.TypeMissingError as e:
      while True:
        fn_name, _ = ctx.fn_stack[-1]
        # print("got ", e.node)
        if (isinstance(e.node, ast.NameDef) and
            e.node.identifier in function_map):
          # If it's seen and not-done, we're recursing.
          if seen.get((e.node.identifier, False), (None, False))[1]:
            raise XlsError(
                'Recursion detected while typechecking; name: {}'.format(
                    e.node.identifier))
          callee = function_map[e.node.identifier]
          assert isinstance(callee, ast.Function), callee
          seen[(e.node.identifier, False)] = (callee, True)
          # stack.append(_make_record(callee, ctx, user=e.user))
          stack.append(
              _TypecheckStackRecord(callee.name.identifier,
                                    isinstance(callee, ast.Test),
                                    e.user))
          break
        if (isinstance(e.node, ast.BuiltinNameDef) and
            e.node.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES):
          logging.vlog(2, 'node: %r; identifier: %r, exception user: %r',
                       e.node, e.node.identifier, e.user)

          if isinstance(e.user, ast.Invocation):
            func = _instantiate(e.node, e.user, ctx)
            if func:
              # We need to figure out what to do with this higher order
              # parametric function.
              e.node = func
              continue
            break

        # Raise if this wasn't a function in this module or a builtin.
        raise


ImportFn = Callable[[Tuple[Text, ...]], Tuple[ast.Module, deduce.NodeToType]]


def check_module(module: Module,
                 f_import: Optional[ImportFn]) -> deduce.NodeToType:
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
  ctx.fn_stack.append(('top', dict()))  # No sym bindings in the global scope.
  for member in ctx.module.top:
    if isinstance(member, ast.Import):
      imported_module, imported_node_to_type = f_import(member.name)
      ctx.node_to_type.add_import(member,
                                  (imported_module, imported_node_to_type))
    elif isinstance(member, (ast.Constant, ast.Enum, ast.Struct, ast.TypeDef)):
      deduce.deduce(member, ctx)
    else:
      assert isinstance(member,
                        (ast.Function, ast.Test, ast.QuickCheck)), member
  ctx.fn_stack.pop()

  quickcheck_map = {
      qc.f.name.identifier: qc for qc in ctx.module.get_quickchecks()
  }
  for qc in quickcheck_map.values():
    assert isinstance(qc, ast.QuickCheck), qc

    f = qc.f
    assert isinstance(f, ast.Function), f
    if f.is_parametric():
      # TODO(cdleary): 2020-08-09 See https://github.com/google/xls/issues/81
      raise PositionalError(
          'Quickchecking parametric '
          'functions is unsupported.', f.span)

    logging.vlog(2, 'Typechecking function: %s', f)
    ctx.fn_stack.append((f.name.identifier, dict()))  # No symbolic bindings.
    check_function_or_test_in_module(f, ctx)

    quickcheck_f_body_type = ctx.node_to_type[f.body]
    if quickcheck_f_body_type != ConcreteType.U1:
      raise XlsTypeError(
          f.span,
          quickcheck_f_body_type,
          ConcreteType.U1,
          suffix='QuickCheck functions must return a bool.')

    logging.vlog(2, 'Finished typechecking function: %s', f)

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  for f in function_map.values():
    assert isinstance(f, ast.Function), f
    if f.is_parametric():
      # Let's typecheck parametric functions per invocation.
      continue

    logging.vlog(2, 'Typechecking function: %s', f)
    ctx.fn_stack.append((f.name.identifier, dict()))  # No symbolic bindings.
    check_function_or_test_in_module(f, ctx)
    logging.vlog(2, 'Finished typechecking function: %s', f)

  test_map = {t.name.identifier: t for t in ctx.module.get_tests()}
  for t in test_map.values():
    assert isinstance(t, ast.Test), t

    if isinstance(t, ast.TestFunction):
      # New-style test constructs are specified using a function.
      # This function shouldn't be parametric and shouldn't take any arguments.
      if t.fn.params:
        raise PositionalError("Test functions shouldn't take arguments.",
                              t.fn.span)

      if t.fn.is_parametric():
        raise PositionalError("Test functions shouldn't be parametric.",
                              t.fn.span)

    # No symbolic bindings inside of a test.
    ctx.fn_stack.append(('{}_test'.format(t.name.identifier), dict()))
    logging.vlog(2, 'Typechecking test: %s', t)
    if isinstance(t, ast.TestFunction):
      # New-style tests are wrapped in a function.
      check_function_or_test_in_module(t.fn, ctx)
    else:
      # Old-style tests are specified in a construct with a body
      # (see check_test()).
      check_function_or_test_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking test: %s', t)

  return ctx.node_to_type

