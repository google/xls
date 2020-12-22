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

"""Implementation of type checking functionality on a parsed AST object."""

import functools
from typing import Optional, Text, Tuple, Callable

from absl import logging

from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_deduce
from xls.dslx.python import cpp_type_info as type_info
from xls.dslx.python import cpp_typecheck
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_deduce import xls_type_error as XlsTypeError
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.span import PositionalError


ImportFn = Callable[[Tuple[Text, ...]], Tuple[ast.Module, type_info.TypeInfo]]


def check_module(module: ast.Module,
                 f_import: Optional[ImportFn]) -> type_info.TypeInfo:
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
  assert f_import is None or callable(f_import), f_import
  ti = type_info.TypeInfo(module)
  import_cache = None if f_import is None else getattr(f_import, 'cache')
  additional_search_paths = () if f_import is None else getattr(
      f_import, 'additional_search_paths')
  ftypecheck = functools.partial(check_module, f_import=f_import)
  ctx = cpp_deduce.DeduceCtx(ti, module, cpp_deduce.deduce,
                             cpp_typecheck.check_top_node_in_module, ftypecheck,
                             additional_search_paths, import_cache)

  # First populate type_info with constants, enums, and resolved imports.
  ctx.add_fn_stack_entry(
      'top', SymbolicBindings())  # No sym bindings in the global scope.
  for member in ctx.module.top:
    if isinstance(member, ast.Import):
      assert isinstance(member.name, tuple), member.name
      imported_module, imported_type_info = f_import(member.name)
      ctx.type_info.add_import(member, (imported_module, imported_type_info))
    elif isinstance(member, (ast.Constant, ast.EnumDef)):
      cpp_deduce.deduce(member, ctx)
    else:
      assert isinstance(member,
                        (ast.Function, ast.Test, ast.StructDef, ast.QuickCheck,
                         ast.TypeDef)), (type(member), member)
  ctx.pop_fn_stack_entry()

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
    ctx.add_fn_stack_entry(f.name.identifier,
                           SymbolicBindings())  # No symbolic bindings.
    cpp_typecheck.check_top_node_in_module(f, ctx)

    quickcheck_f_body_type = ctx.type_info[f.body]
    if quickcheck_f_body_type != ConcreteType.U1:
      raise XlsTypeError(
          f.span,
          quickcheck_f_body_type,
          ConcreteType.U1,
          suffix='QuickCheck functions must return a bool.')

    logging.vlog(2, 'Finished typechecking function: %s', f)

  # We typecheck struct definitions using check_top_node_in_module() so that
  # we can typecheck function calls in parametric bindings, if any.
  struct_map = {s.name.identifier: s for s in ctx.module.get_structs()}
  for s in struct_map.values():
    assert isinstance(s, ast.StructDef), s
    logging.vlog(2, 'Typechecking struct %s', s)
    ctx.add_fn_stack_entry('top', SymbolicBindings())  # No symbolic bindings.
    cpp_typecheck.check_top_node_in_module(s, ctx)
    logging.vlog(2, 'Finished typechecking struct: %s', s)

  typedef_map = {
      t.name.identifier: t
      for t in ctx.module.top
      if isinstance(t, ast.TypeDef)
  }
  for t in typedef_map.values():
    assert isinstance(t, ast.TypeDef), t
    logging.vlog(2, 'Typechecking typedef %s', t)
    ctx.add_fn_stack_entry('top', SymbolicBindings())  # No symbolic bindings.
    cpp_typecheck.check_top_node_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking typedef: %s', t)

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  for f in function_map.values():
    assert isinstance(f, ast.Function), f
    if f.is_parametric():
      # Let's typecheck parametric functions per invocation.
      continue

    logging.vlog(2, 'Typechecking function: %s', f)
    ctx.add_fn_stack_entry(f.name.identifier,
                           SymbolicBindings())  # No symbolic bindings.
    cpp_typecheck.check_top_node_in_module(f, ctx)
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
    ctx.add_fn_stack_entry('{}_test'.format(t.name.identifier),
                           SymbolicBindings())
    logging.vlog(2, 'Typechecking test: %s', t)
    if isinstance(t, ast.TestFunction):
      # New-style tests are wrapped in a function.
      cpp_typecheck.check_top_node_in_module(t.fn, ctx)
    else:
      # Old-style tests are specified in a construct with a body
      # (see cpp_typecheck.check_test()).
      cpp_typecheck.check_top_node_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking test: %s', t)

  return ctx.type_info
