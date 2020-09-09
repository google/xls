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

"""Determines the order in which functions should be converted to IR."""

from typing import Tuple, Text, List, Dict, Union

from absl import logging
import dataclasses

from xls.dslx import dslx_builtins
from xls.dslx import type_info as type_info_mod
from xls.dslx.parametric_instantiator import SymbolicBindings
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_ast_visitor


@dataclasses.dataclass
class Callee:
  f: ast.Function
  m: ast.Module
  type_info: type_info_mod.TypeInfo
  sym_bindings: SymbolicBindings


ImportedInfo = Tuple[ast.Module, type_info_mod.TypeInfo]


class ConversionRecord(object):
  """Record used in sequence, noting order functions should be converted in.

  Describes a function instance that should be emitted (in an order determined
  by an encapsulating sequence). Annotated with metadata that describes the call
  graph instance.

  Attributes:
    f: Function AST node to convert.
    m: Module that f resides in.
    type_info: Node to type mapping for use in converting this
      function instance.
    callees: Function names that 'f' calls.
    bindings: Parametric bindings for this function instance.
    callees: Functions that this instance calls.
  """

  def __init__(self,
               f: ast.Function,
               m: ast.Module,
               type_info: type_info_mod.TypeInfo,
               bindings: SymbolicBindings,
               callees: Tuple[Callee, ...] = ()):
    self.f = f
    self.m = m
    self.type_info = type_info
    self.bindings = bindings
    self.callees = callees

  def __repr__(self) -> Text:
    return 'ConversionRecord(f={!r}, m={!r}, bindings={!r}, callees={!r})'.format(
        self.f.identifier, self.m, self.bindings, self.callees)


def get_callees(func: Union[ast.Function, ast.Test], m: ast.Module,
                type_info: type_info_mod.TypeInfo, imports: Dict[ast.Import,
                                                                 ImportedInfo],
                bindings: SymbolicBindings) -> Tuple[Callee, ...]:
  """Traverses the definition of f to find callees.

  Args:
    func: Function/test construct to inspect for calls.
    m: Module that f resides in.
    type_info: Node to type mapping that should be used with f.
    imports: Mapping of modules imported by m.
    bindings: Bindings used in instantiation of f.

  Returns:
    Callee functions invoked by f, and the parametric bindings used in each of
    those invocations.
  """
  callees = []

  class InvocationVisitor(cpp_ast_visitor.AstVisitor):
    """Visits invocation nodes to build up the callees list."""

    @cpp_ast_visitor.AstVisitor.no_auto_traverse
    def visit_ParametricBinding(self, node: ast.ParametricBinding) -> None:
      pass

    def visit_Invocation(self, node: ast.Invocation) -> None:
      if isinstance(node.callee, ast.ModRef):
        this_m, _ = imports[node.callee.mod]
        f = this_m.get_function(node.callee.value)
        fn_identifier = f.name.identifier
      elif isinstance(node.callee, ast.NameRef):
        this_m = m
        fn_identifier = node.callee.identifier
        if fn_identifier == 'map':
          # We need to make sure we convert the mapped function!
          fn_node = node.args[1]
          if isinstance(fn_node, ast.ModRef):
            fn_identifier = fn_node.value
            this_m = imports[fn_node.mod][0]
          else:
            fn_identifier = fn_node.name_def.identifier
        try:
          f = this_m.get_function(fn_identifier)
        except KeyError:
          if node.callee.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
            return
          raise
      else:
        raise NotImplementedError(
            'Only calls to named functions are currently supported, got callee: {!r}'
            .format(node.callee))

      node_symbolic_bindings = type_info.get_invocation_symbolic_bindings(
          node, bindings)

      # Either use the global type_info or the child type_info
      # chained off of this invocation.
      try:
        invocation_type_info = type_info.get_instantiation(
            node, node_symbolic_bindings)
      except KeyError:
        invocation_type_info = type_info
      callees.append(
          Callee(f, this_m, invocation_type_info, node_symbolic_bindings))

  cpp_ast_visitor.visit(func, InvocationVisitor())
  logging.vlog(3, 'Callees for %s: %s', func,
               [(cr.f.identifier, cr.sym_bindings) for cr in callees])
  return tuple(callees)


def _is_ready(ready: List[ConversionRecord], f: ast.Function, m: ast.Module,
              bindings: SymbolicBindings) -> bool:
  return any(
      cr.f == f and cr.m == m and cr.bindings == bindings for cr in ready)


def _add_to_ready(ready: List[ConversionRecord], imports: Dict[ast.Import,
                                                               ImportedInfo],
                  f: Union[ast.Function, ast.Test], m: ast.Module,
                  type_info: type_info_mod.TypeInfo,
                  bindings: SymbolicBindings) -> None:
  """Adds (f, bindings) to conversion order after deps have been added."""
  if _is_ready(ready, f, m, bindings):
    return

  # Remember the original callees value because we're gonna knock them out
  # from a list.
  orig_callees = tuple(get_callees(f, m, type_info, imports, bindings))

  # Knock out all callees that are already in the order.
  callees = list(orig_callees)
  for t in list(callees):
    if _is_ready(ready, t.f, t.m, t.sym_bindings):
      callees.remove(t)

  # For all of the remaining callees (that were not ready), add them to the
  # list before us, since we depend upon them.
  for callee in callees:
    _add_to_ready(ready, imports, callee.f, callee.m, callee.type_info,
                  callee.sym_bindings)

  assert not _is_ready(ready, f, m, bindings)

  # We don't convert the bodies of test constructs to IR
  if not isinstance(f, ast.Test):
    logging.vlog(3, 'Adding to ready sequence: %s', f.name.identifier)
    ready.append(
        ConversionRecord(f, m, type_info, bindings, callees=orig_callees))


def get_order(module: ast.Module,
              type_info: type_info_mod.TypeInfo,
              imports: Dict[ast.Import, ImportedInfo],
              traverse_tests: bool = False) -> List[ConversionRecord]:
  """Returns (topological) order for functions to be converted to IR.

  Args:
    module: Module to convert the (non-parametric) functions for.
    type_info: Mapping from node to type.
    imports: Transitive imports that are required by "module".
    traverse_tests: Whether to traverse DSLX test constructs. This flag should
      be set if we intend to run functions only called from test constructs
      through the JIT.
  """
  ready = []  # type: List[ConversionRecord]

  # Functions in the module should become ready in dependency order (they
  # referred to each other's names).

  for quickcheck in module.get_quickchecks():
    function = quickcheck.f
    assert not function.is_parametric(), function

    _add_to_ready(ready, imports, function, module, type_info, bindings=())

  for function in module.get_functions():
    if function.is_parametric():
      continue

    _add_to_ready(ready, imports, function, module, type_info, bindings=())

  if traverse_tests:
    for test in module.get_tests():
      _add_to_ready(ready, imports, test, module, type_info, bindings=())

  return ready
