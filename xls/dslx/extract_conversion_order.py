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

from typing import Tuple, Text, List, Dict, NamedTuple

from absl import logging

from xls.dslx import ast
from xls.dslx import deduce
from xls.dslx import dslx_builtins
from xls.dslx.parametric_instantiator import SymbolicBindings

Callee = NamedTuple(
    'Callee',
    [('f', ast.Function), ('m', ast.Module),
     ('sym_bindings', SymbolicBindings)],
)
ImportedInfo = Tuple[ast.Module, deduce.NodeToType]


class ConversionRecord(object):
  """Record used in sequence, noting order functions should be converted in.

  Describes a function instance that should be emitted (in an order determined
  by an encapsulating sequence). Annotated with metadata that describes the call
  graph instance.

  Attributes:
    f: Function AST node to convert.
    m: Module that f resides in.
    callees: Function names that 'f' calls.
    bindings: Parametric bindings for this function instance.
    callees: Functions that this instance calls.
  """

  def __init__(self,
               f: ast.Function,
               m: ast.Module,
               bindings: SymbolicBindings,
               callees: Tuple[Callee, ...] = ()):
    self.f = f
    self.m = m
    self.bindings = bindings
    self.callees = callees

  def __repr__(self) -> Text:
    return 'ConversionRecord(f={!r}, m={!r}, bindings={!r}, callees={!r})'.format(
        self.f.identifier, self.m, self.bindings, self.callees)


def get_callees(func: ast.Function, m: ast.Module, imports: Dict[ast.Import,
                                                                 ImportedInfo],
                bindings: SymbolicBindings) -> Tuple[Callee, ...]:
  """Traverses the definition of f to find callees.

  Args:
    func: Function to inspect for calls.
    m: Module that f resides in.
    imports: Mapping of modules imported by m.
    bindings: Bindings used in instantiation of f.

  Returns:
    Callee functions invoked by f, and the parametric bindings used in each of
    those invocations.
  """
  callees = []

  class InvocationVisitor(ast.AstVisitor):
    """Visits invocation nodes to build up the callees list."""

    def visit_Invocation(self, node: ast.Invocation) -> None:
      if isinstance(node.callee, ast.ModRef):
        this_m, _ = imports[node.callee.mod]
        f = this_m.get_function(node.callee.value_tok.value)
        fn_identifier = f.name.identifier
      elif isinstance(node.callee, ast.NameRef):
        this_m = m
        fn_identifier = node.callee.identifier
        if fn_identifier == 'map':
          # We need to make sure we convert the mapped function!
          fn_node = node.args[1]
          if isinstance(fn_node, ast.ModRef):
            fn_identifier = fn_node.value_tok.value
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

      node_symbolic_bindings = node.symbolic_bindings.get(
          (m.name, func.name.identifier, bindings), ())
      callees.append(Callee(f, this_m, node_symbolic_bindings))

  func.accept(InvocationVisitor())
  logging.vlog(3, 'Callees for %s: %s', func,
               [(cr.f.identifier, cr.sym_bindings) for cr in callees])
  return tuple(callees)


def _is_ready(ready: List[ConversionRecord], f: ast.Function, m: ast.Module,
              bindings: SymbolicBindings) -> bool:
  return any(
      cr.f == f and cr.m == m and cr.bindings == bindings for cr in ready)


def _add_to_ready(ready: List[ConversionRecord],
                  imports: Dict[ast.Import, ImportedInfo], f: ast.Function,
                  m: ast.Module, bindings: SymbolicBindings) -> None:
  """Adds (f, bindings) to conversion order after deps have been added."""
  if _is_ready(ready, f, m, bindings):
    return

  # Remember the original callees value because we're gonna knock them out
  # from a list.
  orig_callees = tuple(get_callees(f, m, imports, bindings))

  # Knock out all callees that are already in the order.
  callees = list(orig_callees)
  for t in list(callees):
    if _is_ready(ready, t.f, t.m, t.sym_bindings):
      callees.remove(t)

  # For all of the remaining callees (that were not ready), add them to the
  # list before us, since we depend upon them.
  for callee in callees:
    _add_to_ready(ready, imports, callee.f, callee.m, callee.sym_bindings)

  assert not _is_ready(ready, f, m, bindings)
  logging.vlog(3, 'Adding to ready sequence: %s', f.name.identifier)
  ready.append(ConversionRecord(f, m, bindings, callees=orig_callees))


def get_order(
    module: ast.Module, imports: Dict[ast.Import,
                                      ImportedInfo]) -> List[ConversionRecord]:
  """Returns (topological) order for functions to be converted to IR.

  Args:
    module: Module to convert the (non-parametric) functions for.
    imports: Transitive imports that are required by "module".
  """
  ready = []  # type: List[ConversionRecord]

  # Functions in the module should become ready in dependency order (they
  # referred to each other's names).
  for function in module.get_functions():
    if function.is_parametric():
      continue

    _add_to_ready(ready, imports, function, module, bindings=())

  return ready
