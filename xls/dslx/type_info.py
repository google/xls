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

# Note: we often access "protected" members on instances of the
# known-same-class, but PyLint flags these as potential errors. We just blanket
# disable instead of annotating every line.
# pylint: disable=protected-access

"""Support for carrying type information from type inferencing."""

from typing import Tuple, Optional, Dict

import dataclasses

from xls.dslx import ast
from xls.dslx import span
from xls.dslx import symbolic_bindings as symbind_mod
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.xls_type_error import TypeInferenceError

ImportedInfo = Tuple[ast.Module, 'TypeInfo']


class TypeMissingError(span.PositionalError):
  """Raised when there is no binding from an AST node to its corresponding type.

  This is useful to raise in order to flag free variables that are dependencies
  for type inference; e.g. functions within a module that invoke other top-level
  functions. The type inference system can catch the error, infer the
  dependency, and re-attempt the deduction of the dependent function.
  """

  def __init__(self, node: ast.AstNode, suffix: str = ''):
    assert isinstance(node, ast.AstNode), repr(node)
    message = 'Missing type for AST node: {node}{suffix}'.format(
        node=node, suffix=' :: ' + suffix if suffix else '')
    # We don't know the real span of the user, we rely on the appropriate caller
    # to catch the error and populate this field properly.
    fake_span = span.Span(span.Pos('<fake>', 0, 0), span.Pos('<fake>', 0, 0))
    super(TypeMissingError, self).__init__(message, fake_span)
    self.node = node
    self.suffix = suffix
    self.user = None


@dataclasses.dataclass
class InvocationData:
  """Parametric instantiation information related to an invocation AST node.

  Instance variables:
    symbolic_bindings_map: Maps from symbolic bindings in the caller to the
      corresponding symbolic bindings in the callee.
    instantiations: Type information that is specialized for a particular
      parametric instantiation of an invocation.
  """
  invocation: ast.Invocation
  symbolic_bindings_map: Dict[
      symbind_mod.SymbolicBindings,
      symbind_mod.SymbolicBindings] = dataclasses.field(
          default_factory=dict)
  instantiations: Dict[symbind_mod.SymbolicBindings,
                       'TypeInfo'] = dataclasses.field(default_factory=dict)

  def update(self, other: 'InvocationData') -> None:
    assert self.invocation == other.invocation
    self.symbolic_bindings_map.update(other.symbolic_bindings_map)
    self.instantiations.update(other.instantiations)


@dataclasses.dataclass
class SliceData:
  node: ast.Slice
  bindings_to_start_width: Dict[symbind_mod.SymbolicBindings,
                                Tuple[int, int]] = dataclasses.field(
                                    default_factory=dict)

  def update(self, other: 'SliceData') -> None:
    assert self.node == other.node
    self.bindings_to_start_width.update(other.bindings_to_start_width)


class TypeInfo:
  """Holds {AstNode: ConcreteType} mapping and other type analysis info.

  Easily "chains" onto an existing mapping of node types when entering a scope
  with parametric bindings; e.g. a new node_to_type mapping is created for
  a parametric function's body after parametric instantiation.

  Also raises a TypeMissingError instead of a KeyError when we encounter a node
  that does not have a type known, so that it can be handled in a more specific
  way versus a KeyError.
  """

  def __init__(self, parent: Optional['TypeInfo'] = None):
    self._dict: Dict[ast.AstNode, ConcreteType] = {}
    self._imports: Dict[ast.Import, ImportedInfo] = {}
    self._name_to_const: Dict[ast.NameDef, ast.Constant] = {}
    self._invocations: Dict[ast.Invocation, InvocationData] = {}
    self._slices: Dict[ast.Slice, SliceData] = {}
    self._parent: Optional['TypeInfo'] = parent

  @property
  def parent(self) -> 'TypeInfo':
    return self._parent

  def update(self, other: 'TypeInfo') -> None:
    """Updates this type information object with the data from 'other'."""
    self._dict.update(other._dict)
    self._imports.update(other._imports)
    # Merge in all the invocation information.
    for invocation, data in other._invocations.items():
      if invocation in self._invocations:
        self._invocations[invocation].update(data)
      else:
        self._invocations[invocation] = data
    # Merge in all the slice information.
    for node, data in other._slices.items():
      if node in self._slices:
        self._slices[node].update(data)
      else:
        self._slices[node] = data

  def _top(self) -> 'TypeInfo':
    """Traverses to the "most parent" TypeInfo."""
    this = self
    while this._parent:
      this = this._parent
    return this

  def add_slice_start_width(
      self, node: ast.Slice,
      symbolic_bindings: symbind_mod.SymbolicBindings,
      start_width: Tuple[int, int]):
    """Notes start/width for a slice operation found during type inference."""
    self._top()._slices.setdefault(node, SliceData(
        node)).bindings_to_start_width[symbolic_bindings] = start_width

  def get_slice_start_width(
      self, node: ast.Slice,
      symbolic_bindings: symbind_mod.SymbolicBindings) -> Tuple[int, int]:
    """Retrieves start/width for slice operation found during type inference."""
    return self._top()._slices[node].bindings_to_start_width[symbolic_bindings]

  def add_invocation_symbolic_bindings(
      self, invocation: ast.Invocation,
      caller: symbind_mod.SymbolicBindings,
      callee: symbind_mod.SymbolicBindings) -> None:
    """Notes caller/callee relation of symbolic bindings at an invocation.

    This is kept from type inferencing time for convenience purposes (so it
    doesn't need to be recalculated anywhere; e.g. in the interpreter).

    Args:
      invocation: The invocation node that (may have) caused parametric
        instantiation.
      caller: The caller's symbolic bindings at the point of invocation.
      callee: The callee's computed symbolic bindings for the invocation.
    """
    self._top()._invocations.setdefault(
        invocation,
        InvocationData(invocation)).symbolic_bindings_map[caller] = callee

  def get_invocation_symbolic_bindings(
      self, invocation: ast.Invocation,
      caller: symbind_mod.SymbolicBindings
  ) -> symbind_mod.SymbolicBindings:
    """Returns callee bindings given caller bindings at an invocation."""
    return self._top()._invocations[invocation].symbolic_bindings_map[caller]

  def add_instantiation(self, invocation: ast.Invocation,
                        caller: symbind_mod.SymbolicBindings,
                        type_info: 'TypeInfo') -> None:
    """Adds derived type info for an "instantiation".

    An "instantiation" is an invocation of a parametric function from some
    caller context (given by the invocation / caller symbolic bindings). These
    have /derived/ type information, where the parametric expressions are
    concretized, and have concrete types corresponding to AST nodes in the
    instantiated parametric function.

    Args:
      invocation: The invocation the type information has been generated for.
      caller: The caller's symbolic bindings that caused this instantiation to
        occur.
      type_info: The type information that has been determined for this
        instantiation.
    """
    self._top()._invocations.setdefault(
        invocation,
        InvocationData(invocation)).instantiations[caller] = type_info

  def has_instantiation(self, invocation: ast.Invocation,
                        caller: symbind_mod.SymbolicBindings) -> bool:
    """Returns if there's type info at invocation with given caller bindings."""
    return caller in self._top()._invocations.get(
        invocation, InvocationData(invocation)).instantiations

  def get_instantiation(
      self, invocation: ast.Invocation,
      caller: symbind_mod.SymbolicBindings) -> 'TypeInfo':
    """Retrieves type info for invocation with given caller bindings."""
    return self._top()._invocations[invocation].instantiations[caller]

  def add_import(self, import_node: ast.Import, info: ImportedInfo) -> None:
    assert import_node not in self._imports, import_node
    self._imports[import_node] = info
    self.update(info[1])

  def note_constant(self, name_def: ast.NameDef, constant: ast.Constant):
    self._name_to_const[name_def] = constant

  def get_const_int(self, name_def: ast.NameDef, user_span: span.Span) -> int:
    if name_def not in self._name_to_const and self.parent:
      return self.parent.get_const_int(name_def, user_span)

    constant = self._name_to_const[name_def]
    value = constant.value
    if isinstance(value, ast.Number):
      return value.get_value_as_int()
    raise TypeInferenceError(
        span=user_span,
        type_=None,
        suffix='Expected to find a constant integral value with the name {};'
        'got: {}'.format(name_def, constant.value))

  def get_imports(self) -> Dict[ast.Import, ImportedInfo]:
    return self._imports if not self.parent else {
        **self._imports,
        **self.parent._imports
    }

  def get_imported(self, import_node: ast.Import) -> ImportedInfo:
    if self.parent:
      if import_node in self._imports:
        return self._imports[import_node]

      return self.parent._imports[import_node]

    return self._imports[import_node]

  def __setitem__(self, k: ast.AstNode, v: ConcreteType) -> None:
    self._dict[k] = v

  def __getitem__(self, k: ast.AstNode) -> ConcreteType:
    """Attempts to resolve AST node 'k' in the node-to-type dictionary.

    Args:
      k: The AST node to resolve to a concrete type.

    Raises:
      TypeMissingError: When the node is not found.

    Returns:
      The previously-determined type of the AST node 'k'.
    """
    assert isinstance(k, ast.AstNode), repr(k)
    try:
      if k in self._dict:
        return self._dict[k]
      if self.parent:
        return self.parent.__getitem__(k)
    except KeyError:
      span_suffix = ' @ {}'.format(k.span) if hasattr(k, 'span') else ''
      raise TypeMissingError(
          k, suffix='resolving type of node{}'.format(span_suffix))
    else:
      span_suffix = ' @ {}'.format(k.span) if hasattr(k, 'span') else ''
      raise TypeMissingError(
          k,
          suffix='resolving type of {} node{}'.format(k.__class__.__name__,
                                                      span_suffix))

  def __contains__(self, k: ast.AstNode) -> bool:
    return (k in self._dict or self.parent.__contains__(k)
            if self.parent else k in self._dict)
