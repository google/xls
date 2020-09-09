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

"""Datatype for binding of identifiers to interpreter value."""

from typing import Text, Callable, List, Union, Optional, Dict, Set, NamedTuple, cast

from xls.dslx.interpreter.value import Value
from xls.dslx.parametric_instantiator import SymbolicBindings
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python.cpp_ast import Span


InterpreterFn = Callable[[List[Value], Span, ast.Invocation], Value]
BindingEntry = Union[Value, InterpreterFn, ast.TypeDef, ast.Enum, ast.Struct,
                     ast.Module]
FnCtx = NamedTuple('FnCtx', [('module_name', Text), ('fn_name', Text),
                             ('sym_bindings', SymbolicBindings)])


class Bindings:
  """Represents the set of bindings (ident: value mappings) for evaluation.

  Acts as a {ident: Value} mapping that can easily "chain" onto an existing set
  of bindings when you enter a new binding scope; e.g. new bindings may be
  created in a loop body that you want to discard when you proceed past the loop
  body.

  Attributes:
    _parent: Bindings from the outer scope (see description for example).
    _map: Maps an identifier to its Value
    fn_ctx: The current (module name, function name, symbolic bindings) that
      these Bindings are being used with.
  """

  def __init__(self, parent: Optional['Bindings'] = None):
    self._parent = parent
    self._map = {}  # type: Dict[Text, BindingEntry]
    self.fn_ctx = None if parent is None else parent.fn_ctx  # type: Optional[FnCtx]

  def add_value(self, identifier: Text, value: Value):
    self._map[identifier] = value

  def add_value_tree(self, name_def_tree: ast.NameDefTree, value: Value):
    """Adds a (tuple) tree of values to the current bindings via name_def_tree.

    If the frontend types are checked properly, you can zip together the
    structure of name_def_tree and value. This also handles the case where the
    name_def_tree is simply a leaf without any tupling (a simple NameDef).

    Args:
      name_def_tree: Tree of identifiers to bind.
      value: Value that should have identical structure to name_def_tree (e.g.
        if the name_def_tree is (a, b, c) this should be a three-value tuple.
    """
    if name_def_tree.is_leaf():
      leaf = name_def_tree.get_leaf()
      if isinstance(leaf, ast.NameDef):
        self.add_value(leaf.identifier, value)
      return
    for subtree, subvalue in zip(name_def_tree.tree, value.tuple_members):
      self.add_value_tree(subtree, subvalue)

  def add_fn(self, identifier: Text, value: InterpreterFn):
    self._map[identifier] = value

  def add_mod(self, identifier: Text, value: ast.Module):
    self._map[identifier] = value

  def add_typedef(self, identifier: Text, value: ast.TypeDef):
    self._map[identifier] = value

  def add_enum(self, identifier: Text, value: ast.Enum):
    self._map[identifier] = value

  def keys(self) -> Set[Text]:
    keys = set(self._map.keys())
    if self._parent is None:
      return keys
    return keys | self._parent.keys()

  def _resolve_entry(self, identifier: Text) -> BindingEntry:
    if identifier in self._map:
      return self._map[identifier]
    if self._parent:
      # Note: we access a protected method here but it's of the same class, so
      # the "protected access" concern is erroneous.
      return self._parent._resolve_entry(identifier)  # pylint: disable=protected-access
    raise KeyError(
        'Cannot resolve identifier in bindings: {!r}; keys: {}'.format(
            identifier, self.keys()))

  def resolve_value_from_identifier(self, identifier: Text) -> Value:
    entry = self._resolve_entry(identifier)
    if callable(entry):
      return Value.make_function(entry)
    elif not isinstance(entry, Value):
      raise TypeError('Attempted to resolve a value but identifier {} '
                      'was not bound to a value; got: {!r}'.format(
                          identifier, entry))
    return entry

  def resolve_value(self, name_ref: ast.NameRef) -> Value:
    """Resolves an interpreter value from the bindings.

    Args:
      name_ref: Name reference to resolve.

    Returns:
      The interpreter value bound to name_ref, if it exists.

    Raises:
      TypeError: If a non-interpreter-value is bound to the name.
      KeyError: If the name is not bound.
    """
    return self.resolve_value_from_identifier(name_ref.identifier)

  def resolve_fn(self, name_ref: ast.NameRef) -> InterpreterFn:
    """Resolves a function value from the bindings.

    Args:
      name_ref: Name reference to resolve.

    Returns:
      The function value bound to name_ref, if it exists.

    Raises:
      TypeError: If a value that is not a function is bound to the name.
      KeyError: If the name is not bound.
    """
    entry = self._resolve_entry(name_ref.identifier)
    if not callable(entry):
      raise TypeError('Attempted to resolve a function but identifier {} '
                      'was not bound to a function; got: {!r}'.format(
                          name_ref.identifier, entry))
    return cast(InterpreterFn, entry)

  def resolve_mod(self, identifier: Text) -> ast.Module:
    entry = self._resolve_entry(identifier)
    if isinstance(entry, ast.Module):
      return entry
    raise TypeError('Attempted to resolve a module but identifier {!r} '
                    'was not bound to a module; got: {!r}'.format(
                        identifier, entry))

  def resolve_type_annotation(self, identifier: Text) -> ast.TypeAnnotation:
    """Resolves 'identifier' to a type binding, or raises."""
    entry = self._resolve_entry(identifier)
    if isinstance(entry, (ast.Enum, ast.TypeDef)):
      return entry.type_  # pytype: disable=attribute-error
    raise TypeError('Attempted to resolve a type but identifier {!r} '
                    'was not bound to a type; got: {!r}'.format(
                        identifier, entry))

  def resolve_type_annotation_or_enum(
      self,
      identifier: Text) -> Union[ast.TypeAnnotation, ast.Enum, ast.Struct]:
    entry = self._resolve_entry(identifier)
    if isinstance(entry, ast.TypeDef):
      return entry.type_  # pytype: disable=attribute-error
    if isinstance(entry, (ast.Enum, ast.TypeAnnotation, ast.Struct)):
      return entry
    raise TypeError('Attempted to resolve a type (or enum) but identifier {!r} '
                    'was not bound to a type (or enum); got: {!r}'.format(
                        identifier, entry))

  def clone_with(self, name_def_tree: ast.NameDefTree,
                 value: Value) -> 'Bindings':
    new_bindings = Bindings(self)
    new_bindings.add_value_tree(name_def_tree, value)
    new_bindings.fn_ctx = self.fn_ctx
    return new_bindings
