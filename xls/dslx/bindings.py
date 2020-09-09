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

"""Bindings class (names to values tracking) for use in parsing."""

from typing import Union, Text, Optional, Dict

from xls.dslx.parse_error import ParseError
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span

BoundNode = Union[ast.Enum, ast.TypeDef, ast.Constant, ast.NameDef,
                  ast.BuiltinNameDef, ast.Struct, ast.Import]
NameDefNode = Union[ast.NameDef, ast.BuiltinNameDef]


class Bindings(object):
  """Maps identifiers to the AST node that bound that identifier.

  This datatype is "stackable" so that we can easily take the bindings at a
  given point in the program (say in a function) and extend it with a new scope
  by stacking a fresh Bindings object on top (sometimes also referred to as a
  "scope chain"). For example:

    builtin_bindings = Bindings(None)
    builtin_bindings.add('range', ast.BuiltinNameDef('range'))

    # Create a fresh scope, with no need to copy the builtin_bindings object.
    function_bindings = Bindings(builtin_bindings)
    f = parse_function(function_bindings)

  We can do this because bindings are immutable and stack according to lexical
  scope; new bindings in the worst case only shadow previous bindings.
  """

  def __init__(self, parent: Optional['Bindings'] = None):
    self.parent = parent
    self._local_bindings = {}  # type: Dict[Text, BoundNode]

  def has_local_bindings(self) -> bool:
    return bool(self._local_bindings)

  def add(self, name: Text, binding: BoundNode) -> None:
    assert isinstance(
        name, str), 'Expected str name for binding; got: %r' % name
    self._local_bindings[name] = binding

  def resolve_node(self, name: Text, span: Span) -> BoundNode:
    """Returns the AST node bound to "name".

    Args:
      name: Identifier to resolve to a defining AST node.
      span: Span that the reference originates from.

    Raises:
      ParseError: If the name is not found in the bindings.
    """
    assert name is not None
    local_binding = self._local_bindings.get(name)
    if local_binding is None:
      if self.parent:
        return self.parent.resolve_node(name, span)
      else:
        raise ParseError(span,
                         'Cannot find a definition for name: {!r}'.format(name))
    return local_binding

  def resolve(self, name: Text, span: Span) -> NameDefNode:
    node = self.resolve_node(name, span)
    if isinstance(node, (ast.Constant, ast.Enum, ast.TypeDef, ast.Struct)):
      return node.name
    assert isinstance(node, (ast.NameDef, ast.BuiltinNameDef)), repr(node)
    return node

  def resolve_node_or_none(self, name: Text) -> Optional[BoundNode]:
    fake_pos = Pos('<fake>', 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    try:
      node = self.resolve_node(name, fake_span)
    except ParseError:
      return None
    else:
      return node

  def resolve_or_none(self, name: Text) -> Optional[NameDefNode]:
    fake_pos = Pos('<fake>', 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    try:
      node = self.resolve(name, fake_span)
    except ParseError:
      return None
    else:
      return node

  def has_name(self, name: str) -> bool:
    return self.resolve_or_none(name) is not None
