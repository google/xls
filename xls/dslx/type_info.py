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
"""Support for carrying type information from type inferencing."""

from typing import Tuple, Optional, Dict

from xls.dslx import ast
from xls.dslx import span
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
    self._parent: Optional['TypeInfo'] = parent

  @property
  def parent(self) -> 'TypeInfo':
    return self._parent

  def update(self, other: 'TypeInfo') -> None:
    self._dict.update(other._dict)  # pylint: disable=protected-access
    self._imports.update(other._imports)  # pylint: disable=protected-access

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
        **self._imports,  # pylint: disable=protected-access
        **self.parent._imports  # pylint: disable=protected-access
    }

  def get_imported(self, import_node: ast.Import) -> ImportedInfo:
    if self.parent:
      if import_node in self._imports:
        return self._imports[import_node]

      return self.parent._imports[import_node]  # pylint: disable=protected-access

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
