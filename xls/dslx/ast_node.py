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

"""Core types for AST nodes.

(Mostly broken out to reduce pytype checking time.)
"""

import abc
from typing import TypeVar, Any

from xls.dslx.span import Pos
from xls.dslx.span import Span


NameDefTree = Any  # pylint: disable=invalid-name
AstNodeOwner = Any


class AstVisitor(metaclass=abc.ABCMeta):
  """Visitor; overridden to support node-parametric behavior."""

  @staticmethod
  def no_auto_traverse(f: TypeVar('T')) -> TypeVar('T'):
    """Decorator for visit methods that don't want the post-order traversal."""
    f.no_auto_traverse = True
    return f


class AstNode(metaclass=abc.ABCMeta):
  """Abstract base class for AST nodes.

  AST nodes hold descriptive data about the grammar of the input program and
  refer back to tokens (and tokens notably refer to their positions in the input
  syntax file).
  """

  def __init__(self, owner: AstNodeOwner):
    assert owner is None or owner.__class__.__name__ == 'Module'
    self._owner = owner

  def accept(self, visitor: AstVisitor) -> None:
    """Visitor pattern: has this AST node accept the given visitor.

    Calls 'visit_{classname}' on the visitor, if it is present, with the actual
    type of this object used to determine the class name.

    Args:
      visitor: Visitor object being accepted.
    """
    m = getattr(visitor, 'visit_{}'.format(self.__class__.__name__),
                lambda x: None)
    # If the visitor says "don't automatically traverse below this node" then
    # we just visit the node itself, and don't call _accept_children on it to
    # recurse below it.
    if getattr(m, 'no_auto_traverse', False):
      m(self)
    else:
      self._accept_children(visitor)
      m(self)

  # TODO(leary): 2019-02-15 Uncomment this as we want full coverage of nodes in
  # describing how to visit their child nodes.
  # @abc.abstractmethod
  def _accept_children(self, visitor: AstVisitor) -> None:
    pass

  def get_span_or_fake(self) -> Span:
    if hasattr(self, 'span'):
      assert isinstance(self.span, Span)
      return self.span
    fake_pos = Pos('<no-file>', 0, 0)
    return Span(fake_pos, fake_pos)
