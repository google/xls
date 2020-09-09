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
"""AST node visitor base class."""

from typing import TypeVar

from absl import logging

from xls.dslx.python import cpp_ast as ast


class AstVisitor:

  # TODO(leary): 2020-09-04 Rename to custom_visit.
  @staticmethod
  def no_auto_traverse(f: TypeVar('T')) -> TypeVar('T'):
    """Decorator for visit methods that don't want the post-order traversal."""
    f.no_auto_traverse = True
    return f


def visit(node: ast.AstNode, visitor: AstVisitor) -> None:
  """Runs a postorder visit for 'visitor' starting at 'node'."""
  class_name = node.__class__.__name__
  handler = getattr(visitor, f'visit_{class_name}', None)

  # Handler can flag (at the function level) that it doesn't want to traverse to
  # the children.
  no_auto_traverse = getattr(handler, 'no_auto_traverse', False)
  if no_auto_traverse:
    pass
  else:
    for child in node.children:
      visit(child, visitor)

  if handler is None:
    logging.vlog(5, 'No handler in %s for node: %s', visitor.__class__.__name__,
                 class_name)
  else:
    logging.vlog(5, 'Visiting %s: %s', class_name, node)
    handler(node)
