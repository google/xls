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
"""Helper utilities for traversing AST nodes."""

from typing import Union, Callable, Tuple, Any

from xls.dslx import ast

ContextType = Any
GetImportedCallback = Callable[[ast.Import, ContextType], Tuple[ast.Module,
                                                                ContextType]]


def evaluate_to_struct_or_enum_or_annotation(
    node: Union[ast.TypeDef, ast.ModRef, ast.Struct],
    get_imported_module: GetImportedCallback, evaluation_context: ContextType
) -> Union[ast.Struct, ast.Enum, ast.TypeAnnotation]:
  """Returns the node dereferenced into a Struct or Enum or TypeAnnotation.

  Will produce TypeAnnotation in the case we bottom out in a tuple, for
  example.

  Args:
    node: Node to resolve to a struct/enum/annotation.
    get_imported_module: Callback that returns the referenced module given a
      ModRef using the provided evaluation_context.
    evaluation_context: Evaluation information being kept by the caller (either
      NodeToType in type deduction or Bindings in interpreter evaluation).
  """
  while isinstance(node, ast.TypeDef):
    annotation = node.type_
    if not annotation.is_typeref():
      return annotation
    node = annotation.typeref.type_def

  if isinstance(node, (ast.Struct, ast.Enum)):
    return node

  assert isinstance(node, ast.ModRef)
  imported_module, evaluation_context = get_imported_module(
      node.mod, evaluation_context)
  td = imported_module.get_typedef(node.value_tok.value)
  # Recurse to dereference it if it's a typedef in the imported module.
  td = evaluate_to_struct_or_enum_or_annotation(td, get_imported_module,
                                                evaluation_context)
  assert isinstance(td, (ast.Struct, ast.Enum, ast.TypeAnnotation)), td
  return td
