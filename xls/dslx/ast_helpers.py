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

from typing import Union, Callable, Tuple, Any, Optional

from xls.dslx import ast
from xls.dslx import scanner
from xls.dslx import span as span_mod

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
    if not isinstance(annotation, ast.TypeRefTypeAnnotation):
      return annotation
    node = annotation.type_ref.type_def

  if isinstance(node, (ast.Struct, ast.Enum)):
    return node

  assert isinstance(node, ast.ModRef)
  imported_module, evaluation_context = get_imported_module(
      node.mod, evaluation_context)
  td = imported_module.get_typedef(node.value)
  # Recurse to dereference it if it's a typedef in the imported module.
  td = evaluate_to_struct_or_enum_or_annotation(td, get_imported_module,
                                                evaluation_context)
  assert isinstance(td, (ast.Struct, ast.Enum, ast.TypeAnnotation)), td
  return td


def tok_to_builtin_type(tok: scanner.Token) -> ast.BuiltinType:
  assert tok.is_keyword_in(scanner.TYPE_KEYWORDS)
  bt = getattr(ast.BuiltinType, tok.value.value.upper())
  assert isinstance(bt, ast.BuiltinType), repr(bt)
  return bt


def make_builtin_type_annotation(
    owner: ast.AstNodeOwner, span: span_mod.Span, tok: scanner.Token,
    dims: Tuple[ast.Expr, ...]) -> ast.TypeAnnotation:
  elem_type = ast.BuiltinTypeAnnotation(owner, span, tok_to_builtin_type(tok))
  for dim in dims:
    elem_type = ast.ArrayTypeAnnotation(owner, span, elem_type, dim)
  return elem_type


def make_type_ref_type_annotation(
    owner: ast.AstNodeOwner,
    span: span_mod.Span,
    type_ref: ast.TypeRef,
    dims: Tuple[ast.Expr, ...],
    parametrics: Optional[Tuple[ast.Expr, ...]] = None) -> ast.TypeAnnotation:
  """Creates a type ref annotation that may be wrapped in array dimensions."""
  assert dims is not None, dims
  elem_type = ast.TypeRefTypeAnnotation(owner, span, type_ref, parametrics)
  for dim in dims:
    elem_type = ast.ArrayTypeAnnotation(owner, span, elem_type, dim)
  return elem_type
