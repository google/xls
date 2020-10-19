# Lint as: python3
#
# Copyright 2020 The XLS Authors
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

from typing import Union, Callable, Tuple, Any, Optional, Sequence

from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_scanner as scanner_mod
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span
from xls.ir.python import bits as ir_bits

ContextType = Any
GetImportedCallback = Callable[[ast.Import, ContextType], Tuple[ast.Module,
                                                                ContextType]]
StructInstanceMembers = Sequence[Tuple[str, ast.Expr]]

# (T, T) -> bool operators.
BINOP_COMPARISON_KIND_LIST = [
    ast.BinopKind.EQ,
    ast.BinopKind.NE,
    ast.BinopKind.GT,
    ast.BinopKind.GE,
    ast.BinopKind.LT,
    ast.BinopKind.LE,
]
BINOP_COMPARISON_KINDS = frozenset(BINOP_COMPARISON_KIND_LIST)
BINOP_ENUM_OK_KINDS = (
    ast.BinopKind.EQ,
    ast.BinopKind.NE,
)
BINOP_SHIFTS = (
    ast.BinopKind.SHLL,
    ast.BinopKind.SHRL,
    ast.BinopKind.SHRA,
)
# (T, T) -> T operators.
BINOP_SAME_TYPE_KIND_LIST = [
    ast.BinopKind.AND,
    ast.BinopKind.OR,
    ast.BinopKind.SHLL,
    ast.BinopKind.SHRL,
    ast.BinopKind.SHRA,
    ast.BinopKind.XOR,
    ast.BinopKind.SUB,
    ast.BinopKind.ADD,
    ast.BinopKind.DIV,
    ast.BinopKind.MUL,
]
# (T) -> T operators.
UNOP_SAME_TYPE_KIND_LIST = [
    ast.UnopKind.INV,
    ast.UnopKind.NEG,
]


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
  td = imported_module.get_typedef_by_name()[node.value]
  # Recurse to dereference it if it's a typedef in the imported module.
  td = evaluate_to_struct_or_enum_or_annotation(td, get_imported_module,
                                                evaluation_context)
  assert isinstance(td, (ast.Struct, ast.Enum, ast.TypeAnnotation)), td
  return td


def tok_to_builtin_type(tok: scanner_mod.Token) -> ast.BuiltinType:
  assert tok.is_keyword_in(scanner_mod.TYPE_KEYWORDS)
  assert isinstance(tok.value, scanner_mod.Keyword), tok.value
  bt = getattr(ast.BuiltinType, tok.value.value.upper())
  assert isinstance(bt, ast.BuiltinType), repr(bt)
  return bt


def get_builtin_type(signed: bool, width: int) -> ast.BuiltinType:
  prefix = 'S' if signed else 'U'
  return getattr(ast.BuiltinType, f'{prefix}{width}')


def make_builtin_type_annotation(
    owner: ast.AstNodeOwner, span: Span, tok: scanner_mod.Token,
    dims: Tuple[ast.Expr, ...]) -> ast.TypeAnnotation:
  elem_type = ast.BuiltinTypeAnnotation(owner, span, tok_to_builtin_type(tok))
  for dim in dims:
    elem_type = ast.ArrayTypeAnnotation(owner, span, elem_type, dim)
  return elem_type


def make_type_ref_type_annotation(
    owner: ast.AstNodeOwner,
    span: Span,
    type_ref: ast.TypeRef,
    dims: Tuple[ast.Expr, ...],
    parametrics: Optional[Tuple[ast.Expr, ...]] = None) -> ast.TypeAnnotation:
  """Creates a type ref annotation that may be wrapped in array dimensions."""
  assert dims is not None, dims
  elem_type = ast.TypeRefTypeAnnotation(owner, span, type_ref, parametrics)
  for dim in dims:
    elem_type = ast.ArrayTypeAnnotation(owner, span, elem_type, dim)
  return elem_type


_FAKE_POS = Pos('<no-file>', 0, 0)
_FAKE_SPAN = Span(_FAKE_POS, _FAKE_POS)


def get_span_or_fake(n: ast.AstNode) -> Span:
  return getattr(n, 'span', _FAKE_SPAN)


def _get_value_as_int(s: str) -> int:
  if s in ('true', 'false'):
    return int(s == 'true')
  if s.startswith(('0x', '-0x')):
    return int(s.replace('_', ''), 16)
  if s.startswith(('0b', '-0b')):
    return int(s.replace('_', ''), 2)
  return int(s.replace('_', ''))


def get_value_as_int(n: ast.Number) -> int:
  """Returns the numerical value contained in the AST node as a Python int."""
  assert isinstance(n, ast.Number), n
  if n.kind == ast.NumberKind.CHARACTER:
    return ord(n.value)
  return _get_value_as_int(n.value)


def get_value_as_bits(n: ast.Number, bit_count: int) -> ir_bits.Bits:
  """Returns the numerical value contained in the AST node as a Python int."""
  x = get_value_as_int(n)
  return ir_bits.from_long(x, bit_count)


def get_token_value_as_int(t: scanner_mod.Token) -> int:
  assert isinstance(t, scanner_mod.Token), t
  assert isinstance(t.value, str), t
  if t.kind == scanner_mod.TokenKind.CHARACTER:
    return ord(t.value)
  return _get_value_as_int(t.value)


def do_preorder(n: ast.NameDefTree,
                f: Callable[[ast.NameDefTree, int, int], None],
                level: int = 1) -> None:
  """Performs a preorder traversal under this node in the NameDefTree.

  Args:
    n: The NameDefTree to do a preorder traversal of.
    f: Callback invoked as f(NameDefTree, level, branchno).
    level: Current level of the node.
  """
  if n.is_leaf():
    return

  for i, item in enumerate(n.tree):
    f(item, level, i)
    do_preorder(item, f, level + 1)
