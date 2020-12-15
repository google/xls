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

# pylint: disable=invalid-name

"""Type system deduction rules for AST nodes."""

import typing
from typing import Callable

from absl import logging

from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_deduce
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_deduce import DeduceCtx


# Dictionary used as registry for rule dispatch based on AST node class.
_RULES = {
    ast.Array:
        cpp_deduce.deduce_Array,
    ast.Attr:
        cpp_deduce.deduce_Attr,
    ast.Binop:
        cpp_deduce.deduce_Binop,
    ast.Carry:
        cpp_deduce.deduce_Carry,
    ast.Cast:
        cpp_deduce.deduce_Cast,
    ast.ColonRef:
        cpp_deduce.deduce_ColonRef,
    ast.Constant:
        cpp_deduce.deduce_ConstantDef,
    ast.ConstRef:
        cpp_deduce.deduce_ConstRef,
    ast.ConstantArray:
        cpp_deduce.deduce_ConstantArray,
    ast.EnumDef:
        cpp_deduce.deduce_EnumDef,
    ast.For:
        cpp_deduce.deduce_For,
    ast.Index:
        cpp_deduce.deduce_Index,
    ast.Invocation:
        cpp_deduce.deduce_Invocation,
    ast.Let:
        cpp_deduce.deduce_Let,
    ast.Match:
        cpp_deduce.deduce_Match,
    ast.MatchArm:
        cpp_deduce.deduce_MatchArm,
    ast.NameRef:
        cpp_deduce.deduce_NameRef,
    ast.Number:
        cpp_deduce.deduce_Number,
    ast.Param:
        cpp_deduce.deduce_Param,
    ast.StructDef:
        cpp_deduce.deduce_StructDef,
    ast.StructInstance:
        cpp_deduce.deduce_StructInstance,
    ast.SplatStructInstance:
        cpp_deduce.deduce_SplatStructInstance,
    ast.Ternary:
        cpp_deduce.deduce_Ternary,
    ast.TypeDef:
        cpp_deduce.deduce_TypeDef,
    ast.TypeRef:
        cpp_deduce.deduce_TypeRef,
    ast.Unop:
        cpp_deduce.deduce_Unop,
    ast.While:
        cpp_deduce.deduce_While,
    ast.XlsTuple:
        cpp_deduce.deduce_XlsTuple,

    # Various type annotations.
    ast.ArrayTypeAnnotation:
        cpp_deduce.deduce_ArrayTypeAnnotation,
    ast.BuiltinTypeAnnotation:
        cpp_deduce.deduce_BuiltinTypeAnnotation,
    ast.TupleTypeAnnotation:
        cpp_deduce.deduce_TupleTypeAnnotation,
    ast.TypeRefTypeAnnotation:
        cpp_deduce.deduce_TypeRefTypeAnnotation,
}


def _deduce(n: ast.AstNode, ctx: DeduceCtx) -> ConcreteType:
  f = _RULES[n.__class__]
  f = typing.cast(Callable[[ast.AstNode, DeduceCtx], ConcreteType], f)
  result = f(n, ctx)
  ctx.type_info[n] = result
  return result


def deduce(n: ast.AstNode, ctx: DeduceCtx) -> ConcreteType:
  """Deduces and returns the type of value produced by this expr.

  Also adds n to ctx.type_info memoization dictionary.

  Args:
    n: The AST node to deduce the type for.
    ctx: Wraps a type_info, a dictionary mapping nodes to their types.

  Returns:
    The type of this expression.

  As a side effect the type_info mapping is filled with all the deductions
  that were necessary to determine (deduce) the resulting type of n.
  """
  assert isinstance(n, ast.AstNode), n
  if n in ctx.type_info:
    result = ctx.type_info[n]
    assert isinstance(result, ConcreteType), result
  else:
    result = ctx.type_info[n] = _deduce(n, ctx)
    logging.vlog(5, 'Deduced type of %s => %s', n, result)
    assert isinstance(result, ConcreteType), \
        '_deduce did not return a ConcreteType; got: {!r}'.format(result)
  return result
