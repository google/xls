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

"""Functions for dealing with concrete types and interpreter values."""

from typing import Tuple

from absl import logging

from xls.dslx.python import cpp_ast as ast
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_scanner import Keyword
from xls.dslx.python.cpp_scanner import Token
from xls.dslx.python.cpp_scanner import TokenKind
from xls.dslx.python.cpp_scanner import TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS
from xls.dslx.python.interp_value import Tag
from xls.dslx.python.interp_value import Value


def _strength_reduce_enum(type_: ast.EnumDef, bit_count: int) -> ConcreteType:
  """Turns an enum to corresponding (bits) concrete type (w/signedness).

  For example, used in conversion checks.

  Args:
    type_: AST node (enum definition) to convert.
    bit_count: The bit count of the underlying bits type for the enum
      definition, as determined by type inference or interpretation.

  Returns:
    The concrete type that represents the enum's underlying bits type.
  """
  assert isinstance(type_, ast.EnumDef), type_
  signed = type_.signed
  assert isinstance(signed, bool), type_
  return BitsType(signed, bit_count)


def concrete_type_from_value(value: Value) -> ConcreteType:
  """Returns the concrete type of 'value'.

  Note that:
  * Non-zero-length arrays are assumed (for zero length arrays we can't
    currently deduce the type from the value because the concrete element type
    is not reified in the array value.
  * Enums are strength-reduced to their underlying bits (storage) type.

  Args:
    value: Value to determine the concrete type for.
  """
  if value.tag in (Tag.UBITS, Tag.SBITS):
    signed = value.tag == Tag.SBITS
    return BitsType(signed, value.get_bit_count())
  elif value.tag == Tag.ARRAY:
    element_type = concrete_type_from_value(value.index(Value.make_u32(0)))
    return ArrayType(element_type, len(value))
  elif value.tag == Tag.TUPLE:
    return TupleType(
        tuple(concrete_type_from_value(m) for m in value.get_elements()))
  else:
    assert value.tag == Tag.ENUM, value
    return _strength_reduce_enum(value.type_, value.get_bit_count())


def concrete_type_from_element_type_and_dims(
    element_type: ConcreteType, dims: Tuple[int, ...]) -> ConcreteType:
  """Wraps element_type in arrays according to `dims`, dims[0] as most minor."""
  t = element_type
  for dim in dims:
    t = ArrayType(t, dim)
  return t


def concrete_type_from_dims(primitive: Token,
                            dims: Tuple[int, ...]) -> 'ConcreteType':
  """Creates a concrete type from the primitive type token and dims.

  Args:
    primitive: The token holding the primitive type as a keyword.
    dims: Dimensions to apply to the primitive type; e.g. () is scalar, (5) is
      1-D array of 5 elements having the primitive type.

  Returns:
    A concrete type object.

  Raises:
    ValueError: If the primitive keyword is unrecognized or dims are empty.
  """
  if primitive.is_keyword(Keyword.BITS) or primitive.is_keyword(Keyword.UN):
    base_type = BitsType(signed=False, size=dims[-1])
  elif primitive.is_keyword(Keyword.SN):
    base_type = BitsType(signed=True, size=dims[-1])
  else:
    assert primitive.kind == TokenKind.KEYWORD
    signedness, bits = TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS[primitive.value]
    element_type = BitsType(signedness, bits)
    while dims:
      dims, minor = dims[:-1], dims[-1]
      element_type = ArrayType(element_type, minor)
    return element_type

  result = concrete_type_from_element_type_and_dims(base_type, dims[:-1])
  logging.vlog(4, '%r %r => %r', primitive, dims, result)
  return result
