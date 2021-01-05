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

from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.scanner import Keyword
from xls.dslx.python.scanner import Token
from xls.dslx.python.scanner import TokenKind
from xls.dslx.python.scanner import TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS


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
