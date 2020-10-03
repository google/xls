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

"""Functions for dealing with concrete types and interpreter values."""

from typing import Tuple, Optional

from absl import logging

from xls.dslx import bit_helpers
from xls.dslx.interpreter.errors import FailureError
from xls.dslx.interpreter.value import Tag
from xls.dslx.interpreter.value import Value
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import EnumType
from xls.dslx.python.cpp_concrete_type import is_ubits
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_pos import Span
from xls.dslx.python.cpp_scanner import Keyword
from xls.dslx.python.cpp_scanner import Token
from xls.dslx.python.cpp_scanner import TokenKind
from xls.dslx.python.cpp_scanner import TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS


def _strength_reduce_enum(type_: ast.Enum, bit_count: int) -> ConcreteType:
  """Turns an enum to corresponding (bits) concrete type (w/signedness).

  For example, used in conversion checks.

  Args:
    type_: AST node (enum definition) to convert.
    bit_count: The bit count of the underlying bits type for the enum
      definition, as determined by type inference or interpretation.

  Returns:
    The concrete type that represents the enum's underlying bits type.
  """
  assert isinstance(type_, ast.Enum), type_
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
    return BitsType(signed, value.bits_payload.bit_count)
  elif value.tag == Tag.ARRAY:
    element_type = concrete_type_from_value(value.array_payload.index(0))
    return ArrayType(element_type, len(value))
  elif value.tag == Tag.TUPLE:
    return TupleType(
        tuple(concrete_type_from_value(m) for m in value.tuple_members))
  else:
    assert value.tag == Tag.ENUM, value
    return _strength_reduce_enum(value.type_, value.bits_payload.bit_count)


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


def _value_compatible_with_type(module: ast.Module, type_: ConcreteType,
                                value: Value) -> bool:
  """Returns whether value is compatible with type_ (recursively)."""
  assert isinstance(value, Value), value

  if isinstance(type_, TupleType) and value.is_tuple():
    return all(
        _value_compatible_with_type(module, ct, m)
        for ct, m in zip(type_.get_unnamed_members(), value.tuple_members))

  if isinstance(type_, ArrayType) and value.is_array():
    et = type_.get_element_type()
    return all(
        _value_compatible_with_type(module, et, m)
        for m in value.array_payload.elements)

  if isinstance(type_, EnumType) and value.tag == Tag.ENUM:
    return type_.get_nominal_type(module) == value.type_

  if isinstance(type_,
                BitsType) and not type_.signed and value.tag == Tag.UBITS:
    return value.bits_payload.bit_count == type_.get_total_bit_count()

  if isinstance(type_, BitsType) and type_.signed and value.tag == Tag.SBITS:
    return value.bits_payload.bit_count == type_.get_total_bit_count()

  if value.tag == Tag.ENUM and isinstance(type_, BitsType):
    return (value.type_.get_signedness() == type_.signed and
            value.bits_payload.bit_count == type_.get_total_bit_count())

  if value.tag == Tag.ARRAY and is_ubits(type_):
    flat_bit_count = value.array_payload.flatten().bits_payload.bit_count
    return flat_bit_count == type_.get_total_bit_count()

  if isinstance(type_, EnumType) and value.is_bits():
    return (type_.signed == (value.tag == Tag.SBITS) and
            type_.get_total_bit_count() == value.get_bit_count())

  raise NotImplementedError(type_, value)


def concrete_type_accepts_value(module: ast.Module, type_: ConcreteType,
                                value: Value) -> bool:
  """Returns whether 'value' conforms to this concrete type."""
  if value.tag == Tag.UBITS:
    return (isinstance(type_, BitsType) and not type_.signed and
            value.bits_payload.bit_count == type_.get_total_bit_count())
  if value.tag == Tag.SBITS:
    return (isinstance(type_, BitsType) and type_.signed and
            value.bits_payload.bit_count == type_.get_total_bit_count())
  if value.tag in (Tag.ARRAY, Tag.TUPLE, Tag.ENUM):
    return _value_compatible_with_type(module, type_, value)
  raise NotImplementedError(type_, value)


def concrete_type_convert_value(module: ast.Module, type_: ConcreteType,
                                value: Value, span: Span,
                                enum_values: Optional[Tuple[Value, ...]],
                                enum_signed: Optional[bool]) -> Value:
  """Converts 'value' into a value of this concrete type."""
  logging.vlog(3, 'Converting value %s to type %s', value, type_)
  if value.tag == Tag.UBITS and isinstance(type_, ArrayType):
    bits_per_element = type_.get_element_type().get_total_bit_count().value
    bits = value.bits_payload

    def bit_slice_value_at_index(i):
      return Value(
          Tag.UBITS,
          bits.slice(
              i * bits_per_element, (i + 1) * bits_per_element, lsb_is_0=False))

    return Value.make_array(
        tuple(bit_slice_value_at_index(i) for i in range(type_.size.value)))

  if (isinstance(type_, EnumType) and
      value.tag in (Tag.UBITS, Tag.SBITS, Tag.ENUM) and
      value.get_bit_count() == type_.get_total_bit_count()):
    # Check that the bits we're converting from are present in the enum type
    # we're converting to.
    nominal_type = type_.get_nominal_type(module)
    for enum_value in enum_values:
      if value.bits_payload == enum_value.bits_payload:
        break
    else:
      raise FailureError(
          span,
          'Value is not valid for enum {}: {}'.format(nominal_type.identifier,
                                                      value))
    return Value.make_enum(value.bits_payload, nominal_type)

  if (value.tag == Tag.ENUM and isinstance(type_, BitsType) and
      type_.get_total_bit_count() == value.get_bit_count()):
    constructor = Value.make_sbits if type_.signed else Value.make_ubits
    bit_count = type_.get_total_bit_count().value
    return constructor(bit_count, value.bits_payload.value)

  def zero_ext() -> Value:
    assert isinstance(type_, BitsType)
    constructor = Value.make_sbits if type_.signed else Value.make_ubits
    bit_count = type_.get_total_bit_count().value
    return constructor(bit_count,
                       value.get_bits_value() & bit_helpers.to_mask(bit_count))

  def sign_ext() -> Value:
    assert isinstance(type_, BitsType)
    constructor = Value.make_sbits if type_.signed else Value.make_ubits
    bit_count = type_.get_total_bit_count().value
    logging.vlog(3, 'Sign extending %s to %s', value, bit_count)
    return constructor(bit_count, value.bits_payload.sign_ext(bit_count).value)

  if value.tag == Tag.UBITS:
    return zero_ext()

  if value.tag == Tag.SBITS:
    return sign_ext()

  if value.tag == Tag.ENUM:
    assert enum_signed is not None
    return sign_ext() if enum_signed else zero_ext()

  # If we're converting an array into bits, flatten the array payload.
  if value.tag == Tag.ARRAY and isinstance(type_, BitsType):
    return value.array_payload.flatten()

  if concrete_type_accepts_value(module, type_, value):  # Vacuous conversion.
    return value

  raise FailureError(
      span,
      'Interpreter failure: cannot convert value %s (of type %s) to type %s' %
      (value, concrete_type_from_value(value), type_))
