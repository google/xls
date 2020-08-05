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

"""Representation / functionality for  a value in the syntax interpreter."""

import enum as enum_mod
from typing import Callable, Text, Union, Tuple, Any, Optional

from xls.dslx.interpreter.array import Array
from xls.dslx.interpreter.bits import Bits
from xls.dslx.span import Span


XlsEnum = Any


class Tag(enum_mod.Enum):
  """Tags a value to denote its payload.

  In cases like distinguishing ubits from sbits, changes the behavior of
  operators like '<' (i.e. changes the unsigned vs signed interpretation of the
  bits payload).
  """
  UBITS = 'ubits'
  SBITS = 'sbits'
  TUPLE = 'tuple'
  ARRAY = 'array'
  ENUM = 'enum'
  FUNCTION = 'function'


class Value:
  """Represents a value in the interpreter evaluation.

  The value type is capable of representing all expression evaluation results.
  It's a tagged union (variant type). It provides support for some key
  operations; a type checker should have validated that all requested operations
  are provably acceptable to avoid getting EvaluateError/TypeErrors raised at
  runtime.
  """

  @classmethod
  def make_tuple(cls, members: Tuple['Value', ...]) -> 'Value':
    return Value(Tag.TUPLE, members)

  @classmethod
  def make_ubits(cls, bit_count: int, value: int) -> 'Value':
    return cls(Tag.UBITS, Bits(bit_count, value))

  @classmethod
  def make_sbits(cls, bit_count: int, value: int) -> 'Value':
    return cls(Tag.SBITS, Bits(bit_count, value))

  @classmethod
  def make_array(cls, elements: Tuple['Value', ...]) -> 'Value':
    return cls(Tag.ARRAY, Array(elements))

  @classmethod
  def make_u32(cls, value: int) -> 'Value':
    return cls.make_ubits(bit_count=32, value=value)

  @classmethod
  def make_bool(cls, value: bool) -> 'Value':
    return cls.make_ubits(bit_count=1, value=int(value))

  @classmethod
  def make_enum(cls, bits: Bits, type_: XlsEnum) -> 'Value':
    return Value(Tag.ENUM, bits, type_=type_)

  @classmethod
  def make_function(cls, fn: Callable[..., 'Value']) -> 'Value':
    return Value(Tag.FUNCTION, fn)

  def __init__(self,
               tag: Tag,
               payload: Union[Bits, Array, Tuple['Value', ...],
                              Callable[..., 'Value']],
               type_: Optional[XlsEnum] = None):
    if tag == Tag.ARRAY:
      assert isinstance(payload, Array), payload
    elif tag in (Tag.UBITS, Tag.SBITS):
      assert isinstance(payload, Bits), payload
    elif tag == Tag.TUPLE:
      assert isinstance(payload, tuple), payload
    elif tag == Tag.ENUM:
      assert type_ is not None
    self.tag = tag
    self.payload = payload
    self.type_ = type_

  def __hash__(self) -> int:
    if self.tag in (Tag.UBITS, Tag.SBITS):
      return hash((self.bits_payload.bit_count, self.bits_payload.value))
    elif self.tag == Tag.TUPLE:
      return hash(tuple(hash(e) for e in self.tuple_members))
    else:
      assert self.tag == Tag.ARRAY
      return hash(tuple(hash(e) for e in self.array_payload.elements))

  def __repr__(self) -> Text:
    return 'Value(tag={!r}, payload={!r})'.format(self.tag, self.payload)

  def __str__(self) -> Text:
    if self.tag == Tag.ARRAY:
      return str(self.array_payload)
    elif self.tag == Tag.UBITS or self.tag == Tag.SBITS:
      return str(self.bits_payload)
    elif self.tag == Tag.TUPLE:
      return '({})'.format(', '.join(str(m) for m in self.tuple_members))
    elif self.tag == Tag.ENUM:
      return '{}:{}'.format(self.type_, self.bits_payload)
    else:
      return repr(self)

  def to_human_str(self) -> Text:
    if self.tag == Tag.ARRAY:
      return self.array_payload.to_human_str()
    elif self.tag in (Tag.UBITS, Tag.SBITS, Tag.ENUM):
      return self.bits_payload.to_human_str()
    elif self.tag == Tag.TUPLE:
      return '({})'.format(', '.join(
          m.to_human_str() for m in self.tuple_members))
    else:
      raise NotImplementedError(self.tag)

  def __iter__(self):
    if self.tag == Tag.ARRAY:
      for element in self.array_payload.elements:
        yield element
      return
    if self.tag == Tag.TUPLE:
      for element in self.tuple_members:
        yield element
      return
    raise TypeError('Value is not iterable: {!r}'.format(self))

  def __len__(self) -> int:
    if self.tag == Tag.ARRAY:
      return len(self.array_payload.elements)
    if self.tag == Tag.TUPLE:
      return len(self.tuple_members)
    raise NotImplementedError(self)

  @property
  def bits_payload(self) -> Bits:
    assert self.tag in (Tag.UBITS, Tag.SBITS, Tag.ENUM)
    assert isinstance(self.payload, Bits), self
    return self.payload

  @property
  def array_payload(self) -> Array:
    assert self.tag == Tag.ARRAY, self
    assert isinstance(self.payload, Array)
    return self.payload

  @property
  def function_payload(self) -> Callable[..., 'Value']:
    assert self.tag == Tag.FUNCTION
    assert callable(self.payload)
    return self.payload

  @property
  def tuple_members(self) -> Tuple['Value', ...]:
    assert self.tag == Tag.TUPLE
    assert isinstance(
        self.payload, tuple
    ), 'Payload of tuple-tagged value is not a tuple; payload: {!r}'.format(
        self.payload)
    return self.payload

  def tuple_replace(self, i: int, v: 'Value') -> 'Value':
    assert isinstance(v, Value), repr(v)
    assert i < len(self.tuple_members), (i, len(self.tuple_members))
    return Value.make_tuple(
        tuple(v if index == i else e
              for index, e in enumerate(self.tuple_members)))

  def is_tuple(self) -> bool:
    return self.tag == Tag.TUPLE

  def is_nil_tuple(self) -> bool:
    return self.is_tuple() and not self.tuple_members

  def is_array(self) -> bool:
    return self.tag == Tag.ARRAY

  def is_bits(self) -> bool:
    return self.tag in (Tag.UBITS, Tag.SBITS)

  def is_ubits(self) -> bool:
    return self.tag == Tag.UBITS

  def is_sbits(self) -> bool:
    return self.tag == Tag.SBITS

  def is_signed_enum(self) -> bool:
    return self.tag == Tag.ENUM and self.type_.get_signedness()

  def is_unsigned_enum(self) -> bool:
    return self.tag == Tag.ENUM and not self.type_.get_signedness()

  def is_function(self):
    return self.tag == Tag.FUNCTION

  def get_bits_value(self) -> int:
    if self.tag not in (Tag.UBITS, Tag.SBITS, Tag.ENUM):
      raise TypeError('Value is not "bits" or "enum" typed.')
    return self.bits_payload.value

  def get_bits_value_signed(self) -> int:
    return self.bits_payload.signed_value

  def get_bit_count(self) -> int:
    assert self.tag in (Tag.UBITS, Tag.SBITS, Tag.ENUM)
    return self.bits_payload.bit_count

  def concat(self, other: 'Value') -> 'Value':
    if self.tag == other.tag == Tag.UBITS:
      return Value(Tag.UBITS, self.bits_payload.concat(other.bits_payload))
    if self.tag == other.tag == Tag.ARRAY:
      return Value(Tag.ARRAY, self.array_payload.concat(other.array_payload))
    raise TypeError('Cannot concat values: {} and {}'.format(self, other))

  def add(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.add(other.bits_payload))
    raise TypeError('Cannot add values with tags:', self.tag, other.tag)

  def add_with_carry(self, other: 'Value') -> 'Value':
    if self.tag == other.tag in (Tag.UBITS, Tag.SBITS):
      carry, result = self.bits_payload.add_with_carry(other.bits_payload)
      return Value(Tag.TUPLE,
                   (Value(Tag.UBITS, carry), Value(self.tag, result)))
    raise TypeError('Cannot add_with_carry values with tags:', self.tag,
                    other.tag)

  def umul(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.umul(other.bits_payload))
    raise TypeError('Cannot umul values with tags:', self.tag, other.tag)

  def smul(self, other: 'Value') -> 'Value':
    if self.tag in (Tag.UBITS, Tag.SBITS) and other.tag in (Tag.UBITS,
                                                            Tag.SBITS):
      return Value(Tag.SBITS, self.bits_payload.smul(other.bits_payload))
    raise TypeError('Cannot smul values with tags:', self.tag, other.tag)

  def sub(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.sub(other.bits_payload))
    raise TypeError('Cannot sub values with tag:', self.tag)

  def mul(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.mul(other.bits_payload))
    raise TypeError('Cannot mul values with tag:', self.tag)

  def shll(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.shll(other.bits_payload))
    raise TypeError('Cannot shll values with tag:', self.tag)

  def shrl(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.shrl(other.bits_payload))
    raise TypeError('Cannot shrl values with tag:', self.tag)

  def shra(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.shra(other.bits_payload))
    raise TypeError('Cannot shra values with tag:', self.tag)

  def bitwise_or(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.bitwise_or(other.bits_payload))
    raise TypeError('Cannot bitwise-or values with tag:', self.tag)

  def bitwise_xor(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.bitwise_xor(other.bits_payload))
    raise TypeError('Cannot bitwise-xor values with tag:', self.tag)

  def bitwise_and(self, other: 'Value') -> 'Value':
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.bitwise_and(other.bits_payload))
    raise TypeError('Cannot bitwise-and values with tag:', self.tag)

  def bitwise_negate(self) -> 'Value':
    if self.tag in (Tag.UBITS, Tag.SBITS):
      return Value(self.tag, self.bits_payload.bitwise_negate())
    raise TypeError('Cannot bitwise-negate values with tag:', self.tag)

  def arithmetic_negate(self) -> 'Value':
    if self.tag in (Tag.UBITS, Tag.SBITS):
      return self.bitwise_negate().add(
          Value(self.tag, Bits(bit_count=self.bits_payload.bit_count, value=1)))
    raise TypeError('Cannot arithmetic-negate values with tag:', self.tag)

  def floordiv(self, other: 'Value') -> 'Value':
    if self.tag == other.tag == Tag.UBITS:
      return Value(Tag.UBITS, self.bits_payload.floordiv(other.bits_payload))
    raise TypeError('Cannot floordiv values with tag:', self.tag)

  def __eq__(self, other: 'Value') -> bool:
    if self.tag == other.tag and self.tag in (Tag.UBITS, Tag.SBITS):
      return self.bits_payload == other.bits_payload
    if self.tag == other.tag == Tag.ENUM:
      if self.type_ == other.type_:
        return self.bits_payload == other.bits_payload
      else:
        raise TypeError(
            'Cannot compare differently typed enums: {} vs {}'.format(
                self.type_, other.type_))
    if self.tag == other.tag == Tag.ARRAY:
      return self.array_payload == other.array_payload
    if self.tag == other.tag == Tag.TUPLE:
      return len(self.tuple_members) == len(other.tuple_members) and all(
          m.eq(o).is_true()
          for m, o in zip(self.tuple_members, other.tuple_members))
    if self.tag == other.tag == Tag.ARRAY:
      return len(self.tuple_members) == len(other.tuple_members) and all(
          m.eq(o).is_true() for m, o in zip(self.array_payload.elements,
                                            other.array_payload.elements))
    if self.tag == Tag.ENUM or other.tag == Tag.ENUM:
      if self.bits_payload.bit_count != other.bits_payload.bit_count:
        return False
      enum, bits = (self, other) if self.tag == Tag.ENUM else (other, self)
      if enum.is_signed_enum() != (bits.tag == Tag.SBITS):
        return False
      return self.bits_payload == other.bits_payload

    raise TypeError('Cannot compare values:'
                    '\n\tself: {}\n\tother: {}'.format(self, other))

  def __ne__(self, other: 'Value') -> bool:
    return not self.__eq__(other)

  def eq(self, other: 'Value') -> 'Value':
    return Value.make_bool(self == other)

  def eq_ignore_sign(self, other: 'Value') -> 'Value':
    """Determines "this == other" in a signedness-insensitive manner."""
    if self.tag in (Tag.UBITS, Tag.SBITS) and other.tag in (Tag.UBITS,
                                                            Tag.SBITS):
      return Value.make_bool(self.bits_payload == other.bits_payload)

    if self.tag == other.tag == Tag.TUPLE:
      result = len(self.tuple_members) == len(other.tuple_members) and all(
          m.eq_ignore_sign(o).is_true()
          for m, o in zip(self.tuple_members, other.tuple_members))
      return Value.make_bool(result)
    elif self.tag == other.tag == Tag.ARRAY:
      result = len(self.array_payload) == len(other.array_payload) and all(
          m.eq_ignore_sign(o).is_true() for m, o in zip(self, other))
      return Value.make_bool(result)

    raise NotImplementedError

  def ne(self, other: 'Value') -> 'Value':
    return Value.make_bool(False if self == other else True)

  def _cmp_helper(self, other: 'Value', suffix: Text) -> 'Value':
    """Dispatches signed/unsigned comparison operations based on tag."""
    if self.tag != other.tag:
      raise TypeError(
          'Cannot compare values with different tags: {!r} vs {!r}'.format(
              self.tag, other.tag))
    if self.tag == Tag.UBITS:
      unsigned = getattr(self.bits_payload, 'u{}'.format(suffix))
      return Value(Tag.UBITS, unsigned(other.bits_payload))
    if self.tag == Tag.SBITS:
      signed = getattr(self.bits_payload, 's{}'.format(suffix))
      return Value(Tag.UBITS, signed(other.bits_payload))
    raise TypeError('Cannot compare values with tag: {!r}'.format(self.tag))

  def lt(self, other: 'Value') -> 'Value':
    return self._cmp_helper(other, 'lt')

  def le(self, other: 'Value') -> 'Value':
    return self._cmp_helper(other, 'le')

  def gt(self, other: 'Value') -> 'Value':
    return self._cmp_helper(other, 'gt')

  def ge(self, other: 'Value') -> 'Value':
    return self._cmp_helper(other, 'ge')

  def scmp(self, other: 'Value', method: Text) -> 'Value':
    if self.tag != other.tag:
      raise TypeError(
          'Cannot compare values with different tags: {!r} vs {!r}'.format(
              self.tag, other.tag))
    signed = getattr(self.bits_payload, method)
    return Value(Tag.UBITS, signed(other.bits_payload))

  def index(self, other: 'Value') -> 'Value':
    if self.tag == Tag.ARRAY and other.tag == Tag.UBITS:
      return self.array_payload.index(other.get_bits_value())
    if self.tag == Tag.TUPLE and other.tag == Tag.UBITS:
      return self.tuple_members[other.get_bits_value()]
    raise TypeError('Cannot index values with tag:', self.tag, self)

  def update(self, index: 'Value', value: 'Value', span: Span) -> 'Value':
    if self.tag == Tag.ARRAY and index.tag == Tag.UBITS:
      return Value(
          Tag.ARRAY,
          self.array_payload.update(index.get_bits_value(), value, span))
    msg = 'Cannot update value; subject: {} index: {} value: {}'.format(
        self.tag, index.tag, value.tag)
    raise TypeError(msg)

  def slice(self, start: 'Value', length: 'Value', span: Span) -> 'Value':
    if self.tag == Tag.ARRAY and start.tag == Tag.UBITS:
      return Value(
          Tag.ARRAY,
          self.array_payload.slice(start.get_bits_value(),
                                   len(length.array_payload), span))
    raise TypeError('Cannot slice values with tag:', self.tag, self)

  def flatten(self) -> 'Value':
    if self.tag == Tag.ARRAY:
      return self.array_payload.flatten()
    if self.tag == Tag.UBITS:
      return self
    raise TypeError('Cannot flatten values with tag:', self.tag)

  def is_false(self) -> bool:
    if self.tag in (Tag.UBITS, Tag.SBITS):
      if self.bits_payload.bit_count != 1:
        raise ValueError(
            'Attempted to check boolean value of a multi-bit payload: {}'
            .format(self))
      return not bool(self.bits_payload.value)
    raise TypeError('Cannot query bool value with tag:', self.tag)

  def is_true(self) -> bool:
    if self.tag in (Tag.UBITS, Tag.SBITS):
      if self.bits_payload.bit_count != 1:
        raise ValueError(
            'Attempted to check boolean value of a multi-bit payload: {}'
            .format(self))
      return bool(self.bits_payload.value)
    raise TypeError('Cannot query bool value with tag:', self.tag)


class Nil(Value):
  """Helper type; represents empty tuple, often used as I/O result type."""

  def __init__(self):
    super(Nil, self).__init__(Tag.TUPLE, ())
