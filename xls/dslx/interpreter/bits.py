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

"""Representation of a (immutable) bits object."""

import operator

from typing import Text, Callable, Iterable, Union, Tuple

from xls.dslx import bit_helpers
from xls.dslx.bit_helpers import bit_slice
from xls.dslx.bit_helpers import concat_bits
from xls.dslx.bit_helpers import to_positive_twos_complement


class Bits(object):
  """The internal payload of a Value that is a vector of bits.

  Values are operated on in two's complement for the operators presented as
  methods.
  """

  def __init__(self, bit_count: int, value: int):
    """Constructs a fixed-width-bit-count object holding the value "value".

    Args:
      bit_count: The width of the value to be represented, in bits.
      value: The value this instance should represent. The value is converted to
        two's complement form and truncated into the provided bit_count.
    """
    self.bit_count = bit_count
    self.value = to_positive_twos_complement(value=value, bit_count=bit_count)

  def __bool__(self):
    return bool(self.value)

  def __len__(self):
    return self.__bool__()

  @property
  def signed_value(self) -> int:
    return self.from_twos_complement()

  def __repr__(self) -> Text:
    return 'Bits(bit_count={!r}, value={!r})'.format(self.bit_count, self.value)

  def __str__(self) -> Text:
    return 'bits[{}]:{:#_x}'.format(self.bit_count, self.value)

  def to_human_str(self) -> Text:
    return str(self.value)

  def __eq__(self, other: 'Bits') -> bool:
    return self.bit_count == other.bit_count and self.value == other.value

  def __ge__(self, other: 'Bits') -> bool:
    return bool(self.uge(other))

  def __add__(self, other: 'Bits') -> 'Bits':
    return self.add(other)

  def __ne__(self, other: 'Bits') -> bool:
    return not self.__eq__(other)

  def __rshift__(self, other: int) -> 'Bits':
    return Bits(bit_count=self.bit_count, value=self.value >> other)

  def __and__(self, other: int) -> 'Bits':
    return Bits(bit_count=self.bit_count, value=self.value & other)

  def from_twos_complement(self) -> int:
    return bit_helpers.from_twos_complement(
        value=self.value, bit_count=self.bit_count)

  def get_mask(self) -> int:
    return bit_helpers.to_mask(self.bit_count)

  def slice(self, start: int, limit: int, lsb_is_0: bool) -> 'Bits':
    assert start >= 0, start
    assert limit >= 0, limit
    assert limit >= start, (start, limit)
    return Bits(
        limit - start,
        bit_slice(self.value, self.bit_count, start, limit, lsb_is_0=lsb_is_0))

  def reverse(self) -> 'Bits':
    return Bits(
        bit_count=self.bit_count,
        value=bit_helpers.reverse(self.value, self.bit_count))

  def get_lsb_slice(self, count: int) -> 'Bits':
    return Bits(bit_count=count, value=self.value & bit_helpers.to_mask(count))

  def get_lsb_index(self, i: int) -> 'Bits':
    return Bits(bit_count=1, value=(self.value >> i) & 0x1)

  def get_sign_bit(self) -> bool:
    if not self.bit_count:
      return False
    return bool((self.value >> (self.bit_count - 1)) & 1)

  def concat(self, other: 'Bits') -> 'Bits':
    """Returns a value that concatenates bits in self with bits in other."""
    return Bits(
        self.bit_count + other.bit_count,
        concat_bits(self.value, self.bit_count, other.value, other.bit_count))

  @classmethod
  def concat_all(cls, xs: Iterable['Bits']) -> 'Bits':
    i = iter(xs)
    accum = next(i)
    for x in i:
      accum = accum.concat(x)
    return accum

  def _same_width_helper(self, other: 'Bits', op: Callable[[int, int],
                                                           int]) -> 'Bits':
    if self.bit_count != other.bit_count:
      raise TypeError(
          'Cannot {} different bit-count "bit" values'.format(op.__name__),
          self, other)
    new_value = op(self.value, other.value)
    return Bits(self.bit_count, new_value & self.get_mask())

  def add(self, other: 'Bits') -> 'Bits':
    """Adds in unsigned arithmetic (values wrap in bit count space)."""
    return self._same_width_helper(other, operator.add)

  def umul(self, other: 'Bits') -> 'Bits':
    return Bits(self.bit_count + other.bit_count, self.value * other.value)

  def smul(self, other: 'Bits') -> 'Bits':
    return Bits(self.bit_count + other.bit_count,
                self.from_twos_complement() * other.from_twos_complement())

  def add_with_carry(self, other: 'Bits') -> Tuple['Bits', 'Bits']:
    """Adds in unsigned arithmetic."""
    lhs_padded = Bits(self.bit_count + 1, self.value)
    rhs_padded = Bits(other.bit_count + 1, other.value)
    result = lhs_padded._same_width_helper(  # pylint: disable=protected-access
        rhs_padded, operator.add)
    return (result.slice(0, 1, lsb_is_0=False),
            result.slice(1, result.bit_count, lsb_is_0=False))

  def sub(self, other: 'Bits') -> 'Bits':
    """Subs in unsigned arithmetic (values wrap in bit count space)."""
    return self._same_width_helper(other, operator.sub)

  def mul(self, other: 'Bits') -> 'Bits':
    if self.bit_count != other.bit_count:
      raise TypeError('Cannot mul different bit-count "bit" values', self,
                      other)

    def same_width_mul(x: int, y: int):
      return (x * y) & self.get_mask()

    return self._same_width_helper(other, same_width_mul)

  def bitwise_or(self, other: 'Bits') -> 'Bits':
    return self._same_width_helper(other, operator.or_)

  def __or__(self, other: 'Bits') -> 'Bits':
    return self.bitwise_or(other)

  def bitwise_xor(self, other: 'Bits') -> 'Bits':
    return self._same_width_helper(other, operator.xor)

  def bitwise_and(self, other: 'Bits') -> 'Bits':
    return self._same_width_helper(other, operator.and_)

  def bitwise_negate(self) -> 'Bits':
    return Bits(self.bit_count, self.value ^ self.get_mask())

  def shll(self, other: 'Bits') -> 'Bits':

    def lshift_in_width(lhs: int, rhs: int) -> int:
      # Normal Python lshift will create giant values.
      if rhs >= self.bit_count:
        return 0
      return (lhs << rhs) & self.get_mask()

    return self._same_width_helper(other, lshift_in_width)

  def shrl(self, other: 'Bits') -> 'Bits':

    def rshift_in_width(lhs: int, rhs: int) -> int:
      # Normal Python rshift will result in overflow errors.
      if rhs >= self.bit_count:
        return 0
      return (lhs >> rhs) & self.get_mask()

    return self._same_width_helper(other, rshift_in_width)

  def shra(self, other: 'Bits') -> 'Bits':
    """Implements shift-right-arithmetic for bit values."""

    def do_shra(lhs: int, rhs: int):
      if (lhs >> (self.bit_count - 1)) & 1:
        if rhs >= self.bit_count:
          return self.get_mask()
        return (lhs >> rhs) | (
            bit_helpers.to_mask(rhs) << (self.bit_count - rhs))
      else:
        if rhs >= self.bit_count:
          return 0
        return lhs >> rhs

    return self._same_width_helper(other, do_shra)

  def floordiv(self, other: 'Bits') -> 'Bits':
    return self._same_width_helper(other, operator.floordiv)

  def _cmp_helper(self, other: Union['Bits', int], op: Callable[[int, int],
                                                                int]) -> 'Bits':
    if isinstance(other, int):
      other = Bits(value=other, bit_count=self.bit_count)
    if self.bit_count != other.bit_count:
      raise TypeError('Cannot compare different bit-count "bit" values', self,
                      other)
    return Bits(1, int(op(self.value, other.value)))

  def _scmp_helper(self, other: Union['Bits', int],
                   op: Callable[[int, int], int]) -> 'Bits':
    if isinstance(other, int):
      other = Bits(value=other, bit_count=self.bit_count)
    if self.bit_count != other.bit_count:
      raise TypeError('Cannot compare different bit-count "bit" values', self,
                      other)
    return Bits(1, int(op(self.signed_value, other.signed_value)))

  def ne(self, other: 'Bits') -> 'Bits':
    return self._cmp_helper(other, operator.ne)

  def eq(self, other: 'Bits') -> 'Bits':
    return self._cmp_helper(other, operator.eq)

  def ult(self, other: 'Bits') -> 'Bits':
    return self._cmp_helper(other, operator.lt)

  def ugt(self, other: 'Bits') -> 'Bits':
    return self._cmp_helper(other, operator.gt)

  def uge(self, other: 'Bits') -> 'Bits':
    return self._cmp_helper(other, operator.ge)

  def ule(self, other: 'Bits') -> 'Bits':
    return self._cmp_helper(other, operator.le)

  def slt(self, other: 'Bits') -> 'Bits':
    return self._scmp_helper(other, operator.lt)

  def sgt(self, other: 'Bits') -> 'Bits':
    return self._scmp_helper(other, operator.gt)

  def sge(self, other: 'Bits') -> 'Bits':
    return self._scmp_helper(other, operator.ge)

  def sle(self, other: 'Bits') -> 'Bits':
    return self._scmp_helper(other, operator.le)

  def zero_ext(self, new_bit_count: int) -> 'Bits':
    return Bits(
        bit_count=new_bit_count,
        value=self.value & bit_helpers.to_mask(new_bit_count))

  def sign_ext(self, new_bit_count: int) -> 'Bits':
    value = self.value
    if self.get_sign_bit() and new_bit_count >= self.bit_count:
      value |= bit_helpers.to_mask(new_bit_count -
                                   self.bit_count) << self.bit_count
    return Bits(
        bit_count=new_bit_count,
        value=value & bit_helpers.to_mask(new_bit_count))
