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

"""Helper routines dealing with bits (in arbitrary-width integers)."""

from typing import Iterable, Text, Optional, Tuple


def to_zext_str(value: int, bit_count: int) -> Text:
  """Pads 'value' from bin(value) with leading zeroes on the left hand side.

  Args:
    value: The value to convert to binary and left-pad.
    bit_count: The number of bits to left-pad out to.

  Returns:
    A string binary value with no leading '0b' prefix.
  """
  bits = bin(value)[2:2 + bit_count]
  assert len(bits) <= bit_count
  return '0' * (bit_count - len(bits)) + bits


def reverse(value: int, bit_count: int) -> int:
  return int(to_zext_str(value, bit_count)[::-1], 2) if bit_count else 0


def bit_slice(value: int, bit_count: int, start: int, limit: int,
              lsb_is_0: bool) -> int:
  """Slices bits out of the original value to create a new value.

  Args:
    value: The original value to slice bits out of.
    bit_count: Bit width of the original value.
    start: Inclusive start.
    limit: Exclusive limit.
    lsb_is_0: Whether a start of 0 means slicing from the lsb towards the msbs,
      or vice versa. lsb_is_0=True is like slicing a number (where we write the
      digits from a_{N-1} on the left to a_0 as the lsb on the right), whereas
      lsb_is_0=False is like slicing an array (where we write the array elements
      a_0 to a_{N-1} from left to right).

  Returns:
    The sliced value.
  """
  assert start >= 0, start
  assert limit >= 0, limit
  assert limit >= start, (start, limit)
  if limit == start:
    return 0
  subject = to_zext_str(value, bit_count)
  if lsb_is_0:
    subject = subject[::-1]
  result = subject[start:limit]
  if lsb_is_0:
    result = result[::-1]
  return int(result, 2)


def join_bits(elements: Iterable[int], element_bit_count: int) -> int:
  """Extends elements to the element_bit_count and concatenates them.

  Args:
    elements: Iterable of integral values that will be converted to binary.
    element_bit_count: The number of bits each element is zero-extended to
      before concatenation.

  Returns:
    The elements concatenated, after having been zero-extended to
    element_bit_count and then converted to an integer (most significant bit
    coming from the first element).
  """
  joined = ''.join(to_zext_str(e, element_bit_count) for e in elements)
  return int(joined, 2)


def concat_bits(x: int, x_bit_count: int, y: int, y_bit_count: int) -> int:
  """Concats x zext'd to x_bit_count with y zext'd to y_bit_count."""
  s = to_zext_str(x, x_bit_count) + to_zext_str(y, y_bit_count)
  if not s:
    return 0
  return int(s, 2)


def to_mask(bit_count: int) -> int:
  return (1 << bit_count) - 1


def fits_in_bits(value: int, bit_count: int) -> bool:
  """Returns whether value is a representable value in bit_count.

  Note that 0 and -1 are allowed for all bit_count, including zero, as they are
  often used as universal quantifier (all bits set / no bits set).

  Args:
    value: Value to consider.
    bit_count: Number of bits to hold a representation of value.

  Implementation note: consider a 2-bit two's complement value:

  ```
      bits    s2_val  u2_val
        10        -2       2
        11        -1       3
        00         0       0
        01         1       1
  ```

  We permit all values in the [min(s2_val), max(u2_val)] range, inclusive,
  i.e. -2 to +3.

  Similarly, for 3-bits:

  ```
    bits  s3_val  u3_val
     000       0       0
     001       1       1
     010       2       2
     011       3       3
     100      -4       4
     101      -3       5
     110      -2       6
     111      -1       7
  ```

  We permit all values in the [min(s3_val), max(u3_val)] range, inclusive.

  Therefore, we allow:
    * all values with abs(x) < -S2_MIN
    * the value U2_MAX
    * the value -1, which canonically means "all bits are set".
      * note that this is an exceptional case vs the above for a bit_count of
        0 for symmetry with 0 being an ok 0-bit number ("all bits set" vs "no
        bits set" as quantifiers).
  """
  unsigned_max = (1 << bit_count) - 1
  signed_min = -(unsigned_max - 1)
  return signed_min <= value <= unsigned_max or value in (0, -1)


def to_positive_twos_complement(value: int, bit_count: int) -> int:
  """Returns 'value' converted to two's complement within bit_count width.

  Args:
    value: Value to turn to two's complement.
    bit_count: The width in which we make the two's complement representation.
  Note: returned values are always non-negative. For example:  >>>
    bin(to_positive_twos_complement(-1, 4)) '0b1111'
  Implemenation note: in Python, masking is enough to give you this desired
    property.
  """
  assert fits_in_bits(value, bit_count), (
      'Value {0!r} (0x{0:x}, 0b{0:b}) did not fit in bit_count {1!r}'.format(
          value, bit_count))
  result = value & to_mask(bit_count)
  assert result >= 0
  return result


def from_twos_complement(value: int, bit_count: int) -> int:
  """Converts a 2's complement value into a Python integer."""
  assert value >= 0, value
  high_bit_set = bool(value & (1 << (bit_count - 1)))
  if high_bit_set:
    flipped = value ^ to_mask(bit_count)
    return -(flipped + 1)
  return value


def resolve_bit_slice_indices(bit_count: int, start: Optional[int],
                              limit: Optional[int]) -> Tuple[int, int]:
  """Returns (start, width), resolving indices via DSLX bit slice semantics."""
  assert bit_count >= 0, bit_count
  if start is None:
    start = 0
  if limit is None:
    limit = bit_count
  if start < 0:
    start += bit_count
  if limit < 0:
    limit += bit_count
  limit = min(max(limit, 0), bit_count)
  start = min(max(start, 0), bit_count, limit)
  assert start >= 0
  assert limit >= start, (start, limit)
  return (start, limit - start)


def resolve_width_slice(bit_count: int, start: int,
                        width: int) -> Tuple[int, int]:
  """Returns (start, width), resolving indices via DSLX width-slice semantics."""
  assert width >= 0, width
  if start < 0:
    start += bit_count
  start = min(max(0, start), bit_count)
  width = min(width, bit_count - start)
  assert start + width <= bit_count, (start, width, bit_count)
  return start, width


def to_hex_string(value: int, width: int) -> Text:
  """Returns a bare textual hex string representing the given value.

  e.g., "fffffffd" for a 32-bit -3, or 1e for a 5-bit 30 (both decimal).

  Args:
    value: The value to encode.
    width: The width of the encoding to generate.
  """
  return '{:x}'.format(value & to_mask(width))


def to_bits_string(value: int) -> str:
  """Converts unsigned value to a bit string with _ separators every nibble."""
  if value < 0:
    raise ValueError(f'Value is not unsigned: {value!r}')
  bits = bin(value)[2:]
  rev = bits[::-1]
  pieces = []
  i = 0
  while i < len(rev):
    pieces.append(rev[i:i + 4])
    i += 4
  return '0b' + '_'.join(pieces)[::-1]
