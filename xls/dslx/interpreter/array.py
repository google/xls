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

"""Array data held in the "guts" of an array Value."""

from typing import Text, Any, Tuple, Optional

from xls.dslx.interpreter import errors
from xls.dslx.span import Span

Value = Any


class Array(object):
  """Represents an array of interpreter values (e.g. bits or other arrays)."""

  def __init__(self, elements: Tuple[Value, ...]):
    self.elements = elements

  def __str__(self) -> Text:
    return '[{}]'.format(', '.join(str(e) for e in self.elements))

  def to_human_str(self) -> Text:
    return '[{}]'.format(', '.join(e.to_human_str() for e in self.elements))

  def __repr__(self) -> Text:
    return 'Array(({},))'.format(', '.join(
        '{!r}'.format(e) for e in self.elements))

  def __len__(self) -> int:
    return len(self.elements)

  def __eq__(self, other: 'Array') -> bool:
    return len(self) == len(other) and all(
        s == o for s, o in zip(self.elements, other.elements))

  def __ne__(self, other: 'Array') -> bool:
    return not self.__eq__(other)

  def concat(self, other: 'Array') -> 'Array':
    return Array(self.elements + other.elements)

  def find_first_differing_index(self, other: 'Array') -> Optional[int]:
    for i, (s, o) in enumerate(zip(self.elements, other.elements)):
      if s.ne(o).get_bits_value():
        return i

  def index(self, i: int) -> Value:
    return self.elements[i]

  def update(self, i: int, v: Value, span: Span) -> 'Array':
    if i >= len(self.elements):
      raise errors.FailureError(
          span=span,
          message='Index {} out of bounds of array length {}.'.format(
              i, len(self.elements)))
    return Array(tuple(v if i == j else e for j, e in enumerate(self.elements)))

  def slice(self, start: int, length: int, span: Span) -> 'Array':
    if start + length > len(self.elements):
      raise errors.FailureError(
          span=span,
          message='Start {} with length {} out of bounds of array length {}.'
          .format(start, length, len(self.elements)))
    return Array(tuple(self.elements[start:start + length]))

  def flatten(self) -> Value:
    flattened_elements = [e.flatten() for e in self.elements]
    accum = flattened_elements[0]
    for e in flattened_elements[1:]:
      accum = accum.concat(e)
    return accum
