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

# Lint as: python3
"""Data structures for positions / spans in source text."""

from typing import Text, Tuple

from xls.common.xls_error import XlsError


class Pos(object):
  """Notes a position (line/col) in the text being scanned."""

  def __init__(self, filename: Text, lineno: int, colno: int):
    self.filename = filename
    self.lineno = lineno
    self.colno = colno

  def __repr__(self) -> Text:
    return (f'Pos(filename={self.filename!r}, lineno={self.lineno}, '
            f'colno={self.colno})')

  def __lt__(self, other: 'Pos') -> bool:
    assert self.filename == other.filename
    return self.lineno < other.lineno or (self.lineno == other.lineno and
                                          self.colno < other.colno)

  def __ge__(self, other: 'Pos') -> bool:
    assert self.filename == other.filename
    return not self < other

  def __eq__(self, other: 'Pos') -> bool:
    return self.filename == other.filename and self.lineno == other.lineno and self.colno == other.colno

  def __ne__(self, other: 'Pos') -> bool:
    return not self.__eq__(other)

  def __str__(self) -> Text:
    """Returns the human readable strinigification.

    Humans generally think line 0 is called "line 1".
    """
    return '{}:{}:{}'.format(self.filename, self.lineno + 1, self.colno + 1)

  def as_tuple(self) -> Tuple[Text, int, int]:
    return (self.filename, self.lineno, self.colno)

  def bump_col(self) -> 'Pos':
    """Returns a new position with the column number bumped by 1."""
    return Pos(self.filename, self.lineno, self.colno + 1)


class Span(object):
  """Describes a contiguous span of text from one position to another."""

  def __init__(self, start: Pos, limit: Pos):
    assert limit >= start, ('Limit of span {!r} must be >= start {!r}'.format(
        limit, start))
    assert start.filename == limit.filename
    self.start = start
    self.limit = limit

  @property
  def filename(self):
    return self.start.filename

  def __repr__(self) -> Text:
    return 'Span(start={!r}, limit={!r})'.format(self.start, self.limit)

  def __str__(self) -> Text:
    assert self.start.filename == self.limit.filename
    if self.start.bump_col() == self.limit:
      return str(self.start)
    return '{}:{}:{}-{}:{}'.format(self.start.filename, self.start.lineno + 1,
                                   self.start.colno + 1, self.limit.lineno + 1,
                                   self.limit.colno + 1)

  def __eq__(self, other: 'Span') -> bool:
    return self.start == other.start and self.limit == other.limit

  def __ne__(self, other: 'Span') -> bool:
    return not self.__eq__(other)

  def as_tuple(self) -> Tuple[Tuple[Text, int, int], Tuple[Text, int, int]]:
    return (self.start.as_tuple(), self.limit.as_tuple())

  def update_limit(self, new_limit: Pos) -> 'Span':
    """Returns a new span value with the same start as self but new_limit."""
    return Span(self.start, new_limit)


class PositionalError(XlsError):
  """An XLS error that's associated with a span position in source text."""

  def __init__(self, message: Text, span: Span):
    super(PositionalError, self).__init__(message)
    self.span = span
    self.printed = False

  @property
  def filename(self):
    return self.span.filename

  @property
  def message(self) -> Text:
    return self.args[0]
