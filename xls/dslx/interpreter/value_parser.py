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

"""Parses values from string text."""

from typing import Text

from xls.common.xls_error import XlsError
from xls.dslx import scanner as scanner_mod
from xls.dslx.interpreter.value import Value
from xls.dslx.scanner import Keyword
from xls.dslx.scanner import ScanError
from xls.dslx.scanner import Scanner
from xls.dslx.scanner import TokenKind


class ValueParseError(XlsError):
  pass


def _bit_value_from_scanner(s: Scanner, signed: bool) -> Value:
  s.drop_or_error(TokenKind.OBRACK)
  bit_count_tok = s.pop_or_error(TokenKind.NUMBER)
  s.drop_or_error(TokenKind.CBRACK)
  s.drop_or_error(TokenKind.COLON)
  value_tok = s.pop_or_error(TokenKind.NUMBER)
  constructor = Value.make_sbits if signed else Value.make_ubits
  return constructor(
      bit_count=bit_count_tok.get_value_as_int(),
      value=value_tok.get_value_as_int())


def value_from_scanner(s: Scanner) -> Value:
  """Recursive call for converting a stream of tokens into a value."""
  if s.try_pop(TokenKind.OPAREN):
    elements = []
    must_end = False
    while True:
      if s.try_pop(TokenKind.CPAREN):
        break
      if must_end:
        s.pop_or_error(TokenKind.CPAREN)
        break
      elements.append(value_from_scanner(s))
      must_end = not s.try_pop(TokenKind.COMMA)
    return Value.make_tuple(tuple(elements))

  if s.try_pop(TokenKind.OBRACK):
    elements = []
    must_end = False
    while True:
      if s.try_pop(TokenKind.CBRACK):
        break
      if must_end:
        s.pop_or_error(TokenKind.CBRACK)
        break
      elements.append(value_from_scanner(s))
      must_end = not s.try_pop(TokenKind.COMMA)
    return Value.make_array(tuple(elements))

  if s.try_pop_keyword(Keyword.BITS) or s.try_pop_keyword(Keyword.UN):
    return _bit_value_from_scanner(s, signed=False)
  if s.try_pop_keyword(Keyword.BITS) or s.try_pop_keyword(Keyword.SN):
    return _bit_value_from_scanner(s, signed=True)

  if s.peek().is_type_keyword():
    type_ = s.pop()
    s.drop_or_error(TokenKind.COLON)
    value_tok = s.pop_or_error(TokenKind.NUMBER)
    signedness, bit_count = scanner_mod.TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS[
        type_.value]
    constructor = Value.make_sbits if signedness else Value.make_ubits
    return constructor(bit_count=bit_count, value=value_tok.get_value_as_int())

  raise ScanError(s.peek().span.start,
                  'Unexpected token in value; found {}'.format(s.peek().kind))


def value_from_string(s: Text) -> Value:
  scanner = Scanner('<text>', s)
  try:
    return value_from_scanner(scanner)
  except ScanError as e:
    raise ValueParseError('Could not parse value from string {!r}: {}'.format(
        s, e))
