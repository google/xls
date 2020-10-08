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

"""Parses values from string text."""

from typing import Optional

from xls.common.xls_error import XlsError
from xls.dslx import ast_helpers
from xls.dslx.interpreter.value import Value
from xls.dslx.python import cpp_scanner as scanner_mod
from xls.dslx.python.cpp_scanner import Keyword
from xls.dslx.python.cpp_scanner import ScanError
from xls.dslx.python.cpp_scanner import Scanner
from xls.dslx.python.cpp_scanner import Token
from xls.dslx.python.cpp_scanner import TokenKind


class ValueParseError(XlsError):
  pass


def _bit_value_from_scanner(s: Scanner, signed: bool) -> Value:
  s.pop_or_error(TokenKind.OBRACK)
  bit_count_tok = s.pop_or_error(TokenKind.NUMBER)
  s.pop_or_error(TokenKind.CBRACK)
  s.pop_or_error(TokenKind.COLON)
  value_tok = s.pop_or_error(TokenKind.NUMBER)
  constructor = Value.make_sbits if signed else Value.make_ubits
  return constructor(
      bit_count=ast_helpers.get_token_value_as_int(bit_count_tok),
      value=ast_helpers.get_token_value_as_int(value_tok))


class _LookaheadWrapper:
  """Wraps a scanner with a token of lookahead capability.

  Lookahead must not be populated when this object is destroyed.
  """

  def __init__(self, scanner: Scanner):
    self.scanner = scanner
    self.lookahead = None

  def __del__(self):
    assert self.lookahead is None

  def _populate_lookahead(self):
    if self.lookahead is None:
      self.lookahead = self.scanner.pop()

  def _pop_lookahead(self) -> Token:
    assert self.lookahead is not None
    result = self.lookahead
    self.lookahead = None
    return result

  def pop(self) -> Token:
    self._populate_lookahead()
    return self._pop_lookahead()

  def pop_or_error(self, target: TokenKind) -> Token:
    tok = self.pop()
    if tok.kind != target:
      raise ValueParseError(f'Required {target} for value parsing; got: {tok}')
    return tok

  def try_pop(self, target: TokenKind) -> Optional[Token]:
    self._populate_lookahead()
    if self.lookahead.kind == target:
      return self._pop_lookahead()
    return None

  def try_pop_keyword(self, target: Keyword) -> Optional[Token]:
    self._populate_lookahead()
    if self.lookahead.is_keyword(target):
      return self._pop_lookahead()
    return None


def value_from_scanner(s: Scanner) -> Value:
  """Recursive call for converting a stream of tokens into a value."""
  s = _LookaheadWrapper(s)
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

  tok = s.pop()
  if tok.is_type_keyword():
    type_ = tok
    s.pop_or_error(TokenKind.COLON)
    value_tok = s.pop_or_error(TokenKind.NUMBER)
    signedness, bit_count = scanner_mod.TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS[
        type_.value]
    constructor = Value.make_sbits if signedness else Value.make_ubits
    return constructor(
        bit_count=bit_count,
        value=ast_helpers.get_token_value_as_int(value_tok))

  raise ScanError(tok.span.start,
                  'Unexpected token in value; found {}'.format(tok.kind))


def value_from_string(s: str) -> Value:
  scanner = Scanner('<text>', s)
  try:
    return value_from_scanner(scanner)
  except ScanError as e:
    raise ValueParseError('Could not parse value from string {!r}: {}'.format(
        s, e))
