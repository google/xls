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

"""Base class for the parser that implements token peeking/popping."""

from typing import Text, Sequence, Union, Optional

from xls.dslx.parse_error import ParseError
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_scanner import Keyword
from xls.dslx.python.cpp_scanner import Scanner
from xls.dslx.python.cpp_scanner import Token
from xls.dslx.python.cpp_scanner import TokenKind


class TokenParser(object):
  """Single-token lookahead parser (base class) that wraps a scanner."""

  def __init__(self, scanner: Scanner):
    self._scanner = scanner
    self._lookahead = None

  def _get_pos(self) -> Pos:
    if self._lookahead:
      return self._lookahead.span.start
    return self._scanner.pos

  def _at_eof(self) -> bool:
    return self._scanner.at_eof()

  def _peekt(self) -> Token:
    """Returns token that has been peeked at (non-destructively) from stream."""
    if self._lookahead is None:
      self._lookahead = self._scanner.pop()
      assert self._lookahead is not None

    return self._lookahead

  def _popt(self) -> Token:
    """Returns a token that has been popped destructively from token stream."""
    tok = self._peekt()
    self._lookahead = None
    return tok

  def _dropt(self) -> None:
    """Wraps _popt() to signify popping a token without needing the value."""
    self._popt()

  def _peekt_is(self, target: TokenKind) -> bool:
    assert not isinstance(target, Keyword), \
        'Not a token kind: {!r}'.format(target)
    return self._peekt().kind == target

  def _peekt_is_keyword(self, target: Keyword) -> bool:
    return self._peekt().is_keyword(target)

  def _peekt_is_identifier(self, target: Text) -> bool:
    return self._peekt().is_identifier(target)

  def _peekt_in(self, targets: Sequence[Union[TokenKind, Keyword]]) -> bool:
    tok = self._peekt()
    for target in targets:
      if isinstance(target, TokenKind) and tok.kind == target:
        return True
      if isinstance(target, Keyword) and tok.is_keyword(target):
        return True
    return False

  def _try_popt(self, target: TokenKind) -> bool:
    tok = self._peekt()
    if tok.kind == target:
      self._dropt()
      return True
    return False

  def _try_pop_keyword(self, target: Keyword) -> bool:
    tok = self._peekt()
    if tok.is_keyword(target):
      self._dropt()
      return True
    return False

  def _try_pop_identifier_token(self, target: Text) -> Optional[Token]:
    tok = self._peekt()
    if tok.is_identifier(target):
      return self._popt()
    return None

  def _try_pop_identifier(self, target: Text) -> bool:
    return bool(self._try_pop_identifier_token(target))

  def _popt_or_error(self,
                     target: TokenKind,
                     start: Optional[Token] = None,
                     context: Optional[Text] = None) -> Token:
    """Pops a token of the target kind 'target' or raises a ParseError."""
    tok = self._peekt()
    if tok.kind == target:
      return self._popt()
    if start is None:
      msg = "Expected '{}', got '{}'".format(target.value, tok.to_error_str())
    else:
      msg = ("Expected '{}' for construct starting with '{}' @ {}, "
             "got '{}'").format(target.value, start.to_error_str(), start.span,
                                tok.to_error_str())
    if context:
      msg += ': ' + context
    raise ParseError(tok.span, msg)

  def _dropt_or_error(self,
                      target: TokenKind,
                      start: Optional[Token] = None,
                      context: Optional[Text] = None) -> None:
    """Just a wrapper around _popt_or_error that doesn't return the token.

    This helps to signify that the intent was to drop the token in caller code
    vs 'forgetting' to do something with a popped token.

    Args:
      target: The token kind that we want to pop from the token stream; if this
        is not the kind we observe from the pop, we raise an error.
      start: An optional 'start of the parsing construct' token for use in error
        messages; e.g. pointing out the start of an open paren when we're trying
        to pop a corresponding closing paren.
      context: Context string to be used in reporting an error.

    Raises:
      ParseError: When the popped token is not of kind 'target'.
    """
    self._popt_or_error(target, start=start, context=context)

  def _pop_keyword_or_error(self,
                            keyword: Keyword,
                            context: Optional[Text] = None) -> Token:
    """Pops a token of target keyword and returns it or raises a ParseError."""
    tok = self._popt()
    if tok.is_keyword(keyword):
      return tok
    msg = 'Expected keyword \'{}\', got {}'.format(keyword.value.lower(),
                                                   tok.to_error_str())
    if context:
      msg += ': ' + context
    raise ParseError(tok.span, msg)

  def _drop_keyword_or_error(self, keyword: Keyword) -> None:
    self._pop_keyword_or_error(keyword)
