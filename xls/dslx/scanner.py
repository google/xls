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

"""Scanner for HLS syntax input.

Turns text (abstractly a sequence of characters) into a sequence of scanner
(lexed) tokens.
"""

import abc
import enum
import re
import string
from typing import List, Optional, Union, Callable, Text, Container

from xls.dslx import dslx_builtins
from xls.dslx.python.cpp_ast import Pos
from xls.dslx.python.cpp_ast import Span
from xls.dslx.span import PositionalError


class TokenKind(enum.Enum):
  """Represents the kinds of tokens we can yield from the scanning process."""
  DOT = '.'
  EOF = 'EOF'
  KEYWORD = 'keyword'
  IDENTIFIER = 'identifier'
  NUMBER = 'number'
  CHARACTER = 'character'
  OPAREN = '('
  CPAREN = ')'
  OBRACE = '{'
  CBRACE = '}'
  PLUS = '+'
  MINUS = '-'
  PLUS_COLON = '+:'
  TRIPLE_CANGLE = '>>>'
  DOUBLE_CANGLE = '>>'
  DOUBLE_OANGLE = '<<'
  EQUALS = '='
  DOUBLE_COLON = '::'
  DOUBLE_PLUS = '++'
  DOUBLE_EQUALS = '=='
  CANGLE_EQUALS = '>='
  OANGLE_EQUALS = '<='
  OANGLE = '<'
  CANGLE = '>'
  BANG_EQUALS = '!='
  BANG = '!'
  OBRACK = '['
  CBRACK = ']'
  COLON = ':'
  COMMA = ','
  DOUBLE_QUOTE = '"'
  STAR = '*'
  SLASH = '/'
  PERCENT = '%'
  ARROW = '->'
  SEMI = ';'
  AMPERSAND = '&'
  DOUBLE_AMPERSAND = '&&'
  BAR = '|'
  DOUBLE_BAR = '||'
  HAT = '^'
  FAT_ARROW = '=>'
  DOUBLE_DOT = '..'
  ELLIPSIS = '...'
  HASH = '#'
  # When in whitespace/comment mode; e.g. for syntax highlighting.
  WHITESPACE = 'whitespace'
  COMMENT = 'comment'


TYPE_KEYWORD_STRINGS = (['u%d' % i for i in range(1, 65)] +
                        ['s%d' % i for i in range(1, 65)] + [
                            'bits',
                            'uN',
                            'sN',
                            'bool',
                        ])
TYPE_KEYWORD_TUPS = [(type_keyword_string.upper(), type_keyword_string)
                     for type_keyword_string in TYPE_KEYWORD_STRINGS]

NON_TYPE_KEYWORD_STRINGS = [
    'as',
    'carry',
    'const',
    'else',
    'enum',
    'false',
    'fn',
    'for',
    'if',
    'import',
    'in',
    'let',
    'match',
    'next',
    'pub',
    'proc',
    'struct',
    'test',
    'true',
    'type',
    'while',
]
NON_TYPE_KEYWORD_TUPS = [
    (non_type_keyword_string.upper(), non_type_keyword_string)
    for non_type_keyword_string in NON_TYPE_KEYWORD_STRINGS
]

NAME_KEYWORD_TUPS = TYPE_KEYWORD_TUPS + NON_TYPE_KEYWORD_TUPS


class Keyword(enum.Enum):
  pass


# Here we use exec to work around a limitation in pytype where it doesn't seem
# to recognize the functional form of Enum() as creating a new type -- so
# instead we clobber the vacuous one created syntactically above with the
# functionally created one, so everybody is happy.
exec(u'Keyword = enum.Enum("Keyword", NAME_KEYWORD_TUPS)', globals())  # pylint: disable=exec-used

KEYWORDS = [t[1] for t in NAME_KEYWORD_TUPS]

TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS = {
    Keyword.BITS: (False, 1),
    Keyword.SN: (True, 1),
    Keyword.UN: (False, 1),
    Keyword.BOOL: (False, 1),
}
TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS.update({
    kv: (kv.value.startswith('s'), int(kv.value[1:]))
    for kv in list(Keyword)
    if re.match(r'[us]\d+', kv.value)
})
TYPE_KEYWORDS = tuple(
    getattr(Keyword, type_keyword_string.upper())
    for type_keyword_string in TYPE_KEYWORD_STRINGS)

SIMPLE_TOKEN_KINDS = '(){}[],"*/%;^'
DOUBLED_SIMPLE_TOKEN_KINDS = ':&|'


class ScanError(PositionalError):

  def __init__(self, pos: Pos, message: Text):
    super(ScanError, self).__init__(message, Span(pos, pos))
    self.pos = pos


class HighlightHandler(object):  # pytype: disable=ignored-metaclass
  """Strategy for highlighting token text in Token.to_highlight_str."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def handle_keyword(self, s: Text) -> Text:
    raise NotImplementedError

  @abc.abstractmethod
  def handle_number(self, s: Text) -> Text:
    raise NotImplementedError

  @abc.abstractmethod
  def handle_comment(self, s: Text) -> Text:
    raise NotImplementedError

  @abc.abstractmethod
  def handle_builtin(self, s: Text) -> Text:
    raise NotImplementedError

  @abc.abstractmethod
  def handle_type(self, s: Text) -> Text:
    raise NotImplementedError

  @abc.abstractmethod
  def handle_other(self, s: Text) -> Text:
    raise NotImplementedError


class Token(object):
  """Represents a token scanned from the character stream."""

  def __init__(self,
               kind: TokenKind,
               span: Span,
               value: Optional[Union[Text, int, Keyword]] = None):
    self.kind = kind
    self.span = span
    self.value = value

  def __repr__(self) -> Text:
    return 'Token(kind={!r}, span={}, value={!r})'.format(
        self.kind, self.span, self.value)

  def __str__(self) -> Text:
    if self.kind == TokenKind.KEYWORD:
      prefix = str(self.value).split('.', 1)[-1]
      if prefix[0:2] == 'UN':
        return 'uN' + prefix[2:]
      elif prefix[0:2] == 'SN':
        return 'sN' + prefix[2:]
      return str(self.value).split('.', 1)[-1].lower()
    if self.kind == TokenKind.IDENTIFIER:
      return str(self.value)
    if self.kind == TokenKind.NUMBER:
      return str(self.value)
    if self.kind == TokenKind.CHARACTER:
      return "'{}'".format(self.value) if self.value != "'" else "'\''"
    return str(self.kind).split('.', 1)[1]

  def to_highlight_str(self, handler: HighlightHandler) -> Text:
    """Converts this token into an ANSI-syntax-highlighted string."""
    if self.is_type_keyword():
      assert isinstance(self.value, Keyword), self.value
      return handler.handle_type(self.value.value)
    if self.kind == TokenKind.KEYWORD:
      assert isinstance(self.value, Keyword), self.value
      return handler.handle_keyword(self.value.value)
    if self.kind in (TokenKind.NUMBER, TokenKind.CHARACTER):
      return handler.handle_number(str(self))
    if self.kind == TokenKind.COMMENT:
      return handler.handle_comment('//{}\n'.format(self.value))
    if self.kind == TokenKind.IDENTIFIER and self.value in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
      return handler.handle_builtin(self.value)

    if all(c in string.punctuation for c in self.kind.value):
      return handler.handle_other(self.kind.value)
    if self.kind == TokenKind.WHITESPACE:
      return handler.handle_other(self.value)
    return handler.handle_other(str(self))

  def to_error_str(self) -> Text:
    """Returns the textual form of the token to be used in error string."""
    if self.kind == TokenKind.KEYWORD:
      assert isinstance(self.value, Keyword)
      return 'keyword:{}'.format(self.value.value.lower())
    return '{}'.format(self.kind.name.lower())

  def get_kind_or_keyword(self) -> Union[TokenKind, Keyword]:
    if self.kind == TokenKind.KEYWORD:
      assert isinstance(self.value, Keyword)
      return self.value
    return self.kind

  def is_kind(self, target: TokenKind) -> bool:
    return self.kind == target

  def is_keyword(self, target: Keyword) -> bool:
    return self.kind == TokenKind.KEYWORD and self.value == target

  def is_keyword_in(self, targets: Container[Keyword]) -> bool:
    return self.kind == TokenKind.KEYWORD and self.value in targets

  def is_kind_or_keyword(self, targets: Container[Union[Keyword, TokenKind]]):
    if self.kind == TokenKind.KEYWORD:
      return self.value in targets
    return self.kind in targets

  def is_type_keyword(self) -> bool:
    return self.kind == TokenKind.KEYWORD and self.value in TYPE_KEYWORDS

  def is_identifier(self, target: Text) -> bool:
    return self.kind == TokenKind.IDENTIFIER and self.value == target

  def is_number(self, target: Text) -> bool:
    return self.kind == TokenKind.NUMBER and self.value == target

  def get_value_as_int(self) -> int:
    """Extracts an integer value from the text of a number token."""
    assert self.kind == TokenKind.NUMBER
    if isinstance(self.value, int):
      return int(self.value)
    else:
      assert isinstance(self.value, str)
      return int(self.value.replace('_', ''), 0)


class Scanner(object):
  """Scans text input (stream of characters) into a stream of scanned tokens."""

  def __init__(self,
               filename: Text,
               text: Text,
               include_whitespace_and_comments: bool = False):
    self._filename = filename
    self._text = text
    self._index = 0
    self._lineno = 0
    self._colno = 0
    self._lookahead = None  # type: Optional[Token]
    self._include_whitespace_and_comments = include_whitespace_and_comments

  @property
  def pos(self) -> Pos:
    return Pos(self._filename, self._lineno, self._colno)

  def _peekc(self) -> Text:
    """Peeks at a character from the stream."""
    return self._text[self._index]

  def _peekc2(self) -> Optional[Text]:
    """Peeks at the second character in the character stream, if it exists.

    Returns:
      The second character in front of the character stream cursor, or None if
      no such character exists.
    """
    if self._index + 1 >= len(self._text):
      return None
    return self._text[self._index + 1]

  def _peekc_is(self, pred: Callable[[Text], bool]) -> bool:
    """Returns true iff lookahead character exists and satisfies pred."""
    if self.at_eof():
      return False
    return pred(self._peekc())

  def _peekc2_is(self, pred: Callable[[Text], bool]) -> bool:
    """Returns true iff 2nd lookahead character exists and satisfies pred."""
    c = self._peekc2()
    if c is None:
      return False
    return pred(c)

  def _popc(self) -> Text:
    """Pops a character from the stream."""
    assert not self._at_char_eof(), 'Cannot pop character when at EOF'
    c = self._peekc()
    self._index += 1
    if c == '\n':
      self._lineno += 1
      self._colno = 0
    else:
      self._colno += 1
    return c

  def _dropc(self, count: int = 1):
    """As _popc(), but does not return the character."""
    for _ in range(count):
      self._popc()

  def _try_dropc(self, target: Text) -> bool:
    """Returns true and drops a char if 'target' is at head of the stream."""
    if not self._at_char_eof() and self._peekc() == target:
      self._popc()
      return True
    return False

  def _scan_while(self, startc: Text,
                  take_while: Callable[[Text], bool]) -> Text:
    """Scans from current position until take_while is false or EOF is reached.

    Args:
      startc: The first character (that has been popped before this scan
        begins), will be the first character of the returned result string.
      take_while: A lambda that indicates when we should stop popping characters
        from the character stream (does not need to consider EOF). Returns the
        resulting characters from the scan, with startc at the front, as a
        string.

    Returns:
      The resulting characters from the scan, with startc at the front, as a
      string.
    """
    chars = [startc]
    while not self._at_char_eof():
      peek = self._peekc()
      if not take_while(peek):
        break
      chars.append(self._popc())
    return ''.join(chars)

  def _scan_identifier_or_keyword(self, startc: Text, start_pos: Pos) -> Token:
    """Scans the identifier-looking entity beginning with startc.

    Args:
      startc: first (already popped) character of the identiifer/keyword token.
      start_pos: start position for the identifier/keyword token.

    Returns:
      Either a keyword (if the scanned identifier turns out to be in the set of
      keywords) or an identifier token.
    """
    s = self._scan_while(
        startc, lambda c: c.isalpha() or c.isdigit() or c in '_!')
    span = Span(start_pos, self.pos)
    if s in KEYWORDS:
      return Token(TokenKind.KEYWORD, span, Keyword(s))
    return Token(TokenKind.IDENTIFIER, span, s)

  def _scan_char(self, start_pos: Pos) -> Token:
    """Scans a TokenKind.CHARACTER token."""
    open_quote = self._popc()
    assert open_quote == '\'', 'Must be called at starting quote.'
    if self._at_char_eof():
      raise ScanError(self.pos,
                      'Expected character after single quote, saw end of file')
    char = self._popc()
    if self._at_char_eof() or self._peekc() != '\'':
      raise ScanError(
          self.pos,
          'Expected closing single quote for character literal; got {!r}'
          .format('end of file' if self._at_char_eof() else self._peekc()))
    self._dropc()
    return Token(TokenKind.CHARACTER, Span(start_pos, self.pos), char)

  def _scan_number(self, startc: Text, start_pos: Pos) -> Token:
    """Scans a number token out of the character stream and returns it."""
    negative = startc == '-'
    if negative:
      startc = self._popc()

    if startc == '0' and self._try_dropc('x'):  # Hex prefix.
      s = self._scan_while(
          '0x',
          lambda c: '0' <= c <= '9' or 'a' <= c.lower() <= 'f' or c == '_')
      if s == '0x':
        raise ScanError(start_pos,
                        'Expected hex characters following 0x prefix.')
      if negative:
        s = '-' + s
      return Token(TokenKind.NUMBER, Span(start_pos, self.pos), s)

    if startc == '0' and self._try_dropc('b'):  # Bin prefix.
      s = self._scan_while('0b', lambda c: '0' <= c and c <= '1' or c == '_')
      if s == '0b':
        raise ScanError(start_pos,
                        'Expected binary characters following 0b prefix.')
      if not self.at_eof() and '0' <= self._peekc() <= '9':
        raise ScanError(
            self.pos,
            'Invalid digit for binary number: {}'.format(self._peekc()))
      if negative:
        s = '-' + s
      return Token(TokenKind.NUMBER, Span(start_pos, self.pos), s)

    s = self._scan_while(startc, lambda c: c.isdigit())
    assert s, 'Must have seen numerical values to attempt to scan a number.'
    if negative:
      s = '-' + s
    return Token(TokenKind.NUMBER, Span(start_pos, self.pos), s)

  def _at_whitespace(self) -> bool:
    return self._peekc() in u' \r\n\t\xa0'

  def _drop_comments_and_leading_whitespace(self):
    """Drops comments/whitespace from the current scan position."""
    while not self._at_char_eof():
      if self._at_whitespace():
        self._dropc()
      elif self._peekc() == '/' and self._peekc2() == '/':
        self._dropc(2)  # Get rid of leading '//'
        while not self._at_char_eof():
          if self._popc() == '\n':
            break
      else:
        break

  def at_eof(self) -> bool:
    """Returns whether scanner reached end of file, no more tokens.

    Note: when the character stream is exhausted but there is a lookahead token
    available, this returns False.
    """
    if self._lookahead:
      return False
    return self._at_char_eof()

  def _at_char_eof(self) -> bool:
    """Returns whether the input character stream has been exhausted."""
    assert self._index <= len(self._text)
    return self._index == len(self._text)

  def _pop_whitespace(self, start_pos: Pos) -> Token:
    assert self._at_whitespace()
    chars = []
    while not self._at_char_eof() and self._at_whitespace():
      chars.append(self._popc())
    return Token(
        TokenKind.WHITESPACE, Span(start_pos, self.pos), value=''.join(chars))

  def _pop_comment(self, start_pos: Pos) -> Token:
    chars = []
    while not self._at_char_eof() and not self._try_dropc('\n'):
      chars.append(self._popc())
    return Token(
        TokenKind.COMMENT, Span(start_pos, self.pos), value=''.join(chars))

  def _try_pop_whitespace_or_comment(self) -> Optional[Token]:
    """Attempts to pop a whitespace or a newline-delimited comment."""
    start_pos = self.pos
    if self._at_char_eof():
      return Token(TokenKind.EOF, Span(start_pos, start_pos))
    if self._at_whitespace():
      return self._pop_whitespace(start_pos)
    if self._peekc() == '/' and self._peekc2() == '/':
      self._dropc(2)
      return self._pop_comment(start_pos)
    else:
      return None

  def peek(self) -> Token:
    """Peeks at a scanned token at the head of the stream.

    Returns:
      The scanned token at the head of the stream. Note this may be an EOF token
      if the character stream is extinguished.

    Raises:
      ScanError: If an unknown character sequence is encountered (that cannot be
        converted into a token).

    This does not destructively update the scan state (i.e. the caller can peek
    at the same token again after this call returns).
    """
    if self._include_whitespace_and_comments:
      tok = self._try_pop_whitespace_or_comment()
      if tok:
        return tok
    else:
      self._drop_comments_and_leading_whitespace()

    # If there's a lookahead token already, we return that as the result of the
    # peek.
    if self._lookahead:
      return self._lookahead

    # Record the position the token starts at.
    start_pos = self.pos

    # Helper that makes a span from start_pos to the current point.
    mk_span = lambda: Span(start_pos, self.pos)

    # After dropping whitespace this may be EOF.
    if self._at_char_eof():
      return Token(TokenKind.EOF, mk_span())

    # Peek at one character for prefix scanning.
    startc = self._peekc()
    assert self._lookahead is None, self._lookahead
    if startc == '\'':
      lookahead = self._scan_char(start_pos)
    elif startc == '#':
      self._dropc()
      lookahead = Token(TokenKind('#'), mk_span())
    elif startc == '!':
      self._dropc()
      if self._try_dropc('='):
        lookahead = Token(TokenKind('!='), mk_span())
      else:
        lookahead = Token(TokenKind('!'), mk_span())
    elif startc == '=':
      self._dropc()
      if self._try_dropc('='):
        lookahead = Token(TokenKind('=='), mk_span())
      elif self._try_dropc('>'):
        lookahead = Token(TokenKind('=>'), mk_span())
      else:
        lookahead = Token(TokenKind('='), mk_span())
    elif startc in SIMPLE_TOKEN_KINDS:
      c = self._popc()
      assert startc == c
      lookahead = Token(TokenKind(c), mk_span())
    elif startc in DOUBLED_SIMPLE_TOKEN_KINDS:
      self._dropc()
      if self._try_dropc(startc):  # Doubled up.
        kind = TokenKind(startc * 2)
        lookahead = Token(kind, mk_span())
      else:  # Not doubled up.
        lookahead = Token(TokenKind(startc), mk_span())
    elif startc == '+':
      self._dropc()
      if self._try_dropc('+'):
        lookahead = Token(TokenKind('++'), mk_span())
      elif self._try_dropc(':'):
        lookahead = Token(TokenKind('+:'), mk_span())
      else:
        lookahead = Token(TokenKind('+'), mk_span())
    elif startc == '<':
      self._dropc()
      if self._try_dropc('<'):
        lookahead = Token(TokenKind('<<'), mk_span())
      elif self._try_dropc('='):
        lookahead = Token(TokenKind('<='), mk_span())
      else:
        lookahead = Token(TokenKind('<'), mk_span())
    elif startc == '>':
      self._dropc()
      if self._try_dropc('>'):
        if self._try_dropc('>'):
          lookahead = Token(TokenKind('>>>'), mk_span())
        else:
          lookahead = Token(TokenKind('>>'), mk_span())
      elif self._try_dropc('='):
        lookahead = Token(TokenKind('>='), mk_span())
      else:
        lookahead = Token(TokenKind('>'), mk_span())
    elif startc.isalpha() or startc == '_':
      lookahead = self._scan_identifier_or_keyword(self._popc(), start_pos)
    elif startc.isdigit(
    ) or startc == '-' and self._peekc2_is(lambda c: c.isdigit()):
      lookahead = self._scan_number(self._popc(), start_pos)
    elif startc == '-':
      self._dropc()
      if self._try_dropc('>'):  # '->' token
        lookahead = Token(TokenKind.ARROW, mk_span())
      else:  # Simply '-' token.
        lookahead = Token(TokenKind.MINUS, mk_span())
    elif startc == '.':
      self._dropc()
      if self._try_dropc('.'):
        if self._try_dropc('.'):
          lookahead = Token(TokenKind.ELLIPSIS, mk_span())
        else:
          lookahead = Token(TokenKind.DOUBLE_DOT, mk_span())
      else:
        lookahead = Token(TokenKind.DOT, mk_span())
    else:
      raise ScanError(start_pos, 'Unrecognized character: {!r}'.format(startc))
    assert lookahead is not None
    assert self._lookahead is None
    self._lookahead = lookahead
    return self._lookahead

  def pop(self) -> Token:
    result = self.peek()
    self._lookahead = None
    return result

  def try_pop(self, target: TokenKind) -> bool:
    """Attempts to pop a token of kind 'target', returns True on success."""
    peek_tok = self.peek()
    if peek_tok.kind == target:
      assert self._lookahead.kind == target
      self._lookahead = None
      return True
    return False

  def try_pop_keyword(self, target: Keyword) -> bool:
    """Attempts to pop a token of keyword 'target', returns True on success."""
    if not self.at_eof() and self.peek().is_keyword(target):
      tok = self.pop()
      assert tok.kind == TokenKind.KEYWORD and tok.value == target
      return True
    return False

  def pop_or_error(self, target: TokenKind) -> Token:
    """Returns a token of kind "target" or raises an error.

    Args:
      target: The token kind to pop from the head of the token stream.

    Raises:
      ScanError: When a token of kind 'target' is not at the head of the token
        stream.
    """
    if self.peek().kind == target:
      result = self.pop()
      assert result.kind == target, (result, target)
      return result
    raise ScanError(
        self.peek().span.start,
        'Expected {} token, found {}.'.format(target,
                                              self.peek().kind))

  def drop_or_error(self, target: TokenKind) -> None:
    """Same as pop_or_error but signifies in code that the token is unused."""
    self.pop_or_error(target)

  def pop_all(self) -> List[Token]:
    results = []
    while not self.at_eof():
      results.append(self.pop())
    return results
