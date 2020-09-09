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

"""Tests for xls.dslx.scanner."""

import random
from typing import Text, Sequence, Any

from absl.testing import absltest
from xls.dslx.python import cpp_pos
from xls.dslx.python import cpp_scanner as scanner
from xls.dslx.python.cpp_scanner import TokenKind
from xls.dslx.python.cpp_scanner import TokenKindFromString


class ScannerTest(absltest.TestCase):

  def assertLen(self, sequence: Sequence[Any], target: int) -> None:
    assert len(sequence) == target, 'Got sequence of length %d, want %d' % (
        len(sequence), target)

  def make_scanner(self, text: Text, **kwargs) -> scanner.Scanner:
    return scanner.Scanner('<fake>', text, **kwargs)

  def test_scan_random(self):
    rng = random.Random(0)
    for _ in range(1024):
      chars = rng.randrange(512)
      text = ''.join(chr(rng.randrange(128)) for _ in range(chars))
      s = self.make_scanner(text)
      try:
        [str(t) for t in s.pop_all()]
      except scanner.ScanError:
        pass

  def test_scan_just_whitespace(self):
    s = self.make_scanner(' ')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertEqual(tokens[0].kind, TokenKind.EOF)

  def test_scan_keyword(self):
    s = self.make_scanner('fn')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertTrue(tokens[0].is_keyword(scanner.Keyword.FN))

  def test_function_definition(self):
    s = self.make_scanner('fn ident(x) { x }')
    tokens = s.pop_all()

    self.assertTrue(
        tokens[0].is_keyword(scanner.Keyword.FN), msg=repr(tokens[0]))
    self.assertEqual('fn', str(tokens[0]))
    self.assertTrue(tokens[1].is_identifier('ident'))
    self.assertEqual('ident', str(tokens[1]))
    self.assertEqual(tokens[2].kind, TokenKindFromString('('))
    self.assertEqual('(', str(tokens[2]))
    self.assertTrue(tokens[3].is_identifier('x'))
    self.assertEqual('x', str(tokens[3]))
    self.assertEqual(tokens[4].kind, TokenKindFromString(')'))
    self.assertEqual(')', str(tokens[4]))
    self.assertEqual(tokens[5].kind, TokenKindFromString('{'))
    self.assertEqual('{', str(tokens[5]))
    self.assertTrue(tokens[6].is_identifier('x'))
    self.assertEqual('x', str(tokens[6]))
    self.assertEqual(tokens[7].kind, TokenKindFromString('}'))
    self.assertEqual('}', str(tokens[7]))

  def test_doubled_up_plus(self):
    s = self.make_scanner('fn concat(x, y) { x ++ y }')
    tokens = s.pop_all()

    self.assertTrue(tokens[-4].is_identifier('x'))
    self.assertEqual(tokens[-3].kind, TokenKindFromString('++'))
    self.assertTrue(tokens[-2].is_identifier('y'))
    self.assertEqual(tokens[-1].kind, TokenKindFromString('}'))

  def test_number_hex(self):
    s = self.make_scanner('0xf00')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertTrue(tokens[0].is_number('0xf00'), msg=str(tokens))

  def test_negative_number_hex(self):
    s = self.make_scanner('-0xf00')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertTrue(tokens[0].is_number('-0xf00'), msg=str(tokens))

  def test_number_bin(self):
    s = self.make_scanner('0b10')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertTrue(tokens[0].is_number('0b10'), msg=str(tokens))

    with self.assertRaisesRegex(scanner.ScanError,
                                'Invalid digit for binary number: \'2\'') as cm:
      s = self.make_scanner('0b102')
      tokens = s.pop_all()

    self.assertIsInstance(cm.exception, scanner.ScanError)
    self.assertIsInstance(cm.exception, Exception)
    self.assertEqual(
        str(cm.exception),
        "ScanError: <fake>:1:5 Invalid digit for binary number: '2'")

  def test_negative_number_bin(self):
    s = self.make_scanner('-0b10')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertTrue(tokens[0].is_number('-0b10'), msg=str(tokens))

  def test_negative_number(self):
    s = self.make_scanner('-42')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertTrue(tokens[0].is_number('-42'), msg=str(tokens))

  def test_number_with_underscores(self):
    s = self.make_scanner('0b11_1100')
    tokens = s.pop_all()
    self.assertLen(tokens, 1)
    self.assertTrue(tokens[0].is_number('0b11_1100'), msg=str(tokens))

  def test_pos_lt(self):
    self.assertLess(cpp_pos.Pos('<fake>', 0, 0), cpp_pos.Pos('<fake>', 0, 1))
    self.assertLess(cpp_pos.Pos('<fake>', 0, 0), cpp_pos.Pos('<fake>', 1, 0))
    self.assertGreaterEqual(
        cpp_pos.Pos('<fake>', 0, 0), cpp_pos.Pos('<fake>', 0, 0))

  def test_scan_incomplete_number(self):
    with self.assertRaisesRegex(scanner.ScanError, 'Expected hex characters'):
      self.make_scanner('0x').pop_all()
    with self.assertRaisesRegex(scanner.ScanError,
                                'Expected binary characters'):
      self.make_scanner('0b').pop_all()

  def test_scan_incomplete_character(self):
    with self.assertRaisesRegex(scanner.ScanError,
                                'Expected closing single quote'):
      self.make_scanner("'a").pop_all()
    with self.assertRaisesRegex(scanner.ScanError,
                                'Expected character after single quote'):
      self.make_scanner("'").pop_all()

  def test_scan_in_whitespace_and_comments_mode(self):
    program = """// Hello comment world.
    42
  // EOF"""
    tokens = self.make_scanner(
        program, include_whitespace_and_comments=True).pop_all()
    self.assertLen(tokens, 5)
    self.assertEqual(tokens[0].kind, TokenKind.COMMENT)
    self.assertEqual(tokens[1].kind, TokenKind.WHITESPACE)
    self.assertEqual(tokens[2].kind, TokenKind.NUMBER)
    self.assertEqual(tokens[3].kind, TokenKind.WHITESPACE)
    self.assertEqual(tokens[4].kind, TokenKind.COMMENT)

  def test_peek_pop_drop_try_drop(self):
    text = '[!](-)'
    expected = [
        TokenKind.OBRACK,
        TokenKind.BANG,
        TokenKind.CBRACK,
        TokenKind.OPAREN,
        TokenKind.MINUS,
        TokenKind.CPAREN,
    ]
    s = self.make_scanner(text)
    for i, tk in enumerate(expected):
      self.assertFalse(s.at_eof())
      self.assertEqual(s.peek().kind, tk)
      if i % 2 == 0:
        t = s.pop()
        self.assertIsInstance(t, scanner.Token)
        self.assertEqual(t.kind, tk)
      else:
        self.assertTrue(s.try_pop(tk))
    self.assertTrue(s.at_eof())

if __name__ == '__main__':
  absltest.main()
