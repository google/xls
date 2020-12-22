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

"""Tests for xls.dslx.fuzzer.ast_generator."""

import random

from xls.common import test_base
from xls.dslx import ast_helpers
from xls.dslx import fakefs_test_util
from xls.dslx import parser_helpers
from xls.dslx.fuzzer import ast_generator
from xls.dslx.python import cpp_scanner as scanner
from xls.dslx.python import cpp_typecheck
from xls.dslx.python.import_routines import ImportCache
from xls.dslx.span import PositionalError


class AstGeneratorTest(test_base.TestCase):

  def test_generates_valid_functions(self):
    g = ast_generator.AstGenerator(
        random.Random(0), ast_generator.AstGeneratorOptions())
    for i in range(32):
      print('Generating sample', i)
      _, m = g.generate_function_in_module('main', 'test')
      text = str(m)
      filename = '/fake/test_sample.x'
      import_cache = ImportCache()
      additional_search_paths = ()
      with fakefs_test_util.scoped_fakefs(filename, text):
        try:
          module = parser_helpers.parse_text(
              text, name='test_sample', print_on_error=True, filename=filename)
          cpp_typecheck.check_module(module, import_cache,
                                     additional_search_paths)
        except PositionalError as e:
          parser_helpers.pprint_positional_error(e)
          raise

  def _test_repeatable(self, seed: int) -> None:

    def generate_one(g):
      _, m = g.generate_function_in_module('main', 'test')
      text = str(m)
      return text

    count_to_generate = 128
    g = ast_generator.AstGenerator(
        random.Random(seed), ast_generator.AstGeneratorOptions())
    fs = [generate_one(g) for _ in range(count_to_generate)]
    g.reset(seed=seed)
    gs = [generate_one(g) for _ in range(count_to_generate)]
    self.assertEqual(fs, gs)

    # Create a fresh generator as well.
    g = ast_generator.AstGenerator(
        random.Random(seed), ast_generator.AstGeneratorOptions())
    hs = [generate_one(g) for _ in range(count_to_generate)]
    self.assertEqual(gs, hs)

  def test_repeatable_seed0(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    self._test_repeatable(seed=0)

  def test_repeatable_seed1(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    self._test_repeatable(seed=1)

  def test_types(self):
    g = ast_generator.AstGenerator(
        random.Random(0), ast_generator.AstGeneratorOptions())
    _, m = g.generate_function_in_module('main', 'test')

    u1 = g._make_type_annotation(signed=False, width=1)
    self.assertEqual(g._get_type_bit_count(u1), 1)
    self.assertTrue(g._is_bit_vector(u1))
    self.assertTrue(g._is_unsigned_bit_vector(u1))
    self.assertFalse(g._is_array(u1))
    self.assertFalse(g._is_tuple(u1))

    s1 = g._make_type_annotation(signed=True, width=1)
    self.assertTrue(g._is_bit_vector(s1))
    self.assertFalse(g._is_unsigned_bit_vector(s1))
    self.assertEqual(g._get_type_bit_count(s1), 1)
    self.assertFalse(g._is_array(s1))
    self.assertFalse(g._is_tuple(s1))

    u42 = g._make_type_annotation(signed=False, width=42)
    self.assertEqual(g._get_type_bit_count(u42), 42)
    self.assertTrue(g._is_bit_vector(u42))
    self.assertTrue(g._is_unsigned_bit_vector(u42))
    self.assertFalse(g._is_array(u42))
    self.assertFalse(g._is_tuple(u42))

    empty_tuple = g._make_tuple_type(())
    self.assertEqual(g._get_type_bit_count(empty_tuple), 0)
    self.assertFalse(g._is_bit_vector(empty_tuple))
    self.assertFalse(g._is_unsigned_bit_vector(empty_tuple))
    self.assertFalse(g._is_array(empty_tuple))
    self.assertTrue(g._is_tuple(empty_tuple))

    tple = g._make_tuple_type((empty_tuple, s1, u1, u42, u42))
    self.assertEqual(g._get_type_bit_count(tple), 86)
    self.assertFalse(g._is_bit_vector(tple))
    self.assertFalse(g._is_unsigned_bit_vector(tple))
    self.assertFalse(g._is_array(tple))
    self.assertTrue(g._is_tuple(tple))

    nested_tuple = g._make_tuple_type((tple, u42, tple))
    self.assertEqual(g._get_type_bit_count(nested_tuple), 214)
    self.assertFalse(g._is_bit_vector(nested_tuple))
    self.assertFalse(g._is_unsigned_bit_vector(nested_tuple))
    self.assertFalse(g._is_array(nested_tuple))
    self.assertTrue(g._is_tuple(nested_tuple))

    array_of_u42 = g._make_array_type(u42, 100)
    self.assertEqual(g._get_type_bit_count(array_of_u42), 4200)
    self.assertFalse(g._is_bit_vector(array_of_u42))
    self.assertFalse(g._is_unsigned_bit_vector(array_of_u42))
    self.assertEqual(g._get_array_size(array_of_u42), 100)
    self.assertTrue(g._is_array(array_of_u42))
    self.assertFalse(g._is_tuple(array_of_u42))

    array_of_tuple = g._make_array_type(nested_tuple, 10)
    self.assertEqual(g._get_type_bit_count(array_of_tuple), 2140)
    self.assertFalse(g._is_bit_vector(array_of_tuple))
    self.assertFalse(g._is_unsigned_bit_vector(array_of_tuple))
    self.assertEqual(g._get_array_size(array_of_tuple), 10)
    self.assertTrue(g._is_array(array_of_tuple))
    self.assertFalse(g._is_tuple(array_of_tuple))

    u11_token = scanner.Token(g.fake_span, scanner.KeywordFromString('u11'))
    u11 = ast_helpers.make_builtin_type_annotation(
        m, g.fake_span, u11_token, dims=())
    self.assertEqual(g._get_type_bit_count(u11), 11)
    self.assertTrue(g._is_bit_vector(u11))
    self.assertTrue(g._is_unsigned_bit_vector(u11))
    self.assertFalse(g._is_array(u11))
    self.assertFalse(g._is_tuple(u11))

    un_token = scanner.Token(g.fake_span, scanner.KeywordFromString('uN'))
    un1234 = ast_helpers.make_builtin_type_annotation(
        m, g.fake_span, un_token, dims=(g._make_number(1234, None),))
    self.assertEqual(g._get_type_bit_count(un1234), 1234)
    self.assertTrue(g._is_bit_vector(un1234))
    self.assertTrue(g._is_unsigned_bit_vector(un1234))
    self.assertFalse(g._is_array(un1234))
    self.assertFalse(g._is_tuple(un1234))

    sn_token = scanner.Token(g.fake_span, scanner.KeywordFromString('sN'))
    sn1234 = ast_helpers.make_builtin_type_annotation(
        m, g.fake_span, sn_token, dims=(g._make_number(1234, None),))
    self.assertEqual(g._get_type_bit_count(sn1234), 1234)
    self.assertTrue(g._is_bit_vector(sn1234))
    self.assertFalse(g._is_unsigned_bit_vector(sn1234))
    self.assertFalse(g._is_array(sn1234))
    self.assertFalse(g._is_tuple(sn1234))

    bits_token = scanner.Token(g.fake_span, scanner.KeywordFromString('bits'))
    bits1234 = ast_helpers.make_builtin_type_annotation(
        m, g.fake_span, bits_token, dims=(g._make_number(1234, None),))
    self.assertEqual(g._get_type_bit_count(bits1234), 1234)
    self.assertTrue(g._is_bit_vector(bits1234))
    self.assertTrue(g._is_unsigned_bit_vector(bits1234))
    self.assertFalse(g._is_array(bits1234))
    self.assertFalse(g._is_tuple(bits1234))

    un1234_10 = ast_helpers.make_builtin_type_annotation(
        m,
        g.fake_span,
        un_token,
        dims=(g._make_number(1234, None), g._make_number(10, None)))
    self.assertEqual(g._get_type_bit_count(un1234_10), 12340)
    self.assertFalse(g._is_bit_vector(un1234_10))
    self.assertFalse(g._is_unsigned_bit_vector(un1234_10))
    self.assertTrue(g._is_array(un1234_10))
    self.assertFalse(g._is_tuple(un1234_10))


if __name__ == '__main__':
  test_base.main()
