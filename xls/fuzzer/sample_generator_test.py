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

import collections
import random

from xls.common import test_base
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.fuzzer import ast_generator
from xls.fuzzer import sample_generator
from xls.fuzzer.sample import SampleOptions


class SampleGeneratorTest(test_base.TestCase):

  def test_randrange_biased_towards_zero(self):
    rng = random.Random(0)
    counter = collections.Counter()
    for _ in range(1024):
      counter[sample_generator.randrange_biased_towards_zero(16, rng)] += 1
    print(sorted(counter.items()))
    self.assertTrue(all(k < 16 for k in counter.keys()))
    self.assertTrue(all(counter[0] > counter[i] for i in range(1, 16)))

  def test_generate_empty_arguments(self):
    rng = random.Random(0)
    self.assertEqual(sample_generator.generate_arguments((), rng), ())

  def test_generate_single_bits_arguments(self):
    rng = random.Random(0)
    args = sample_generator.generate_arguments(
        (BitsType(signed=False, size=42),), rng)
    self.assertLen(args, 1)
    self.assertTrue(args[0].is_ubits())
    self.assertEqual(args[0].get_bit_count(), 42)

  def test_generate_mixed_bits_arguments(self):
    rng = random.Random(0)
    args = sample_generator.generate_arguments(
        (BitsType(signed=False, size=123), BitsType(signed=True, size=22)), rng)
    self.assertLen(args, 2)
    self.assertTrue(args[0].is_ubits())
    self.assertEqual(args[0].get_bit_count(), 123)
    self.assertTrue(args[1].is_sbits())
    self.assertEqual(args[1].get_bit_count(), 22)

  def test_generate_tuple_argument(self):
    rng = random.Random(0)
    args = sample_generator.generate_arguments((TupleType(
        (BitsType(signed=False, size=123),
         BitsType(signed=True, size=22))),), rng)
    self.assertLen(args, 1)
    self.assertTrue(args[0].is_tuple())
    self.assertEqual(args[0].get_elements()[0].get_bit_count(), 123)
    self.assertEqual(args[0].get_elements()[1].get_bit_count(), 22)

  def test_generate_array_argument(self):
    rng = random.Random(0)
    args = sample_generator.generate_arguments(
        (ArrayType(BitsType(signed=True, size=4), 24),), rng)
    self.assertLen(args, 1)
    self.assertTrue(args[0].is_array())
    self.assertLen(args[0].get_elements(), 24)
    self.assertTrue(args[0].index(0).is_sbits())
    self.assertTrue(args[0].index(0).get_bit_count(), 4)

  def test_generate_basic_sample(self):
    rng = random.Random(0)
    sample = sample_generator.generate_sample(
        rng,
        ast_generator.AstGeneratorOptions(),
        calls_per_sample=3,
        default_options=SampleOptions(
            convert_to_ir=True, optimize_ir=True, codegen=False,
            simulate=False))
    self.assertTrue(sample.options.input_is_dslx)
    self.assertTrue(sample.options.convert_to_ir)
    self.assertTrue(sample.options.optimize_ir)
    self.assertFalse(sample.options.codegen)
    self.assertFalse(sample.options.simulate)
    self.assertLen(sample.args_batch, 3)
    self.assertIn('fn main', sample.input_text)

  def test_generate_codegen_sample(self):
    rng = random.Random(0)
    sample = sample_generator.generate_sample(
        rng,
        ast_generator.AstGeneratorOptions(),
        calls_per_sample=0,
        default_options=SampleOptions(
            convert_to_ir=True, optimize_ir=True, codegen=True, simulate=True))
    self.assertTrue(sample.options.input_is_dslx)
    self.assertTrue(sample.options.convert_to_ir)
    self.assertTrue(sample.options.optimize_ir)
    self.assertTrue(sample.options.codegen)
    self.assertTrue(sample.options.simulate)
    self.assertNotEmpty(sample.options.codegen_args)
    self.assertEmpty(sample.args_batch)


if __name__ == '__main__':
  test_base.main()
