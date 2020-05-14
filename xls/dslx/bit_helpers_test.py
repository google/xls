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
#
# Lint as: python3
"""Tests for xls.dslx.bit_helpers."""

from xls.dslx import bit_helpers
from absl.testing import absltest


class BitHelpersTest(absltest.TestCase):

  def test_concat_bits(self):
    x = 0b10
    y = 0b011
    self.assertEqual(0b10011, bit_helpers.concat_bits(x, 2, y, 3))

  def test_to_positive_twos_complement(self):
    value = -1
    bit_count = 2
    self.assertEqual(0b11,
                     bit_helpers.to_positive_twos_complement(value, bit_count))

    value = -2
    bit_count = 3
    self.assertEqual(0b110,
                     bit_helpers.to_positive_twos_complement(value, bit_count))

  def test_join_bits(self):
    elements = [0b001, 0b011, 0b010]
    got = bit_helpers.join_bits(elements, element_bit_count=3)
    self.assertEqual(0b001011010, got)

  def test_bit_slice(self):
    original = 0b01010101
    bit_count = 8
    self.assertEqual(
        0b01, bit_helpers.bit_slice(original, bit_count, 0, 2, lsb_is_0=False))
    self.assertEqual(
        0b010, bit_helpers.bit_slice(original, bit_count, 0, 3, lsb_is_0=False))
    self.assertEqual(
        0b101, bit_helpers.bit_slice(original, bit_count, 5, 8, lsb_is_0=False))

  def test_to_zext_str_empty(self):
    self.assertEqual('', bit_helpers.to_zext_str(value=0, bit_count=0))

  def test_concat_bits_empty(self):
    # Empty both sides.
    self.assertEqual(0, bit_helpers.concat_bits(0, 0, 0, 0))
    # Empty RHS.
    self.assertEqual(3, bit_helpers.concat_bits(3, 2, 0, 0))
    # Empty LHS.
    self.assertEqual(3, bit_helpers.concat_bits(0, 0, 3, 2))

  def test_fits_in_bits(self):
    self.assertTrue(bit_helpers.fits_in_bits(value=0xf, bit_count=4))
    self.assertTrue(bit_helpers.fits_in_bits(value=0x7, bit_count=3))
    self.assertTrue(bit_helpers.fits_in_bits(value=0x3, bit_count=2))
    self.assertTrue(bit_helpers.fits_in_bits(value=0x1, bit_count=1))
    self.assertTrue(bit_helpers.fits_in_bits(value=0x0, bit_count=0))
    self.assertTrue(bit_helpers.fits_in_bits(value=0x0, bit_count=1))
    self.assertTrue(bit_helpers.fits_in_bits(value=0x0, bit_count=2))
    self.assertTrue(bit_helpers.fits_in_bits(value=0b110, bit_count=3))

    self.assertFalse(bit_helpers.fits_in_bits(value=0xf, bit_count=3))
    self.assertFalse(bit_helpers.fits_in_bits(value=0xf, bit_count=0))
    self.assertFalse(bit_helpers.fits_in_bits(value=0x7, bit_count=2))
    self.assertFalse(bit_helpers.fits_in_bits(value=0x3, bit_count=1))
    self.assertFalse(bit_helpers.fits_in_bits(value=0x1, bit_count=0))

    # Note: negative numbers that become too negative (large absolute value)
    # will not be expressible in a given bit count.
    self.assertTrue(bit_helpers.fits_in_bits(value=-1, bit_count=4))
    self.assertTrue(bit_helpers.fits_in_bits(value=-1, bit_count=3))
    self.assertTrue(bit_helpers.fits_in_bits(value=-1, bit_count=2))
    self.assertTrue(bit_helpers.fits_in_bits(value=-1, bit_count=1))

    self.assertTrue(bit_helpers.fits_in_bits(value=-1, bit_count=0))
    self.assertFalse(bit_helpers.fits_in_bits(value=-2, bit_count=1))
    self.assertFalse(bit_helpers.fits_in_bits(value=-3, bit_count=2))

    self.assertFalse(bit_helpers.fits_in_bits(value=0x100000000, bit_count=32))

  def test_from_twos_complement(self):
    self.assertEqual(-1,
                     bit_helpers.from_twos_complement(value=0xf, bit_count=4))
    self.assertEqual(-1,
                     bit_helpers.from_twos_complement(value=0x1f, bit_count=5))
    self.assertEqual(-2,
                     bit_helpers.from_twos_complement(value=0x1e, bit_count=5))
    self.assertEqual(15,
                     bit_helpers.from_twos_complement(value=0xf, bit_count=5))


if __name__ == '__main__':
  absltest.main()
