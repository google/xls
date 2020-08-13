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

"""Tests for xls.dslx.interpreter.value."""

from xls.dslx.interpreter import value
from absl.testing import absltest


class ValueTest(absltest.TestCase):

  def test_bits_equivalence(self):
    a = value.Bits(value=4, bit_count=4)
    self.assertEqual(a, a)
    b = value.Bits(value=5, bit_count=4)
    self.assertEqual(b, b)
    self.assertNotEqual(a, b)

  def test_flatten_array_of_bits(self):
    a = value.Value.make_ubits(bit_count=12, value=0xf00)
    b = value.Value.make_ubits(bit_count=12, value=0xba5)
    array = value.Value.make_array((a, b))
    o = array.flatten()
    self.assertEqual(24, o.bits_payload.bit_count)
    self.assertEqual(0xf00ba5, o.bits_payload.value)

  def test_bitwise_negate(self):
    v = value.Value.make_ubits(3, 0x7)
    expected = value.Value.make_ubits(3, 0)
    self.assertEqual(v.bitwise_negate(), expected)

    v = value.Value.make_ubits(3, 0x6)
    expected = value.Value.make_ubits(3, 1)
    self.assertEqual(v.bitwise_negate(), expected)

    v = value.Value.make_ubits(3, 0x5)
    expected = value.Value.make_ubits(3, 0x2)
    self.assertEqual(v.bitwise_negate(), expected)

  def test_less_than(self):
    uf = value.Value.make_ubits(4, 0xf)
    sf = value.Value.make_sbits(4, 0xf)
    uzero = value.Value.make_ubits(4, 0)
    szero = value.Value.make_sbits(4, 0)
    true = value.Value.make_bool(True)
    false = value.Value.make_bool(False)
    self.assertEqual(true, uf.gt(uzero))
    self.assertEqual(false, uf.lt(uzero))
    self.assertEqual(false, sf.gt(szero))
    self.assertEqual(true, sf.lt(szero))

  def test_negate(self):
    uone = value.Value.make_ubits(4, 1)
    uf = value.Value.make_ubits(4, 0xf)
    self.assertEqual(uone.arithmetic_negate(), uf)

    sone = value.Value.make_sbits(4, 1)
    sf = value.Value.make_sbits(4, 0xf)
    self.assertEqual(sone.arithmetic_negate(), sf)

  def _sample_ops(self, x: value.Value) -> value.Value:
    """Runs a string of Value operations against the given value."""
    return x.shrl(x).bitwise_xor(x).shra(x).bitwise_or(x).bitwise_and(x)\
        .bitwise_negate().arithmetic_negate().sub(x)

  def test_sample_ops(self):
    uone = value.Value.make_ubits(4, value=5)
    uzero = value.Value.make_ubits(4, value=1)
    self.assertEqual(uzero, self._sample_ops(uone))

    sone = value.Value.make_sbits(4, value=5)
    szero = value.Value.make_sbits(4, value=1)
    self.assertEqual(szero, self._sample_ops(sone))

  def test_array_of_u32_human_str(self):
    elements = (value.Value.make_u32(2), value.Value.make_u32(3),
                value.Value.make_u32(4))
    array = value.Value.make_array(elements)
    self.assertEqual(array.to_human_str(), '[2, 3, 4]')


if __name__ == '__main__':
  absltest.main()
