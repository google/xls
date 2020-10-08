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

"""Tests for xls.dslx.interpreter.bits."""

from xls.dslx.interpreter.bits import Bits
from absl.testing import absltest


class BitsTest(absltest.TestCase):

  def test_sign_ext(self):
    a = Bits(bit_count=0, value=0b0)
    self.assertEqual(Bits(bit_count=2, value=0b0), a.sign_ext(new_bit_count=2))

    b = Bits(bit_count=2, value=0b01)
    self.assertEqual(Bits(bit_count=2, value=0b01), b.sign_ext(new_bit_count=2))
    self.assertEqual(Bits(bit_count=3, value=0b01), b.sign_ext(new_bit_count=3))
    self.assertEqual(Bits(bit_count=4, value=0b01), b.sign_ext(new_bit_count=4))

    c = Bits(bit_count=2, value=0b10)
    self.assertEqual(
        Bits(bit_count=3, value=0b110), c.sign_ext(new_bit_count=3))
    self.assertEqual(
        Bits(bit_count=4, value=0b1110), c.sign_ext(new_bit_count=4))


if __name__ == '__main__':
  absltest.main()
