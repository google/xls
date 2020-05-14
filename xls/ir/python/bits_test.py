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

"""Tests for xls.ir.python.bits."""

from xls.ir.python import bits
from absl.testing import absltest


class BitsTest(absltest.TestCase):

  def test_bits(self):
    self.assertEqual(43, bits.UBits(43, 7).to_uint())
    self.assertEqual(53, bits.UBits(53, 7).to_int())
    self.assertEqual(33, bits.SBits(33, 8).to_uint())
    self.assertEqual(83, bits.SBits(83, 8).to_int())
    self.assertEqual(-83, bits.SBits(-83, 8).to_int())


if __name__ == '__main__':
  absltest.main()
