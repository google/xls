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

"""Tests for xls.ir.python.number_parser."""

from xls.ir.python import number_parser
from absl.testing import absltest


class NumberParserTest(absltest.TestCase):

  def test_number_parser(self):
    self.assertEqual(15, number_parser.bits_from_string('0xf').to_uint())


if __name__ == '__main__':
  absltest.main()
