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

"""Tests for xls.dslx.xls_type_error."""

from xls.common import test_base
from xls.dslx.concrete_type import BitsType
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span
from xls.dslx.xls_type_error import XlsTypeError


class XlsTypeErrorTest(test_base.TestCase):

  def test_stringify(self):
    # Test without a suffix.
    t = BitsType(signed=False, size=3)
    fake_pos = Pos('<fake>', 9, 10)
    fake_span = Span(fake_pos, fake_pos)
    e = XlsTypeError(fake_span, t, t)
    self.assertEndsWith(str(e), '@ <fake>:10:11-10:11')


if __name__ == '__main__':
  test_base.main()
