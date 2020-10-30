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
"""Tests for importing a DSLX module that has a type error (from another module)."""

from xls.common import runfiles
from xls.dslx.interpreter import parse_and_interpret
from absl.testing import absltest


class ImportModuleWithTypeErrorTest(absltest.TestCase):

  def test_imports_module_with_type_error(self):
    path = runfiles.get_path(
        'xls/dslx/interpreter/tests/imports_has_type_error.x')
    with self.assertRaises(Exception) as cm:
      parse_and_interpret.parse_and_test_path(path)

    self.assertIn('XlsTypeError', str(cm.exception))
    self.assertIn('xls/dslx/interpreter/tests/has_type_error.x:16:3-16:4',
                  str(cm.exception))


if __name__ == '__main__':
  absltest.main()
