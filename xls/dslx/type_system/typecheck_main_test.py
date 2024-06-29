# Copyright 2022 The XLS Authors
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
"""Tests for typecheck_main."""

import os
import subprocess as subp

from absl.testing import absltest
from xls.common import runfiles

_TYPECHECK_MAIN_PATH = runfiles.get_path('xls/dslx/type_system/typecheck_main')


class TypecheckMainTest(absltest.TestCase):

  def test_module_with_import(self):
    mod_path = runfiles.get_path('xls/dslx/tests/mod_const_enum_importer.x')
    basedir = mod_path
    for _ in range(3):
      basedir, _ = os.path.split(basedir)
    output = subp.check_output(
        [_TYPECHECK_MAIN_PATH, mod_path, '--dslx_path=' + basedir],
        encoding='utf-8',
    )
    self.assertIn('TYPE_REF :: `mod_simple_const_enum::MyEnum`', output)


if __name__ == '__main__':
  absltest.main()
