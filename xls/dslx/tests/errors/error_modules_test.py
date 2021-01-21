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
"""Tests for DSLX modules with various forms of errors."""

import subprocess as subp

from xls.common import runfiles
from xls.common import test_base

_INTERP_PATH = runfiles.get_path('xls/dslx/cpp_interpreter_main')


class ImportModuleWithTypeErrorTest(test_base.TestCase):

  def _run(self, path: str) -> str:
    full_path = runfiles.get_path(path)
    p = subp.run([_INTERP_PATH, full_path],
                 stderr=subp.PIPE,
                 check=False,
                 encoding='utf-8')
    self.assertNotEqual(p.returncode, 0)
    return p.stderr

  def test_imports_module_with_type_error(self):
    stderr = self._run('xls/dslx/tests/errors/imports_has_type_error.x')
    self.assertIn('xls/dslx/tests/errors/has_type_error.x:16:3-16:4', stderr)
    self.assertIn('did not match the annotated return type', stderr)

  def test_imports_and_causes_ref_error(self):
    stderr = self._run('xls/dslx/tests/errors/imports_and_causes_ref_error.x')
    self.assertIn('ParseError', stderr)
    self.assertIn(
        'xls/dslx/tests/errors/imports_and_causes_ref_error.x:17:29-17:31',
        stderr)

  def test_imports_private_enum(self):
    stderr = self._run('xls/dslx/tests/errors/imports_private_enum.x')
    self.assertIn('xls/dslx/tests/errors/imports_private_enum.x:17:14-17:40',
                  stderr)

  def test_imports_dne(self):
    stderr = self._run('xls/dslx/tests/errors/imports_and_typedefs_dne_type.x')
    self.assertIn(
        'xls/dslx/tests/errors/imports_and_typedefs_dne_type.x:17:12-17:48',
        stderr)
    self.assertIn(
        "xls.dslx.tests.errors.mod_private_enum member 'ReallyDoesNotExist' which does not exist",
        stderr)

  def test_colon_ref_builtin(self):
    stderr = self._run('xls/dslx/tests/errors/colon_ref_builtin.x')
    self.assertIn('xls/dslx/tests/errors/colon_ref_builtin.x:16:9-16:25',
                  stderr)
    self.assertIn("Builtin 'update' has no attributes", stderr)

  def constant_without_type_annotation(self):
    stderr = self._run('xls/dslx/tests/constant_without_type_annot.x')
    self.assertIn('xls/dslx/tests/constant_without_type_annot.x:2:13-2:15',
                  stderr)
    self.assertIn('please annotate a type.', stderr)


if __name__ == '__main__':
  test_base.main()
