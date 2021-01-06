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
from xls.common import test_base
from xls.dslx.interpreter import parse_and_interpret
from xls.dslx.python.cpp_deduce import TypeInferenceError
from xls.dslx.python.cpp_parser import CppParseError


class ImportModuleWithTypeErrorTest(test_base.TestCase):

  def test_imports_module_with_type_error(self):
    path = runfiles.get_path('xls/dslx/tests/errors/imports_has_type_error.x')
    with self.assertRaises(Exception) as cm:
      parse_and_interpret.parse_and_test_path(path)

    self.assertIn('xls/dslx/tests/errors/has_type_error.x:16:3-16:4',
                  str(cm.exception))
    self.assertIn('did not match the annotated return type', str(cm.exception))

  def test_imports_and_causes_ref_error(self):
    path = runfiles.get_path(
        'xls/dslx/tests/errors/imports_and_causes_ref_error.x')
    with self.assertRaises(CppParseError) as cm:
      parse_and_interpret.parse_and_test_path(path)

    self.assertIn('ParseError', str(cm.exception.message))
    self.assertIn(
        'xls/dslx/tests/errors/imports_and_causes_ref_error.x:17:29-17:31',
        str(cm.exception.message))

  def test_imports_private_enum(self):
    path = runfiles.get_path('xls/dslx/tests/errors/imports_private_enum.x')
    with self.assertRaises(TypeInferenceError) as cm:
      parse_and_interpret.parse_and_test_path(path)

    self.assertIn('xls/dslx/tests/errors/imports_private_enum.x:17:14-17:40',
                  str(cm.exception.span))

  def test_imports_dne(self):
    path = runfiles.get_path(
        'xls/dslx/tests/errors/imports_and_typedefs_dne_type.x')
    with self.assertRaises(TypeInferenceError) as cm:
      parse_and_interpret.parse_and_test_path(path)

    self.assertIn(
        'xls/dslx/tests/errors/imports_and_typedefs_dne_type.x:17:12-17:48',
        str(cm.exception.span))
    self.assertIn(
        "xls.dslx.tests.errors.mod_private_enum member 'ReallyDoesNotExist' which does not exist",
        str(cm.exception))

  def test_colon_ref_builtin(self):
    path = runfiles.get_path('xls/dslx/tests/errors/colon_ref_builtin.x')
    with self.assertRaises(TypeInferenceError) as cm:
      parse_and_interpret.parse_and_test_path(path)

    self.assertIn('xls/dslx/tests/errors/colon_ref_builtin.x:16:9-16:25',
                  str(cm.exception.span))
    self.assertIn("Builtin 'update' has no attributes", str(cm.exception))

  def constant_without_type_annotation(self):
    path = runfiles.get_path('xls/dslx/tests/constant_without_type_annot.x')
    with self.assertRaises(TypeInferenceError) as cm:
      parse_and_interpret.parse_and_test_path(path)

    self.assertIn('xls/dslx/tests/constant_without_type_annot.x:2:13-2:15',
                  str(cm.exception.span))
    self.assertIn('please annotate a type.', str(cm.exception))


if __name__ == '__main__':
  test_base.main()
