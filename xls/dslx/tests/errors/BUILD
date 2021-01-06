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

# Tests for DSLX files that have various forms of errors.

# pytype tests are present in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_test(
    name = "error_modules_test",
    srcs = ["error_modules_test.py"],
    data = [
        "colon_ref_builtin.x",
        "has_type_error.x",
        "imports_and_causes_ref_error.x",
        "imports_and_typedefs_dne_type.x",
        "imports_has_type_error.x",
        "imports_private_enum.x",
        "mod_private_enum.x",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/dslx/interpreter:parse_and_interpret",
        "//xls/dslx/python:cpp_deduce",
        "//xls/dslx/python:cpp_parser",
    ],
)
