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
        "bad_annotation.x",
        "bad_dim.x",
        "brace_scope.x",
        "co_recursion.x",
        "colon_ref_builtin.x",
        "constant_without_type_annot.x",
        "double_define_parameter.x",
        "double_define_top_level_function.x",
        "enum_with_type_on_value.x",
        "has_type_error.x",
        "imports_and_causes_ref_error.x",
        "imports_and_typedefs_dne_type.x",
        "imports_has_type_error.x",
        "imports_private_enum.x",
        "invalid_array_expression_size.x",
        "invalid_array_expression_type.x",
        "invalid_parameter_cast.x",
        "let_destructure_same_name.x",
        "match_multi_pattern_with_bindings.x",
        "mod_private_enum.x",
        "multiple_mod_level_const_bindings.x",
        "non_constexpr_slice.x",
        "self_recursion.x",
        "tail_call.x",
        "two_failing_tests.x",
        "//xls/dslx:interpreter_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)
