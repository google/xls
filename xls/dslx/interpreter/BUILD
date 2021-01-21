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

# Reference interpreter executor for XLS syntax.

# pytype test and library

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx:span",
        "//xls/dslx/python:cpp_pos",
    ],
)

py_library(
    name = "value_parser",
    srcs = ["value_parser.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/common:xls_error",
        "//xls/dslx:ast_helpers",
        "//xls/dslx/python:interp_value",
        "//xls/dslx/python:scanner",
        "//xls/ir/python:bits",
    ],
)

py_test(
    name = "value_test",
    srcs = ["value_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:interp_value",
        "//xls/ir/python:bits",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "value_parser_test",
    srcs = ["value_parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":value_parser",
        "//xls/dslx/python:interp_value",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "jit_comparison",
    srcs = ["jit_comparison.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx:bit_helpers",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:interp_value",
        "//xls/ir/python:bits",
        "//xls/ir/python:number_parser",
        "//xls/ir/python:value",
    ],
)

py_test(
    name = "jit_comparison_test",
    srcs = ["jit_comparison_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jit_comparison",
        "//xls/dslx/python:interp_value",
        "//xls/ir/python:bits",
        "//xls/ir/python:value",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [
        "//xls/dslx:cpp_interpreter_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("pyfakefs"),  # unittest
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_deduce",
        "//xls/dslx/python:interpreter",
    ],
)

filegroup(
    name = "dslx_test",
    srcs = ["dslx_test.sh"],
    visibility = ["//xls:xls_users"],
)
