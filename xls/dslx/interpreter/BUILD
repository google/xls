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

# pytype binary, test, library

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
        "//xls/dslx/python:cpp_scanner",
        "//xls/dslx/python:interp_value",
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
    name = "concrete_type_helpers",
    srcs = ["concrete_type_helpers.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
        "//xls/dslx/python:interp_value",
        "@com_google_absl_py//absl/logging",
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

py_library(
    name = "interpreter_helpers",
    srcs = ["interpreter_helpers.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx:deduce",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_type_info",
        "//xls/dslx/python:import_routines",
        "//xls/dslx/python:interp_bindings",
        "//xls/dslx/python:interpreter",
    ],
)

py_library(
    name = "quickcheck_helpers",
    srcs = ["quickcheck_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":jit_comparison",
        "//xls/dslx:ir_name_mangler",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:interpreter",
        "//xls/ir/python:package",
        "//xls/jit/python:ir_jit",
    ],
)

py_library(
    name = "parse_and_interpret",
    srcs = ["parse_and_interpret.py"],
    srcs_version = "PY3",
    deps = [
        ":quickcheck_helpers",
        "//xls/dslx:import_helpers",
        "//xls/dslx:ir_converter",
        "//xls/dslx:parser_helpers",
        "//xls/dslx:span",
        "//xls/dslx:typecheck",
        "//xls/dslx/python:cpp_parser",
        "//xls/dslx/python:cpp_scanner",
        "//xls/dslx/python:interpreter",
    ],
)

py_test(
    name = "parse_and_interpret_test",
    srcs = ["parse_and_interpret_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parse_and_interpret",
        requirement("pyfakefs"),  # unittest
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [
        ":interpreter_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parse_and_interpret",
        requirement("pyfakefs"),  # unittest
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_deduce",
        "//xls/dslx/python:interpreter",
    ],
)

py_binary(
    name = "interpreter_main",
    srcs = ["interpreter_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parse_and_interpret",
        "//xls/common/python:init_xls",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "repl",
    srcs = ["repl.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("pyfakefs"),
        "//xls/common/python:init_xls",
        "//xls/dslx:bit_helpers",
        "//xls/dslx:import_helpers",
        "//xls/dslx:parser_helpers",
        "//xls/dslx:span",
        "//xls/dslx:typecheck",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_deduce",
        "//xls/dslx/python:cpp_parser",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
        "//xls/dslx/python:interp_value",
        "//xls/dslx/python:interpreter",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_test(
    name = "repl_test",
    srcs = ["repl_test.py"],
    data = [":repl"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

filegroup(
    name = "dslx_test",
    srcs = ["dslx_test.sh"],
)
