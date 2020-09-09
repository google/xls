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
        "//xls/dslx/python:cpp_ast",
    ],
)

py_library(
    name = "bits",
    srcs = ["bits.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx:bit_helpers",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "bits_test",
    srcs = ["bits_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bits",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "array",
    srcs = ["array.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        "//xls/dslx/python:cpp_ast",
    ],
)

py_library(
    name = "value",
    srcs = ["value.py"],
    srcs_version = "PY3",
    deps = [
        ":array",
        ":bits",
        "//xls/dslx/python:cpp_ast",
    ],
)

py_library(
    name = "value_parser",
    srcs = ["value_parser.py"],
    srcs_version = "PY3",
    deps = [
        ":value",
        "//xls/common:xls_error",
        "//xls/dslx:scanner",
    ],
)

py_test(
    name = "value_test",
    srcs = ["value_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":value",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "value_parser_test",
    srcs = ["value_parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":value",
        ":value_parser",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "bindings",
    srcs = ["bindings.py"],
    srcs_version = "PY3",
    deps = [
        ":value",
        "//xls/dslx:parametric_instantiator",
        "//xls/dslx:span",
        "//xls/dslx/python:cpp_ast",
    ],
)

py_library(
    name = "concrete_type_helpers",
    srcs = ["concrete_type_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":value",
        "//xls/dslx:bit_helpers",
        "//xls/dslx:concrete_type",
        "//xls/dslx:scanner",
        "//xls/dslx:span",
        "//xls/dslx/python:cpp_ast",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "interpreter",
    srcs = ["interpreter.py"],
    srcs_version = "PY3",
    deps = [
        ":bindings",
        ":concrete_type_helpers",
        ":errors",
        ":jit_comparison",
        ":value",
        requirement("termcolor"),
        "//xls/dslx:ast_helpers",
        "//xls/dslx:bit_helpers",
        "//xls/dslx:concrete_type",
        "//xls/dslx:deduce",
        "//xls/dslx:import_fn",
        "//xls/dslx:ir_name_mangler",
        "//xls/dslx:parametric_expression",
        "//xls/dslx:parametric_instantiator",
        "//xls/dslx:span",
        "//xls/dslx:type_info",
        "//xls/dslx/python:cpp_ast",
        "//xls/ir/python:package",
        "//xls/jit/python:llvm_ir_jit",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "jit_comparison",
    srcs = ["jit_comparison.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx:bit_helpers",
        "//xls/dslx:concrete_type",
        "//xls/dslx/interpreter:value",
        "//xls/ir/python:bits",
        "//xls/ir/python:number_parser",
        "//xls/ir/python:value",
        "//xls/jit/python:llvm_ir_jit",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "jit_comparison_test",
    srcs = ["jit_comparison_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jit_comparison",
        ":value",
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
        ":bindings",
        ":interpreter",
        "//xls/dslx:deduce",
        "//xls/dslx:type_info",
        "//xls/dslx/python:cpp_ast",
    ],
)

py_library(
    name = "parse_and_interpret",
    srcs = ["parse_and_interpret.py"],
    srcs_version = "PY3",
    deps = [
        ":bindings",
        ":interpreter",
        "//xls/dslx:import_routines",
        "//xls/dslx:ir_converter",
        "//xls/dslx:parser",
        "//xls/dslx:parser_helpers",
        "//xls/dslx:scanner",
        "//xls/dslx:span",
        "//xls/dslx:typecheck",
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":interpreter",
        ":parse_and_interpret",
        requirement("pyfakefs"),  # unittest
        "//xls/dslx:concrete_type",
        "//xls/dslx:xls_type_error",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_binary(
    name = "interpreter_main",
    srcs = ["interpreter_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parse_and_interpret",
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
        ":interpreter",
        ":value",
        requirement("pyfakefs"),
        "//xls/dslx:bit_helpers",
        "//xls/dslx:concrete_type",
        "//xls/dslx:import_routines",
        "//xls/dslx:parser",
        "//xls/dslx:parser_helpers",
        "//xls/dslx:scanner",
        "//xls/dslx:span",
        "//xls/dslx:typecheck",
        "//xls/dslx:xls_type_error",
        "//xls/dslx/python:cpp_ast",
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
