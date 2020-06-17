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

# Frontend syntax.

# pytype binary, test, library

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "bit_helpers",
    srcs = ["bit_helpers.py"],
    srcs_version = "PY3",
)

py_test(
    name = "bit_helpers_test",
    srcs = ["bit_helpers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bit_helpers",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "span",
    srcs = ["span.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/common:xls_error",
    ],
)

py_library(
    name = "scanner",
    srcs = ["scanner.py"],
    srcs_version = "PY3",
    deps = [
        ":dslx_builtins",
        ":span",
    ],
)

py_test(
    name = "scanner_test",
    srcs = ["scanner_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":scanner",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_binary(
    name = "scanner_main",
    srcs = ["scanner_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":scanner",
        "@com_google_absl_py//absl:app",
    ],
)

py_library(
    name = "parametric_expression",
    srcs = ["parametric_expression.py"],
    srcs_version = "PY3",
    deps = [":span"],
)

py_test(
    name = "parametric_expression_test",
    srcs = ["parametric_expression_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parametric_expression",
        ":span",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "parametric_instantiator",
    srcs = ["parametric_instantiator.py"],
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":parametric_expression",
        ":span",
        ":xls_type_error",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "core_ast_nodes",
    srcs = ["core_ast_nodes.py"],
    srcs_version = "PY3",
    deps = [
        ":ast_node",
        ":bit_helpers",
        ":concrete_type",
        ":free_variables",
        ":parametric_expression",
        ":scanner",
        ":span",
        ":xls_type_error",
    ],
)

py_library(
    name = "leaf_ast_nodes",
    srcs = ["leaf_ast_nodes.py"],
    srcs_version = "PY3",
    deps = [
        ":ast_node",
        ":bit_helpers",
        ":concrete_type",
        ":core_ast_nodes",
        ":free_variables",
        ":parametric_expression",
        ":parametric_instantiator",
        ":scanner",
        ":span",
        ":xls_type_error",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "ast",
    srcs = ["ast.py"],
    srcs_version = "PY3",
    deps = [
        ":ast_node",
        ":core_ast_nodes",
        ":leaf_ast_nodes",
    ],
)

py_test(
    name = "ast_test",
    srcs = ["ast_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":scanner",
        ":span",
        "//xls/common:test_base",
    ],
)

py_library(
    name = "token_parser",
    srcs = ["token_parser.py"],
    srcs_version = "PY3",
    deps = [
        ":parse_error",
        ":scanner",
        ":span",
    ],
)

py_library(
    name = "parse_error",
    srcs = ["parse_error.py"],
    srcs_version = "PY3",
    deps = [
        ":span",
    ],
)

py_library(
    name = "bindings",
    srcs = ["bindings.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":parse_error",
        ":span",
    ],
)

py_library(
    name = "parser",
    srcs = ["parser.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":bindings",
        ":dslx_builtins",
        ":parse_error",
        ":scanner",
        ":span",
        ":token_parser",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "parser_helpers",
    srcs = ["parser_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":parser",
        ":scanner",
        ":span",
        requirement("termcolor"),
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":fakefs_util",
        ":parser",
        ":parser_helpers",
        ":scanner",
        ":span",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "parse_and_typecheck",
    srcs = ["parse_and_typecheck.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":deduce",
        ":import_fn",
        ":parser_helpers",
        ":typecheck",
        ":xls_type_error",
        "@pyfakefs_archive//:pyfakefs",
    ],
)

py_binary(
    name = "parser_main",
    srcs = ["parser_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":import_routines",
        ":parse_and_typecheck",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_library(
    name = "math_helpers",
    srcs = ["math_helpers.py"],
    srcs_version = "PY3",
)

py_library(
    name = "deduce",
    srcs = ["deduce.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":bit_helpers",
        ":concrete_type",
        ":debug_utils",
        ":dslx_builtins",
        ":parametric_expression",
        ":parametric_instantiator",
        ":scanner",
        ":span",
        ":xls_type_error",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "dslx_builtins",
    srcs = ["dslx_builtins.py"],
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":parametric_instantiator",
        ":span",
        ":xls_type_error",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "typecheck",
    srcs = ["typecheck.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":concrete_type",
        ":deduce",
        ":dslx_builtins",
        ":parametric_expression",
        ":parametric_instantiator",
        ":scanner",
        ":span",
        ":xls_type_error",
        "//xls/common:xls_error",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "typecheck_test",
    srcs = ["typecheck_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fakefs_util",
        ":parser_helpers",
        ":span",
        ":typecheck",
        ":xls_type_error",
        "//xls/common:xls_error",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "xls_type_error",
    srcs = ["xls_type_error.py"],
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":span",
    ],
)

py_test(
    name = "xls_type_error_test",
    srcs = ["xls_type_error_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":span",
        ":xls_type_error",
        "//xls/common:test_base",
    ],
)

py_library(
    name = "concrete_type",
    srcs = ["concrete_type.py"],
    srcs_version = "PY3",
    deps = [
        ":math_helpers",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "concrete_type_test",
    srcs = ["concrete_type_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":parametric_expression",
        ":span",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "ir_converter",
    srcs = ["ir_converter.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":bit_helpers",
        ":concrete_type",
        ":deduce",
        ":dslx_builtins",
        ":extract_conversion_order",
        ":parametric_expression",
        ":parametric_instantiator",
        ":span",
        "//xls/common:xls_error",
        "//xls/dslx/interpreter",
        "//xls/ir/python:bits",
        "//xls/ir/python:fileno",
        "//xls/ir/python:function",
        "//xls/ir/python:function_builder",
        "//xls/ir/python:lsb_or_msb",
        "//xls/ir/python:number_parser",
        "//xls/ir/python:package",
        "//xls/ir/python:source_location",
        "//xls/ir/python:type",
        "//xls/ir/python:value",
        "//xls/ir/python:verifier",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "ir_converter_test",
    srcs = ["ir_converter_test.py"],
    data = [
        "//xls/tools:parse_ir",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fakefs_util",
        ":ir_converter",
        ":parser_helpers",
        ":typecheck",
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

# TODO(b/136618878): 2019-07-08 Switching this to PY3 causes ASAN to fail
# because of the CLIF module.
py_binary(
    name = "ir_converter_main",
    srcs = ["ir_converter_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":import_routines",
        ":ir_converter",
        ":parser_helpers",
        ":span",
        ":typecheck",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "highlight_main",
    srcs = ["highlight_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":scanner",
        requirement("termcolor"),
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_library(
    name = "free_variables",
    srcs = ["free_variables.py"],
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
    ],
)

py_library(
    name = "ast_node",
    srcs = ["ast_node.py"],
    srcs_version = "PY3",
    deps = [":span"],
)

py_library(
    name = "extract_conversion_order",
    srcs = ["extract_conversion_order.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":deduce",
        ":dslx_builtins",
        ":parametric_expression",
        ":parametric_instantiator",
        ":parser",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "extract_conversion_order_test",
    srcs = ["extract_conversion_order_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":extract_conversion_order",
        ":fakefs_util",
        ":parse_and_typecheck",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

filegroup(
    name = "ir_converter_test_sh",
    srcs = ["ir_converter_test.sh"],
)

py_library(
    name = "ir_converter_helpers",
    srcs = ["ir_converter_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":import_fn",
        ":ir_converter",
        ":parse_and_typecheck",
        "//xls/ir/python:package",
    ],
)

py_library(
    name = "debug_utils",
    srcs = ["debug_utils.py"],
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "import_routines",
    srcs = ["import_routines.py"],
    data = ["//xls/dslx/stdlib:x_files"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":deduce",
        ":import_fn",
        ":parse_and_typecheck",
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "fakefs_util",
    srcs = ["fakefs_util.py"],
    srcs_version = "PY3",
    deps = [
        "@pyfakefs_archive//:pyfakefs",  # fake_filesystem_unittest
    ],
)

py_library(
    name = "import_fn",
    srcs = ["import_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":ast",
        ":deduce",
    ],
)
