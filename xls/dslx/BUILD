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
    deps = [
        "//xls/dslx:parametric_expression",
    ],
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
        "//xls/dslx/python:cpp_pos",
    ],
)

py_test(
    name = "scanner_test",
    srcs = ["scanner_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_binary(
    name = "scanner_main",
    srcs = ["scanner_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_scanner",
        "@com_google_absl_py//absl:app",
    ],
)

py_library(
    name = "parametric_expression",
    srcs = ["parametric_expression.py"],
    srcs_version = "PY3",
    deps = ["//xls/dslx/python:cpp_pos"],
)

py_test(
    name = "parametric_expression_test",
    srcs = ["parametric_expression_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parametric_expression",
        "//xls/dslx/python:cpp_pos",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "symbolic_bindings",
    srcs = ["symbolic_bindings.py"],
    srcs_version = "PY3",
)

py_library(
    name = "parametric_instantiator",
    srcs = ["parametric_instantiator.py"],
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":parametric_expression",
        ":symbolic_bindings",
        ":xls_type_error",
        "//xls/dslx/python:cpp_pos",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "ast_test",
    srcs = ["ast_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast_helpers",
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_ast_visitor",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
    ],
)

py_library(
    name = "token_parser",
    srcs = ["token_parser.py"],
    srcs_version = "PY3",
    deps = [
        ":parse_error",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
    ],
)

py_library(
    name = "parse_error",
    srcs = ["parse_error.py"],
    srcs_version = "PY3",
    deps = [
        ":span",
        "//xls/dslx/python:cpp_pos",
    ],
)

py_library(
    name = "bindings",
    srcs = ["bindings.py"],
    srcs_version = "PY3",
    deps = [
        ":parse_error",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_pos",
    ],
)

py_library(
    name = "parser",
    srcs = ["parser.py"],
    srcs_version = "PY3",
    deps = [
        ":ast_helpers",
        ":bindings",
        ":dslx_builtins",
        ":parse_error",
        ":token_parser",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "parser_helpers",
    srcs = ["parser_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":parser",
        ":span",
        requirement("termcolor"),
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_scanner",
    ],
)

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fakefs_test_util",
        ":parser",
        ":parser_helpers",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "parse_and_typecheck",
    srcs = ["parse_and_typecheck.py"],
    srcs_version = "PY3",
    deps = [
        ":fakefs_util",
        ":import_fn",
        ":parser_helpers",
        ":type_info",
        ":typecheck",
        ":xls_type_error",
        requirement("pyfakefs"),
        "//xls/dslx/python:cpp_ast",
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
    name = "type_info",
    srcs = ["type_info.py"],
    deps = [
        ":ast_helpers",
        ":concrete_type",
        ":span",
        ":symbolic_bindings",
        ":xls_type_error",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_pos",
    ],
)

py_library(
    name = "deduce",
    srcs = ["deduce.py"],
    srcs_version = "PY3",
    deps = [
        ":ast_helpers",
        ":bit_helpers",
        ":concrete_type",
        ":dslx_builtins",
        ":parametric_expression",
        ":parametric_instantiator",
        ":type_info",
        ":xls_type_error",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
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
        ":xls_type_error",
        "//xls/dslx/python:cpp_pos",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "typecheck",
    srcs = ["typecheck.py"],
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":deduce",
        ":dslx_builtins",
        ":span",
        ":type_info",
        ":xls_type_error",
        "//xls/common:xls_error",
        "//xls/dslx/interpreter:interpreter_helpers",
        "//xls/dslx/python:cpp_ast",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "typecheck_test",
    srcs = ["typecheck_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fakefs_test_util",
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
        "//xls/dslx/python:cpp_pos",
    ],
)

py_test(
    name = "xls_type_error_test",
    srcs = ["xls_type_error_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concrete_type",
        ":xls_type_error",
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_pos",
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
        "//xls/dslx/python:cpp_pos",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "ir_converter",
    srcs = ["ir_converter.py"],
    srcs_version = "PY3",
    deps = [
        ":ast_helpers",
        ":bit_helpers",
        ":concrete_type",
        ":deduce",
        ":dslx_builtins",
        ":extract_conversion_order",
        ":ir_name_mangler",
        ":parametric_expression",
        ":parametric_instantiator",
        ":span",
        ":type_info",
        "//xls/common:xls_error",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_ast_visitor",
        "//xls/dslx/python:cpp_pos",
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

py_library(
    name = "ir_name_mangler",
    srcs = ["ir_name_mangler.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_ast",
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
        ":fakefs_test_util",
        ":ir_converter",
        ":parser_helpers",
        ":span",
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
        requirement("termcolor"),
        "//xls/dslx/python:cpp_scanner",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_library(
    name = "extract_conversion_order",
    srcs = ["extract_conversion_order.py"],
    srcs_version = "PY3",
    deps = [
        ":dslx_builtins",
        ":parametric_instantiator",
        ":type_info",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_ast_visitor",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "extract_conversion_order_test",
    srcs = ["extract_conversion_order_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":extract_conversion_order",
        ":fakefs_test_util",
        ":parse_and_typecheck",
        ":type_info",
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_ast",
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
        requirement("pyfakefs"),
    ],
)

py_library(
    name = "fakefs_test_util",
    # If we don't mark this as testonly we can get warnings (when the modified
    # third party version drags in googletest as a dependency and the test
    # main is not called).
    testonly = True,
    srcs = ["fakefs_test_util.py"],
    srcs_version = "PY3",
    deps = [
        ":fakefs_util",
        requirement("pyfakefs"),  # unittest
    ],
)

py_library(
    name = "import_fn",
    srcs = ["import_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":deduce",
        ":type_info",
        "//xls/dslx/python:cpp_ast",
    ],
)

py_library(
    name = "ast_helpers",
    srcs = ["ast_helpers.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
    ],
)

cc_library(
    name = "cpp_pos",
    srcs = ["cpp_pos.cc"],
    hdrs = ["cpp_pos.h"],
    deps = [
        "//xls/common:integral_types",
        "//xls/common/logging",
        "//xls/common/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_re2//:re2",
    ],
)

cc_library(
    name = "cpp_ast_builtin_types",
    hdrs = ["cpp_ast_builtin_types.inc"],
)

cc_library(
    name = "cpp_ast",
    srcs = ["cpp_ast.cc"],
    hdrs = ["cpp_ast.h"],
    deps = [
        ":cpp_ast_builtin_types",
        ":cpp_pos",
        "//xls/common:indent",
        "//xls/common/logging",
        "//xls/common/status:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "cpp_ast_test",
    srcs = ["cpp_ast_test.cc"],
    deps = [
        ":cpp_ast",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cpp_scanner",
    srcs = ["cpp_scanner.cc"],
    hdrs = [
        "cpp_scanner.h",
        "cpp_scanner_keywords.inc",
    ],
    deps = [
        ":cpp_ast",
        ":cpp_pos",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "cpp_scanner_test",
    srcs = ["cpp_scanner_test.cc"],
    deps = [
        ":cpp_scanner",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
