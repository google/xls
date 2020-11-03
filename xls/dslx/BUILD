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

# Frontend syntax.

# pytype binary, test, library

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "import_helpers",
    srcs = ["import_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":typecheck",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_type_info",
        "//xls/dslx/python:import_routines",
    ],
)

py_library(
    name = "bit_helpers",
    srcs = ["bit_helpers.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_parametric_expression",
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

py_test(
    name = "parametric_expression_test",
    srcs = ["parametric_expression_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_parametric_expression",
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
        ":concrete_type_helpers",
        ":symbolic_bindings",
        ":xls_type_error",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_parametric_expression",
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
    name = "parse_error",
    srcs = ["parse_error.py"],
    srcs_version = "PY3",
    deps = [
        ":span",
        "//xls/dslx/python:cpp_pos",
    ],
)

py_library(
    name = "parser_helpers",
    srcs = ["parser_helpers.py"],
    srcs_version = "PY3",
    deps = [
        ":parse_error",
        ":span",
        requirement("termcolor"),
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_parser",
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
        ":parser_helpers",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_parser",
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
        ":typecheck",
        ":xls_type_error",
        requirement("pyfakefs"),
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_type_info",
    ],
)

py_binary(
    name = "parser_main",
    srcs = ["parser_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":import_helpers",
        ":parse_and_typecheck",
        "//xls/dslx/python:import_routines",
        "@com_google_absl_py//absl:app",
    ],
)

py_library(
    name = "math_helpers",
    srcs = ["math_helpers.py"],
    srcs_version = "PY3",
)

cc_library(
    name = "cpp_type_info",
    srcs = ["type_info.cc"],
    hdrs = ["type_info.h"],
    deps = [
        ":cpp_ast",
        ":cpp_concrete_type",
    ],
)

cc_test(
    name = "cpp_type_info_test",
    srcs = ["type_info_test.cc"],
    deps = [
        ":cpp_type_info",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "deduce",
    srcs = ["deduce.py"],
    srcs_version = "PY3",
    deps = [
        ":ast_helpers",
        ":bit_helpers",
        ":concrete_type_helpers",
        ":dslx_builtins",
        ":parametric_instantiator",
        ":xls_type_error",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_parametric_expression",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_scanner",
        "//xls/dslx/python:cpp_type_info",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "dslx_builtins",
    srcs = ["dslx_builtins.py"],
    srcs_version = "PY3",
    deps = [
        ":parametric_instantiator",
        ":xls_type_error",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_pos",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "typecheck",
    srcs = ["typecheck.py"],
    srcs_version = "PY3",
    deps = [
        ":deduce",
        ":dslx_builtins",
        ":span",
        ":xls_type_error",
        "//xls/common:xls_error",
        "//xls/dslx/interpreter:interpreter_helpers",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_type_info",
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
        "//xls/dslx/python:cpp_parser",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "xls_type_error",
    srcs = ["xls_type_error.py"],
    srcs_version = "PY3",
    deps = [
        ":span",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_pos",
    ],
)

py_test(
    name = "xls_type_error_test",
    srcs = ["xls_type_error_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":xls_type_error",
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_pos",
    ],
)

py_test(
    name = "concrete_type_test",
    srcs = ["concrete_type_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_parametric_expression",
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
        ":concrete_type_helpers",
        ":deduce",
        ":dslx_builtins",
        ":extract_conversion_order",
        ":ir_name_mangler",
        ":parametric_instantiator",
        ":span",
        "//xls/common:xls_error",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_ast_visitor",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_parametric_expression",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:cpp_type_info",
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
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_parser",
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
        ":import_helpers",
        ":ir_converter",
        ":parser_helpers",
        ":span",
        ":typecheck",
        "//xls/common/python:init_xls",
        "//xls/dslx/python:cpp_parser",
        "//xls/dslx/python:import_routines",
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
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_ast_visitor",
        "//xls/dslx/python:cpp_type_info",
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
        "//xls/common:test_base",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_type_info",
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
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_type_info",
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
        "//xls/ir/python:bits",
    ],
)

cc_library(
    name = "cpp_pos",
    srcs = ["cpp_pos.cc"],
    hdrs = ["cpp_pos.h"],
    deps = [
        "//xls/common:integral_types",
        "//xls/common/logging",
        "@com_google_absl//absl/status:statusor",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
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

cc_library(
    name = "cpp_bindings",
    srcs = ["cpp_bindings.cc"],
    hdrs = ["cpp_bindings.h"],
    deps = [
        ":cpp_ast",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cpp_token_parser",
    hdrs = ["token_parser.h"],
    deps = [
        ":cpp_bindings",
        ":cpp_scanner",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "cpp_parser",
    srcs = ["parser.cc"],
    hdrs = ["parser.h"],
    deps = [
        ":cpp_ast",
        ":cpp_bindings",
        ":cpp_token_parser",
        "//xls/common:cleanup",
        "@com_google_absl//absl/debugging:demangle_internal",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "cpp_parser_test",
    srcs = ["parser_test.cc"],
    deps = [
        ":cpp_parser",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cpp_parametric_expression",
    srcs = ["parametric_expression.cc"],
    hdrs = ["parametric_expression.h"],
    deps = [
        ":cpp_pos",
        "//xls/common:integral_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "cpp_concrete_type",
    srcs = ["concrete_type.cc"],
    hdrs = ["concrete_type.h"],
    deps = [
        ":cpp_ast",
        ":cpp_parametric_expression",
        "//xls/common:integral_types",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "cpp_concrete_type_test",
    srcs = ["concrete_type_test.cc"],
    deps = [
        ":cpp_concrete_type",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "concrete_type_helpers",
    srcs = ["concrete_type_helpers.py"],
    deps = [
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_concrete_type",
    ],
)

cc_library(
    name = "interp_value",
    srcs = ["interp_value.cc"],
    hdrs = ["interp_value.h"],
    deps = [
        ":cpp_ast",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
    ],
)

cc_library(
    name = "interp_bindings",
    srcs = ["interp_bindings.cc"],
    hdrs = ["interp_bindings.h"],
    deps = [
        ":cpp_ast",
        ":cpp_type_info",
        ":interp_value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "interp_bindings_test",
    srcs = ["interp_bindings_test.cc"],
    deps = [
        ":interp_bindings",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "builtins",
    srcs = ["builtins.cc"],
    hdrs = ["builtins.h"],
    deps = [
        ":cpp_ast",
        ":cpp_type_info",
        ":interp_value",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits_ops",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "cpp_evaluate",
    srcs = ["cpp_evaluate.cc"],
    hdrs = ["cpp_evaluate.h"],
    deps = [
        ":cpp_type_info",
        ":interp_bindings",
        ":interp_value",
        "//xls/common/status:ret_check",
        "//xls/ir:bits",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "import_routines",
    srcs = ["import_routines.cc"],
    hdrs = ["import_routines.h"],
    data = ["//xls/dslx/stdlib:x_files"],
    deps = [
        ":cpp_ast",
        ":cpp_parser",
        ":cpp_scanner",
        ":cpp_type_info",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
    ],
)
