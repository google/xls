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

# pytype test and library

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

cc_library(
    name = "cpp_transpiler",
    srcs = ["cpp_transpiler.cc"],
    hdrs = ["cpp_transpiler.h"],
    deps = [
        ":ast",
        ":import_data",
        ":interpreter",
        ":type_info",
        ":typecheck",
        "//xls/common:case_converters",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "cpp_transpiler_test",
    srcs = ["cpp_transpiler_test.cc"],
    deps = [
        ":ast",
        ":cpp_transpiler",
        ":import_data",
        ":parse_and_typecheck",
        ":type_info",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "symbolic_bindings",
    srcs = ["symbolic_bindings.cc"],
    hdrs = ["symbolic_bindings.h"],
    deps = [
        ":interp_value",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "type_and_bindings",
    hdrs = ["type_and_bindings.h"],
    deps = [
        ":concrete_type",
        ":symbolic_bindings",
    ],
)

cc_library(
    name = "type_info",
    srcs = ["type_info.cc"],
    hdrs = ["type_info.h"],
    deps = [
        ":ast",
        ":concrete_type",
        ":symbolic_bindings",
        "//xls/common/status:ret_check",
    ],
)

cc_test(
    name = "type_info_test",
    srcs = ["type_info_test.cc"],
    deps = [
        ":type_info",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ir_converter_test",
    srcs = ["ir_converter_test.cc"],
    data = glob(["testdata/*.ir"]),
    deps = [
        ":ir_converter",
        ":parse_and_typecheck",
        "//xls/common:init_xls",
        "//xls/common:update_golden_files",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "highlight_main",
    srcs = ["highlight_main.cc"],
    deps = [
        ":dslx_builtins",
        ":scanner",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "@com_google_absl//absl/strings",
    ],
)

py_test(
    name = "highlight_main_test",
    srcs = ["highlight_main_test.py"],
    data = [
        ":highlight_main",
        "//xls/examples:sha256.x",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

filegroup(
    name = "ir_converter_test_sh",
    srcs = ["ir_converter_test.sh"],
    visibility = ["//xls:xls_users"],
)

cc_library(
    name = "pos",
    srcs = ["pos.cc"],
    hdrs = ["pos.h"],
    deps = [
        "//xls/common/logging",
        "@com_github_google_re2//:re2",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "pos_test",
    srcs = ["pos_test.cc"],
    deps = [
        ":pos",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ast_builtin_types",
    hdrs = ["ast_builtin_types.inc"],
)

cc_library(
    name = "ast",
    srcs = ["ast.cc"],
    hdrs = ["ast.h"],
    deps = [
        ":ast_builtin_types",
        ":pos",
        "//xls/common:casts",
        "//xls/common:indent",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:number_parser",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "ast_test",
    srcs = ["ast_test.cc"],
    deps = [
        ":ast",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "scanner",
    srcs = ["scanner.cc"],
    hdrs = [
        "scanner.h",
        "scanner_keywords.inc",
    ],
    deps = [
        ":pos",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "scanner_test",
    srcs = ["scanner_test.cc"],
    deps = [
        ":scanner",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "bindings",
    srcs = ["bindings.cc"],
    hdrs = ["bindings.h"],
    deps = [
        ":ast",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_github_google_re2//:re2",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "bindings_test",
    srcs = ["bindings_test.cc"],
    deps = [
        ":bindings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "token_parser",
    srcs = ["token_parser.cc"],
    hdrs = ["token_parser.h"],
    deps = [
        ":bindings",
        ":scanner",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "parser",
    srcs = ["parser.cc"],
    hdrs = ["parser.h"],
    deps = [
        ":ast",
        ":bindings",
        ":token_parser",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "parser_test",
    srcs = ["parser_test.cc"],
    deps = [
        ":parser",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parametric_expression",
    srcs = ["parametric_expression.cc"],
    hdrs = ["parametric_expression.h"],
    deps = [
        ":interp_value",
        ":pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "parametric_expression_test",
    srcs = ["parametric_expression_test.cc"],
    deps = [
        ":parametric_expression",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "concrete_type",
    srcs = ["concrete_type.cc"],
    hdrs = ["concrete_type.h"],
    deps = [
        ":ast",
        ":interp_value",
        ":parametric_expression",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_github_google_re2//:re2",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "concrete_type_test",
    srcs = ["concrete_type_test.cc"],
    deps = [
        ":concrete_type",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "interp_value",
    srcs = ["interp_value.cc"],
    hdrs = ["interp_value.h"],
    deps = [
        ":ast",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:value",
    ],
)

cc_test(
    name = "interp_value_test",
    srcs = ["interp_value_test.cc"],
    deps = [
        ":interp_value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "interp_bindings",
    srcs = ["interp_bindings.cc"],
    hdrs = ["interp_bindings.h"],
    deps = [
        ":ast",
        ":interp_value",
        ":type_info",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "builtins",
    srcs = ["builtins.cc"],
    hdrs = ["builtins.h"],
    deps = [
        ":abstract_interpreter",
        ":ast",
        ":interp_value",
        ":type_info",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits_ops",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "abstract_interpreter",
    srcs = ["abstract_interpreter.cc"],
    hdrs = ["abstract_interpreter.h"],
    deps = [
        ":ast",
        ":concrete_type",
        ":import_routines",
        ":interp_bindings",
        ":interp_value",
        ":type_info",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "evaluate",
    srcs = ["evaluate.cc"],
    hdrs = ["evaluate.h"],
    deps = [
        ":abstract_interpreter",
        ":ast",
        ":import_routines",
        ":interp_bindings",
        ":interp_value",
        ":type_info",
        "//xls/common/status:ret_check",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.cc"],
    deps = [
        ":evaluate",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "import_data",
    srcs = ["import_data.cc"],
    hdrs = ["import_data.h"],
    deps = [
        ":ast",
        ":interp_bindings",
        ":type_info",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "import_routines",
    srcs = ["import_routines.cc"],
    hdrs = ["import_routines.h"],
    data = ["//xls/dslx/stdlib:x_files"],
    deps = [
        ":ast",
        ":import_data",
        ":parser",
        ":scanner",
        ":type_info",
        "//xls/common/config:xls_config",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mangle",
    srcs = ["mangle.cc"],
    hdrs = ["mangle.h"],
    deps = [
        ":ast",
        ":symbolic_bindings",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "mangle_test",
    srcs = ["mangle_test.cc"],
    deps = [
        ":mangle",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "interpreter",
    srcs = ["interpreter.cc"],
    hdrs = ["interpreter.h"],
    deps = [
        ":abstract_interpreter",
        ":ast",
        ":builtins",
        ":evaluate",
        ":import_routines",
        ":interp_bindings",
        ":interp_value",
        ":mangle",
        ":type_info",
        "//xls/common/status:ret_check",
        "//xls/ir",
    ],
)

cc_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.cc"],
    deps = [
        ":interpreter",
        ":parse_and_typecheck",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "deduce_ctx",
    srcs = ["deduce_ctx.cc"],
    hdrs = ["deduce_ctx.h"],
    deps = [
        ":concrete_type",
        ":import_routines",
        ":interp_bindings",
        "//xls/common:string_to_int",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "deduce",
    srcs = ["deduce.cc"],
    hdrs = ["deduce.h"],
    deps = [
        ":ast",
        ":concrete_type",
        ":deduce_ctx",
        ":import_routines",
        ":interp_bindings",
        ":interpreter",
        ":parametric_instantiator",
        ":scanner",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "parametric_instantiator",
    srcs = ["parametric_instantiator.cc"],
    hdrs = ["parametric_instantiator.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        ":interpreter",
        ":type_and_bindings",
        ":type_info",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "typecheck",
    srcs = ["typecheck.cc"],
    hdrs = ["typecheck.h"],
    deps = [
        ":deduce",
        ":deduce_ctx",
        ":dslx_builtins",
        ":import_routines",
        ":interpreter",
        ":parametric_instantiator",
        "@com_github_google_re2//:re2",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "typecheck_test",
    srcs = ["typecheck_test.cc"],
    deps = [
        ":parse_and_typecheck",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "dslx_builtins",
    srcs = ["dslx_builtins.cc"],
    hdrs = ["dslx_builtins.h"],
    deps = [
        ":ast",
        ":concrete_type",
        ":deduce_ctx",
        ":interp_value",
        ":parametric_instantiator",
        ":type_and_bindings",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ir_converter",
    srcs = ["ir_converter.cc"],
    hdrs = ["ir_converter.h"],
    deps = [
        ":ast",
        ":builtins",
        ":deduce_ctx",
        ":dslx_builtins",
        ":extract_conversion_order",
        ":import_routines",
        ":interp_value",
        ":interpreter",
        ":mangle",
        ":type_info",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "extract_conversion_order",
    srcs = ["extract_conversion_order.cc"],
    hdrs = ["extract_conversion_order.h"],
    deps = [
        ":ast",
        ":dslx_builtins",
        ":symbolic_bindings",
        ":type_info",
        "//xls/common:symbolized_stacktrace",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "extract_conversion_order_test",
    srcs = ["extract_conversion_order_test.cc"],
    deps = [
        ":extract_conversion_order",
        ":parse_and_typecheck",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "error_printer",
    srcs = ["error_printer.cc"],
    hdrs = ["error_printer.h"],
    deps = [
        ":pos",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "error_printer_test",
    srcs = ["error_printer_test.cc"],
    deps = [
        ":error_printer",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "run_routines",
    srcs = ["run_routines.cc"],
    hdrs = ["run_routines.h"],
    deps = [
        ":bindings",
        ":command_line_utils",
        ":error_printer",
        ":interp_value",
        ":interpreter",
        ":ir_converter",
        ":mangle",
        ":parse_and_typecheck",
        ":symbolic_bindings",
        ":typecheck",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:random_value",
        "//xls/jit:ir_jit",
    ],
)

cc_test(
    name = "run_routines_test",
    srcs = ["run_routines_test.cc"],
    deps = [
        ":run_routines",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/ir:ir_parser",
        "@com_google_googletest//:gtest",
    ],
)

# -- Command line utilities.

cc_library(
    name = "command_line_utils",
    srcs = ["command_line_utils.cc"],
    hdrs = ["command_line_utils.h"],
    deps = [
        ":bindings",
        ":error_printer",
        ":pos",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "ir_converter_main",
    srcs = ["ir_converter_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":command_line_utils",
        ":error_printer",
        ":ir_converter",
        ":parser",
        ":scanner",
        ":typecheck",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "interpreter_main",
    srcs = ["interpreter_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":builtins",
        ":command_line_utils",
        ":error_printer",
        ":interpreter",
        ":ir_converter",
        ":mangle",
        ":parse_and_typecheck",
        ":parser",
        ":run_routines",
        ":scanner",
        ":typecheck",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/jit:ir_jit",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "parse_and_typecheck",
    srcs = ["parse_and_typecheck.cc"],
    hdrs = ["parse_and_typecheck.h"],
    deps = [
        ":ast",
        ":import_routines",
        ":parser",
        ":scanner",
        ":type_info",
        ":typecheck",
    ],
)
