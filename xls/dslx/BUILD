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

# pytype tests are present in this file

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "constexpr_evaluator",
    srcs = ["constexpr_evaluator.cc"],
    hdrs = ["constexpr_evaluator.h"],
    deps = [
        ":errors",
        ":import_data",
        ":interp_value",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:builtins_metadata",
        "//xls/dslx/type_system:concrete_type_zero_value",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "constexpr_evaluator_test",
    srcs = ["constexpr_evaluator_test.cc"],
    deps = [
        ":constexpr_evaluator",
        ":create_import_data",
        ":import_data",
        ":parse_and_typecheck",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:typecheck",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "cpp_transpiler",
    srcs = ["cpp_transpiler.cc"],
    hdrs = ["cpp_transpiler.h"],
    deps = [
        ":import_data",
        "//xls/common:case_converters",
        "//xls/common:visitor",
        "//xls/common/status:status_macros",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:typecheck",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "cpp_transpiler_test",
    srcs = ["cpp_transpiler_test.cc"],
    deps = [
        ":cpp_transpiler",
        ":create_import_data",
        ":import_data",
        ":parse_and_typecheck",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "create_import_data",
    srcs = ["create_import_data.cc"],
    hdrs = ["create_import_data.h"],
    deps = [
        ":default_dslx_stdlib_path",
        ":import_data",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_cache",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    deps = [
        ":import_record",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:concrete_type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "interp_value_helpers",
    srcs = ["interp_value_helpers.cc"],
    hdrs = ["interp_value_helpers.h"],
    deps = [
        ":interp_value",
        "//xls/common/status:ret_check",
        "//xls/dslx/type_system:concrete_type",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "interp_value_helpers_test",
    srcs = ["interp_value_helpers_test.cc"],
    deps = [
        ":interp_value",
        ":interp_value_helpers",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:concrete_type",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "trait_visitor",
    srcs = ["trait_visitor.cc"],
    hdrs = ["trait_visitor.h"],
    deps = [
        "//xls/dslx/frontend:ast",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "value_format_descriptor",
    hdrs = ["value_format_descriptor.h"],
    deps = [
        "//xls/ir:bits",
        "//xls/ir:format_preference",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "interp_value",
    srcs = ["interp_value.cc"],
    hdrs = ["interp_value.h"],
    deps = [
        ":value_format_descriptor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:value",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "interp_value_test",
    srcs = ["interp_value_test.cc"],
    deps = [
        ":interp_value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "interp_bindings",
    srcs = ["interp_bindings.cc"],
    hdrs = ["interp_bindings.h"],
    deps = [
        ":interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "import_record",
    hdrs = ["import_record.h"],
    deps = [
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "import_data",
    srcs = ["import_data.cc"],
    hdrs = ["import_data.h"],
    deps = [
        ":default_dslx_stdlib_path",
        ":errors",
        ":import_record",
        ":interp_bindings",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/dslx/bytecode:bytecode_cache_interface",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "import_routines",
    srcs = ["import_routines.cc"],
    hdrs = ["import_routines.h"],
    data = ["//xls/dslx/stdlib:x_files"],
    deps = [
        ":import_data",
        "//xls/common/config:xls_config",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "mangle",
    srcs = ["mangle.cc"],
    hdrs = ["mangle.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:parametric_env",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
    name = "dslx_builtins",
    srcs = ["dslx_builtins.cc"],
    hdrs = ["dslx_builtins.h"],
    deps = [
        ":errors",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:builtins_metadata",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:deduce_ctx",
        "//xls/dslx/type_system:parametric_instantiator",
        "//xls/dslx/type_system:type_and_bindings",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "error_printer",
    srcs = ["error_printer.cc"],
    hdrs = ["error_printer.h"],
    deps = [
        ":warning_collector",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "run_routines",
    srcs = ["run_routines.cc"],
    hdrs = ["run_routines.h"],
    deps = [
        ":command_line_utils",
        ":create_import_data",
        ":default_dslx_stdlib_path",
        ":error_printer",
        ":errors",
        ":interp_value",
        ":interp_value_helpers",
        ":mangle",
        ":parse_and_typecheck",
        "//xls/common:test_macros",
        "//xls/dslx/bytecode:bytecode_cache",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:bindings",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:typecheck",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/jit:function_jit",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parse_and_typecheck",
    srcs = ["parse_and_typecheck.cc"],
    hdrs = ["parse_and_typecheck.h"],
    deps = [
        ":import_routines",
        ":warning_collector",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:typecheck",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "default_dslx_stdlib_path",
    srcs = ["default_dslx_stdlib_path.cc"],
    hdrs = ["default_dslx_stdlib_path.h"],
)

cc_library(
    name = "command_line_utils",
    srcs = ["command_line_utils.cc"],
    hdrs = ["command_line_utils.h"],
    deps = [
        ":error_printer",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/dslx/frontend:bindings",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

#  -- DSLX specific tool binaries / command line utilities

cc_binary(
    name = "interpreter_main",
    srcs = ["interpreter_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":command_line_utils",
        ":error_printer",
        ":run_routines",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "cpp_transpiler_main",
    srcs = ["cpp_transpiler_main.cc"],
    visibility = ["//xls:xls_public"],
    deps = [
        ":cpp_transpiler",
        ":create_import_data",
        ":default_dslx_stdlib_path",
        ":import_data",
        ":parse_and_typecheck",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "highlight_main",
    srcs = ["highlight_main.cc"],
    deps = [
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/dslx/frontend:builtins_metadata",
        "//xls/dslx/frontend:scanner",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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

py_test(
    name = "value_test",
    srcs = ["value_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/dslx/python:interp_value",
        "//xls/dslx/python:interpreter",
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
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/dslx/python:interpreter",
    ],
)

cc_library(
    name = "warning_collector",
    hdrs = ["warning_collector.h"],
    deps = [
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "make_value_format_descriptor",
    srcs = ["make_value_format_descriptor.cc"],
    hdrs = ["make_value_format_descriptor.h"],
    deps = [
        ":value_format_descriptor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/type_system:concrete_type",
    ],
)

cc_library(
    name = "channel_direction",
    hdrs = ["channel_direction.h"],
)
