# Copyright 2023 The XLS Authors
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

# Bytecode interpreter.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bytecode",
    srcs = ["bytecode.cc"],
    hdrs = ["bytecode.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:strong_int",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx:value_format_descriptor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:format_strings",
        "//xls/ir:number_parser",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "bytecode_cache",
    srcs = ["bytecode_cache.cc"],
    hdrs = ["bytecode_cache.h"],
    deps = [
        ":bytecode",
        ":bytecode_cache_interface",
        ":bytecode_emitter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_library(
    name = "bytecode_cache_interface",
    hdrs = ["bytecode_cache_interface.h"],
    deps = [
        ":bytecode",
        "@com_google_absl//absl/status:statusor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_library(
    name = "bytecode_emitter",
    srcs = ["bytecode_emitter.cc"],
    hdrs = ["bytecode_emitter.h"],
    deps = [
        ":bytecode",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:casts",
        "//xls/common:symbolized_stacktrace",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx:make_value_format_descriptor",
        "//xls/dslx:value_format_descriptor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:bits",
        "//xls/ir:format_preference",
        "//xls/ir:format_strings",
    ],
)

cc_test(
    name = "bytecode_emitter_test",
    srcs = ["bytecode_emitter_test.cc"],
    deps = [
        ":bytecode",
        ":bytecode_emitter",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "interpreter_stack",
    srcs = ["interpreter_stack.cc"],
    hdrs = ["interpreter_stack.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/dslx:interp_value",
    ],
)

cc_test(
    name = "interpreter_stack_test",
    srcs = ["interpreter_stack_test.cc"],
    deps = [
        ":interpreter_stack",
        "@com_google_absl//absl/status",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
    ],
)

cc_library(
    name = "bytecode_interpreter_options",
    hdrs = ["bytecode_interpreter_options.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:parametric_env",
        "//xls/ir:format_preference",
    ],
)

cc_library(
    name = "frame",
    srcs = ["frame.cc"],
    hdrs = ["frame.h"],
    deps = [
        ":bytecode",
        "@com_google_absl//absl/log:check",
        "//xls/common/logging",
        "//xls/dslx:interp_value",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_library(
    name = "builtins",
    srcs = ["builtins.cc"],
    hdrs = ["builtins.h"],
    deps = [
        ":bytecode",
        ":bytecode_interpreter_options",
        ":frame",
        ":interpreter_stack",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
    ],
)

cc_library(
    name = "bytecode_interpreter",
    srcs = ["bytecode_interpreter.cc"],
    hdrs = ["bytecode_interpreter.h"],
    deps = [
        ":builtins",
        ":bytecode",
        ":bytecode_cache_interface",
        ":bytecode_emitter",
        ":bytecode_interpreter_options",
        ":frame",
        ":interpreter_stack",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:dslx_builtins",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx:value_format_descriptor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:big_int",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:format_strings",
    ],
)

cc_test(
    name = "bytecode_interpreter_test",
    srcs = ["bytecode_interpreter_test.cc"],
    deps = [
        ":builtins",
        ":bytecode",
        ":bytecode_emitter",
        ":bytecode_interpreter",
        ":bytecode_interpreter_options",
        ":interpreter_stack",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/run_routines",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:format_preference",
        "//xls/ir:format_strings",
    ],
)
