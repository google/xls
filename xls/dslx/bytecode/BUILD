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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:strong_int",
        "//xls/common/status:ret_check",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:bits_ops",
        "//xls/ir:format_strings",
        "//xls/ir:number_parser",
        "@com_github_google_re2//:re2",
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
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_helpers",
        "//xls/dslx:make_value_format_descriptor",
        "//xls/dslx:value_format_descriptor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:concrete_type_zero_value",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
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
        "@com_google_absl//absl/types:optional",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:parametric_env",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "bytecode_interpreter",
    srcs = ["bytecode_interpreter.cc"],
    hdrs = ["bytecode_interpreter.h"],
    deps = [
        ":bytecode",
        ":bytecode_emitter",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_helpers",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:parametric_env",
        "//xls/ir:bits_ops",
    ],
)

cc_test(
    name = "bytecode_interpreter_test",
    srcs = ["bytecode_interpreter_test.cc"],
    deps = [
        ":bytecode_emitter",
        ":bytecode_interpreter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:run_routines",
        "@com_google_googletest//:gtest",
    ],
)
