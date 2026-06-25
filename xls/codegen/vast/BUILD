# Copyright 2024 The XLS Authors
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "vast",
    srcs = ["vast.cc"],
    hdrs = ["vast.h"],
    deps = [
        ":verilog_keywords",
        "//xls/common:indent",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:code_template",
        "//xls/ir:format_preference",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:die_if_null",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@abseil-cpp//absl/types:variant",
    ],
)

cc_test(
    name = "vast_test",
    srcs = ["vast_test.cc"],
    deps = [
        ":vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:format_preference",
        "//xls/ir:number_parser",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "infer_vast_types",
    srcs = ["infer_vast_types.cc"],
    hdrs = ["infer_vast_types.h"],
    deps = [
        ":fold_vast_constants",
        ":vast",
        "//xls/common/status:status_macros",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "infer_vast_types_test",
    srcs = ["infer_vast_types_test.cc"],
    deps = [
        ":infer_vast_types",
        ":vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:number_parser",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "fold_vast_constants",
    srcs = ["fold_vast_constants.cc"],
    hdrs = ["fold_vast_constants.h"],
    deps = [
        ":vast",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:format_preference",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/numeric:bits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "fold_vast_constants_test",
    srcs = ["fold_vast_constants_test.cc"],
    deps = [
        ":fold_vast_constants",
        ":vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:number_parser",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "dslx_type_fixer",
    srcs = ["dslx_type_fixer.cc"],
    hdrs = ["dslx_type_fixer.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system_v2:inference_table",
        "//xls/dslx/type_system_v2:type_annotation_utils",
        "//xls/dslx/type_system_v2:type_inference_error_handler",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "dslx_builder",
    srcs = ["dslx_builder.cc"],
    hdrs = ["dslx_builder.h"],
    deps = [
        ":dslx_type_fixer",
        ":fold_vast_constants",
        ":vast",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/fmt:ast_fmt",
        "//xls/dslx/fmt:comments",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:comment_data",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/frontend:semantics_analysis",
        "//xls/dslx/ir_convert:convert_options",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system_v2:type_inference_error_handler",
        "//xls/dslx/type_system_v2:typecheck_module_v2",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:variant",
    ],
)

cc_library(
    name = "translate_vast_to_dslx",
    srcs = ["translate_vast_to_dslx.cc"],
    hdrs = ["translate_vast_to_dslx.h"],
    deps = [
        ":dslx_builder",
        ":fold_vast_constants",
        ":infer_vast_types",
        ":vast",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:trait_visitor",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:token",
        "//xls/dslx/type_system_v2:type_annotation_utils",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "verilog_keywords",
    srcs = ["verilog_keywords.cc"],
    hdrs = ["verilog_keywords.h"],
    deps = [
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "translate_vast_to_dslx_test",
    srcs = ["translate_vast_to_dslx_test.cc"],
    deps = [
        ":translate_vast_to_dslx",
        ":vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:number_parser",
        "//xls/ir:source_location",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest",
    ],
)
