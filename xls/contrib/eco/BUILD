# Copyright 2025 The XLS Authors
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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
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

proto_library(
    name = "ir_patch_proto",
    srcs = ["ir_patch.proto"],
    deps = [
        "//xls/ir:xls_type_proto",
        "//xls/ir:xls_value_proto",
    ],
)

cc_library(
    name = "patch_ir_lib",
    srcs = ["patch_ir.cc"],
    hdrs = ["patch_ir.h"],
    deps = [
        ":ir_patch_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:format_strings",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:xls_type_cc_proto",
        "//xls/ir:xls_value_cc_proto",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "//xls/tools:codegen_flags",
        "//xls/tools:scheduling_options_flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "patch_ir_main",
    srcs = ["patch_ir_main.cc"],
    deps = [
        ":ir_patch_cc_proto",
        ":patch_ir_lib",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_proto_library(
    name = "ir_patch_cc_proto",
    deps = [":ir_patch_proto"],
)

# Graph Edit Distance (GED) and Maximum Common Subgraph (MCS) implementation

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    deps = [
        "//xls/ir:op",
        "//xls/ir:xls_type_cc_proto",
        "//xls/ir:xls_value_cc_proto",
        "//xls/visualization/ir_viz:visualization_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xls_ir_to_graph",
    srcs = ["xls_ir_to_graph.cc"],
    hdrs = ["xls_ir_to_graph.h"],
    deps = [
        ":graph",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:format_strings",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/visualization/ir_viz:node_attribute_visitor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "lap_solver",
    srcs = ["lap_solver.cc"],
    hdrs = ["lap_solver.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "ged",
    srcs = [
        "ged.cc",
        "ged_cost_functions.cc",
    ],
    hdrs = [
        "ged.h",
        "ged_cost_functions.h",
    ],
    deps = [
        ":graph",
        ":lap_solver",
        "//xls/common:stopwatch",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "mcs",
    srcs = ["mcs.cc"],
    hdrs = ["mcs.h"],
    deps = [
        ":graph",
        "//xls/common:stopwatch",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "mcs_test",
    size = "small",
    srcs = ["mcs_test.cc"],
    deps = [
        ":mcs",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "xls_ir_to_graph_test",
    size = "small",
    srcs = ["xls_ir_to_graph_test.cc"],
    deps = [
        ":graph",
        ":xls_ir_to_graph",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "ir_patch_gen_lib",
    srcs = ["ir_patch_gen.cc"],
    hdrs = ["ir_patch_gen.h"],
    deps = [
        ":ged",
        ":graph",
        ":ir_patch_cc_proto",
        "//xls/ir:op",
    ],
)

cc_binary(
    name = "ged_main",
    srcs = ["ged_main.cc"],
    deps = [
        ":ged",
        ":ir_patch_gen_lib",
        ":mcs",
        ":xls_ir_to_graph",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:stopwatch",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)
