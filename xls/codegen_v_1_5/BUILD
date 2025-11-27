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
    name = "block_conversion_pass",
    hdrs = ["block_conversion_pass.h"],
    deps = [
        "//xls/codegen:codegen_options",
        "//xls/passes:pass_base",
        "//xls/scheduling:pipeline_schedule_cc_proto",
    ],
)

cc_library(
    name = "block_conversion_pass_pipeline",
    srcs = ["block_conversion_pass_pipeline.cc"],
    hdrs = ["block_conversion_pass_pipeline.h"],
    deps = [
        ":block_conversion_pass",
        ":block_finalization_pass",
        ":channel_to_port_io_lowering_pass",
        ":flow_control_insertion_pass",
        ":param_and_return_value_lowering_pass",
        ":pipeline_register_insertion_pass",
        ":scheduling_pass",
        ":state_to_register_io_lowering_pass",
    ],
)

cc_library(
    name = "block_finalization_pass",
    srcs = ["block_finalization_pass.cc"],
    hdrs = ["block_finalization_pass.h"],
    deps = [
        ":block_conversion_pass",
        "//xls/common:casts",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:name_uniquer",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "block_finalization_pass_test",
    srcs = ["block_finalization_pass_test.cc"],
    deps = [
        ":block_conversion_pass",
        ":block_finalization_pass",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:register",
        "//xls/ir:source_location",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/functional:overload",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "channel_to_port_io_lowering_pass",
    srcs = ["channel_to_port_io_lowering_pass.cc"],
    hdrs = ["channel_to_port_io_lowering_pass.h"],
    deps = [
        ":block_conversion_pass",
        "//xls/ir",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "convert_to_block",
    srcs = ["convert_to_block.cc"],
    hdrs = ["convert_to_block.h"],
    deps = [
        ":block_conversion_pass",
        ":block_conversion_pass_pipeline",
        "//xls/codegen:codegen_options",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/passes:pass_base",
        "//xls/scheduling:scheduling_options",
        "//xls/scheduling:scheduling_result",
        "//xls/tools:codegen",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "convert_to_block_test",
    srcs = ["convert_to_block_test.cc"],
    deps = [
        ":convert_to_block",
        "//xls/codegen:codegen_options",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/scheduling:scheduling_options",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "flow_control_insertion_pass",
    srcs = ["flow_control_insertion_pass.cc"],
    hdrs = ["flow_control_insertion_pass.h"],
    deps = [
        ":block_conversion_pass",
        "//xls/ir",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "param_and_return_value_lowering_pass",
    srcs = ["param_and_return_value_lowering_pass.cc"],
    hdrs = ["param_and_return_value_lowering_pass.h"],
    deps = [
        ":block_conversion_pass",
        "//xls/ir",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "pipeline_register_insertion_pass",
    srcs = ["pipeline_register_insertion_pass.cc"],
    hdrs = ["pipeline_register_insertion_pass.h"],
    deps = [
        ":block_conversion_pass",
        "//xls/ir",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "state_to_register_io_lowering_pass",
    srcs = ["state_to_register_io_lowering_pass.cc"],
    hdrs = ["state_to_register_io_lowering_pass.h"],
    deps = [
        ":block_conversion_pass",
        "//xls/ir",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "scheduling_pass",
    srcs = ["scheduling_pass.cc"],
    hdrs = ["scheduling_pass.h"],
    deps = [
        ":block_conversion_pass",
        "//xls/common:casts",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/passes:pass_base",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "scheduling_pass_test",
    srcs = ["scheduling_pass_test.cc"],
    deps = [
        ":block_conversion_pass",
        ":scheduling_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/passes:pass_base",
        "//xls/scheduling:scheduling_options",
        "//xls/scheduling:scheduling_result",
        "//xls/tools:codegen",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)
