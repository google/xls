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

# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "pipeline_schedule_proto",
    srcs = ["pipeline_schedule.proto"],
)

cc_proto_library(
    name = "pipeline_schedule_cc_proto",
    deps = [":pipeline_schedule_proto"],
)

cc_library(
    name = "extract_stage",
    srcs = ["extract_stage.cc"],
    hdrs = ["extract_stage.h"],
    deps = [
        ":pipeline_schedule",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_test(
    name = "extract_stage_test",
    srcs = ["extract_stage_test.cc"],
    deps = [
        ":extract_stage",
        "//xls/common/status:matchers",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pipeline_schedule",
    srcs = ["pipeline_schedule.cc"],
    hdrs = ["pipeline_schedule.h"],
    deps = [
        ":function_partition",
        ":pipeline_schedule_cc_proto",
        ":schedule_bounds",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/data_structures:binary_search",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
    ],
)

cc_test(
    name = "pipeline_schedule_test",
    srcs = ["pipeline_schedule_test.cc"],
    deps = [
        ":pipeline_schedule",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/ir:type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "schedule_bounds_test",
    srcs = ["schedule_bounds_test.cc"],
    deps = [
        ":schedule_bounds",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "function_partition_test",
    srcs = ["function_partition_test.cc"],
    # 2020-01-08: //xls/scheduling/function_partition_test \
    #               --gtest_list_tests
    # shows 5
    shard_count = 5,
    deps = [
        ":function_partition",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "schedule_bounds",
    srcs = ["schedule_bounds.cc"],
    hdrs = ["schedule_bounds.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:integral_types",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
    ],
)

cc_library(
    name = "function_partition",
    srcs = ["function_partition.cc"],
    hdrs = ["function_partition.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:vlog_is_on",
        "//xls/data_structures:min_cut",
        "//xls/ir",
    ],
)

cc_library(
    name = "scheduling_pass",
    srcs = ["scheduling_pass.cc"],
    hdrs = ["scheduling_pass.h"],
    deps = [
        ":pipeline_schedule",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/passes:pass_base",
    ],
)

cc_library(
    name = "pipeline_scheduling_pass",
    srcs = ["pipeline_scheduling_pass.cc"],
    hdrs = ["pipeline_scheduling_pass.h"],
    deps = [
        ":scheduling_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_library(
    name = "scheduling_checker",
    srcs = ["scheduling_checker.cc"],
    hdrs = ["scheduling_checker.h"],
    deps = [
        ":scheduling_pass",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/passes",
    ],
)
