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

load("@rules_python//python:proto.bzl", "py_proto_library")
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "pipeline_schedule_proto",
    srcs = ["pipeline_schedule.proto"],
)

py_proto_library(
    name = "pipeline_schedule_py_pb2",
    deps = [":pipeline_schedule_proto"],
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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:type",
        "//xls/ir:value_utils",
    ],
)

cc_test(
    name = "extract_stage_test",
    srcs = ["extract_stage_test.cc"],
    deps = [
        ":extract_stage",
        ":run_pipeline_schedule",
        ":scheduling_options",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
    ],
)

cc_library(
    name = "scheduling_options",
    hdrs = ["scheduling_options.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/tools:scheduling_options_flags_cc_proto",
    ],
)

cc_library(
    name = "min_cut_scheduler",
    srcs = ["min_cut_scheduler.cc"],
    hdrs = ["min_cut_scheduler.h"],
    deps = [
        ":function_partition",
        ":schedule_bounds",
        ":scheduling_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:node_util",
    ],
)

cc_test(
    name = "min_cut_scheduler_test",
    srcs = ["min_cut_scheduler_test.cc"],
    deps = [
        ":min_cut_scheduler",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "sdc_scheduler",
    srcs = ["sdc_scheduler.cc"],
    hdrs = ["sdc_scheduler.h"],
    deps = [
        ":scheduling_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "@com_google_ortools//ortools/math_opt/cpp:math_opt",
        "@com_google_ortools//ortools/math_opt/solvers:glop_solver",
    ],
)

cc_library(
    name = "pipeline_schedule",
    srcs = ["pipeline_schedule.cc"],
    hdrs = ["pipeline_schedule.h"],
    deps = [
        ":pipeline_schedule_cc_proto",
        ":scheduling_options",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/fdo:delay_manager",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:node_util",
        "//xls/ir:op",
    ],
)

cc_library(
    name = "run_pipeline_schedule",
    srcs = ["run_pipeline_schedule.cc"],
    hdrs = ["run_pipeline_schedule.h"],
    deps = [
        ":min_cut_scheduler",
        ":pipeline_schedule",
        ":schedule_bounds",
        ":scheduling_options",
        ":sdc_scheduler",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:binary_search",
        "//xls/delay_model:delay_estimator",
        "//xls/fdo:delay_manager",
        "//xls/fdo:iterative_sdc_scheduler",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "//xls/ir:op",
    ],
)

cc_test(
    name = "pipeline_schedule_test",
    srcs = ["pipeline_schedule_test.cc"],
    deps = [
        ":pipeline_schedule",
        ":pipeline_schedule_cc_proto",
        ":run_pipeline_schedule",
        ":scheduling_options",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/fdo:delay_manager",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
    ],
)

cc_test(
    name = "schedule_bounds_test",
    srcs = ["schedule_bounds_test.cc"],
    deps = [
        ":schedule_bounds",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
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
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/data_structures:min_cut",
        "//xls/ir",
    ],
)

cc_library(
    name = "proc_state_legalization_pass",
    srcs = ["proc_state_legalization_pass.cc"],
    hdrs = ["proc_state_legalization_pass.h"],
    deps = [
        ":scheduling_pass",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/solvers:z3_ir_translator",
    ],
)

cc_test(
    name = "proc_state_legalization_pass_test",
    srcs = ["proc_state_legalization_pass_test.cc"],
    deps = [
        ":proc_state_legalization_pass",
        ":scheduling_options",
        ":scheduling_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/passes:cse_pass",
        "//xls/passes:dce_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
    ],
)

cc_library(
    name = "scheduling_pass_pipeline",
    srcs = ["scheduling_pass_pipeline.cc"],
    hdrs = ["scheduling_pass_pipeline.h"],
    deps = [
        ":mutual_exclusion_pass",
        ":pipeline_scheduling_pass",
        ":proc_state_legalization_pass",
        ":scheduling_checker",
        ":scheduling_pass",
        ":scheduling_wrapper_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/passes:dce_pass",
        "//xls/passes:literal_uncommoning_pass",
        "//xls/passes:optimization_pass_pipeline",
    ],
)

cc_library(
    name = "scheduling_wrapper_pass",
    srcs = ["scheduling_wrapper_pass.cc"],
    hdrs = ["scheduling_wrapper_pass.h"],
    deps = [
        ":pipeline_schedule",
        ":scheduling_pass",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
    ],
)

cc_test(
    name = "scheduling_wrapper_pass_test",
    srcs = ["scheduling_wrapper_pass_test.cc"],
    deps = [
        ":pipeline_schedule",
        ":pipeline_scheduling_pass",
        ":scheduling_options",
        ":scheduling_pass",
        ":scheduling_wrapper_pass",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:source_location",
        "//xls/passes:dce_pass",
        "//xls/passes:dfe_pass",
        "//xls/passes:inlining_pass",
        "//xls/passes:literal_uncommoning_pass",
        "//xls/passes:optimization_pass",
    ],
)

cc_library(
    name = "scheduling_checker",
    srcs = ["scheduling_checker.cc"],
    hdrs = ["scheduling_checker.h"],
    deps = [
        ":pipeline_schedule",
        ":scheduling_pass",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/passes:optimization_pass",
    ],
)

cc_library(
    name = "scheduling_pass",
    srcs = ["scheduling_pass.cc"],
    hdrs = ["scheduling_pass.h"],
    deps = [
        ":pipeline_schedule",
        ":scheduling_options",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "//xls/passes:pass_base",
    ],
)

cc_library(
    name = "mutual_exclusion_pass",
    srcs = ["mutual_exclusion_pass.cc"],
    hdrs = ["mutual_exclusion_pass.h"],
    deps = [
        ":scheduling_options",
        ":scheduling_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:casts",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:graph_coloring",
        "//xls/data_structures:transitive_closure",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:channel",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/passes:optimization_pass",
        "//xls/passes:post_dominator_analysis",
        "//xls/passes:token_provenance_analysis",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
        "@z3//:api",
    ],
)

cc_test(
    name = "mutual_exclusion_pass_test",
    srcs = ["mutual_exclusion_pass_test.cc"],
    deps = [
        ":mutual_exclusion_pass",
        ":scheduling_options",
        ":scheduling_pass",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/ir:value",
        "//xls/passes:cse_pass",
        "//xls/passes:dce_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
    ],
)

cc_library(
    name = "pipeline_scheduling_pass",
    srcs = ["pipeline_scheduling_pass.cc"],
    hdrs = ["pipeline_scheduling_pass.h"],
    deps = [
        ":pipeline_schedule",
        ":run_pipeline_schedule",
        ":scheduling_options",
        ":scheduling_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_test(
    name = "pipeline_scheduling_pass_test",
    srcs = ["pipeline_scheduling_pass_test.cc"],
    data = [
        "@org_theopenroadproject//:opensta",
        "@com_google_skywater_pdk_sky130_fd_sc_hd//:sky130_fd_sc_hd",
        "@at_clifford_yosys//:yosys",
    ],
    deps = [
        ":pipeline_scheduling_pass",
        ":scheduling_options",
        ":scheduling_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel_ops",
        "//xls/ir:foreign_function",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "//xls/tools:scheduling_options_flags",
    ],
)
