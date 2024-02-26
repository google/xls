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

cc_library(
    name = "name_to_bit_count",
    hdrs = ["name_to_bit_count.h"],
    deps = [
        "//xls/ir:bits",
        "//xls/ir:type",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "combinational_generator",
    srcs = ["combinational_generator.cc"],
    hdrs = ["combinational_generator.h"],
    deps = [
        ":block_conversion",
        ":block_generator",
        ":codegen_options",
        ":codegen_pass",
        ":codegen_pass_pipeline",
        ":flattening",
        ":module_builder",
        ":module_signature",
        ":name_to_bit_count",
        ":node_expressions",
        ":signature_generator",
        ":vast",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:node_util",
        "//xls/ir:type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "combinational_generator_test",
    srcs = ["combinational_generator_test.cc"],
    data = glob([
        "testdata/combinational_generator_test_*",
    ]),
    shard_count = 50,
    deps = [
        ":combinational_generator",
        ":vast",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "//xls/simulation:module_simulator",
        "//xls/simulation:module_testbench",
        "//xls/simulation:module_testbench_thread",
        "//xls/simulation:verilog_test_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "trace_test",
    srcs = ["trace_test.cc"],
    data = glob([
        "testdata/trace_test_*",
    ]),
    deps = [
        ":codegen_options",
        ":combinational_generator",
        ":pipeline_generator",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimators",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "//xls/simulation:module_testbench",
        "//xls/simulation:module_testbench_thread",
        "//xls/simulation:verilog_simulators",
        "//xls/simulation:verilog_test_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "vast",
    srcs = ["vast.cc"],
    hdrs = ["vast.h"],
    deps = [
        ":module_signature_cc_proto",
        "//xls/common:indent",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:code_template",
        "//xls/ir:format_preference",
        "//xls/ir:source_location",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "finite_state_machine",
    srcs = ["finite_state_machine.cc"],
    hdrs = ["finite_state_machine.h"],
    deps = [
        ":vast",
        "//xls/common:casts",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:source_location",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "vast_test",
    srcs = ["vast_test.cc"],
    deps = [
        ":vast",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:foreign_function",
        "//xls/ir:number_parser",
        "//xls/ir:source_location",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "finite_state_machine_test",
    srcs = ["finite_state_machine_test.cc"],
    data = glob(["testdata/finite_state_machine_test_*"]),
    deps = [
        ":finite_state_machine",
        ":vast",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/simulation:verilog_test_base",
    ],
)

cc_library(
    name = "pipeline_generator",
    srcs = ["pipeline_generator.cc"],
    hdrs = ["pipeline_generator.h"],
    deps = [
        ":block_conversion",
        ":block_generator",
        ":codegen_options",
        ":codegen_pass",
        ":codegen_pass_pipeline",
        ":module_signature",
        ":module_signature_cc_proto",
        ":name_to_bit_count",
        ":vast",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "module_signature",
    srcs = ["module_signature.cc"],
    hdrs = ["module_signature.h"],
    deps = [
        ":module_signature_cc_proto",
        ":verilog_line_map_cc_proto",
        ":xls_metrics_cc_proto",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "node_expressions",
    srcs = ["node_expressions.cc"],
    hdrs = ["node_expressions.h"],
    deps = [
        ":codegen_options",
        ":flattening",
        ":vast",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "flattening",
    srcs = ["flattening.cc"],
    hdrs = ["flattening.h"],
    deps = [
        ":vast",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:xls_type_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lint_annotate",
    srcs = ["lint_annotate.cc"],
    hdrs = ["lint_annotate.h"],
    deps = [
        ":vast",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "module_builder",
    srcs = ["module_builder.cc"],
    hdrs = ["module_builder.h"],
    deps = [
        ":codegen_options",
        ":flattening",
        ":lint_annotate",
        ":module_signature_cc_proto",
        ":node_expressions",
        ":node_representation",
        ":vast",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_strings",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/passes:bdd_function",
        "//xls/passes:bdd_query_engine",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "block_conversion",
    srcs = ["block_conversion.cc"],
    hdrs = ["block_conversion.h"],
    deps = [
        ":bdd_io_analysis",
        ":codegen_checker",
        ":codegen_options",
        ":codegen_pass",
        ":codegen_wrapper_pass",
        ":concurrent_stage_groups",
        ":register_legalization_pass",
        ":vast",
        "//xls/common:casts",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/passes:dce_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "//xls/passes:tuple_simplification_pass",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "codegen_pass",
    srcs = ["codegen_pass.cc"],
    hdrs = ["codegen_pass.h"],
    deps = [
        ":codegen_options",
        ":concurrent_stage_groups",
        ":module_signature",
        "//xls/common/logging",
        "//xls/data_structures:inline_bitmap",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:value",
        "//xls/passes:pass_base",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "codegen_options",
    srcs = ["codegen_options.cc"],
    hdrs = ["codegen_options.h"],
    deps = [
        ":module_signature_cc_proto",
        ":op_override",
        ":ram_configuration",
        "//xls/common:proto_adaptor_utils",
        "//xls/ir:op",
        "//xls/ir:register",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "signature_generation_pass",
    srcs = ["signature_generation_pass.cc"],
    hdrs = ["signature_generation_pass.h"],
    deps = [
        ":codegen_pass",
        ":signature_generator",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "codegen_pass_pipeline",
    srcs = ["codegen_pass_pipeline.cc"],
    hdrs = ["codegen_pass_pipeline.h"],
    deps = [
        ":block_metrics_generation_pass",
        ":block_stitching_pass",
        ":codegen_checker",
        ":codegen_pass",
        ":codegen_wrapper_pass",
        ":ffi_instantiation_pass",
        ":mulp_combining_pass",
        ":port_legalization_pass",
        ":ram_rewrite_pass",
        ":register_legalization_pass",
        ":side_effect_condition_pass",
        ":signature_generation_pass",
        "//xls/ir",
        "//xls/passes:dce_pass",
        "//xls/passes:identity_removal_pass",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "port_legalization_pass",
    srcs = ["port_legalization_pass.cc"],
    hdrs = ["port_legalization_pass.h"],
    deps = [
        ":codegen_pass",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:value_utils",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "signature_generator",
    srcs = ["signature_generator.cc"],
    hdrs = ["signature_generator.h"],
    deps = [
        ":codegen_options",
        ":module_signature",
        "//xls/common:casts",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:type",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "codegen_checker",
    srcs = ["codegen_checker.cc"],
    hdrs = ["codegen_checker.h"],
    deps = [
        ":codegen_pass",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:register",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "block_generator",
    srcs = ["block_generator.cc"],
    hdrs = ["block_generator.h"],
    deps = [
        ":block_conversion",
        ":codegen_options",
        ":flattening",
        ":module_builder",
        ":node_expressions",
        ":node_representation",
        ":op_override",
        ":vast",
        ":verilog_line_map_cc_proto",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:format_preference",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
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
    name = "register_chaining_analysis",
    srcs = ["register_chaining_analysis.cc"],
    hdrs = ["register_chaining_analysis.h"],
    deps = [
        ":codegen_pass",
        "//xls/ir",
        "//xls/ir:register",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_test(
    name = "register_chaining_analysis_test",
    srcs = ["register_chaining_analysis_test.cc"],
    deps = [
        ":register_chaining_analysis",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:register",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "concurrent_stage_groups",
    srcs = ["concurrent_stage_groups.cc"],
    hdrs = ["concurrent_stage_groups.h"],
    deps = [
        "//xls/common/logging",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "register_legalization_pass",
    srcs = ["register_legalization_pass.cc"],
    hdrs = ["register_legalization_pass.h"],
    deps = [
        ":codegen_pass",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:register",
        "//xls/ir:value_utils",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "codegen_wrapper_pass",
    srcs = ["codegen_wrapper_pass.cc"],
    hdrs = ["codegen_wrapper_pass.h"],
    deps = [
        ":codegen_pass",
        "//xls/common/status:status_macros",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

proto_library(
    name = "module_signature_proto",
    srcs = ["module_signature.proto"],
    deps = [
        ":xls_metrics_proto",
        "//xls/ir:channel_proto",
        "//xls/ir:foreign_function_data_proto",
        "//xls/ir:xls_type_proto",
        "@com_google_googleapis//google/api:field_behavior_proto",
    ],
)

cc_proto_library(
    name = "module_signature_cc_proto",
    deps = [":module_signature_proto"],
)

py_proto_library(
    name = "module_signature_py_pb2",
    deps = [":module_signature_proto"],
)

proto_library(
    name = "xls_metrics_proto",
    srcs = ["xls_metrics.proto"],
    deps = ["//xls/ir:op_proto"],
)

cc_proto_library(
    name = "xls_metrics_cc_proto",
    deps = [
        ":xls_metrics_proto",
    ],
)

py_proto_library(
    name = "xls_metrics_py_pb2",
    deps = [":xls_metrics_proto"],
)

proto_library(
    name = "verilog_line_map_proto",
    srcs = ["verilog_line_map.proto"],
)

cc_proto_library(
    name = "verilog_line_map_cc_proto",
    deps = [":verilog_line_map_proto"],
)

py_proto_library(
    name = "verilog_line_map_py_pb2",
    deps = [":verilog_line_map_proto"],
)

cc_test(
    name = "pipeline_generator_test",
    srcs = ["pipeline_generator_test.cc"],
    data = glob([
        "testdata/pipeline_generator_test_*",
    ]),
    shard_count = 50,
    deps = [
        ":module_signature",
        ":module_signature_cc_proto",
        ":pipeline_generator",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "//xls/simulation:module_simulator",
        "//xls/simulation:module_testbench",
        "//xls/simulation:module_testbench_thread",
        "//xls/simulation:verilog_test_base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "module_signature_test",
    srcs = ["module_signature_test.cc"],
    deps = [
        ":module_signature",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:ir_matcher",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "flattening_test",
    srcs = ["flattening_test.cc"],
    deps = [
        ":flattening",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
    ],
)

cc_test(
    name = "module_builder_test",
    srcs = ["module_builder_test.cc"],
    data = glob(["testdata/module_builder_test_*"]),
    shard_count = 10,
    deps = [
        ":module_builder",
        ":vast",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/simulation:verilog_test_base",
    ],
)

cc_test(
    name = "block_conversion_test",
    srcs = ["block_conversion_test.cc"],
    shard_count = 50,
    deps = [
        ":block_conversion",
        ":codegen_options",
        ":codegen_pass",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/interpreter:block_evaluator",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "signature_generator_test",
    srcs = ["signature_generator_test.cc"],
    deps = [
        ":block_conversion",
        ":codegen_options",
        ":codegen_pass",
        ":module_signature",
        ":module_signature_cc_proto",
        ":signature_generator",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
    ],
)

cc_test(
    name = "port_legalization_pass_test",
    srcs = ["port_legalization_pass_test.cc"],
    deps = [
        ":codegen_pass",
        ":port_legalization_pass",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "block_generator_test",
    srcs = ["block_generator_test.cc"],
    data = glob([
        "testdata/block_generator_test_*",
    ]),
    shard_count = 10,
    deps = [
        ":block_conversion",
        ":block_generator",
        ":codegen_options",
        ":codegen_pass",
        ":codegen_pass_pipeline",
        ":module_signature",
        ":op_override_impls",
        ":signature_generator",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/passes:pass_base",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "//xls/simulation:module_simulator",
        "//xls/simulation:module_testbench",
        "//xls/simulation:module_testbench_thread",
        "//xls/simulation:testbench_signal_capture",
        "//xls/simulation:verilog_test_base",
        "//xls/tools:codegen",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:verilog_include",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "register_legalization_pass_test",
    srcs = ["register_legalization_pass_test.cc"],
    deps = [
        ":codegen_pass",
        ":register_legalization_pass",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "codegen_wrapper_pass_test",
    srcs = ["codegen_wrapper_pass_test.cc"],
    deps = [
        ":codegen_pass",
        ":codegen_wrapper_pass",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/passes:dce_pass",
    ],
)

cc_library(
    name = "block_metrics_generation_pass",
    srcs = ["block_metrics_generation_pass.cc"],
    hdrs = ["block_metrics_generation_pass.h"],
    deps = [
        ":block_metrics",
        ":codegen_pass",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "block_metrics",
    srcs = ["block_metrics.cc"],
    hdrs = ["block_metrics.h"],
    deps = [
        ":xls_metrics_cc_proto",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "block_metrics_test",
    srcs = ["block_metrics_test.cc"],
    deps = [
        ":block_conversion",
        ":block_metrics",
        ":codegen_options",
        ":codegen_pass",
        ":xls_metrics_cc_proto",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
    ],
)

cc_library(
    name = "bdd_io_analysis",
    srcs = ["bdd_io_analysis.cc"],
    hdrs = ["bdd_io_analysis.h"],
    deps = [
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:node_util",
        "//xls/passes:bdd_function",
        "//xls/passes:bdd_query_engine",
        "//xls/passes:query_engine",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "lint_annotate_cc",
    srcs = ["lint_annotate.cc"],
    hdrs = ["lint_annotate.h"],
    deps = [":vast"],
)

cc_test(
    name = "bdd_io_analysis_test",
    srcs = ["bdd_io_analysis_test.cc"],
    deps = [
        ":bdd_io_analysis",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/passes:bdd_cse_pass",
        "//xls/passes:dce_pass",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "op_override",
    srcs = [],
    hdrs = ["op_override.h"],
    deps = [
        ":node_representation",
        ":vast",
        "//xls/ir",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "op_override_impls",
    srcs = ["op_override_impls.cc"],
    hdrs = ["op_override_impls.h"],
    deps = [
        ":module_builder",
        ":op_override",
        ":vast",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "node_representation",
    srcs = [],
    hdrs = ["node_representation.h"],
    deps = [
        ":vast",
    ],
)

cc_library(
    name = "mulp_combining_pass",
    srcs = ["mulp_combining_pass.cc"],
    hdrs = ["mulp_combining_pass.h"],
    deps = [
        ":codegen_pass",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "mulp_combining_pass_test",
    srcs = ["mulp_combining_pass_test.cc"],
    deps = [
        ":codegen_pass",
        ":mulp_combining_pass",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
    ],
)

cc_library(
    name = "ffi_instantiation_pass",
    srcs = ["ffi_instantiation_pass.cc"],
    hdrs = ["ffi_instantiation_pass.h"],
    deps = [
        ":codegen_pass",
        ":vast",
        "//xls/common:casts",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:type",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "ffi_instantiation_pass_test",
    srcs = ["ffi_instantiation_pass_test.cc"],
    deps = [
        ":codegen_pass",
        ":ffi_instantiation_pass",
        "//xls/common:casts",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:foreign_function",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "ram_configuration",
    srcs = ["ram_configuration.cc"],
    hdrs = ["ram_configuration.h"],
    deps = [
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ram_rewrite_pass",
    srcs = ["ram_rewrite_pass.cc"],
    hdrs = ["ram_rewrite_pass.h"],
    deps = [
        ":block_conversion",
        ":codegen_pass",
        ":module_signature",
        ":ram_configuration",
        "//xls/common:casts",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "ram_rewrite_pass_test",
    srcs = ["ram_rewrite_pass_test.cc"],
    deps = [
        ":block_conversion",
        ":codegen_options",
        ":codegen_pass",
        ":codegen_pass_pipeline",
        ":ram_configuration",
        ":ram_rewrite_pass",
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:ir_parser",
        "//xls/passes:pass_base",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "side_effect_condition_pass",
    srcs = ["side_effect_condition_pass.cc"],
    hdrs = ["side_effect_condition_pass.h"],
    deps = [
        ":codegen_pass",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "side_effect_condition_pass_test",
    srcs = ["side_effect_condition_pass_test.cc"],
    deps = [
        ":block_conversion",
        ":codegen_options",
        ":codegen_pass",
        ":codegen_pass_pipeline",
        ":side_effect_condition_pass",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimators",
        "//xls/interpreter:block_evaluator",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:register",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/passes:pass_base",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "//xls/scheduling:scheduling_pass",
        "//xls/scheduling:scheduling_pass_pipeline",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "block_stitching_pass",
    srcs = ["block_stitching_pass.cc"],
    hdrs = ["block_stitching_pass.h"],
    deps = [
        ":codegen_pass",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
