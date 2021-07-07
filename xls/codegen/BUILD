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
load("//xls/build_rules:py_proto_library.bzl", "xls_py_proto_library")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "name_to_bit_count",
    hdrs = ["name_to_bit_count.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//xls/ir:bits",
        "//xls/ir:type",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:node_util",
        "//xls/ir:type",
    ],
)

cc_test(
    name = "combinational_generator_test",
    srcs = ["combinational_generator_test.cc"],
    data = glob([
        "testdata/combinational_generator_test_*",
    ]),
    shard_count = 10,
    deps = [
        ":combinational_generator",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:random_value",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
        "//xls/simulation:verilog_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "sequential_generator",
    srcs = ["sequential_generator.cc"],
    hdrs = ["sequential_generator.h"],
    deps = [
        ":finite_state_machine",
        ":module_builder",
        ":module_signature",
        ":module_signature_cc_proto",
        ":pipeline_generator",
        ":vast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:type",
        "//xls/passes",
        "//xls/scheduling:pipeline_schedule",
    ],
)

cc_test(
    name = "sequential_generator_test",
    srcs = ["sequential_generator_test.cc"],
    data = glob([
        "testdata/sequential_generator_test_*",
    ]),
    shard_count = 10,
    deps = [
        ":module_builder",
        ":module_signature",
        ":module_signature_cc_proto",
        ":pipeline_generator",
        ":sequential_generator",
        ":vast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/scheduling:pipeline_schedule",
        "//xls/simulation:module_simulator",
        "//xls/simulation:module_testbench",
        "//xls/simulation:verilog_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "vast",
    srcs = ["vast.cc"],
    hdrs = ["vast.h"],
    deps = [
        ":module_signature_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "//xls/common:indent",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "@com_github_google_re2//:re2",
    ],
)

cc_library(
    name = "finite_state_machine",
    srcs = ["finite_state_machine.cc"],
    hdrs = ["finite_state_machine.h"],
    deps = [
        ":vast",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//xls/common:casts",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "vast_test",
    srcs = ["vast_test.cc"],
    deps = [
        ":vast",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:number_parser",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "finite_state_machine_test",
    srcs = ["finite_state_machine_test.cc"],
    data = glob(["testdata/finite_state_machine_test_*"]),
    deps = [
        ":finite_state_machine",
        ":vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/simulation:verilog_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "pipeline_generator",
    srcs = ["pipeline_generator.cc"],
    hdrs = ["pipeline_generator.h"],
    deps = [
        ":finite_state_machine",
        ":flattening",
        ":module_builder",
        ":module_signature",
        ":module_signature_cc_proto",
        ":name_to_bit_count",
        ":node_expressions",
        ":vast",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/scheduling:pipeline_schedule",
    ],
)

cc_library(
    name = "module_signature",
    srcs = ["module_signature.cc"],
    hdrs = ["module_signature.h"],
    deps = [
        ":module_signature_cc_proto",
        ":vast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:type",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "node_expressions",
    srcs = ["node_expressions.cc"],
    hdrs = ["node_expressions.h"],
    deps = [
        ":flattening",
        ":vast",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:type",
    ],
)

cc_library(
    name = "flattening",
    srcs = ["flattening.cc"],
    hdrs = ["flattening.h"],
    deps = [
        ":vast",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:xls_type_cc_proto",
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
        ":flattening",
        ":lint_annotate",
        ":node_expressions",
        ":vast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_github_google_re2//:re2",
    ],
)

cc_library(
    name = "state_removal_pass",
    srcs = ["state_removal_pass.cc"],
    hdrs = ["state_removal_pass.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/passes",
        "//xls/passes:pass_base",
    ],
)

cc_library(
    name = "block_conversion",
    srcs = ["block_conversion.cc"],
    hdrs = ["block_conversion.h"],
    deps = [
        ":vast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:node_util",
        "//xls/ir:type",
        "//xls/ir:value_helpers",
        "//xls/passes:dce_pass",
        "//xls/passes:tuple_simplification_pass",
        "//xls/scheduling:pipeline_schedule",
    ],
)

cc_library(
    name = "codegen_pass",
    srcs = ["codegen_pass.cc"],
    hdrs = ["codegen_pass.h"],
    deps = [
        ":codegen_options",
        ":module_signature",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//xls/ir",
        "//xls/passes:pass_base",
        "//xls/scheduling:pipeline_schedule",
    ],
)

cc_library(
    name = "codegen_options",
    srcs = ["codegen_options.cc"],
    hdrs = ["codegen_options.h"],
    deps = [
        ":module_signature_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//xls/common:proto_adaptor_utils",
        "//xls/scheduling:pipeline_schedule",
    ],
)

cc_library(
    name = "signature_generation_pass",
    srcs = ["signature_generation_pass.cc"],
    hdrs = ["signature_generation_pass.h"],
    deps = [
        ":codegen_pass",
        ":signature_generator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
    ],
)

cc_library(
    name = "codegen_pass_pipeline",
    srcs = ["codegen_pass_pipeline.cc"],
    hdrs = ["codegen_pass_pipeline.h"],
    deps = [
        ":codegen_checker",
        ":codegen_pass",
        ":port_legalization_pass",
        ":signature_generation_pass",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "port_legalization_pass",
    srcs = ["port_legalization_pass.cc"],
    hdrs = ["port_legalization_pass.h"],
    deps = [
        ":codegen_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:value_helpers",
    ],
)

cc_library(
    name = "signature_generator",
    srcs = ["signature_generator.cc"],
    hdrs = ["signature_generator.h"],
    deps = [
        ":codegen_options",
        ":module_signature",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
        "//xls/scheduling:pipeline_schedule",
    ],
)

cc_library(
    name = "codegen_checker",
    srcs = ["codegen_checker.cc"],
    hdrs = ["codegen_checker.h"],
    deps = [
        ":codegen_pass",
        "@com_google_absl//absl/status",
        "//xls/ir",
    ],
)

cc_library(
    name = "block_generator",
    srcs = ["block_generator.cc"],
    hdrs = ["block_generator.h"],
    deps = [
        ":codegen_options",
        ":flattening",
        ":module_builder",
        ":vast",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging:log_lines",
        "//xls/ir",
        "//xls/ir:node_util",
    ],
)

proto_library(
    name = "module_signature_proto",
    srcs = ["module_signature.proto"],
    deps = ["//xls/ir:xls_type_proto"],
)

cc_proto_library(
    name = "module_signature_cc_proto",
    deps = [":module_signature_proto"],
)

xls_py_proto_library(
    name = "module_signature_py_pb2",
    srcs = ["module_signature.proto"],
    internal_deps = [
        ":module_signature_proto",
        "//xls/ir:xls_type_proto",
    ],
    deps = [
        "//xls/ir:xls_type_py_pb2",
    ],
)

cc_test(
    name = "pipeline_generator_test",
    srcs = ["pipeline_generator_test.cc"],
    data = glob([
        "testdata/pipeline_generator_test_*",
    ]),
    shard_count = 10,
    deps = [
        ":flattening",
        ":pipeline_generator",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/scheduling:pipeline_schedule",
        "//xls/simulation:module_simulator",
        "//xls/simulation:module_testbench",
        "//xls/simulation:verilog_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "module_signature_test",
    srcs = ["module_signature_test.cc"],
    deps = [
        ":module_signature",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "flattening_test",
    srcs = ["flattening_test.cc"],
    deps = [
        ":flattening",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
        "@com_google_googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/simulation:verilog_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "state_removal_pass_test",
    srcs = ["state_removal_pass_test.cc"],
    deps = [
        ":state_removal_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:proc_network_interpreter",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "block_conversion_test",
    srcs = ["block_conversion_test.cc"],
    deps = [
        ":block_conversion",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signature_generator_test",
    srcs = ["signature_generator_test.cc"],
    deps = [
        ":codegen_options",
        ":module_signature_cc_proto",
        ":signature_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:type",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "port_legalization_pass_test",
    srcs = ["port_legalization_pass_test.cc"],
    deps = [
        ":codegen_pass",
        ":port_legalization_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "block_generator_test",
    srcs = ["block_generator_test.cc"],
    data = glob([
        "testdata/block_generator_test_*",
    ]),
    deps = [
        ":block_generator",
        ":signature_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:channel_cc_proto",
        "//xls/ir:function_builder",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_test_base",
    ],
)
