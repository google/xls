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

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "verilog_simulator_test",
    srcs = ["verilog_simulator_test.cc"],
    deps = [
        ":verilog_simulator",
        ":verilog_simulators",
        ":verilog_test_base",
        "//xls/codegen:vast",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "verilog_simulator",
    srcs = ["verilog_simulator.cc"],
    hdrs = ["verilog_simulator.h"],
    deps = [
        "//xls/codegen:name_to_bit_count",
        "//xls/codegen:vast",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/tools:verilog_include",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "module_simulator",
    srcs = ["module_simulator.cc"],
    hdrs = ["module_simulator.h"],
    deps = [
        ":module_testbench",
        ":module_testbench_thread",
        ":verilog_simulator",
        "//xls/codegen:flattening",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common/logging",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/tools:eval_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "module_simulator_test",
    srcs = ["module_simulator_test.cc"],
    data = ["input_channel_monitor.v"] + glob([
        "testdata/module_simulator_test_*",
    ]),
    shard_count = 10,
    deps = [
        ":module_simulator",
        ":module_testbench",
        ":verilog_test_base",
        "//xls/codegen:module_signature",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "module_testbench",
    srcs = ["module_testbench.cc"],
    hdrs = ["module_testbench.h"],
    deps = [
        ":module_testbench_thread",
        ":testbench_metadata",
        ":testbench_signal_capture",
        ":testbench_stream",
        ":verilog_simulator",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common:source_location",
        "//xls/common/file:temp_directory",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:number_parser",
        "//xls/ir:source_location",
        "//xls/tools:verilog_include",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "module_testbench_test",
    srcs = ["module_testbench_test.cc"],
    data = glob([
        "testdata/module_testbench_test_*",
    ]),
    shard_count = 10,
    deps = [
        ":module_testbench",
        ":module_testbench_thread",
        ":testbench_signal_capture",
        ":testbench_stream",
        ":verilog_simulator",
        ":verilog_test_base",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/codegen:vast",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:source_location",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "verilog_test_base",
    testonly = True,
    hdrs = [
        "simulation_targets.inc",
        "verilog_test_base.h",
    ],
    deps = [
        ":module_simulator",
        ":module_testbench",
        ":verilog_simulator",
        ":verilog_simulators",
        "//xls/codegen:codegen_options",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common:golden_files",
        "//xls/common:source_location",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:matchers",
        "//xls/tools:verilog_include",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "verilog_simulators",
    srcs = ["verilog_simulators.cc"],
    hdrs = ["verilog_simulators.h"],
    deps = [
        ":verilog_simulator",
        "//xls/simulation/simulators:iverilog_simulator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "verilog_test_base_test",
    srcs = ["verilog_test_base_test.cc"],
    data = glob(["testdata/verilog_test_base_test_*"]),
    deps = [
        ":verilog_test_base",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "module_simulator_codegen_test",
    srcs = ["module_simulator_codegen_test.cc"],
    shard_count = 10,
    deps = [
        ":module_simulator",
        ":verilog_test_base",
        "//xls/codegen:block_generator",
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/codegen:pipeline_generator",
        "//xls/codegen:signature_generator",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "testbench_io_test",
    srcs = ["testbench_io_test.cc"],
    deps = [
        ":verilog_test_base",
        "//xls/common:thread",
        "//xls/common:undeclared_outputs",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:named_pipe",
        "//xls/common/file:temp_directory",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:number_parser",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "check_simulator",
    srcs = ["check_simulator.cc"],
    hdrs = ["check_simulator.h"],
    deps = [
        "//xls/common/logging",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "testbench_metadata",
    srcs = ["testbench_metadata.cc"],
    hdrs = ["testbench_metadata.h"],
    deps = [
        "//xls/codegen:module_signature",
        "//xls/common/logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "testbench_signal_capture",
    srcs = ["testbench_signal_capture.cc"],
    hdrs = ["testbench_signal_capture.h"],
    deps = [
        ":testbench_metadata",
        ":testbench_stream",
        "//xls/common:source_location",
        "//xls/common/logging",
        "//xls/ir:bits",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "module_testbench_thread",
    srcs = ["module_testbench_thread.cc"],
    hdrs = ["module_testbench_thread.h"],
    deps = [
        ":testbench_metadata",
        ":testbench_signal_capture",
        ":testbench_stream",
        "//xls/codegen:vast",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/ir:bits",
        "//xls/ir:source_location",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "testbench_stream",
    srcs = ["testbench_stream.cc"],
    hdrs = ["testbench_stream.h"],
    deps = [
        "//xls/codegen:vast",
        "//xls/common:thread",
        "//xls/common/file:named_pipe",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:number_parser",
        "//xls/ir:source_location",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)
