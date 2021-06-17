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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "verilog_simulator",
    srcs = ["verilog_simulator.cc"],
    hdrs = ["verilog_simulator.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:name_to_bit_count",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/file:temp_file",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/tools:verilog_include",
        "@com_github_google_re2//:re2",
    ],
)

cc_library(
    name = "module_simulator",
    srcs = ["module_simulator.cc"],
    hdrs = ["module_simulator.h"],
    deps = [
        ":module_testbench",
        ":verilog_simulator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/codegen:flattening",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common/logging",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:value",
    ],
)

cc_test(
    name = "module_simulator_test",
    srcs = ["module_simulator_test.cc"],
    deps = [
        ":module_simulator",
        ":verilog_simulators",
        ":verilog_test_base",
        "@com_google_absl//absl/status:statusor",
        "//xls/codegen:module_signature",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "module_testbench",
    srcs = ["module_testbench.cc"],
    hdrs = ["module_testbench.h"],
    deps = [
        ":verilog_simulator",
        ":verilog_simulators",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_parser",
        "//xls/ir:number_parser",
        "@com_github_google_re2//:re2",
    ],
)

cc_test(
    name = "module_testbench_test",
    srcs = ["module_testbench_test.cc"],
    deps = [
        ":module_testbench",
        ":verilog_simulators",
        ":verilog_test_base",
        "//xls/codegen:vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "verilog_test_base",
    testonly = True,
    srcs = ["verilog_test_base.cc"],
    hdrs = [
        "simulation_targets.inc",
        "verilog_test_base.h",
    ],
    deps = [
        ":verilog_simulator",
        ":verilog_simulators",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:vast",
        "//xls/common:update_golden_files",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "verilog_simulators",
    srcs = ["verilog_simulators.cc"],
    hdrs = ["verilog_simulators.h"],
    deps = [
        ":verilog_simulator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/simulation/simulators:iverilog_simulator",
    ],
)

cc_test(
    name = "verilog_test_base_test",
    srcs = ["verilog_test_base_test.cc"],
    data = glob(["testdata/verilog_test_base_test_*"]),
    deps = [
        ":verilog_test_base",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "module_simulator_codegen_test",
    srcs = ["module_simulator_codegen_test.cc"],
    deps = [
        ":module_simulator",
        ":verilog_test_base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/codegen:block_generator",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/codegen:pipeline_generator",
        "//xls/codegen:signature_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_googletest//:gtest",
    ],
)
