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

load("@bazel_skylib//rules:build_test.bzl", "build_test")

# pytype test and library
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")

# Load proto_library
load("//xls/dslx:strip_comments.bzl", "dslx_strip_comments")
load("//xls/fuzzer:build_defs.bzl", "generate_crasher_regression_tests")
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "x_files",
    srcs = glob(["crashers/*.x"]),
)

dslx_strip_comments(
    name = "x_files_stripped",
    srcs = [":x_files"],
    visibility = ["//xls:xls_internal"],
)

py_library(
    name = "cli_utils",
    srcs = ["cli_utils.py"],
)

cc_library(
    name = "run_fuzz",
    srcs = ["run_fuzz.cc"],
    hdrs = ["run_fuzz.h"],
    data = [
        ":sample_runner_main",
        ":summarize_ir_main",
    ],
    deps = [
        ":ast_generator",
        ":cpp_run_fuzz",
        ":sample",
        ":sample_generator",
        ":sample_runner",
        ":sample_summary_cc_proto",
        "//xls/common:stopwatch",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "//xls/tests:testvector_cc_proto",
        "@boringssl//:crypto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "run_fuzz_minimizer_test",
    timeout = "long",
    srcs = ["run_fuzz_minimizer_test.cc"],
    data = [
        "//xls/dev_tools:parse_ir",
    ],
    tags = ["optonly"],
    deps = [
        ":cpp_run_fuzz",
        ":cpp_sample_runner",
        ":run_fuzz",
        ":sample",
        "//xls/common:subprocess",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value_utils",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "run_fuzz_function_test",
    srcs = ["run_fuzz_test.cc"],
    shard_count = 50,
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "run_fuzz_function_wide_test",
    srcs = ["run_fuzz_test.cc"],
    args = ["--wide=true"],
    shard_count = 50,
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "run_fuzz_proc_test",
    srcs = ["run_fuzz_test.cc"],
    args = ["--generate_proc=true"],
    shard_count = 50,
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "run_fuzz_function_codegen_test",
    srcs = ["run_fuzz_codegen_test.cc"],
    shard_count = 50,
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "run_fuzz_proc_codegen_test",
    srcs = ["run_fuzz_codegen_test.cc"],
    args = ["--generate_proc=true"],
    shard_count = 50,
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "run_fuzz_function_wide_codegen_test",
    srcs = ["run_fuzz_codegen_test.cc"],
    args = ["--wide=true"],
    shard_count = 50,
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_binary(
    name = "run_crasher",
    srcs = ["run_crasher.cc"],
    deps = [
        ":run_fuzz",
        ":sample",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "run_crasher_sh",
    srcs = ["run_crasher.sh"],
)

cc_library(
    name = "sample_runner",
    srcs = ["sample_runner.cc"],
    hdrs = ["sample_runner.h"],
    data = [
        "//xls/dslx/ir_convert:ir_converter_main",
        "//xls/tools:codegen_main",
        "//xls/tools:eval_ir_main",
        "//xls/tools:eval_proc_main",
        "//xls/tools:opt_main",
        "//xls/tools:simulate_module_main",
    ],
    deps = [
        ":cpp_sample_runner",
        ":sample",
        ":sample_cc_proto",
        ":sample_summary_cc_proto",
        "//xls/common:revision",
        "//xls/common:stopwatch",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/bytecode:proc_hierarchy_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/public:runtime_build_actions",
        "//xls/simulation:check_simulator",
        "//xls/tests:testvector_cc_proto",
        "//xls/tools:eval_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@re2",
    ],
)

cc_test(
    name = "sample_runner_test",
    srcs = ["sample_runner_test.cc"],
    deps = [
        ":cpp_sample_runner",
        ":sample",
        ":sample_cc_proto",
        ":sample_runner",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/simulation:check_simulator",
        "//xls/tools:eval_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)

# Create tests of all the known non-failing crashers.
# Calling the generated manual target :regression_tests includes the failures
# and :failing_regression_tests only runs the failures.
generate_crasher_regression_tests(
    name = "regression_tests",
    srcs = glob(["crashers/*"]),
    failing = [
        # TODO(https://github.com/google/xls/issues/746): Remove when fixed.
        "crashers/crasher_2022-10-19_a3a3.x",
        # TODO(https://github.com/google/xls/issues/1140): Remove when fixed.
        "crashers/crasher_2023-10-01_eb60.x",
        # TODO(https://github.com/google/xls/issues/1141): Remove when fixed.
        "crashers/crasher_2023-10-02_7c14.x",
        # TODO(https://github.com/google/xls/issues/1609): Remove when fixed.
        "crashers/crasher_2024-09-14_6bbb.x",
        # TODO(https://github.com/google/xls/issues/1618): Remove when fixed.
        "crashers/crasher_2024-09-20_67c5.x",
        # TODO(https://github.com/google/xls/issues/2011): Remove when no longer flaky
        "crashers/crasher_2025-03-19_2e1f.x",
        # TODO(https://github.com/google/xls/issues/2049): Remove when fixed.
        "crashers/crasher_2025-04-23_0cd2.x",
        # TODO(https://github.com/google/xls/issues/2142): Remove when fixed.
        "crashers/crasher_2025-05-07_2d16.x",
        # TODO(https://github.com/google/xls/issues/2458): Remove when fixed.
        "crashers/crasher_2025-06-23_5597.x",
        # TODO(https://github.com/google/xls/issues/2692): Remove when fixed.
        "crashers/crasher_2025-07-25_1957.x",
    ],
    # Tests which are too slow in the unopt-ir interpreter.
    no_unopt_interpreter = [
        # TODO(https://github.com/google/xls/issues/2263): Enormous copies of
        # values in unopt-ir. Remove when fixed.
        "crashers/crasher_2025-05-25_eb23.x",
        # TODO(https://github.com/google/xls/issues/2263): Enormous copies of
        # values in unopt-ir. Remove when fixed.
        "crashers/crasher_2025-05-26_bdec.x",
    ],
    prefix = package_name(),
    tags = {
        # Tests which timeout unless run in opt mode.
        "crashers/crasher_2021-03-27_2259.x": [
            "optonly",
        ],
        "crashers/crasher_2023-07-06_feae.x": [
            "optonly",
        ],
        # Tests which use iverilog and iverilog does not work with sanitizers.
        "crashers/crasher_2025-06-06_9c04.x": [
            "optonly",
        ],
        "crashers/crasher_2025-05-25_eb23.x": [
            "optonly",
        ],
        "crashers/crasher_2022-09-03_4d43.x": [
            "optonly",
        ],
        "crashers/crasher_2024-07-16_ddf8.x": [
            "optonly",
        ],
        "crashers/crasher_2023-11-19_3094.x": [
            "optonly",
        ],
        "crashers/crasher_2022-10-24_40eb.x": [
            "optonly",
        ],
        "crashers/crasher_2022-10-25_133f.x": [
            "optonly",
        ],
        "crashers/multiple-sends.x": [
            "optonly",
        ],
        "crashers/crasher_2022-12-01_9a43.x": [
            "optonly",
        ],
        "crashers/crasher_2023-02-26_ccb9.x": [
            "optonly",
        ],
        "crashers/crasher_2023-03-27_83b8.x": [
            "optonly",
        ],
        "crashers/crasher_2023-06-05_c720.x": [
            "optonly",
        ],
        "crashers/crasher_2023-07-02_c303.x": [
            "optonly",
        ],
        "crashers/crasher_2023-04-09_1f76.x": [
            "optonly",
        ],
        "crashers/crasher_2024-02-23_b88e.x": [
            "optonly",
        ],
        "crashers/crasher_2024-02-25_1b34.x": [
            "optonly",
        ],
        "crashers/crasher_2024-07-08_5973.x": [
            "optonly",
        ],
        "crashers/crasher_2024-07-14_b78c.x": [
            "optonly",
        ],
        "crashers/crasher_2024-09-11_55dd.x": [
            "optonly",
        ],
        "crashers/crasher_2024-09-23_0330.x": [
            "optonly",
        ],
        "crashers/crasher_2024-09-23_0330-with-holdoff.x": [
            "optonly",
        ],
        "crashers/crasher_2024-09-24_6def.x": [
            "optonly",
        ],
    },
)

cc_library(
    name = "run_fuzz_multiprocess_lib",
    srcs = ["run_fuzz_multiprocess.cc"],
    hdrs = ["run_fuzz_multiprocess.h"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "//xls/common:stopwatch",
        "//xls/common:thread",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "sample",
    srcs = ["sample.cc"],
    hdrs = ["sample.h"],
    deps = [
        ":sample_cc_proto",
        ":scrub_crasher",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/tests:testvector_cc_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@re2",
    ],
)

cc_binary(
    name = "run_fuzz_multiprocess",
    srcs = ["run_fuzz_multiprocess_main.cc"],
    deps = [
        ":ast_generator",
        ":run_fuzz_multiprocess_lib",
        ":sample",
        ":sample_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:thread",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "sample_runner_main",
    srcs = ["sample_runner_main.cc"],
    deps = [
        ":sample_runner",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:temp_directory",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "summarize_ir_main",
    srcs = ["summarize_ir_main.cc"],
    deps = [
        ":sample_summary_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:type",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_binary(
    name = "read_summary_main",
    srcs = ["read_summary_main.cc"],
    deps = [
        ":sample_summary_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir:op",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "find_failing_input_main",
    srcs = ["find_failing_input_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/jit:function_jit",
        "//xls/tests:testvector_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

py_test(
    name = "find_failing_input_main_test",
    srcs = ["find_failing_input_main_test.py"],
    data = [
        ":find_failing_input_main",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

cc_proto_library(
    name = "sample_summary_cc_proto",
    deps = [":sample_summary_proto"],
)

py_proto_library(
    name = "sample_summary_py_pb2",
    deps = [":sample_summary_proto"],
)

proto_library(
    name = "sample_summary_proto",
    srcs = ["sample_summary.proto"],
)

py_test(
    name = "run_fuzz_multiprocess_test",
    timeout = "long",
    srcs = ["run_fuzz_multiprocess_test.py"],
    data = [":run_fuzz_multiprocess"],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

py_test(
    name = "fuzz_coverage_test",
    timeout = "eternal",
    srcs = ["fuzz_coverage_test.py"],
    data = [":run_fuzz_multiprocess"],
    shard_count = 3,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":sample_summary_py_pb2",
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/ir:op_py_pb2",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

build_test(
    name = "fuzz_coverage_build_test",
    targets = [":fuzz_coverage_test"],
)

cc_proto_library(
    name = "ast_generator_options_cc_proto",
    deps = [":ast_generator_options_proto"],
)

py_proto_library(
    name = "ast_generator_options_py_pb2",
    deps = [":ast_generator_options_proto"],
)

proto_library(
    name = "ast_generator_options_proto",
    srcs = ["ast_generator_options.proto"],
)

cc_library(
    name = "ast_generator",
    srcs = ["ast_generator.cc"],
    hdrs = ["ast_generator.h"],
    deps = [
        ":ast_generator_options_cc_proto",
        ":value_generator",
        "//xls/common:casts",
        "//xls/common:math_util",
        "//xls/common:random_util",
        "//xls/common:symbolized_stacktrace",
        "//xls/common:test_macros",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc",
        "//xls/dslx/frontend:token",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
    name = "ast_generator_test",
    srcs = ["ast_generator_test.cc"],
    deps = [
        ":ast_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
        "@re2",
    ],
)

cc_library(
    name = "sample_generator",
    srcs = ["sample_generator.cc"],
    hdrs = ["sample_generator.h"],
    deps = [
        ":ast_generator",
        ":sample",
        ":sample_cc_proto",
        ":value_generator",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:unwrap_meta_type",
        "//xls/ir:format_preference",
        "//xls/tests:testvector_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dslx_mutator",
    srcs = ["dslx_mutator.cc"],
    hdrs = ["dslx_mutator.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/frontend:token",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dslx_mutator_test",
    srcs = ["dslx_mutator_test.cc"],
    deps = [
        ":dslx_mutator",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/random:mock_distributions",
        "@com_google_absl//absl/random:mocking_bit_gen",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "scrub_crasher",
    srcs = ["scrub_crasher.cc"],
    hdrs = ["scrub_crasher.h"],
    deps = ["@re2"],
)

cc_test(
    name = "sample_generator_test",
    srcs = ["sample_generator_test.cc"],
    deps = [
        ":ast_generator",
        ":sample",
        ":sample_cc_proto",
        ":sample_generator",
        ":value_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:channel_direction",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

# By default (no flags) the integration test generates only 10 function samples which
# serves as a smoke test for the health of the test itself.
cc_test(
    name = "fuzz_integration_test",
    srcs = ["fuzz_integration_test.cc"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        "//xls/common:stopwatch",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@googletest//:gtest",
    ],
)

# Integration test generates 10 proc samples.
cc_test(
    name = "fuzz_integration_proc_test",
    srcs = ["fuzz_integration_test.cc"],
    args = ["--generate_proc=true"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        "//xls/common:stopwatch",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "value_generator",
    srcs = ["value_generator.cc"],
    hdrs = ["value_generator.h"],
    deps = [
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:inline_bitmap",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:number_parser",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "value_generator_test",
    srcs = ["value_generator_test.cc"],
    deps = [
        ":value_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "sample_test",
    srcs = ["sample_test.cc"],
    deps = [
        ":sample",
        ":sample_cc_proto",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
        "@com_google_absl//absl/log",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "cpp_run_fuzz",
    srcs = ["cpp_run_fuzz.cc"],
    hdrs = ["cpp_run_fuzz.h"],
    data = [
        ":find_failing_input_main",
        ":sample_runner_main",
        "//xls/dev_tools:ir_minimizer_main",
    ],
    deps = [
        ":sample",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "cpp_sample_runner",
    srcs = ["cpp_sample_runner.cc"],
    hdrs = ["cpp_sample_runner.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/ir:format_preference",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_proto_library(
    name = "sample_cc_proto",
    deps = [":sample_proto"],
)

proto_library(
    name = "sample_proto",
    srcs = ["sample.proto"],
    deps = ["//xls/tests:testvector_proto"],
)
