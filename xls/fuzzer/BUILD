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
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_library")
load("//xls/dslx:strip_comments.bzl", "dslx_strip_comments")
load("//xls/fuzzer:build_defs.bzl", "generate_crasher_regression_tests")

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

pytype_strict_library(
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@boringssl//:crypto",
        "@protobuf",
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
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
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
        "//xls/public:runtime_dslx_actions",
        "//xls/simulation:check_simulator",
        "//xls/tests:testvector_cc_proto",
        "//xls/tools:eval_utils",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
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
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
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
        # TODO(https://github.com/google/xls/issues/1141): Remove when fixed.
        "crashers/crasher_2023-10-02_7c14.x",
        # TODO(https://github.com/google/xls/issues/2049): Remove when fixed.
        "crashers/crasher_2025-04-23_0cd2.x",
        # TODO(https://github.com/google/xls/issues/2142): Remove when fixed.
        "crashers/crasher_2025-05-07_2d16.x",
        # TODO: https://github.com/google/xls/issues/2989 - Remove when fixed.
        "crashers/crasher_2025-09-03_c1e3.x",

        # TODO: https://github.com/google/xls/issues/4076 - Remove when fixed.
        "crashers/crasher_2026-04-08_85f2.x",

        # TODO: https://github.com/google/xls/issues/4142 - Remove when fixed.
        "crashers/crasher_2026-04-22_2292.x",
        # TODO: https://github.com/google/xls/issues/4302 - Remove when fixed.
        "crashers/crasher_2026-05-21_c47d.x",
        # TODO: https://github.com/google/xls/issues/4361 - Remove when fixed.
        "crashers/crasher_2026-06-03_919f.x",
        # TODO: https://github.com/google/xls/issues/4372 - Remove when fixed.
        "crashers/crasher_2026-06-09_8e88.x",
        # TODO: https://github.com/google/xls/issues/4380 - Remove when fixed.
        "crashers/crasher_2026-05-23_85e7.x",
        # TODO: https://github.com/google/xls/issues/4380 - Remove when fixed.
        "crashers/crasher_2026-05-23_87e5.x",
        # TODO: https://github.com/google/xls/issues/4446 - Remove when fixed.
        "crashers/crasher_2026-06-22_ea6c.x",
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
        "crashers/crasher_2025-05-25_eb23.x": [
            "optonly",
        ],
        "crashers/crasher_2026-04-10_7b7e.x": [
            "optonly",
        ],
        # TODO - b/520095765: For unknown reasons iverilog fails to run under hwasan.
        "crashers/crasher_2022-09-03_4d43.x": ["nohwasan"],
        "crashers/crasher_2022-10-24_40eb.x": ["nohwasan"],
        "crashers/crasher_2022-10-25_133f.x": ["nohwasan"],
        "crashers/crasher_2022-12-01_9a43.x": ["nohwasan"],
        "crashers/crasher_2023-02-26_ccb9.x": ["nohwasan"],
        "crashers/crasher_2023-03-27_83b8.x": ["nohwasan"],
        "crashers/crasher_2023-04-09_1f76.x": ["nohwasan"],
        "crashers/crasher_2023-06-05_c720.x": ["nohwasan"],
        "crashers/crasher_2023-07-02_c303.x": ["nohwasan"],
        "crashers/crasher_2023-11-19_3094.x": ["nohwasan"],
        "crashers/crasher_2024-02-23_b88e.x": ["nohwasan"],
        "crashers/crasher_2024-02-25_1b34.x": ["nohwasan"],
        "crashers/crasher_2024-07-08_5973.x": ["nohwasan"],
        "crashers/crasher_2024-07-14_b78c.x": ["nohwasan"],
        "crashers/crasher_2024-07-16_ddf8.x": ["nohwasan"],
        "crashers/crasher_2024-09-11_55dd.x": ["nohwasan"],
        "crashers/crasher_2024-09-23_0330-with-holdoff.x": ["nohwasan"],
        "crashers/crasher_2024-09-23_0330.x": ["nohwasan"],
        "crashers/crasher_2025-03-19_2e1f.x": ["nohwasan"],
        "crashers/crasher_2025-06-06_9c04.x": ["nohwasan"],
        "crashers/crasher_2025-06-23_5597.x": ["nohwasan"],
        "crashers/crasher_2025-09-04_5f36.x": ["nohwasan"],
        "crashers/crasher_2026-03-31_9c27.x": ["nohwasan"],
        "crashers/crasher_2026-05-11_d422.x": ["nohwasan"],
        "crashers/crasher_multiple-sends.x": ["nohwasan"],
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
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "sample",
    srcs = ["sample.cc"],
    hdrs = ["sample.h"],
    deps = [
        ":sample_cc_proto",
        ":scrub_crasher",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/tests:testvector_cc_proto",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@protobuf",
        "@protobuf//:differencer",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@protobuf",
        "@protobuf//:protobuf_lite",
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
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

pytype_strict_contrib_test(
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

pytype_strict_contrib_test(
    name = "run_fuzz_multiprocess_test",
    timeout = "long",
    srcs = ["run_fuzz_multiprocess_test.py"],
    data = [":run_fuzz_multiprocess"],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

pytype_strict_contrib_test(
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
        "@protobuf//:protobuf_python",
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
        "//xls/common:math_util",
        "//xls/common:random_util",
        "//xls/common:symbolized_stacktrace",
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
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_prod",
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
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
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
        "//xls/dslx/ir_convert:convert_options",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:unwrap_meta_type",
        "//xls/ir:format_preference",
        "//xls/tests:testvector_cc_proto",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
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
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "dslx_mutator_test",
    srcs = ["dslx_mutator_test.cc"],
    deps = [
        ":dslx_mutator",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/random:mock_distributions",
        "@abseil-cpp//absl/random:mocking_bit_gen",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
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
        "@abseil-cpp//absl/status:status_matchers",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/time",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/time",
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
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@abseil-cpp//absl/types:variant",
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
        "@abseil-cpp//absl/status:status_matchers",
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
        "@abseil-cpp//absl/log",
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
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@protobuf",
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
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
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

xls_dslx_library(
    name = "crashers/crasher_2025_07_07_6866",
    srcs = ["crashers/crasher_2025-07-07_6866.x"],
)

xls_dslx_library(
    name = "crashers/crasher_2026_06_03_919f",
    srcs = ["crashers/crasher_2026-06-03_919f.x"],
)
