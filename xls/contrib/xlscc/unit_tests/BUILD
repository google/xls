# Copyright 2023 The XLS Authors
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//xls/contrib/xlscc/build_rules:cc_generator.bzl", "xls_ac_fuzz_binaries")

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "unit_test",
    testonly = 1,
    srcs = ["unit_test.cc"],
    hdrs = ["unit_test.h"],
    deps = [
        "//xls/codegen:module_signature_cc_proto",
        "//xls/common:source_location",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:cc_parser",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/contrib/xlscc:translator_types",
        "//xls/contrib/xlscc:xlscc_logging",
        "//xls/interpreter:block_evaluator",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:interpreter_proc_runtime",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:events",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:state_element",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/simulation:sim_test_base",
        "//xls/tools:codegen",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:opt",
        "//xls/tools:scheduling_options_flags_cc_proto",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
        "@llvm-project//clang:ast",
    ],
)

cc_test(
    name = "translator_logic_test",
    srcs = ["translator_logic_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:source_location",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/contrib/xlscc:translator_types",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "translator_io_test",
    srcs = ["translator_io_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:translator_types",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "continuations_test",
    srcs = ["continuations_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:tracked_bvalue",
        "//xls/contrib/xlscc:translator",
        "//xls/contrib/xlscc:translator_types",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:op",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
        "@llvm-project//clang:ast",
    ],
)

cc_test(
    name = "fsm_layout_test",
    srcs = ["fsm_layout_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:generate_fsm",
        "//xls/contrib/xlscc:translator_types",
        "//xls/ir",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
        "@llvm-project//clang:ast",
    ],
)

cc_test(
    name = "translator_memory_test",
    srcs = ["translator_memory_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "translator_proc_test",
    srcs = ["translator_proc_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_cc_proto",
        "//xls/ir:events",
        "//xls/ir:function_builder",
        "//xls/ir:op",
        "//xls/ir:state_element",
        "//xls/ir:value",
        "//xls/passes",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "translator_pointer_test",
    srcs = ["translator_pointer_test.cc"],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "translator_static_test",
    srcs = ["translator_static_test.cc"],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:translator_types",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "translator_metadata_test",
    srcs = ["translator_metadata_test.cc"],
    data = [
        "//xls/contrib/xlscc:synth_only_headers",
        "@com_github_hlslibs_ac_types//:ac_types_as_data",
    ],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/contrib/xlscc:translator",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "translator_verilog_test",
    srcs = ["translator_verilog_test.cc"],
    data = glob([
        "testdata/translator_verilog_test_*",
    ]),
    deps = [
        ":unit_test",
        "//xls/codegen:block_conversion",
        "//xls/codegen:block_generator",
        "//xls/codegen:codegen_pass",
        "//xls/codegen:module_signature",
        "//xls/codegen:signature_generator",
        "//xls/codegen/vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:cc_parser",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/contrib/xlscc:translator_types",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/passes",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_test_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

py_test(
    name = "xlscc_main_test",
    srcs = ["xlscc_main_test.py"],
    data = [
        "//xls/common:runfiles",
        "//xls/contrib/xlscc",
        "//xls/interpreter:ir_interpreter",
        "//xls/tools:eval_ir_main",
    ],
    deps = [
        "//xls/contrib/xlscc:hls_block_py_pb2",
        "@abseil-py//absl/testing:absltest",
    ],
)

cc_test(
    name = "cc_parser_test",
    srcs = ["cc_parser_test.cc"],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:cc_parser",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/ir:channel",
        "//xls/ir:source_location",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "synth_only_test",
    srcs = ["synth_only_test.cc"],
    data = [
        "//xls/contrib/xlscc:synth_only_headers",
        "@com_github_hlslibs_ac_types//:ac_types_as_data",
    ],
    shard_count = 10,
    tags = [
        "noasan",
        "nomsan",
    ],
    deps = [
        ":unit_test",
        "//xls/common:source_location",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "xls_int_test",
    srcs = ["xls_int_test.cc"],
    data = [
        "//xls/contrib/xlscc:synth_only_headers",
        "@com_github_hlslibs_ac_types//:ac_types_as_data",
    ],
    shard_count = 10,
    tags = [
        "noasan",
        "nomsan",
    ],
    deps = [
        ":unit_test",
        "//xls/common:source_location",
        "//xls/common:xls_gunit_main",
        "@com_github_hlslibs_ac_types//:ac_int",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "xls_intrinsics_test",
    srcs = ["xls_intrinsics_test.cc"],
    data = [
        "//xls/contrib/xlscc:synth_only_headers",
        "@com_github_hlslibs_ac_types//:ac_types_as_data",
    ],
    tags = [
        "noasan",
        "nomsan",
    ],
    deps = [
        ":unit_test",
        "//xls/common:source_location",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "xls_fixed_test",
    srcs = ["xls_fixed_test.cc"],
    data = [
        "//xls/contrib/xlscc:synth_only_headers",
        "@com_github_hlslibs_ac_types//:ac_types_as_data",
    ],
    shard_count = 10,
    tags = [
        "noasan",
        "nomsan",
    ],
    deps = [
        ":unit_test",
        "//xls/common:source_location",
        "//xls/common:xls_gunit_main",
        "@com_github_hlslibs_ac_types//:ac_fixed",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "cc_generator",
    srcs = ["cc_generator.cc"],
    hdrs = ["cc_generator.h"],
    deps = [
        "//xls/common:random_util",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_binary(
    name = "cc_generate_test",
    srcs = ["cc_generate_test.cc"],
    data = [
        "//xls/contrib/xlscc",
        "//xls/contrib/xlscc:synth_only_headers",
        "@com_github_hlslibs_ac_types//:ac_types_as_data",
    ],
    deps = [
        ":cc_generator",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xls_ac_fuzz_binaries(
    name = "fuzz_int",
    seed_count = 1,
    seed_start = 10,
    test_ac_fixed = False,
    test_ac_int = True,
    deps = [
        "//xls/codegen:codegen_options",
        "//xls/codegen/vast",
        "//xls/common:source_location",
        "//xls/ir:bits",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@googletest//:gtest",
    ],
)

xls_ac_fuzz_binaries(
    name = "fuzz_fixed",
    seed_count = 1,
    seed_start = 10,
    test_ac_fixed = True,
    test_ac_int = False,
    deps = [
        "//xls/codegen:codegen_options",
        "//xls/codegen/vast",
        "//xls/common:source_location",
        "//xls/ir:bits",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@googletest//:gtest",
    ],
)
