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

load("//xls/contrib/xlscc/build_rules:cc_generator.bzl", "xls_int_fuzz_binaries")

cc_library(
    name = "unit_test",
    testonly = 1,
    srcs = ["unit_test.cc"],
    hdrs = ["unit_test.h"],
    deps = [
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_file",
        "//xls/common/logging",
        "//xls/common/logging:log_entry",
        "//xls/common/logging:log_sink",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:cc_parser",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/contrib/xlscc:xlscc_logging",
        "//xls/interpreter:interpreter_proc_runtime",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:ir_test_base",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@llvm-project//clang:ast",
    ],
)

cc_test(
    name = "translator_logic_test",
    srcs = ["translator_logic_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "translator_io_test",
    srcs = ["translator_io_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "translator_memory_test",
    srcs = ["translator_memory_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/ir",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "translator_proc_test",
    srcs = ["translator_proc_test.cc"],
    shard_count = 32,
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:events",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "translator_pointer_test",
    srcs = ["translator_pointer_test.cc"],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:translator",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "translator_static_test",
    srcs = ["translator_static_test.cc"],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:translator",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "translator_metadata_test",
    srcs = ["translator_metadata_test.cc"],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "//xls/contrib/xlscc:translator",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
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
        "//xls/codegen:signature_generator",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/contrib/xlscc:translator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
        "//xls/simulation:verilog_test_base",
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
    python_version = "PY3",
    deps = [
        "//xls/contrib/xlscc:hls_block_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

cc_test(
    name = "cc_parser_test",
    srcs = ["cc_parser_test.cc"],
    deps = [
        ":unit_test",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/xlscc:cc_parser",
        "//xls/contrib/xlscc:metadata_output_cc_proto",
        "@llvm-project//clang:basic",
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
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "cc_generator",
    srcs = ["cc_generator.cc"],
    hdrs = ["cc_generator.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//xls/codegen:combinational_generator",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/contrib/xlscc/unit_tests:cc_generator",
        "//xls/passes:standard_pipeline",
        "//xls/simulation:verilog_simulators",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xls_int_fuzz_binaries(
    name = "sample_fuzz",
    seed_count = 9,
    seed_start = 10,
)
