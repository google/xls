# Copyright 2021 The XLS Authors
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

# Dependencies locked to avoid transitive dependencies on llvm.
# Use the XLS[cc] binary for C++ to XLS IR translation, rather than directly
#  depending on the translation library.
# Email xls-team@ or seanhaskell@ if a dependency is desired.

# cc_proto_library is used in this file
load("//xls/build_rules:py_proto_library.bzl", "xls_py_proto_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "hls_block_proto",
    srcs = ["hls_block.proto"],
    features = ["-proto_dynamic_mode_static_link"],
)

cc_proto_library(
    name = "hls_block_cc_proto",
    deps = [":hls_block_proto"],
)

xls_py_proto_library(
    name = "hls_block_py_pb2",
    srcs = ["hls_block.proto"],
    internal_deps = [":hls_block_proto"],
)

proto_library(
    name = "metadata_output_proto",
    srcs = ["metadata_output.proto"],
    features = ["-proto_dynamic_mode_static_link"],
)

cc_proto_library(
    name = "metadata_output_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":metadata_output_proto"],
)

xls_py_proto_library(
    name = "metadata_output_py_pb2",
    srcs = ["metadata_output.proto"],
    internal_deps = [":metadata_output_proto"],
)

build_test(
    name = "metadata_proto_libraries_build",
    targets = [
        ":metadata_output_cc_proto",
        ":metadata_output_py_pb2",
    ],
)

cc_library(
    name = "unit_test",
    testonly = 1,
    srcs = ["unit_test.cc"],
    hdrs = ["unit_test.h"],
    deps = [
        ":translator",
        "//xls/codegen:combinational_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:proc_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "translator",
    srcs = ["translator.cc"],
    hdrs = ["translator.h"],
    deps = [
        ":hls_block_cc_proto",
        ":metadata_output_cc_proto",
        "//xls/common:thread",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/passes:dce_pass",
        "//xls/passes:dfe_pass",
        "//xls/passes:identity_removal_pass",
        "//xls/passes:inlining_pass",
        "//xls/passes:standard_pipeline",
        "//xls/passes:tuple_simplification_pass",
        "//xls/passes:verifier_checker",
        "@com_github_google_re2//:re2",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:tooling",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "translator_test",
    srcs = ["translator_test.cc"],
    deps = [
        ":metadata_output_cc_proto",
        ":translator",
        ":unit_test",
        "//xls/codegen:block_conversion",
        "//xls/codegen:block_generator",
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:proc_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
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
        ":hls_block_cc_proto",
        ":translator",
        ":unit_test",
        "//xls/codegen:block_conversion",
        "//xls/codegen:block_generator",
        "//xls/codegen:signature_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
        "//xls/simulation:verilog_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "xlscc",
    srcs = ["main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":hls_block_cc_proto",
        ":metadata_output_cc_proto",
        ":translator",
        "//xls/codegen:block_conversion",
        "//xls/codegen:block_generator",
        "//xls/common:init_xls",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/passes:standard_pipeline",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//clang:ast",
    ],
)

py_test(
    name = "xlscc_main_test",
    srcs = ["xlscc_main_test.py"],
    data = [
        ":xlscc",
        "//xls/common:runfiles",
        "//xls/interpreter:ir_interpreter",
        "//xls/tools:eval_ir_main",
    ],
    python_version = "PY3",
    deps = [
        ":hls_block_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
