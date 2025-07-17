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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
# Load proto_library
# cc_proto_library is used in this file

# pytype binary only
# @rules_cc//cc:cc_binary.bzl
# @rules_cc//cc:cc_library.bzl
load("@rules_python//python:proto.bzl", "py_proto_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(
    ["bazel_rules_macros.md"],
    visibility = ["//xls:xls_internal"],
)

proto_library(
    name = "hls_block_proto",
    srcs = ["hls_block.proto"],
    features = ["-proto_dynamic_mode_static_link"],
    deps = ["//xls/ir:channel_proto"],
)

cc_proto_library(
    name = "hls_block_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":hls_block_proto"],
)

py_proto_library(
    name = "hls_block_py_pb2",
    visibility = ["//xls:xls_users"],
    deps = [":hls_block_proto"],
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

py_proto_library(
    name = "metadata_output_py_pb2",
    deps = [":metadata_output_proto"],
)

build_test(
    name = "metadata_proto_libraries_build",
    targets = [
        ":metadata_output_cc_proto",
        ":metadata_output_py_pb2",
    ],
)

py_binary(
    name = "get_top_func_from_proto",
    srcs = ["get_top_func_from_proto.py"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":metadata_output_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

cc_library(
    name = "xlscc_logging",
    hdrs = [
        "xlscc_logging.h",
    ],
    deps = ["@com_google_absl//absl/log:check"],
)

cc_library(
    name = "tracked_bvalue",
    srcs = ["tracked_bvalue.cc"],
    hdrs = ["tracked_bvalue.h"],
    deps = [
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:source_location",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "node_manipulation",
    srcs = ["node_manipulation.cc"],
    hdrs = ["node_manipulation.h"],
    deps = [
        "//xls/ir",
        "//xls/ir:op",
    ],
)

cc_library(
    name = "translator_types",
    srcs = [
        "translator_types.cc",
    ],
    hdrs = [
        "translator_types.h",
    ],
    deps = [
        ":metadata_output_cc_proto",
        ":node_manipulation",
        ":tracked_bvalue",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:state_element",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
    ],
)

cc_library(
    name = "generate_fsm",
    srcs = [
        "generate_fsm.cc",
    ],
    hdrs = [
        "generate_fsm.h",
    ],
    deps = [
        ":tracked_bvalue",
        ":translator_types",
        ":xlscc_logging",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:source_location",
        "//xls/ir:state_element",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//clang:ast",
    ],
)

cc_library(
    name = "translator",
    srcs = [
        "continuations.cc",
        "translate_block.cc",
        "translate_io.cc",
        "translate_loops.cc",
        "translate_metadata.cc",
        "translator.cc",
    ],
    hdrs = [
        "translator.h",
    ],
    deps = [
        ":cc_parser",
        ":generate_fsm",
        ":hls_block_cc_proto",
        ":metadata_output_cc_proto",
        ":node_manipulation",
        ":tracked_bvalue",
        ":translator_types",
        ":xlscc_logging",
        "//xls/common:math_util",
        "//xls/common:stopwatch",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:caret",
        "//xls/ir:channel",
        "//xls/ir:channel_cc_proto",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:state_element",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/passes",
        "//xls/passes:dce_pass",
        "//xls/passes:node_source_analysis",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//llvm:Support",
        "@re2",
        "@z3//:api",
    ],
)

cc_library(
    name = "cc_parser",
    srcs = ["cc_parser.cc"],
    hdrs = ["cc_parser.h"],
    deps = [
        ":metadata_output_cc_proto",
        "//xls/common:thread",
        "//xls/common/file:filesystem",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:source_location",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:sema",
        "@llvm-project//clang:tooling",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "synth_only_headers",
    srcs = [
        "synth_only/ac_compat/ac_channel.h",
        "synth_only/ac_compat/ac_fixed.h",
        "synth_only/ac_compat/ac_int.h",
        "synth_only/ac_compat/ac_reg.h",
        "synth_only/ac_compat/ac_wait.h",
        "synth_only/algorithm",
        "synth_only/assert.h",
        "synth_only/atomic",
        "synth_only/cassert",
        "synth_only/cmath",
        "synth_only/cstddef",
        "synth_only/cstdint",
        "synth_only/cstdio",
        "synth_only/cstdlib",
        "synth_only/fstream",
        "synth_only/inttypes.h",
        "synth_only/iomanip",
        "synth_only/ios",
        "synth_only/iostream",
        "synth_only/istream",
        "synth_only/limits",
        "synth_only/math.h",
        "synth_only/ostream",
        "synth_only/sstream",
        "synth_only/stdarg.h",
        "synth_only/stddef.h",
        "synth_only/stdint.h",
        "synth_only/stdio.h",
        "synth_only/string",
        "synth_only/type_traits",
        "synth_only/utility",
        "synth_only/vector",
        "synth_only/xls_fixed.h",
        "synth_only/xls_int.h",
        "synth_only/xls_intrinsics.h",
    ],
    visibility = ["//xls:xls_users"],
)

cc_binary(
    name = "xlscc",
    srcs = [
        "flags.h",
        "main.cc",
    ],
    visibility = ["//xls:xls_users"],
    deps = [
        ":hls_block_cc_proto",
        ":metadata_output_cc_proto",
        ":translator",
        ":translator_types",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging:log_flags",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//clang:ast",
    ],
)

py_binary(
    name = "instrument_ir",
    srcs = ["instrument_ir.py"],
    data = [
        "//xls/tools:eval_ir_main",
        "//xls/tools:eval_proc_main",
    ],
    deps = [
        ":hls_block_py_pb2",
        ":metadata_output_py_pb2",
        "//xls/common:runfiles",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)
