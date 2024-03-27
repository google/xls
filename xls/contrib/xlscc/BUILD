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
# cc_proto_library is used in this file

# pytype binary only
load("@rules_python//python:proto.bzl", "py_proto_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
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
)

cc_proto_library(
    name = "hls_block_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":hls_block_proto"],
)

py_proto_library(
    name = "hls_block_py_pb2",
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
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
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
    name = "translator",
    srcs = [
        "translate_block.cc",
        "translate_io.cc",
        "translate_loops.cc",
        "translate_metadata.cc",
        "translator.cc",
        "translator_types.cc",
    ],
    hdrs = [
        "translator.h",
    ],
    deps = [
        ":cc_parser",
        ":hls_block_cc_proto",
        ":metadata_output_cc_proto",
        ":xlscc_logging",
        "@com_google_absl//absl/base",
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
        "//xls/common:math_util",
        "//xls/common:stopwatch",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:caret",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
        "@z3//:api",
        "@com_googlesource_code_re2//:re2",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "cc_parser",
    srcs = ["cc_parser.cc"],
    hdrs = ["cc_parser.h"],
    deps = [
        ":metadata_output_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//xls/common:thread",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:source_location",
        "@com_googlesource_code_re2//:re2",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:frontend",
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
        "synth_only/cassert",
        "synth_only/cmath",
        "synth_only/cstdint",
        "synth_only/cstdlib",
        "synth_only/inttypes.h",
        "synth_only/iostream",
        "synth_only/limits",
        "synth_only/math.h",
        "synth_only/ostream",
        "synth_only/sstream",
        "synth_only/stdarg.h",
        "synth_only/stddef.h",
        "synth_only/stdint.h",
        "synth_only/string",
        "synth_only/type_traits",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging:log_flags",
        "//xls/common/status:status_macros",
        "//xls/ir:channel",
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
    python_version = "PY3",
    deps = [
        ":hls_block_py_pb2",
        ":metadata_output_py_pb2",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
    ],
)
