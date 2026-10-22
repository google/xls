# Copyright 2026 The XLS Authors
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

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "test_codegen_flags_proto",
    testonly = True,
    srcs = ["test_codegen_flags.proto"],
    deps = ["//xls/tools:codegen_flags_proto"],
)

cc_proto_library(
    name = "test_codegen_flags_cc_proto",
    testonly = True,
    deps = [":test_codegen_flags_proto"],
)

cc_library(
    name = "test_codegen_flag_handler",
    testonly = True,
    srcs = ["test_codegen_flag_handler.cc"],
    hdrs = ["test_codegen_flag_handler.h"],
    deps = [
        ":test_codegen_flags_cc_proto",
        "//xls/codegen:codegen_options",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:codegen_flags_handler_registry",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status",
    ],
    alwayslink = True,
)

proto_library(
    name = "test_opt_flags_proto",
    testonly = True,
    srcs = ["test_opt_flags.proto"],
    deps = ["//xls/tools:opt_flags_proto"],
)

cc_proto_library(
    name = "test_opt_flags_cc_proto",
    testonly = True,
    deps = [":test_opt_flags_proto"],
)

cc_library(
    name = "test_opt_flag_handler",
    testonly = True,
    srcs = ["test_opt_flag_handler.cc"],
    hdrs = ["test_opt_flag_handler.h"],
    deps = [
        ":test_opt_flags_cc_proto",
        "//xls/common/status:status_macros",
        "//xls/passes:optimization_pass",
        "//xls/tools:opt_flags_cc_proto",
        "//xls/tools:opt_flags_handler_registry",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status",
    ],
    alwayslink = True,
)
