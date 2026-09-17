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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls/experimental/busperf:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

bzl_library(
    name = "busperf_yaml_rules_bzl",
    srcs = ["busperf_yaml_rules.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "busperf_report_rules_bzl",
    srcs = ["busperf_report_rules.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "busperf_macros_bzl",
    srcs = ["busperf_macros.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":busperf_report_rules_bzl",
        ":busperf_yaml_rules_bzl",
        "//xls/build_rules:xls_build_defs_bzl",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        ":busperf_macros_bzl",
        ":busperf_report_rules_bzl",
    ],
)
