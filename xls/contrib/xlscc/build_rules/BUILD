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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(
    [
        "xlscc_providers.bzl",
        "xlscc_rules.bzl",
        "xlscc_rules_build_defs.bzl",
    ],
    visibility = ["//visibility:private"],
)

exports_files(
    ["xlscc_build_defs.bzl"],
    visibility = ["//xls:xls_users"],
)

bzl_library(
    name = "xlscc_rules_bzl",
    srcs = ["xlscc_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xlscc_providers_bzl",
        "//xls/build_rules:xls_codegen_rules_bzl",
        "//xls/build_rules:xls_common_rules_bzl",
        "//xls/build_rules:xls_config_rules_bzl",
        "//xls/build_rules:xls_ir_rules_bzl",
        "//xls/build_rules:xls_providers_bzl",
        "//xls/build_rules:xls_toolchains_bzl",
        "//xls/build_rules:xls_type_check_utils_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xlscc_rules_build_defs_bzl",
    srcs = ["xlscc_rules_build_defs.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xlscc_rules_bzl",
    ],
)

bzl_library(
    name = "xlscc_build_defs_bzl",
    srcs = ["xlscc_build_defs.bzl"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":xlscc_rules_bzl",
    ],
)

bzl_library(
    name = "xlscc_providers_bzl",
    srcs = ["xlscc_providers.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "cc_generator_bzl",
    srcs = ["cc_generator.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "@rules_cc//cc:core_rules",
    ],
)
