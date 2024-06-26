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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(":xls_toolchains.bzl", "xls_toolchain")

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
    name = "release_files",
    srcs = glob(["*.bzl"]) + [
        ":default_xls_toolchain",
        ":xls_macros_bzl",
    ],
)

exports_files(
    [
        "xls_codegen_macros.bzl",
        "xls_codegen_rules.bzl",
        "xls_codegen_fdo_rules.bzl",
        "xls_common_rules.bzl",
        "xls_config_rules.bzl",
        "xls_dslx_rules.bzl",
        "xls_ir_rules.bzl",
        "xls_ir_macros.bzl",
        "xls_jit_wrapper_rules.bzl",
        "xls_macros.bzl",
        "xls_oss_config.bzl",
        "xls_providers.bzl",
        "xls_rules.bzl",
        "xls_rules_build_defs.bzl",
        "xls_toolchains.bzl",
        "xls_utilities.bzl",
    ],
    visibility = ["//visibility:private"],
)

exports_files(
    ["xls_build_defs.bzl"],
    visibility = ["//xls:xls_users"],
)

bzl_library(
    name = "xls_type_check_utils_bzl",
    srcs = ["xls_type_check_utils.bzl"],
)

bzl_library(
    name = "xls_utilities_bzl",
    srcs = ["xls_utilities.bzl"],
    deps = [
        ":xls_common_rules_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_oss_config_rules_bzl",
    srcs = ["xls_oss_config_rules.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "xls_config_rules_bzl",
    srcs = ["xls_config_rules.bzl"],
    deps = [
        ":xls_oss_config_rules_bzl",
    ],
)

bzl_library(
    name = "xls_common_rules_bzl",
    srcs = ["xls_common_rules.bzl"],
    deps = [":xls_providers_bzl"],
)

bzl_library(
    name = "xls_providers_bzl",
    srcs = ["xls_providers.bzl"],
    deps = [
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_dslx_rules_bzl",
    srcs = ["xls_dslx_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_ir_rules_bzl",
    srcs = ["xls_ir_rules.bzl"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_dslx_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        ":xls_type_check_utils_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_internal_build_defs_bzl",
    srcs = ["xls_internal_build_defs.bzl"],
)

bzl_library(
    name = "xls_ir_macros_bzl",
    srcs = ["xls_ir_macros.bzl"],
    deps = [
        ":xls_config_rules_bzl",
        ":xls_internal_aot_rules_bzl",
        ":xls_internal_build_defs_bzl",
        ":xls_ir_rules_bzl",
        ":xls_type_check_utils_bzl",
    ],
)

bzl_library(
    name = "xls_codegen_rules_bzl",
    srcs = [
        "xls_codegen_fdo_rules.bzl",
        "xls_codegen_rules.bzl",
    ],
    visibility = ["//xls:xls_users"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_ir_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        "@bazel_skylib//lib:dicts",
        "@rules_hdl//pdk:build_defs.bzl",
    ],
)

bzl_library(
    name = "xls_jit_wrapper_rules_bzl",
    srcs = ["xls_jit_wrapper_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_internal_aot_rules_bzl",
        ":xls_internal_build_defs_bzl",
        ":xls_ir_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        ":xls_type_check_utils_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_codegen_macros_bzl",
    srcs = ["xls_codegen_macros.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_codegen_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_type_check_utils_bzl",
        "@bazel_skylib//rules:build_test",
    ],
)

bzl_library(
    name = "xls_rules_bzl",
    srcs = ["xls_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_codegen_rules_bzl",
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_dslx_rules_bzl",
        ":xls_ir_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_macros_bzl",
    srcs = ["xls_macros.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_codegen_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_ir_macros_bzl",
        ":xls_ir_rules_bzl",
        ":xls_rules_bzl",
        ":xls_type_check_utils_bzl",
        "@bazel_skylib//rules:build_test",
        "@bazel_skylib//rules:diff_test",
    ],
)

bzl_library(
    name = "xls_rules_build_defs_bzl",
    srcs = ["xls_rules_build_defs.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_codegen_rules_bzl",
        ":xls_ir_rules_bzl",
        ":xls_jit_wrapper_rules_bzl",
        ":xls_rules_bzl",
    ],
)

bzl_library(
    name = "xls_build_defs_bzl",
    srcs = ["xls_build_defs.bzl"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":xls_codegen_macros_bzl",
        ":xls_codegen_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_dslx_rules_bzl",
        ":xls_ir_macros_bzl",
        ":xls_ir_rules_bzl",
        ":xls_jit_wrapper_rules_bzl",
        ":xls_macros_bzl",
        ":xls_rules_bzl",
        ":xls_toolchains_bzl",
        ":xls_utilities_bzl",
    ],
)

bzl_library(
    name = "xls_toolchains_bzl",
    srcs = ["xls_toolchains.bzl"],
)

bzl_library(
    name = "xls_internal_aot_rules_bzl",
    srcs = ["xls_internal_aot_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_ir_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        ":xls_utilities_bzl",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//rules:common_settings",
    ],
)

xls_toolchain(
    name = "default_xls_toolchain",
    visibility = ["//xls:xls_public"],
)
