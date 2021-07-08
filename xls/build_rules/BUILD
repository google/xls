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
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(
    [
        "xls_codegen_macros.bzl",
        "xls_codegen_rules.bzl",
        "xls_common_rules.bzl",
        "xls_config_rules.bzl",
        "xls_codegen_rules.bzl",
        "xls_dslx_rules.bzl",
        "xls_ir_rules.bzl",
        "xls_jit_wrapper_rules.bzl",
        "xls_macros.bzl",
        "xls_oss_config.bzl",
        "xls_providers.bzl",
        "xls_rules.bzl",
        "xls_toolchains.bzl",
    ],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "xls_oss_config_rules_bzl",
    srcs = ["xls_oss_config_rules.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "xls_config_rules_bzl",
    srcs = ["xls_config_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "xls_oss_config_rules_bzl",
    ],
)

bzl_library(
    name = "xls_common_rules_bzl",
    srcs = ["xls_common_rules.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "xls_providers_bzl",
    srcs = ["xls_providers.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "xls_dslx_rules_bzl",
    srcs = ["xls_dslx_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_ir_rules_bzl",
    srcs = ["xls_ir_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_dslx_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_codegen_rules_bzl",
    srcs = ["xls_codegen_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_common_rules_bzl",
        ":xls_config_rules_bzl",
        ":xls_ir_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "xls_jit_wrapper_rules_bzl",
    srcs = ["xls_jit_wrapper_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_config_rules_bzl",
        ":xls_ir_rules_bzl",
        ":xls_providers_bzl",
        ":xls_toolchains_bzl",
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
    ],
)

bzl_library(
    name = "xls_rules_bzl",
    srcs = ["xls_rules.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        ":xls_codegen_rules_bzl",
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
        ":xls_config_rules_bzl",
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
        ":xls_ir_rules_bzl",
        ":xls_jit_wrapper_rules_bzl",
        ":xls_macros_bzl",
        ":xls_rules_bzl",
    ],
)

bzl_library(
    name = "xls_toolchains_bzl",
    srcs = ["xls_toolchains.bzl"],
    visibility = ["//visibility:private"],
)

xls_toolchain(
    name = "default_xls_toolchain",
)
