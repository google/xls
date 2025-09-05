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

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xls/build_rules:xls_utilities.bzl", "bool_config_setting")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

alias(
    name = "xls_config",
    actual = ":xls_config_oss",
)

cc_library(
    name = "xls_config_oss",
    srcs = ["xls_config_oss.cc"],
    hdrs = ["xls_config.h"],
)

# LINT.Start
bool_config_setting(
    name = "save_temps_is_requested",
    values = {"save_temps": "true"},
    visibility = ["//xls:xls_internal"],
)

# Tell AOT targets to emit llvm ir and asm in addition to their normal outputs.
#
# This can be set even if --save_temps is not passed. If --save_temps is passed this is ignored.
bool_flag(
    name = "emit_aot_intermediates",
    build_setting_default = False,
    visibility = ["//xls:xls_internal"],
)

# Tell opt-rules to record a pprof of pass executions.
bool_flag(
    name = "profile_opt_passes",
    build_setting_default = False,
    visibility = ["//xls:xls_internal"],
)
# LINT.End
