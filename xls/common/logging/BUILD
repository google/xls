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

# Logging library that provides logging and assertion helpers and macros.

# pytype tests are present in this file
# @rules_cc//cc:cc_binary.bzl
# @rules_cc//cc:cc_library.bzl
# @rules_cc//cc:cc_test.bzl

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "log_lines",
    srcs = ["log_lines.cc"],
    hdrs = ["log_lines.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":log_flags",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "log_lines_test",
    srcs = ["log_lines_test.cc"],
    deps = [
        ":log_lines",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "log_flags",
    srcs = [
        "log_flags.cc",
        "log_flags.inc",
    ],
    hdrs = ["log_flags.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/log:globals",
    ],
    alwayslink = True,  # Provides some flags such as --alsologtostderr
)

py_test(
    name = "logging_flags_test",
    srcs = ["logging_flags_test.py"],
    data = [":log_initialization_tester"],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

cc_binary(
    name = "log_initialization_tester",
    testonly = True,
    srcs = ["log_initialization_tester.cc"],
    deps = [
        "//xls/common:init_xls",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:vlog_is_on",
    ],
)

cc_library(
    name = "scoped_vlog_level",
    hdrs = ["scoped_vlog_level.h"],
    visibility = ["//xls:xls_internal"],
    deps = ["@com_google_absl//absl/log:globals"],
)
