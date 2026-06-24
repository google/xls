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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_contrib_test")

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
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "log_lines_test",
    srcs = ["log_lines_test.cc"],
    deps = [
        ":log_lines",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/log:scoped_mock_log",
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
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:flags",
        "@abseil-cpp//absl/log:globals",
    ],
    alwayslink = True,  # Provides some flags such as --alsologtostderr
)

pytype_strict_contrib_test(
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
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:vlog_is_on",
    ],
)

cc_library(
    name = "scoped_vlog_level",
    srcs = ["scoped_vlog_level.cc"],
    hdrs = ["scoped_vlog_level.h"],
    visibility = ["//xls:xls_internal"],
    deps = [
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "scoped_vlog_level_test",
    srcs = [
        "scoped_vlog_level_test.cc",
        "scoped_vlog_level_test_helper.cc",
        "scoped_vlog_level_test_helper.h",
    ],
    deps = [
        ":collect_logs",
        ":scoped_vlog_level",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:log_sink_registry",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "scoped_record_logs",
    testonly = 1,
    srcs = ["scoped_record_logs.cc"],
    hdrs = ["scoped_record_logs.h"],
    visibility = ["//xls:xls_internal"],
    deps = [
        ":collect_logs",
        "@abseil-cpp//absl/log:log_sink_registry",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "collect_logs",
    hdrs = ["collect_logs.h"],
    visibility = ["//xls:xls_internal"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:log_entry",
        "@abseil-cpp//absl/log:log_sink",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_test(
    name = "collect_logs_test",
    srcs = ["collect_logs_test.cc"],
    deps = [
        ":collect_logs",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:log_sink_registry",
        "@googletest//:gtest",
    ],
)
