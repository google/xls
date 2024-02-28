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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "capture_stream",
    testonly = 1,
    srcs = ["capture_stream.cc"],
    hdrs = ["capture_stream.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "//xls/common/file:file_descriptor",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_file",
        "//xls/common/status:error_code_to_status",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "capture_stream_test",
    srcs = ["capture_stream_test.cc"],
    deps = [
        ":capture_stream",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_library(
    name = "errno_saver",
    hdrs = [
        "errno_saver.h",
    ],
    visibility = ["//visibility:private"],
)

cc_test(
    name = "errno_saver_test",
    srcs = ["errno_saver_test.cc"],
    deps = [
        ":errno_saver",
        "//xls/common:strerror",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "log_lines",
    srcs = ["log_lines.cc"],
    hdrs = ["log_lines.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":log_flags",
        ":logging",
        ":vlog_is_on",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "log_lines_test",
    srcs = ["log_lines_test.cc"],
    deps = [
        ":log_lines",
        ":scoped_mock_log",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "log_flags",
    srcs = [
        "log_flags.cc",
        "log_flags.inc",
    ],
    hdrs = ["log_flags.h"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:flags",
    ],
    alwayslink = 1,
)

cc_test(
    name = "log_flags_test",
    srcs = ["log_flags_test.cc"],
    deps = [
        ":capture_stream",
        ":log_flags",
        ":logging",
        ":logging_test_base",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
    ],
)

config_setting(
    name = "opt",
    values = {"compilation_mode": "opt"},
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":log_flags",
        ":vlog_is_on",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    deps = [
        ":capture_stream",
        ":logging",
        ":logging_test_base",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_library(
    name = "logging_test_base",
    testonly = True,
    srcs = ["logging_test_base.cc"],
    hdrs = ["logging_test_base.h"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "scoped_mock_log",
    testonly = True,
    hdrs = ["scoped_mock_log.h"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "vlog_is_on",
    hdrs = ["vlog_is_on.h"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
    ],
)

cc_test(
    name = "vlog_is_on_test",
    srcs = ["vlog_is_on_test.cc"],
    deps = [
        ":logging",
        ":vlog_is_on",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:globals",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

py_test(
    name = "logging_flags_test",
    srcs = ["logging_flags_test.py"],
    data = [":logger"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

cc_binary(
    name = "logger",
    srcs = ["logger.cc"],
    deps = [
        ":logging",
        "//xls/common:init_xls",
    ],
)
