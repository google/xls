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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "condition",
    hdrs = [
        "condition.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":log_message",
        ":null_stream",
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
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "log_entry",
    srcs = ["log_entry.cc"],
    hdrs = ["log_entry.h"],
    deps = [
        ":log_flags",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "log_entry_test",
    srcs = ["log_entry_test.cc"],
    deps = [
        ":log_entry",
        "@com_google_absl//absl/base:log_severity",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "log_lines",
    srcs = ["log_lines.cc"],
    hdrs = ["log_lines.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":logging",
        ":vlog_is_on",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "log_lines_test",
    srcs = ["log_lines_test.cc"],
    deps = [
        ":log_lines",
        ":scoped_mock_log",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
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
    ],
)

cc_test(
    name = "log_flags_test",
    srcs = ["log_flags_test.cc"],
    deps = [
        ":capture_stream",
        ":log_flags",
        ":logging_test_base",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "log_message",
    srcs = [
        "log_message.cc",
    ],
    hdrs = [
        "log_message.h",
    ],
    deps = [
        ":errno_saver",
        ":log_entry",
        ":log_flags",
        ":log_sink",
        ":null_guard",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//xls/common:source_location",
        "//xls/common:strerror",
        "//xls/common:symbolized_stacktrace",
    ],
)

cc_test(
    name = "log_message_test",
    srcs = ["log_message_test.cc"],
    deps = [
        ":log_flags",
        ":log_message",
        ":logging_test_base",
        ":scoped_mock_log",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:source_location",
        "//xls/common:strerror",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "log_sink",
    hdrs = [
        "log_sink.h",
    ],
    deps = [
        ":log_entry",
    ],
)

cc_library(
    name = "logging",
    srcs = [
        "logging.cc",
    ],
    hdrs = [
        "check_ops.h",
        "logging.h",
        "logging_internal.h",
    ],
    deps = [
        ":condition",
        ":log_message",
        ":log_sink",
        ":null_guard",
        ":vlog_is_on",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    deps = [
        ":capture_stream",
        ":log_entry",
        ":logging",
        ":logging_test_base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "logging_test_base",
    testonly = True,
    srcs = ["logging_test_base.cc"],
    hdrs = ["logging_test_base.h"],
    deps = [
        ":log_entry",
        ":log_sink",
        ":logging",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "null_guard",
    hdrs = [
        "null_guard.h",
    ],
    visibility = ["//visibility:private"],
)

cc_test(
    name = "null_guard_test",
    srcs = ["null_guard_test.cc"],
    deps = [
        ":null_guard",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "null_stream",
    hdrs = [
        "null_stream.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":log_sink",
        "@com_google_absl//absl/strings",
        "//xls/common:source_location",
    ],
)

cc_test(
    name = "null_stream_test",
    srcs = ["null_stream_test.cc"],
    deps = [
        ":null_stream",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "scoped_mock_log",
    testonly = True,
    hdrs = ["scoped_mock_log.h"],
    deps = [
        ":log_entry",
        ":logging",
        "@com_google_absl//absl/base:log_severity",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "vlog_is_on",
    srcs = [
        "vlog_is_on.cc",
        "vlog_is_on.inc",
    ],
    hdrs = ["vlog_is_on.h"],
    deps = [
        ":errno_saver",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "vlog_is_on_test",
    srcs = ["vlog_is_on_test.cc"],
    deps = [
        ":vlog_is_on",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
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
