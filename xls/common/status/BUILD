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

# Common utilities related to absl::Status.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "error_code_to_status",
    srcs = ["error_code_to_status.cc"],
    hdrs = ["error_code_to_status.h"],
    # The main XLS code base should use absl::Status.
    visibility = ["//xls/common:__subpackages__"],
    deps = [
        ":status_builder",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "error_code_to_status_test",
    srcs = ["error_code_to_status_test.cc"],
    deps = [
        ":error_code_to_status",
        ":matchers",
        "@com_google_absl//absl/status",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "ret_check",
    srcs = ["ret_check.cc"],
    hdrs = ["ret_check.h"],
    deps = [
        ":status_builder",
        ":status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:source_location",
    ],
)

cc_test(
    name = "ret_check_test",
    srcs = ["ret_check_test.cc"],
    deps = [
        ":matchers",
        ":ret_check",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:scoped_mock_log",
    ],
)

cc_library(
    name = "status_builder",
    srcs = [
        "status_builder.cc",
        "status_builder_oss.cc",
    ],
    hdrs = ["status_builder.h"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xls/common:source_location",
        "//xls/common:symbolized_stacktrace",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "status_builder_test",
    srcs = ["status_builder_test.cc"],
    deps = [
        ":status_builder",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//xls/common:source_location",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:scoped_mock_log",
        "//xls/common/logging:vlog_is_on",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":status_builder",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "//xls/common:source_location",
    ],
)

cc_test(
    name = "status_macros_test",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":status_builder",
        ":status_macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    srcs = ["matchers.cc"],
    hdrs = ["matchers.h"],
    deps = [
        ":status_builder",
        ":status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)
