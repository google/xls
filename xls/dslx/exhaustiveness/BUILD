# Copyright 2025 The XLS Authors
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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "interp_value_interval",
    srcs = ["interp_value_interval.cc"],
    hdrs = ["interp_value_interval.h"],
    deps = [
        "//xls/dslx:interp_value",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "interp_value_interval_test",
    srcs = ["interp_value_interval_test.cc"],
    deps = [
        ":interp_value_interval",
        "//xls/common:xls_gunit_main",
        "//xls/dslx:interp_value",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "nd_region",
    srcs = ["nd_region.cc"],
    hdrs = ["nd_region.h"],
    deps = [
        ":interp_value_interval",
        "//xls/dslx:interp_value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "nd_region_test",
    srcs = ["nd_region_test.cc"],
    deps = [
        ":interp_value_interval",
        ":nd_region",
        "//xls/common:xls_gunit_main",
        "//xls/dslx:interp_value",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "match_exhaustiveness_checker",
    srcs = ["match_exhaustiveness_checker.cc"],
    hdrs = ["match_exhaustiveness_checker.h"],
    deps = [
        ":interp_value_interval",
        ":nd_region",
        "//xls/common:visitor",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "exhaustiveness_match_test",
    srcs = ["exhaustiveness_match_test.cc"],
    deps = [
        ":match_exhaustiveness_checker",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)
