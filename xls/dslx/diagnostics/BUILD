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
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "warn_on_defined_but_unused",
    srcs = ["warn_on_defined_but_unused.cc"],
    hdrs = ["warn_on_defined_but_unused.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "warn_on_defined_but_unused_test",
    srcs = ["warn_on_defined_but_unused_test.cc"],
    deps = [
        ":warn_on_defined_but_unused",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "maybe_explain_error",
    srcs = ["maybe_explain_error.cc"],
    hdrs = ["maybe_explain_error.h"],
    deps = [
        ":format_type_mismatch",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_mismatch_error_data",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "maybe_explain_error_test",
    srcs = ["maybe_explain_error_test.cc"],
    deps = [
        ":maybe_explain_error",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:deduce",
        "//xls/dslx/type_system:deduce_ctx",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:type_mismatch_error_data",
        "//xls/dslx/type_system:typecheck_module",
        "@com_google_absl//absl/status",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "format_type_mismatch",
    srcs = ["format_type_mismatch.cc"],
    hdrs = ["format_type_mismatch.h"],
    deps = [
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:zip_types",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "format_type_mismatch_test",
    srcs = ["format_type_mismatch_test.cc"],
    deps = [
        ":format_type_mismatch",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:channel_direction",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "@googletest//:gtest",
    ],
)
