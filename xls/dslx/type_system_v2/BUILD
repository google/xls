# Copyright 2024 The XLS Authors
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
    name = "inference_table",
    srcs = ["inference_table.cc"],
    hdrs = ["inference_table.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "inference_table_to_type_info",
    srcs = ["inference_table_to_type_info.cc"],
    hdrs = ["inference_table_to_type_info.h"],
    deps = [
        ":inference_table",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "inference_table_test",
    srcs = ["inference_table_test.cc"],
    deps = [
        ":inference_table",
        ":inference_table_to_type_info",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)
