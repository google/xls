# Copyright 2020 Google LLC
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
    name = "integration_options",
    srcs = ["integration_options.cc"],
    hdrs = ["integration_options.h"],
    deps = [
    ],
)

cc_library(
    name = "ir_integrator",
    srcs = ["ir_integrator.cc"],
    hdrs = ["ir_integrator.h"],
    deps = [
        ":integration_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
    ],
)

cc_test(
    name = "ir_integrator_test",
    srcs = ["ir_integrator_test.cc"],
    deps = [
        ":integration_options",
        ":ir_integrator",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:verifier",
    ],
)

cc_library(
    name = "integration_builder",
    srcs = ["integration_builder.cc"],
    hdrs = ["integration_builder.h"],
    deps = [
        ":integration_options",
        ":ir_integrator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:status_macros",
        "//xls/contrib/integrator/integration_algorithms:basic_integration_algorithm",
        "//xls/ir",
        "//xls/ir:op",
    ],
)

cc_test(
    name = "integration_builder_test",
    srcs = ["integration_builder_test.cc"],
    deps = [
        ":integration_builder",
        "@com_google_googletest//:gtest",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:source_location",
        "//xls/ir:verifier",
    ],
)
