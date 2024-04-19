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
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "basic_integration_algorithm",
    srcs = ["basic_integration_algorithm.cc"],
    hdrs = [
        "basic_integration_algorithm.h",
        "integration_algorithm.h",
        "integration_algorithm_implementation.h",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/contrib/integrator:integration_options",
        "//xls/contrib/integrator:ir_integrator",
        "//xls/ir",
        "//xls/ir:verifier",
    ],
)

cc_test(
    name = "basic_integration_algorithm_test",
    srcs = ["basic_integration_algorithm_test.cc"],
    deps = [
        ":basic_integration_algorithm",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/contrib/integrator:integration_builder",
        "//xls/contrib/integrator:ir_integrator",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/ir:verifier",
    ],
)
