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

# Delay models.

load("//xls/delay_model:build_defs.bzl", "delay_model")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)


cc_library(
    name = "area_estimator",
    srcs = ["area_estimator.cc"],
    hdrs = ["area_estimator.h"],
    deps = [
        ":models",
        "//xls/common:integral_types",
        "//xls/common/status:status_builder",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/netlist:logical_effort",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "area_estimator_test",
    srcs = ["area_estimator_test.cc"],
    deps = [
        ":area_estimator",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

# A srcs-less library which gathers together the models into a single
# dependency.
cc_library(
    name = "models",
    srcs = [],
    deps = [
        "area_model_testing_2_point_5_mux_per_node",  # build_cleaner: keep
    ],
)

delay_model(
    name = "area_model_testing_2_point_5_mux_per_node",
    srcs = ["area_model_testing_2_point_5_mux_per_node.textproto"],
    model_name = "area_model_testing_2_point_5_mux_per_node",
    precedence = "kLow",
)
