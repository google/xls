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

# pytype binary only
load("//xls/estimators/delay_model:build_defs.bzl", "delay_model")

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "area_characterization_client_main",
    srcs = ["area_characterization_client_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
        "//xls/estimators:estimator_model",
        "//xls/estimators:estimator_model_py_pb2",
        "//xls/estimators/delay_model:op_module_generator",
        "//xls/synthesis:client_credentials",
        "//xls/synthesis:synthesis_py_pb2",
        "//xls/synthesis:synthesis_service_py_pb2_grpc",
    ],
)

cc_library(
    name = "area_estimator",
    srcs = ["area_estimator.cc"],
    hdrs = ["area_estimator.h"],
    deps = [
        ":models",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/ir",
    ],
)

cc_test(
    name = "area_estimators_test",
    srcs = ["area_estimators_test.cc"],
    deps = [
        ":area_estimator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:source_location",
        "@com_google_googletest//:gtest",
    ],
)

# A srcs-less library which gathers together the alwaylink=1 models into a
# single library.
cc_library(
    name = "models",
    srcs = [],
    deps = [
        ":area_model_ice40_lut4",  # build_cleaner: keep
        ":area_model_testing_2_point_5_mux_per_node",  # build_cleaner: keep
    ],
)

delay_model(
    name = "area_model_testing_2_point_5_mux_per_node",
    srcs = ["area_model_testing_2_point_5_mux_per_node.textproto"],
    model_name = "area_model_testing_2_point_5_mux_per_node",
    precedence = "kLow",
)

delay_model(
    name = "area_model_ice40_lut4",
    srcs = ["ice40_lut4.textproto"],
    model_name = "ice40_lut4",
    precedence = "kHigh",
)
