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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_hdl//dependency_support/com_google_skywater_pdk:cells_info.bzl", "for_each_sky130_cells")
load("@rules_hdl//dependency_support/org_theopenroadproject_asap7_pdk_r1p7:cells_info.bzl", "for_each_asap7_cells")
load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//xls/build_rules:xls_build_defs.bzl", "xls_model_generation")
load("//xls/estimators/area_model:build_defs.bzl", "area_model")

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
    name = "area_estimator",
    srcs = ["area_estimator.cc"],
    hdrs = ["area_estimator.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/estimators:estimator_model_cc_proto",
        "//xls/estimators:proto_estimator_utils",
        "//xls/ir",
        "//xls/ir:op",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "area_estimator_test",
    srcs = ["area_estimator_test.cc"],
    deps = [
        ":area_estimator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/estimators:estimator_model_cc_proto",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
        "@protobuf",
    ],
)

cc_library(
    name = "area_estimators",
    srcs = ["area_estimators.cc"],
    hdrs = ["area_estimators.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":area_estimator",
        "//xls/ir",
        "@abseil-cpp//absl/status:statusor",
    ],
)

# Synthetic model used to test that `area_model` works without a checked-in
# `precomputed_proto` by generating the compiled estimator proto at build time.
area_model(
    name = "synthetic_test_area_model",
    testonly = True,
    srcs = ["testdata/synthetic_test_area_model.textproto"],
    model_name = "synthetic_test_area",
)

cc_test(
    name = "area_estimators_test",
    srcs = ["area_estimators_test.cc"],
    deps = [
        ":area_estimator",
        ":area_estimators",
        ":synthetic_test_area_model",  # build_cleaner: keep
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/estimators/area_model/models",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

pytype_strict_library(
    name = "area_model_utils",
    srcs = ["area_model_utils.py"],
    deps = ["//xls/estimators:estimator_model"],
)

pytype_strict_contrib_test(
    name = "area_model_utils_test",
    srcs = ["area_model_utils_test.py"],
    data = [
        ":area_model_utils",
    ],
    deps = [
        ":area_model_utils",
        "//xls/common:test_base",
        "//xls/estimators:estimator_model",
        "//xls/estimators:estimator_model_py_pb2",
        "@abseil-py//absl/testing:absltest",
        "@protobuf//:protobuf_python",
    ],
)

exports_files([
    "area_model_registration.cc.tmpl",
    "samples.textproto",
])

# Skywater 130nm open source PDK
# ========================================================================
# Default model for SKY130
xls_model_generation(
    name = "data_points-sky130",
    samples_file = ":samples.textproto",
    standard_cells = "@com_google_skywater_pdk_sky130_fd_sc_hd//:sky130_fd_sc_hd",
)

[
    xls_model_generation(
        name = "data_points-" + cell_name,
        samples_file = ":samples.textproto",
        standard_cells = cell_target,
        tags = ["manual"],
    )
    for cell_name, cell_target in for_each_sky130_cells("sc_hd")
]

# ASAP 7nm pseudo open source PDK
# ========================================================================
# Default model for ASAP7
xls_model_generation(
    name = "data_points-asap7",
    samples_file = ":samples.textproto",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
)

[
    xls_model_generation(
        name = "data_points-" + cell_name,
        samples_file = ":samples.textproto",
        standard_cells = cell_target,
        tags = ["manual"],
    )
    for cell_name, cell_target in for_each_asap7_cells("sc7p5t_rev27")
]

[
    xls_model_generation(
        name = "data_points-" + cell_name,
        samples_file = ":samples.textproto",
        standard_cells = cell_target,
        tags = ["manual"],
    )
    for cell_name, cell_target in for_each_asap7_cells("sc7p5t_rev28")
]

# Special ASAP7 rev27 standard cells scaled up 4x
xls_model_generation(
    name = "data_points-asap7-sc7p5t_rev27_rvt_4x",
    samples_file = ":samples.textproto",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt_4x",
    tags = ["manual"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:private"],
    deps = ["//xls/estimators:build_defs_bzl"],
)
