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

# pytype binary, test, library
load("@rules_hdl//dependency_support/com_google_skywater_pdk:cells_info.bzl", "for_each_sky130_cells")
load("@rules_hdl//dependency_support/org_theopenroadproject_asap7_pdk_r1p7:cells_info.bzl", "for_each_asap7_cells")
load("@xls_pip_deps//:requirements.bzl", "requirement")
load("//xls/build_rules:xls_build_defs.bzl", "xls_model_generation")

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
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "area_estimators",
    srcs = ["area_estimators.cc"],
    hdrs = ["area_estimators.h"],
    deps = [
        ":area_estimator",
        "//xls/estimators/area_model/models",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "area_estimators_test",
    srcs = ["area_estimators_test.cc"],
    deps = [
        ":area_estimator",
        ":area_estimators",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

py_library(
    name = "area_model_utils",
    srcs = ["area_model_utils.py"],
    deps = ["//xls/estimators:estimator_model"],
)

py_test(
    name = "area_model_utils_test",
    srcs = ["area_model_utils_test.py"],
    data = [
        ":area_model_utils",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":area_model_utils",
        "//xls/common:test_base",
        "//xls/estimators:estimator_model",
        "//xls/estimators:estimator_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_binary(
    name = "generate_area_lookup",
    srcs = ["generate_area_lookup.py"],
    data = ["generate_area_lookup.tmpl"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "//xls/estimators:estimator_model",
        "//xls/estimators:estimator_model_py_pb2",
        "//xls/estimators/area_model:area_model_utils",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

exports_files([
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
