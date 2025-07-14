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

load("@rules_hdl//dependency_support/com_google_skywater_pdk:cells_info.bzl", "for_each_sky130_cells")
load("@rules_hdl//dependency_support/org_theopenroadproject_asap7_pdk_r1p7:cells_info.bzl", "for_each_asap7_cells")

# pytype binary only
# @rules_cc//cc:cc_library.bzl
# @rules_cc//cc:cc_test.bzl
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")

# Load proto_library
load("//xls/build_rules:xls_build_defs.bzl", "xls_model_generation")
# cc_proto_library is used in this file

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
    name = "delay_estimator",
    srcs = ["delay_estimator.cc"],
    hdrs = ["delay_estimator.h"],
    deps = [
        "//xls/common:test_macros",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/netlist:cell_library",
        "//xls/netlist:logical_effort",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ffi_delay_estimator",
    srcs = ["ffi_delay_estimator.cc"],
    hdrs = ["ffi_delay_estimator.h"],
    deps = [
        ":delay_estimator",
        "//xls/ir",
        "//xls/ir:op",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "delay_estimator_test",
    srcs = ["delay_estimator_test.cc"],
    deps = [
        ":delay_estimator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "delay_heap_test",
    srcs = ["delay_heap_test.cc"],
    deps = [
        ":analyze_critical_path",
        ":delay_estimator",
        ":delay_estimators",
        ":delay_heap",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "delay_estimators_test",
    srcs = ["delay_estimators_test.cc"],
    deps = [
        ":delay_estimator",
        ":delay_estimators",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "analyze_critical_path",
    srcs = ["analyze_critical_path.cc"],
    hdrs = ["analyze_critical_path.h"],
    deps = [
        ":delay_estimator",
        ":delay_info_cc_proto",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "analyze_critical_path_test",
    srcs = ["analyze_critical_path_test.cc"],
    deps = [
        ":analyze_critical_path",
        ":delay_estimator",
        ":delay_estimators",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "delay_heap",
    srcs = ["delay_heap.cc"],
    hdrs = ["delay_heap.h"],
    deps = [
        ":delay_estimator",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "delay_estimators",
    srcs = ["delay_estimators.cc"],
    hdrs = ["delay_estimators.h"],
    deps = [
        ":delay_estimator",
        "//xls/estimators/delay_model/models",
        "@com_google_absl//absl/status:statusor",
    ],
)

py_binary(
    name = "generate_delay_lookup",
    srcs = ["generate_delay_lookup.py"],
    data = ["generate_delay_lookup.tmpl"],
    deps = [
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "//xls/estimators:estimator_model",
        "//xls/estimators:estimator_model_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

cc_library(
    name = "sample_point_extraction_utils",
    srcs = ["sample_point_extraction_utils.cc"],
    hdrs = ["sample_point_extraction_utils.h"],
    deps = [
        ":delay_estimator",
        "//xls/common:case_converters",
        "//xls/common/status:status_macros",
        "//xls/estimators:estimator_model_cc_proto",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/ir:type",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "sample_point_extraction_utils_test",
    srcs = ["sample_point_extraction_utils_test.cc"],
    deps = [
        ":delay_estimator",
        ":sample_point_extraction_utils",
        "//xls/common:proto_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/estimators:estimator_model_cc_proto",
        "//xls/ir",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
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

# ASAP 7nm fake open source PDK
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

proto_library(
    name = "delay_info_proto",
    srcs = ["delay_info.proto"],
    visibility = ["//xls:xls_users"],
    deps = ["//xls/ir:op_proto"],
)

cc_proto_library(
    name = "delay_info_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":delay_info_proto"],
)

py_proto_library(
    name = "delay_info_py_pb2",
    visibility = ["//xls:xls_users"],
    deps = [":delay_info_proto"],
)
