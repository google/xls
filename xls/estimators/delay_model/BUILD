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

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_hdl//dependency_support/com_google_skywater_pdk:cells_info.bzl", "for_each_sky130_cells")
load("@rules_hdl//dependency_support/org_theopenroadproject_asap7_pdk_r1p7:cells_info.bzl", "for_each_asap7_cells")
load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_binary")
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
    name = "delay_estimator",
    srcs = ["delay_estimator.cc"],
    hdrs = ["delay_estimator.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/netlist:cell_library",
        "//xls/netlist:logical_effort",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:die_if_null",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_prod",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
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
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "analyze_critical_path",
    srcs = ["analyze_critical_path.cc"],
    hdrs = ["analyze_critical_path.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":delay_estimator",
        ":delay_info_cc_proto",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
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
        "@abseil-cpp//absl/status:statusor",
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
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "delay_estimators",
    srcs = ["delay_estimators.cc"],
    hdrs = ["delay_estimators.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":delay_estimator",
        "//xls/ir",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

pytype_strict_binary(
    name = "generate_delay_lookup",
    srcs = ["generate_delay_lookup.py"],
    data = ["generate_delay_lookup.tmpl"],
    deps = [
        "//xls/common:runfiles",
        "//xls/estimators:estimator_model",
        "//xls/estimators:estimator_model_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@protobuf//:protobuf_python",
        "@xls_pip_deps//jinja2",
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
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
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
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
        "@protobuf",
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
