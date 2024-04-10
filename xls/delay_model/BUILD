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

# pytype binary, test, library
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")
load("//xls/build_rules:xls_build_defs.bzl", "xls_delay_model_generation")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
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
        ":delay_estimators",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/ir:type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "delay_estimators_test",
    srcs = ["delay_estimators_test.cc"],
    deps = [
        ":delay_estimator",
        ":delay_estimators",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
    ],
)

cc_library(
    name = "analyze_critical_path",
    srcs = ["analyze_critical_path.cc"],
    hdrs = ["analyze_critical_path.h"],
    deps = [
        ":delay_estimator",
        "//xls/common/status:status_macros",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "analyze_critical_path_test",
    srcs = ["analyze_critical_path_test.cc"],
    deps = [
        ":analyze_critical_path",
        ":delay_estimator",
        ":delay_estimators",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "delay_estimators",
    srcs = ["delay_estimators.cc"],
    hdrs = ["delay_estimators.h"],
    deps = [
        ":delay_estimator",
        "//xls/delay_model/models",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "delay_model_proto",
    srcs = ["delay_model.proto"],
)

py_proto_library(
    name = "delay_model_py_pb2",
    deps = [":delay_model_proto"],
)

py_binary(
    name = "generate_delay_lookup",
    srcs = ["generate_delay_lookup.py"],
    data = ["generate_delay_lookup.tmpl"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":delay_model",
        ":delay_model_py_pb2",
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "delay_model",
    srcs = ["delay_model.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":delay_model_py_pb2",
        requirement("numpy"),
        requirement("scipy"),
    ],
)

py_test(
    name = "delay_model_test",
    srcs = ["delay_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":delay_model",
        ":delay_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "op_module_generator",
    srcs = ["op_module_generator.py"],
    data = [
        "//xls/tools:codegen_main",
        "//xls/tools:parse_ir",
    ],
    srcs_version = "PY3",
    deps = [
        "//xls/codegen:module_signature_py_pb2",
        "//xls/common:runfiles",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "op_module_generator_test",
    srcs = ["op_module_generator_test.py"],
    data = glob(["testdata/*"]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":op_module_generator",
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_binary(
    name = "delay_model_join",
    srcs = ["delay_model_join.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":delay_model_py_pb2",
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "delay_model_join_test",
    srcs = ["delay_model_join_test.py"],
    data = [
        ":delay_model_join",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_binary(
    name = "extract_sample_points_from_delay_model",
    srcs = ["extract_sample_points_from_delay_model.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":delay_model_py_pb2",
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "extract_sample_points_from_delay_model_test",
    srcs = ["extract_sample_points_from_delay_model_test.py"],
    data = [
        ":extract_sample_points_from_delay_model",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

exports_files([
    "samples.textproto",
])

# Skywater 130nm open source PDK
# ========================================================================
# Default model for SKY130
xls_delay_model_generation(
    name = "data_points-sky130",
    samples_file = ":samples.textproto",
    standard_cells = "@com_google_skywater_pdk_sky130_fd_sc_hd//:sky130_fd_sc_hd",
)

[
    xls_delay_model_generation(
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
xls_delay_model_generation(
    name = "data_points-asap7",
    samples_file = ":samples.textproto",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
)

[
    xls_delay_model_generation(
        name = "data_points-" + cell_name,
        samples_file = ":samples.textproto",
        standard_cells = cell_target,
        tags = ["manual"],
    )
    for cell_name, cell_target in for_each_asap7_cells("sc7p5t_rev27")
]

[
    xls_delay_model_generation(
        name = "data_points-" + cell_name,
        samples_file = ":samples.textproto",
        standard_cells = cell_target,
        tags = ["manual"],
    )
    for cell_name, cell_target in for_each_asap7_cells("sc7p5t_rev28")
]

# Special ASAP7 rev27 standard cells scaled up 4x
xls_delay_model_generation(
    name = "data_points-asap7-sc7p5t_rev27_rvt_4x",
    samples_file = ":samples.textproto",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt_4x",
    tags = ["manual"],
)
