# Copyright 2025 The XLS Authors
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_python//python:proto.bzl", "py_proto_library")
# Load proto_library
# cc_proto_library is used in this file

load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "xls_ir_to_cytoscape",
    srcs = ["xls_ir_to_cytoscape.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/visualization/ir_viz:node_attribute_visitor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:json_util",
    ],
)

pytype_strict_library(
    name = "xls_ir_to_networkx",
    srcs = ["xls_ir_to_networkx.py"],
    data = [":xls_ir_to_cytoscape"],
    deps = [
        "@abseil-py//absl/logging",
        "@xls_pip_deps//networkx",
        # networkx annoyingly marks scipy as an optional dep, but our usage requires it.
        # Also see b/394358042.
        "@xls_pip_deps//scipy",  # buildcleaner: keep
        "//xls/common:runfiles",
    ],
)

pytype_strict_contrib_test(
    name = "xls_ir_to_networkx_test",
    srcs = ["xls_ir_to_networkx_test.py"],
    data = ["//xls/examples:riscv_simple.opt.ir"],
    deps = [
        ":xls_ir_to_networkx",
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
        "@xls_pip_deps//networkx",
    ],
)

pytype_strict_library(
    name = "ir_diff",
    srcs = ["ir_diff.py"],
    deps = [
        "@xls_pip_deps//networkx",
    ],
)

py_library(
    name = "ir2nx",
    srcs = ["ir2nx.py"],
    deps = [
        ":xls_types",
        ":xls_values",
        requirement("networkx"),
    ],
)

py_binary(
    name = "ir_stats",
    srcs = ["ir_stats.py"],
    deps = [
        ":ir2nx",
    ],
)

py_binary(
    name = "ir2gxl",
    srcs = ["ir2gxl.py"],
    main = "ir2gxl.py",
    deps = [
        ":ir2nx",
        requirement("networkx"),
    ],
)

pytype_strict_library(
    name = "ir_diff_utils",
    srcs = ["ir_diff_utils.py"],
    deps = [
        ":ir_diff",
        "@xls_pip_deps//matplotlib",
        "@xls_pip_deps//networkx",
    ],
)

pytype_strict_library(
    name = "ir_patch_gen",
    srcs = ["ir_patch_gen.py"],
    deps = [
        ":ir_diff",
        ":ir_patch_py_pb2",
        ":xls_types",
        ":xls_values",
    ],
)

pytype_strict_library(
    name = "xls_types",
    srcs = ["xls_types.py"],
    deps = ["//xls/ir:xls_type_py_pb2"],
)

pytype_strict_library(
    name = "xls_values",
    srcs = ["xls_values.py"],
    deps = [
        ":xls_types",
        "//xls/ir:xls_value_py_pb2",
    ],
)

pytype_strict_library(
    name = "ir_diff_main_lib",
    srcs = ["ir_diff_main.py"],
    deps = [
        ":ir_diff",
        ":ir_diff_utils",
        ":ir_patch_gen",
        ":xls_ir_to_networkx",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "ir_diff_main",
    srcs = ["ir_diff_main.py"],
    deps = [":ir_diff_main_lib"],
)

pytype_strict_contrib_test(
    name = "ir_diff_main_test",
    srcs = ["ir_diff_main_test.py"],
    data = [
        "//xls/examples:riscv_simple.ir",  # unopt
        "//xls/examples:riscv_simple.opt.ir",  #opt
    ],
    deps = [
        ":ir_diff",
        ":ir_diff_main_lib",
        ":ir_patch_py_pb2",
        "//xls/common:runfiles",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:flagsaver",
    ],
)

proto_library(
    name = "ir_patch_proto",
    srcs = ["ir_patch.proto"],
    deps = [
        "//xls/ir:xls_type_proto",
        "//xls/ir:xls_value_proto",
    ],
)

py_proto_library(
    name = "ir_patch_py_pb2",
    deps = [":ir_patch_proto"],
)

cc_library(
    name = "patch_ir_lib",
    srcs = ["patch_ir.cc"],
    hdrs = ["patch_ir.h"],
    deps = [
        ":ir_patch_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:xls_type_cc_proto",
        "//xls/ir:xls_value_cc_proto",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "//xls/tools:codegen_flags",
        "//xls/tools:scheduling_options_flags",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "patch_ir_main",
    srcs = ["patch_ir_main.cc"],
    deps = [
        ":ir_patch_cc_proto",
        ":patch_ir_lib",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_proto_library(
    name = "ir_patch_cc_proto",
    deps = [":ir_patch_proto"],
)

# Graph Edit Distance (GED) and Maximum Common Subgraph (MCS) implementation

cc_library(
    name = "tinyxml2",
    srcs = ["libs/tinyxml2/tinyxml2.cpp"],
    hdrs = ["libs/tinyxml2/tinyxml2.h"],
    strip_include_prefix = "libs/tinyxml2",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "scipy_lap",
    srcs = ["libs/scipy_lap/rectangular_lsap.cpp"],
    hdrs = ["libs/scipy_lap/rectangular_lsap.h"],
    strip_include_prefix = "libs",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    deps = [
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "gxl_parser",
    srcs = ["gxl_parser.cc"],
    hdrs = ["gxl_parser.h"],
    deps = [
        ":graph",
        ":tinyxml2",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "lap_solver",
    srcs = ["lap_solver.cc"],
    hdrs = ["lap_solver.h"],
    deps = [
        ":scipy_lap",
    ],
)

cc_library(
    name = "ged",
    srcs = ["ged.cc"],
    hdrs = [
        "ged.h",
        "ged_cost_functions.h",
    ],
    deps = [
        ":graph",
        ":lap_solver",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "mcs",
    srcs = ["mcs.cc"],
    hdrs = ["mcs.h"],
    deps = [
        ":graph",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "ir_patch_gen_lib",
    srcs = ["ir_patch_gen.cc"],
    hdrs = ["ir_patch_gen.h"],
    deps = [
        ":ged",
        ":graph",
        ":ir_patch_cc_proto",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:xls_value_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "ged_main",
    srcs = ["ged_main.cc"],
    deps = [
        ":ged",
        ":gxl_parser",
        ":ir_patch_gen_lib",
        ":mcs",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
