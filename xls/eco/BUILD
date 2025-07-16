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

# pytype binary, test, library
# @rules_cc//cc:cc_binary.bzl
load("@rules_python//python:proto.bzl", "py_proto_library")
# Load proto_library
# cc_proto_library is used in this file

load("@xls_pip_deps//:requirements.bzl", "requirement")

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

py_library(
    name = "xls_ir_to_networkx",
    srcs = ["xls_ir_to_networkx.py"],
    data = [":xls_ir_to_cytoscape"],
    deps = [
        "@abseil-py//absl/logging",
        requirement("networkx"),
        # networkx annoyingly marks scipy as an optional dep, but our usage requires it.
        # Also see b/394358042.
        requirement("scipy"),  # buildcleaner: keep
        "//xls/common:runfiles",
    ],
)

py_test(
    name = "xls_ir_to_networkx_test",
    srcs = ["xls_ir_to_networkx_test.py"],
    data = ["//xls/examples:riscv_simple.opt.ir"],
    deps = [
        ":xls_ir_to_networkx",
        requirement("networkx"),
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "ir_diff",
    srcs = ["ir_diff.py"],
    deps = [
        requirement("networkx"),
    ],
)

py_library(
    name = "ir_diff_utils",
    srcs = ["ir_diff_utils.py"],
    deps = [
        ":ir_diff",
        requirement("matplotlib"),
        requirement("networkx"),
    ],
)

py_library(
    name = "ir_patch_gen",
    srcs = ["ir_patch_gen.py"],
    deps = [
        ":ir_diff",
        ":ir_patch_py_pb2",
        ":xls_types",
        ":xls_values",
    ],
)

py_library(
    name = "xls_types",
    srcs = ["xls_types.py"],
    deps = ["//xls/ir:xls_type_py_pb2"],
)

py_library(
    name = "xls_values",
    srcs = ["xls_values.py"],
    deps = [
        ":xls_types",
        "//xls/ir:xls_value_py_pb2",
    ],
)

py_binary(
    name = "ir_diff_main",
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

py_test(
    name = "ir_diff_main_test",
    srcs = ["ir_diff_main_test.py"],
    data = [
        ":ir_diff_main",
        "//xls/examples:riscv_simple.ir",  # unopt
        "//xls/examples:riscv_simple.opt.ir",  #opt
    ],
    deps = [
        ":ir_patch_py_pb2",
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
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

cc_proto_library(
    name = "ir_patch_cc_proto",
    deps = [":ir_patch_proto"],
)
