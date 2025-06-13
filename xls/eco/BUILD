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
    tags = ["notap"],
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
    name = "ir2nx",
    srcs = ["ir2nx.py"],
    deps = [
        ":xls_types",
        ":xls_values",
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
    tags = ["notap"],
    deps = [
        ":ir2nx",
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
    tags = ["notap"],
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

cc_binary(
    name = "patch_ir_main",
    srcs = ["patch_ir_main.cc"],
    deps = [
        ":ir_patch_cc_proto",
        ":patch_ir",
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
    ],
)

cc_library(
    name = "patch_ir",
    srcs = [
        "patch_ir.cc",
    ],
    hdrs = ["patch_ir.h"],
    deps = [
        ":ir_patch_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:state_element",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:xls_type_cc_proto",
        "//xls/ir:xls_value_cc_proto",
        "//xls/scheduling:pipeline_schedule",
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
    name = "codegen_scheduled_main",
    srcs = ["codegen_scheduled_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/codegen:codegen_result",
        "//xls/codegen:module_signature",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dev_tools:tool_timeout",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:verifier",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "//xls/tools:codegen",
        "//xls/tools:codegen_flags",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:scheduling_options_flags",
        "//xls/tools:scheduling_options_flags_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)
