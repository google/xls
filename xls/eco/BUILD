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

# pytype test and library

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
        "@com_google_absl_py//absl/logging",
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
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
