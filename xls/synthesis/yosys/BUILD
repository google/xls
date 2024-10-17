# Copyright 2020 Google LLC
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

# pytype tests are present in this file

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
    name = "yosys_server_main",
    srcs = ["yosys_server_main.cc"],
    deps = [
        ":yosys_synthesis_service",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/synthesis:credentials",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "yosys_synthesis_service",
    srcs = ["yosys_synthesis_service.cc"],
    hdrs = ["yosys_synthesis_service.h"],
    deps = [
        ":yosys_util",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/synthesis:synthesis_cc_proto",
        "//xls/synthesis:synthesis_service_cc_grpc",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "bogusys",
    srcs = ["bogusys.cc"],
    data = ["testdata/netlist.json"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "nextpbr",
    srcs = ["nextpbr.cc"],
    data = ["testdata/nextpnr.out"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

py_test(
    name = "yosys_server_test",
    srcs = ["yosys_server_test.py"],
    data = [
        ":bogusys",
        ":nextpbr",
        ":yosys_server_main",
        "//xls/synthesis:synthesis_client_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("portpicker"),
        "//xls/common:runfiles",
        "//xls/synthesis:synthesis_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

cc_library(
    name = "yosys_util",
    srcs = ["yosys_util.cc"],
    hdrs = ["yosys_util.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "yosys_util_test",
    srcs = ["yosys_util_test.cc"],
    data = ["testdata/nextpnr.out"],
    deps = [
        ":yosys_util",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

exports_files(["synth_by_stage.tcl"])
