# Copyright 2022 The XLS Authors
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
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "dummy_metrics_main",
    srcs = ["dummy_metrics_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "@at_clifford_yosys//:json11",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "json_metrics_server_main",
    srcs = ["json_metrics_server_main.cc"],
    deps = [
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:status_macros",
        "//xls/synthesis:credentials",
        "//xls/synthesis:synthesis_cc_proto",
        "//xls/synthesis:synthesis_service_cc_grpc",
        "@at_clifford_yosys//:json11",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

py_test(
    name = "json_metrics_server_test",
    srcs = ["json_metrics_server_test.py"],
    data = [
        ":dummy_metrics_main",
        ":json_metrics_server_main",
        "//xls/synthesis:synthesis_client_main",
    ],
    python_version = "PY3",
    # One shard for each different server test.
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        requirement("portpicker"),
        "//xls/common:runfiles",
        "//xls/synthesis:synthesis_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

exports_files(["sta_by_stage.tcl"])
