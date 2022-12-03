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

load("@xls_pip_deps//:requirements.bzl", "requirement")

# pytype binary, test, library
load("//xls/build_rules:py_proto_library.bzl", "xls_py_proto_library")
load("//dependency_support/pybind11:pybind11.bzl", "xls_pybind_extension")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "characterize_binops",
    srcs = ["characterize_binops.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lec_characterizer",
        ":lec_characterizer_py_pb2",
        ":z3_lec",
        "//xls/common:gfile",
        "//xls/ir:op_py_pb2",
        "//xls/ir:xls_type_py_pb2",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "lec_characterizer",
    srcs = ["lec_characterizer.py"],
    srcs_version = "PY3",
    deps = [
        ":lec_characterizer_py_pb2",
        ":z3_lec",
        "//xls/common:gfile",
        "//xls/delay_model:op_module_generator",
        "//xls/ir:op_py_pb2",
        "//xls/ir/python:ir_parser",
        "//xls/ir/python:package",
        "//xls/ir/python:type",
        "//xls/synthesis:client_credentials",
        "//xls/synthesis:synthesis_py_pb2",
        "//xls/synthesis:synthesis_service_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

proto_library(
    name = "lec_characterizer_proto",
    srcs = ["lec_characterizer.proto"],
    deps = [
        "//xls/ir:xls_type_proto",
    ],
)

xls_py_proto_library(
    name = "lec_characterizer_py_pb2",
    srcs = ["lec_characterizer.proto"],
    internal_deps = [
        ":lec_characterizer_proto",
        "//xls/ir:xls_type_proto",
    ],
    deps = [
        "//xls/ir:xls_type_py_pb2",
    ],
)

py_test(
    name = "lec_characterizer_test",
    srcs = ["lec_characterizer_test.py"],
    data = [
        "//xls/netlist:fake_cell_library.textproto",
        "//xls/synthesis:dummy_synthesis_server_main",
    ],
    python_version = "PY3",
    deps = [
        ":lec_characterizer",
        ":lec_characterizer_py_pb2",
        requirement("portpicker"),
        "//xls/common:gfile",
        "//xls/common:runfiles",
        "//xls/ir:op_py_pb2",
        "//xls/ir:xls_type_py_pb2",
        "//xls/ir/python:package",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

xls_pybind_extension(
    name = "z3_lec",
    srcs = ["z3_lec.cc"],
    deps = [
        "//xls/common/file:filesystem",
        "//xls/common/status:import_status_module",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:netlist_cc_proto",
        "//xls/netlist:netlist_parser",
        "//xls/solvers:z3_lec",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

py_test(
    name = "z3_lec_test",
    srcs = ["z3_lec_test.py"],
    data = ["//xls/netlist:fake_cell_library.textproto"],
    python_version = "PY3",
    deps = [
        ":z3_lec",
        "//xls/common:runfiles",
        "//xls/common/python:init_xls",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
