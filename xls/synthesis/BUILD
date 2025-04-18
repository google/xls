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

# pytype binary, test, library
load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@grpc//bazel:python_rules.bzl", "py_grpc_library")
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

proto_library(
    name = "synthesis_proto",
    srcs = ["synthesis.proto"],
    deps = ["//xls/codegen:module_signature_proto"],
)

cc_proto_library(
    name = "synthesis_cc_proto",
    deps = [":synthesis_proto"],
)

proto_library(
    name = "synthesis_service_proto",
    srcs = ["synthesis_service.proto"],
    deps = [
        ":synthesis_proto",
    ],
)

py_proto_library(
    name = "synthesis_service_py_pb2",
    deps = [":synthesis_service_proto"],
)

py_proto_library(
    name = "synthesis_py_pb2",
    deps = [":synthesis_proto"],
)

cc_proto_library(
    name = "synthesis_service_cc_proto",
    deps = [":synthesis_service_proto"],
)

cc_grpc_library(
    name = "synthesis_service_cc_grpc",
    srcs = [":synthesis_service_proto"],
    grpc_only = 1,
    deps = [
        ":synthesis_service_cc_proto",
        "@grpc//:grpc++",
    ],
)

py_grpc_library(
    name = "synthesis_service_py_pb2_grpc",
    srcs = [":synthesis_service_proto"],
    deps = [":synthesis_service_py_pb2"],
)

py_binary(
    name = "synthesis_client_py",
    srcs = ["synthesis_client_main.py"],
    main = "synthesis_client_main.py",
    deps = [
        ":client_credentials",
        ":synthesis_py_pb2",
        ":synthesis_service_py_pb2_grpc",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/logging",
    ],
)

py_binary(
    name = "op_characterization_client_main",
    srcs = ["op_characterization_client_main.py"],
    deps = [
        ":client_credentials",
        ":op_characterization_client",
        ":synthesis_service_py_pb2_grpc",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

py_test(
    name = "op_characterization_client_test",
    srcs = ["op_characterization_client_test.py"],
    deps = [
        ":op_characterization_client",
        "//xls/estimators:estimator_model_py_pb2",
        "//xls/estimators:estimator_model_utils",
        "@abseil-py//absl/testing:absltest",
    ],
)

cc_binary(
    name = "fake_synthesis_server_main",
    srcs = ["fake_synthesis_server_main.cc"],
    deps = [
        ":credentials",
        ":synthesis_cc_proto",
        ":synthesis_service_cc_grpc",
        "//xls/common:init_xls",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@grpc//:grpc++",
    ],
)

py_test(
    name = "synthesis_server_test",
    srcs = ["synthesis_server_test.py"],
    data = [
        ":fake_synthesis_server_main",
        ":synthesis_client_main",
    ],
    # 2020-01-08: Two test cases at the moment.
    shard_count = 2,
    deps = [
        ":synthesis_py_pb2",
        requirement("portpicker"),
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

cc_library(
    name = "credentials",
    srcs = ["credentials.cc"],
    hdrs = ["credentials.h"],
    deps = [
        "@grpc",
        "@grpc//:grpc++",
    ],
)

py_library(
    name = "client_credentials",
    srcs = ["client_credentials.py"],
    deps = ["@grpc//src/python/grpcio/grpc:grpcio"],
)

py_library(
    name = "synthesis_utils",
    srcs = ["synthesis_utils.py"],
    deps = [
        ":synthesis_py_pb2",
        ":synthesis_service_py_pb2_grpc",
        "@abseil-py//absl/logging",
    ],
)

py_library(
    name = "op_characterization_client",
    srcs = ["op_characterization_client.py"],
    data = [
        "//xls/ir:op_name_representations_proto",
    ],
    deps = [
        ":synthesis_py_pb2",
        ":synthesis_service_py_pb2_grpc",
        "//xls/common:gfile",
        "//xls/common:runfiles",
        "//xls/estimators:estimator_model_py_pb2",
        "//xls/estimators:estimator_model_utils",
        "//xls/estimators:op_module_generator",
        "//xls/ir:xls_op_name_py_pb2",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "synthesis_utils_test",
    srcs = ["synthesis_utils_test.py"],
    data = [":fake_synthesis_server_main"],
    shard_count = 7,
    deps = [
        ":client_credentials",
        ":synthesis_utils",
        requirement("portpicker"),
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
    ],
)

cc_library(
    name = "synthesis_client",
    srcs = ["synthesis_client.cc"],
    hdrs = ["synthesis_client.h"],
    deps = [
        ":credentials",
        ":synthesis_cc_proto",
        ":synthesis_service_cc_grpc",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@grpc//:grpc++",
    ],
)

cc_binary(
    name = "synthesis_client_main",
    srcs = ["synthesis_client_main.cc"],
    deps = [
        ":synthesis_cc_proto",
        ":synthesis_client",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)
