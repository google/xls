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
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
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
        "@com_github_grpc_grpc//:grpc++",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_credentials",
        ":synthesis_py_pb2",
        ":synthesis_service_py_pb2_grpc",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
    ],
)

py_binary(
    name = "timing_characterization_client_main",
    srcs = ["timing_characterization_client_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_credentials",
        ":synthesis_service_py_pb2_grpc",
        ":timing_characterization_client",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_test(
    name = "timing_characterization_client_test",
    srcs = ["timing_characterization_client_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":timing_characterization_client",
        "//xls/delay_model:delay_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
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
        "//xls/common/logging",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

py_test(
    name = "synthesis_server_test",
    srcs = ["synthesis_server_test.py"],
    data = [
        ":fake_synthesis_server_main",
        ":synthesis_client_main",
    ],
    python_version = "PY3",
    # 2020-01-08: Two test cases at the moment.
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":synthesis_py_pb2",
        requirement("portpicker"),
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

cc_library(
    name = "credentials",
    srcs = ["credentials.cc"],
    hdrs = ["credentials.h"],
    deps = ["@com_github_grpc_grpc//:grpc++"],
)

py_library(
    name = "client_credentials",
    srcs = ["client_credentials.py"],
    srcs_version = "PY3",
    deps = ["@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio"],
)

py_library(
    name = "synthesis_utils",
    srcs = ["synthesis_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":client_credentials",
        ":synthesis_py_pb2",
        ":synthesis_service_py_pb2_grpc",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "timing_characterization_client",
    srcs = ["timing_characterization_client.py"],
    srcs_version = "PY3",
    deps = [
        ":synthesis_py_pb2",
        ":synthesis_service_py_pb2_grpc",
        "//xls/common:gfile",
        "//xls/delay_model:delay_model_py_pb2",
        "//xls/delay_model:op_module_generator",
        "//xls/ir:op_specification_typechecked",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "synthesis_utils_test",
    srcs = ["synthesis_utils_test.py"],
    data = [":fake_synthesis_server_main"],
    python_version = "PY3",
    shard_count = 7,
    srcs_version = "PY3",
    deps = [
        ":client_credentials",
        ":synthesis_utils",
        requirement("portpicker"),
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/testing:absltest",
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
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//xls/common/logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)
