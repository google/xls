# Copyright 2024 The XLS Authors
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
load("@rules_python//python:proto.bzl", "py_proto_library")
# cc_proto_library is used in this file

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "estimator_model_proto",
    srcs = ["estimator_model.proto"],
)

cc_proto_library(
    name = "estimator_model_cc_proto",
    deps = [":estimator_model_proto"],
)

py_proto_library(
    name = "estimator_model_py_pb2",
    deps = [":estimator_model_proto"],
)

py_library(
    name = "estimator_model",
    srcs = ["estimator_model.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":estimator_model_py_pb2",
        requirement("numpy"),
        requirement("scipy"),
    ],
)

py_test(
    name = "estimator_model_test",
    srcs = ["estimator_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nomsan"],  # scipy seems not to be msan-clean.
    deps = [
        ":estimator_model",
        ":estimator_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)
