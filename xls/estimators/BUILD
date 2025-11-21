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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
# Load proto_library
# cc_proto_library is used in this file

load("@xls_pip_deps//:requirements.bzl", "requirement")
load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

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

# Helper alias for all the configured area and delay models.
cc_library(
    name = "estimators",
    # Don't remove this. It registers the models.
    tags = ["keep_dep"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/estimators/area_model/models",
        "//xls/estimators/delay_model/models",
    ],
    alwayslink = 1,
)

pytype_strict_library(
    name = "estimator_model",
    srcs = ["estimator_model.py"],
    deps = [
        ":estimator_model_py_pb2",
        requirement("numpy"),
        requirement("scipy"),
    ],
)

pytype_strict_contrib_test(
    name = "estimator_model_test",
    srcs = ["estimator_model_test.py"],
    tags = ["nomsan"],  # scipy seems not to be msan-clean.
    deps = [
        ":estimator_model",
        ":estimator_model_py_pb2",
        "@abseil-py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_library(
    name = "op_module_generator",
    srcs = ["op_module_generator.py"],
    data = [
        "//xls/dev_tools:parse_ir",
        "//xls/tools:codegen_main",
    ],
    deps = [
        "//xls/codegen:module_signature_py_pb2",
        "//xls/common:runfiles",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_contrib_test(
    name = "op_module_generator_test",
    srcs = ["op_module_generator_test.py"],
    data = glob(["testdata/*"]),
    deps = [
        ":op_module_generator",
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
    ],
)

pytype_strict_binary(
    name = "estimator_model_join",
    srcs = ["estimator_model_join.py"],
    deps = [
        ":estimator_model",
        ":estimator_model_py_pb2",
        ":estimator_model_utils",
        "//xls/common:gfile",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_contrib_test(
    name = "estimator_model_join_test",
    srcs = ["estimator_model_join_test.py"],
    data = [
        ":estimator_model_join",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@abseil-py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "estimator_model_utils",
    srcs = ["estimator_model_utils.py"],
    deps = [":estimator_model_py_pb2"],
)

pytype_strict_contrib_test(
    name = "estimator_model_utils_test",
    srcs = ["estimator_model_utils_test.py"],
    deps = [
        ":estimator_model_py_pb2",
        ":estimator_model_utils",
        "//xls/common:test_base",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

pytype_strict_binary(
    name = "extract_sample_points_from_estimator_model",
    srcs = ["extract_sample_points_from_estimator_model.py"],
    deps = [
        ":estimator_model_py_pb2",
        "//xls/common:gfile",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_contrib_test(
    name = "extract_sample_points_from_estimator_model_test",
    srcs = ["extract_sample_points_from_estimator_model_test.py"],
    data = [
        ":extract_sample_points_from_estimator_model",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@abseil-py//absl/testing:absltest",
    ],
)

pytype_strict_binary(
    name = "estimator_model_stats",
    srcs = ["estimator_model_stats.py"],
    deps = [
        requirement("numpy"),
        "//xls/estimators:estimator_model",
        "//xls/estimators:estimator_model_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_contrib_test(
    name = "estimator_model_stats_test",
    srcs = ["estimator_model_stats_test.py"],
    data = [
        "testdata/asap7_delay_data_points.textproto",
        "testdata/asap7_delay_model_stats.csv",
        "testdata/sky130_delay_data_points.textproto",
        "testdata/sky130_delay_model_stats.csv",
        "testdata/unit_delay_data_points.textproto",
        "testdata/unit_delay_model_stats.csv",
        ":estimator_model_stats",
    ],
    tags = [
        "noasan",  # TODO: b/436249343 - Remove this when SciPy is patched.
        "nomsan",  # scipy seems not to be msan-clean.
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@abseil-py//absl/testing:absltest",
    ],
)

pytype_strict_binary(
    name = "run_op_characterization",
    srcs = ["run_op_characterization.py"],
    deps = [
        requirement("portpicker"),
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/logging",
    ],
)
