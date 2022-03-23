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

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "delay_estimator",
    srcs = ["delay_estimator.cc"],
    hdrs = ["delay_estimator.h"],
    deps = [
        "//xls/common/status:status_builder",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/netlist:logical_effort",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "delay_estimator_test",
    srcs = ["delay_estimator_test.cc"],
    deps = [
        ":delay_estimator",
        ":delay_estimators",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "delay_heap_test",
    srcs = ["delay_heap_test.cc"],
    deps = [
        ":analyze_critical_path",
        ":delay_estimator",
        ":delay_estimators",
        ":delay_heap",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/ir:type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "delay_estimators_test",
    srcs = ["delay_estimators_test.cc"],
    deps = [
        ":delay_estimator",
        ":delay_estimators",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "analyze_critical_path",
    srcs = ["analyze_critical_path.cc"],
    hdrs = ["analyze_critical_path.h"],
    deps = [
        ":delay_estimator",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "analyze_critical_path_test",
    srcs = ["analyze_critical_path_test.cc"],
    deps = [
        ":analyze_critical_path",
        ":delay_estimator",
        ":delay_estimators",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "delay_heap",
    srcs = ["delay_heap.cc"],
    hdrs = ["delay_heap.h"],
    deps = [
        ":delay_estimator",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "delay_estimators",
    srcs = ["delay_estimators.cc"],
    hdrs = ["delay_estimators.h"],
    deps = [
        ":delay_estimator",
        "//xls/delay_model/models",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "delay_model_proto",
    srcs = ["delay_model.proto"],
)

xls_py_proto_library(
    name = "delay_model_py_pb2",
    srcs = ["delay_model.proto"],
    internal_deps = [":delay_model_proto"],
)

py_binary(
    name = "generate_delay_lookup",
    srcs = ["generate_delay_lookup.py"],
    data = ["generate_delay_lookup.tmpl"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":delay_model",
        ":delay_model_py_pb2",
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "delay_model",
    srcs = ["delay_model.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":delay_model_py_pb2",
        requirement("numpy"),
        requirement("scipy"),
    ],
)

py_test(
    name = "delay_model_test",
    srcs = ["delay_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":delay_model",
        ":delay_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "op_module_generator",
    srcs = ["op_module_generator.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/codegen:module_signature_py_pb2",
        "//xls/codegen/python:module_signature",
        "//xls/codegen/python:pipeline_generator",
        "//xls/ir/python:ir_parser",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "op_module_generator_test",
    srcs = ["op_module_generator_test.py"],
    data = glob(["testdata/*"]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":op_module_generator",
        "//xls/common:runfiles",
        "//xls/common/python:init_xls",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
