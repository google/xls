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

# pytype tests are present in this file
load("//dependency_support/pybind11:pybind11.bzl", "xls_pybind_extension")

package(
    default_visibility = [
        "//xls:xls_internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "bits",
    srcs = ["bits.cc"],
    deps = [
        "//xls/common/status:import_status_module",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

py_test(
    name = "bits_test",
    srcs = ["bits_test.py"],
    python_version = "PY3",
    deps = [
        ":bits",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

xls_pybind_extension(
    name = "fileno",
    srcs = ["fileno.cc"],
    deps = [
        "//xls/ir:source_location",
    ],
)

py_test(
    name = "fileno_test",
    srcs = ["fileno_test.py"],
    python_version = "PY3",
    deps = [
        ":fileno",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

xls_pybind_extension(
    name = "function",
    srcs = ["function.cc"],
    py_deps = [
        ":type",
    ],
    deps = [
        ":wrapper_types",
        "//xls/ir",
    ],
)

py_test(
    name = "function_test",
    srcs = ["function_test.py"],
    python_version = "PY3",
    deps = [
        ":bits",
        ":function_builder",
        ":package",
        ":value",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

xls_pybind_extension(
    name = "function_builder",
    srcs = ["function_builder.cc"],
    py_deps = [
        ":bits",  # build_cleaner: keep
        ":function",  # build_cleaner: keep
        ":lsb_or_msb",  # build_cleaner: keep
        ":package",  # build_cleaner: keep
        ":source_location",  # build_cleaner: keep
        ":type",  # build_cleaner: keep
        ":value",  # build_cleaner: keep
    ],
    deps = [
        ":wrapper_types",
        "//xls/common/status:import_status_module",
        "//xls/ir",
        "//xls/ir:function_builder",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

xls_pybind_extension(
    name = "ir_parser",
    srcs = ["ir_parser.cc"],
    py_deps = [
        ":package",  # build_cleaner: keep
        ":type",  # build_cleaner: keep
        ":value",  # build_cleaner: keep
    ],
    deps = [
        ":wrapper_types",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:import_status_module",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

py_test(
    name = "ir_parser_test",
    srcs = ["ir_parser_test.py"],
    python_version = "PY3",
    deps = [
        ":ir_parser",
        ":package",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "function_builder_test",
    srcs = ["function_builder_test.py"],
    python_version = "PY3",
    deps = [
        ":bits",
        ":fileno",
        ":function_builder",
        ":lsb_or_msb",
        ":package",
        ":source_location",
        ":value",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

xls_pybind_extension(
    name = "lsb_or_msb",
    srcs = ["lsb_or_msb.cc"],
    deps = [
        "//xls/ir",
    ],
)

xls_pybind_extension(
    name = "package",
    srcs = ["package.cc"],
    py_deps = [
        ":fileno",  # build_cleaner: keep
        ":function",  # build_cleaner: keep
        ":type",  # build_cleaner: keep
    ],
    deps = [
        ":wrapper_types",
        "//xls/common/status:import_status_module",
        "//xls/ir",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

py_test(
    name = "package_test",
    srcs = ["package_test.py"],
    python_version = "PY3",
    deps = [
        ":bits",
        ":fileno",
        ":function",
        ":function_builder",
        ":package",
        ":type",
        ":value",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

xls_pybind_extension(
    name = "source_location",
    srcs = ["source_location.cc"],
    py_deps = [
        ":fileno",  # build_cleaner: keep
    ],
    deps = [
        "//xls/ir:source_location",
    ],
)

xls_pybind_extension(
    name = "type",
    srcs = ["type.cc"],
    deps = [
        ":wrapper_types",
        "//xls/ir",  # build_cleaner: keep
        "//xls/ir:type",
        "@com_google_protobuf//:protobuf",
    ],
)

py_test(
    name = "type_test",
    srcs = ["type_test.py"],
    python_version = "PY3",
    deps = [
        ":package",
        ":type",
        "//xls/common:test_base",
    ],
)

cc_library(
    name = "wrapper_types",
    hdrs = ["wrapper_types.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:type",
        "@pybind11",
    ],
)

xls_pybind_extension(
    name = "op",
    srcs = ["op.cc"],
    deps = [
        "@com_google_absl//absl/strings",
        "//xls/ir:op",
    ],
)

xls_pybind_extension(
    name = "value",
    srcs = ["value.cc"],
    py_deps = [
        ":bits",  # build_cleaner: keep
    ],
    deps = [
        "//xls/common/status:import_status_module",
        "//xls/ir:value",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

xls_pybind_extension(
    name = "verifier",
    srcs = ["verifier.cc"],
    py_deps = [
        ":function",  # build_cleaner: keep
        ":package",  # build_cleaner: keep
    ],
    deps = [
        ":wrapper_types",
        "//xls/common/status:import_status_module",
        "//xls/ir",
        "@pybind11_abseil//pybind11_abseil:status_caster",
    ],
)

py_test(
    name = "verifier_test",
    srcs = ["verifier_test.py"],
    python_version = "PY3",
    deps = [
        ":bits",
        ":function_builder",
        ":package",
        ":value",
        ":verifier",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "op_test",
    srcs = ["op_test.py"],
    deps = [
        ":op",
        "//xls/common:test_base",
    ],
)
