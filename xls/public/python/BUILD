# Copyright 2021 The XLS Authors
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
load("//dependency_support/pybind11:pybind11.bzl", "xls_pybind_extension")

package(
    default_visibility = [
        "//xls:xls_internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "runtime_build_actions",
    srcs = ["runtime_build_actions.cc"],
    deps = [
        "//xls/common/status:import_status_module",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/public:runtime_build_actions",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

py_test(
    name = "runtime_build_actions_test",
    srcs = ["runtime_build_actions_test.py"],
    python_version = "PY3",
    deps = [
        ":runtime_build_actions",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common/python:init_xls",
    ],
)

py_library(
    name = "builder",
    srcs = ["builder.py"],
    srcs_version = "PY3",
    deps = [
        "//xls/ir/python:function_builder",
        "//xls/ir/python:package",
    ],
)

py_test(
    name = "builder_test",
    srcs = ["builder_test.py"],
    python_version = "PY3",
    deps = [
        ":builder",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common/python:init_xls",
    ],
)
