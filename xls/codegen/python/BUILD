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
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "module_signature",
    srcs = ["module_signature.cc"],
    deps = [
        "//xls/codegen:module_signature",
    ],
)

xls_pybind_extension(
    name = "pipeline_generator",
    srcs = ["pipeline_generator.cc"],
    py_deps = [
        ":module_signature",  # build_cleaner: keep
        "//xls/ir/python:package",  # build_cleaner: keep
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/codegen:module_signature",
        "//xls/codegen:pipeline_generator",
        "//xls/common/status:import_status_module",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir/python:wrapper_types",
        "//xls/scheduling:pipeline_schedule",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

py_test(
    name = "pipeline_generator_test",
    srcs = ["pipeline_generator_test.py"],
    python_version = "PY3",
    deps = [
        ":pipeline_generator",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common/python:init_xls",
        "//xls/ir/python:function_builder",
        "//xls/ir/python:package",
    ],
)
