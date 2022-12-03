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
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "standard_pipeline",
    srcs = ["standard_pipeline.cc"],
    py_deps = [
        "//xls/ir/python:package",  # build_cleaner: keep
    ],
    deps = [
        "//xls/common/status:import_status_module",
        "//xls/ir/python:wrapper_types",
        "//xls/passes:standard_pipeline",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

py_test(
    name = "standard_pipeline_test",
    srcs = ["standard_pipeline_test.py"],
    python_version = "PY3",
    deps = [
        ":standard_pipeline",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common/python:init_xls",
        "//xls/ir/python:bits",
        "//xls/ir/python:function_builder",
        "//xls/ir/python:package",
    ],
)
