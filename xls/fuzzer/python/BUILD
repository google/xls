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

load("//dependency_support/pybind11:pybind11.bzl", "xls_pybind_extension")

package(
    default_visibility = [
        "//xls:xls_internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "cpp_ast_generator",
    srcs = [
        "cpp_ast_generator.cc",
    ],
    py_deps = [
        "//xls/dslx/python:interp_value",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:import_status_module",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:ast",
        "//xls/fuzzer:ast_generator",
        "//xls/fuzzer:cpp_sample_generator",
        "//xls/fuzzer:value_generator",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)

xls_pybind_extension(
    name = "cpp_sample",
    srcs = [
        "cpp_sample.cc",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@at_clifford_yosys//:json11",
        "//xls/common/status:import_status_module",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value_helpers",
        "//xls/fuzzer:ast_generator",
        "//xls/fuzzer:cpp_sample",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)
