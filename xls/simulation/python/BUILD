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

load("//dependency_support/pybind11:pybind11.bzl", "xls_pybind_extension")

package(
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "module_simulator",
    srcs = ["module_simulator.cc"],
    py_deps = [
        "//xls/codegen/python:module_signature",  # build_cleaner: keep
        "//xls/ir/python:value",  # build_cleaner: keep
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:vast",
        "//xls/common/status:import_status_module",
        "//xls/ir:value",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:statusor_caster",
    ],
)
