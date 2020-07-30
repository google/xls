# Copyright 2020 Google LLC
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
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "z3_lec",
    srcs = ["z3_lec.cc"],
    deps = [
        "//xls/common/file:filesystem",
        "//xls/common/python:absl_casters",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:netlist_cc_proto",
        "//xls/netlist:netlist_parser",
        "//xls/solvers:z3_lec",
        "@com_google_protobuf//:protobuf",
    ],
)

py_test(
    name = "z3_lec_test",
    srcs = ["z3_lec_test.py"],
    data = ["//xls/netlist:fake_cell_library.textproto"],
    python_version = "PY3",
    deps = [
        ":z3_lec",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common:runfiles",
        "//xls/common/python:init_xls",
    ],
)
