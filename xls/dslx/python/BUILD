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
    default_visibility = [
        "//xls:xls_internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "errors",
    hdrs = ["errors.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/strings",
        "//xls/common:string_to_int",
        "//xls/dslx:bindings",
        "//xls/dslx:concrete_type",
        "//xls/dslx:deduce_ctx",
        "@pybind11",
    ],
)

xls_pybind_extension(
    name = "ast",
    srcs = [
        "ast.cc",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:ast",
    ],
)

xls_pybind_extension(
    name = "interp_value",
    srcs = [
        "interp_value.cc",
    ],
    py_deps = [
        "//xls/ir/python:bits",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:interp_value",
        "//xls/dslx:ir_converter",
        "//xls/ir:ir_parser",
    ],
)

xls_pybind_extension(
    name = "interpreter",
    srcs = ["interpreter.cc"],
    py_deps = [
        ":interp_value",  # build_cleaner: keep
    ],
    deps = [
        ":errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:ast",
        "//xls/dslx:concrete_type",
        "//xls/dslx:evaluate",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_bindings",
        "//xls/dslx:interpreter",
        "//xls/dslx:ir_converter",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:parser",
        "//xls/dslx:symbolic_bindings",
        "//xls/dslx:typecheck",
        "//xls/ir:ir_parser",
        "//xls/ir/python:wrapper_types",
    ],
)
