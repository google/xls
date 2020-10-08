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

# pytype test and library
load("//dependency_support/pybind11:pybind11.bzl", "xls_pybind_extension")

package(
    default_visibility = [
        "//xls:xls_internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_pybind_extension(
    name = "cpp_ast",
    srcs = [
        "cpp_ast.cc",
        "cpp_ast.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_ast",
    ],
)

py_test(
    name = "cpp_ast_test",
    srcs = ["cpp_ast_test.py"],
    python_version = "PY3",
    deps = [
        ":cpp_ast",
        ":cpp_pos",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "cpp_ast_visitor",
    srcs = ["cpp_ast_visitor.py"],
    deps = [
        ":cpp_ast",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "cpp_ast_visitor_test",
    srcs = ["cpp_ast_visitor_test.py"],
    python_version = "PY3",
    deps = [
        ":cpp_ast",
        ":cpp_ast_visitor",
        ":cpp_pos",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

xls_pybind_extension(
    name = "cpp_scanner",
    srcs = ["cpp_scanner.cc"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_ast_builtin_types",
        "//xls/dslx:cpp_pos",
        "//xls/dslx:cpp_scanner",
    ],
)

xls_pybind_extension(
    name = "cpp_pos",
    srcs = ["cpp_pos.cc"],
    deps = [
        "@com_google_absl//absl/base",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_pos",
    ],
)

xls_pybind_extension(
    name = "cpp_parser",
    srcs = [
        "cpp_ast.h",
        "cpp_parser.cc",
    ],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
        ":cpp_scanner",  # build_cleaner: keep
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_parser",
    ],
)

xls_pybind_extension(
    name = "cpp_parametric_expression",
    srcs = ["cpp_parametric_expression.cc"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_parametric_expression",
    ],
)

xls_pybind_extension(
    name = "cpp_concrete_type",
    srcs = [
        "cpp_ast.h",
        "cpp_concrete_type.cc",
    ],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_concrete_type",
    ],
)

xls_pybind_extension(
    name = "cpp_type_info",
    srcs = [
        "cpp_ast.h",
        "cpp_type_info.cc",
    ],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_type_info",
    ],
)
