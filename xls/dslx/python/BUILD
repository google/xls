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

# Usually the pybind extension modules don't need shared definitions, but the
# "Holder" wrapper objects for AST nodes are used across a few different pybind
# extensions, so it's broken out here.
cc_library(
    name = "cpp_ast_header",
    hdrs = ["cpp_ast.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "//xls/dslx:cpp_ast",
        "@pybind11",
    ],
)

cc_library(
    name = "errors",
    hdrs = ["errors.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/strings",
        "//xls/common:string_to_int",
        "//xls/dslx:cpp_bindings",
        "//xls/dslx:cpp_concrete_type",
        "//xls/dslx:deduce_ctx",
        "@pybind11",
    ],
)

xls_pybind_extension(
    name = "cpp_ast",
    srcs = [
        "cpp_ast.cc",
    ],
    deps = [
        ":cpp_ast_header",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
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
    name = "scanner",
    srcs = ["scanner.cc"],
    deps = [
        ":errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_ast_builtin_types",
        "//xls/dslx:cpp_pos",
        "//xls/dslx:scanner",
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
        "cpp_parser.cc",
    ],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
        ":scanner",  # build_cleaner: keep
    ],
    deps = [
        ":cpp_ast_header",
        ":errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
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
        "cpp_concrete_type.cc",
    ],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
    ],
    deps = [
        ":cpp_ast_header",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_concrete_type",
    ],
)

xls_pybind_extension(
    name = "interp_value",
    srcs = [
        "interp_value.cc",
    ],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
    ],
    deps = [
        ":cpp_ast_header",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:interp_value",
    ],
)

xls_pybind_extension(
    name = "import_routines",
    srcs = [
        "import_routines.cc",
    ],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
        ":interp_value",  # build_cleaner: keep
    ],
    deps = [
        ":cpp_ast_header",
        ":errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_ast",
        "//xls/dslx:cpp_bindings",
        "//xls/dslx:cpp_evaluate",
        "//xls/dslx:cpp_type_info",
        "//xls/dslx:import_routines",
    ],
)

xls_pybind_extension(
    name = "interpreter",
    srcs = ["interpreter.cc"],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
        ":interp_value",  # build_cleaner: keep
        ":cpp_concrete_type",  # build_cleaner: keep
    ],
    deps = [
        ":cpp_ast_header",
        ":errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_ast",
        "//xls/dslx:cpp_concrete_type",
        "//xls/dslx:cpp_evaluate",
        "//xls/dslx:cpp_parse_and_typecheck",
        "//xls/dslx:cpp_parser",
        "//xls/dslx:cpp_typecheck",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_bindings",
        "//xls/dslx:interpreter",
        "//xls/dslx:scanner",
        "//xls/dslx:symbolic_bindings",
        "//xls/ir/python:wrapper_types",
    ],
)

xls_pybind_extension(
    name = "cpp_deduce",
    srcs = ["cpp_deduce.cc"],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
        ":cpp_concrete_type",  # build_cleaner: keep
        ":import_routines",  # build_cleaner: keep
    ],
    deps = [
        ":cpp_ast_header",
        ":errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_ast",
        "//xls/dslx:cpp_concrete_type",
        "//xls/dslx:cpp_deduce",
        "//xls/dslx:cpp_evaluate",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_bindings",
        "//xls/dslx:interpreter",
        "//xls/ir/python:wrapper_types",
    ],
)

xls_pybind_extension(
    name = "cpp_typecheck",
    srcs = ["cpp_typecheck.cc"],
    py_deps = [
        ":cpp_ast",  # build_cleaner: keep
        ":cpp_deduce",  # build_cleaner: keep
        ":cpp_concrete_type",  # build_cleaner: keep
        ":import_routines",  # build_cleaner: keep
    ],
    deps = [
        ":cpp_ast_header",
        ":errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor_pybind_caster",
        "//xls/dslx:cpp_ast",
        "//xls/dslx:cpp_concrete_type",
        "//xls/dslx:cpp_deduce",
        "//xls/dslx:cpp_evaluate",
        "//xls/dslx:cpp_typecheck",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_bindings",
        "//xls/dslx:interpreter",
        "//xls/ir/python:wrapper_types",
    ],
)
