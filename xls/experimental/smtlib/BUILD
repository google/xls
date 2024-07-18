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

# Compiler data structures for converting high level algorithms into Hardware
# Description Language.

# pytype binary, library

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "flags_checks",
    srcs = [
        "flags_checks.py",
    ],
)

py_binary(
    name = "n_bit_add_generator",
    srcs = [
        "n_bit_add_generator.py",
    ],
    python_version = "PY3",
    deps = [
        ":flags_checks",
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "n_bit_mul_generator",
    srcs = [
        "n_bit_mul_generator.py",
    ],
    python_version = "PY3",
    deps = [
        ":flags_checks",
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_library(
    name = "n_bit_nested_add_generator",
    srcs = [
        "n_bit_nested_add_generator.py",
    ],
    deps = ["//xls/common:gfile"],
)

py_library(
    name = "n_bit_nested_mul_generator",
    srcs = [
        "n_bit_nested_mul_generator.py",
    ],
    deps = ["//xls/common:gfile"],
)

py_library(
    name = "n_bit_nested_shift_generator",
    srcs = [
        "n_bit_nested_shift_generator.py",
    ],
    deps = [
        "//xls/common:gfile",
    ],
)

py_library(
    name = "solvers_op_comparison_functions",
    srcs = [
        "solvers_op_comparison_functions.py",
    ],
)

py_binary(
    name = "solvers_op_comparison_bits_list",
    srcs = [
        "solvers_op_comparison_bits_list.py",
    ],
    python_version = "PY3",
    deps = [
        ":flags_checks",
        ":n_bit_nested_add_generator",
        ":n_bit_nested_mul_generator",
        ":n_bit_nested_shift_generator",
        ":solvers_op_comparison_functions",
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "solvers_op_comparison_nests_list",
    srcs = [
        "solvers_op_comparison_nests_list.py",
    ],
    python_version = "PY3",
    deps = [
        ":flags_checks",
        ":n_bit_nested_add_generator",
        ":n_bit_nested_mul_generator",
        ":n_bit_nested_shift_generator",
        ":solvers_op_comparison_functions",
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "plot_csv_solver_speed_data",
    srcs = [
        "plot_csv_solver_speed_data.py",
    ],
    python_version = "PY3",
    deps = [
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)
