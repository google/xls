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

load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pytype_strict_library(
    name = "flags_checks",
    srcs = [
        "flags_checks.py",
    ],
)

pytype_strict_binary(
    name = "n_bit_add_generator",
    srcs = [
        "n_bit_add_generator.py",
    ],
    deps = [
        ":flags_checks",
        "//xls/common:gfile",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "n_bit_mul_generator",
    srcs = [
        "n_bit_mul_generator.py",
    ],
    deps = [
        ":flags_checks",
        "//xls/common:gfile",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

pytype_strict_library(
    name = "n_bit_nested_add_generator",
    srcs = [
        "n_bit_nested_add_generator.py",
    ],
    deps = ["//xls/common:gfile"],
)

pytype_strict_library(
    name = "n_bit_nested_mul_generator",
    srcs = [
        "n_bit_nested_mul_generator.py",
    ],
    deps = ["//xls/common:gfile"],
)

pytype_strict_library(
    name = "n_bit_nested_shift_generator",
    srcs = [
        "n_bit_nested_shift_generator.py",
    ],
    deps = [
        "//xls/common:gfile",
    ],
)

pytype_strict_library(
    name = "solvers_op_comparison_functions",
    srcs = [
        "solvers_op_comparison_functions.py",
    ],
)

pytype_strict_binary(
    name = "solvers_op_comparison_bits_list",
    srcs = [
        "solvers_op_comparison_bits_list.py",
    ],
    deps = [
        ":flags_checks",
        ":n_bit_nested_add_generator",
        ":n_bit_nested_mul_generator",
        ":n_bit_nested_shift_generator",
        ":solvers_op_comparison_functions",
        "//xls/common:gfile",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "solvers_op_comparison_nests_list",
    srcs = [
        "solvers_op_comparison_nests_list.py",
    ],
    deps = [
        ":flags_checks",
        ":n_bit_nested_add_generator",
        ":n_bit_nested_mul_generator",
        ":n_bit_nested_shift_generator",
        ":solvers_op_comparison_functions",
        "//xls/common:gfile",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "plot_csv_solver_speed_data",
    srcs = [
        "plot_csv_solver_speed_data.py",
    ],
    deps = [
        "//xls/common:gfile",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@xls_pip_deps//matplotlib",
    ],
)
