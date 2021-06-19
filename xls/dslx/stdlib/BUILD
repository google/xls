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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_module_library",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_module_library(
    name = "std_dslx_module",
    src = "std.x",
)

xls_dslx_test(
    name = "std_dslx_test",
    dep = "std_dslx_module",
)

xls_dslx_module_library(
    name = "acm_random_dslx_module",
    src = "acm_random.x",
)

xls_dslx_test(
    name = "acm_random_dslx_test",
    dep = "acm_random_dslx_module",
)

xls_dslx_module_library(
    name = "apfloat_dslx_module",
    src = "apfloat.x",
)

xls_dslx_test(
    name = "apfloat_dslx_test",
    dep = "apfloat_dslx_module",
)

xls_dslx_module_library(
    name = "bfloat16_dslx_module",
    src = "bfloat16.x",
)

xls_dslx_test(
    name = "bfloat16_dslx_test",
    dep = "bfloat16_dslx_module",
)

xls_dslx_module_library(
    name = "float32_dslx_module",
    src = "float32.x",
)

xls_dslx_test(
    name = "float32_dslx_test",
    dep = "float32_dslx_module",
)

xls_dslx_module_library(
    name = "float64_dslx_module",
    src = "float64.x",
)

xls_dslx_test(
    name = "float64_dslx_test",
    dep = "float64_dslx_module",
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_users"],
)
