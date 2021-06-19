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

# Build rules for XLS examples.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_module_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)
# Examples used in the DSLX introduction document.

xls_dslx_module_library(
    name = "crc32_one_byte_dslx_module",
    src = "crc32_one_byte.x",
)

xls_dslx_opt_ir(
    name = "crc32_one_byte",
    dep = ":crc32_one_byte_dslx_module",
)

xls_dslx_opt_ir_test(
    name = "crc32_one_byte_test",
    dep = ":crc32_one_byte",
)

xls_dslx_module_library(
    name = "crc32_one_byte_inferred_dslx_module",
    src = "crc32_one_byte_inferred.x",
)

xls_dslx_opt_ir(
    name = "crc32_one_byte_inferred",
    dep = ":crc32_one_byte_inferred_dslx_module",
)

xls_dslx_opt_ir_test(
    name = "crc32_one_byte_inferred_test",
    dep = ":crc32_one_byte_inferred",
)

xls_dslx_module_library(
    name = "prefix_scan_equality_dslx_module",
    src = "prefix_scan_equality.x",
)

# TODO(leary): 2019-07-24 Missing conversion of 'for/enumerate'.
xls_dslx_test(
    name = "prefix_scan_equality_dslx_test",
    dep = ":prefix_scan_equality_dslx_module",
    dslx_test_args = {
        "compare": "none",
    },
)
