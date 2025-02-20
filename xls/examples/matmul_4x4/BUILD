# Copyright 2024 The XLS Authors
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
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "matmul_4x4_dslx",
    srcs = ["matmul_4x4.x"],
)

xls_dslx_test(
    name = "matmul_4x4_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":matmul_4x4_dslx",
)

xls_dslx_opt_ir(
    name = "matmul_4x4_opt_ir",
    dslx_top = "matmul_4x4",
    library = ":matmul_4x4_dslx",
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
