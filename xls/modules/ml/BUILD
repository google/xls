# Copyright 2026 The XLS Authors
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

# Machine-learning primitives for DSLX (float32).

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_fmt_test",
    "xls_dslx_library",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "activation_dslx",
    srcs = ["activation.x"],
)

xls_dslx_fmt_test(
    name = "activation_dslx_fmt_test",
    src = "activation.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "activation_dslx_test",
    size = "small",
    srcs = ["activation.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "ws_matmul_dslx",
    srcs = ["ws_matmul.x"],
)

xls_dslx_fmt_test(
    name = "ws_matmul_dslx_fmt_test",
    src = "ws_matmul.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "ws_matmul_dslx_test",
    size = "small",
    srcs = ["ws_matmul.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "os_matmul_dslx",
    srcs = ["os_matmul.x"],
)

xls_dslx_fmt_test(
    name = "os_matmul_dslx_fmt_test",
    src = "os_matmul.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "os_matmul_dslx_test",
    size = "small",
    srcs = ["os_matmul.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "mac_dslx",
    srcs = ["mac.x"],
)

xls_dslx_fmt_test(
    name = "mac_dslx_fmt_test",
    src = "mac.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "mac_dslx_test",
    size = "small",
    srcs = ["mac.x"],
    dslx_test_args = {"compare": "jit"},
)
