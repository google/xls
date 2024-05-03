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
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_fmt_test",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "crc32_dslx",
    srcs = ["crc32.x"],
)

xls_dslx_fmt_test(
    name = "crc32_dslx_fmt_test",
    src = "crc32.x",
)

xls_dslx_test(
    name = "crc32_dslx_test",
    srcs = ["crc32.x"],
    deps = [":crc32_dslx"],
)

xls_dslx_opt_ir(
    name = "crc32",
    srcs = ["crc32.x"],
    dslx_top = "main",
    ir_file = "crc32.ir",
    opt_ir_file = "crc32.opt.ir",
)

xls_benchmark_ir(
    name = "crc32_benchmark_ir",
    src = ":crc32.ir",
    tags = ["optonly"],
)

xls_dslx_opt_ir_test(
    name = "crc32_test",
    dep = ":crc32",
)

cc_xls_ir_jit_wrapper(
    name = "crc32_jit_wrapper",
    src = ":crc32.opt.ir",
    jit_wrapper_args = {
        "class_name": "Crc32",
        "namespace": "xls::examples",
    },
)

cc_binary(
    name = "crc32_reference",
    srcs = ["crc32_reference.cc"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
    ],
)
