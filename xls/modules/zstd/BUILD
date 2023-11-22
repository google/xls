# Copyright 2023 The XLS Authors
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

# Build rules for XLS ZSTD codec implementation.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "buffer_dslx",
    srcs = [
        "buffer.x",
    ],
)

xls_dslx_test(
    name = "buffer_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":buffer_dslx",
)

xls_dslx_library(
    name = "zstd_magic_dslx",
    srcs = [
        "zstd_magic.x",
    ],
    deps = [
        ":buffer_dslx",
    ]
)

xls_dslx_test(
    name = "zstd_magic_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":zstd_magic_dslx",
)

xls_dslx_library(
    name = "zstd_frame_header_dslx",
    srcs = [
        "zstd_frame_header.x",
    ],
    deps = [
        ":buffer_dslx",
    ],
)

xls_dslx_test(
    name = "zstd_frame_header_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":zstd_frame_header_dslx",
)

xls_dslx_library(
    name = "zstd_dec_dslx",
    srcs = [
        "zstd_dec.x",
    ],
    deps = [
        ":buffer_dslx",
        ":zstd_frame_header_dslx",
        ":zstd_magic_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "zstd_dec_opt_ir",
    dslx_top = "ZstdDecoder",
    library = "zstd_dec_dslx",
)

cc_test(
    name = "zstd_dec_dslx_test",
    srcs = [
        "zstd_dec_test.cc",
    ],
    data = [
        "data.zst",
        ":zstd_dec_opt_ir",
    ],
    deps = [
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/interpreter:interpreter_proc_runtime",
        "//xls/ir:events",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "@com_github_facebook_zstd//:libzstd",
        "@com_google_googletest//:gtest",
    ],
)
