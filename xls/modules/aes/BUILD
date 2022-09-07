# Copyright 2022 The XLS Authors
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

# Build rules for XLS AES implementations.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_test",
)
load(
    "//xls/build_rules:xls_ir_macros.bzl",
    "xls_ir_cc_library_macro",
)

package(
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "aes_common_dslx",
    srcs = ["aes_common.x"],
    deps = [
        ":constants_dslx",
    ],
)

xls_dslx_library(
    name = "aes_128_common_dslx",
    srcs = ["aes_128_common.x"],
    deps = [
        ":aes_common_dslx",
    ],
)

xls_dslx_library(
    name = "aes_128_dslx",
    srcs = ["aes_128.x"],
    deps = [
        ":aes_128_common_dslx",
        ":aes_common_dslx",
        ":constants_dslx",
    ],
)

xls_dslx_ir(
    name = "aes_128_encrypt",
    dslx_top = "aes_encrypt",
    library = ":aes_128_dslx",
)

xls_dslx_ir(
    name = "aes_128_decrypt",
    dslx_top = "aes_decrypt",
    library = ":aes_128_dslx",
)

xls_dslx_test(
    name = "aes_128_dslx_test",
    library = ":aes_128_dslx",
)

# Note that we're using unoptimized IR: unoptimized compiles in a handful of
# seconds, while optimized takes ~7 minutes and yields roughly the same
# performance.
xls_ir_cc_library_macro(
    name = "aes_128_encrypt_cc",
    src = ":aes_128_encrypt.ir",
    namespaces = "xls,aes",
    top = "aes_128_encrypt",
)

# Note that we're using unoptimized IR: unoptimized compiles in a handful of
# seconds, while optimized takes ~7 minutes and yields roughly the same
# performance.
xls_ir_cc_library_macro(
    name = "aes_128_decrypt_cc",
    src = ":aes_128_decrypt.ir",
    namespaces = "xls,aes",
    top = "aes_128_decrypt",
)

cc_test(
    name = "aes_128_test",
    srcs = ["aes_128_test.cc"],
    deps = [
        ":aes_128_decrypt_cc",
        ":aes_128_encrypt_cc",
        ":aes_test_common",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common:init_xls",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value",
        "@boringssl//:crypto",
    ],
)

cc_library(
    name = "aes_test_common",
    srcs = ["aes_test_common.cc"],
    hdrs = ["aes_test_common.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:value",
    ],
)

xls_dslx_library(
    name = "aes_256_common_dslx",
    srcs = ["aes_256_common.x"],
    deps = [
        ":aes_common_dslx",
        ":constants_dslx",
    ],
)

xls_dslx_library(
    name = "aes_256_dslx",
    srcs = ["aes_256.x"],
    deps = [
        ":aes_256_common_dslx",
        ":aes_common_dslx",
    ],
)

xls_dslx_ir(
    name = "aes_256_encrypt",
    dslx_top = "encrypt",
    library = ":aes_256_dslx",
)

xls_ir_cc_library_macro(
    name = "aes_256_encrypt_cc",
    src = ":aes_256_encrypt.ir",
    namespaces = "xls,aes_256",
    top = "aes_256_encrypt",
)

xls_dslx_ir(
    name = "aes_256_decrypt",
    dslx_top = "decrypt",
    library = ":aes_256_dslx",
)

xls_ir_cc_library_macro(
    name = "aes_256_decrypt_cc",
    src = ":aes_256_decrypt.ir",
    namespaces = "xls,aes_256",
    top = "aes_256_decrypt",
)

cc_test(
    name = "aes_256_test",
    srcs = ["aes_256_test.cc"],
    deps = [
        ":aes_256_decrypt_cc",
        ":aes_256_encrypt_cc",
        ":aes_test_common",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common:init_xls",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value",
        "@boringssl//:crypto",
    ],
)

xls_dslx_library(
    name = "constants_dslx",
    srcs = ["constants.x"],
)

xls_dslx_library(
    name = "aes_128_ctr_dslx",
    srcs = ["aes_128_ctr.x"],
    deps = [
        ":aes_128_common_dslx",
        ":aes_128_dslx",
        ":aes_common_dslx",
    ],
)

xls_dslx_test(
    name = "aes_128_ctr_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":aes_128_ctr_dslx",
)

xls_dslx_ir(
    name = "aes_128_ctr",
    dslx_top = "aes_128_ctr",
    ir_conv_args = {
        "top_proc_initial_state": "'(0, (0, [0, 0, 0, 0], 0, 0), 0, 0)'",
    },
    library = ":aes_128_ctr_dslx",
)

cc_test(
    name = "aes_128_ctr_test",
    srcs = ["aes_128_ctr_test.cc"],
    data = [
        ":aes_128_ctr.ir",
    ],
    deps = [
        ":aes_test_common",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/jit:jit_channel_queue",
        "//xls/jit:proc_jit",
        "@boringssl//:crypto",
    ],
)

xls_dslx_library(
    name = "aes_128_ghash_dslx",
    srcs = ["aes_128_ghash.x"],
    deps = [
        ":aes_128_common_dslx",
        ":aes_128_dslx",
        ":aes_common_dslx",
    ],
)

xls_dslx_test(
    name = "aes_128_ghash_dslx_test",
    # TODO(rspringer): Remove when GitHub issue #673 is resolved.
    dslx_test_args = {
        "compare": "none",
    },
    library = ":aes_128_ghash_dslx",
)

xls_dslx_library(
    name = "aes_128_gcm_dslx",
    srcs = ["aes_128_gcm.x"],
    deps = [
        ":aes_128_common_dslx",
        ":aes_128_ctr_dslx",
        ":aes_128_dslx",
        ":aes_128_ghash_dslx",
        ":aes_common_dslx",
    ],
)

xls_dslx_test(
    name = "aes_128_gcm_dslx_test",
    # TODO(rspringer): Remove when GitHub issue #673 is resolved.
    dslx_test_args = {
        "compare": "none",
    },
    library = ":aes_128_gcm_dslx",
)

xls_dslx_ir(
    name = "aes_128_gcm",
    dslx_top = "aes_128_gcm",
    ir_conv_args = {
        "top_proc_initial_state": "'(0, (0, 0, 0, [0, 0, 0, 0], 0), 0, 0)'",
    },
    library = ":aes_128_gcm_dslx",
)

cc_test(
    name = "aes_128_gcm_test",
    srcs = ["aes_128_gcm_test.cc"],
    data = [
        ":aes_128_gcm.ir",
    ],
    deps = [
        ":aes_test_common",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/jit:jit_channel_queue",
        "//xls/jit:proc_jit",
        "//xls/jit:serial_proc_runtime",
        "@boringssl//:crypto",
    ],
)
