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
    "PROC_WRAPPER_TYPE",
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_test",
)
load(
    "//xls/build_rules:xls_ir_macros.bzl",
    "xls_ir_cc_library_macro",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    features = [
        "layering_check",
        "parse_headers",
    ],
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
    name = "aes_ctr_dslx",
    srcs = ["aes_ctr.x"],
    deps = [
        ":aes_common_dslx",
        ":aes_dslx",
    ],
)

xls_dslx_test(
    name = "aes_ctr_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":aes_ctr_dslx",
)

xls_dslx_ir(
    name = "aes_ctr",
    dslx_top = "aes_ctr",
    library = ":aes_ctr_dslx",
)

cc_xls_ir_jit_wrapper(
    name = "aes_ctr_wrapper",
    src = ":aes_ctr.ir",
    jit_wrapper_args = {
        "class_name": "AesCtr",
        "namespace": "xls::aes::wrapped",
    },
    wrapper_type = PROC_WRAPPER_TYPE,
)

cc_test(
    name = "aes_ctr_test",
    srcs = ["aes_ctr_test.cc"],
    deps = [
        ":aes_ctr_wrapper",
        ":aes_test_common",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "//xls/interpreter:proc_runtime",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:events",
        "//xls/jit:jit_channel_queue",
        "@boringssl//:crypto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

xls_dslx_ir(
    name = "aes_decrypt",
    dslx_top = "decrypt",
    library = ":aes_dslx",
)

xls_ir_cc_library_macro(
    name = "aes_decrypt_cc",
    src = ":aes_decrypt.ir",
    namespaces = "xls,aes",
)

xls_dslx_library(
    name = "aes_dslx",
    srcs = ["aes.x"],
    deps = [
        ":aes_common_dslx",
        ":constants_dslx",
    ],
)

xls_dslx_ir(
    name = "aes_encrypt",
    dslx_top = "encrypt",
    library = ":aes_dslx",
)

xls_ir_cc_library_macro(
    name = "aes_encrypt_cc",
    src = ":aes_encrypt.ir",
    namespaces = "xls,aes",
)

xls_dslx_library(
    name = "aes_gcm_dslx",
    srcs = ["aes_gcm.x"],
    deps = [
        ":aes_common_dslx",
        ":aes_ctr_dslx",
        ":aes_dslx",
        ":ghash_dslx",
    ],
)

xls_dslx_test(
    name = "aes_gcm_dslx_test",
    # TODO(rspringer): Remove when GitHub issue #673 is resolved.
    dslx_test_args = {
        "compare": "none",
    },
    library = ":aes_gcm_dslx",
)

xls_dslx_ir(
    name = "aes_gcm",
    dslx_top = "aes_gcm",
    library = ":aes_gcm_dslx",
)

cc_xls_ir_jit_wrapper(
    name = "aes_gcm_wrapper",
    src = ":aes_gcm.ir",
    jit_wrapper_args = {
        "class_name": "AesGcm",
        "namespace": "xls::aes::wrapped",
    },
    wrapper_type = PROC_WRAPPER_TYPE,
)

cc_test(
    name = "aes_gcm_test",
    srcs = ["aes_gcm_test.cc"],
    tags = ["optonly"],
    deps = [
        ":aes_gcm_wrapper",
        ":aes_test_common",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "//xls/interpreter:proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:value",
        "@boringssl//:crypto",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xls_dslx_library(
    name = "constants_dslx",
    srcs = ["constants.x"],
)

xls_dslx_library(
    name = "ghash_dslx",
    srcs = ["ghash.x"],
    deps = [
        ":aes_common_dslx",
        ":aes_dslx",
    ],
)

xls_dslx_test(
    name = "ghash_dslx_test",
    # TODO(rspringer): Remove when GitHub issue #673 is resolved.
    dslx_test_args = {
        "compare": "none",
    },
    library = ":ghash_dslx",
)

xls_dslx_ir(
    name = "aes_ghash",
    dslx_top = "ghash",
    library = ":ghash_dslx",
)

cc_test(
    name = "aes_test",
    srcs = ["aes_test.cc"],
    # The XLS AOT compiler does not currently support cross-compilation.
    deps = [
        ":aes_decrypt_cc",
        ":aes_encrypt_cc",
        ":aes_test_common",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@boringssl//:crypto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "aes_test_common",
    srcs = ["aes_test_common.cc"],
    hdrs = ["aes_test_common.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:value",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xls_benchmark_ir(
    name = "aes_ctr_benchmark_ir",
    src = ":aes_ctr.ir",
    # TODO: Hits OOM issues during synthesis; should get this fixed.
    synthesize = False,
)

xls_benchmark_ir(
    name = "aes_ghash_benchmark_ir",
    src = ":aes_ghash.ir",
)

cc_binary(
    name = "sample_generator",
    srcs = ["sample_generator.cc"],
    deps = [
        ":aes_test_common",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "@boringssl//:crypto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
