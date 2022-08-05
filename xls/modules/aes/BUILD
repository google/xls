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
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "aes_128_gcm_dslx",
    srcs = ["aes_128_gcm.x"],
)

xls_dslx_opt_ir(
    name = "aes_128_gcm",
    # This will be changed to "_gcm" once implemented.
    dslx_top = "aes_encrypt",
    library = ":aes_128_gcm_dslx",
)

xls_dslx_test(
    name = "aes_128_gcm_dslx_test",
    library = ":aes_128_gcm_dslx",
)

cc_xls_ir_jit_wrapper(
    name = "aes_128_gcm_jit_wrapper",
    src = ":aes_128_gcm.opt.ir",
    jit_wrapper_args = {
        "class_name": "aes_128_gcm",
    },
)

# This tickles much slowness in instruction selection during JIT compilation,
# so we'll mark it as a manual test for now. Same goes for ASAN.
cc_test(
    name = "aes_128_gcm_test",
    srcs = ["aes_128_gcm_test.cc"],
    tags = [
        "manual",
        "noasan",
    ],
    deps = [
        ":aes_128_gcm_jit_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/logging",
        "@boringssl//:crypto",
    ],
)
