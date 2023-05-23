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

# Build rules for XLS RLE Encoder implementation.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_ir",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

xls_dslx_library(
    name = "dbe_common_dslx",
    srcs = [
        "common.x",
        "common_test.x"
    ],
)

# ---------------------------------------------------------------------------
# Universal decoder
# ---------------------------------------------------------------------------

xls_dslx_library(
    name = "dbe_decoder_dslx",
    srcs = [
        "decoder.x"
    ],
    deps = [
        ":dbe_common_dslx"
    ],
)

xls_dslx_test(
    name = "dbe_decoder_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "dbe_decoder_dslx",
)

# ---------------------------------------------------------------------------
# LZ4 encoder
# ---------------------------------------------------------------------------

xls_dslx_library(
    name = "dbe_lz4_encoder_dslx",
    srcs = [
        "lz4_encoder.x"
    ],
    deps = [
        ":dbe_common_dslx",
        "//xls/examples:ram_dslx"
    ],
)

xls_dslx_test(
    name = "dbe_lz4_encoder_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "dbe_lz4_encoder_dslx",
)

xls_dslx_ir(
    name = "dbe_lz4_encoder_large_ir",
    dslx_top = "encoder_large",
    library = "dbe_lz4_encoder_dslx",
    ir_file = "dbe_lz4_encoder_large_ir.ir",
)

# xls_ir_opt_ir(
#     name = "dbe_lz4_encoder_large_opt_ir",
#     src = "dbe_lz4_encoder_large_ir.ir",
#     top = "tbd",
# )
