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
    "xls_dslx_test",
    "xls_dslx_verilog",
    "xls_benchmark_ir",
    "xls_benchmark_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "rle_dslx",
    srcs = ["rle.x"],
)

xls_dslx_test(
    name = "rle_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_dslx",
)

xls_dslx_test(
    name = "rle_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_dslx",
)

xls_dslx_test(
    name = "rle_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_dslx",
)

xls_dslx_verilog (
    name = "rle_enc_verilog",
    dslx_top = "RLEEnc32",
    verilog_file = "rle_enc.v",
    library = "rle_dslx",
    opt_ir_args = {
        "top": "__rle__RLEEnc32__RLEEnc_0__2_32_next",
    },
    codegen_args = {
        "module_name": "rle_enc",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
)

xls_benchmark_ir(
    name = "rle_enc_ir_benchmark",
    src = "rle_enc_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    }
)

xls_benchmark_verilog(
    name = "rle_enc_verilog_benchmark",
    verilog_target = "rle_enc_verilog",
)
