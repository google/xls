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
    name = "rle_interface_dslx",
    srcs = [
        "rle_interface.x",
    ],
)

xls_dslx_library(
    name = "rle_enc_dslx",
    srcs = [
        "rle_enc.x",
    ],
    deps = [
        ":rle_interface_dslx"
    ],
)

xls_dslx_test(
    name = "rle_enc_dslx_test",
    dslx_test_args = {},
    compare_list = ["none", "interpreter", "jit"],
    library = "rle_enc_dslx",
)

xls_dslx_verilog (
    name = "rle_enc_verilog",
    dslx_top = "RLEEnc32",
    verilog_file = "rle_enc.v",
    library = "rle_enc_dslx",
    opt_ir_args = {
        "top": "__rle_enc__RLEEnc32__RLEEnc_0__2_32_next",
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

xls_dslx_library(
    name = "rle_mux_dslx",
    srcs = [
        "muxer.x",
    ],
    deps = [
        ":rle_interface_dslx"
    ],
)

xls_dslx_test(
    name = "rle_mux_dslx_test",
    dslx_test_args = {},
    compare_list = ["none", "interpreter", "jit"],
    library = "rle_mux_dslx",
)

xls_dslx_verilog (
    name = "rle_mux_verilog",
    dslx_top = "StreamMuxerSimple32",
    verilog_file = "rle_mux.v",
    library = "rle_mux_dslx",
    opt_ir_args = {
        "top": "__muxer__StreamMuxerSimple32__StreamMuxerSimple_0__32_next",
    },
    codegen_args = {
        "module_name": "rle_mux",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
)

xls_benchmark_ir(
    name = "rle_mux_ir_benchmark",
    src = "rle_mux_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    }
)

xls_dslx_library(
    name = "rle_dec_dslx",
    srcs = [
        "rle_dec.x",
    ],
    deps = [
        ":rle_interface_dslx"
    ],
)

xls_dslx_test(
    name = "rle_dec_dslx_test",
    dslx_test_args = {},
    compare_list = ["none", "interpreter", "jit"],
    library = "rle_dec_dslx",
)

xls_dslx_verilog (
    name = "rle_dec_verilog",
    dslx_top = "RLEDec32",
    verilog_file = "rle_dec.v",
    library = "rle_dec_dslx",
    opt_ir_args = {
        "top": "__rle_dec__RLEDec32__RLEDec_0__2_32_next",
    },
    codegen_args = {
        "module_name": "rle_dec",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
)

xls_benchmark_ir(
    name = "rle_dec_ir_benchmark",
    src = "rle_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    }
)

xls_benchmark_verilog(
    name = "rle_dec_verilog_benchmark",
    verilog_target = "rle_dec_verilog",
)
