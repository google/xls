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
    "xls_benchmark_ir",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "rle_common_dslx",
    srcs = [
        "rle_common.x",
    ],
)

xls_dslx_library(
    name = "rle_enc_dslx",
    srcs = [
        "rle_enc.x",
    ],
    deps = [
        ":rle_common_dslx",
    ],
)

xls_dslx_test(
    name = "rle_enc_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_enc_dslx",
)

xls_dslx_test(
    name = "rle_enc_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_enc_dslx",
)

xls_dslx_test(
    name = "rle_enc_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_enc_dslx",
)

xls_dslx_ir(
    name = "rle_enc_ir",
    dslx_top = "RunLengthEncoder32",
    ir_file = "rle_enc.ir",
    library = "rle_enc_dslx",
)

xls_ir_opt_ir(
    name = "rle_enc_opt_ir",
    src = "rle_enc.ir",
    top = "__rle_enc__RunLengthEncoder32__RunLengthEncoder_0__2_32_next",
)

xls_ir_verilog(
    name = "rle_enc_verilog",
    src = ":rle_enc_opt_ir.opt.ir",
    codegen_args = {
        "module_name": "rle_enc",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    verilog_file = "rle_enc.v",
)

xls_benchmark_ir(
    name = "rle_enc_ir_benchmark",
    src = ":rle_enc_opt_ir.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    },
)

xls_dslx_library(
    name = "rle_enc_adv_reduce_stage_dslx",
    srcs = [
        "rle_enc_adv_reduce_stage.x",
    ],
    deps = [
        ":rle_common_dslx"
    ],
)

xls_dslx_test(
    name = "rle_enc_adv_reduce_stage_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_enc_adv_reduce_stage_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_reduce_stage_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_enc_adv_reduce_stage_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_reduce_stage_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_enc_adv_reduce_stage_dslx",
)

xls_dslx_library(
    name = "rle_enc_adv_realign_stage_dslx",
    srcs = [
        "rle_enc_adv_realign_stage.x",
    ],
    deps = [
        ":rle_common_dslx"
    ],
)

xls_dslx_test(
    name = "rle_enc_adv_realign_stage_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_enc_adv_realign_stage_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_realign_stage_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_enc_adv_realign_stage_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_realign_stage_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_enc_adv_realign_stage_dslx",
)

xls_dslx_library(
    name = "rle_enc_adv_core_dslx",
    srcs = [
        "rle_enc_adv_core.x",
    ],
    deps = [
        ":rle_common_dslx"
    ],
)

xls_dslx_test(
    name = "rle_enc_adv_core_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_enc_adv_core_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_core_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_enc_adv_core_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_core_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_enc_adv_core_dslx",
)

xls_dslx_library(
    name = "rle_enc_adv_adjust_width_stage_dslx",
    srcs = [
        "rle_enc_adv_adjust_width_stage.x",
    ],
    deps = [
        ":rle_common_dslx"
    ],
)

xls_dslx_test(
    name = "rle_enc_adv_adjust_width_stage_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_enc_adv_adjust_width_stage_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_adjust_width_stage_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_enc_adv_adjust_width_stage_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_adjust_width_stage_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_enc_adv_adjust_width_stage_dslx",
)

xls_dslx_library(
    name = "rle_enc_adv_dslx",
    srcs = [
        "rle_enc_adv.x",
    ],
    deps = [
        ":rle_common_dslx",
        ":rle_enc_adv_reduce_stage_dslx",
        ":rle_enc_adv_realign_stage_dslx",
        ":rle_enc_adv_core_dslx",
        ":rle_enc_adv_adjust_width_stage_dslx",
    ],
)

xls_dslx_test(
    name = "rle_enc_adv_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_enc_adv_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_enc_adv_dslx",
)

xls_dslx_test(
    name = "rle_enc_adv_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_enc_adv_dslx",
)

xls_dslx_ir(
    name = "rle_enc_adv_ir",
    dslx_top = "RunLengthEncoder8_8_4_2",
    library = "rle_enc_adv_dslx",
    ir_file = "rle_enc_adv.ir",
)

xls_ir_opt_ir(
    name = "rle_enc_adv_opt_ir",
    src = "rle_enc_adv.ir",
    top = "__xls_modules_rle_rle_enc_adv_core__RunLengthEncoder8_8_4_2__RunLengthEncoderAdvanced__RunLengthEncoderAdvancedCoreStage_0__8_4_8_next",
)

xls_ir_verilog(
    name = "rle_enc_adv_verilog",
    src = ":rle_enc_adv_opt_ir.opt.ir",
    verilog_file = "rle_enc_adv.v",
    codegen_args = {
        "module_name": "rle_enc_adv",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
)

xls_benchmark_ir(
    name = "rle_enc_adv_ir_benchmark",
    src = ":rle_enc_adv_opt_ir.opt.ir",
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
        ":rle_common_dslx",
    ],
)

xls_dslx_test(
    name = "rle_dec_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_dec_dslx",
)

xls_dslx_test(
    name = "rle_dec_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_dec_dslx",
)

xls_dslx_test(
    name = "rle_dec_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_dec_dslx",
)

xls_dslx_ir(
    name = "rle_dec_ir",
    dslx_top = "RunLengthDecoder32",
    ir_file = "rle_dec.ir",
    library = "rle_dec_dslx",
)

xls_ir_opt_ir(
    name = "rle_dec_opt_ir",
    src = "rle_dec.ir",
    top = "__rle_dec__RunLengthDecoder32__RunLengthDecoder_0__2_32_next",
)

xls_ir_verilog(
    name = "rle_dec_verilog",
    src = ":rle_dec_opt_ir.opt.ir",
    codegen_args = {
        "module_name": "rle_dec",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    verilog_file = "rle_dec.v",
)

xls_benchmark_ir(
    name = "rle_dec_ir_benchmark",
    src = ":rle_dec_opt_ir.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    },
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
