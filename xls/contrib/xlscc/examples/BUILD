# Copyright 2021 The XLS Authors
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
    "proto_data",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)
load(
    "//xls/contrib/xlscc/build_rules:xlscc_build_defs.bzl",
    "xls_cc_ir",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_data(
    name = "mux3_block_pb",
    src = "mux3.textproto",
)

proto_data(
    name = "switch_block_pb",
    src = "switch.textproto",
)

proto_data(
    name = "mux1to2_block_pb",
    src = "mux1to2.textproto",
)

xls_cc_ir(
    name = "mux3_ir",
    src = "mux3.cc",
    block = ":mux3_block_pb",
    ir_file = "mux3.ir",
)

xls_cc_ir(
    name = "switch_ir",
    src = "switch.cc",
    block = ":switch_block_pb",
    ir_file = "switch.ir",
)

xls_cc_ir(
    name = "mux1to2_ir",
    src = "mux1to2.cc",
    block = ":mux1to2_block_pb",
    ir_file = "mux1to2.ir",
)

xls_ir_opt_ir(
    name = "mux3_opt_ir",
    src = ":mux3.ir",
    opt_ir_file = "mux3_opt_ir.opt.ir",
)

xls_ir_opt_ir(
    name = "switch_opt_ir",
    src = ":switch.ir",
    opt_ir_file = "switch_opt_ir.opt.ir",
)

xls_ir_opt_ir(
    name = "mux1to2_opt_ir",
    src = ":mux1to2.ir",
    opt_ir_file = "mux1to2_opt_ir.opt.ir",
)

xls_ir_verilog(
    name = "mux3_comb_v",
    src = ":mux3_opt_ir.opt.ir",
    codegen_args = {
        "generator": "combinational",
        "module_name": "mux3_comb",
        "top": "Mux3_proc",
    },
    verilog_file = "mux3_comb_v.sv",
)

xls_ir_verilog(
    name = "mux3_stages_5_v",
    src = ":mux3_opt_ir.opt.ir",
    codegen_args = {
        "module_name": "mux3_stages_5",
        "pipeline_stages": "5",
        "generator": "pipeline",
        "input_valid_signal": "input_valid",
        "output_valid_signal": "output_valid",
        "reset": "rst",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "reset_data_path": "false",
        "flop_inputs": "false",
        "flop_outputs": "false",
        "top": "Mux3_proc",
    },
    verilog_file = "mux3_stages_5_v.sv",
)

xls_ir_verilog(
    name = "mux1to2_stages_5_v",
    src = ":mux1to2_opt_ir.opt.ir",
    codegen_args = {
        "module_name": "mux1to2_stages_5",
        "pipeline_stages": "5",
        "generator": "pipeline",
        "input_valid_signal": "input_valid",
        "output_valid_signal": "output_valid",
        "reset": "rst",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "reset_data_path": "false",
        "flop_inputs": "false",
        "flop_outputs": "false",
        "top": "Mux1To2_proc",
    },
    verilog_file = "mux1to2_stages_5_v.sv",
)

cc_test(
    name = "mux3_opt_ir_test",
    srcs = ["mux3_opt_ir_test.cc"],
    data = [":mux3_opt_ir.opt.ir"],
    deps = [
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "mux3_v_test",
    srcs = ["mux3_v_test.cc"],
    data = [
        ":mux3_comb_v.sv",
        ":mux3_stages_5_v.sv",
    ],
    deps = [
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest",
    ],
)
