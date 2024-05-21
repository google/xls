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
    "xls_benchmark_verilog",
)
load(
    "//xls/contrib/xlscc/build_rules:xlscc_build_defs.bzl",
    "xls_cc_ir",
    "xls_cc_verilog",
)
load(
    "//xls/contrib/xlscc/build_rules:xlscc_rules_build_defs.bzl",
    xls_cc_ir_native_rule = "xls_cc_ir",
    xls_cc_verilog_native_rule = "xls_cc_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

proto_data(
    name = "packet_selector_block_pb",
    src = "packet_selector.textproto",
)

################################################################################
# Standalone tests for the native rule and macro versions of xls_cc_ir.
################################################################################
xls_cc_ir_native_rule(
    name = "packet_selector_ir",
    src = "packet_selector.cc",
    block = ":packet_selector_block_pb",
)

xls_cc_ir(
    name = "packet_selector_ir_macro",
    src = "packet_selector.cc",
    block = ":packet_selector_block_pb",
)

################################################################################
# Standalone tests for the native rule and macro versions of xls_cc_verilog.
################################################################################
xls_cc_verilog_native_rule(
    name = "packet_selector_verilog",
    src = "packet_selector.cc",
    block = ":packet_selector_block_pb",
    codegen_args = {
        "generator": "combinational",
        "module_name": "PacketSelector",
        "top": "PacketSelector_proc",
    },
    verilog_file = "packet_selector_verilog.sv",
)

xls_cc_verilog(
    name = "packet_selector_verilog_macro",
    src = "packet_selector.cc",
    block = ":packet_selector_block_pb",
    codegen_args = {
        "generator": "combinational",
        "module_name": "PacketSelector",
        "top": "PacketSelector_proc",
    },
    verilog_file = "packet_selector_verilog_macro.sv",
)

xls_benchmark_verilog(
    name = "packet_selector_verilog_benchmark",
    verilog_target = ":packet_selector_verilog",
)

################################################################################
# Tests for user defined output filenames.
################################################################################
xls_cc_ir_native_rule(
    name = "packet_selector_user_defined_name_ir",
    src = "packet_selector.cc",
    block = ":packet_selector_block_pb",
    ir_file = "xls_cc_ir.ir",
)

xls_cc_verilog_native_rule(
    name = "packet_selector_user_defined_name_verilog",
    src = "packet_selector.cc",
    block = ":packet_selector_block_pb",
    codegen_args = {
        "generator": "combinational",
        "module_name": "PacketSelector",
        "top": "PacketSelector_proc",
    },
    ir_file = "xls_cc_verilog.ir",
    module_sig_file = "xls_cc_verilog.sig.textproto",
    opt_ir_file = "xls_cc_verilog.opt.ir",
    verilog_file = "xls_cc_verilog.sv",
)
