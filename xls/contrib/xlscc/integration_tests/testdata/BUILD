# Copyright 2025 The XLS Authors
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
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)
load(
    "//xls/contrib/xlscc/build_rules:xlscc_rules.bzl",
    "xls_cc_ir",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_cc_ir(
    name = "eval_proc_main_test_test_memory_cc_ir",
    src = "eval_proc_main_test.test_memory.cc",
    block_from_class = "Test",
    block_pb_out = "eval_proc_main_test.test_memory.block.pb",
    ir_file = "eval_proc_main_test.test_memory.ir",
    xlscc_args = {"top": ""},
)

xls_ir_opt_ir(
    name = "eval_proc_main_test_test_memory_opt_ir",
    src = ":eval_proc_main_test_test_memory_cc_ir",
    opt_ir_file = "eval_proc_main_test.test_memory.opt.ir",
    ram_rewrites = [
        "eval_proc_main_test.ram_rewrites.textproto",
    ],
)

xls_ir_verilog(
    name = "eval_proc_main_test_test_memory_block_ir",
    src = ":eval_proc_main_test_test_memory_opt_ir",
    block_ir_file = "eval_proc_main_test.test_memory.block.ir",
    codegen_args = {
        "pipeline_stages": "2",
        "reset": "rst",
        "ram_configurations": "mem:1RW:mem_req:mem_resp:mem_write_completion",
    },
    verilog_file = "eval_proc_main_test.test_memory.sv",
)

# Allow any file here to be used by tests directly.
exports_files(
    glob(
        include = [
            "*",
        ],
        exclude = [
            "*.x",
            "*.cc",
            "BUILD",
        ],
    ),
)
