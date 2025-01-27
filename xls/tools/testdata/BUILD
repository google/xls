# Copyright 2024 The XLS Authors
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
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "disable_port_main",
    testonly = True,
    srcs = ["disable_port_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:source_location",
        "//xls/ir:value_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "add_reset_trace_main",
    testonly = True,
    srcs = ["add_reset_trace_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:format_strings",
        "//xls/ir:ir_parser",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

xls_dslx_library(
    name = "eval_proc_main_test",
    srcs = ["eval_proc_main_test.x"],
)

xls_dslx_ir(
    name = "eval_proc_main_test_ir",
    dslx_top = "test_proc",
    library = ":eval_proc_main_test",
)

xls_ir_opt_ir(
    name = "eval_proc_main_test_opt_ir",
    src = ":eval_proc_main_test_ir",
    opt_ir_file = "eval_proc_main_test.opt.ir",
)

xls_ir_verilog(
    name = "eval_proc_main_test_sv",
    src = ":eval_proc_main_test_opt_ir",
    codegen_args = {
        "module_name": "test_block",
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
        "worst_case_throughput": "3",
    },
    verilog_file = "eval_proc_main_test.sv",
)

xls_dslx_library(
    name = "eval_proc_main_zero_size_test",
    srcs = ["eval_proc_main_zero_size_test.x"],
)

xls_dslx_ir(
    name = "eval_proc_main_zero_size_test_ir",
    dslx_top = "test_proc",
    library = ":eval_proc_main_zero_size_test",
)

xls_ir_opt_ir(
    name = "eval_proc_main_zero_size_test_opt_ir",
    src = ":eval_proc_main_zero_size_test_ir",
    opt_ir_file = "eval_proc_main_zero_size_test.opt.ir",
)

xls_ir_verilog(
    name = "eval_proc_main_zero_size_test_sv",
    src = ":eval_proc_main_zero_size_test_opt_ir",
    codegen_args = {
        "module_name": "test_block",
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
        "worst_case_throughput": "3",
    },
    verilog_file = "eval_proc_main_zero_size_test.sv",
)

xls_dslx_library(
    name = "eval_proc_main_conditional_test",
    srcs = ["eval_proc_main_conditional_test.x"],
)

xls_dslx_ir(
    name = "eval_proc_main_conditional_test_ir",
    dslx_top = "test_proc",
    library = ":eval_proc_main_conditional_test",
)

xls_ir_opt_ir(
    name = "eval_proc_main_conditional_test_opt_ir",
    src = ":eval_proc_main_conditional_test_ir",
    opt_ir_file = "eval_proc_main_conditional_test.opt.ir",
)

genrule(
    name = "add_trace_to_eval_proc_main_test_blk",
    testonly = True,
    srcs = [":eval_proc_main_test.block.ir"],
    outs = ["eval_proc_main_test_with_trace.block.ir"],
    cmd = "$(location :add_reset_trace_main) --reset_name rst $< > $@",
    tools = [":add_reset_trace_main"],
)

# Force this port to always say it's not valid
genrule(
    name = "break_eval_proc_main_test_blk",
    testonly = True,
    srcs = [":eval_proc_main_test.block.ir"],
    outs = ["eval_proc_main_test_broken.block.ir"],
    cmd = "$(location :disable_port_main) --port_name eval_proc_main_test__out_ch_2_valid $< > $@",
    tools = [":disable_port_main"],
)

filegroup(
    name = "static_vtxt_files",
    testonly = True,
    srcs = glob(["*.vtxt"]),
)

filegroup(
    name = "static_ir_files",
    testonly = True,
    srcs = glob(["*.ir"]),
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

xls_dslx_library(
    name = "eval_proc_main_test_memory_lib",
    srcs = ["eval_proc_main_test_memory.x"],
)

xls_dslx_ir(
    name = "eval_proc_main_test_memory",
    dslx_top = "test_proc",
    library = ":eval_proc_main_test_memory_lib",
)

xls_ir_opt_ir(
    name = "eval_proc_main_test_memory_opt_ir",
    src = ":eval_proc_main_test_memory",
    opt_ir_file = "eval_proc_main_test_memory.opt.ir",
    ram_rewrites = [
        "//xls/tools/testdata:eval_proc_main_test_memory.ram_rewrites.textproto",
    ],
)

xls_ir_verilog(
    name = "eval_proc_main_test_memory_block_ir",
    src = ":eval_proc_main_test_memory_opt_ir",
    block_ir_file = "eval_proc_main_test_memory.block.ir",
    codegen_args = {
        "pipeline_stages": "2",
        "reset": "rst",
        "ram_configurations": "mem:1RW:mem_req:mem_resp:mem_write_completion",
    },
    verilog_file = "eval_proc_main_test_memory.sv",
)
