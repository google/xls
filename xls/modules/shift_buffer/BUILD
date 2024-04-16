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

# This is a pipeline-aware data buffer. The implementation aligns incoming data
# before it enters the buffer and mutates its state.

load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "benchmark_synth", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
    "xls_benchmark_verilog",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_dslx_verilog",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "fixme_dslx",
    srcs = ["fixme.x"],
)

xls_dslx_test(
    name = "fixme_dslx_test",
    library = ":fixme_dslx",
)

xls_dslx_library(
    name = "math_dslx",
    srcs = ["math.x"],
    deps = [":fixme_dslx"]
)

xls_dslx_test(
    name = "math_dslx_test",
    library = ":math_dslx",
)

## ShiftBuffer

xls_dslx_library(
    name = "shift_buffer_dslx",
    srcs = ["shift_buffer.x"],
    deps = [
        ":math_dslx",
        ":fixme_dslx",
    ],
)

xls_dslx_test(
    name = "shift_buffer_dslx_test",
    library = ":shift_buffer_dslx",
)

xls_dslx_library(
    name = "shift_buffer_inst_dslx",
    srcs = ["shift_buffer_inst.x"],
    deps = [
        ":shift_buffer_dslx",
    ],
)

# ShiftBufferAligner

xls_dslx_verilog(
    name = "shift_buffer_aligner_verilog",
    codegen_args = {
        "module_name": "ShiftBufferAligner",
        "generator": "pipeline",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "worst_case_throughput": "1",
        "use_system_verilog": "false",
    },
    dslx_top = "ShiftBufferAlignerInst",
    library = ":shift_buffer_inst_dslx",
    opt_ir_args = {
        "top": "__xls_modules_shift_buffer_shift_buffer__ShiftBufferAlignerInst__ShiftBufferAligner_0__64_128_7_next",
    },
    verilog_file = "shift_buffer_aligner.v",
)

xls_benchmark_ir(
    name = "shift_buffer_aligner_opt_ir_benchmark",
    src = ":shift_buffer_aligner_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "shift_buffer_aligner_verilog_lib",
    srcs = [
        ":shift_buffer_aligner.v",
    ],
)

synthesize_rtl(
    name = "shift_buffer_aligner_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "ShiftBufferAligner",
    deps = [
        ":shift_buffer_aligner_verilog_lib",
    ],
)

benchmark_synth(
    name = "shift_buffer_aligner_benchmark_synth",
    synth_target = ":shift_buffer_aligner_synth_asap7",
)

place_and_route(
    name = "shift_buffer_aligner_place_and_route",
    clock_period = "650",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":shift_buffer_aligner_synth_asap7",
    target_die_utilization_percentage = "5",
)

xls_benchmark_verilog(
    name = "shift_buffer_aligner_verilog_benchmark",
    verilog_target = "shift_buffer_aligner_verilog",
)

## ShiftBufferStorage

xls_dslx_verilog(
    name = "shift_buffer_storage_verilog",
    codegen_args = {
        "module_name": "ShiftBufferStorage",
        "generator": "pipeline",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "worst_case_throughput": "1",
        "use_system_verilog": "false",
    },
    dslx_top = "ShiftBufferStorageInst",
    library = ":shift_buffer_inst_dslx",
    opt_ir_args = {
        "top": "__xls_modules_shift_buffer_shift_buffer__ShiftBufferStorageInst__ShiftBufferStorage_0__64_7_next",
    },
    verilog_file = "shift_buffer_storage.v",
)

xls_benchmark_ir(
    name = "shift_buffer_storage_opt_ir_benchmark",
    src = ":shift_buffer_storage_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "shift_buffer_storage_verilog_lib",
    srcs = [
        ":shift_buffer_storage.v",
    ],
)

synthesize_rtl(
    name = "shift_buffer_storage_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "ShiftBufferStorage",
    deps = [
        ":shift_buffer_storage_verilog_lib",
    ],
)

benchmark_synth(
    name = "shift_buffer_storage_benchmark_synth",
    synth_target = ":shift_buffer_storage_synth_asap7",
)

place_and_route(
    name = "shift_buffer_storage_place_and_route",
    clock_period = "650",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":shift_buffer_storage_synth_asap7",
    target_die_utilization_percentage = "5",
)

xls_benchmark_verilog(
    name = "shift_buffer_storage_verilog_benchmark",
    verilog_target = "shift_buffer_storage_verilog",
)

## ShiftBuffer

xls_dslx_verilog(
    name = "shift_buffer_verilog",
    codegen_args = {
        "module_name": "ShiftBuffer",
        "generator": "pipeline",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "worst_case_throughput": "1",
        "use_system_verilog": "false",
    },
    dslx_top = "ShiftBufferInst",
    library = ":shift_buffer_inst_dslx",
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__xls_modules_shift_buffer_shift_buffer__ShiftBufferInst__ShiftBuffer_0__ShiftBufferStorage_0__64_7_next",
    },
    verilog_file = "shift_buffer.v",
)

xls_benchmark_ir(
    name = "shift_buffer_opt_ir_benchmark",
    src = ":shift_buffer_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "shift_buffer_verilog_lib",
    srcs = [
        ":shift_buffer.v",
    ],
)

synthesize_rtl(
    name = "shift_buffer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "ShiftBuffer",
    deps = [
        ":shift_buffer_verilog_lib",
    ],
)

benchmark_synth(
    name = "shift_buffer_benchmark_synth",
    synth_target = ":shift_buffer_synth_asap7",
)

place_and_route(
    name = "shift_buffer_place_and_route",
    clock_period = "650",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":shift_buffer_synth_asap7",
    target_die_utilization_percentage = "5",
)

xls_benchmark_verilog(
    name = "shift_buffer_verilog_benchmark",
    verilog_target = "shift_buffer_verilog",
)
