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

# Build rules for ZSTD Sequence Executor

load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "benchmark_synth", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
    "xls_benchmark_verilog",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_dslx_verilog",
)
load("//xls/common:openroad_warnings.bzl", "ASAP7_SUPPPRESSED_WARNINGS")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

CLOCK_PERIOD_PS = "750"

COMMON_CODEGEN_ARGS = {
    "delay_model": "asap7",
    "reset": "rst",
    "worst_case_throughput": "1",
    "use_system_verilog": "false",
    "clock_period_ps": CLOCK_PERIOD_PS,
    "clock_margin_percent": "20",
    "multi_proc": "true",
    "materialize_internal_fifos": "true",
}

xls_dslx_library(
    name = "history_buffer_dslx",
    srcs = ["history_buffer.x"],
    deps = [
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "history_buffer_dslx_test",
    library = ":history_buffer_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "history_copy_executor_dslx",
    srcs = [
        "history_copy_executor.x",
    ],
    deps = [
        "//xls/modules/zstd:common_dslx",
        "//xls/modules/zstd/memory:mem_writer_dslx",
        ":history_buffer_dslx",
    ],
)

xls_dslx_test(
    name = "history_copy_executor_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":history_copy_executor_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "join_output_dslx",
    srcs = [
        "join_output.x",
    ],
    deps = [
        "//xls/modules/zstd:common_dslx",
        "//xls/modules/zstd/memory:mem_writer_dslx",
    ],
)

xls_dslx_test(
    name = "join_output_dslx_test",
    library = ":join_output_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "sequence_executor_ctrl_dslx",
    srcs = [
        "sequence_executor_ctrl.x",
    ],
    deps = [
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd:common_dslx",
        "//xls/modules/zstd:parallel_rams_dslx",
        "//xls/modules/zstd:ram_printer_dslx",
        "//xls/modules/zstd/memory:mem_writer_dslx",
        ":join_output_dslx",
        ":history_copy_executor_dslx",
        ":literals_executor_dslx",
    ],
)

xls_dslx_test(
    name = "sequence_executor_ctrl_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":sequence_executor_ctrl_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "literals_prefetch_dslx",
    srcs = [
        "literals_prefetch.x",
    ],
    deps = [
        "//xls/modules/zstd:common_dslx",
    ],
)

xls_dslx_test(
    name = "literals_prefetch_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":literals_prefetch_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "literals_executor_dslx",
    srcs = [
        "literals_executor.x",
    ],
    deps = [
        "//xls/modules/zstd:common_dslx",
        "//xls/modules/zstd/memory:mem_writer_dslx",
        ":literals_prefetch_dslx",
        ":history_buffer_dslx",
    ],
)

xls_dslx_test(
    name = "literals_executor_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":literals_executor_dslx",
    tags = ["manual"],
)

LITERALS_EXECUTOR_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "literals_executor",
    "pipeline_stages": "4",
    "worst_case_throughput": "2",
}

xls_dslx_verilog(
    name = "literals_executor_verilog",
    codegen_args = LITERALS_EXECUTOR_CODEGEN_ARGS,
    dslx_top = "LiteralsExecutorInst",
    library = ":literals_executor_dslx",
    tags = ["manual"],
    verilog_file = "literals_executor.v",
)

xls_benchmark_ir(
    name = "literals_executor_opt_ir_benchmark",
    src = ":literals_executor_verilog.opt.ir",
    benchmark_ir_args = LITERALS_EXECUTOR_CODEGEN_ARGS,
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "literals_executor_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "literals_executor_verilog",
)

verilog_library(
    name = "literals_executor_lib",
    srcs = [
        ":literals_executor.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "literals_executor_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "literals_executor",
    deps = [
        ":literals_executor_lib",
    ],
)

benchmark_synth(
    name = "literals_executor_benchmark_synth",
    synth_target = ":literals_executor_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "literals_executor_place_and_route_skip",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":literals_executor_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "sequence_executor_dslx",
    srcs = [
        "sequence_executor.x",
    ],
    deps = [
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd:common_dslx",
        "//xls/modules/zstd:parallel_rams_dslx",
        "//xls/modules/zstd:ram_printer_dslx",
        "//xls/modules/zstd/memory:mem_writer_dslx",
        ":history_buffer_dslx",
        ":history_copy_executor_dslx",
        ":literals_executor_dslx",
        ":join_output_dslx",
        ":sequence_executor_ctrl_dslx",
    ],
)

xls_dslx_test(
    name = "sequence_executor_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":sequence_executor_dslx",
    tags = ["manual"],
)

SEQUENCE_EXECUTOR_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "sequence_executor",
    "clock_period_ps": "0",
    "generator": "pipeline",
    "delay_model": "asap7",
    "worst_case_throughput": "2",
    "pipeline_stages": "3",
    "reset": "rst",
    "reset_data_path": "true",
    "reset_active_low": "false",
    "reset_asynchronous": "true",
    "flop_inputs": "false",
    "flop_single_value_channels": "false",
    "flop_outputs": "false",
}

xls_dslx_verilog(
    name = "sequence_executor_verilog",
    codegen_args = SEQUENCE_EXECUTOR_CODEGEN_ARGS,
    dslx_top = "SequenceExecutorZstd",
    library = ":sequence_executor_dslx",
    tags = ["manual"],
    verilog_file = "sequence_executor.v",
)

xls_benchmark_ir(
    name = "sequence_executor_opt_ir_benchmark",
    src = ":sequence_executor_verilog.opt.ir",
    benchmark_ir_args = SEQUENCE_EXECUTOR_CODEGEN_ARGS,
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "sequence_executor_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "sequence_executor_verilog",
)
