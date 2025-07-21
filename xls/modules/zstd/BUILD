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

# Build rules for XLS ZSTD codec implementation.

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
    name = "math_dslx",
    srcs = [
        "math.x",
    ],
)

xls_dslx_test(
    name = "math_dslx_test",
    library = ":math_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "buffer_dslx",
    srcs = [
        "buffer.x",
    ],
)

xls_dslx_test(
    name = "buffer_dslx_test",
    library = ":buffer_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "window_buffer_dslx",
    srcs = [
        "window_buffer.x",
    ],
    deps = [
        ":buffer_dslx",
    ],
)

xls_dslx_test(
    name = "window_buffer_dslx_test",
    library = ":window_buffer_dslx",
    tags = ["manual"],
)

WINDOW_BUFFER_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "WindowBuffer64",
    "worst_case_throughput": "2",
}

xls_dslx_verilog(
    name = "window_buffer_verilog",
    codegen_args = WINDOW_BUFFER_CODEGEN_ARGS,
    dslx_top = "WindowBuffer64",
    library = ":window_buffer_dslx",
    tags = ["manual"],
    verilog_file = "window_buffer.v",
)

xls_benchmark_ir(
    name = "window_buffer_opt_ir_benchmark",
    src = ":window_buffer_verilog.opt.ir",
    benchmark_ir_args = WINDOW_BUFFER_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "window_buffer_verilog_lib",
    srcs = [
        ":window_buffer.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "window_buffer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "WindowBuffer64",
    deps = [
        ":window_buffer_verilog_lib",
    ],
)

benchmark_synth(
    name = "window_buffer_benchmark_synth",
    synth_target = ":window_buffer_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "window_buffer_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":window_buffer_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "shift_buffer_dslx",
    srcs = ["shift_buffer.x"],
    deps = [
        ":math_dslx",
    ],
)

xls_dslx_test(
    name = "shift_buffer_dslx_test",
    library = ":shift_buffer_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "shift_buffer_aligner_verilog",
    codegen_args = COMMON_CODEGEN_ARGS | {"module_name": "ShiftBufferAligner"},
    dslx_top = "ShiftBufferAlignerInst",
    library = ":shift_buffer_dslx",
    tags = ["manual"],
    verilog_file = "shift_buffer_aligner.v",
)

xls_benchmark_ir(
    name = "shift_buffer_aligner_opt_ir_benchmark",
    src = ":shift_buffer_aligner_verilog.opt.ir",
    benchmark_ir_args = COMMON_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "shift_buffer_aligner_verilog_lib",
    srcs = [
        ":shift_buffer_aligner.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "shift_buffer_aligner_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "ShiftBufferAligner",
    deps = [
        ":shift_buffer_aligner_verilog_lib",
    ],
)

benchmark_synth(
    name = "shift_buffer_aligner_benchmark_synth",
    synth_target = ":shift_buffer_aligner_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "shift_buffer_aligner_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":shift_buffer_aligner_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "5",
)

xls_benchmark_verilog(
    name = "shift_buffer_aligner_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "shift_buffer_aligner_verilog",
)

SHIFT_BUFFER_STORAGE_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "clock_period_ps": "0",
    "pipeline_stages": "2",
    "module_name": "ShiftBufferStorage",
}

xls_dslx_verilog(
    name = "shift_buffer_storage_verilog",
    codegen_args = SHIFT_BUFFER_STORAGE_CODEGEN_ARGS,
    dslx_top = "ShiftBufferStorageInst",
    library = ":shift_buffer_dslx",
    tags = ["manual"],
    verilog_file = "shift_buffer_storage.v",
)

xls_benchmark_ir(
    name = "shift_buffer_storage_opt_ir_benchmark",
    src = ":shift_buffer_storage_verilog.opt.ir",
    benchmark_ir_args = SHIFT_BUFFER_STORAGE_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "shift_buffer_storage_verilog_lib",
    srcs = [
        ":shift_buffer_storage.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "shift_buffer_storage_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "ShiftBufferStorage",
    deps = [
        ":shift_buffer_storage_verilog_lib",
    ],
)

benchmark_synth(
    name = "shift_buffer_storage_benchmark_synth",
    synth_target = ":shift_buffer_storage_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "shift_buffer_storage_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":shift_buffer_storage_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "5",
)

xls_benchmark_verilog(
    name = "shift_buffer_storage_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "shift_buffer_storage_verilog",
)

# FIXME: Improve the proc to achieve CLOCK_PERIOD_PS
SHIFT_BUFFER_CLOCK_PERIOD_PS = "1000"

SHIFT_BUFFER_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "clock_period_ps": SHIFT_BUFFER_CLOCK_PERIOD_PS,
    "module_name": "ShiftBuffer",
}

xls_dslx_verilog(
    name = "shift_buffer_verilog",
    codegen_args = SHIFT_BUFFER_CODEGEN_ARGS,
    dslx_top = "ShiftBufferInst",
    library = ":shift_buffer_dslx",
    tags = ["manual"],
    verilog_file = "shift_buffer.v",
)

xls_benchmark_ir(
    name = "shift_buffer_opt_ir_benchmark",
    src = ":shift_buffer_verilog.opt.ir",
    benchmark_ir_args = SHIFT_BUFFER_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "shift_buffer_verilog_lib",
    srcs = [
        ":shift_buffer.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "shift_buffer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "ShiftBuffer",
    deps = [
        ":shift_buffer_verilog_lib",
    ],
)

benchmark_synth(
    name = "shift_buffer_benchmark_synth",
    synth_target = ":shift_buffer_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "shift_buffer_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":shift_buffer_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "5",
)

xls_benchmark_verilog(
    name = "shift_buffer_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "shift_buffer_verilog",
)

xls_dslx_library(
    name = "common_dslx",
    srcs = [
        "common.x",
    ],
    deps = [
        ":shift_buffer_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_library(
    name = "frame_header_dec_dslx",
    srcs = [
        "frame_header_dec.x",
    ],
    deps = [
        "//xls/modules/zstd/memory:axi_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "frame_header_dec_dslx_test",
    library = ":frame_header_dec_dslx",
    tags = ["manual"],
)

FRAME_HEADER_DEC_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "FrameHeaderDecoder",
    "clock_period_ps": "0",
    "pipeline_stages": "6",
}

xls_dslx_verilog(
    name = "frame_header_dec_verilog",
    codegen_args = FRAME_HEADER_DEC_CODEGEN_ARGS,
    dslx_top = "FrameHeaderDecoderInst",
    library = ":frame_header_dec_dslx",
    tags = ["manual"],
    verilog_file = "frame_header_dec.v",
)

xls_benchmark_ir(
    name = "frame_header_dec_opt_ir_benchmark",
    src = ":frame_header_dec_verilog.opt.ir",
    benchmark_ir_args = FRAME_HEADER_DEC_CODEGEN_ARGS | {
        "top": "__frame_header_dec__FrameHeaderDecoderInst__FrameHeaderDecoder_0__16_32_30_5_next",
        "pipeline_stages": "10",
    },
    tags = ["manual"],
)

verilog_library(
    name = "frame_header_dec_verilog_lib",
    srcs = [
        ":frame_header_dec.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "frame_header_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "FrameHeaderDecoder",
    deps = [
        ":frame_header_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "frame_header_dec_benchmark_synth",
    synth_target = ":frame_header_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "frame_header_dec_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":frame_header_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "block_header_dslx",
    srcs = [
        "block_header.x",
    ],
    deps = [
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "block_header_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":block_header_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "block_header_dec_dslx",
    srcs = [
        "block_header_dec.x",
    ],
    deps = [
        ":block_header_dslx",
        ":common_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "block_header_dec_dslx_test",
    library = ":block_header_dec_dslx",
    tags = ["manual"],
)

BLOCK_HEADER_DEC_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "BlockHeaderDec",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "block_header_dec_verilog",
    codegen_args = BLOCK_HEADER_DEC_CODEGEN_ARGS,
    dslx_top = "BlockHeaderDecoderInst",
    library = ":block_header_dec_dslx",
    tags = ["manual"],
    verilog_file = "block_header_dec.v",
)

xls_benchmark_ir(
    name = "block_header_dec_opt_ir_benchmark",
    src = ":block_header_dec_verilog.opt.ir",
    benchmark_ir_args = BLOCK_HEADER_DEC_CODEGEN_ARGS | {
        "pipeline_stages": "10",
        "top": "__block_header_dec__BlockHeaderDecoderInst__BlockHeaderDecoder_0__16_64_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "block_header_dec_verilog_lib",
    srcs = [
        ":block_header_dec.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "block_header_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "BlockHeaderDec",
    deps = [
        ":block_header_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "block_header_dec_benchmark_synth",
    synth_target = ":block_header_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "block_header_dec_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":block_header_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "raw_block_dec_dslx",
    srcs = [
        "raw_block_dec.x",
    ],
    deps = [
        ":common_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "raw_block_dec_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":raw_block_dec_dslx",
    tags = ["manual"],
)

RAW_BLOCK_DEC_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "RawBlockDecoder",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "raw_block_dec_verilog",
    codegen_args = RAW_BLOCK_DEC_CODEGEN_ARGS,
    dslx_top = "RawBlockDecoderInst",
    library = ":raw_block_dec_dslx",
    tags = ["manual"],
    verilog_file = "raw_block_dec.v",
)

xls_benchmark_ir(
    name = "raw_block_dec_opt_ir_benchmark",
    src = ":raw_block_dec_verilog.opt.ir",
    benchmark_ir_args = RAW_BLOCK_DEC_CODEGEN_ARGS | {
        "pipeline_stages": "10",
        "top": "__raw_block_dec__RawBlockDecoderInst__RawBlockDecoder_0__32_32_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "raw_block_dec_verilog_lib",
    srcs = [
        ":raw_block_dec.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "raw_block_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RawBlockDecoder",
    deps = [
        ":raw_block_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "raw_block_dec_benchmark_synth",
    synth_target = ":raw_block_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "raw_block_dec_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":raw_block_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "rle_block_dec_dslx",
    srcs = [
        "rle_block_dec.x",
    ],
    deps = [
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "rle_block_dec_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":rle_block_dec_dslx",
    tags = ["manual"],
)

RLE_BLOCK_DEC_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "RleBlockDecoder",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "rle_block_dec_verilog",
    codegen_args = RLE_BLOCK_DEC_CODEGEN_ARGS,
    dslx_top = "RleBlockDecoderInst",
    library = ":rle_block_dec_dslx",
    tags = ["manual"],
    verilog_file = "rle_block_dec.v",
)

xls_benchmark_ir(
    name = "rle_block_dec_opt_ir_benchmark",
    src = ":rle_block_dec_verilog.opt.ir",
    benchmark_ir_args = RLE_BLOCK_DEC_CODEGEN_ARGS | {
        "pipeline_stages": "10",
        "top": "__rle_block_dec__RleBlockDecoderInst__RleBlockDecoder_0__64_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "rle_block_dec_verilog_lib",
    srcs = [
        ":rle_block_dec.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "rle_block_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RleBlockDecoder",
    deps = [
        ":rle_block_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "rle_block_dec_benchmark_synth",
    synth_target = ":rle_block_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "rle_block_dec_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":rle_block_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "dec_mux_dslx",
    srcs = [
        "dec_mux.x",
    ],
    deps = [
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "dec_mux_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":dec_mux_dslx",
    tags = ["manual"],
)

DEC_MUX_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "DecoderMux",
    "clock_period_ps": "0",
    "pipeline_stages": "3",
}

xls_dslx_verilog(
    name = "dec_mux_verilog",
    codegen_args = DEC_MUX_CODEGEN_ARGS,
    dslx_top = "DecoderMux",
    library = ":dec_mux_dslx",
    tags = ["manual"],
    verilog_file = "dec_mux.v",
)

xls_benchmark_ir(
    name = "dec_mux_opt_ir_benchmark",
    src = ":dec_mux_verilog.opt.ir",
    benchmark_ir_args = DEC_MUX_CODEGEN_ARGS | {
        "pipeline_stages": "10",
    },
    tags = ["manual"],
)

verilog_library(
    name = "dec_mux_verilog_lib",
    srcs = [
        ":dec_mux.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "dec_mux_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "DecoderMux",
    deps = [
        ":dec_mux_verilog_lib",
    ],
)

benchmark_synth(
    name = "dec_mux_benchmark_synth",
    synth_target = ":dec_mux_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "dec_mux_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":dec_mux_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "ram_printer_dslx",
    srcs = ["ram_printer.x"],
    deps = [
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "ram_printer_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":ram_printer_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "parallel_rams_dslx",
    srcs = ["parallel_rams.x"],
    deps = [
        ":common_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "parallel_rams_dslx_test",
    library = ":parallel_rams_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "sequence_executor_dslx",
    srcs = [
        "sequence_executor.x",
    ],
    deps = [
        ":common_dslx",
        ":parallel_rams_dslx",
        ":ram_printer_dslx",
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd/memory:mem_writer_dslx",
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
    "ram_configurations": ",".join([
        "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
            latency = 5,
            ram_name = "ram{}".format(num),
            rd_req = "sequence_executor__rd_req_m{}_s".format(num),
            rd_resp = "sequence_executor__rd_resp_m{}_r".format(num),
            wr_req = "sequence_executor__wr_req_m{}_s".format(num),
            wr_resp = "sequence_executor__wr_resp_m{}_r".format(num),
        )
        for num in range(7)
    ]),
    "pipeline_stages": "6",
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
    benchmark_ir_args = SEQUENCE_EXECUTOR_CODEGEN_ARGS | {
        "pipeline_stages": "10",
    },
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "sequence_executor_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "sequence_executor_verilog",
)

verilog_library(
    name = "sequence_executor_lib",
    srcs = [
        ":sequence_executor.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "sequence_executor_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "sequence_executor",
    deps = [
        ":sequence_executor_lib",
    ],
)

benchmark_synth(
    name = "sequence_executor_benchmark_synth",
    synth_target = ":sequence_executor_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "sequence_executor_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    die_height_microns = 120,
    die_width_microns = 120,
    min_pin_distance = "0.4",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":sequence_executor_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "axi_csr_accessor_dslx",
    srcs = [
        "axi_csr_accessor.x",
    ],
    deps = [
        ":csr_config_dslx",
        "//xls/modules/zstd/memory:axi_dslx",
    ],
)

xls_dslx_test(
    name = "axi_csr_accessor_dslx_test",
    library = ":axi_csr_accessor_dslx",
    tags = ["manual"],
)

AXI_CSR_ACCESSOR_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "AxiCsrAccessor",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "axi_csr_accessor_verilog",
    codegen_args = AXI_CSR_ACCESSOR_CODEGEN_ARGS,
    dslx_top = "AxiCsrAccessorInst",
    library = ":axi_csr_accessor_dslx",
    tags = ["manual"],
    verilog_file = "axi_csr_accessor.v",
)

xls_benchmark_ir(
    name = "axi_csr_accessor_opt_ir_benchmark",
    src = ":axi_csr_accessor_verilog.opt.ir",
    benchmark_ir_args = AXI_CSR_ACCESSOR_CODEGEN_ARGS | {
        "pipeline_stages": "10",
        "top": "__axi_csr_accessor__AxiCsrAccessorInst__AxiCsrAccessor_0__16_32_4_4_2_4_16_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "axi_csr_accessor_verilog_lib",
    srcs = [
        ":axi_csr_accessor.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "axi_csr_accessor_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "AxiCsrAccessor",
    deps = [
        ":axi_csr_accessor_verilog_lib",
    ],
)

benchmark_synth(
    name = "axi_csr_accessor_benchmark_synth",
    synth_target = ":axi_csr_accessor_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "axi_csr_accessor_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":axi_csr_accessor_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "csr_config_dslx",
    srcs = [
        "csr_config.x",
    ],
    deps = [
        "//xls/modules/zstd/memory:axi_dslx",
    ],
)

xls_dslx_test(
    name = "csr_config_dslx_test",
    library = ":csr_config_dslx",
    tags = ["manual"],
)

CSR_CONFIG_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "CsrConfig",
    "pipeline_stages": "3",
}

xls_dslx_verilog(
    name = "csr_config_verilog",
    codegen_args = CSR_CONFIG_CODEGEN_ARGS,
    dslx_top = "CsrConfigInst",
    library = ":csr_config_dslx",
    tags = ["manual"],
    verilog_file = "csr_config.v",
)

xls_benchmark_ir(
    name = "csr_config_opt_ir_benchmark",
    src = ":csr_config_verilog.opt.ir",
    benchmark_ir_args = CSR_CONFIG_CODEGEN_ARGS | {
        "pipeline_stages": "10",
        "top": "__csr_config__CsrConfigInst__CsrConfig_0__2_32_4_32_2_4_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "csr_config_verilog_lib",
    srcs = [
        ":csr_config.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "csr_config_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "CsrConfig",
    deps = [
        ":csr_config_verilog_lib",
    ],
)

benchmark_synth(
    name = "csr_config_benchmark_synth",
    synth_target = ":csr_config_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "csr_config_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":csr_config_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "ram_wr_handler_dslx",
    srcs = ["ram_wr_handler.x"],
    deps = [
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "ram_wr_handler_dslx_test",
    library = ":ram_wr_handler_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "ram_rw_handler_verilog",
    codegen_args = {
        "module_name": "RamWrRespHandler",
        "delay_model": "asap7",
        "pipeline_stages": "1",
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "RamWrRespHandlerInst",
    library = ":ram_wr_handler_dslx",
    opt_ir_args = {
        "top": "__ram_wr_handler__RamWrRespHandlerInst__RamWrRespHandler_0__32_next",
    },
    tags = ["manual"],
    verilog_file = "ram_rw_handler.v",
)

xls_benchmark_ir(
    name = "ram_rw_handler_opt_ir_benchmark",
    src = ":ram_rw_handler_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

verilog_library(
    name = "ram_rw_handler_verilog_lib",
    srcs = [
        ":ram_rw_handler.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "ram_rw_handler_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RamWrRespHandler",
    deps = [
        ":ram_rw_handler_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_rw_handler_benchmark_synth",
    synth_target = ":ram_rw_handler_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "ram_rw_handler_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_rw_handler_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "fse_proba_freq_dec_dslx",
    srcs = ["fse_proba_freq_dec.x"],
    deps = [
        ":common_dslx",
        ":ram_wr_handler_dslx",
        ":refilling_shift_buffer_dslx",
        ":shift_buffer_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "fse_proba_freq_dec_dslx_test",
    library = ":fse_proba_freq_dec_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "fse_proba_freq_dec_verilog",
    codegen_args = {
        "module_name": "FseProbaFreqDec",
        "generator": "pipeline",
        "delay_model": "asap7",
        # FIXME: update ram rewrite
        #"ram_configurations": "ram:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
        #    latency = 5,
        #    rd_req = "fse_proba_freq_dec__rd_req_s",
        #    rd_resp = "fse_proba_freq_dec__rd_resp_r",
        #    wr_req = "fse_proba_freq_dec__wr_req_s",
        #    wr_resp = "fse_proba_freq_dec__wr_resp_r",
        #),
        "pipeline_stages": "6",
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "FseProbaFreqDecoderInst",
    library = ":fse_proba_freq_dec_dslx",
    tags = ["manual"],
    verilog_file = "fse_proba_freq_dec.v",
)

xls_benchmark_ir(
    name = "fse_proba_freq_dec_opt_ir_benchmark",
    src = ":fse_proba_freq_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
        "reset": "rst",
        # FIXME: update ram rewrite
        #"ram_configurations": "ram:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
        #    latency = 5,
        #    rd_req = "fse_proba_freq_dec__rd_req_s",
        #    rd_resp = "fse_proba_freq_dec__rd_resp_r",
        #    wr_req = "fse_proba_freq_dec__wr_req_s",
        #    wr_resp = "fse_proba_freq_dec__wr_resp_r",
        #),
    },
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "fse_proba_freq_dec_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "fse_proba_freq_dec_verilog",
)

verilog_library(
    name = "fse_proba_freq_dec_lib",
    srcs = [
        ":fse_proba_freq_dec.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "fse_proba_freq_dec_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "FseProbaFreqDec",
    deps = [
        ":fse_proba_freq_dec_lib",
    ],
)

benchmark_synth(
    name = "fse_proba_freq_dec_benchmark_synth",
    synth_target = ":fse_proba_freq_dec_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "fse_proba_freq_dec_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "detailed_routing",
    synthesized_rtl = ":fse_proba_freq_dec_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "literals_block_header_dec_dslx",
    srcs = ["literals_block_header_dec.x"],
    deps = [
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "literals_block_header_dec_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":literals_block_header_dec_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "literals_block_header_dec_verilog",
    codegen_args = COMMON_CODEGEN_ARGS | {
        "module_name": "LiteralsBlockHeaderDecoder",
    },
    dslx_top = "LiteralsHeaderDecoderInst",
    library = ":literals_block_header_dec_dslx",
    tags = ["manual"],
    verilog_file = "literals_block_header_dec.v",
)

xls_benchmark_ir(
    name = "literals_block_header_dec_opt_ir_benchmark",
    src = ":literals_block_header_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "top": "__literals_block_header_dec__LiteralsHeaderDecoderInst__LiteralsHeaderDecoder_0__16_64_next",
        "pipeline_stages": "10",
    },
    tags = ["manual"],
)

xls_dslx_library(
    name = "sequence_conf_dec_dslx",
    srcs = ["sequence_conf_dec.x"],
    deps = [
        ":common_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "sequence_conf_dec_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":sequence_conf_dec_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "sequence_conf_dec_verilog",
    codegen_args = COMMON_CODEGEN_ARGS | {
        "module_name": "SequenceConfDecoder",
    },
    dslx_top = "SequenceConfDecoderInst",
    library = ":sequence_conf_dec_dslx",
    tags = ["manual"],
    verilog_file = "sequence_conf_dec.v",
)

xls_benchmark_ir(
    name = "sequence_conf_dec_opt_ir_benchmark",
    src = ":sequence_conf_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "top": "__sequence_conf_dec__SequenceConfDecoderInst__SequenceConfDecoder_0__16_64_next",
        "pipeline_stages": "10",
    },
    tags = ["manual"],
)

xls_dslx_library(
    name = "refilling_shift_buffer_dslx",
    srcs = ["refilling_shift_buffer.x"],
    deps = [
        ":shift_buffer_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "refilling_shift_buffer_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":refilling_shift_buffer_dslx",
    tags = ["manual"],
)

REFILLING_SHIFT_BUFFER_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "RefillingShiftBufferInternalInst",
    "worst_case_throughput": "3",
}

xls_dslx_verilog(
    name = "refilling_shift_buffer_internal_verilog",
    codegen_args = REFILLING_SHIFT_BUFFER_CODEGEN_ARGS,
    dslx_top = "RefillingShiftBufferInternalInst",
    library = ":refilling_shift_buffer_dslx",
    tags = ["manual"],
    verilog_file = "refilling_shift_buffer_internal.v",
)

xls_benchmark_ir(
    name = "refilling_shift_buffer_internal_opt_ir_benchmark",
    src = ":refilling_shift_buffer_internal_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
    },
    codegen_args = REFILLING_SHIFT_BUFFER_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "refilling_shift_buffer_internal_verilog_lib",
    srcs = [
        ":refilling_shift_buffer_internal.v",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "refilling_shift_buffer_internal_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RefillingShiftBufferInternalInst",
    deps = [
        ":refilling_shift_buffer_internal_verilog_lib",
    ],
)

benchmark_synth(
    name = "refilling_shift_buffer_internal_benchmark_synth",
    synth_target = ":refilling_shift_buffer_internal_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "refilling_shift_buffer_internal_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.4",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":refilling_shift_buffer_internal_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "comp_block_dec_dslx",
    srcs = ["comp_block_dec.x"],
    deps = [
        ":command_constructor_dslx",
        ":common_dslx",
        ":fse_proba_freq_dec_dslx",
        ":huffman_literals_dec_dslx",
        ":literals_block_header_dec_dslx",
        ":literals_buffer_dslx",
        ":literals_decoder_dslx",
        ":parallel_rams_dslx",
        ":sequence_dec_dslx",
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd/memory:axi_ram_reader_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "comp_block_dec_dslx_test",
    size = "enormous",
    library = ":comp_block_dec_dslx",
    tags = [
        "local",
        "manual",
    ],
)

zstd_dec_deps = [
    ":axi_csr_accessor_dslx",
    ":block_header_dec_dslx",
    ":block_header_dslx",
    ":common_dslx",
    ":csr_config_dslx",
    ":dec_mux_dslx",
    ":frame_header_dec_dslx",
    ":raw_block_dec_dslx",
    ":rle_block_dec_dslx",
    ":comp_block_dec_dslx",
    ":sequence_executor_dslx",
    ":huffman_literals_dec_dslx",
    ":literals_buffer_dslx",
    ":parallel_rams_dslx",
    "//xls/examples:ram_dslx",
    "//xls/modules/zstd/memory:mem_reader_dslx",
    "//xls/modules/zstd/memory:mem_writer_dslx",
    "//xls/modules/zstd/memory:axi_ram_reader_dslx",
]

xls_dslx_library(
    name = "zstd_dec_dslx",
    srcs = [
        "zstd_dec.x",
    ],
    deps = zstd_dec_deps,
)

xls_dslx_test(
    name = "zstd_dec_dslx_test",
    size = "large",
    srcs = [
        "data/comp_frame.x",
        "data/comp_frame_fse_comp.x",
        "data/comp_frame_fse_repeated.x",
        "data/comp_frame_huffman.x",
        "data/comp_frame_huffman_fse.x",
        "zstd_dec.x",
        "zstd_dec_test.x",
    ],
    tags = [
        "local",
        "manual",
    ],
    deps = zstd_dec_deps,
)

ZSTD_DEC_INTERNAL_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "ZstdDecoderInternal",
    "pipeline_stages": "2",
    "worst_case_throughput": "2",
}

xls_dslx_verilog(
    name = "zstd_dec_internal_verilog",
    codegen_args = ZSTD_DEC_INTERNAL_CODEGEN_ARGS,
    dslx_top = "ZstdDecoderInternalInst",
    library = ":zstd_dec_dslx",
    tags = ["manual"],
    verilog_file = "zstd_dec_internal.v",
)

xls_benchmark_ir(
    name = "zstd_dec_internal_opt_ir_benchmark",
    src = ":zstd_dec_internal_verilog.opt.ir",
    benchmark_ir_args = ZSTD_DEC_INTERNAL_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "zstd_dec_internal_verilog_lib",
    srcs = [
        ":zstd_dec_internal.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "zstd_dec_internal_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "ZstdDecoderInternal",
    deps = [
        ":zstd_dec_internal_verilog_lib",
    ],
)

benchmark_synth(
    name = "zstd_dec_internal_benchmark_synth",
    synth_target = ":zstd_dec_internal_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "zstd_dec_internal_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.35",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":zstd_dec_internal_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "comp_lookup_dec_dslx",
    srcs = [
        "comp_lookup_dec.x",
    ],
    deps = [
        ":common_dslx",
        ":fse_proba_freq_dec_dslx",
        ":fse_table_creator_dslx",
        ":refilling_shift_buffer_dslx",
        ":shift_buffer_dslx",
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd/memory:axi_dslx",
        "//xls/modules/zstd/memory:axi_ram_reader_dslx",
    ],
)

xls_dslx_test(
    name = "comp_lookup_dec_dslx_test",
    library = ":comp_lookup_dec_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "rle_lookup_dec_dslx",
    srcs = [
        "rle_lookup_dec.x",
    ],
    deps = [
        ":common_dslx",
        ":fse_table_creator_dslx",
        ":refilling_shift_buffer_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "rle_lookup_dec_dslx_test",
    library = ":rle_lookup_dec_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "fse_lookup_dec_dslx",
    srcs = [
        "fse_lookup_dec.x",
    ],
    deps = [
        ":comp_lookup_dec_dslx",
        ":ram_mux_dslx",
        ":refilling_shift_buffer_dslx",
        ":refilling_shift_buffer_mux_dslx",
        ":rle_lookup_dec_dslx",
    ],
)

xls_dslx_test(
    name = "fse_lookup_dec_dslx_test",
    library = ":fse_lookup_dec_dslx",
    tags = ["manual"],
)

xls_dslx_library(
    name = "fse_table_iterator_dslx",
    srcs = ["fse_table_iterator.x"],
    deps = [
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "fse_table_iterator_dslx_test",
    library = ":fse_table_iterator_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "fse_table_iterator_verilog",
    codegen_args = {
        "module_name": "FseTableIterator",
        "delay_model": "asap7",
        "pipeline_stages": "1",
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "FseTableIterator",
    library = ":fse_table_iterator_dslx",
    tags = ["manual"],
    verilog_file = "fse_table_iterator.v",
)

xls_benchmark_ir(
    name = "fse_table_iterator_opt_ir_benchmark",
    src = ":fse_table_iterator_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

verilog_library(
    name = "fse_table_iterator_verilog_lib",
    srcs = [
        ":fse_table_iterator.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "fse_table_iterator_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "FseTableIterator",
    deps = [
        ":fse_table_iterator_verilog_lib",
    ],
)

benchmark_synth(
    name = "fse_table_iterator_benchmark_synth",
    synth_target = ":fse_table_iterator_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "fse_table_iterator_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":fse_table_iterator_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "fse_table_creator_dslx",
    srcs = ["fse_table_creator.x"],
    deps = [
        ":common_dslx",
        ":fse_table_iterator_dslx",
        ":ram_wr_handler_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "fse_table_creator_dslx_test",
    library = ":fse_table_creator_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "fse_table_creator_verilog",
    codegen_args = {
        "module_name": "FseTableCreator",
        "delay_model": "asap7",
        "pipeline_stages": "4",
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "FseTableCreatorInst",
    library = ":fse_table_creator_dslx",
    tags = ["manual"],
    verilog_file = "fse_table_creator.v",
)

xls_benchmark_ir(
    name = "fse_table_creator_opt_ir_benchmark",
    src = ":fse_table_creator_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "fse_table_creator_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "fse_table_creator_verilog",
)

verilog_library(
    name = "fse_table_creator_lib",
    srcs = [
        ":fse_table_creator.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "fse_table_creator_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "FseTableCreator",
    deps = [
        ":fse_table_creator_lib",
    ],
)

benchmark_synth(
    name = "fse_table_creator_benchmark_synth",
    synth_target = ":fse_table_creator_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "fse_table_creator_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":fse_table_creator_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "command_constructor_dslx",
    srcs = ["command_constructor.x"],
    deps = [
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "command_constructor_dslx_test",
    dslx_test_args = {"compare": "none"},
    library = ":command_constructor_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "command_constructor_verilog",
    codegen_args = {
        "module_name": "CommandConstructor",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "CommandConstructor",
    library = ":command_constructor_dslx",
    tags = ["manual"],
    verilog_file = "command_constructor.v",
)

xls_benchmark_ir(
    name = "command_constructor_opt_ir_benchmark",
    src = ":command_constructor_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "8",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "command_constructor_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "command_constructor_verilog",
)

verilog_library(
    name = "command_constructor_lib",
    srcs = [
        ":command_constructor.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "command_constructor_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "CommandConstructor",
    deps = [
        ":command_constructor_lib",
    ],
)

benchmark_synth(
    name = "command_constructor_benchmark_synth",
    synth_target = ":command_constructor_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "command_constructor_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.3",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":command_constructor_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "ram_demux_dslx",
    srcs = ["ram_demux.x"],
    deps = [
        ":ram_passthrough_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "ram_demux_dslx_test",
    dslx_test_args = {"compare": "none"},
    library = ":ram_demux_dslx",
    tags = ["manual"],
)

RAM_DEMUX_CODEGEN_ARGS = {
    "module_name": "RamDemux",
    "generator": "pipeline",
    "delay_model": "asap7",
    "worst_case_throughput": "6",
    "ram_configurations": ",".join([
        "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
            latency = 5,
            ram_name = "ram{}".format(num),
            rd_req = "ram_demux__rd_req{}_s".format(num),
            rd_resp = "ram_demux__rd_resp{}_r".format(num),
            wr_req = "ram_demux__wr_req{}_s".format(num),
            wr_resp = "ram_demux__wr_resp{}_r".format(num),
        )
        for num in range(2)
    ]),
    "clock_period_ps": "750",
    "reset": "rst",
    "use_system_verilog": "false",
    "materialize_internal_fifos": "true",
}

xls_dslx_verilog(
    name = "ram_demux_verilog",
    codegen_args = RAM_DEMUX_CODEGEN_ARGS,
    dslx_top = "RamDemuxWrappedInst",
    library = ":ram_demux_dslx",
    tags = ["manual"],
    verilog_file = "ram_demux.v",
)

xls_benchmark_ir(
    name = "ram_demux_opt_ir_benchmark",
    src = "ram_demux_verilog.opt.ir",
    codegen_args = RAM_DEMUX_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "ram_demux_verilog_lib",
    srcs = [
        ":ram_demux.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "ram_demux_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RamDemux",
    deps = [
        ":ram_demux_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_demux_benchmark_synth",
    synth_target = ":ram_demux_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "ram_demux_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_demux_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "5",
)

ram_demux_naive_codegen_args = {
    "module_name": "RamDemuxNaive",
    "generator": "pipeline",
    "delay_model": "asap7",
    "ram_configurations": ",".join([
        "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
            latency = 1,
            ram_name = "ram{}".format(num),
            rd_req = "ram_demux__rd_req{}_s".format(num),
            rd_resp = "ram_demux__rd_resp{}_r".format(num),
            wr_req = "ram_demux__wr_req{}_s".format(num),
            wr_resp = "ram_demux__wr_resp{}_r".format(num),
        )
        for num in range(2)
    ]),
    "pipeline_stages": "6",
    "reset": "rst",
    "use_system_verilog": "false",
    "materialize_internal_fifos": "true",
}

xls_dslx_verilog(
    name = "ram_demux_naive_verilog",
    codegen_args = ram_demux_naive_codegen_args,
    dslx_top = "RamDemuxNaiveWrappedInst",
    library = ":ram_demux_dslx",
    tags = ["manual"],
    verilog_file = "ram_demux_naive.v",
)

xls_benchmark_ir(
    name = "ram_demux_naive_opt_ir_benchmark",
    src = "ram_demux_naive_verilog.opt.ir",
    codegen_args = ram_demux_naive_codegen_args,
    tags = ["manual"],
)

verilog_library(
    name = "ram_demux_naive_verilog_lib",
    srcs = [
        ":ram_demux_naive.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "ram_demux_naive_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RamDemuxNaive",
    deps = [
        ":ram_demux_naive_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_demux_naive_benchmark_synth",
    synth_target = ":ram_demux_naive_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "ram_demux_naive_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_demux_naive_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "5",
)

xls_dslx_library(
    name = "ram_passthrough_dslx",
    srcs = ["ram_passthrough.x"],
    deps = [
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "ram_passthrough_dslx_test",
    dslx_test_args = {"compare": "none"},
    library = ":ram_passthrough_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "ram_passthrough_verilog",
    codegen_args = {
        "module_name": "RamPassthrough",
        "generator": "pipeline",
        "delay_model": "asap7",
        "ram_configurations": "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
            latency = 1,
            ram_name = "ram",
            rd_req = "ram_passthrough__rd_req_s",
            rd_resp = "ram_passthrough__rd_resp_r",
            wr_req = "ram_passthrough__wr_req_s",
            wr_resp = "ram_passthrough__wr_resp_r",
        ),
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
        "clock_period_ps": "750",
    },
    dslx_top = "RamPassthroughInst",
    library = ":ram_passthrough_dslx",
    tags = ["manual"],
    verilog_file = "ram_passthrough.v",
)

xls_dslx_library(
    name = "fse_dec_dslx",
    srcs = [
        "fse_dec.x",
    ],
    deps = [
        ":common_dslx",
        ":fse_table_creator_dslx",
        ":math_dslx",
        ":refilling_shift_buffer_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "fse_dec_dslx_test",
    library = ":fse_dec_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "fse_dec_verilog",
    codegen_args = {
        "module_name": "FseDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "8",
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "FseDecoderInst",
    library = ":fse_dec_dslx",
    opt_ir_args = {
        "top": "__fse_dec__FseDecoderInst__FseDecoder_0__64_8_32_4_64_7_next",
    },
    tags = ["manual"],
    verilog_file = "fse_dec.v",
)

xls_benchmark_ir(
    name = "fse_dec_opt_ir_benchmark",
    src = ":fse_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "delay_model": "asap7",
        "pipeline_stages": "3",
    },
    tags = ["manual"],
)

verilog_library(
    name = "fse_dec_verilog_lib",
    srcs = [
        ":fse_dec.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "fse_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "FseDecoder",
    deps = [
        ":fse_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "fse_dec_benchmark_synth",
    synth_target = ":fse_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "fse_dec_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    die_height_microns = 100,
    die_width_microns = 100,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":fse_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "ram_demux3_dslx",
    srcs = ["ram_demux3.x"],
    deps = [":ram_demux_dslx"],
)

xls_dslx_test(
    name = "ram_demux3_dslx_test",
    library = ":ram_demux3_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "ram_demux3_verilog",
    codegen_args = {
        "module_name": "RamDemux3",
        "generator": "pipeline",
        "worst_case_throughput": "6",
        "delay_model": "asap7",
        "ram_configurations": ",".join([
            "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
                latency = 5,
                ram_name = "ram{}".format(num),
                rd_req = "ram_demux3__rd_req{}_s".format(num),
                rd_resp = "ram_demux3__rd_resp{}_r".format(num),
                wr_req = "ram_demux3__wr_req{}_s".format(num),
                wr_resp = "ram_demux3__wr_resp{}_r".format(num),
            )
            for num in range(3)
        ]),
        "pipeline_stages": "6",
        "reset": "rst",
        "use_system_verilog": "false",
        "multi_proc": "true",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "RamDemux3Inst",
    library = ":ram_demux3_dslx",
    tags = ["manual"],
    verilog_file = "ram_demux3.v",
)

xls_benchmark_ir(
    name = "ram_demux3_opt_ir_benchmark",
    src = ":ram_demux3_verilog.opt.ir",
    benchmark_ir_args = {
        "delay_model": "asap7",
        "pipeline_stages": "3",
        "worst_case_throughput": "6",
    },
    tags = ["manual"],
)

verilog_library(
    name = "ram_demux3_verilog_lib",
    srcs = [
        ":ram_demux3.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "ram_demux3_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RamDemux3",
    deps = [
        ":ram_demux3_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_demux3_benchmark_synth",
    synth_target = ":ram_demux3_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "ram_demux3_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    die_height_microns = 100,
    die_width_microns = 100,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_demux3_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "ram_mux_dslx",
    srcs = [
        "ram_mux.x",
    ],
    deps = [
        ":ram_passthrough_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "ram_mux_dslx_test",
    library = ":ram_mux_dslx",
    tags = ["manual"],
)

RAM_MUX_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "RamMux",
    "worst_case_throughput": "4",
    "ram_configurations":
        "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
            latency = 1,
            ram_name = "ram",
            rd_req = "ram_mux__rd_req_s",
            rd_resp = "ram_mux__rd_resp_r",
            wr_req = "ram_mux__wr_req_s",
            wr_resp = "ram_mux__wr_resp_r",
        )
}

xls_dslx_verilog(
    name = "ram_mux_verilog",
    codegen_args = RAM_MUX_CODEGEN_ARGS,
    dslx_top = "RamMuxWrappedInst",
    library = ":ram_mux_dslx",
    tags = ["manual"],
    verilog_file = "ram_mux.v",
)

xls_benchmark_ir(
    name = "ram_mux_opt_ir_benchmark",
    src = ":ram_mux_verilog.opt.ir",
    codegen_args = RAM_MUX_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "ram_mux_verilog_lib",
    srcs = [
        ":ram_mux.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "ram_mux_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RamMux",
    deps = [
        ":ram_mux_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_mux_benchmark_synth",
    synth_target = ":ram_mux_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "ram_mux_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    die_height_microns = 120,
    die_width_microns = 120,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_mux_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "sequence_dec_dslx",
    srcs = [
        "sequence_dec.x",
    ],
    deps = [
        ":common_dslx",
        ":comp_lookup_dec_dslx",
        ":fse_dec_dslx",
        ":fse_lookup_dec_dslx",
        ":fse_table_creator_dslx",
        ":ram_demux3_dslx",
        ":ram_mux_dslx",
        ":refilling_shift_buffer_dslx",
        ":sequence_conf_dec_dslx",
        ":shift_buffer_dslx",
        "//xls/modules/zstd/memory:axi_ram_reader_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "sequence_dec_dslx_test",
    library = ":sequence_dec_dslx",
    tags = [
        "local",
        "manual",
    ],
)

xls_dslx_verilog(
    name = "fse_lookup_ctrl_verilog",
    codegen_args = {
        "module_name": "FseLookupCtrl",
        "generator": "pipeline",
        "delay_model": "asap7",
        "pipeline_stages": "6",
        "reset": "rst",
        "use_system_verilog": "false",
        "multi_proc": "true",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "FseLookupCtrlInst",
    library = ":sequence_dec_dslx",
    opt_ir_args = {
        "top": "__sequence_dec__FseLookupCtrlInst__FseLookupCtrl_0_next",
    },
    tags = ["manual"],
    verilog_file = "fse_lookup_ctrl.v",
)

xls_benchmark_ir(
    name = "fse_lookup_ctrl_opt_ir_benchmark",
    src = ":fse_lookup_ctrl_verilog.opt.ir",
    benchmark_ir_args = {
        "delay_model": "asap7",
        "pipeline_stages": "6",
    },
    tags = ["manual"],
)

xls_dslx_library(
    name = "rle_literals_dec_dslx",
    srcs = [
        "rle_literals_dec.x",
    ],
    deps = [
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "rle_literals_dec_dslx_test",
    library = ":rle_literals_dec_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "rle_literals_dec_verilog",
    codegen_args = {
        "module_name": "rle_literals_dec",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "worst_case_throughput": "1",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "RleLiteralsDecoderInst",
    library = ":rle_literals_dec_dslx",
    tags = ["manual"],
    verilog_file = "rle_literals_dec.v",
)

xls_benchmark_ir(
    name = "rle_literals_dec_opt_ir_benchmark",
    src = ":rle_literals_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

verilog_library(
    name = "rle_literals_dec_verilog_lib",
    srcs = [
        ":rle_literals_dec.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "rle_literals_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "rle_literals_dec",
    deps = [
        ":rle_literals_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "rle_literals_dec_benchmark_synth",
    synth_target = ":rle_literals_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "rle_literals_dec_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":rle_literals_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "raw_literals_dec_dslx",
    srcs = ["raw_literals_dec.x"],
    deps = [
        ":common_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "raw_literals_dec_dslx_test",
    library = ":raw_literals_dec_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "raw_literals_dec_verilog",
    codegen_args = {
        "module_name": "RawLiteralsDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "RawLiteralsDecoderInst",
    library = ":raw_literals_dec_dslx",
    opt_ir_args = {
        "top": "__raw_literals_dec__RawLiteralsDecoderInst__RawLiteralsDecoder_0__16_64_next",
    },
    tags = ["manual"],
    verilog_file = "raw_literals_dec.v",
)

xls_benchmark_ir(
    name = "raw_literals_dec_opt_ir_benchmark",
    src = ":raw_literals_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
        "top": "__raw_literals_dec__RawLiteralsDecoderInst__RawLiteralsDecoder_0__16_64_next",
    },
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "raw_literals_dec_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "raw_literals_dec_verilog",
)

verilog_library(
    name = "raw_literals_dec_lib",
    srcs = [
        ":raw_literals_dec.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "raw_literals_dec_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "RawLiteralsDecoder",
    deps = [
        ":raw_literals_dec_lib",
    ],
)

benchmark_synth(
    name = "raw_literals_dec_benchmark_synth",
    synth_target = ":raw_literals_dec_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "raw_literals_dec_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":raw_literals_dec_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "literals_buffer_dslx",
    srcs = [
        "literals_buffer.x",
    ],
    deps = [
        ":common_dslx",
        ":parallel_rams_dslx",
        ":ram_printer_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "literals_buffer_dslx_test",
    library = ":literals_buffer_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "literals_buffer_verilog",
    codegen_args = {
        "module_name": "LiteralsBuffer",
        "delay_model": "asap7",
        "ram_configurations": ",".join([
            "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
                latency = 5,
                ram_name = "ram{}".format(num),
                rd_req = "literals_buffer__rd_req_m{}_s".format(num),
                rd_resp = "literals_buffer__rd_resp_m{}_r".format(num),
                wr_req = "literals_buffer__wr_req_m{}_s".format(num),
                wr_resp = "literals_buffer__wr_resp_m{}_r".format(num),
            )
            for num in range(7)
        ]),
        "pipeline_stages": "6",
        "reset": "rst",
        "worst_case_throughput": "1",
        "use_system_verilog": "false",
        "multi_proc": "true",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "LiteralsBufferInst",
    library = ":literals_buffer_dslx",
    tags = ["manual"],
    verilog_file = "literals_buffer.v",
)

xls_benchmark_ir(
    name = "literals_buffer_opt_ir_benchmark",
    src = ":literals_buffer_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "6",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "literals_buffer_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "literals_buffer_verilog",
)

verilog_library(
    name = "literals_buffer_verilog_lib",
    srcs = [
        ":literals_buffer.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "literals_buffer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "LiteralsBuffer",
    deps = [
        ":literals_buffer_verilog_lib",
    ],
)

benchmark_synth(
    name = "literals_buffer_benchmark_synth",
    synth_target = ":literals_buffer_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "literals_buffer_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":literals_buffer_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "literals_decoder_dslx",
    srcs = [
        "literals_decoder.x",
    ],
    deps = [
        ":common_dslx",
        ":huffman_literals_dec_dslx",
        ":literals_block_header_dec_dslx",
        ":literals_buffer_dslx",
        ":parallel_rams_dslx",
        ":ram_printer_dslx",
        ":raw_literals_dec_dslx",
        ":rle_literals_dec_dslx",
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd/memory:axi_dslx",
        "//xls/modules/zstd/memory:axi_ram_reader_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "literals_decoder_dslx_test",
    library = ":literals_decoder_dslx",
    tags = ["manual"],
)

LITERALS_DECODER_CTRL_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "LiteralsDecoderCtrl",
    "pipeline_stages": "10",
}

xls_dslx_verilog(
    name = "literals_decoder_ctrl_verilog",
    codegen_args = LITERALS_DECODER_CTRL_CODEGEN_ARGS,
    dslx_top = "LiteralsDecoderCtrlInst",
    library = ":literals_decoder_dslx",
    tags = ["manual"],
    verilog_file = "literals_decoder_ctrl.v",
)

xls_benchmark_ir(
    name = "literals_decoder_ctrl_opt_ir_benchmark",
    src = ":literals_decoder_ctrl_verilog.opt.ir",
    benchmark_ir_args = LITERALS_DECODER_CTRL_CODEGEN_ARGS | {
        "multi_proc": "true",
    },
    tags = ["manual"],
)

verilog_library(
    name = "literals_decoder_ctrl_verilog_lib",
    srcs = [
        ":literals_decoder_ctrl.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "literals_decoder_ctrl_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "LiteralsDecoderCtrl",
    deps = [
        ":literals_decoder_ctrl_verilog_lib",
    ],
)

benchmark_synth(
    name = "literals_decoder_ctrl_benchmark_synth",
    synth_target = ":literals_decoder_ctrl_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "literals_decoder_ctrl_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":literals_decoder_ctrl_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

LITERALS_DECODER_CODEGEN_ARGS = {
    "module_name": "LiteralsDecoder",
    "delay_model": "asap7",
    "ram_configurations": ",".join([
        "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
            latency = 5,
            ram_name = "ram{}".format(num),
            rd_req = "literals_decoder__rd_req_m{}_s".format(num),
            rd_resp = "literals_decoder__rd_resp_m{}_r".format(num),
            wr_req = "literals_decoder__wr_req_m{}_s".format(num),
            wr_resp = "literals_decoder__wr_resp_m{}_r".format(num),
        )
        for num in range(7)
    ]),
    "pipeline_stages": "8",
    "reset": "rst",
    "worst_case_throughput": "3",
    "use_system_verilog": "false",
    "materialize_internal_fifos": "false",  # TODO: set to true once this option works with loopback channels
}

xls_dslx_verilog(
    name = "literals_decoder_verilog",
    codegen_args = LITERALS_DECODER_CODEGEN_ARGS,
    dslx_top = "LiteralsDecoderInst",
    library = ":literals_decoder_dslx",
    tags = ["manual"],
    verilog_file = "literals_decoder.v",
)

xls_benchmark_ir(
    name = "literals_decoder_opt_ir_benchmark",
    src = ":literals_decoder_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
    },
    codegen_args = LITERALS_DECODER_CODEGEN_ARGS,
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "literals_decoder_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "literals_decoder_verilog",
)

verilog_library(
    name = "literals_decoder_verilog_lib",
    srcs = [
        ":literals_decoder.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "literals_decoder_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "LiteralsDecoder",
    deps = [
        ":literals_decoder_verilog_lib",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper_verilog_lib",
    ],
)

benchmark_synth(
    name = "literals_decoder_benchmark_synth",
    synth_target = ":literals_decoder_synth_asap7",
    tags = ["manual"],
)

# TODO: Remove _skip suffix after fixing long P&R time. Used now to skip this target in CI.
place_and_route(
    name = "literals_decoder_place_and_route_skip",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":literals_decoder_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_common_dslx",
    srcs = [
        "huffman_common.x",
    ],
    deps = [],
)

xls_dslx_library(
    name = "huffman_prescan_dslx",
    srcs = [
        "huffman_prescan.x",
    ],
    deps = [
        ":common_dslx",
        ":huffman_common_dslx",
        "//xls/dslx/stdlib:acm_random_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_prescan_dslx_test",
    library = ":huffman_prescan_dslx",
    tags = ["manual"],
)

PRESCAN_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanPrescan",
    "pipeline_stages": "16",
    "clock_period_ps": "750",
    "worst_case_throughput": "1",
    "ram_configurations": "InternalRam:1R1W:huffman_prescan__internal_read_req_s" +
                          ":huffman_prescan__internal_read_rsp_r:" +
                          "huffman_prescan__internal_write_req_s:" +
                          "huffman_prescan__internal_write_rsp_r:5",
    "io_constraints": "huffman_prescan__read_req_s:send:" +
                      "huffman_prescan__read_rsp_r:recv:5:5",
    "materialize_internal_fifos": "false",  # TODO: remove once this option works with loopback channels
}

xls_dslx_verilog(
    name = "huffman_prescan_verilog",
    codegen_args = PRESCAN_CODEGEN_ARGS,
    dslx_top = "WeightPreScan",
    library = ":huffman_prescan_dslx",
    tags = ["manual"],
    verilog_file = "huffman_prescan.v",
)

xls_benchmark_ir(
    name = "huffman_prescan_opt_ir_benchmark",
    src = ":huffman_prescan_verilog.opt.ir",
    benchmark_ir_args = PRESCAN_CODEGEN_ARGS,
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "huffman_prescan_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "huffman_prescan_verilog",
)

verilog_library(
    name = "huffman_prescan_verilog_lib",
    srcs = [
        ":huffman_prescan.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_prescan_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanPrescan",
    deps = [
        ":huffman_prescan_verilog_lib",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_prescan_benchmark_synth",
    synth_target = ":huffman_prescan_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "huffman_prescan_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    synthesized_rtl = ":huffman_prescan_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_code_builder_dslx",
    srcs = [
        "huffman_code_builder.x",
    ],
    deps = [
        ":common_dslx",
        ":huffman_common_dslx",
        "//xls/dslx/stdlib:acm_random_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_code_builder_dslx_test",
    library = ":huffman_code_builder_dslx",
    tags = ["manual"],
)

HUFFMAN_CODE_BUILDER_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanCodeBuilder",
    "pipeline_stages": "8",
    "clock_period_ps": "750",
    "worst_case_throughput": "5",
}

xls_dslx_verilog(
    name = "huffman_code_builder_verilog",
    codegen_args = HUFFMAN_CODE_BUILDER_CODEGEN_ARGS,
    dslx_top = "WeightCodeBuilder",
    library = ":huffman_code_builder_dslx",
    tags = ["manual"],
    verilog_file = "huffman_code_builder.v",
)

xls_benchmark_ir(
    name = "huffman_code_builder_opt_ir_benchmark",
    src = ":huffman_code_builder_verilog.opt.ir",
    benchmark_ir_args = HUFFMAN_CODE_BUILDER_CODEGEN_ARGS,
    tags = ["manual"],
)

xls_benchmark_verilog(
    name = "huffman_code_builder_verilog_benchmark",
    tags = ["manual"],
    verilog_target = "huffman_code_builder_verilog",
)

verilog_library(
    name = "huffman_code_builder_verilog_lib",
    srcs = [
        ":huffman_code_builder.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_code_builder_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanCodeBuilder",
    deps = [
        ":huffman_code_builder_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_code_builder_benchmark_synth",
    synth_target = ":huffman_code_builder_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "huffman_code_builder_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    synthesized_rtl = ":huffman_code_builder_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_axi_reader_dslx",
    srcs = [
        "huffman_axi_reader.x",
    ],
    deps = [
        "//xls/modules/zstd/memory:axi_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_axi_reader_dslx_test",
    library = ":huffman_axi_reader_dslx",
    tags = ["manual"],
)

HUFFMAN_AXI_READER_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanAxiReader",
    "clock_period_ps": "750",
    "worst_case_throughput": "4",
}

xls_dslx_verilog(
    name = "huffman_axi_reader_verilog",
    codegen_args = HUFFMAN_AXI_READER_CODEGEN_ARGS,
    dslx_top = "HuffmanAxiReaderInst",
    library = ":huffman_axi_reader_dslx",
    tags = ["manual"],
    verilog_file = "huffman_axi_reader.v",
)

xls_benchmark_ir(
    name = "huffman_axi_reader_opt_ir_benchmark",
    src = ":huffman_axi_reader_verilog.opt.ir",
    benchmark_ir_args = HUFFMAN_AXI_READER_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "huffman_axi_reader_verilog_lib",
    srcs = [
        ":huffman_axi_reader.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_axi_reader_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanAxiReader",
    deps = [
        ":huffman_axi_reader_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_axi_reader_benchmark_synth",
    synth_target = ":huffman_axi_reader_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "huffman_axi_reader_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":huffman_axi_reader_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_data_preprocessor_dslx",
    srcs = [
        "huffman_data_preprocessor.x",
    ],
    deps = [
        ":common_dslx",
        ":huffman_axi_reader_dslx",
        ":huffman_common_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_data_preprocessor_dslx_test",
    library = ":huffman_data_preprocessor_dslx",
    tags = ["manual"],
)

HUFFMAN_DATA_PREPROCESSOR_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanDataPreprocessor",
    "pipeline_stages": "36",
    "clock_period_ps": "810",
    "clock_margin_percent": "0",
    "worst_case_throughput": "1",
}

xls_dslx_verilog(
    name = "huffman_data_preprocessor_verilog",
    codegen_args = HUFFMAN_DATA_PREPROCESSOR_CODEGEN_ARGS,
    dslx_top = "HuffmanDataPreprocessor",
    library = ":huffman_data_preprocessor_dslx",
    tags = ["manual"],
    verilog_file = "huffman_data_preprocessor.v",
)

xls_benchmark_ir(
    name = "huffman_data_preprocessor_opt_ir_benchmark",
    src = ":huffman_data_preprocessor_verilog.opt.ir",
    benchmark_ir_args = HUFFMAN_DATA_PREPROCESSOR_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "huffman_data_preprocessor_verilog_lib",
    srcs = [
        ":huffman_data_preprocessor.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_data_preprocessor_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanDataPreprocessor",
    deps = [
        ":huffman_data_preprocessor_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_data_preprocessor_benchmark_synth",
    synth_target = ":huffman_data_preprocessor_synth_asap7",
    tags = ["manual"],
)

# TODO: Remove _skip suffix after fixing long P&R time. Used now to skip this target in CI.
place_and_route(
    name = "huffman_data_preprocessor_place_and_route_skip",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":huffman_data_preprocessor_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_decoder_dslx",
    srcs = [
        "huffman_decoder.x",
    ],
    deps = [
        ":common_dslx",
        ":huffman_common_dslx",
        ":huffman_data_preprocessor_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_decoder_dslx_test",
    library = ":huffman_decoder_dslx",
    tags = ["manual"],
)

HUFFMAN_DECODER_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanDecoder",
    "pipeline_stages": "8",
    "clock_period_ps": "0",
    "worst_case_throughput": "1",
}

xls_dslx_verilog(
    name = "huffman_decoder_verilog",
    codegen_args = HUFFMAN_DECODER_CODEGEN_ARGS,
    dslx_top = "HuffmanDecoder",
    library = ":huffman_decoder_dslx",
    tags = ["manual"],
    verilog_file = "huffman_decoder.v",
)

xls_benchmark_ir(
    name = "huffman_decoder_opt_ir_benchmark",
    src = ":huffman_decoder_verilog.opt.ir",
    benchmark_ir_args = HUFFMAN_DECODER_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "huffman_decoder_verilog_lib",
    srcs = [
        ":huffman_decoder.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_decoder_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanDecoder",
    deps = [
        ":huffman_decoder_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_decoder_benchmark_synth",
    synth_target = ":huffman_decoder_synth_asap7",
    tags = ["manual"],
)

# TODO: Remove _skip suffix after fixing long P&R time. Used now to skip this target in CI.
place_and_route(
    name = "huffman_decoder_place_and_route_skip",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":huffman_decoder_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_ctrl_dslx",
    srcs = [
        "huffman_ctrl.x",
    ],
    deps = [
        ":common_dslx",
        ":huffman_axi_reader_dslx",
        ":huffman_code_builder_dslx",
        ":huffman_common_dslx",
        ":huffman_data_preprocessor_dslx",
        ":huffman_decoder_dslx",
        ":huffman_prescan_dslx",
        ":huffman_weights_dec_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_ctrl_dslx_test",
    library = ":huffman_ctrl_dslx",
    tags = ["manual"],
)

HUFFMAN_CTRL_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanCtrl",
    "pipeline_stages": "8",
    "clock_period_ps": "750",
    "worst_case_throughput": "3",
}

xls_dslx_verilog(
    name = "huffman_ctrl_verilog",
    codegen_args = HUFFMAN_CTRL_CODEGEN_ARGS,
    dslx_top = "HuffmanControlAndSequenceInst",
    library = ":huffman_ctrl_dslx",
    tags = ["manual"],
    verilog_file = "huffman_ctrl.v",
)

xls_benchmark_ir(
    name = "huffman_ctrl_opt_ir_benchmark",
    src = ":huffman_ctrl_verilog.opt.ir",
    benchmark_ir_args = HUFFMAN_CTRL_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "huffman_ctrl_verilog_lib",
    srcs = [
        ":huffman_ctrl.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_ctrl_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanCtrl",
    deps = [
        ":huffman_ctrl_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_ctrl_benchmark_synth",
    synth_target = ":huffman_ctrl_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "huffman_ctrl_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":huffman_ctrl_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_weights_dec_dslx",
    srcs = [
        "huffman_weights_dec.x",
    ],
    deps = [
        ":comp_lookup_dec_dslx",
        ":fse_table_creator_dslx",
        ":huffman_prescan_dslx",
        ":math_dslx",
        ":ram_mux_dslx",
        ":refilling_shift_buffer_dslx",
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd/memory:axi_ram_reader_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_weights_dec_dslx_test",
    size = "large",
    library = ":huffman_weights_dec_dslx",
    tags = [
        "local",
        "manual",
    ],
)

HUFFMAN_WEIGHTS_DEC_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanWeightsDecoder",
    "pipeline_stages": "25",
    "clock_period_ps": "750",
    "worst_case_throughput": "17",
    "multi_proc": "true",
}

xls_dslx_verilog(
    name = "huffman_weights_dec_verilog",
    codegen_args = HUFFMAN_WEIGHTS_DEC_CODEGEN_ARGS,
    dslx_top = "HuffmanWeightsDecoderInst",
    library = ":huffman_weights_dec_dslx",
    opt_ir_args = {
        "top": "__huffman_weights_dec__HuffmanWeightsDecoderInst__HuffmanWeightsDecoder_0__32_64_8_8_16_1_8_32_4_9_8_1_8_16_1_6_32_8_next",
    },
    tags = ["manual"],
    verilog_file = "huffman_weights_dec.v",
)

xls_benchmark_ir(
    name = "huffman_weights_dec_opt_ir_benchmark",
    src = ":huffman_weights_dec_verilog.opt.ir",
    benchmark_ir_args = HUFFMAN_WEIGHTS_DEC_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "huffman_weights_dec_verilog_lib",
    srcs = [
        ":huffman_weights_dec.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_weights_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanWeightsDecoder",
    deps = [
        ":huffman_weights_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_weights_dec_benchmark_synth",
    synth_target = ":huffman_weights_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "huffman_weights_dec_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":huffman_weights_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "huffman_literals_dec_dslx",
    srcs = [
        "huffman_literals_dec.x",
    ],
    deps = [
        ":common_dslx",
        ":huffman_axi_reader_dslx",
        ":huffman_code_builder_dslx",
        ":huffman_common_dslx",
        ":huffman_ctrl_dslx",
        ":huffman_data_preprocessor_dslx",
        ":huffman_decoder_dslx",
        ":huffman_prescan_dslx",
        "//xls/modules/zstd/memory:axi_ram_reader_dslx",
    ],
)

xls_dslx_test(
    name = "huffman_literals_dec_dslx_test",
    library = ":huffman_literals_dec_dslx",
    tags = ["manual"],
)

HUFFMAN_LITERALS_DEC_CODEGEN_ARGS = COMMON_CODEGEN_ARGS | {
    "module_name": "HuffmanLiteralsDecoder",
    "pipeline_stages": "64",
    "clock_period_ps": "0",
    "worst_case_throughput": "0",
    "minimize_worst_case_throughput": "true",
    "materialize_internal_fifos": "false",  # TODO: remove once this option works with loopback channels
}

xls_dslx_verilog(
    name = "huffman_literals_dec_verilog",
    codegen_args = HUFFMAN_LITERALS_DEC_CODEGEN_ARGS,
    dslx_top = "HuffmanLiteralsDecoderInst",
    library = ":huffman_literals_dec_dslx",
    tags = ["manual"],
    verilog_file = "huffman_literals_dec.v",
)

xls_benchmark_ir(
    name = "huffman_literals_dec_opt_ir_benchmark",
    src = ":huffman_literals_dec_verilog.opt.ir",
    benchmark_ir_args = HUFFMAN_LITERALS_DEC_CODEGEN_ARGS,
    tags = ["manual"],
)

verilog_library(
    name = "huffman_literals_dec_verilog_lib",
    srcs = [
        ":huffman_literals_dec.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "huffman_literals_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "HuffmanLiteralsDecoder",
    deps = [
        ":huffman_literals_dec_verilog_lib",
        "//xls/modules/zstd/rtl:xls_fifo_wrapper_verilog_lib",
    ],
)

benchmark_synth(
    name = "huffman_literals_dec_benchmark_synth",
    synth_target = ":huffman_literals_dec_synth_asap7",
    tags = ["manual"],
)

# TODO: Remove _skip suffix after fixing long P&R time. Used now to skip this target in CI.
place_and_route(
    name = "huffman_literals_dec_place_and_route_skip",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":huffman_literals_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "refilling_shift_buffer_mux_dslx",
    srcs = [
        "refilling_shift_buffer_mux.x",
    ],
    deps = [
        ":refilling_shift_buffer_dslx",
    ],
)

xls_dslx_test(
    name = "refilling_shift_buffer_mux_dslx_test",
    library = ":refilling_shift_buffer_mux_dslx",
)
