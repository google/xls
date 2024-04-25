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
load("@xls_pip_deps//:requirements.bzl", "requirement")
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

exports_files(["xls_fifo_wrapper.v"])

CLOCK_PERIOD_PS = "750"
# Clock periods for modules that exceed the 750ps critical path in IR benchmark

common_codegen_args = {
    "delay_model": "asap7",
    "reset": "rst",
    "worst_case_throughput": "1",
    "use_system_verilog": "false",
    "clock_period_ps": CLOCK_PERIOD_PS,
    "clock_margin_percent": "20",
    "multi_proc": "true",
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
    dslx_test_args = {"compare": "jit"},
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
    dslx_test_args = {"compare": "jit"},
    library = ":window_buffer_dslx",
    tags = ["manual"],
)

window_buffer_codegen_args = common_codegen_args | {
    "module_name": "WindowBuffer64",
    "clock_period_ps": "0",
    "pipeline_stages": "1",
    "worst_case_throughput": "2",
}

xls_dslx_verilog(
    name = "window_buffer_verilog",
    codegen_args = window_buffer_codegen_args,
    dslx_top = "WindowBuffer64",
    library = ":window_buffer_dslx",
    tags = ["manual"],
    verilog_file = "window_buffer.v",
)

xls_benchmark_ir(
    name = "window_buffer_opt_ir_benchmark",
    src = ":window_buffer_verilog.opt.ir",
    benchmark_ir_args = window_buffer_codegen_args | {
        "pipeline_stages": "10",
        "top": "__window_buffer__WindowBuffer64__WindowBuffer_0__64_32_48_next",
    },
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

cc_library(
    name = "data_generator",
    srcs = ["data_generator.cc"],
    hdrs = ["data_generator.h"],
    data = [
        "@zstd//:decodecorpus",
    ],
    deps = [
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

xls_dslx_library(
    name = "common_dslx",
    srcs = [
        "common.x",
    ],
    deps = [
        "//xls/modules/shift_buffer:shift_buffer_dslx",
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

frame_header_dec_codegen_args = common_codegen_args | {
    "module_name": "FrameHeaderDecoder",
    "clock_period_ps": "0",
    "pipeline_stages": "6",
}

xls_dslx_verilog(
    name = "frame_header_dec_verilog",
    codegen_args = frame_header_dec_codegen_args,
    dslx_top = "FrameHeaderDecoderInst",
    library = ":frame_header_dec_dslx",
    tags = ["manual"],
    verilog_file = "frame_header_dec.v",
)

xls_benchmark_ir(
    name = "frame_header_dec_opt_ir_benchmark",
    src = ":frame_header_dec_verilog.opt.ir",
    benchmark_ir_args = frame_header_dec_codegen_args | {
        "top": "__frame_header_dec__FrameHeaderDecoderInst__FrameHeaderDecoder_0__16_32_30_5_next",
        "pipeline_stages": "10",
    },
    tags = ["manual"],
)

verilog_library(
    name = "frame_header_dec_verilog_lib",
    srcs = [
        ":frame_header_dec.v",
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

block_header_dec_codegen_args = common_codegen_args | {
    "module_name": "BlockHeaderDec",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "block_header_dec_verilog",
    codegen_args = block_header_dec_codegen_args,
    dslx_top = "BlockHeaderDecoderInst",
    library = ":block_header_dec_dslx",
    tags = ["manual"],
    verilog_file = "block_header_dec.v",
)

xls_benchmark_ir(
    name = "block_header_dec_opt_ir_benchmark",
    src = ":block_header_dec_verilog.opt.ir",
    benchmark_ir_args = block_header_dec_codegen_args | {
        "pipeline_stages": "10",
        "top": "__block_header_dec__BlockHeaderDecoderInst__BlockHeaderDecoder_0__16_64_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "block_header_dec_verilog_lib",
    srcs = [
        ":block_header_dec.v",
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

raw_block_dec_codegen_args = common_codegen_args | {
    "module_name": "RawBlockDecoder",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "raw_block_dec_verilog",
    codegen_args = raw_block_dec_codegen_args,
    dslx_top = "RawBlockDecoderInst",
    library = ":raw_block_dec_dslx",
    tags = ["manual"],
    verilog_file = "raw_block_dec.v",
)

xls_benchmark_ir(
    name = "raw_block_dec_opt_ir_benchmark",
    src = ":raw_block_dec_verilog.opt.ir",
    benchmark_ir_args = raw_block_dec_codegen_args | {
        "pipeline_stages": "10",
        "top": "__raw_block_dec__RawBlockDecoderInst__RawBlockDecoder_0__32_32_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "raw_block_dec_verilog_lib",
    srcs = [
        ":raw_block_dec.v",
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

rle_block_dec_codegen_args = common_codegen_args | {
    "module_name": "RleBlockDecoder",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "rle_block_dec_verilog",
    codegen_args = rle_block_dec_codegen_args,
    dslx_top = "RleBlockDecoderInst",
    library = ":rle_block_dec_dslx",
    tags = ["manual"],
    verilog_file = "rle_block_dec.v",
)

xls_benchmark_ir(
    name = "rle_block_dec_opt_ir_benchmark",
    src = ":rle_block_dec_verilog.opt.ir",
    benchmark_ir_args = rle_block_dec_codegen_args | {
        "pipeline_stages": "10",
        "top": "__rle_block_dec__RleBlockDecoderInst__RleBlockDecoder_0__64_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "rle_block_dec_verilog_lib",
    srcs = [
        ":rle_block_dec.v",
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

dec_mux_codegen_args = common_codegen_args | {
    "module_name": "DecoderMux",
    "clock_period_ps": "0",
    "pipeline_stages": "3",
}

xls_dslx_verilog(
    name = "dec_mux_verilog",
    codegen_args = dec_mux_codegen_args,
    dslx_top = "DecoderMux",
    library = ":dec_mux_dslx",
    tags = ["manual"],
    verilog_file = "dec_mux.v",
)

xls_benchmark_ir(
    name = "dec_mux_opt_ir_benchmark",
    src = ":dec_mux_verilog.opt.ir",
    benchmark_ir_args = dec_mux_codegen_args | {
        "pipeline_stages": "10",
    },
    tags = ["manual"],
)

verilog_library(
    name = "dec_mux_verilog_lib",
    srcs = [
        ":dec_mux.v",
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
    name = "sequence_executor_dslx",
    srcs = [
        "sequence_executor.x",
    ],
    deps = [
        ":common_dslx",
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

sequence_executor_codegen_args = common_codegen_args | {
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
    codegen_args = sequence_executor_codegen_args,
    dslx_top = "SequenceExecutorZstd",
    library = ":sequence_executor_dslx",
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__sequence_executor__SequenceExecutorZstd__SequenceExecutor_0__16_64_64_0_0_0_13_8192_65536_next",
    },
    tags = ["manual"],
    verilog_file = "sequence_executor.v",
)

xls_benchmark_ir(
    name = "sequence_executor_opt_ir_benchmark",
    src = ":sequence_executor_verilog.opt.ir",
    benchmark_ir_args = sequence_executor_codegen_args | {
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

axi_csr_accessor_codegen_args = common_codegen_args | {
    "module_name": "AxiCsrAccessor",
    "pipeline_stages": "1",
}

xls_dslx_verilog(
    name = "axi_csr_accessor_verilog",
    codegen_args = axi_csr_accessor_codegen_args,
    dslx_top = "AxiCsrAccessorInst",
    library = ":axi_csr_accessor_dslx",
    tags = ["manual"],
    verilog_file = "axi_csr_accessor.v",
)

xls_benchmark_ir(
    name = "axi_csr_accessor_opt_ir_benchmark",
    src = ":axi_csr_accessor_verilog.opt.ir",
    benchmark_ir_args = axi_csr_accessor_codegen_args | {
        "pipeline_stages": "10",
        "top": "__axi_csr_accessor__AxiCsrAccessorInst__AxiCsrAccessor_0__16_32_4_4_2_4_16_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "axi_csr_accessor_verilog_lib",
    srcs = [
        ":axi_csr_accessor.v",
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

csr_config_codegen_args = common_codegen_args | {
    "module_name": "CsrConfig",
    "pipeline_stages": "3",
}

xls_dslx_verilog(
    name = "csr_config_verilog",
    codegen_args = csr_config_codegen_args,
    dslx_top = "CsrConfigInst",
    library = ":csr_config_dslx",
    tags = ["manual"],
    verilog_file = "csr_config.v",
)

xls_benchmark_ir(
    name = "csr_config_opt_ir_benchmark",
    src = ":csr_config_verilog.opt.ir",
    benchmark_ir_args = csr_config_codegen_args | {
        "pipeline_stages": "10",
        "top": "__csr_config__CsrConfigInst__CsrConfig_0__2_32_4_32_2_4_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "csr_config_verilog_lib",
    srcs = [
        ":csr_config.v",
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
)

xls_dslx_verilog(
    name = "ram_rw_handler_verilog",
    codegen_args = {
        "module_name": "RamWrRespHandler",
        "delay_model": "asap7",
        "pipeline_stages": "1",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "RamWrRespHandlerInst",
    library = ":ram_wr_handler_dslx",
    opt_ir_args = {
        "top": "__ram_wr_handler__RamWrRespHandlerInst__RamWrRespHandler_0__32_next",
    },
    verilog_file = "ram_rw_handler.v",
)

xls_benchmark_ir(
    name = "ram_rw_handler_opt_ir_benchmark",
    src = ":ram_rw_handler_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "ram_rw_handler_verilog_lib",
    srcs = [
        ":ram_rw_handler.v",
    ],
)

synthesize_rtl(
    name = "ram_rw_handler_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "RamWrRespHandler",
    deps = [
        ":ram_rw_handler_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_rw_handler_benchmark_synth",
    synth_target = ":ram_rw_handler_synth_asap7",
)

place_and_route(
    name = "ram_rw_handler_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_rw_handler_synth_asap7",
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "fse_proba_freq_dec_dslx",
    srcs = ["fse_proba_freq_dec.x"],
    deps = [
        ":common_dslx",
        ":ram_wr_handler_dslx",
        ":refilling_shift_buffer_dslx",
        "//xls/modules/shift_buffer:shift_buffer_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "fse_proba_freq_dec_dslx_test",
    library = ":fse_proba_freq_dec_dslx",
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
    },
    dslx_top = "FseProbaFreqDecoderInst",
    library = ":fse_proba_freq_dec_dslx",
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__fse_proba_freq_dec__FseProbaFreqDecoderInst__FseProbaFreqDecoder_0__64_7_8_10_1_next",
    },
    verilog_file = "fse_proba_freq_dec.v",
)

xls_benchmark_ir(
    name = "fse_proba_freq_dec_opt_ir_benchmark",
    src = ":fse_proba_freq_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
        "inline_procs": "true",
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
)

xls_benchmark_verilog(
    name = "fse_proba_freq_dec_verilog_benchmark",
    verilog_target = "fse_proba_freq_dec_verilog",
)

verilog_library(
    name = "fse_proba_freq_dec_lib",
    srcs = [
        ":fse_proba_freq_dec.v",
    ],
)

synthesize_rtl(
    name = "fse_proba_freq_dec_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "FseProbaFreqDec",
    deps = [
        ":fse_proba_freq_dec_lib",
    ],
)

benchmark_synth(
    name = "fse_proba_freq_dec_benchmark_synth",
    synth_target = ":fse_proba_freq_dec_asap7",
)

place_and_route(
    name = "fse_proba_freq_dec_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "detailed_routing",
    synthesized_rtl = ":fse_proba_freq_dec_asap7",
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
)


xls_dslx_verilog(
    name = "literals_block_header_dec_verilog",
    codegen_args = window_buffer_codegen_args,
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
        "//xls/modules/zstd/memory:mem_reader_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "sequence_conf_dec_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":sequence_conf_dec_dslx",
)

xls_dslx_verilog(
    name = "sequence_conf_dec_verilog",
    codegen_args = window_buffer_codegen_args,
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

#xls_dslx_library(
#    name = "comp_block_dec_dslx",
#    srcs = ["comp_block_dec.x"],
#    deps = [
#        "//xls/examples:ram_dslx",
#        "//xls/modules/zstd/memory:mem_reader_dslx",
#        "//xls/modules/zstd/memory:axi_ram_dslx",
#        ":fse_proba_freq_dec_dslx",
#        ":literals_block_header_dec_dslx",
#    ],
#)
#
#xls_dslx_test(
#    name = "comp_block_dec_dslx_test",
#    library = ":comp_block_dec_dslx",
#)

xls_dslx_library(
    name = "refilling_shift_buffer_dslx",
    srcs = ["refilling_shift_buffer.x"],
    deps = [
        "//xls/modules/shift_buffer:shift_buffer_dslx",
        "//xls/modules/zstd/memory:mem_reader_dslx",
    ],
)

xls_dslx_test(
    name = "refilling_shift_buffer_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":refilling_shift_buffer_dslx",
)

xls_dslx_verilog(
    name = "refilling_shift_buffer_internal_verilog",
    codegen_args = common_codegen_args | {
        "module_name": "RefillingShiftBufferInternalInst"
    },
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
        "inline_procs": "false",
    },
    tags = ["manual"],
)

verilog_library(
    name = "refilling_shift_buffer_internal_verilog_lib",
    srcs = [
        ":refilling_shift_buffer_internal.v",
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

#xls_dslx_verilog(
#    name = "refilling_shift_buffer_verilog",
#    codegen_args = common_codegen_args,
#    dslx_top = "RefillingShiftBufferInst",
#    library = ":refilling_shift_buffer_dslx",
#    tags = ["manual"],
#    verilog_file = "refilling_shift_buffer.v",
#)

py_binary(
    name = "zstd_test_frames_generator",
    srcs = ["zstd_frame_dslx.py"],
    imports = ["."],
    main = "zstd_frame_dslx.py",
    tags = ["manual"],
    visibility = ["//xls:xls_users"],
    deps = [
        requirement("zstandard"),
        "//xls/common:runfiles",
        "//xls/modules/zstd/cocotb:data_generator",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

genrule(
    name = "zstd_test_frames_generate",
    srcs = [],
    outs = ["zstd_frame_testcases.x"],
    cmd = "$(location :zstd_test_frames_generator) -n 2 --btype RAW RLE -o $@",
    tools = [":zstd_test_frames_generator"],
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
    ":sequence_executor_dslx",
    "//xls/examples:ram_dslx",
    "//xls/modules/zstd/memory:mem_reader_dslx",
    "//xls/modules/zstd/memory:mem_writer_dslx",
    "//xls/modules/zstd/memory:axi_ram_dslx",
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
    srcs = [
        "zstd_dec.x",
        "zstd_dec_test.x",
        "zstd_frame_testcases.x",
    ],
    tags = ["manual"],
    deps = zstd_dec_deps,
)

zstd_dec_codegen_args = common_codegen_args | {
    "module_name": "ZstdDecoder",
    "clock_period_ps": "0",
    "pipeline_stages": "10",
    "flop_inputs_kind": "skid",
    "flop_outputs_kind": "skid",
}

xls_dslx_verilog(
    name = "zstd_dec_verilog",
    codegen_args = zstd_dec_codegen_args,
    dslx_top = "ZstdDecoderInst",
    library = ":zstd_dec_dslx",
    tags = ["manual"],
    verilog_file = "zstd_dec.v",
)

zstd_dec_internal_codegen_args = common_codegen_args | {
    "module_name": "ZstdDecoderInternal",
    "pipeline_stages": "2",
}

xls_dslx_verilog(
    name = "zstd_dec_internal_verilog",
    codegen_args = zstd_dec_internal_codegen_args,
    dslx_top = "ZstdDecoderInternalInst",
    library = ":zstd_dec_dslx",
    tags = ["manual"],
    verilog_file = "zstd_dec_internal.v",
)

xls_benchmark_ir(
    name = "zstd_dec_internal_opt_ir_benchmark",
    src = ":zstd_dec_internal_verilog.opt.ir",
    benchmark_ir_args = {
        "top": "__zstd_dec__ZstdDecoderInternalInst__ZstdDecoderInternal_0__16_64_8_4_16_next",
        "pipeline_stages": "10",
    },
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

verilog_library(
    name = "zstd_dec_verilog_lib",
    srcs = [
        ":xls_fifo_wrapper.v",
        ":zstd_dec.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "zstd_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "ZstdDecoder",
    deps = [
        ":zstd_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "zstd_dec_benchmark_synth",
    synth_target = ":zstd_dec_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "zstd_dec_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.4",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":zstd_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

py_test(
    name = "zstd_dec_cocotb_test",
    srcs = ["zstd_dec_cocotb_test.py"],
    data = [
        ":xls_fifo_wrapper.v",
        ":zstd_dec.v",
        ":zstd_dec_wrapper.v",
        "//xls/modules/zstd/external:arbiter.v",
        "//xls/modules/zstd/external:axi_crossbar.v",
        "//xls/modules/zstd/external:axi_crossbar_addr.v",
        "//xls/modules/zstd/external:axi_crossbar_rd.v",
        "//xls/modules/zstd/external:axi_crossbar_wr.v",
        "//xls/modules/zstd/external:axi_crossbar_wrapper.v",
        "//xls/modules/zstd/external:axi_register_rd.v",
        "//xls/modules/zstd/external:axi_register_wr.v",
        "//xls/modules/zstd/external:priority_encoder.v",
        "@com_icarus_iverilog//:iverilog",
        "@com_icarus_iverilog//:vvp",
    ],
    env = {"BUILD_WORKING_DIRECTORY": "sim_build"},
    imports = ["."],
    tags = ["manual"],
    visibility = ["//xls:xls_users"],
    deps = [
        requirement("cocotb"),
        requirement("cocotbext-axi"),
        requirement("pytest"),
        requirement("zstandard"),
        "//xls/common:runfiles",
        "//xls/modules/zstd/cocotb:channel",
        "//xls/modules/zstd/cocotb:data_generator",
        "//xls/modules/zstd/cocotb:memory",
        "//xls/modules/zstd/cocotb:utils",
        "//xls/modules/zstd/cocotb:xlsstruct",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

xls_dslx_library(
    name = "fse_lookup_dec_dslx",
    srcs = [
        "fse_lookup_dec.x",
    ],
    deps = [
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd/memory:axi_dslx",
        "//xls/modules/zstd/memory:axi_ram_dslx",
        "//xls/modules/zstd:common_dslx",
        "//xls/modules/zstd:fse_table_creator_dslx",
        "//xls/modules/zstd:refilling_shift_buffer_dslx",
        "//xls/modules/zstd:fse_proba_freq_dec_dslx",
        "//xls/modules/shift_buffer:shift_buffer_dslx",
    ]
)

xls_dslx_test(
    name = "fse_lookup_dec_dslx_test",
    library = ":fse_lookup_dec_dslx",
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
)

xls_dslx_verilog(
    name = "fse_table_iterator_verilog",
    codegen_args = {
        "module_name": "FseTableIterator",
        "delay_model": "asap7",
        "pipeline_stages": "1",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "FseTableIterator",
    library = ":fse_table_iterator_dslx",
    verilog_file = "fse_table_iterator.v",
)

xls_benchmark_ir(
    name = "fse_table_iterator_opt_ir_benchmark",
    src = ":fse_table_iterator_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "fse_table_iterator_verilog_lib",
    srcs = [
        ":fse_table_iterator.v",
    ],
)

synthesize_rtl(
    name = "fse_table_iterator_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "FseTableIterator",
    deps = [
        ":fse_table_iterator_verilog_lib",
    ],
)

benchmark_synth(
    name = "fse_table_iterator_benchmark_synth",
    synth_target = ":fse_table_iterator_synth_asap7",
)

place_and_route(
    name = "fse_table_iterator_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":fse_table_iterator_synth_asap7",
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
)

xls_dslx_verilog(
    name = "fse_table_creator_verilog",
    codegen_args = {
        "module_name": "FseTableCreator",
        "delay_model": "asap7",
        "pipeline_stages": "4",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "FseTableCreatorInst",
    library = ":fse_table_creator_dslx",
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__fse_table_creator__FseTableCreatorInst__FseTableCreator_0__8_16_1_8_32_4_8_16_1_next",
    },
    verilog_file = "fse_table_creator.v",
    tags = ["manual"],
)

xls_benchmark_ir(
    name = "fse_table_creator_opt_ir_benchmark",
    src = ":fse_table_creator_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
    },
)

xls_benchmark_verilog(
    name = "fse_table_creator_verilog_benchmark",
    verilog_target = "fse_table_creator_verilog",
)

verilog_library(
    name = "fse_table_creator_lib",
    srcs = [
        ":fse_table_creator.v",
    ],
)

synthesize_rtl(
    name = "fse_table_creator_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "FseTableCreator",
    deps = [
        ":fse_table_creator_lib",
    ],
)

benchmark_synth(
    name = "fse_table_creator_benchmark_synth",
    synth_target = ":fse_table_creator_asap7",
)

place_and_route(
    name = "fse_table_creator_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":fse_table_creator_asap7",
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
)

xls_dslx_verilog(
    name = "command_constructor_verilog",
    codegen_args = {
        "module_name": "CommandConstructor",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "CommandConstructor",
    library = ":command_constructor_dslx",
    verilog_file = "command_constructor.v",
)

xls_benchmark_ir(
    name = "command_constructor_opt_ir_benchmark",
    src = ":command_constructor_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "8",
        "delay_model": "asap7",
    },
)

xls_benchmark_verilog(
    name = "command_constructor_verilog_benchmark",
    verilog_target = "command_constructor_verilog",
)

verilog_library(
    name = "command_constructor_lib",
    srcs = [
        ":command_constructor.v",
    ],
)

synthesize_rtl(
    name = "command_constructor_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "CommandConstructor",
    deps = [
        ":command_constructor_lib",
    ],
)

benchmark_synth(
    name = "command_constructor_benchmark_synth",
    synth_target = ":command_constructor_asap7",
)

place_and_route(
    name = "command_constructor_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.3",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":command_constructor_asap7",
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "ram_demux_dslx",
    srcs = ["ram_demux.x"],
    deps = [
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "ram_demux_dslx_test",
    dslx_test_args = {"compare": "none"},
    library = ":ram_demux_dslx",
)

xls_dslx_verilog(
    name = "ram_demux_verilog",
    codegen_args = {
        "module_name": "RamDemux",
        "generator": "pipeline",
        "delay_model": "asap7",
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
        "pipeline_stages": "6",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "RamDemuxInst",
    library = ":ram_demux_dslx",
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__ram_demux__RamDemuxInst__RamDemux_0__5_8_0_8_5_next",
    },
    verilog_file = "ram_demux.v",
)

xls_benchmark_ir(
    name = "ram_demux_opt_ir_benchmark",
    src = "ram_demux_verilog.opt.ir",
    benchmark_ir_args = {
        "top": "__ram_demux__RamDemuxInst__RamDemux_0__5_8_0_8_5_next",
    },
    codegen_args = {
        "pipeline_stages": "10",
    },
)

verilog_library(
    name = "ram_demux_verilog_lib",
    srcs = [
        ":ram_demux.v",
    ],
)

synthesize_rtl(
    name = "ram_demux_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "RamDemux",
    deps = [
        ":ram_demux_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_demux_benchmark_synth",
    synth_target = ":ram_demux_synth_asap7",
)

place_and_route(
    name = "ram_demux_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_demux_synth_asap7",
    target_die_utilization_percentage = "5",
)

xls_dslx_verilog(
    name = "ram_demux_naive_verilog",
    codegen_args = {
        "module_name": "RamDemuxNaive",
        "generator": "pipeline",
        "delay_model": "asap7",
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
        "pipeline_stages": "6",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "RamDemuxNaiveInst",
    library = ":ram_demux_dslx",
    opt_ir_args = {
        "top": "__ram_demux__RamDemuxNaiveInst__RamDemuxNaive_0__5_8_0_8_next",
    },
    verilog_file = "ram_demux_naive.v",
)

xls_benchmark_ir(
    name = "ram_demux_naive_opt_ir_benchmark",
    src = "ram_demux_naive_verilog.opt.ir",
    benchmark_ir_args = {
        "top": "__ram_demux__RamDemuxNaiveInst__RamDemuxNaive_0__5_8_0_8_next",
    },
    codegen_args = {
        "pipeline_stages": "10",
    },
)

verilog_library(
    name = "ram_demux_naive_verilog_lib",
    srcs = [
        ":ram_demux_naive.v",
    ],
)

synthesize_rtl(
    name = "ram_demux_naive_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "RamDemuxNaive",
    deps = [
        ":ram_demux_naive_verilog_lib",
    ],
)

benchmark_synth(
    name = "ram_demux_naive_benchmark_synth",
    synth_target = ":ram_demux_naive_synth_asap7",
)

place_and_route(
    name = "ram_demux_naive_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":ram_demux_naive_synth_asap7",
    target_die_utilization_percentage = "5",
)

xls_dslx_library(
    name = "fse_dec_dslx",
    srcs = [
        "fse_dec.x",
    ],
    deps = [
        "//xls/examples:ram_dslx",
        ":common_dslx",
        ":math_dslx",
        ":fse_table_creator_dslx",
        "//xls/modules/zstd:refilling_shift_buffer_dslx",
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
    },
    dslx_top = "FseDecoderInst",
    opt_ir_args = {
        "top": "__fse_dec__FseDecoderInst__FseDecoder_0__64_8_32_4_64_7_next"
    },
    library = ":fse_dec_dslx",
    tags = ["manual"],
    verilog_file = "fse_dec.v",
)

xls_benchmark_ir(
    name = "fse_dec_opt_ir_benchmark",
    src = ":fse_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "delay_model": "asap7",
        "pipeline_stages": "3",
        "inline_procs": "false"
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
    min_pin_distance = "0.5",
    placement_density = "0.30",
    die_height_microns = 100,
    die_width_microns = 100,
    stop_after_step = "global_routing",
    synthesized_rtl = ":fse_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

