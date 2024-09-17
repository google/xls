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

load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "benchmark_synth", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_dslx_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "axi_dslx",
    srcs = ["axi.x"],
)

xls_dslx_library(
    name = "axi_st_dslx",
    srcs = ["axi_st.x"],
)

xls_dslx_library(
    name = "common_dslx",
    srcs = [
        "common.x",
    ],
    deps = [":axi_dslx"],
)

xls_dslx_test(
    name = "common_dslx_test",
    library = ":common_dslx",
    tags = ["manual"],
)

CLOCK_PERIOD_PS = "750"
# Clock periods for modules that exceed the 750ps critical path in IR benchmark
AXI_READER_CLOCK_PERIOD_PS = "1800"
AXI_STREAM_REMOVE_EMPTY_CLOCK_PERIOD_PS = "1300"
MEM_READER_CLOCK_PERIOD_PS = "2600"

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
    name = "axi_reader_dslx",
    srcs = ["axi_reader.x"],
    deps = [
        ":axi_dslx",
        ":axi_st_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "axi_reader_dslx_test",
    library = ":axi_reader_dslx",
    tags = ["manual"],
)

axi_reader_codegen_args = common_codegen_args | {
    "module_name": "axi_reader",
    "pipeline_stages": "2",
    "streaming_channel_data_suffix": "_data",
    "flop_inputs_kind": "skid",
    "flop_outputs_kind": "skid",
    "clock_period_ps": AXI_READER_CLOCK_PERIOD_PS,
}

xls_dslx_verilog(
    name = "axi_reader_verilog",
    codegen_args = axi_reader_codegen_args,
    dslx_top = "AxiReaderInst",
    library = ":axi_reader_dslx",
    tags = ["manual"],
    verilog_file = "axi_reader.v",
)

xls_benchmark_ir(
    name = "axi_reader_opt_ir_benchmark",
    src = ":axi_reader_verilog.opt.ir",
    benchmark_ir_args = axi_reader_codegen_args | {
        "pipeline_stages": "10",
        "top": "__axi_reader__AxiReaderInst__AxiReader_0__16_32_4_4_4_3_2_14_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "axi_reader_verilog_lib",
    srcs = [
        ":axi_reader.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "axi_reader_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "axi_reader",
    deps = [
        ":axi_reader_verilog_lib",
    ],
)

benchmark_synth(
    name = "axi_reader_benchmark_synth",
    synth_target = ":axi_reader_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "axi_reader_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":axi_reader_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "axi_stream_remove_empty_dslx",
    srcs = ["axi_stream_remove_empty.x"],
    deps = [
        ":axi_st_dslx",
    ],
)

xls_dslx_test(
    name = "axi_stream_remove_empty_dslx_test",
    library = ":axi_stream_remove_empty_dslx",
    tags = ["manual"],
)

axi_stream_remove_empty_codegen_args = common_codegen_args | {
    "module_name": "axi_stream_remove_empty",
    "clock_period_ps": AXI_STREAM_REMOVE_EMPTY_CLOCK_PERIOD_PS,
    "pipeline_stages": "2",
}

xls_dslx_verilog(
    name = "axi_stream_remove_empty_verilog",
    codegen_args = axi_stream_remove_empty_codegen_args,
    dslx_top = "AxiStreamRemoveEmptyInst",
    library = ":axi_stream_remove_empty_dslx",
    tags = ["manual"],
    verilog_file = "axi_stream_remove_empty.v",
)

xls_benchmark_ir(
    name = "axi_stream_remove_empty_opt_ir_benchmark",
    src = ":axi_stream_remove_empty_verilog.opt.ir",
    benchmark_ir_args = axi_stream_remove_empty_codegen_args | {
        "pipeline_stages": "10",
        "top": "__axi_stream_remove_empty__AxiStreamRemoveEmptyInst__AxiStreamRemoveEmpty_0__32_4_6_32_32_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "axi_stream_remove_empty_verilog_lib",
    srcs = [
        ":axi_stream_remove_empty.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "axi_stream_remove_empty_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "axi_stream_remove_empty",
    deps = [
        ":axi_stream_remove_empty_verilog_lib",
    ],
)

benchmark_synth(
    name = "axi_stream_remove_empty_benchmark_synth",
    synth_target = ":axi_stream_remove_empty_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "axi_stream_remove_empty_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":axi_stream_remove_empty_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "axi_stream_downscaler_dslx",
    srcs = ["axi_stream_downscaler.x"],
    deps = [
        ":axi_st_dslx",
    ],
)

xls_dslx_test(
    name = "axi_stream_downscaler_dslx_test",
    library = ":axi_stream_downscaler_dslx",
    tags = ["manual"],
)

axi_stream_downscaler_codegen_args = common_codegen_args | {
    "module_name": "axi_stream_downscaler",
    "pipeline_stages": "2",
}

xls_dslx_verilog(
    name = "axi_stream_downscaler_verilog",
    codegen_args = axi_stream_downscaler_codegen_args,
    dslx_top = "AxiStreamDownscalerInst",
    library = ":axi_stream_downscaler_dslx",
    tags = ["manual"],
    verilog_file = "axi_stream_downscaler.v",
)

xls_benchmark_ir(
    name = "axi_stream_downscaler_opt_ir_benchmark",
    src = ":axi_stream_downscaler_verilog.opt.ir",
    benchmark_ir_args = axi_stream_downscaler_codegen_args | {
        "pipeline_stages": "10",
        "top": "__axi_stream_downscaler__AxiStreamDownscalerInst__AxiStreamDownscaler_0__8_8_128_16_32_4_4_3_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "axi_stream_downscaler_verilog_lib",
    srcs = [
        ":axi_stream_downscaler.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "axi_stream_downscaler_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "axi_stream_downscaler",
    deps = [
        ":axi_stream_downscaler_verilog_lib",
    ],
)

benchmark_synth(
    name = "axi_stream_downscaler_benchmark_synth",
    synth_target = ":axi_stream_downscaler_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "axi_stream_downscaler_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":axi_stream_downscaler_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "mem_reader_dslx",
    srcs = ["mem_reader.x"],
    deps = [
        ":axi_dslx",
        ":axi_reader_dslx",
        ":axi_st_dslx",
        ":axi_stream_downscaler_dslx",
        ":axi_stream_remove_empty_dslx",
    ],
)

xls_dslx_test(
    name = "mem_reader_dslx_test",
    library = ":mem_reader_dslx",
    tags = ["manual"],
)

mem_reader_internal_codegen_args = common_codegen_args | {
    "module_name": "mem_reader_internal",
    "pipeline_stages": "10",
}

xls_dslx_verilog(
    name = "mem_reader_internal_verilog",
    codegen_args = mem_reader_internal_codegen_args,
    dslx_top = "MemReaderInternalInst",
    library = ":mem_reader_dslx",
    tags = ["manual"],
    verilog_file = "mem_reader_internal.v",
)

xls_benchmark_ir(
    name = "mem_reader_internal_opt_ir_benchmark",
    src = ":mem_reader_internal_verilog.opt.ir",
    benchmark_ir_args = common_codegen_args | {
        "pipeline_stages": "10",
        "top": "__mem_reader__MemReaderInternalInst__MemReaderInternal_0__16_128_16_8_8_2_2_16_64_8_next",
    },
    tags = ["manual"],
)

verilog_library(
    name = "mem_reader_internal_verilog_lib",
    srcs = [
        ":mem_reader_internal.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "mem_reader_internal_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "mem_reader_internal",
    deps = [
        ":mem_reader_internal_verilog_lib",
    ],
)

benchmark_synth(
    name = "mem_reader_internal_benchmark_synth",
    synth_target = ":mem_reader_internal_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "mem_reader_internal_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":mem_reader_internal_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

mem_reader_codegen_args = common_codegen_args | {
    "module_name": "mem_reader",
    "pipeline_stages": "4",
    "streaming_channel_data_suffix": "_data",
    "flop_inputs_kind": "skid",
    "flop_outputs_kind": "skid",
    "clock_period_ps": MEM_READER_CLOCK_PERIOD_PS,
}

xls_dslx_verilog(
    name = "mem_reader_verilog",
    codegen_args = mem_reader_codegen_args,
    dslx_top = "MemReaderInst",
    library = ":mem_reader_dslx",
    tags = ["manual"],
    verilog_file = "mem_reader.v",
)

verilog_library(
    name = "mem_reader_verilog_lib",
    srcs = [
        ":mem_reader.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "mem_reader_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "mem_reader",
    deps = [
        ":mem_reader_verilog_lib",
    ],
)

benchmark_synth(
    name = "mem_reader_benchmark_synth",
    synth_target = ":mem_reader_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "mem_reader_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":mem_reader_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

mem_reader_adv_codegen_args = common_codegen_args | {
    "module_name": "mem_reader_adv",
    "pipeline_stages": "4",
    "streaming_channel_data_suffix": "_data",
    "flop_inputs_kind": "skid",
    "flop_outputs_kind": "skid",
    "clock_period_ps": MEM_READER_CLOCK_PERIOD_PS,
}

xls_dslx_verilog(
    name = "mem_reader_adv_verilog",
    codegen_args = mem_reader_adv_codegen_args,
    dslx_top = "MemReaderAdvInst",
    library = ":mem_reader_dslx",
    tags = ["manual"],
    verilog_file = "mem_reader_adv.v",
)

verilog_library(
    name = "mem_reader_adv_verilog_lib",
    srcs = [
        ":mem_reader_adv.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "mem_reader_adv_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "mem_reader_adv",
    deps = [
        ":mem_reader_adv_verilog_lib",
    ],
)

benchmark_synth(
    name = "mem_reader_adv_benchmark_synth",
    synth_target = ":mem_reader_adv_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "mem_reader_adv_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":mem_reader_adv_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)
