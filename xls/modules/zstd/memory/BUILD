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

# pytype binary, library
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
load("//xls/common:openroad_warnings.bzl", "ASAP7_SUPPPRESSED_WARNINGS")

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

common_codegen_args = {
    "delay_model": "asap7",
    "reset": "rst",
    "worst_case_throughput": "1",
    "use_system_verilog": "false",
    "clock_period_ps": CLOCK_PERIOD_PS,
    "clock_margin_percent": "20",
    "multi_proc": "true",
    "streaming_channel_data_suffix": "_data",
    "fifo_module": "",
    "materialize_internal_fifos": "true",

    # TODO: This should be adjusted when per channel separation of IO options is enabled
    "flop_inputs_kind": "skid",
    "flop_outputs_kind": "skid",
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

# FIXME: Improve the proc to achieve CLOCK_PERIOD_PS
AXI_READER_CLOCK_PERIOD_PS = "1700"

axi_reader_codegen_args = common_codegen_args | {
    "clock_period_ps": AXI_READER_CLOCK_PERIOD_PS,
    "module_name": "axi_reader",
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
    benchmark_ir_args = axi_reader_codegen_args,
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
    clock_period = AXI_READER_CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
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

xls_dslx_verilog(
    name = "axi_stream_remove_empty_verilog",
    codegen_args = common_codegen_args | {"module_name": "axi_stream_remove_empty"},
    dslx_top = "AxiStreamRemoveEmptyInst",
    library = ":axi_stream_remove_empty_dslx",
    tags = ["manual"],
    verilog_file = "axi_stream_remove_empty.v",
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
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":axi_stream_remove_empty_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_verilog(
    name = "remove_empty_bytes_verilog",
    codegen_args = common_codegen_args | {"module_name": "remove_empty_bytes"},
    dslx_top = "RemoveEmptyBytesInst",
    library = ":axi_stream_remove_empty_dslx",
    tags = ["manual"],
    verilog_file = "remove_empty_bytes.v",
)

xls_benchmark_ir(
    name = "remove_empty_bytes_opt_ir_benchmark",
    src = ":remove_empty_bytes_verilog.opt.ir",
    benchmark_ir_args = common_codegen_args,
    tags = ["manual"],
)

verilog_library(
    name = "remove_empty_bytes_verilog_lib",
    srcs = [
        ":remove_empty_bytes.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "remove_empty_bytes_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "remove_empty_bytes",
    deps = [
        ":remove_empty_bytes_verilog_lib",
    ],
)

benchmark_synth(
    name = "remove_empty_bytes_benchmark_synth",
    synth_target = ":remove_empty_bytes_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "remove_empty_bytes_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":remove_empty_bytes_synth_asap7",
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

xls_dslx_verilog(
    name = "axi_stream_downscaler_verilog",
    codegen_args = common_codegen_args | {"module_name": "axi_stream_downscaler"},
    dslx_top = "AxiStreamDownscalerInst",
    library = ":axi_stream_downscaler_dslx",
    tags = ["manual"],
    verilog_file = "axi_stream_downscaler.v",
)

xls_benchmark_ir(
    name = "axi_stream_downscaler_opt_ir_benchmark",
    src = ":axi_stream_downscaler_verilog.opt.ir",
    benchmark_ir_args = common_codegen_args,
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
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":axi_stream_downscaler_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "axi_ram_reader_dslx",
    srcs = ["axi_ram_reader.x"],
    deps = [
        ":axi_dslx",
        "//xls/examples:ram_dslx",
        "//xls/modules/zstd:math_dslx",
    ],
)

xls_dslx_test(
    name = "axi_ram_reader_dslx_test",
    library = ":axi_ram_reader_dslx",
)

# FIXME: Improve the proc to achieve CLOCK_PERIOD_PS
AXI_RAM_READER_CLOCK_PERIOD_PS = "850"

axi_ram_reader_codegen_args = common_codegen_args | {
    "module_name": "axi_ram_reader",
    "clock_period_ps": AXI_RAM_READER_CLOCK_PERIOD_PS,
    "ram_configurations": "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
        latency = 5,
        ram_name = "ram",
        rd_req = "axi_ram_reader__rd_req_s",
        rd_resp = "axi_ram_reader__rd_resp_r",
        wr_req = "axi_ram_reader__wr_req_s",
        wr_resp = "axi_ram_reader__wr_resp_r",
    ),
}

xls_dslx_verilog(
    name = "axi_ram_reader_verilog",
    codegen_args = axi_ram_reader_codegen_args,
    dslx_top = "AxiRamReaderInstWithEmptyWrites",
    library = ":axi_ram_reader_dslx",
    tags = ["manual"],
    verilog_file = "axi_ram_reader.v",
)

verilog_library(
    name = "axi_ram_reader_verilog_lib",
    srcs = [
        ":axi_ram_reader.v",
    ],
    tags = ["manual"],
)

xls_benchmark_ir(
    name = "axi_ram_reader_opt_ir_benchmark",
    src = ":axi_ram_reader_verilog.opt.ir",
    benchmark_ir_args = axi_ram_reader_codegen_args,
    tags = ["manual"],
)

synthesize_rtl(
    name = "axi_ram_reader_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "axi_ram_reader",
    deps = [
        ":axi_ram_reader_verilog_lib",
    ],
)

benchmark_synth(
    name = "axi_ram_reader_benchmark_synth",
    synth_target = ":axi_ram_reader_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "axi_ram_reader_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":axi_ram_reader_synth_asap7",
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

# FIXME: Improve the proc to achieve CLOCK_PERIOD_PS
MEM_READER_CLOCK_PERIOD_PS = "2600"

mem_reader_codegen_args = common_codegen_args | {
    "clock_period_ps": MEM_READER_CLOCK_PERIOD_PS,
    "module_name": "mem_reader",
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

xls_benchmark_ir(
    name = "mem_reader_opt_ir_benchmark",
    src = ":mem_reader_verilog.opt.ir",
    benchmark_ir_args = mem_reader_codegen_args,
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
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":mem_reader_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_verilog(
    name = "mem_reader_adv_verilog",
    codegen_args = mem_reader_codegen_args | {"module_name": "mem_reader_adv"},
    dslx_top = "MemReaderAdvInst",
    library = ":mem_reader_dslx",
    tags = ["manual"],
    verilog_file = "mem_reader_adv.v",
)

xls_benchmark_ir(
    name = "mem_reader_adv_opt_ir_benchmark",
    src = ":mem_reader_adv_verilog.opt.ir",
    benchmark_ir_args = mem_reader_codegen_args,
    tags = ["manual"],
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
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":mem_reader_adv_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

py_test(
    name = "mem_reader_cocotb_test",
    srcs = ["mem_reader_cocotb_test.py"],
    data = [
        ":mem_reader_adv.v",
        ":mem_reader_wrapper.v",
        "@com_icarus_iverilog//:iverilog",
        "@com_icarus_iverilog//:vvp",
    ],
    env = {"BUILD_WORKING_DIRECTORY": "sim_build"},
    tags = ["manual"],
    visibility = ["//xls:xls_users"],
    deps = [
        requirement("cocotb"),
        requirement("cocotbext-axi"),
        requirement("pytest"),
        "//xls/common:runfiles",
        "//xls/modules/zstd/cocotb:channel",
        "//xls/modules/zstd/cocotb:memory",
        "//xls/modules/zstd/cocotb:utils",
        "//xls/modules/zstd/cocotb:xlsstruct",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

xls_dslx_library(
    name = "axi_writer_dslx",
    srcs = ["axi_writer.x"],
    deps = [
        ":axi_dslx",
        ":axi_st_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "axi_writer_dslx_test",
    library = ":axi_writer_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "axi_writer_verilog",
    codegen_args = common_codegen_args | {"module_name": "axi_writer"},
    dslx_top = "AxiWriterInst",
    library = ":axi_writer_dslx",
    tags = ["manual"],
    verilog_file = "axi_writer.v",
)

xls_benchmark_ir(
    name = "axi_writer_opt_ir_benchmark",
    src = ":axi_writer_verilog.opt.ir",
    benchmark_ir_args = common_codegen_args,
    tags = ["manual"],
)

verilog_library(
    name = "axi_writer_verilog_lib",
    srcs = [
        ":axi_writer.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "axi_writer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "axi_writer",
    deps = [
        ":axi_writer_verilog_lib",
    ],
)

benchmark_synth(
    name = "axi_writer_benchmark_synth",
    synth_target = ":axi_writer_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "axi_writer_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":axi_writer_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

py_test(
    name = "axi_writer_cocotb_test",
    srcs = ["axi_writer_cocotb_test.py"],
    data = [
        ":axi_writer.v",
        ":axi_writer_wrapper.v",
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
        "//xls/common:runfiles",
        "//xls/modules/zstd/cocotb:channel",
        "//xls/modules/zstd/cocotb:utils",
        "//xls/modules/zstd/cocotb:xlsstruct",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

xls_dslx_library(
    name = "axi_stream_add_empty_dslx",
    srcs = ["axi_stream_add_empty.x"],
    deps = [
        ":axi_dslx",
        ":axi_st_dslx",
        ":axi_writer_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "axi_stream_add_empty_dslx_test",
    library = ":axi_stream_add_empty_dslx",
)

axi_stream_add_empty_codegen_args = common_codegen_args | {
    "module_name": "axi_stream_add_empty",
    "pipeline_stages": "2",
    "streaming_channel_data_suffix": "_data",
}

xls_dslx_verilog(
    name = "axi_stream_add_empty_verilog",
    codegen_args = axi_stream_add_empty_codegen_args,
    dslx_top = "AxiStreamAddEmptyInst",
    library = ":axi_stream_add_empty_dslx",
    tags = ["manual"],
    verilog_file = "axi_stream_add_empty.v",
)

xls_benchmark_ir(
    name = "axi_stream_add_empty_opt_ir_benchmark",
    src = ":axi_stream_add_empty_verilog.opt.ir",
    benchmark_ir_args = axi_stream_add_empty_codegen_args,
    tags = ["manual"],
)

verilog_library(
    name = "axi_stream_add_empty_verilog_lib",
    srcs = [
        ":axi_stream_add_empty.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "axi_stream_add_empty_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "axi_stream_add_empty",
    deps = [
        ":axi_stream_add_empty_verilog_lib",
    ],
)

benchmark_synth(
    name = "axi_stream_add_empty_benchmark_synth",
    synth_target = ":axi_stream_add_empty_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "axi_stream_add_empty_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":axi_stream_add_empty_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "mem_writer_dslx",
    srcs = ["mem_writer.x"],
    deps = [
        ":axi_dslx",
        ":axi_st_dslx",
        ":axi_stream_add_empty_dslx",
        ":axi_stream_remove_empty_dslx",
        ":axi_writer_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "mem_writer_dslx_test",
    library = ":mem_writer_dslx",
)

xls_dslx_verilog(
    name = "mem_writer_verilog",
    codegen_args = common_codegen_args | {"module_name": "mem_writer"},
    dslx_top = "MemWriterInst",
    library = ":mem_writer_dslx",
    tags = ["manual"],
    verilog_file = "mem_writer.v",
)

verilog_library(
    name = "mem_writer_verilog_lib",
    srcs = [
        ":mem_writer.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "mem_writer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "mem_writer",
    deps = [
        ":mem_writer_verilog_lib",
    ],
)

benchmark_synth(
    name = "mem_writer_benchmark_synth",
    synth_target = ":mem_writer_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "mem_writer_place_and_route",
    clock_period = CLOCK_PERIOD_PS,
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    suppress_warnings = ASAP7_SUPPPRESSED_WARNINGS,
    synthesized_rtl = ":mem_writer_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

py_test(
    name = "mem_writer_cocotb_test",
    srcs = ["mem_writer_cocotb_test.py"],
    data = [
        ":mem_writer.v",
        ":mem_writer_wrapper.v",
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
        "//xls/common:runfiles",
        "//xls/modules/zstd/cocotb:channel",
        "//xls/modules/zstd/cocotb:utils",
        "//xls/modules/zstd/cocotb:xlsstruct",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)
