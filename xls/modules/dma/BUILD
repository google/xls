# Copyright 2023 The XLS Authors
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
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
    "xls_dslx_fmt_test",
    "xls_dslx_ir",
    "xls_dslx_library",
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

# Common
xls_dslx_library(
    name = "dma_common",
    srcs = [
        "bus/axi_pkg.x",
        "bus/axi_st_pkg.x",
        "common.x",
        "config.x",
        "gpf.x",
    ],
)

xls_dslx_test(
    name = "test_common",
    library = "dma_common",
)

# CSR
xls_dslx_library(
    name = "csr_lib",
    srcs = [
        "csr.x",
    ],
    deps = [
        ":dma_common",
    ],
)

xls_dslx_test(
    name = "test_csr",
    library = "csr_lib",
)

xls_dslx_verilog(
    name = "verilog_csr",
    codegen_args = {
        "module_name": "csr",
        "delay_model": "asap7",
        "pipeline_stages": "4",
        "worst_case_throughput": "4",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "csr",
    library = "csr_lib",
    opt_ir_args = {
        "top": "__csr__csr__Csr_0__8_32_14_next",
    },
    verilog_file = "csr.v",
)

xls_benchmark_ir(
    name = "csr_opt_ir_benchmark",
    src = ":verilog_csr.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "4",
        "worst_case_throughput": "4",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "verilog_csr_lib",
    srcs = [
        ":csr.v",
    ],
)

synthesize_rtl(
    name = "csr_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "csr",
    deps = [
        ":verilog_csr_lib",
    ],
)

benchmark_synth(
    name = "csr_benchmark_synth",
    synth_target = ":csr_synth_asap7",
)

place_and_route(
    name = "csr_place_and_route",
    clock_period = "2000",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":csr_synth_asap7",
    target_die_utilization_percentage = "10",
)

# AXI CSR
xls_dslx_library(
    name = "axi_csr_lib",
    srcs = [
        "axi_csr.x",
    ],
    deps = [
        ":csr_lib",
        ":dma_common",
    ],
)

xls_dslx_test(
    name = "test_axi_csr",
    library = "axi_csr_lib",
)

xls_dslx_verilog(
    name = "verilog_axi_csr",
    codegen_args = {
        "module_name": "axi_csr",
        "delay_model": "asap7",
        "pipeline_stages": "4",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "axi_csr",
    library = "axi_csr_lib",
    opt_ir_args = {
        "top": "__axi_csr__axi_csr__AxiCsr_0__32_32_4_14_4_next",
    },
    verilog_file = "axi_csr.v",
)

xls_benchmark_ir(
    name = "axi_csr_opt_ir_benchmark",
    src = ":verilog_axi_csr.opt.ir",
    benchmark_ir_args = {
        "delay_model": "asap7",
        "pipeline_stages": "4",
    },
)

verilog_library(
    name = "verilog_axi_csr_lib",
    srcs = [
        ":axi_csr.v",
    ],
)

synthesize_rtl(
    name = "axi_csr_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "axi_csr",
    deps = [
        ":verilog_axi_csr_lib",
    ],
)

benchmark_synth(
    name = "axi_csr_benchmark_synth",
    synth_target = ":axi_csr_synth_asap7",
)

place_and_route(
    name = "axi_csr_place_and_route",
    clock_period = "2000",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":axi_csr_synth_asap7",
    target_die_utilization_percentage = "10",
)

# FIFO
xls_dslx_library(
    name = "fifo",
    srcs = [
        "fifo.x",
    ],
    deps = [
        ":dma_common",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "test_fifo",
    library = "fifo",
)

# xls_dslx_ir(
#     name = "fifo_ir",
#     dslx_top = "fifo_synth",
#     ir_file = "fifo_ir.ir",
#     library = "fifo",
# )

# xls_ir_opt_ir(
#     name = "fifo_ir_opt",
#     src = "fifo_ir.ir",
#     # FIXME: Top level is not correctly generated in verilog
#     top = "__fifo__fifo_synth__FIFO__Writer_0__4_8_1_1_16_1_next"
# )

# xls_ir_verilog(
#     name = "verilog_fifo",
#     src = ":fifo_ir_opt.opt.ir",
#     codegen_args = {
#         "module_name": "fifo",
#         "delay_model": "unit",
#         "pipeline_stages": "3",
#         "worst_case_throughput": "2",
#         "reset": "rst",
#         "use_system_verilog": "false",
#         # TODO: setup configuration for RAM macro generation
#         # https://google.github.io/xls/codegen_options/#rams-experimental
#         # https://github.com/google/xls/blob/609a7ab89d96d2a7396d2418d00d30e4cb57b119/xls/codegen/ram_configuration.h#L99
#         # https://github.com/google/xls/blob/609a7ab89d96d2a7396d2418d00d30e4cb57b119/docs_src/tutorials/xlscc_memory.md?plain=1#L140
#     },
#     verilog_file = "fifo.v",
# )

# Address Generator
xls_dslx_library(
    name = "address_generator_lib",
    srcs = [
        "address_generator.x",
    ],
    deps = [
        ":dma_common",
    ],
)

xls_dslx_test(
    name = "test_address_generator",
    library = "address_generator_lib",
)

xls_dslx_verilog(
    name = "verilog_address_generator",
    codegen_args = {
        "module_name": "address_generator",
        "delay_model": "asap7",
        "pipeline_stages": "3",
        "worst_case_throughput": "3",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "address_generator",
    library = "address_generator_lib",
    opt_ir_args = {
        "top": "__address_generator__address_generator__AddressGenerator_0__32_4_next",
    },
    verilog_file = "address_generator.v",
)

xls_benchmark_ir(
    name = "address_generator_opt_ir_benchmark",
    src = ":verilog_address_generator.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "3",
        "worst_case_throughput": "3",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "verilog_address_generator_lib",
    srcs = [
        ":address_generator.v",
    ],
)

synthesize_rtl(
    name = "address_generator_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "address_generator",
    deps = [
        ":verilog_address_generator_lib",
    ],
)

benchmark_synth(
    name = "address_generator_benchmark_synth",
    synth_target = ":address_generator_synth_asap7",
)

place_and_route(
    name = "address_generator_place_and_route",
    clock_period = "2000",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":address_generator_synth_asap7",
    target_die_utilization_percentage = "10",
)

# Frontend Reader
xls_dslx_library(
    name = "frontend_reader_lib",
    srcs = [
        "frontend_reader.x",
    ],
    deps = [
        ":dma_common",
    ],
)

xls_dslx_test(
    name = "test_frontend_reader",
    library = "frontend_reader_lib",
)

xls_dslx_verilog(
    name = "verilog_frontend_reader",
    codegen_args = {
        "module_name": "frontend_reader",
        "delay_model": "asap7",
        "pipeline_stages": "4",
        "worst_case_throughput": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "frontend_reader",
    library = "frontend_reader_lib",
    opt_ir_args = {
        "top": "__frontend_reader__frontend_reader__FrontendReader_0__32_32_4_4_4_next",
    },
    verilog_file = "frontend_reader.v",
)

xls_benchmark_ir(
    name = "frontend_reader_opt_ir_benchmark",
    src = ":verilog_frontend_reader.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "4",
        "worst_case_throughput": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "verilog_frontend_reader_lib",
    srcs = [
        ":frontend_reader.v",
    ],
)

synthesize_rtl(
    name = "frontend_reader_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "frontend_reader",
    deps = [
        ":verilog_frontend_reader_lib",
    ],
)

benchmark_synth(
    name = "frontend_reader_benchmark_synth",
    synth_target = ":frontend_reader_synth_asap7",
)

place_and_route(
    name = "frontend_reader_place_and_route",
    clock_period = "2000",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":frontend_reader_synth_asap7",
    target_die_utilization_percentage = "10",
)

# Frontend writer
xls_dslx_library(
    name = "frontend_writer_lib",
    srcs = [
        "frontend_writer.x",
    ],
    deps = [
        ":dma_common",
    ],
)

xls_dslx_test(
    name = "test_frontend_writer",
    library = "frontend_writer_lib",
)

xls_dslx_verilog(
    name = "verilog_frontend_writer",
    codegen_args = {
        "module_name": "frontend_writer",
        "delay_model": "asap7",
        "pipeline_stages": "4",
        "worst_case_throughput": "3",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "frontend_writer",
    library = "frontend_writer_lib",
    opt_ir_args = {
        "top": "__frontend_writer__frontend_writer__FrontendWriter_0__32_32_4_4_4_4_next",
    },
    verilog_file = "frontend_writer.v",
)

xls_benchmark_ir(
    name = "frontend_writer_opt_ir_benchmark",
    src = ":verilog_frontend_writer.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "4",
        "worst_case_throughput": "3",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "verilog_frontend_writer_lib",
    srcs = [
        ":frontend_writer.v",
    ],
)

synthesize_rtl(
    name = "frontend_writer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "frontend_writer",
    deps = [
        ":verilog_frontend_writer_lib",
    ],
)

benchmark_synth(
    name = "frontend_writer_benchmark_synth",
    synth_target = ":frontend_writer_synth_asap7",
)

place_and_route(
    name = "frontend_writer_place_and_route",
    clock_period = "2000",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":frontend_writer_synth_asap7",
    target_die_utilization_percentage = "10",
)

# Main controller
xls_dslx_library(
    name = "main_controller_lib",
    srcs = [
        "main_controller.x",
    ],
    deps = [
        ":address_generator_lib",
        ":axi_csr_lib",
        ":dma_common",
        ":frontend_reader_lib",
        ":frontend_writer_lib",
    ],
)

xls_dslx_test(
    name = "test_main_controller",
    library = "main_controller_lib",
)

# FIXME: Two instances of address generator with the same parameters conflict with each other
# related issue: https://github.com/google/xls/issues/1202

# xls_dslx_verilog(
#     name = "verilog_main_controller",
#     opt_ir_args = {
#         "inline_procs": "true",
#         "top" : "__main_controller__main_controller__MainController_0__32_32_4_4_4_14_4_next",
#     },
#     codegen_args = {
#         "module_name": "main_controller",
#         "delay_model": "asap7",
#         "pipeline_stages": "4",
#         "reset": "rst",
#         "use_system_verilog": "false",
#     },
#     dslx_top = "main_controller",
#     library = "main_controller_lib",
#     verilog_file = "main_controller.v",
# )

# xls_benchmark_ir(
#     name = "main_controller_opt_ir_benchmark",
#     src = ":verilog_main_controller.opt.ir",
#     benchmark_ir_args = {
#         "pipeline_stages": "3",
#         "worst_case_throughput": "3",
#         "delay_model": "asap7",
#     },
# )

# verilog_library(
#     name = "verilog_main_controller_lib",
#     srcs = [
#         ":main_controller.v",
#     ],
# )

# synthesize_rtl(
#     name = "main_controller_synth_asap7",
#     standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
#     top_module = "main_controller",
#     deps = [
#         ":verilog_main_controller_lib",
#     ],
# )

# benchmark_synth(
#     name = "main_controller_benchmark_synth",
#     synth_target = ":main_controller_synth_asap7",
# )

# place_and_route(
#     name = "main_controller_place_and_route",
#     clock_period = "750",
#     core_padding_microns = 2,
#     min_pin_distance = "0.5",
#     placement_density = "0.30",
#     stop_after_step = "global_routing",
#     synthesized_rtl = ":main_controller_synth_asap7",
#     target_die_utilization_percentage = "10",
# )

# Formatting
xls_dslx_fmt_test(
    name = "fmt_address_generator",
    src = "address_generator.x",
)

xls_dslx_fmt_test(
    name = "fmt_axi_csr",
    src = "axi_csr.x",
)

xls_dslx_fmt_test(
    name = "fmt_common",
    src = "common.x",
)

xls_dslx_fmt_test(
    name = "fmt_config",
    src = "config.x",
)

xls_dslx_fmt_test(
    name = "fmt_csr",
    src = "csr.x",
)

xls_dslx_fmt_test(
    name = "fmt_fifo",
    src = "fifo.x",
)

xls_dslx_fmt_test(
    name = "fmt_frontend_reader",
    src = "frontend_reader.x",
)

xls_dslx_fmt_test(
    name = "fmt_frontend_writer",
    src = "frontend_writer.x",
)

xls_dslx_fmt_test(
    name = "fmt_gpf",
    src = "gpf.x",
)

xls_dslx_fmt_test(
    name = "fmt_main_controller",
    src = "main_controller.x",
)

xls_dslx_fmt_test(
    name = "fmt_bus_axi_pkg",
    src = "bus/axi_pkg.x",
)

xls_dslx_fmt_test(
    name = "fmt_bus_axi_st_pkg",
    src = "bus/axi_st_pkg.x",
)
