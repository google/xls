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

# Build rules for XLS RLE Encoder implementation.

load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "synthesize_rtl")
load("@rules_hdl//gds_write:build_defs.bzl", "gds_write")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "rle_common_dslx",
    srcs = [
        "rle_common.x",
    ],
)

xls_dslx_library(
    name = "rle_enc_dslx",
    srcs = [
        "rle_enc.x",
    ],
    deps = [
        ":rle_common_dslx",
    ],
)

xls_dslx_test(
    name = "rle_enc_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_enc_dslx",
)

xls_dslx_test(
    name = "rle_enc_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_enc_dslx",
)

xls_dslx_test(
    name = "rle_enc_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_enc_dslx",
)

xls_dslx_ir(
    name = "rle_enc_ir",
    dslx_top = "RunLengthEncoder32",
    ir_file = "rle_enc.ir",
    library = "rle_enc_dslx",
)

xls_ir_opt_ir(
    name = "rle_enc_opt_ir",
    src = "rle_enc.ir",
    top = "__rle_enc__RunLengthEncoder32__RunLengthEncoder_0__2_32_next",
)

xls_ir_verilog(
    name = "rle_enc_verilog",
    src = ":rle_enc_opt_ir.opt.ir",
    codegen_args = {
        "module_name": "rle_enc",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    verilog_file = "rle_enc.v",
)

verilog_library(
    name = "rle_enc_verilog_lib",
    srcs = [
        ":rle_enc.v",
    ],
)

synthesize_rtl(
    name = "rle_enc_synth_sky130",
    top_module = "rle_enc",
    deps = [
        ":rle_enc_verilog_lib",
    ],
)

place_and_route(
    name = "rle_enc_place_and_route_sky130",
    # ~0.67 GHz
    clock_period = "1.5",
    core_padding_microns = 2,
    die_height_microns = 120,
    die_width_microns = 120,
    min_pin_distance = "2",
    placement_density = "0.96",
    synthesized_rtl = ":rle_enc_synth_sky130",
)

gds_write(
    name = "rle_enc_gds_sky130",
    implemented_rtl = ":rle_enc_place_and_route_sky130",
)

synthesize_rtl(
    name = "rle_enc_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt",
    top_module = "rle_enc",
    deps = [
        ":rle_enc_verilog_lib",
    ],
)

place_and_route(
    name = "rle_enc_place_and_route_asap7",
    # ~1 GHz
    clock_period = "750",
    core_padding_microns = 1,
    die_height_microns = 25,
    die_width_microns = 25,
    min_pin_distance = "0.2",
    placement_density = "0.95",
    synthesized_rtl = ":rle_enc_synth_asap7",
)

gds_write(
    name = "rle_enc_gds_asap7",
    implemented_rtl = ":rle_enc_place_and_route_asap7",
)

xls_benchmark_ir(
    name = "rle_enc_ir_benchmark",
    src = ":rle_enc_opt_ir.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
    },
)

xls_dslx_library(
    name = "rle_dec_dslx",
    srcs = [
        "rle_dec.x",
    ],
    deps = [
        ":rle_common_dslx",
    ],
)

xls_dslx_test(
    name = "rle_dec_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "rle_dec_dslx",
)

xls_dslx_test(
    name = "rle_dec_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "rle_dec_dslx",
)

xls_dslx_test(
    name = "rle_dec_dslx_jit_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = "rle_dec_dslx",
)

xls_dslx_ir(
    name = "rle_dec_ir",
    dslx_top = "RunLengthDecoder32",
    ir_file = "rle_dec.ir",
    library = "rle_dec_dslx",
)

xls_ir_opt_ir(
    name = "rle_dec_opt_ir",
    src = "rle_dec.ir",
    top = "__rle_dec__RunLengthDecoder32__RunLengthDecoder_0__2_32_next",
)

xls_ir_verilog(
    name = "rle_dec_verilog",
    src = ":rle_dec_opt_ir.opt.ir",
    codegen_args = {
        "module_name": "rle_dec",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    verilog_file = "rle_dec.v",
)

verilog_library(
    name = "rle_dec_verilog_lib",
    srcs = [
        ":rle_dec.v",
    ],
)

synthesize_rtl(
    name = "rle_dec_synth_sky130",
    top_module = "rle_dec",
    deps = [
        ":rle_dec_verilog_lib",
    ],
)

place_and_route(
    name = "rle_dec_place_and_route_sky130",
    # ~0.83 GHz
    clock_period = "1.2",
    core_padding_microns = 2,
    die_height_microns = 120,
    die_width_microns = 120,
    min_pin_distance = "2",
    placement_density = "0.95",
    synthesized_rtl = ":rle_dec_synth_sky130",
)

gds_write(
    name = "rle_dec_gds_sky130",
    implemented_rtl = ":rle_dec_place_and_route_sky130",
)

synthesize_rtl(
    name = "rle_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt",
    top_module = "rle_dec",
    deps = [
        ":rle_dec_verilog_lib",
    ],
)

place_and_route(
    name = "rle_dec_place_and_route_asap7",
    # ~2.5GHz
    clock_period = "400",
    core_padding_microns = 1,
    die_height_microns = 15,
    die_width_microns = 15,
    min_pin_distance = "0.2",
    placement_density = "0.95",
    synthesized_rtl = ":rle_dec_synth_asap7",
)

gds_write(
    name = "rle_dec_gds_asap7",
    implemented_rtl = ":rle_dec_place_and_route_asap7",
)

xls_benchmark_ir(
    name = "rle_dec_ir_benchmark",
    src = ":rle_dec_opt_ir.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    },
)

# TODO(b/319286627): 2024-01-09 Disabled due to failure at HEAD.
# build_test(
#     name = "rle_build_test",
#     targets = [
#         "rle_enc_verilog",
#         "rle_enc_gds_sky130",
#         "rle_enc_gds_asap7",
#         "rle_dec_verilog",
#         "rle_dec_gds_sky130",
#         "rle_dec_gds_asap7",
#     ],
# )

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
