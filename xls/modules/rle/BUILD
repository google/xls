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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)
load("//xls/build_rules:dashboard.bzl", "dashboard")
load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
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
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
        "streaming_channel_data_suffix": "_data",
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
    clock_period = "1",
    core_padding_microns = 10,
    placement_density = "0.8",
    synthesized_rtl = ":rle_enc_synth_sky130",
    target_die_utilization_percentage = "30",
)

xls_benchmark_ir(
    name = "rle_enc_ir_benchmark",
    src = ":rle_enc_opt_ir.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
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

xls_benchmark_ir(
    name = "rle_dec_ir_benchmark",
    src = ":rle_dec_opt_ir.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    },
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)

py_test(
    name = "rle_enc_cocotb_test",
    srcs = ["rle_enc_cocotb_test.py"],
    data = [
        ":rle_enc_verilog",
        "@com_icarus_iverilog//:iverilog",
        "@com_icarus_iverilog//:vvp",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/simulation/cocotb:cocotb_struct",
        "//xls/simulation/cocotb:cocotb_xls",
        "//xls/tools/dashboard:utils",
    ],
)

dashboard(
    name = "rle_dashboard",
    title = "RLE Dashboard",
    files = [
        {
            "name": "//xls/modules/rle:rle_enc_place_and_route_sky130",
            "file_parsers": {
                "//xls/tools/dashboard:chip_area_results_textproto_parser": "xls/modules/rle/rle_enc_place_and_route_sky130_general_routing_area_results.textproto",
                "//xls/tools/dashboard:chip_power_results_textproto_parser": "xls/modules/rle/rle_enc_place_and_route_sky130_general_routing_power_results.textproto",
            },
        },
    ],
    tests = [
        {
            "name": "rle_enc_cocotb_test",
            "output_parsers": [
                "//xls/tools/dashboard:generic_parser",
            ],
            "file_parsers": {
                "//xls/tools/dashboard:cocotb_results_xml_parser": "sim_build/results.xml",
            },
        },
        {
            "name": "rle_enc_dslx_ir_test",
            "output_parsers": [
                "//xls/tools/dashboard:dslx_test_parser",
            ],
        },
    ],
)
