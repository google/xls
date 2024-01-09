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

# Build rules for XLS ZSTD codec implementation.

load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "benchmark_synth", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
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
    name = "buffer_dslx",
    srcs = [
        "buffer.x",
    ],
)

xls_dslx_test(
    name = "buffer_dslx_test",
    library = ":buffer_dslx",
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
)

xls_dslx_verilog(
    name = "window_buffer_verilog",
    codegen_args = {
        "module_name": "WindowBuffer64",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "WindowBuffer64",
    library = ":window_buffer_dslx",
    # TODO: 2024-01-25: Workaround for https://github.com/google/xls/issues/869
    # Force proc inlining and set last internal proc as top proc for IR optimization
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__window_buffer__WindowBuffer64__WindowBuffer_0__64_32_48_next",
    },
    verilog_file = "window_buffer.v",
)

xls_benchmark_ir(
    name = "window_buffer_opt_ir_benchmark",
    src = ":window_buffer_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "window_buffer_verilog_lib",
    srcs = [
        ":window_buffer.v",
    ],
)

synthesize_rtl(
    name = "window_buffer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "WindowBuffer64",
    deps = [
        ":window_buffer_verilog_lib",
    ],
)

benchmark_synth(
    name = "window_buffer_benchmark_synth",
    synth_target = ":window_buffer_synth_asap7",
)

place_and_route(
    name = "window_buffer_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    skip_detailed_routing = True,
    synthesized_rtl = ":window_buffer_synth_asap7",
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "magic_dslx",
    srcs = [
        "magic.x",
    ],
    deps = [
        ":buffer_dslx",
    ],
)

xls_dslx_test(
    name = "magic_dslx_test",
    library = ":magic_dslx",
)

cc_library(
    name = "data_generator",
    srcs = ["data_generator.cc"],
    hdrs = ["data_generator.h"],
    data = [
        "@com_github_facebook_zstd//:decodecorpus",
    ],
    deps = [
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

xls_dslx_library(
    name = "frame_header_dslx",
    srcs = [
        "frame_header.x",
    ],
    deps = [
        ":buffer_dslx",
    ],
)

xls_dslx_library(
    name = "common_dslx",
    srcs = [
        "common.x",
    ],
    deps = [],
)

xls_dslx_test(
    name = "frame_header_dslx_test",
    library = ":frame_header_dslx",
)

xls_dslx_library(
    name = "frame_header_test_dslx",
    srcs = [
        "frame_header_test.x",
    ],
    deps = [
        ":buffer_dslx",
        ":frame_header_dslx",
    ],
)

cc_test(
    name = "frame_header_cc_test",
    srcs = [
        "frame_header_test.cc",
    ],
    data = [
        ":frame_header_test_dslx",
    ],
    shard_count = 50,
    deps = [
        ":data_generator",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/dslx/type_system:parametric_env",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_github_facebook_zstd//:zstd",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:googletest_fixture_adapter",
        "@com_google_googletest//:gtest",
    ],
)

xls_dslx_verilog(
    name = "frame_header_verilog",
    codegen_args = {
        "module_name": "FrameHeaderDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "9",
        "use_system_verilog": "false",
    },
    dslx_top = "parse_frame_header_128",
    library = ":frame_header_test_dslx",
    verilog_file = "frame_header.v",
)

xls_benchmark_ir(
    name = "frame_header_opt_ir_benchmark",
    src = ":frame_header_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "9",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "frame_header_verilog_lib",
    srcs = [
        ":frame_header.v",
    ],
)

synthesize_rtl(
    name = "frame_header_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "FrameHeaderDecoder",
    deps = [
        ":frame_header_verilog_lib",
    ],
)

benchmark_synth(
    name = "frame_header_benchmark_synth",
    synth_target = ":frame_header_synth_asap7",
)

place_and_route(
    name = "frame_header_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    skip_detailed_routing = True,
    synthesized_rtl = ":frame_header_synth_asap7",
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "raw_block_dec_dslx",
    srcs = [
        "raw_block_dec.x",
    ],
    deps = [
        ":buffer_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "raw_block_dec_dslx_test",
    library = ":raw_block_dec_dslx",
)

xls_dslx_verilog(
    name = "raw_block_dec_verilog",
    codegen_args = {
        "module_name": "RawBlockDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "RawBlockDecoder",
    library = ":raw_block_dec_dslx",
    verilog_file = "raw_block_dec.v",
)

xls_benchmark_ir(
    name = "raw_block_dec_opt_ir_benchmark",
    src = ":raw_block_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "raw_block_dec_verilog_lib",
    srcs = [
        ":raw_block_dec.v",
    ],
)

synthesize_rtl(
    name = "raw_block_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "RawBlockDecoder",
    deps = [
        ":raw_block_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "raw_block_dec_benchmark_synth",
    synth_target = ":raw_block_dec_synth_asap7",
)

place_and_route(
    name = "raw_block_dec_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    skip_detailed_routing = True,
    synthesized_rtl = ":raw_block_dec_synth_asap7",
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "rle_block_dec_dslx",
    srcs = [
        "rle_block_dec.x",
    ],
    deps = [
        ":buffer_dslx",
        ":common_dslx",
        "//xls/modules/rle:rle_common_dslx",
        "//xls/modules/rle:rle_dec_dslx",
    ],
)

xls_dslx_test(
    name = "rle_block_dec_dslx_test",
    library = ":rle_block_dec_dslx",
)

xls_dslx_verilog(
    name = "rle_block_dec_verilog",
    codegen_args = {
        "module_name": "RleBlockDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "3",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "RleBlockDecoder",
    library = ":rle_block_dec_dslx",
    # TODO: 2024-01-15: Workaround for https://github.com/google/xls/issues/869
    # Force proc inlining and set last internal proc as top proc for IR optimization
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__rle_block_dec__RleBlockDecoder__BatchPacker_0_next",
    },
    verilog_file = "rle_block_dec.v",
)

xls_benchmark_ir(
    name = "rle_block_dec_opt_ir_benchmark",
    src = ":rle_block_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "3",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "rle_block_dec_verilog_lib",
    srcs = [
        ":rle_block_dec.v",
    ],
)

synthesize_rtl(
    name = "rle_block_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "RleBlockDecoder",
    deps = [
        ":rle_block_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "rle_block_dec_benchmark_synth",
    synth_target = ":rle_block_dec_synth_asap7",
)

place_and_route(
    name = "rle_block_dec_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    skip_detailed_routing = True,
    synthesized_rtl = ":rle_block_dec_synth_asap7",
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "block_header_dslx",
    srcs = [
        "block_header.x",
    ],
    deps = [
        ":buffer_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "block_header_dslx_test",
    library = ":block_header_dslx",
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
    library = ":dec_mux_dslx",
)

xls_dslx_verilog(
    name = "dec_mux_verilog",
    codegen_args = {
        "module_name": "DecoderMux",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "DecoderMux",
    library = ":dec_mux_dslx",
    verilog_file = "dec_mux.v",
)

xls_benchmark_ir(
    name = "dec_mux_opt_ir_benchmark",
    src = ":dec_mux_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "dec_mux_verilog_lib",
    srcs = [
        ":dec_mux.v",
    ],
)

synthesize_rtl(
    name = "dec_mux_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "DecoderMux",
    deps = [
        ":dec_mux_verilog_lib",
    ],
)

benchmark_synth(
    name = "dec_mux_benchmark_synth",
    synth_target = ":dec_mux_synth_asap7",
)

place_and_route(
    name = "dec_mux_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    skip_detailed_routing = True,
    synthesized_rtl = ":dec_mux_synth_asap7",
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "dec_demux_dslx",
    srcs = [
        "dec_demux.x",
    ],
    deps = [
        ":block_header_dslx",
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "dec_demux_dslx_test",
    library = ":dec_demux_dslx",
)

xls_dslx_verilog(
    name = "dec_demux_verilog",
    codegen_args = {
        "module_name": "DecoderDemux",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "DecoderDemux",
    library = ":dec_demux_dslx",
    verilog_file = "dec_demux.v",
)

xls_benchmark_ir(
    name = "dec_demux_opt_ir_benchmark",
    src = ":dec_demux_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "dec_demux_verilog_lib",
    srcs = [
        ":dec_demux.v",
    ],
)

synthesize_rtl(
    name = "dec_demux_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "DecoderDemux",
    deps = [
        ":dec_demux_verilog_lib",
    ],
)

benchmark_synth(
    name = "dec_demux_benchmark_synth",
    synth_target = ":dec_demux_synth_asap7",
)

place_and_route(
    name = "dec_demux_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    skip_detailed_routing = True,
    synthesized_rtl = ":dec_demux_synth_asap7",
    target_die_utilization_percentage = "5",
)

xls_dslx_library(
    name = "block_dec_dslx",
    srcs = [
        "block_dec.x",
    ],
    deps = [
        ":common_dslx",
        ":dec_demux_dslx",
        ":dec_mux_dslx",
        ":raw_block_dec_dslx",
        ":rle_block_dec_dslx",
    ],
)

xls_dslx_test(
    name = "block_dec_dslx_test",
    library = ":block_dec_dslx",
)

xls_dslx_verilog(
    name = "block_dec_verilog",
    codegen_args = {
        "module_name": "BlockDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "BlockDecoder",
    library = ":block_dec_dslx",
    # TODO: 2024-01-15: Workaround for https://github.com/google/xls/issues/869
    # Force proc inlining and set last internal proc as top proc for IR optimization
    opt_ir_args = {
        "inline_procs": "true",
        "top": "__xls_modules_zstd_dec_mux__BlockDecoder__DecoderMux_0_next",
    },
    verilog_file = "block_dec.v",
)

xls_benchmark_ir(
    name = "block_dec_opt_ir_benchmark",
    src = ":block_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
)

verilog_library(
    name = "block_dec_verilog_lib",
    srcs = [
        ":block_dec.v",
    ],
)

synthesize_rtl(
    name = "block_dec_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    top_module = "BlockDecoder",
    deps = [
        ":block_dec_verilog_lib",
    ],
)

benchmark_synth(
    name = "block_dec_benchmark_synth",
    synth_target = ":block_dec_synth_asap7",
)

place_and_route(
    name = "block_dec_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    skip_detailed_routing = True,
    synthesized_rtl = ":block_dec_synth_asap7",
    target_die_utilization_percentage = "10",
)
