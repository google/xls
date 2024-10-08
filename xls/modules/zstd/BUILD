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
    "xls_dslx_ir",
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
    opt_ir_args = {
        "top": "__window_buffer__WindowBuffer64__WindowBuffer_0__64_32_48_next",
    },
    tags = ["manual"],
    verilog_file = "window_buffer.v",
)

xls_benchmark_ir(
    name = "window_buffer_opt_ir_benchmark",
    src = ":window_buffer_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
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
    clock_period = "750",
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
    dslx_test_args = {"compare": "jit"},
    library = ":frame_header_dslx",
    tags = ["manual"],
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
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/ir_convert:convert_options",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/dslx/type_system:parametric_env",
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/simulation:sim_test_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
        "@zstd",
    ],
)

xls_dslx_verilog(
    name = "frame_header_verilog",
    codegen_args = {
        "module_name": "FrameHeaderDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "9",
        "reset": "rst",
        "reset_data_path": "false",
        "use_system_verilog": "false",
    },
    dslx_top = "parse_frame_header_128",
    library = ":frame_header_test_dslx",
    tags = ["manual"],
    verilog_file = "frame_header.v",
)

xls_benchmark_ir(
    name = "frame_header_opt_ir_benchmark",
    src = ":frame_header_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "9",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

verilog_library(
    name = "frame_header_verilog_lib",
    srcs = [
        ":frame_header.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "frame_header_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "FrameHeaderDecoder",
    deps = [
        ":frame_header_verilog_lib",
    ],
)

benchmark_synth(
    name = "frame_header_benchmark_synth",
    synth_target = ":frame_header_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "frame_header_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":frame_header_synth_asap7",
    tags = ["manual"],
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
    dslx_test_args = {"compare": "jit"},
    library = ":raw_block_dec_dslx",
    tags = ["manual"],
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
    tags = ["manual"],
    verilog_file = "raw_block_dec.v",
)

xls_benchmark_ir(
    name = "raw_block_dec_opt_ir_benchmark",
    src = ":raw_block_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
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
    clock_period = "750",
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
        ":buffer_dslx",
        ":common_dslx",
        "//xls/modules/rle:rle_common_dslx",
        "//xls/modules/rle:rle_dec_dslx",
    ],
)

xls_dslx_test(
    name = "rle_block_dec_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":rle_block_dec_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "rle_block_dec_verilog",
    codegen_args = {
        "module_name": "RleBlockDecoder",
        "delay_model": "asap7",
        "pipeline_stages": "3",
        "reset": "rst",
        "use_system_verilog": "false",
        "fifo_module": "",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "RleBlockDecoder",
    library = ":rle_block_dec_dslx",
    opt_ir_args = {
        "top": "__rle_block_dec__RleBlockDecoder__BatchPacker_0_next",
    },
    tags = ["manual"],
    verilog_file = "rle_block_dec.v",
)

xls_benchmark_ir(
    name = "rle_block_dec_opt_ir_benchmark",
    src = ":rle_block_dec_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "3",
        "delay_model": "asap7",
        "fifo_module": "",
        "materialize_internal_fifos": "true",
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
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":rle_block_dec_synth_asap7",
    tags = ["manual"],
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
    dslx_test_args = {"compare": "jit"},
    library = ":block_header_dslx",
    tags = ["manual"],
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

xls_dslx_verilog(
    name = "dec_mux_verilog",
    codegen_args = {
        "module_name": "DecoderMux",
        "delay_model": "asap7",
        "pipeline_stages": "3",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "DecoderMux",
    library = ":dec_mux_dslx",
    tags = ["manual"],
    verilog_file = "dec_mux.v",
)

xls_benchmark_ir(
    name = "dec_mux_opt_ir_benchmark",
    src = ":dec_mux_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "10",
        "delay_model": "asap7",
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
    clock_period = "750",
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

xls_dslx_verilog(
    name = "sequence_executor_verilog",
    codegen_args = {
        "module_name": "sequence_executor",
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
        "pipeline_stages": "8",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "true",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "worst_case_throughput": "1",
        "use_system_verilog": "false",
        "fifo_module": "",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "SequenceExecutorZstd",
    library = ":sequence_executor_dslx",
    opt_ir_args = {
        "top": "__sequence_executor__SequenceExecutorZstd__SequenceExecutor_0__64_0_0_0_13_8192_65536_next",
    },
    tags = ["manual"],
    verilog_file = "sequence_executor.v",
)

xls_benchmark_ir(
    name = "sequence_executor_ir_benchmark",
    src = ":sequence_executor_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "8",
        "delay_model": "asap7",
        "fifo_module": "",
        "materialize_internal_fifos": "true",
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
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.4",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":sequence_executor_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "repacketizer_dslx",
    srcs = [
        "repacketizer.x",
    ],
    deps = [
        ":common_dslx",
    ],
)

xls_dslx_test(
    name = "repacketizer_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":repacketizer_dslx",
    tags = ["manual"],
)

xls_dslx_verilog(
    name = "repacketizer_verilog",
    codegen_args = {
        "module_name": "Repacketizer",
        "delay_model": "asap7",
        "pipeline_stages": "2",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "Repacketizer",
    library = ":repacketizer_dslx",
    tags = ["manual"],
    verilog_file = "repacketizer.v",
)

xls_benchmark_ir(
    name = "repacketizer_opt_ir_benchmark",
    src = ":repacketizer_verilog.opt.ir",
    benchmark_ir_args = {
        "pipeline_stages": "2",
        "delay_model": "asap7",
    },
    tags = ["manual"],
)

verilog_library(
    name = "repacketizer_verilog_lib",
    srcs = [
        ":repacketizer.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "repacketizer_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_28//:asap7-sc7p5t_rev28_rvt",
    tags = ["manual"],
    top_module = "Repacketizer",
    deps = [
        ":repacketizer_verilog_lib",
    ],
)

benchmark_synth(
    name = "repacketizer_benchmark_synth",
    synth_target = ":repacketizer_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "repacketizer_place_and_route",
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":repacketizer_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)

xls_dslx_library(
    name = "zstd_dec_dslx",
    srcs = [
        "zstd_dec.x",
    ],
    deps = [
        ":block_header_dslx",
        ":buffer_dslx",
        ":common_dslx",
        ":frame_header_dslx",
        ":frame_header_test_dslx",
        ":ram_printer_dslx",
        ":repacketizer_dslx",
        ":sequence_executor_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_verilog(
    name = "zstd_dec_verilog",
    codegen_args = {
        "module_name": "ZstdDecoder",
        "generator": "pipeline",
        "delay_model": "asap7",
        "ram_configurations": ",".join([
            "{ram_name}:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_resp}:{latency}".format(
                latency = 5,
                ram_name = "ram{}".format(num),
                rd_req = "zstd_dec__ram_rd_req_{}_s".format(num),
                rd_resp = "zstd_dec__ram_rd_resp_{}_r".format(num),
                wr_req = "zstd_dec__ram_wr_req_{}_s".format(num),
                wr_resp = "zstd_dec__ram_wr_resp_{}_r".format(num),
            )
            for num in range(7)
        ]),
        "pipeline_stages": "10",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "true",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "worst_case_throughput": "1",
        "use_system_verilog": "false",
        "fifo_module": "",
        "materialize_internal_fifos": "true",
    },
    dslx_top = "ZstdDecoder",
    library = ":zstd_dec_dslx",
    tags = ["manual"],
    verilog_file = "zstd_dec.v",
)

xls_dslx_ir(
    name = "zstd_dec_test_ir",
    dslx_top = "ZstdDecoderTest",
    ir_file = "zstd_dec_test.ir",
    library = ":zstd_dec_dslx",
    tags = ["manual"],
)

cc_test(
    name = "zstd_dec_cc_test",
    size = "large",
    srcs = [
        "zstd_dec_test.cc",
    ],
    data = [
        ":zstd_dec_test.ir",
    ],
    shard_count = 50,
    deps = [
        ":data_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:events",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/jit:jit_proc_runtime",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
        "@zstd",
    ],
)

xls_benchmark_ir(
    name = "zstd_dec_opt_ir_benchmark",
    src = ":zstd_dec_verilog.opt.ir",
    benchmark_ir_args = {
        #TODO: rewrite ram in opt_ir step to perform valid IR benchmark
        "pipeline_stages": "1",
        "delay_model": "asap7",
        "fifo_module": "",
        "materialize_internal_fifos": "true",
    },
    tags = ["manual"],
)

verilog_library(
    name = "zstd_dec_verilog_lib",
    srcs = [
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
    clock_period = "750",
    core_padding_microns = 2,
    min_pin_distance = "0.5",
    placement_density = "0.30",
    stop_after_step = "global_routing",
    synthesized_rtl = ":zstd_dec_synth_asap7",
    tags = ["manual"],
    target_die_utilization_percentage = "10",
)
