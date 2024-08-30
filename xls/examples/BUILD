# Copyright 2020 The XLS Authors
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

# Build rules for XLS examples.

load("@rules_hdl//gds_write:build_defs.bzl", "gds_write")
load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "benchmark_synth", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_fmt_test",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_prove_quickcheck_test",
    "xls_dslx_test",
    "xls_dslx_verilog",
    "xls_eval_ir_test",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)
load("//xls/examples:list_filegroup_files.bzl", "list_filegroup_files")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

exports_files(["sha256.x"])

filegroup(
    name = "ir_examples",
    srcs = [
        ":sha256.ir",
        ":sha256.opt.ir",
        "//xls/examples/adler32:adler32.ir",
        "//xls/examples/adler32:adler32.opt.ir",
        "//xls/examples/crc32:crc32.ir",
        "//xls/examples/crc32:crc32.opt.ir",
        "//xls/examples/matmul_4x4:matmul_4x4.ir",
    ],
)

xls_dslx_library(
    name = "lfsr_dslx",
    srcs = ["lfsr.x"],
)

xls_dslx_fmt_test(
    name = "lfsr_fmt_test",
    src = "lfsr.x",
)

xls_dslx_test(
    name = "lfsr_dslx_test",
    srcs = ["lfsr.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "lfsr_proc_dslx",
    srcs = ["lfsr_proc.x"],
    deps = [":lfsr_dslx"],
)

xls_dslx_test(
    name = "lfsr_proc_dslx_test",
    srcs = ["lfsr_proc.x"],
    dslx_test_args = {"compare": "jit"},
    deps = [":lfsr_proc_dslx"],
)

xls_dslx_library(
    name = "capitalize_dslx",
    srcs = ["capitalize.x"],
)

xls_dslx_test(
    name = "capitalize_dslx_test",
    srcs = ["capitalize.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "capitalize",
    srcs = ["capitalize.x"],
    dslx_top = "main",
)

xls_dslx_opt_ir_test(
    name = "capitalize_test",
    dep = ":capitalize",
)

xls_dslx_opt_ir(
    name = "custom_schedule",
    srcs = ["custom_schedule.x"],
    dslx_top = "Accumulator",
)

xls_dslx_fmt_test(
    name = "custom_schedule_fmt_test",
    src = "custom_schedule.x",
)

xls_dslx_verilog(
    name = "custom_schedule_codegen",
    codegen_args = {
        "module_name": "custom_schedule_mod",
        "generator": "pipeline",
        "delay_model": "sky130",
        "use_system_verilog": "true",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "true",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "io_constraints": "custom_schedule__data_in:recv:custom_schedule__old_state:send:0:0,custom_schedule__data_in:recv:custom_schedule__data_out:send:4:4,custom_schedule__data_in:recv:custom_schedule__activate:recv:8:8",
        "worst_case_throughput": "9",
        "pipeline_stages": "9",
    },
    dslx_top = "Accumulator",
    library = ":custom_schedule",
    verilog_file = "custom_schedule.sv",
)

xls_dslx_library(
    name = "dot_product_dslx",
    srcs = ["dot_product.x"],
)

xls_dslx_test(
    name = "dot_product_dslx_test",
    srcs = ["dot_product.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "fir_filter_dslx",
    srcs = ["fir_filter.x"],
)

xls_dslx_test(
    name = "fir_filter_dslx_test",
    srcs = ["fir_filter.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "riscv_simple_dslx",
    srcs = ["riscv_simple.x"],
)

xls_dslx_opt_ir(
    name = "riscv_simple_opt_ir",
    srcs = ["riscv_simple.x"],
    dslx_top = "run_instruction",
    ir_file = "riscv_simple.ir",
    opt_ir_file = "riscv_simple.opt.ir",
)

xls_ir_verilog(
    name = "riscv_simple_sv",
    src = ":riscv_simple_opt_ir",
    codegen_args = {
        "module_name": "run_instruction",
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
        "reset_data_path": "false",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
        "worst_case_throughput": "3",
        "assert_format": "\\;",
    },
    verilog_file = "riscv_simple.sv",
)

verilog_library(
    name = "riscv_simple_verilog",
    srcs = [":riscv_simple.sv"],
    tags = ["DSLX"],
)

synthesize_rtl(
    name = "riscv_simple_verilog_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt_4x",
    tags = ["manual"],
    top_module = "run_instruction",
    deps = [":riscv_simple_verilog"],
)

benchmark_synth(
    name = "riscv_simple_benchmark_synth_asap7",
    synth_target = ":riscv_simple_verilog_synth_asap7",
    tags = ["manual"],
)

place_and_route(
    name = "riscv_simple_place_and_route_asap7",
    # ~1 GHz
    clock_period = "1000",  # units of clock period for ASAP7 are picoseconds
    core_padding_microns = 1,
    die_height_microns = 500,
    die_width_microns = 500,
    min_pin_distance = "0.2",
    placement_density = "0.6",
    synthesized_rtl = ":riscv_simple_verilog_synth_asap7",
    tags = ["manual"],
)

xls_ir_verilog(
    name = "riscv_simple_sv_sky130",
    src = ":riscv_simple_opt_ir",
    codegen_args = {
        "module_name": "run_instruction",
        "generator": "pipeline",
        "delay_model": "sky130",
        "pipeline_stages": "4",
        "reset": "rst",
        "reset_data_path": "false",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
        "worst_case_throughput": "3",
        "assert_format": "\\;",
    },
    verilog_file = "riscv_simple_sky130.sv",
)

verilog_library(
    name = "riscv_simple_verilog_sky130",
    srcs = [":riscv_simple_sky130.sv"],
    tags = ["DSLX"],
)

synthesize_rtl(
    name = "riscv_simple_verilog_synth_sky130",
    tags = ["manual"],
    top_module = "run_instruction",
    deps = [":riscv_simple_verilog_sky130"],
)

benchmark_synth(
    name = "riscv_simple_benchmark_synth_sky130",
    synth_target = ":riscv_simple_verilog_synth_sky130",
    tags = ["manual"],
)

place_and_route(
    name = "riscv_simple_place_and_route_sky130",
    # 1 GHZ
    clock_period = "1.0",  # units of clock period for SKY130 are nanoseconds
    core_padding_microns = 2,
    die_height_microns = 5000,
    die_width_microns = 5000,
    min_pin_distance = "2",
    placement_density = "0.5",
    synthesized_rtl = ":riscv_simple_verilog_synth_sky130",
    tags = ["manual"],
)

xls_benchmark_ir(
    name = "riscv_simple_benchmark_ir",
    src = ":riscv_simple.ir",
    tags = ["optonly"],
)

xls_dslx_library(
    name = "ffi",
    srcs = ["ffi.x"],
)

xls_dslx_test(
    name = "ffi_test",
    srcs = ["ffi.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_verilog(
    name = "ffi_codegen",
    codegen_args = {
        "module_name": "ffi",
        "generator": "pipeline",
        "pipeline_stages": "1",
        "delay_model": "sky130",
        "ffi_fallback_delay_ps": "123",
        "use_system_verilog": "true",
    },
    dslx_top = "main",
    library = ":ffi",
    verilog_file = "ffi.sv",
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test, xls_eval_ir_test and
# xls_benchmark_ir.
xls_dslx_test(
    name = "riscv_simple_dslx_test",
    srcs = ["riscv_simple.x"],
    dslx_test_args = {
        # TODO(hjmontero): run_instruction segfaults in the JIT.
        "compare": "interpreter",
    },
)

xls_dslx_fmt_test(
    name = "riscv_simple_fmt_test",
    src = "riscv_simple.x",
)

xls_dslx_opt_ir(
    name = "sha256",
    srcs = ["sha256.x"],
    dslx_top = "main",
    ir_file = "sha256.ir",
    opt_ir_file = "sha256.opt.ir",
    tags = ["optonly"],
)

xls_dslx_test(
    name = "sha256_dslx_test",
    srcs = ["sha256.x"],
    dslx_test_args = {"compare": "jit"},
)

# TODO(rspringer): OOMs on some machines. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "sha256_eval_ir_test",
    src = ":sha256.ir",
    tags = ["optonly"],
)

xls_benchmark_ir(
    name = "sha256_benchmark_ir",
    src = ":sha256.ir",
    # TODO: Takes >15m for synthesis; too long for routine benchmarks.
    synthesize = False,
    tags = ["optonly"],
)

cc_xls_ir_jit_wrapper(
    name = "sha256_jit_wrapper",
    src = ":sha256",
    jit_wrapper_args = {
        "class_name": "Sha256",
        "function": "main",
        "namespace": "xls::examples",
    },
)

xls_dslx_library(
    name = "sobel_filter_dslx",
    srcs = ["sobel_filter.x"],
    deps = [
        "//third_party/xls_go_math:fpsqrt_32_dslx",
    ],
)

xls_dslx_test(
    name = "sobel_filter_dslx_test",
    srcs = ["sobel_filter.x"],
    dslx_test_args = {"compare": "jit"},
    deps = [":sobel_filter_dslx"],
)

xls_dslx_test(
    name = "sobel_filter_test",
    srcs = ["sobel_filter.x"],
    dslx_test_args = {"compare": "jit"},
    deps = [":sobel_filter_dslx"],
)

xls_dslx_library(
    name = "sobel_filter_benchmark_dslx",
    srcs = ["sobel_filter_benchmark.x"],
    deps = [":sobel_filter_dslx"],
)

xls_dslx_opt_ir(
    name = "sobel_filter_ir",
    srcs = ["sobel_filter_benchmark.x"],
    dslx_top = "apply_stencil_float32_8x8",
    deps = [":sobel_filter_dslx"],
)

xls_benchmark_ir(
    name = "sobel_filter_benchmark_ir",
    src = ":sobel_filter_ir.ir",
    # TODO: Takes >15m for synthesis; too long for routine benchmarks.
    synthesize = False,
)

list_filegroup_files(
    name = "ir_example_file_list",
    src = ":ir_examples",
    out = "ir_example_file_list.txt",
)

cc_library(
    name = "sample_packages",
    srcs = [
        "sample_packages.cc",
        "sample_packages.inc.h",
    ],
    hdrs = ["sample_packages.h"],
    data = [
        ":ir_example_file_list.txt",
        ":ir_examples",
        "//xls/examples/adler32:adler32.opt.ir",
    ],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:path",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:verifier",
    ],
)

cc_library(
    name = "proc_fir_filter",
    srcs = ["proc_fir_filter.cc"],
    hdrs = ["proc_fir_filter.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
    ],
)

cc_test(
    name = "proc_fir_filter_test",
    srcs = ["proc_fir_filter_test.cc"],
    deps = [
        ":proc_fir_filter",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:interpreter_proc_runtime",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel_ops",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

xls_dslx_library(
    name = "proc_iota",
    srcs = ["proc_iota.x"],
)

xls_dslx_ir(
    name = "proc_iota_ir",
    dslx_top = "main",
    ir_file = "proc_iota.ir",
    library = ":proc_iota",
)

xls_ir_opt_ir(
    name = "proc_iota_consumer_opt_ir",
    src = "proc_iota.ir",
    top = "__proc_iota__main__consumer_0__2_next",
)

xls_ir_opt_ir(
    name = "proc_iota_producer_opt_ir",
    src = "proc_iota.ir",
    top = "__proc_iota__main__producer_0_next",
)

[
    (
        xls_ir_verilog(
            name = "proc_iota_" + procmodule + "_sv",
            src = ":proc_iota_" + procmodule + "_opt_ir",
            codegen_args = {
                "module_name": "proc_iota_" + procmodule,
                "generator": "pipeline",
                "delay_model": "asap7",
                "pipeline_stages": "2",
                "reset": "rst",
                "reset_asynchronous": "true",
                "flop_inputs": "false",
                "streaming_channel_data_suffix": "_data",
                "streaming_channel_ready_suffix": "_ready",
                "streaming_channel_valid_suffix": "_valid",
                "use_system_verilog": "true",
            },
            verilog_file = "proc_iota_" + procmodule + ".sv",
        ),
    )
    for procmodule in [
        "consumer",
        "producer",
    ]
]

xls_dslx_library(
    name = "memory_proc_dslx",
    srcs = ["memory_proc.x"],
)

xls_dslx_test(
    name = "memory_proc_test",
    dslx_test_args = {"compare": "jit"},
    library = ":memory_proc_dslx",
)

# -- tiny_adder sample

xls_dslx_library(
    name = "tiny_adder_dslx",
    srcs = ["tiny_adder.x"],
)

xls_dslx_test(
    name = "tiny_adder_test",
    dslx_test_args = {"compare": "jit"},
    library = ":tiny_adder_dslx",
)

xls_dslx_opt_ir(
    name = "tiny_adder_opt_ir",
    srcs = ["tiny_adder.x"],
    dslx_top = "tiny_adder",
    ir_file = "tiny_adder.ir",
    opt_ir_file = "tiny_adder.opt.ir",
)

xls_dslx_verilog(
    name = "tiny_adder_sv",
    codegen_args = {
        "module_name": "tiny_adder",
        "generator": "combinational",
        "delay_model": "unit",
        "use_system_verilog": "true",
        "output_port_name": "result",
    },
    dslx_top = "tiny_adder",
    library = ":tiny_adder_dslx",
    verilog_file = "tiny_adder.sv",
)

# -- find_index sample, with OpenROAD synthesis

xls_dslx_library(
    name = "find_index_dslx",
    srcs = ["find_index.x"],
)

xls_dslx_test(
    name = "find_index_test",
    dslx_test_args = {"compare": "jit"},
    library = ":find_index_dslx",
)

xls_dslx_opt_ir(
    name = "find_index_opt_ir",
    srcs = ["find_index.x"],
    dslx_top = "find_index",
    ir_file = "find_index.ir",
    opt_ir_file = "find_index.opt.ir",
)

xls_dslx_verilog(
    name = "find_index_5000ps_model_unit",
    codegen_args = {
        "delay_model": "unit",
        "clock_period_ps": "5000",
        "reset_data_path": "false",
        "module_name": "find_index",
        "use_system_verilog": "false",
    },
    dslx_top = "find_index",
    library = ":find_index_dslx",
    verilog_file = "find_index_5000ps_model_unit.v",
)

verilog_library(
    name = "find_index_5000ps_model_unit_verilog",
    srcs = [
        ":find_index_5000ps_model_unit.v",
    ],
)

synthesize_rtl(
    name = "find_index_5000ps_model_unit_verilog_synth_sky130",
    top_module = "find_index",
    deps = [
        ":find_index_5000ps_model_unit_verilog",
    ],
)

synthesize_rtl(
    name = "find_index_5000ps_model_unit_verilog_synth_asap7",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt",
    top_module = "find_index",
    deps = [
        ":find_index_5000ps_model_unit_verilog",
    ],
)

synthesize_rtl(
    name = "find_index_5000ps_model_unit_verilog_synth_by_stage",
    synth_tcl = "//xls/synthesis/yosys:synth_by_stage.tcl",
    top_module = "find_index",
    deps = [
        ":find_index_5000ps_model_unit_verilog",
    ],
)

xls_dslx_library(
    name = "large_array_dslx",
    srcs = ["large_array.x"],
)

xls_dslx_test(
    name = "large_array_dslx_test",
    srcs = ["large_array.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "large_array",
    srcs = ["large_array.x"],
    dslx_top = "large_array",
    ir_file = "large_array.ir",
    opt_ir_file = "large_array.opt.ir",
)

xls_benchmark_ir(
    name = "large_array_benchmark_ir",
    src = ":large_array.ir",
    # TODO: Takes >15m for synthesis; too long for routine benchmarks.
    synthesize = False,
)

xls_dslx_library(
    name = "prefix_sum_dslx",
    srcs = ["prefix_sum.x"],
)

xls_dslx_test(
    name = "prefix_sum_dslx_test",
    srcs = ["prefix_sum.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "prefix_sum",
    srcs = ["prefix_sum.x"],
    dslx_top = "prefix_sum",
    ir_file = "prefix_sum.ir",
    opt_ir_file = "prefix_sum.opt.ir",
)

xls_benchmark_ir(
    name = "prefix_sum_benchmark_ir",
    src = ":prefix_sum.ir",
    # TODO: Takes >15m for synthesis; too long for routine benchmarks.
    synthesize = False,
)

xls_dslx_library(
    name = "sparse_prefix_sum_dslx",
    srcs = ["sparse_prefix_sum.x"],
)

xls_dslx_test(
    name = "sparse_prefix_sum_dslx_test",
    srcs = ["sparse_prefix_sum.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "sparse_prefix_sum",
    srcs = ["sparse_prefix_sum.x"],
    dslx_top = "sparse_prefix_sum",
    ir_file = "sparse_prefix_sum.ir",
    opt_ir_file = "sparse_prefix_sum.opt.ir",
)

xls_benchmark_ir(
    name = "sparse_prefix_sum_benchmark_ir",
    src = ":sparse_prefix_sum.ir",
    # TODO: Takes >15m for synthesis; too long for routine benchmarks.
    synthesize = False,
)

xls_dslx_library(
    name = "ram_dslx",
    srcs = ["ram.x"],
)

xls_dslx_test(
    name = "ram_test",
    dslx_test_args = {"compare": "jit"},
    library = ":ram_dslx",
)

xls_dslx_library(
    name = "delay_dslx",
    srcs = ["delay.x"],
    deps = [":ram_dslx"],
)

xls_dslx_test(
    name = "delay_test",
    dslx_test_args = {"compare": "jit"},
    library = ":delay_dslx",
)

xls_dslx_ir(
    name = "delay_ir",
    dslx_top = "Delay32x2048_init3",
    ir_file = "delay.ir",
    library = ":delay_dslx",
)

xls_ir_opt_ir(
    name = "delay_opt_ir",
    src = "delay.ir",
    top = "__delay__Delay32x2048_init3__Delay_0__DelayInternal_0__10_32_2048_64_1024_3_next",
)

xls_ir_verilog(
    name = "delay_sv",
    src = ":delay_opt_ir",
    codegen_args = {
        "module_name": "delay_top",
        "generator": "pipeline",
        "delay_model": "unit",
        # Enable codegen pass that rewrites the req and resp channels to drive
        # an external RAM. The configuration below means that the pair of
        # channels `delay__ram_req` and `delay__ram_resp` drive a 1RW RAM
        # and output ports should be prefixed with `ram`, e.g. `ram_addr`.
        # See https://google.github.io/xls/codegen_options/#rams-experimental
        # for more information about this option.
        "ram_configurations": "ram:1RW:{req}:{resp}:{wr_comp}".format(
            req = "delay__ram_req",
            resp = "delay__ram_resp",
            wr_comp = "delay__ram_wr_comp",
        ),
        "pipeline_stages": "2",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
    },
    verilog_file = "delay.sv",
)

verilog_library(
    name = "delay",
    srcs = [":delay.sv"],
    tags = ["DSLX"],
)

synthesize_rtl(
    name = "delay_verilog_synth",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt_4x",
    tags = ["manual"],
    top_module = "delay_top",
    deps = [":delay"],
)

benchmark_synth(
    name = "delay_benchmark_synth",
    synth_target = ":delay_verilog_synth",
    tags = ["manual"],
)

xls_dslx_library(
    name = "delay_loopback_channel_dslx",
    srcs = ["delay_loopback_channel.x"],
)

xls_dslx_fmt_test(
    name = "delay_loopback_channel_fmt_test",
    src = "delay_loopback_channel.x",
)

xls_dslx_test(
    name = "delay_loopback_channel_test",
    dslx_test_args = {"compare": "jit"},
    library = ":delay_loopback_channel_dslx",
)

xls_dslx_ir(
    name = "delay_loopback_channel_ir",
    dslx_top = "Delay",
    ir_file = "delay_loopback_channel.ir",
    library = ":delay_loopback_channel_dslx",
)

xls_ir_opt_ir(
    name = "delay_loopback_channel_opt_ir",
    src = "delay_loopback_channel.ir",
)

xls_benchmark_ir(
    name = "delay_loopback_channel_benchmark_ir",
    src = ":delay_loopback_channel_opt_ir",
    benchmark_ir_args = {
        # TODO: With schedule generation this generates InstantiationOutput
        # nodes which are not supported by the block-jit. For now just disable
        # scheduling.
        "pipeline_stages": "0",
    },
    # TODO: Enable this once loopback channels work in synthesis.
    synthesize = False,
)

xls_ir_verilog(
    name = "delay_loopback_channel_sv",
    src = ":delay_loopback_channel_opt_ir",
    codegen_args = {
        "module_name": "delay_top",
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "1",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
    },
    verilog_file = "delay_loopback_channel.sv",
)

verilog_library(
    name = "delay_loopback_channel",
    srcs = [":delay_loopback_channel.sv"],
    tags = ["DSLX_LOOPBACK"],
)

xls_dslx_library(
    name = "quickcheck_dslx",
    srcs = ["quickcheck.x"],
)

xls_dslx_test(
    name = "quickcheck_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = ":quickcheck_dslx",
)

xls_dslx_prove_quickcheck_test(
    name = "quickcheck_prove_test",
    library = ":quickcheck_dslx",
)

xls_dslx_library(
    name = "constraint_dslx",
    srcs = ["constraint.x"],
)

xls_dslx_test(
    name = "constraint_dslx_test",
    srcs = ["constraint.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_ir(
    name = "constraint_ir",
    dslx_top = "main",
    ir_file = "constraint.ir",
    library = ":constraint_dslx",
)

xls_ir_opt_ir(
    name = "constraint_opt_ir",
    src = "constraint.ir",
)

xls_ir_verilog(
    name = "constraint_sv",
    src = ":constraint_opt_ir",
    codegen_args = {
        "module_name": "constraint_top",
        "generator": "pipeline",
        "delay_model": "unit",
        # Set a constraint between req and resp that they must be 2 cycles apart
        # See https://google.github.io/xls/codegen_options/#pipelining-and-scheduling-options
        # for more information about this option.
        "io_constraints": "constraint__req:send:constraint__resp:recv:2:2",
        "pipeline_stages": "4",
        "clock_period_ps": "2",
        "reset": "rst",
        "use_system_verilog": "true",
    },
    verilog_file = "constraint.sv",
)

verilog_library(
    name = "constraint",
    srcs = [":constraint.sv"],
)

place_and_route(
    name = "find_index_place_and_route_sky130",
    # ~1.43 GHZ
    clock_period = "0.70",  # units of clock period for SKY130 are nanoseconds
    core_padding_microns = 2,
    die_height_microns = 45,
    die_width_microns = 45,
    min_pin_distance = "2",
    placement_density = "0.95",
    synthesized_rtl = ":find_index_5000ps_model_unit_verilog_synth_sky130",
)

gds_write(
    name = "find_index_gds_sky130",
    implemented_rtl = ":find_index_place_and_route_sky130",
)

place_and_route(
    name = "find_index_place_and_route_asap7",
    # ~3 GHz
    clock_period = "325",  # units of clock period for ASAP7 are picoseconds
    core_padding_microns = 1,
    die_height_microns = 7,
    die_width_microns = 7,
    min_pin_distance = "0.2",
    placement_density = "0.95",
    synthesized_rtl = ":find_index_5000ps_model_unit_verilog_synth_asap7",
)

gds_write(
    name = "find_index_gds_asap7",
    implemented_rtl = ":find_index_place_and_route_asap7",
)

xls_dslx_library(
    name = "apfloat_fmac_dslx",
    srcs = ["apfloat_fmac.x"],
)

xls_dslx_fmt_test(
    name = "apfloat_fmac_dslx_fmt_test",
    src = "apfloat_fmac.x",
)

xls_dslx_library(
    name = "fp32_fmac_dslx",
    srcs = ["fp32_fmac.x"],
    deps = [":apfloat_fmac_dslx"],
)

xls_dslx_opt_ir(
    name = "fp32_fmac",
    dslx_top = "fp32_fmac",
    library = ":fp32_fmac_dslx",
)

xls_dslx_test(
    name = "fp32_fmac_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":fp32_fmac_dslx",
)

xls_dslx_library(
    name = "overflow_detect_dslx",
    srcs = ["overflow_detect.x"],
)

xls_dslx_test(
    name = "overflow_detect_dslx_test",
    srcs = ["overflow_detect.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "overflow_detect",
    srcs = ["overflow_detect.x"],
    dslx_top = "main",
    ir_file = "overflow_detect.ir",
    opt_ir_file = "overflow_detect.opt.ir",
)

xls_dslx_library(
    name = "nested_sel_dslx",
    srcs = ["nested_sel.x"],
)

xls_dslx_test(
    name = "nested_sel_dslx_test",
    srcs = ["nested_sel.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "nested_sel",
    srcs = ["nested_sel.x"],
    dslx_top = "main",
)

xls_dslx_opt_ir_test(
    name = "nested_sel_test",
    dep = ":nested_sel",
)

xls_dslx_library(
    name = "cubic_bezier_dslx",
    srcs = ["cubic_bezier.x"],
)

xls_dslx_test(
    name = "cubic_bezier_dslx_test",
    srcs = ["cubic_bezier.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "bitonic_sort_dslx",
    srcs = ["bitonic_sort.x"],
)

xls_dslx_test(
    name = "bitonic_sort_dslx_test",
    srcs = ["bitonic_sort.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "hack_cpu_dslx",
    srcs = ["hack_cpu.x"],
)

xls_dslx_test(
    name = "hack_cpu_dslx_test",
    srcs = ["hack_cpu.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_library(
    name = "serialized_decomposer_dslx",
    srcs = ["serialized_decomposer.x"],
)

xls_dslx_test(
    name = "serialized_decomposer_test",
    dslx_test_args = {"compare": "jit"},
    library = ":serialized_decomposer_dslx",
)

xls_dslx_ir(
    name = "serialized_decomposer_ir",
    dslx_top = "serialized_decomposer",
    ir_file = "serialized_decomposer.ir",
    library = ":serialized_decomposer_dslx",
)

xls_ir_opt_ir(
    name = "serialized_decomposer_opt_ir",
    src = "serialized_decomposer.ir",
    top = "__serialized_decomposer__serialized_decomposer_0_next",
)

xls_ir_verilog(
    name = "serialized_decomposer_sv",
    src = ":serialized_decomposer_opt_ir",
    codegen_args = {
        "module_name": "serialized_decomposer_top",
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
        "worst_case_throughput": "3",
    },
    verilog_file = "serialized_decomposer.sv",
)

verilog_library(
    name = "serialized_decomposer",
    srcs = [":serialized_decomposer.sv"],
    tags = ["DSLX"],
)

synthesize_rtl(
    name = "serialized_decomposer_verilog_synth",
    standard_cells = "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt_4x",
    tags = ["manual"],
    top_module = "serialized_decomposer_top",
    deps = [":serialized_decomposer"],
)

benchmark_synth(
    name = "serialized_decomposer_benchmark_synth",
    synth_target = ":serialized_decomposer_verilog_synth",
    tags = ["manual"],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)

xls_dslx_library(
    name = "passthrough_dslx",
    srcs = [
        "passthrough.x",
    ],
)

xls_dslx_test(
    name = "passthrough_dslx_test",
    srcs = ["passthrough.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_verilog(
    name = "passthrough_verilog",
    codegen_args = {
        "module_name": "passthrough",
        "delay_model": "unit",
        "pipeline_stages": "1",
        "reset": "rst",
    },
    dslx_top = "Passthrough",
    library = "passthrough_dslx",
    verilog_file = "passthrough.sv",
)

xls_dslx_library(
    name = "gcd_dslx",
    srcs = ["gcd.x"],
)

xls_dslx_test(
    name = "gcd_test",
    size = "small",
    srcs = ["gcd.x"],
    dslx_test_args = {"compare": "jit"},
)
