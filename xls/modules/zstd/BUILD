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
    tags = ["manual"],
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
    top_module = "WindowBuffer64",
    deps = [
        ":window_buffer_verilog_lib",
    ],
    tags = ["manual"],
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
    target_die_utilization_percentage = "10",
    tags = ["manual"],
)

