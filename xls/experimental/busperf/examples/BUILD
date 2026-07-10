# Copyright 2026 The XLS Authors
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

# Example designs demonstrating xls_busperf_setup (//xls/experimental/busperf);
# see README.md in this directory.

load("//xls/experimental/busperf/build_rules:build_defs.bzl", "xls_busperf_setup")
load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_library", "xls_dslx_test")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "bottleneck_dslx",
    srcs = ["bottleneck.x"],
)

xls_dslx_test(
    name = "bottleneck_dslx_test",
    srcs = ["bottleneck.x"],
    deps = [":bottleneck_dslx"],
)

xls_busperf_setup(
    name = "bottleneck_no_stall",
    children = ["Passthrough", "SlowConsumerNoStall"],
    codegen_args = {
        "clock_period_ps": "10",
        "materialize_internal_fifos": "true",
        "module_name": "bottleneck_no_stall",
        "pipeline_stages": "1",
    },
    dslx_top = "BottleneckNoStall",
    library = ":bottleneck_dslx",
    reset = "rst",
    scope = "tb_bottleneck_no_stall.dut",
    testbench = "tb_bottleneck.v",
    testbench_defines = {
        "DUT_MODULE": "bottleneck_no_stall",
        "TB_MODULE": "tb_bottleneck_no_stall",
        "VCD_NAME": "\"bottleneck_no_stall.vcd\"",
    },
    vcd_filename = "bottleneck_no_stall.vcd",
)

xls_busperf_setup(
    name = "bottleneck_stall",
    children = ["Passthrough", "SlowConsumerStall4"],
    codegen_args = {
        "clock_period_ps": "10",
        "materialize_internal_fifos": "true",
        "module_name": "bottleneck_stall",
        "pipeline_stages": "1",
    },
    dslx_top = "BottleneckStall",
    library = ":bottleneck_dslx",
    reset = "rst",
    scope = "tb_bottleneck_stall.dut",
    testbench = "tb_bottleneck.v",
    testbench_defines = {
        "DUT_MODULE": "bottleneck_stall",
        "TB_MODULE": "tb_bottleneck_stall",
        "VCD_NAME": "\"bottleneck_stall.vcd\"",
    },
    vcd_filename = "bottleneck_stall.vcd",
)
