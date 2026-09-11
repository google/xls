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

# Example design demonstrating busperf config generation; see README.md.

load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_library", "xls_dslx_verilog")
load("//xls/experimental/busperf:build_defs.bzl", "busperf_yaml")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "bottleneck_dslx",
    srcs = ["bottleneck.x"],
)

xls_dslx_verilog(
    name = "bottleneck_no_stall_verilog",
    dslx_top = "BottleneckNoStall",
    library = ":bottleneck_dslx",
    verilog_file = "bottleneck_no_stall_verilog.sv",
    codegen_args = {
        "clock_period_ps": "10",
        "pipeline_stages": "1",
        "reset": "rst",
        "module_name": "bottleneck_no_stall",
        "materialize_internal_fifos": "true",
        "embed_child_block_signatures": "true",
    },
)

busperf_yaml(
    name = "bottleneck_no_stall_bus_yaml",
    signature = ":bottleneck_no_stall_verilog.sig.textproto",
    scope = "tb_bottleneck_no_stall.dut",
)

xls_dslx_verilog(
    name = "bottleneck_stall_verilog",
    dslx_top = "BottleneckStall",
    library = ":bottleneck_dslx",
    verilog_file = "bottleneck_stall_verilog.sv",
    codegen_args = {
        "clock_period_ps": "10",
        "pipeline_stages": "1",
        "reset": "rst",
        "module_name": "bottleneck_stall",
        "materialize_internal_fifos": "true",
        "embed_child_block_signatures": "true",
    },
)

busperf_yaml(
    name = "bottleneck_stall_bus_yaml",
    signature = ":bottleneck_stall_verilog.sig.textproto",
    scope = "tb_bottleneck_stall.dut",
)
