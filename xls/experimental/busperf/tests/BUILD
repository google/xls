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

# Runs the real pipeline on a hierarchical design and diffs against a golden
# YAML. Run :hier_test_update_golden to update it after an intentional change.

load("//xls/build_rules:xls_build_defs.bzl", "xls_diff_test", "xls_dslx_library")
load("//xls/experimental/busperf/build_rules:build_defs.bzl", "xls_busperf_setup")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "hier_dslx",
    srcs = ["hier.x"],
)

xls_busperf_setup(
    name = "hier",
    codegen_args = {
        "clock_period_ps": "10",
        "module_name": "hier",
        "pipeline_stages": "1",
    },
    dslx_top = "Parent",
    library = ":hier_dslx",
    reset = "rst",
    scope = "tb_hier.dut",
    testbench = "testdata/tb_hier.v",
    testbench_defines = {
        "DUT_MODULE": "hier",
        "TB_MODULE": "tb_hier",
        "VCD_NAME": "\"hier.vcd\"",
    },
    vcd_filename = "hier.vcd",
)

xls_diff_test(
    name = "hier_test",
    file = ":hier_bus_yaml",
    golden = "testdata/hier.busperf.yaml",
)
