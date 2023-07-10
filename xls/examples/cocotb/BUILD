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

load("@xls_pip_deps//:requirements.bzl", "requirement")
load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_verilog")
load("//xls/build_rules:cocotb_xls_test.bzl", "cocotb_xls_test")

xls_dslx_verilog(
    name = "running_counter_verilog",
    srcs = ["running_counter.x"],
    deps = [],
    codegen_args = {
        "delay_model": "unit",
        "clock_period_ps": "5000",
        "reset": "rst",
        "module_name": "RunningCounter",
        "use_system_verilog": "false",
        "streaming_channel_data_suffix": "_data",
    },
    dslx_top = "RunningCounter",
    verilog_file = "running_counter.v",
)

cocotb_xls_test(
    name = "running_counter_cocotb",
    hdl_toplevel = "RunningCounter",
    hdl_toplevel_lang = "verilog",
    test_module = [
        "cocotb_running_counter.py",
    ],
    verilog_sources = [
        "dumpvcd.v",
        ":running_counter.v",
    ],
    timescale = {
        "unit": "1ns",
        "precission": "1ps",
    },
    deps = [
        requirement("cocotb"),
        requirement("cocotb_bus"),
        "//xls/simulation/cocotb:cocotb_xls",
    ],
)
