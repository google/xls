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

# Build rules for XLS DBE/LZ4 algorithm implementation.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_ir",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
    "xls_dslx_test",
)

load("@xls_pip_deps//:requirements.bzl", "requirement")
load("//xls/build_rules:cocotb_xls_test.bzl", "cocotb_xls_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

xls_dslx_library(
    name = "dbe_common_dslx",
    srcs = [
        "common.x",
        "common_test.x"
    ],
)

py_library(
    name = "dbe_common_test_py",
    srcs = [
        "dbe_common_test.py",
    ],
    imports = [
        "."
    ],
    deps = [
        "//xls/modules/dbe/scripts:dbe",
        requirement("cocotb"),
    ],
)

# ---------------------------------------------------------------------------
# LZ4 decoder
# ---------------------------------------------------------------------------

xls_dslx_library(
    name = "dbe_lz4_decoder_dslx",
    srcs = [
        "lz4_decoder.x"
    ],
    deps = [
        ":dbe_common_dslx",
        "//xls/examples:ram_dslx",
    ],
)

xls_dslx_test(
    name = "dbe_lz4_decoder_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "dbe_lz4_decoder_dslx",
)

xls_dslx_ir(
    name = "dbe_lz4_decoder_ir",
    dslx_top = "decoder",
    library = "dbe_lz4_decoder_dslx",
    ir_file = "dbe_lz4_decoder_ir.ir",
)

xls_ir_opt_ir(
    name = "dbe_lz4_decoder_opt_ir",
    src = "dbe_lz4_decoder_ir.ir",
    top = "__lz4_decoder__decoder__decoder_base_0__16_16_8_next",
)

xls_ir_verilog(
    name = "dbe_lz4_decoder_verilog",
    src = "dbe_lz4_decoder_opt_ir.opt.ir",
    verilog_file = "dbe_lz4_decoder.v",
    codegen_args = {
        "module_name": "dbe_lz4_decoder",
        "delay_model": "unit",
        "pipeline_stages": "3",
        "reset": "rst",
        "use_system_verilog": "false",
        "streaming_channel_data_suffix": "_data",
        "io_constraints": ",".join([
            "lz4_decoder__o_ram_hb_rd_req:send:lz4_decoder__i_ram_hb_rd_resp:recv:1:none",
            "lz4_decoder__o_ram_hb_wr_req:send:lz4_decoder__i_ram_hb_wr_resp:recv:1:none",
        ]),
    },
)

cocotb_xls_test(
    name = "dbe_lz4_decoder_cocotb_test",
    sim_name = "icarus",
    hdl_toplevel = "lz4_decoder_wrap",
    hdl_toplevel_lang = "verilog",
    test_module = [
        "lz4_decoder_test.py",
    ],
    verilog_sources = [
        ":dbe_lz4_decoder.v",
        "ram_model.v",
        "lz4_decoder_wrap.v",
    ],
    timescale = {
        "unit": "1ns",
        "precission": "1ps",
    },
    deps = [
        "dbe_common_test_py",
        "//xls/modules/dbe/scripts:dbe",
        "//xls/simulation/cocotb:cocotb_xls",
        requirement("cocotb"),
        requirement("cocotb_bus"),
    ],
)
