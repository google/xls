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
    ram_rewrites = [
        ":lz4_decoder_ram1r1w_rewrites.textproto",
    ],
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
        "ram_configurations": "ram_hb:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_comp}".format(
            rd_req = "ram_hb_read_req",
            rd_resp = "ram_hb_read_resp",
            wr_req = "ram_hb_write_req",
            wr_comp = "ram_hb_write_completion",
        ),
    },
)

# ---------------------------------------------------------------------------
# LZ4 encoder
# ---------------------------------------------------------------------------

xls_dslx_library(
    name = "dbe_lz4_encoder_dslx",
    srcs = [
        "lz4_encoder.x"
    ],
    deps = [
        ":dbe_common_dslx",
        "//xls/examples:ram_dslx",
        # decoder is referenced by test code
        ":dbe_lz4_decoder_dslx",
    ],
)

xls_dslx_test(
    name = "dbe_lz4_encoder_dslx_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = "dbe_lz4_encoder_dslx",
)

#8K hash specialization
xls_dslx_ir(
    name = "dbe_lz4_encoder_8k_ir",
    dslx_top = "encoder_8k",
    library = "dbe_lz4_encoder_dslx",
    ir_file = "dbe_lz4_encoder_8k_ir.ir",
)

xls_ir_opt_ir(
    name = "dbe_lz4_encoder_8k_opt_ir",
    src = "dbe_lz4_encoder_8k_ir.ir",
    top = "__lz4_encoder__encoder_8k__encoder_base_0__16_3_13_12_13_65536_8192_4_16_8_next",
    ram_rewrites = [
        ":lz4_encoder_8k_ram1r1w_rewrites.textproto",
    ],
)

xls_ir_verilog(
    name = "dbe_lz4_encoder_8k_verilog",
    src = "dbe_lz4_encoder_8k_opt_ir.opt.ir",
    verilog_file = "dbe_lz4_encoder_8k.v",
    codegen_args = {
        "module_name": "dbe_lz4_encoder",
        "delay_model": "unit",
        "pipeline_stages": "3",
        "worst_case_throughput": "3",
        "reset": "rst",
        "use_system_verilog": "false",
        "streaming_channel_data_suffix": "_data",
        "ram_configurations": ",".join([
            "ram_hb:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_comp}".format(
                rd_req = "ram_hb_read_req",
                rd_resp = "ram_hb_read_resp",
                wr_req = "ram_hb_write_req",
                wr_comp = "ram_hb_write_completion",
            ),
            "ram_ht:1R1W:{rd_req}:{rd_resp}:{wr_req}:{wr_comp}".format(
                rd_req = "ram_ht_read_req",
                rd_resp = "ram_ht_read_resp",
                wr_req = "ram_ht_write_req",
                wr_comp = "ram_ht_write_completion",
            ),
        ]),
    },
)
