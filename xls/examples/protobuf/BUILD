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

# DSLX Protobuf Examples

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "varint_encode_dslx",
    srcs = ["varint_encode.x"],
)

xls_dslx_test(
    name = "varint_encode_test",
    library = ":varint_encode_dslx",
)

xls_dslx_ir(
    name = "verint_encode_ir",
    dslx_top = "varint_encode_u32",
    ir_file = "varint_encode.ir",
    library = ":varint_encode_dslx",
)

xls_ir_opt_ir(
    name = "varint_encode_opt_ir",
    src = ":varint_encode.ir",
)

xls_ir_verilog(
    name = "varint_encode_sv",
    src = ":varint_encode_opt_ir.opt.ir",
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "asap7",
        "clock_period_ps": "4000",
        "use_system_verilog": "true",
    },
    verilog_file = "varint_encode_u32.sv",
)

xls_dslx_library(
    name = "varint_decode_dslx",
    srcs = ["varint_decode.x"],
    deps = [":varint_encode_dslx"],
)

xls_dslx_test(
    name = "varint_decode_test",
    dslx_test_args = {
        "compare": "jit",
    },
    library = ":varint_decode_dslx",
)

xls_dslx_library(
    name = "varint_streaming_decode",
    srcs = ["varint_streaming_decode.x"],
    deps = [":varint_decode_dslx"],
)

xls_dslx_test(
    name = "varint_streaming_decode_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":varint_streaming_decode",
)

# TODO(google/xls#889): enable targets when parameters can be pushed in IR conversion.
# xls_dslx_ir(
#     name = "varint_streaming_decode_ir",
#     dslx_top = "varint_streaming_u32_decode",
#     library = ":varint_streaming_decode",
# )
#
# xls_ir_opt_ir(
#     name = "varint_streaming_decode_opt_ir",
#     src = ":varint_streaming_decode_ir.ir",
# )
#
# xls_ir_verilog(
#     name = "varint_streaming_decode_sv",
#     src = ":varint_streaming_decode_opt_ir.opt.ir",
#     codegen_args = {
#         "module_name": "varint_streaming_decode",
#         "generator": "pipeline",
#         "delay_model": "asap7",
#         "pipeline_stages": "4",
#         "worst_case_throughput": "4",
#         "clock_period_ps": "5000",
#         "reset": "rst",
#         "reset_data_path": "true",
#         "reset_active_low": "false",
#         "reset_asynchronous": "false",
#         "flop_inputs": "false",
#         "flop_single_value_channels": "false",
#         "flop_outputs": "false",
#         "add_idle_output": "false",
#         "streaming_channel_data_suffix": "_data",
#         "streaming_channel_ready_suffix": "_ready",
#         "streaming_channel_valid_suffix": "_valid",
#         "use_system_verilog": "true",
#     },
#     verilog_file = "varint_streaming_decode.sv",
# )
#

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
