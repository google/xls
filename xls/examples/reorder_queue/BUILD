# Copyright 2025 The XLS Authors
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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_dslx_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
)

xls_dslx_library(
    name = "reorder_queue_dslx",
    srcs = [
        "reorder_queue.x",
    ],
)

xls_dslx_test(
    name = "reorder_queue_test",
    library = ":reorder_queue_dslx",
)

xls_dslx_verilog(
    name = "reorder_queue_verilog",
    codegen_args = {
        "module_name": "reorder_queue",
        "generator": "pipeline",
        "delay_model": "asap7",
        "pipeline_stages": "1",
        "input_valid_signal": "input_valid",
        "output_valid_signal": "output_valid",
        "reset": "rst",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_rdy",
        "streaming_channel_valid_suffix": "_vld",
        "use_system_verilog": "true",
    },
    dslx_top = "reorder_queue_32_16",
    library = ":reorder_queue_dslx",
    verilog_file = "reorder_queue.sv",
)
