# Copyright 2024 The XLS Authors
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

"""Collection of simulation sources

This package exports verilog sources required by the ZSTD Decoder in the verilog
tests.

The sources contain:
 * Wrapper for the ZSTD Decoder used in the verilog tests
 * Verilog implementation of the channel fifo

"""

# pytype binary, library
load("@rules_hdl//verilog:providers.bzl", "verilog_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

exports_files(
    [
        "zstd_dec_wrapper.sv",
        "xls_fifo_wrapper.sv",
        "ram_demux_wrapper.v",
        "ram_1r1w.v",
    ],
)

verilog_library(
    name = "xls_fifo_wrapper_verilog_lib",
    srcs = [
        ":xls_fifo_wrapper.sv",
    ],
    tags = ["manual"],
)
