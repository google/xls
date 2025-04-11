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

"""Collection of external RTL sources

This package exports external verilog sources required by the ZSTD Decoder
in the verilog tests.

The exported files come from the Open Source library of the AXI4 and AXI4 lite
bus components released under MIT license.

The sources contain an implementation of the AXI4 crossbar which is used in the
verilog tests of the ZSTD Decoder to connect multiple AXI interfaces into a single
AXI interface that forms the IO of the ZSTD Decoder verilog wrapper used in the tests.

Source: https://github.com/alexforencich/verilog-axi
"""

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

exports_files(
    [
        "arbiter.v",
        "axi_crossbar.v",
        "axi_crossbar_addr.v",
        "axi_crossbar_rd.v",
        "axi_crossbar_wr.v",
        "axi_crossbar_wrapper.v",
        "axi_register_rd.v",
        "axi_register_wr.v",
        "priority_encoder.v",
    ],
)
