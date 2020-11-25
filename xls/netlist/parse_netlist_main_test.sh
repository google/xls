#!/bin/bash -e
# Copyright 2020 The XLS Authors
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


# Test with an cell library
echo 'entries: { kind: INVERTER, name: "INV" input_names: "A" output_pin_list { pins { name: "ZN" function: "F" } } }' > "${TEST_TMPDIR}/fake_cell_library.textproto"
echo 'module main(i, o); input i; output o; INV inv_0(.A(i), .ZN(o)); endmodule' > "${TEST_TMPDIR}/netlist.v"

BINPATH=./xls/netlist/parse_netlist_main
$BINPATH "${TEST_TMPDIR}/netlist.v" "${TEST_TMPDIR}/fake_cell_library.textproto"

# Test without a cell library
echo 'module main(a0, a1, a2, a3, q0); input a0, a1, a2, a3; output q0; SB_LUT4 #(.LUT_INIT(16'"'"'h8000)) q0_lut (.I0(a0), .I1(a1), .I2(a2), .I3(a3), .O(q0)); endmodule' > "${TEST_TMPDIR}/netlist2.v"

$BINPATH "${TEST_TMPDIR}/netlist2.v"

echo "PASS"
