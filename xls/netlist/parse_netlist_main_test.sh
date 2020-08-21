#!/bin/bash
# Copyright 2020 Google LLC
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


# Find input files
echo 'entries: { kind: INVERTER, name: "INV" input_names: "A" output_pin_list { pins { name: "ZN" function: "F" } } }' > "${TEST_TMPDIR}/fake_cell_library.textproto"
echo 'module main(i, o); input i; output o; INV inv_0(.A(i), .ZN(o)); endmodule' > "${TEST_TMPDIR}/netlist.v"

BINPATH=./xls/netlist/parse_netlist_main
$BINPATH "${TEST_TMPDIR}/netlist.v" "${TEST_TMPDIR}/fake_cell_library.textproto" || exit -1

echo "PASS"
