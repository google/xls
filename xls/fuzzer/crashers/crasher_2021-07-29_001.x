// Copyright 2021 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// would not converge.
//
// BEGIN_CONFIG
// issue: "was causing an optimizer hang -- attempts to constant fold the gate op"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "(); (bits[10]:0x2aa)"
//     args: "(); (bits[10]:0x3ff)"
//     args: "(); (bits[10]:0x2aa)"
//     args: "(); (bits[10]:0x0)"
//   }
// }
// END_CONFIG
fn main(x0: (), x1: (s10,)) -> (u11, u11, s10, u11, u11, u11) {
  let x3: u11 = u11:0x325;
  let x4: u11 = gate!((x3) != (x3), x3);
  let x5: u11 = clz(x3);
  let x6: s10 = (x1).0;
  let x7: u11 = !(x3);
  (x3, x4, x6, x7, x4, x3)
}
