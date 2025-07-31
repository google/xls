// Copyright 2020 The XLS Authors
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
// BEGIN_CONFIG
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
//     args: "bits[45]:0x20; bits[50]:0x2_0b70_1041_0050; bits[38]:0x1_0000; bits[8]:0x5; bits[49]:0x4000_0000_0000; bits[13]:0x1555; bits[12]:0xe9c; bits[14]:0xa08; bits[35]:0x5_5540_0000"
//   }
// }
// END_CONFIG
type x10 = uN[0x31];
type x23 = uN[0x1];
fn main(x0: s45, x1: s50, x2: s38, x3: u8, x4: u49, x5: u13, x6: u12, x7: s14, x8: s35) -> (s14, u8) {
    let x9: x10[0x1] = (x4 as x10[0x1]);
    let x11: (u49, s50, s45, s45, u12, u49) = (x4, x1, x0, x0, x6, x4);
    let x12: s35 = -(x8);
    let x13: u12 = clz(x6);
    let x14: u12 = clz(x13);
    let x15: u12 = clz(x6);
    let x16: uN[13] = one_hot(x15, (u1:1));
    let x17: u49 = !(x4);
    let x18: u8 = ((x8 as u8)) | (x3);
    let x19: uN[42] = (x4)[x15+:uN[42]];
    let x20: u13 = ((x2 as u13)) + (x5);
    let x21: s26 = (s26:0x40000);
    let x22: x23[0x8] = (x18 as x23[0x8]);
    let x24: u1 = xor_reduce(x14);
    let x25: s14 = for (i, x): (u4, s14) in (u4:0x0)..(u4:0x3) {
    x
  }(x7)
  ;
    let x26: s50 = one_hot_sel(x24, [x1]);
    let x27: u8 = !(x18);
    let x28: uN[50] = (x1 as u50)[x3+:uN[50]];
    let x29: u49 = one_hot_sel(x24, [x4]);
    let x30: u8 = one_hot_sel(x24, [x18]);
    let x31: s26 = for (i, x): (u4, s26) in (u4:0x0)..(u4:0x6) {
    x
  }(x21)
  ;
    let x32: s50 = (x11).0x1;
    let x33: u1 = or_reduce(x4);
    let x34: u49 = (x11).0x0;
    let x35: u1 = xor_reduce(x20);
    (x25, x30)
}
