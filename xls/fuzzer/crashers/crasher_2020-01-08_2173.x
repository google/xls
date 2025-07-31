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
//   codegen: true
//   codegen_args: "--generator=combinational"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[57]:0x400000000; bits[36]:0x40; bits[24]:0x80000; bits[12]:0xc34"
//     args: "bits[57]:0x20000; bits[36]:0x800; bits[24]:0x80; bits[12]:0x0"
//     args: "bits[57]:0x40000000; bits[36]:0x10000000; bits[24]:0x440058; bits[12]:0x555"
//     args: "bits[57]:0x10000; bits[36]:0x100000; bits[24]:0x26e9d; bits[12]:0x603"
//     args: "bits[57]:0x200000000000; bits[36]:0x1; bits[24]:0x201882; bits[12]:0x0"
//     args: "bits[57]:0x1000000; bits[36]:0x2; bits[24]:0x4; bits[12]:0x400"
//     args: "bits[57]:0x8000000; bits[36]:0x800000000; bits[24]:0x80000; bits[12]:0x4"
//     args: "bits[57]:0x40000000000000; bits[36]:0xfffffffff; bits[24]:0xf4164d; bits[12]:0x7ff"
//     args: "bits[57]:0x16515626c77b396; bits[36]:0x6089b41f3; bits[24]:0x584991; bits[12]:0xb28"
//     args: "bits[57]:0x1e16f6fbc8500a2; bits[36]:0x42a53786b; bits[24]:0xf0b7b7; bits[12]:0x51e"
//     args: "bits[57]:0x40; bits[36]:0x8000; bits[24]:0x83c973; bits[12]:0x3a"
//     args: "bits[57]:0x4000000; bits[36]:0xc6012867b; bits[24]:0xee19ce; bits[12]:0x30"
//     args: "bits[57]:0x8000000; bits[36]:0x400; bits[24]:0x20; bits[12]:0xaaa"
//     args: "bits[57]:0x1cc8b8ca9439cc; bits[36]:0x8d7c5d500; bits[24]:0xaa71d; bits[12]:0x9a"
//     args: "bits[57]:0xf138d2ac9b3dd5; bits[36]:0x800; bits[24]:0x1d0e82; bits[12]:0x542"
//     args: "bits[57]:0x200000; bits[36]:0x800; bits[24]:0x0; bits[12]:0x1"
//     args: "bits[57]:0x40000000000; bits[36]:0xc403280d0; bits[24]:0x200200; bits[12]:0x420"
//     args: "bits[57]:0x100000000000; bits[36]:0x400000000; bits[24]:0x0; bits[12]:0x274"
//     args: "bits[57]:0x8000; bits[36]:0x800000; bits[24]:0x800; bits[12]:0x100"
//     args: "bits[57]:0x8; bits[36]:0x10000; bits[24]:0x82c9c4; bits[12]:0x310"
//   }
// }
// END_CONFIG
fn main(x0: s57, x1: s36, x2: s24, x3: u12) -> (u62, u45, u62) {
    let x4: uN[6] = (x2 as u24)[0x12+:uN[6]];
    let x5: uN[84] = ((((((x3) ++ (x3)) ++ (x3)) ++ (x3)) ++ (x3)) ++ (x3)) ++ (x3);
    let x6: u62 = (u62:0x400000);
    let x7: uN[6] = (x3)[:0x6];
    let x8: s24 = for (i, x): (u4, s24) in (u4:0x0)..(u4:0x5) {
    x
  }(x2)
  ;
    let x9: uN[136] = ((x3) ++ (x6)) ++ (x6);
    let x10: u62 = clz(x6);
    let x11: uN[4] = (x1 as u36)[0x14+:uN[4]];
    let x12: uN[62] = (x6)[:];
    let x13: s36 = (x1) + ((x10 as s36));
    let x14: uN[260] = ((((x6) ++ (x3)) ++ (x6)) ++ (x6)) ++ (x10);
    let x15: s24 = for (i, x): (u4, s24) in (u4:0x0)..(u4:0x3) {
    x
  }(x8)
  ;
    let x16: uN[186] = ((x10) ++ (x6)) ++ (x6);
    let x17: uN[13] = one_hot(x3, (u1:0));
    let x18: u45 = (u45:0x8000000);
    let x19: u62 = for (i, x): (u4, u62) in (u4:0x0)..(u4:0x3) {
    x
  }(x10)
  ;
    let x20: s29 = (s29:0x40000);
    let x21: uN[57] = (x0 as u57)[:];
    let x22: uN[1320] = x10 ++ x18 ++ x10 ++ x10 ++ x18 ++ x10 ++ x19 ++ x6 ++ x10 ++ x6 ++ x18 ++ x10 ++ x19 ++ x19 ++ x6 ++ x6 ++ x10 ++ x19 ++ x18 ++ x6 ++ x3 ++ x3 ++ x10 ++ x6;
    let x23: s24 = for (i, x): (u4, s24) in (u4:0x0)..(u4:0x7) {
    x
  }(x15)
  ;
    let x24: uN[169] = ((x19) ++ (x18)) ++ (x10);
    (x6, x18, x6)
}
