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
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=9"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[7]:0x4; bits[5]:0x1; bits[59]:0x3ffffffffffffff; bits[60]:0x4000; bits[23]:0x2aaaaa; bits[13]:0x1000"
//     args: "bits[7]:0x1; bits[5]:0x3; bits[59]:0x2c3846000000c4a; bits[60]:0x40000; bits[23]:0x61a4b0; bits[13]:0xb00"
//     args: "bits[7]:0x4; bits[5]:0x0; bits[59]:0x8000; bits[60]:0x800000000000; bits[23]:0x10000; bits[13]:0x100"
//     args: "bits[7]:0x0; bits[5]:0x0; bits[59]:0x20000000000; bits[60]:0x800000000000; bits[23]:0x20000; bits[13]:0x200"
//     args: "bits[7]:0x4c; bits[5]:0x19; bits[59]:0x5c000401e9031a1; bits[60]:0xc7ae00f4df9ab0; bits[23]:0x6c5958; bits[13]:0x900"
//     args: "bits[7]:0x0; bits[5]:0x10; bits[59]:0x880280a080; bits[60]:0x21a0c8185fa128a; bits[23]:0x20000; bits[13]:0x100"
//     args: "bits[7]:0x0; bits[5]:0x4; bits[59]:0x10cc00580410230; bits[60]:0x391801200824671; bits[23]:0x547068; bits[13]:0x8a9"
//     args: "bits[7]:0x2a; bits[5]:0xa; bits[59]:0x8000; bits[60]:0x5134d1e61463579; bits[23]:0x2aaaaa; bits[13]:0x1673"
//     args: "bits[7]:0x1; bits[5]:0x0; bits[59]:0x100000; bits[60]:0x900a48712e20408; bits[23]:0x200; bits[13]:0x2"
//     args: "bits[7]:0x4; bits[5]:0x1; bits[59]:0x10000000; bits[60]:0x800000; bits[23]:0x2519c2; bits[13]:0x1915"
//     args: "bits[7]:0x0; bits[5]:0x0; bits[59]:0x25b03fd7a39f9c; bits[60]:0x4b60f6e74cbe49; bits[23]:0x2; bits[13]:0x1555"
//     args: "bits[7]:0x10; bits[5]:0x4; bits[59]:0x1000000000000; bits[60]:0x800; bits[23]:0x100130; bits[13]:0x80"
//     args: "bits[7]:0x7f; bits[5]:0x1f; bits[59]:0x76207b98551a264; bits[60]:0x40000000; bits[23]:0x7fffff; bits[13]:0x553"
//     args: "bits[7]:0x20; bits[5]:0x8; bits[59]:0x200000000000; bits[60]:0x400000; bits[23]:0x408350; bits[13]:0x1020"
//     args: "bits[7]:0x10; bits[5]:0x1; bits[59]:0x8000000; bits[60]:0xc0941201001a001; bits[23]:0x4d9091; bits[13]:0x146"
//     args: "bits[7]:0x2; bits[5]:0x0; bits[59]:0x2000000000000; bits[60]:0x100000000000000; bits[23]:0x2000; bits[13]:0x2"
//     args: "bits[7]:0x40; bits[5]:0x10; bits[59]:0x400000000000000; bits[60]:0x200000000000000; bits[23]:0x2000; bits[13]:0x400"
//     args: "bits[7]:0x0; bits[5]:0x0; bits[59]:0x681001020c0280e; bits[60]:0x9c209a283f0f59b; bits[23]:0x200000; bits[13]:0x800"
//     args: "bits[7]:0x2a; bits[5]:0xa; bits[59]:0x162c055c427c41; bits[60]:0x10000000000; bits[23]:0x30e8e5; bits[13]:0x40"
//     args: "bits[7]:0x1; bits[5]:0xc; bits[59]:0x32bcaaaa2abaaac; bits[60]:0x4; bits[23]:0x509207; bits[13]:0x400"
//   }
// }
// END_CONFIG
type x23 = u1;
type x24 = u1;fn x8(x9: u1) -> u1 {
    let x10: u1 = -(x9);
    let x11: uN[4] = (((x10) ++ (x10)) ++ (x10)) ++ (x9);
    let x12: s14 = (s14:0x400);
    let x13: (s14,) = (x12,);
    let x14: s14 = (x13).0x0;
    let x15: s44 = (s44:0x100000);
    let x16: u1 = clz(x10);
    let x17: uN[1] = (x10)[0x0+:uN[1]];
    let x18: u1 = one_hot_sel(x9, [x9]);
    let x19: uN[3] = ((x9) ++ (x10)) ++ (x16);
    let x20: s14 = for (i, x): (u4, s14) in (u4:0x0)..(u4:0x1) {
    x
  }(x12)
  ;
    let x21: u1 = one_hot_sel(x16, [x18]);
    let x22: uN[1] = (x9)[0x0+:uN[1]];
    x10
}
fn main(x0: u7, x1: s5, x2: u59, x3: u60, x4: u23, x5: s13) -> (s13, uN[15], s5, u7, uN[306], x24[0xa], s53, s13, u59, uN[15]) {
    let x6: uN[232] = (((((x3) ++ (x3)) ++ (x2)) ++ (x4)) ++ (x4)) ++ (x0);
    let x7: x24[0xa] = map(x23[0xa]:[(u1:0x0), (u1:0x0), (u1:0x1), (u1:0x0), (u1:0x0), (u1:0x0), (u1:0x1), (u1:0x0), (u1:0x0), (u1:0x1)], x8);
    let x25: u7 = for (i, x): (u4, u7) in (u4:0x0)..(u4:0x1) {
    x
  }(x0)
  ;
    let x26: uN[306] = ((((((((x25) ++ (x4)) ++ (x3)) ++ (x4)) ++ (x3)) ++ (x0)) ++ (x3)) ++ (x2)) ++ (x25);
    let x27: s13 = -(x5);
    let x28: u7 = ((x4 as u7)) + (x0);
    let x29: u7 = ((x3 as u7)) & (x25);
    let x30: (u59, u7) = (x2, x25);
    let x31: (u60,) = (x3,);
    let x32: uN[35] = ((((x0) ++ (x0)) ++ (x28)) ++ (x25)) ++ (x25);
    let x33: u59 = (x30).0x0;
    let x34: s53 = (s53:0x2000000000000);
    let x35: s13 = one_hot_sel(x25, [x5, x27, x27, x5, x27, x5, x5]);
    let x36: s13 = ((x29 as s13)) & (x35);
    let x37: s13 = !(x36);
    let x38: s13 = (x36) >> (if ((x27) >= ((s13:0x5))) { ((u13:0x5)) } else { (x27 as u13) });
    let x39: u14 = (u14:0x2000);
    let x40: uN[15] = one_hot(x39, (u1:1));
    let x41: u7 = (x25) ^ (x0);
    let x42: uN[31] = (x3)[:-0x1d];
    let x43: uN[5] = (x37 as u13)[-0x9:-0x4];
    let x44: s13 = one_hot_sel(x29, [x27, x38, x36, x37, x27, x37, x37]);
    (x35, x40, x1, x28, x26, x7, x34, x44, x2, x40)
}
