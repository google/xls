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
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=combinational"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x13"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0x13"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x19"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x1c"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x19"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x18"
//     args: "bits[5]:0x1b"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x1b"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0xc"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x7"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x1f"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0x1e"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x14"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x2"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0x10"
//     args: "bits[5]:0x8"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0x1"
//     args: "bits[5]:0xf"
//     args: "bits[5]:0x18"
//     args: "bits[5]:0x15"
//     args: "bits[5]:0xa"
//     args: "bits[5]:0x4"
//     args: "bits[5]:0x0"
//     args: "bits[5]:0x8"
//   }
// }
// END_CONFIG
type x8 = uN[0x1];
type x21 = uN[0x1];
fn main(x0: s5) -> (u1, x8[0x2], u1, s5, x21[0x1], x21[0x1], s5, u1, s5, (s5, s5, u1, s5, u1, s5, s5, s5, s5, s5, s5, s5, u1, s5), s5, s5, x8[0x1], u5, (s5,), s5) {
  let x1: s5 = one_hot_sel(x0 as u5, [x0, x0, x0, x0, x0]);
  let x2: s5 = one_hot_sel(x1 as u5, [x1, x1, x1, x1, x1]);
  let x3: s5 = !(x1);
  let x4: s5 = (x3) ^ (x1);
  let x5: u1 = (x3) <= (x1);
  let x6: s5 = one_hot_sel(x2 as u5, [x1, x0, x4, x3, x0]);
  let x7: x8[0x1] = ((x5) as x8[0x1]);
  let x9: x8[0x2] = (x7) ++ (x7);
  let x10: s5 = for (i, x): (u4, s5) in u4:0x0..u4:0x4 {
    x
  }(x3);
  let x11: (s5,) = (x3,);
  let x12: s5 = one_hot_sel(x0 as u5, [x1, x6, x3, x3, x10]);
  let x13: u1 = ctz(x5);
  let x14: u5 = (x12 as u5)[x5+:u5];
  let x15: (s5, s5, u1, s5, u1, s5, s5, s5, s5, s5, s5, s5, u1, s5) = (x4, x4, x13, x10, x5, x2, x6, x0, x12, x6, x10, x1, x13, x3);
  let x16: u1 = ctz(x13);
  let x17: s5 = one_hot_sel(x16, [x3]);
  let x18: s5 = one_hot_sel(x3 as u5, [x4, x3, x10, x10, x2]);
  let x19: s47 = s47:0x4000;
  let x20: x21[0x1] = ((x16) as x21[0x1]);
  let x22: s60 = s60:0x7ffffffffffffff;
  let x23: s5 = one_hot_sel(x17 as u5, [x17, x17, x3, x2, x0]);
  let x24: s47 = one_hot_sel(x18 as u5, [x19, x19, x19, x19, x19]);
  (x13, x9, x5, x6, x20, x20, x18, x13, x17, x15, x23, x6, x7, x14, x11, x23)
}
