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
//
// BEGIN_CONFIG
// issue: "https://github.com/google/xls/issues/222"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=9"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x1"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//     args: "bits[1]:0x0"
//   }
// }
// END_CONFIG
type x9 = u3;
type x21 = u1;
fn main(x0: u1) -> (u1, u1, x21[0x4]) {
  let x1: u1 = for (i, x): (u4, u1) in u4:0x0..u4:0x5 {
    x
  }(x0);
  let x2: u1 = (x0) != (x0);
  let x3: u1 = (x1) >> (x0);
  let x4: u1 = ((x1) != (u1:0x0)) && ((x3) != (u1:0x0));
  let x5: u1 = -(x1);
  let x6: u1 = (x5) << (if ((((x2) as u1)) >= (u1:0x0)) { (u1:0x0) } else { (((x2) as u1)) });
  let x7: u3 = ((x2) ++ (x3)) ++ (x6);
  let x8: x9[0x1] = ((x7) as x9[0x1]);
  let x10: u3 = clz(x7);
  let x11: u2 = (x10)[0x1+:u2];
  let x12: u1 = one_hot_sel(x1, [x1]);
  let x13: s30 = s30:0x1;
  let x14: u1 = for (i, x): (u4, u1) in u4:0x0..u4:0x7 {
    x
  }(x2);
  let x15: u4 = ((x0) ++ (x11)) ++ (x0);
  let x16: u1 = (x5) - (((x10) as u1));
  let x17: u1 = -(x16);
  let x18: u1 = rev(x0);
  let x19: s30 = one_hot_sel(x11, [x13, x13]);
  let x20: x21[0x4] = ((x15) as x21[0x4]);
  let x22: u2 = one_hot_sel(x2, [x11]);
  let x23: u49 = u49:0x1555555555555;
  let x24: s29 = s29:0x1fffffff;
  let x25: s29 = -(x24);
  let x26: u1 = (x16) ^ (((x7) as u1));
  let x27: u3 = !(x7);
  let x28: u4 = ctz(x15);
  let x29: x21[0x8] = (x20) ++ (x20);
  let x30: x21[0x8] = (x20) ++ (x20);
  let x31: u4 = for (i, x): (u4, u4) in u4:0x0..u4:0x2 {
    x
  }(x15);
  (x0, x3, x20)
}
