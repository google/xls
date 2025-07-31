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
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=7"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x26"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x2"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x31"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x4f"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x2c"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0xe"
//     args: "bits[7]:0x75"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x9"
//     args: "bits[7]:0x7f"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x7c"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x3f"
//     args: "bits[7]:0x5e"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x7"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x7f"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x2"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x2"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x7f"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x7f"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x4f"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x7f"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x2"
//     args: "bits[7]:0x40"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x7f"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x2"
//     args: "bits[7]:0x20"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x10"
//     args: "bits[7]:0x0"
//     args: "bits[7]:0x8"
//     args: "bits[7]:0x4"
//     args: "bits[7]:0x1"
//     args: "bits[7]:0x55"
//     args: "bits[7]:0x2a"
//     args: "bits[7]:0x55"
//   }
// }
// END_CONFIG
type x21 = u1;
fn main(x0: s7) -> (u1, s7, u1, u54, u27, s7, u27, u27, u1, s38, s38, u1, u54, u54, u1, s38, s7, (s48, u54, u27, u1, u54, u1, u27, u27), u1, u54, u27, s48, u27, u27) {
  let x1: u27 = u27:0x1000000;
  let x2: u27 = ctz(x1);
  let x3: u54 = (x1) ++ (x1);
  let x4: u27 = clz(x1);
  let x5: u27 = for (i, x): (u4, u27) in u4:0x0..u4:0x6 {
    x
  }(x1);
  let x6: u54 = (x3) + (((x4) as u54));
  let x7: u1 = (((x4) as u54)) != (x3);
  let x8: u54 = one_hot_sel(x7, [x6]);
  let x9: u54 = for (i, x): (u4, u54) in u4:0x0..u4:0x0 {
    x
  }(x3);
  let x10: s48 = s48:0x200000;
  let x11: u27 = for (i, x): (u4, u27) in u4:0x0..u4:0x1 {
    x
  }(x5);
  let x12: u27 = -(x11);
  let x13: s38 = s38:0x100;
  let x14: u27 = -(x5);
  let x15: u27 = (x13 as u38)[x11+:u27];
  let x16: u54 = for (i, x): (u4, u54) in u4:0x0..u4:0x3 {
    x
  }(x9);
  let x17: s48 = s48:0x400000;
  let x18: u1 = ((x11) != (u27:0x0)) || ((x15) != (u27:0x0));
  let x19: (s48, u54, u27, u1, u54, u1, u27, u27) = (x10, x3, x14, x7, x9, x7, x4, x4);
  let x20: x21[0x1] = ((x7) as x21[0x1]);
  let x22: u54 = ctz(x8);
  let x23: u54 = rev(x3);
  let x24: u1 = (x19).0x3;
  (x7, x0, x18, x9, x5, x0, x1, x4, x18, x13, x13, x7, x22, x23, x18, x13, x0, x19, x7, x23, x12, x17, x2, x11)
}
