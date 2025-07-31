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
//   codegen_args: "--pipeline_stages=2"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[26]:0x2000; bits[1]:0x0; bits[14]:0x263; bits[2]:0x3"
//     args: "bits[26]:0x4_0000; bits[1]:0x0; bits[14]:0x3280; bits[2]:0x0"
//     args: "bits[26]:0x0; bits[1]:0x0; bits[14]:0x2; bits[2]:0x1"
//     args: "bits[26]:0x80_0000; bits[1]:0x0; bits[14]:0x2472; bits[2]:0x2"
//     args: "bits[26]:0x20; bits[1]:0x0; bits[14]:0x2a; bits[2]:0x1"
//     args: "bits[26]:0x1_0000; bits[1]:0x0; bits[14]:0x1038; bits[2]:0x1"
//     args: "bits[26]:0x0; bits[1]:0x0; bits[14]:0x80; bits[2]:0x1"
//     args: "bits[26]:0x2; bits[1]:0x0; bits[14]:0x1102; bits[2]:0x2"
//     args: "bits[26]:0x40_0000; bits[1]:0x0; bits[14]:0x1; bits[2]:0x0"
//     args: "bits[26]:0x100; bits[1]:0x1; bits[14]:0x2487; bits[2]:0x2"
//     args: "bits[26]:0x3ff_ffff; bits[1]:0x1; bits[14]:0x3fef; bits[2]:0x0"
//     args: "bits[26]:0x200_0000; bits[1]:0x0; bits[14]:0x2898; bits[2]:0x1"
//     args: "bits[26]:0x1_0000; bits[1]:0x0; bits[14]:0x100; bits[2]:0x3"
//     args: "bits[26]:0x2; bits[1]:0x0; bits[14]:0x82; bits[2]:0x2"
//     args: "bits[26]:0x40; bits[1]:0x0; bits[14]:0x2; bits[2]:0x2"
//     args: "bits[26]:0x100; bits[1]:0x0; bits[14]:0x800; bits[2]:0x3"
//     args: "bits[26]:0x200; bits[1]:0x0; bits[14]:0x800; bits[2]:0x0"
//     args: "bits[26]:0x20_0000; bits[1]:0x1; bits[14]:0x2000; bits[2]:0x1"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0x40; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; bits[1]:0x1; bits[14]:0x1000; bits[2]:0x0"
//     args: "bits[26]:0x40; bits[1]:0x0; bits[14]:0x20cb; bits[2]:0x0"
//     args: "bits[26]:0x10_0000; bits[1]:0x0; bits[14]:0x20; bits[2]:0x1"
//     args: "bits[26]:0x1000; bits[1]:0x0; bits[14]:0x2000; bits[2]:0x0"
//     args: "bits[26]:0x200_0000; bits[1]:0x1; bits[14]:0x1; bits[2]:0x1"
//     args: "bits[26]:0x2_0000; bits[1]:0x0; bits[14]:0x10; bits[2]:0x2"
//     args: "bits[26]:0x200_0000; bits[1]:0x0; bits[14]:0x27a4; bits[2]:0x0"
//     args: "bits[26]:0x2_0000; bits[1]:0x0; bits[14]:0x1020; bits[2]:0x1"
//     args: "bits[26]:0x20; bits[1]:0x0; bits[14]:0xa13; bits[2]:0x3"
//     args: "bits[26]:0x8000; bits[1]:0x1; bits[14]:0x1555; bits[2]:0x1"
//     args: "bits[26]:0x80; bits[1]:0x0; bits[14]:0x2a2; bits[2]:0x0"
//     args: "bits[26]:0x2_0000; bits[1]:0x0; bits[14]:0x2; bits[2]:0x1"
//     args: "bits[26]:0x10_0000; bits[1]:0x0; bits[14]:0x3210; bits[2]:0x1"
//     args: "bits[26]:0x20_0000; bits[1]:0x1; bits[14]:0x0; bits[2]:0x0"
//     args: "bits[26]:0x1_0000; bits[1]:0x0; bits[14]:0x143; bits[2]:0x3"
//     args: "bits[26]:0x246_0b51; bits[1]:0x0; bits[14]:0x200; bits[2]:0x0"
//     args: "bits[26]:0x8; bits[1]:0x0; bits[14]:0x940; bits[2]:0x0"
//     args: "bits[26]:0x2000; bits[1]:0x1; bits[14]:0x2022; bits[2]:0x0"
//     args: "bits[26]:0x1; bits[1]:0x0; bits[14]:0x3fff; bits[2]:0x1"
//     args: "bits[26]:0x100_0000; bits[1]:0x0; bits[14]:0x40; bits[2]:0x1"
//     args: "bits[26]:0x8000; bits[1]:0x0; bits[14]:0x400; bits[2]:0x1"
//     args: "bits[26]:0x100; bits[1]:0x0; bits[14]:0x102; bits[2]:0x2"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0x1555; bits[2]:0x0"
//     args: "bits[26]:0x2_0000; bits[1]:0x0; bits[14]:0x200; bits[2]:0x3"
//     args: "bits[26]:0x400; bits[1]:0x0; bits[14]:0x908; bits[2]:0x0"
//     args: "bits[26]:0x8000; bits[1]:0x0; bits[14]:0x1000; bits[2]:0x1"
//     args: "bits[26]:0x0; bits[1]:0x0; bits[14]:0x2294; bits[2]:0x0"
//     args: "bits[26]:0x40_0000; bits[1]:0x1; bits[14]:0x3fff; bits[2]:0x0"
//     args: "bits[26]:0x400; bits[1]:0x1; bits[14]:0x400; bits[2]:0x1"
//     args: "bits[26]:0x2000; bits[1]:0x0; bits[14]:0x2391; bits[2]:0x0"
//     args: "bits[26]:0x254_658a; bits[1]:0x0; bits[14]:0x1fff; bits[2]:0x1"
//     args: "bits[26]:0x109_2f09; bits[1]:0x0; bits[14]:0x572; bits[2]:0x1"
//     args: "bits[26]:0x3ab_fb89; bits[1]:0x1; bits[14]:0x3b83; bits[2]:0x1"
//     args: "bits[26]:0x4; bits[1]:0x0; bits[14]:0xdf4; bits[2]:0x2"
//     args: "bits[26]:0x20; bits[1]:0x1; bits[14]:0x10a2; bits[2]:0x2"
//     args: "bits[26]:0x40; bits[1]:0x1; bits[14]:0x4a5; bits[2]:0x3"
//     args: "bits[26]:0x1000; bits[1]:0x1; bits[14]:0x400; bits[2]:0x2"
//     args: "bits[26]:0x2_0000; bits[1]:0x0; bits[14]:0x2aaa; bits[2]:0x0"
//     args: "bits[26]:0x1f6_8530; bits[1]:0x0; bits[14]:0x200; bits[2]:0x1"
//     args: "bits[26]:0x1_0000; bits[1]:0x0; bits[14]:0x100; bits[2]:0x0"
//     args: "bits[26]:0x80_0000; bits[1]:0x0; bits[14]:0x4; bits[2]:0x1"
//     args: "bits[26]:0x80; bits[1]:0x0; bits[14]:0x2aaa; bits[2]:0x1"
//     args: "bits[26]:0x10; bits[1]:0x0; bits[14]:0x3fff; bits[2]:0x1"
//     args: "bits[26]:0x40; bits[1]:0x0; bits[14]:0x40; bits[2]:0x0"
//     args: "bits[26]:0x4_0000; bits[1]:0x1; bits[14]:0x2aaa; bits[2]:0x3"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0x8; bits[2]:0x3"
//     args: "bits[26]:0x4000; bits[1]:0x1; bits[14]:0x100; bits[2]:0x1"
//     args: "bits[26]:0x20_0000; bits[1]:0x0; bits[14]:0x2aaa; bits[2]:0x1"
//     args: "bits[26]:0x4_0000; bits[1]:0x0; bits[14]:0x0; bits[2]:0x0"
//     args: "bits[26]:0x8_0000; bits[1]:0x0; bits[14]:0x80; bits[2]:0x1"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0x400; bits[2]:0x2"
//     args: "bits[26]:0x3ff_ffff; bits[1]:0x1; bits[14]:0x1555; bits[2]:0x1"
//     args: "bits[26]:0x40; bits[1]:0x1; bits[14]:0x2202; bits[2]:0x2"
//     args: "bits[26]:0x100_0000; bits[1]:0x0; bits[14]:0x2020; bits[2]:0x0"
//     args: "bits[26]:0x3ff_ffff; bits[1]:0x1; bits[14]:0x1555; bits[2]:0x2"
//     args: "bits[26]:0x1000; bits[1]:0x0; bits[14]:0x1080; bits[2]:0x2"
//     args: "bits[26]:0x4_0000; bits[1]:0x0; bits[14]:0x2aad; bits[2]:0x1"
//     args: "bits[26]:0x1000; bits[1]:0x0; bits[14]:0x1314; bits[2]:0x1"
//     args: "bits[26]:0x1; bits[1]:0x1; bits[14]:0x2000; bits[2]:0x1"
//     args: "bits[26]:0x20; bits[1]:0x0; bits[14]:0x1555; bits[2]:0x1"
//     args: "bits[26]:0x10; bits[1]:0x0; bits[14]:0x2000; bits[2]:0x0"
//     args: "bits[26]:0x40_0000; bits[1]:0x1; bits[14]:0x0; bits[2]:0x0"
//     args: "bits[26]:0x155_5555; bits[1]:0x1; bits[14]:0x3f66; bits[2]:0x3"
//     args: "bits[26]:0x4_0000; bits[1]:0x1; bits[14]:0x0; bits[2]:0x2"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0x2aaa; bits[2]:0x1"
//     args: "bits[26]:0x8; bits[1]:0x0; bits[14]:0x2356; bits[2]:0x1"
//     args: "bits[26]:0x800; bits[1]:0x0; bits[14]:0x800; bits[2]:0x0"
//     args: "bits[26]:0x4; bits[1]:0x0; bits[14]:0x0; bits[2]:0x0"
//     args: "bits[26]:0x33d_696c; bits[1]:0x0; bits[14]:0x20; bits[2]:0x1"
//     args: "bits[26]:0x40; bits[1]:0x0; bits[14]:0x371c; bits[2]:0x0"
//     args: "bits[26]:0x4; bits[1]:0x0; bits[14]:0x800; bits[2]:0x0"
//     args: "bits[26]:0x33f_8b54; bits[1]:0x0; bits[14]:0x641; bits[2]:0x0"
//     args: "bits[26]:0x4_0000; bits[1]:0x0; bits[14]:0x418; bits[2]:0x3"
//     args: "bits[26]:0x8000; bits[1]:0x0; bits[14]:0x300a; bits[2]:0x0"
//     args: "bits[26]:0x40_0000; bits[1]:0x0; bits[14]:0x400; bits[2]:0x0"
//     args: "bits[26]:0x400; bits[1]:0x0; bits[14]:0x1555; bits[2]:0x1"
//     args: "bits[26]:0x8; bits[1]:0x0; bits[14]:0x1587; bits[2]:0x0"
//     args: "bits[26]:0x4; bits[1]:0x0; bits[14]:0x39c3; bits[2]:0x0"
//     args: "bits[26]:0x10_0000; bits[1]:0x1; bits[14]:0xb39; bits[2]:0x0"
//     args: "bits[26]:0x1_0000; bits[1]:0x0; bits[14]:0x2600; bits[2]:0x0"
//     args: "bits[26]:0x8_0000; bits[1]:0x1; bits[14]:0x2c14; bits[2]:0x2"
//     args: "bits[26]:0x4; bits[1]:0x1; bits[14]:0x4; bits[2]:0x1"
//     args: "bits[26]:0x100; bits[1]:0x0; bits[14]:0x1a11; bits[2]:0x1"
//     args: "bits[26]:0x4000; bits[1]:0x0; bits[14]:0x1fff; bits[2]:0x1"
//     args: "bits[26]:0x155_5555; bits[1]:0x1; bits[14]:0x20; bits[2]:0x2"
//     args: "bits[26]:0x1_0000; bits[1]:0x0; bits[14]:0x80; bits[2]:0x0"
//     args: "bits[26]:0x40; bits[1]:0x0; bits[14]:0x10; bits[2]:0x1"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0x3e03; bits[2]:0x2"
//     args: "bits[26]:0x1_0000; bits[1]:0x0; bits[14]:0x43; bits[2]:0x0"
//     args: "bits[26]:0x20_0000; bits[1]:0x0; bits[14]:0x2809; bits[2]:0x1"
//     args: "bits[26]:0x100; bits[1]:0x0; bits[14]:0x2111; bits[2]:0x0"
//     args: "bits[26]:0xc3_ddfe; bits[1]:0x0; bits[14]:0x100; bits[2]:0x2"
//     args: "bits[26]:0x10e_8053; bits[1]:0x0; bits[14]:0x800; bits[2]:0x1"
//     args: "bits[26]:0x1000; bits[1]:0x0; bits[14]:0x1555; bits[2]:0x1"
//     args: "bits[26]:0x2000; bits[1]:0x0; bits[14]:0x1000; bits[2]:0x1"
//     args: "bits[26]:0x20_0000; bits[1]:0x0; bits[14]:0x1674; bits[2]:0x0"
//     args: "bits[26]:0x100; bits[1]:0x1; bits[14]:0x2a4a; bits[2]:0x1"
//     args: "bits[26]:0x2_0000; bits[1]:0x0; bits[14]:0x294; bits[2]:0x0"
//     args: "bits[26]:0x400; bits[1]:0x0; bits[14]:0x1555; bits[2]:0x0"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0x20; bits[2]:0x0"
//     args: "bits[26]:0x2cc_3489; bits[1]:0x0; bits[14]:0x1034; bits[2]:0x3"
//     args: "bits[26]:0x4000; bits[1]:0x0; bits[14]:0x610; bits[2]:0x0"
//     args: "bits[26]:0x10; bits[1]:0x0; bits[14]:0x10; bits[2]:0x1"
//     args: "bits[26]:0x100; bits[1]:0x0; bits[14]:0x400; bits[2]:0x2"
//     args: "bits[26]:0x2aa_aaaa; bits[1]:0x0; bits[14]:0xaaa; bits[2]:0x1"
//     args: "bits[26]:0x1_0000; bits[1]:0x1; bits[14]:0x800; bits[2]:0x3"
//     args: "bits[26]:0x80_0000; bits[1]:0x0; bits[14]:0x200; bits[2]:0x0"
//     args: "bits[26]:0x40_0000; bits[1]:0x0; bits[14]:0x301; bits[2]:0x1"
//     args: "bits[26]:0x40; bits[1]:0x0; bits[14]:0x1000; bits[2]:0x1"
//   }
// }
// END_CONFIG
const W32_V5 = u32:0x5;
type x10 = u1;
type x31 = u7;
fn main(x0: u26, x1: u1, x2: s14, x3: u2) -> (u35, u35, u35) {
  let x4: u35 = u35:0x400;
  let x5: u3 = one_hot(x3, u1:0x1);
  let x6: u35 = rev(x4);
  let x7: u35 = -(x6);
  let x8: u35 = !(x4);
  let x9: x10[0x23] = ((x6) as x10[0x23]);
  let x11: u35 = -(x6);
  let x12: (u35, u26, u3, u35, u35, u3, u2, u35, u35, u35, u35) = (x8, x0, x5, x11, x4, x5, x3, x6, x4, x7, x7);
  let x13: u35 = one_hot_sel(x5, [x8, x11, x6]);
  let x14: (u35, (u35, u26, u3, u35, u35, u3, u2, u35, u35, u35, u35)) = (x4, x12);
  let x15: s36 = s36:0x40000;
  let x16: s54 = s54:0x1000;
  let x17: u38 = (x5) ++ (x8);
  let x18: u35 = rev(x13);
  let x19: (u35, u26, u3, u35, u35, u3, u2, u35, u35, u35, u35) = for (i, x): (u4, (u35, u26, u3, u35, u35, u3, u2, u35, u35, u35, u35)) in u4:0x0..u4:0x2 {
    x
  }(x12);
  let x20: u2 = one_hot(x1, u1:0x1);
  let x21: u32 = u32:0x80000000;
  let x22: u35 = (x11)[x13+:u35];
  let x23: u1 = (((x20) as u35)) > (x11);
  let x24: u3 = one_hot_sel(x3, [x5, x5]);
  let x25: u15 = u15:0x40;
  let x26: s36 = (x15) >> (if ((((x23) as u36)) >= (u36:0xb)) { (u36:0xb) } else { (((x23) as u36)) });
  let x27: s63 = s63:0x200000000000;
  let x28: u2 = one_hot_sel(x5, [x3, x3, x3]);
  let x29: u3 = (x12).2;
  let x30: x31[W32_V5] = ((x6) as x31[W32_V5]);
  (x6, x11, x18)
}
