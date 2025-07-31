// Copyright 2022 The XLS Authors
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
// evaluated opt IR (JIT) =
//    (bits[39]:0x0, bits[38]:0x0, bits[18]:0x1, bits[20]:0x5_5555)
// evaluated opt IR (interpreter), evaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[39]:0x10_0000_0000, bits[38]:0x0, bits[18]:0x1, bits[20]:0x5_5555)
//
// BEGIN_CONFIG
// exception: "// Result miscompare for sample 0:"
// issue: "https://github.com/google/xls/issues/641"
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
//   codegen_args: "--pipeline_stages=10"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x18_b7a5_95e2; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1e_6f23_5a7c; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x4_47cf_2fb4; bits[1]:0x1"
//     args: "bits[38]:0xf_4452_4f80; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x40_0000; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x8; bits[1]:0x1"
//     args: "bits[38]:0x40_0000; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x24_60c7_315f; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1_4000_3a41; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x80_0000; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x16_3cc2_8b80; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x19_7045_5aae; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x2f_2c71_9ace; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x0"
//     args: "bits[38]:0x39_b72f_7a72; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x31_f176_1999; bits[1]:0x1"
//     args: "bits[38]:0x8; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x14_8146_c914; bits[1]:0x0"
//     args: "bits[38]:0x23_8f9a_52fa; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x26_23f4_70fe; bits[1]:0x0"
//     args: "bits[38]:0x4_0000; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x80; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x3_94ea_f180; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0xc_c787_58d1; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x0"
//     args: "bits[38]:0x17_e15f_9567; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x100; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x400_0000; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2c_b0de_5045; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x2000; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x7_806b_2cfd; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x1_0000; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x3c_a6fe_bb7e; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x1d_ab13_7aa5; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x800; bits[1]:0x0"
//     args: "bits[38]:0x10_0000; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2_0000_0000; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x200_0000; bits[1]:0x0"
//   }
// }
// END_CONFIG
type x26 = u10;
fn main(x0: u38, x1: s1) -> (u39, u38, u18, u20) {
  let x2: u38 = (x0) - (x0);
  let x3: u2 = u2:0x2;
  let x4: u2 = (x3) - (((x2) as u2));
  let x5: u38 = (x0) >> (if (x2) >= (u38:6) { u38:6 } else { x2 });
  let x6: uN[154] = ((((x0) ++ (x5)) ++ (x5)) ++ (x2)) ++ (x4);
  let x7: u38 = -(x2);
  let x8: u20 = (x5)[0+:u20];
  let x9: u2 = (((x7) as u2)) | (x4);
  let x10: u38 = for (i, x): (u4, u38) in u4:0..u4:0 {
    x
  }(x5);
  let x11: bool = (x5) != (((x6) as u38));
  let x12: (uN[154], bool) = (x6, x11);
  let x13: u38 = (x5) + (x10);
  let x14: s51 = s51:0x40_0000;
  let x15: bool = (x9)[1+:bool];
  let x16: u20 = (((x14) as u20)) + (x8);
  let x17: u2 = -(x9);
  let x18: u3 = one_hot(x4, bool:true);
  let x19: (u2, u3, u38) = (x9, x18, x0);
  let x20: u18 = (x6)[x11+:u18];
  let x21: u20 = clz(x8);
  let x22: u38 = clz(x0);
  let x23: bool = ctz(x15);
  let x24: u38 = (((x20) as u38)) | (x5);
  let x25: u20 = (x16)[x3+:u20];
  let x27: x26[2] = ((x25) as x26[2]);
  let x28: u14 = (x25)[6+:u14];
  let x29: u2 = (x17) | (((x20) as u2));
  let x30: u39 = one_hot(x24, bool:false);
  let x31: x26 = (x27)[if (x6) >= (uN[154]:1) { uN[154]:1 } else { x6 }];
  (x30, x2, x20, x8)
}
