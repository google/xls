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
// evaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[2]:0x2, bits[2]:0x2, bits[2]:0x3, bits[2]:0x2)
// evaluated unopt IR (JIT) =
//    (bits[2]:0x2, bits[2]:0x0, bits[2]:0x3, bits[2]:0x0)
//
// BEGIN_CONFIG
// exception: "// Result miscompare for sample 3:"
// issue: "https://github.com/google/xls/issues/375"
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
//   codegen_args: "--pipeline_stages=3"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x80_0000; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x21_a196_6158; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x10; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x8000_0000; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x3f_13a0_490c; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x10_0000; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x3c_0e29_e99f; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x100_0000; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x25_d273_3fcc; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x2f_6f53_87b2; bits[1]:0x0"
//     args: "bits[38]:0x1e_5920_3b09; bits[1]:0x1"
//     args: "bits[38]:0x800_0000; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x1000_0000; bits[1]:0x1"
//     args: "bits[38]:0x200; bits[1]:0x1"
//     args: "bits[38]:0x40_0000; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x17_b9ae_e47d; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x2_0000_0000; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x800_0000; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x26_23f7_c887; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0xb_61bc_5162; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0xe_68b7_5d40; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x80_0000; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x400_0000; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0xb_a268_d3f8; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x80; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x1"
//     args: "bits[38]:0x11_f89b_957d; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x1; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x4000_0000; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x30_ee9d_d60b; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x8000; bits[1]:0x0"
//     args: "bits[38]:0x0; bits[1]:0x1"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x0; bits[1]:0x0"
//     args: "bits[38]:0x10_a656_1fd0; bits[1]:0x0"
//     args: "bits[38]:0x10; bits[1]:0x0"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x15_5555_5555; bits[1]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; bits[1]:0x1"
//     args: "bits[38]:0x200; bits[1]:0x0"
//     args: "bits[38]:0x1_0000_0000; bits[1]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; bits[1]:0x0"
//     args: "bits[38]:0x9_d22a_658d; bits[1]:0x0"
//   }
// }
// END_CONFIG
type x16 = u1;
fn main(x0: s38, x1: u1) -> (u2, u2, u2, u2) {
  let x2: u2 = one_hot(x1, bool:true);
  let x3: u2 = -(x2);
  let x4: u2 = !(x2);
  let x5: u2 = (x2) - (x4);
  let x6: u2 = !(x3);
  let x7: bool = (x4)[x2+:bool];
  let x9: s38 = (x0) & (((x6) as s38));
  let x10: u2 = (x3) * (((x9) as u2));
  let x11: bool = xor_reduce(x10);
  let x12: u2 = -(x6);
  let x13: u11 = (((((x2) ++ (x12)) ++ (x5)) ++ (x1)) ++ (x12)) ++ (x2);
  let x14: u2 = (x12)[:];
  let x15: u11 = (((((x6) ++ (x11)) ++ (x6)) ++ (x3)) ++ (x4)) ++ (x4);
  let x17: x16[1] = [x1];
  let x18: x16[2] = (x17) ++ (x17);
  let x19: u2 = for (i, x): (u4, u2) in u4:0..u4:6 {
    x
  }(x4);
  let x20: u2 = (x4) | (x2);
  let x21: s38 = -(x0);
  let x22: u2 = (((x13) as u2)) - (x14);
  let x23: u11 = (x15) ^ (((x1) as u11));
  let x24: u2 = bit_slice_update(x2, x11, x7);
  let x25: u2 = -(x12);
  (x2, x10, x22, x10)
}
