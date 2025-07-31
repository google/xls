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
//     args: "bits[32]:0x2000_0000; bits[26]:0x8_0000"
//     args: "bits[32]:0x8; bits[26]:0x8_0000"
//     args: "bits[32]:0x200_0000; bits[26]:0x0"
//     args: "bits[32]:0x0; bits[26]:0x20_9008"
//     args: "bits[32]:0x2_0000; bits[26]:0x4"
//     args: "bits[32]:0x10; bits[26]:0x8_0000"
//     args: "bits[32]:0x1000; bits[26]:0x40"
//     args: "bits[32]:0x200; bits[26]:0x310_0302"
//     args: "bits[32]:0x400_0000; bits[26]:0x8000"
//     args: "bits[32]:0x5c0a_e196; bits[26]:0x2"
//     args: "bits[32]:0x1000; bits[26]:0x20_1000"
//     args: "bits[32]:0x1000; bits[26]:0x80_9000"
//     args: "bits[32]:0x4; bits[26]:0x40_9a94"
//     args: "bits[32]:0x4000; bits[26]:0x155_5555"
//     args: "bits[32]:0x4; bits[26]:0x60_2804"
//     args: "bits[32]:0x10_0000; bits[26]:0x10_0000"
//     args: "bits[32]:0x20_0000; bits[26]:0x20_0001"
//     args: "bits[32]:0x8_0000; bits[26]:0x24a_8110"
//     args: "bits[32]:0x5555_5555; bits[26]:0x8000"
//     args: "bits[32]:0x40_0000; bits[26]:0x49_c702"
//     args: "bits[32]:0xc899_d54e; bits[26]:0x159_dd04"
//     args: "bits[32]:0x8_0000; bits[26]:0x8_0000"
//     args: "bits[32]:0xaaaa_aaaa; bits[26]:0x2a8_aaaa"
//     args: "bits[32]:0x8; bits[26]:0x3_0208"
//     args: "bits[32]:0x100_0000; bits[26]:0x184_f010"
//     args: "bits[32]:0x400_0000; bits[26]:0x10_0000"
//     args: "bits[32]:0x23c2_7051; bits[26]:0x400"
//     args: "bits[32]:0x8000; bits[26]:0x4000"
//     args: "bits[32]:0x400_0000; bits[26]:0x100_0000"
//     args: "bits[32]:0x80; bits[26]:0x20d_0080"
//     args: "bits[32]:0x59bc_0f7d; bits[26]:0x1"
//     args: "bits[32]:0x200_0000; bits[26]:0x4000"
//     args: "bits[32]:0x80_0000; bits[26]:0x1e3_da97"
//     args: "bits[32]:0x400_0000; bits[26]:0x80"
//     args: "bits[32]:0x400_0000; bits[26]:0x21_403c"
//     args: "bits[32]:0x800_0000; bits[26]:0x82_2402"
//     args: "bits[32]:0x10; bits[26]:0x4"
//     args: "bits[32]:0x10; bits[26]:0x380_2454"
//     args: "bits[32]:0x5555_5555; bits[26]:0x1d7_5735"
//     args: "bits[32]:0xffff_ffff; bits[26]:0x3ff_77fe"
//     args: "bits[32]:0x5555_5555; bits[26]:0x145_155d"
//     args: "bits[32]:0x7fff_ffff; bits[26]:0x100_0000"
//     args: "bits[32]:0x80_0000; bits[26]:0x18c_2280"
//     args: "bits[32]:0x80; bits[26]:0x80"
//     args: "bits[32]:0xaaaa_aaaa; bits[26]:0x2ba_8e6a"
//     args: "bits[32]:0x8_0000; bits[26]:0x208_4061"
//     args: "bits[32]:0x9a76_1905; bits[26]:0x227_1d45"
//     args: "bits[32]:0x10; bits[26]:0x400"
//     args: "bits[32]:0xaaaa_aaaa; bits[26]:0x40_0000"
//     args: "bits[32]:0x6784_df29; bits[26]:0x384_ff29"
//     args: "bits[32]:0x400_0000; bits[26]:0x1308"
//     args: "bits[32]:0x80; bits[26]:0xc_0204"
//     args: "bits[32]:0x1_0000; bits[26]:0xa0_4084"
//     args: "bits[32]:0x200_0000; bits[26]:0x315_4084"
//     args: "bits[32]:0x2; bits[26]:0x2000"
//     args: "bits[32]:0x80; bits[26]:0x4"
//     args: "bits[32]:0x4000_0000; bits[26]:0x4"
//     args: "bits[32]:0x400; bits[26]:0x4_0000"
//     args: "bits[32]:0x8000_0000; bits[26]:0x201_500c"
//     args: "bits[32]:0x80; bits[26]:0x1c2_82eb"
//     args: "bits[32]:0x4_0000; bits[26]:0x100"
//     args: "bits[32]:0x100; bits[26]:0x20"
//     args: "bits[32]:0x80_0000; bits[26]:0x4c_724f"
//     args: "bits[32]:0xaaaa_aaaa; bits[26]:0x200"
//   }
// }
// END_CONFIG
type x19 = uN[0x9];
type x28 = uN[0x1];
fn main(x0: s32, x1: u26) -> u62 {
  let x2: u26 = -(x1);
  let x3: s39 = s39:0x8;
  let x4: u36 = (x3 as u39)[0x3+:u36];
  let x5: u26 = (x1)[0x0+:u26];
  let x6: u1 = xor_reduce(x2);
  let x7: s32 = one_hot_sel(x6, [x0]);
  let x8: u1 = one_hot_sel(x6, [x6]);
  let x9: u62 = (x4) ++ (x2);
  let x10: s16 = s16:0x8;
  let x11: u1 = rev(x6);
  let x12: u27 = u27:0x4000000;
  let x13: s11 = s11:0x4;
  let x14: u62 = rev(x9);
  let x15: u1 = (x0) > (x7);
  let x16: (u1, s39, s32) = (x15, x3, x7);
  let x17: u1 = (x10 as u16)[:-0xf];
  let x18: x19[0x4] = ((x4) as x19[0x4]);
  let x20: u1 = for (i, x): (u4, u1) in u4:0x0..u4:0x1 {
    x
  }(x11);
  let x21: s11 = (((x20) as s11)) << (if ((x13) >= (s11:0x9)) { (u11:0x9) } else { (x13 as u11) });
  let x22: u30 = (x0 as u32)[0x2:];
  let x23: s32 = -(x7);
  let x24: u1 = (((x23) as u1)) >> (if ((x6) >= (u1:0x0)) { (u1:0x0) } else { (x6) });
  let x25: u1 = one_hot_sel(x15, [x8]);
  let x26: s39 = for (i, x): (u4, s39) in u4:0x0..u4:0x0 {
    x
  }(x3);
  let x27: x28[0x1] = ((x8) as x28[0x1]);
  let x29: u1 = (x25)[x24+:u1];
  let x30: u53 = u53:0x800;
  x14
}
