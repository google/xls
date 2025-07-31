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
//   codegen_args: "--pipeline_stages=4"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[62]:0x31e7_8642_b6d6_eccc; bits[25]:0x9f_389a; bits[24]:0x10_0000"
//     args: "bits[62]:0x4000; bits[25]:0x100; bits[24]:0x1_0000"
//     args: "bits[62]:0x40; bits[25]:0x81_0540; bits[24]:0x80_0043"
//     args: "bits[62]:0x200_0000; bits[25]:0x0; bits[24]:0x4000"
//     args: "bits[62]:0x4000; bits[25]:0x48a0; bits[24]:0x40"
//     args: "bits[62]:0x800; bits[25]:0x32_4c00; bits[24]:0x4"
//     args: "bits[62]:0x4_0000_0000; bits[25]:0x0; bits[24]:0x40_0000"
//     args: "bits[62]:0x80; bits[25]:0x4_2080; bits[24]:0x8"
//     args: "bits[62]:0x200_0000_0000; bits[25]:0x185_4437; bits[24]:0x1000"
//     args: "bits[62]:0x1fdf_1c9d_ad47_2bc2; bits[25]:0x0; bits[24]:0x480"
//     args: "bits[62]:0x2_0000_0000; bits[25]:0x1_0000; bits[24]:0xc8_0040"
//     args: "bits[62]:0x3fff_ffff_ffff_ffff; bits[25]:0x1ff_ffff; bits[24]:0x1"
//     args: "bits[62]:0x1fff_ffff_ffff_ffff; bits[25]:0x8000; bits[24]:0xff_73bd"
//     args: "bits[62]:0x10_0000_0000; bits[25]:0x800; bits[24]:0x7f_ffff"
//     args: "bits[62]:0x2000_0000; bits[25]:0x40; bits[24]:0x40_3920"
//     args: "bits[62]:0x2_0000_0000_0000; bits[25]:0x80_a84e; bits[24]:0x0"
//     args: "bits[62]:0x902_5143_bb27_1b6e; bits[25]:0x14f_2603; bits[24]:0x20"
//     args: "bits[62]:0x400_0000_0000; bits[25]:0xe3_eba9; bits[24]:0x1_0000"
//     args: "bits[62]:0x200_0000; bits[25]:0x12_12a8; bits[24]:0xa_13ec"
//     args: "bits[62]:0x8000; bits[25]:0x8_8000; bits[24]:0x8000"
//     args: "bits[62]:0x8000_0000; bits[25]:0x400; bits[24]:0x400"
//     args: "bits[62]:0x2a6f_6a9f_dedf_bd7a; bits[25]:0x81_f0e6; bits[24]:0x1"
//     args: "bits[62]:0x80_0000_0000_0000; bits[25]:0x195_4d40; bits[24]:0x8"
//     args: "bits[62]:0x1_0000_0000; bits[25]:0x40; bits[24]:0x7f_ffff"
//     args: "bits[62]:0x1000_0000; bits[25]:0x2_0000; bits[24]:0x2_0818"
//     args: "bits[62]:0x4; bits[25]:0x106_8004; bits[24]:0xa_d5ac"
//     args: "bits[62]:0x400_0000_0000_0000; bits[25]:0x91_e593; bits[24]:0x8_0000"
//     args: "bits[62]:0x10_0000_0000; bits[25]:0x200; bits[24]:0x20_0000"
//     args: "bits[62]:0x20_0000_0000; bits[25]:0x40_2228; bits[24]:0x20"
//     args: "bits[62]:0x10_0000_0000; bits[25]:0x21_2400; bits[24]:0xb0_1602"
//     args: "bits[62]:0x200_0000_0000; bits[25]:0x110_0890; bits[24]:0x55_5555"
//     args: "bits[62]:0x400_0000_0000; bits[25]:0xaa_aaaa; bits[24]:0xba_acaa"
//     args: "bits[62]:0x400_0000_0000_0000; bits[25]:0x200; bits[24]:0x80_0020"
//     args: "bits[62]:0x800_0000; bits[25]:0x2_0000; bits[24]:0x8000"
//     args: "bits[62]:0x3dd4_1010_6741_e636; bits[25]:0x4; bits[24]:0x21_0230"
//     args: "bits[62]:0x1000_0000_0000; bits[25]:0x4_0000; bits[24]:0x2000"
//     args: "bits[62]:0x400; bits[25]:0x400; bits[24]:0x10"
//     args: "bits[62]:0x8_0000_0000_0000; bits[25]:0x1ff_ffff; bits[24]:0x1_0000"
//     args: "bits[62]:0x800_0000_0000; bits[25]:0x32_2464; bits[24]:0x20"
//     args: "bits[62]:0x1_0000_0000; bits[25]:0x40; bits[24]:0x24_0629"
//     args: "bits[62]:0x229e_d58e_aa09_27cc; bits[25]:0xcd_132e; bits[24]:0x55_5555"
//     args: "bits[62]:0x10_0000_0000_0000; bits[25]:0x80_0401; bits[24]:0x8_0246"
//     args: "bits[62]:0x1555_5555_5555_5555; bits[25]:0x155_d555; bits[24]:0x5d_905d"
//     args: "bits[62]:0x8_0000; bits[25]:0x2; bits[24]:0x5_122f"
//     args: "bits[62]:0x40_0000_0000; bits[25]:0x1ff_ffff; bits[24]:0xd3_dcaf"
//     args: "bits[62]:0x35d5_9dec_e305_d9e4; bits[25]:0x1ab_cafc; bits[24]:0xab_cadd"
//     args: "bits[62]:0x100_0000_0000_0000; bits[25]:0x1200; bits[24]:0x68_d220"
//     args: "bits[62]:0x3fff_ffff_ffff_ffff; bits[25]:0x4000; bits[24]:0xaa_aaaa"
//     args: "bits[62]:0x10_0000_0000_0000; bits[25]:0x1205; bits[24]:0x89"
//     args: "bits[62]:0x100; bits[25]:0x1; bits[24]:0x60_135c"
//     args: "bits[62]:0x80_0000; bits[25]:0x20; bits[24]:0x3a_281d"
//     args: "bits[62]:0x8000; bits[25]:0x0; bits[24]:0x10_0000"
//     args: "bits[62]:0x10_0000; bits[25]:0x1_0000; bits[24]:0x4_0000"
//     args: "bits[62]:0x19f_e5c9_906d_a2aa; bits[25]:0x1; bits[24]:0x40"
//     args: "bits[62]:0x4000; bits[25]:0x400; bits[24]:0x2000"
//     args: "bits[62]:0x8_0000; bits[25]:0xaa_aaaa; bits[24]:0xab_aaaa"
//     args: "bits[62]:0x0; bits[25]:0x84_c910; bits[24]:0x16_0001"
//     args: "bits[62]:0x8000_0000_0000; bits[25]:0x4000; bits[24]:0x20"
//     args: "bits[62]:0x100_0000; bits[25]:0x1_0000; bits[24]:0x29_208a"
//     args: "bits[62]:0x2000; bits[25]:0x1a9_b2b3; bits[24]:0x9_54c0"
//     args: "bits[62]:0x800_0000_0000_0000; bits[25]:0x2000; bits[24]:0x88_0006"
//     args: "bits[62]:0x2_0000_0000; bits[25]:0x0; bits[24]:0x20_8100"
//     args: "bits[62]:0x80; bits[25]:0x400; bits[24]:0x40_0080"
//     args: "bits[62]:0x40_0000_0000; bits[25]:0x400; bits[24]:0x4000"
//     args: "bits[62]:0x3e15_8e83_7e1f_9bf9; bits[25]:0xde_f3fd; bits[24]:0xff_ffff"
//     args: "bits[62]:0x10_0000; bits[25]:0x14_4507; bits[24]:0x4_4587"
//     args: "bits[62]:0x4000_0000_0000; bits[25]:0x200; bits[24]:0x8"
//     args: "bits[62]:0x841_c9b3_d315_3ca7; bits[25]:0x155_5555; bits[24]:0x20"
//     args: "bits[62]:0x8_0000_0000_0000; bits[25]:0x44_a22f; bits[24]:0x20_0020"
//     args: "bits[62]:0x80_0000; bits[25]:0x40; bits[24]:0x80_0102"
//     args: "bits[62]:0x4_0000; bits[25]:0x4_0000; bits[24]:0x100"
//     args: "bits[62]:0x800; bits[25]:0x12c_6945; bits[24]:0x84_4945"
//     args: "bits[62]:0x2000_0000_0000; bits[25]:0x200; bits[24]:0x5_d0f3"
//     args: "bits[62]:0x294f_d086_8b20_3211; bits[25]:0x1a1_f8a0; bits[24]:0x1000"
//     args: "bits[62]:0x1000_0000_0000; bits[25]:0x1c6_8130; bits[24]:0x49_85f2"
//     args: "bits[62]:0x1fff_ffff_ffff_ffff; bits[25]:0x1_0000; bits[24]:0x4"
//     args: "bits[62]:0x40_0000; bits[25]:0x8_0000; bits[24]:0x8_b515"
//     args: "bits[62]:0x1000; bits[25]:0x1000; bits[24]:0x2000"
//     args: "bits[62]:0x10; bits[25]:0x120_80d9; bits[24]:0x5_0060"
//     args: "bits[62]:0x10; bits[25]:0x100_0000; bits[24]:0x40"
//     args: "bits[62]:0x240d_d883_4b36_c4db; bits[25]:0xb3_55e9; bits[24]:0x31_75e9"
//     args: "bits[62]:0x40; bits[25]:0x8_0000; bits[24]:0x40_0000"
//     args: "bits[62]:0x2000; bits[25]:0x1000; bits[24]:0x1"
//     args: "bits[62]:0x200_0000_0000_0000; bits[25]:0x4d_001c; bits[24]:0x20_0000"
//     args: "bits[62]:0x1c9d_3dd4_5cad_8d6e; bits[25]:0xad_0d44; bits[24]:0x4a_0d4b"
//     args: "bits[62]:0x20_0000; bits[25]:0x33_0004; bits[24]:0x80"
//     args: "bits[62]:0x2_0000_0000_0000; bits[25]:0x1; bits[24]:0x2309"
//     args: "bits[62]:0x1_0000; bits[25]:0x44_88b8; bits[24]:0x41_0000"
//     args: "bits[62]:0x800_0000_0000; bits[25]:0x12d_13a0; bits[24]:0x800"
//     args: "bits[62]:0x20; bits[25]:0x10; bits[24]:0x400"
//     args: "bits[62]:0x10; bits[25]:0x85_0b11; bits[24]:0x1a_8401"
//     args: "bits[62]:0x40_0000_0000_0000; bits[25]:0x80_0000; bits[24]:0x90_4047"
//     args: "bits[62]:0x4a4_861d_4310_92d1; bits[25]:0x10; bits[24]:0x4_9250"
//     args: "bits[62]:0x8000_0000_0000; bits[25]:0x8d12; bits[24]:0xb_6c06"
//     args: "bits[62]:0x8; bits[25]:0x64_4e3a; bits[24]:0x1000"
//     args: "bits[62]:0x400_0000_0000_0000; bits[25]:0x1ff_ffff; bits[24]:0x1000"
//     args: "bits[62]:0x2000_0000_0000; bits[25]:0x0; bits[24]:0x8_0012"
//     args: "bits[62]:0x4; bits[25]:0xa4_3144; bits[24]:0xa4_314c"
//     args: "bits[62]:0x20_0000_0000_0000; bits[25]:0x40_0000; bits[24]:0x21_5f68"
//     args: "bits[62]:0x800_0000_0000_0000; bits[25]:0x1ff_ffff; bits[24]:0xb8_33f9"
//     args: "bits[62]:0x1000_0000_0000_0000; bits[25]:0x400; bits[24]:0x4c_4460"
//     args: "bits[62]:0xfb7_5f54_f5d3_e99d; bits[25]:0x4_0000; bits[24]:0xd6_4884"
//     args: "bits[62]:0x10_0000_0000_0000; bits[25]:0x2000; bits[24]:0x80_4142"
//     args: "bits[62]:0x40; bits[25]:0x8_0040; bits[24]:0x10_0000"
//     args: "bits[62]:0x15c9_c9fd_b12e_3a40; bits[25]:0x4_0000; bits[24]:0x8_0000"
//     args: "bits[62]:0x2_0000_0000; bits[25]:0x0; bits[24]:0x10"
//     args: "bits[62]:0x800_0000_0000; bits[25]:0x1ff_ffff; bits[24]:0x401a"
//     args: "bits[62]:0x4000; bits[25]:0x1_0000; bits[24]:0x2_0000"
//     args: "bits[62]:0x80; bits[25]:0x8000; bits[24]:0x10_6406"
//     args: "bits[62]:0x4_0000_0000; bits[25]:0x188_3ac6; bits[24]:0x20"
//     args: "bits[62]:0x800_0000_0000; bits[25]:0x3_a842; bits[24]:0x2_0000"
//     args: "bits[62]:0x1000_0000; bits[25]:0x149_4154; bits[24]:0x9_613c"
//     args: "bits[62]:0x1; bits[25]:0x41; bits[24]:0x80_0000"
//     args: "bits[62]:0x8_0000; bits[25]:0x2; bits[24]:0x0"
//     args: "bits[62]:0x1000_0000_0000_0000; bits[25]:0x200; bits[24]:0x8"
//     args: "bits[62]:0x4000; bits[25]:0x6800; bits[24]:0xaa_aaaa"
//     args: "bits[62]:0x2_0000_0000_0000; bits[25]:0x1ff_ffff; bits[24]:0x5f_d794"
//     args: "bits[62]:0x1000_0000_0000_0000; bits[25]:0x8000; bits[24]:0xe9_3c79"
//     args: "bits[62]:0x80_0000_0000_0000; bits[25]:0x42_05a1; bits[24]:0x63_0045"
//     args: "bits[62]:0x4000; bits[25]:0x1_4408; bits[24]:0x2000"
//     args: "bits[62]:0x2000; bits[25]:0x109_2040; bits[24]:0x8"
//     args: "bits[62]:0x2ced_4821_d72b_0ca3; bits[25]:0x1; bits[24]:0x2831"
//     args: "bits[62]:0x2_0000_0000; bits[25]:0x1000; bits[24]:0x800"
//     args: "bits[62]:0x8; bits[25]:0x10_00ac; bits[24]:0x31_20ad"
//     args: "bits[62]:0x1_0000_0000_0000; bits[25]:0x54_142c; bits[24]:0x8_0000"
//     args: "bits[62]:0x1000_0000_0000_0000; bits[25]:0x100; bits[24]:0x8"
//     args: "bits[62]:0x100_0000_0000; bits[25]:0x20_0000; bits[24]:0x21_0301"
//     args: "bits[62]:0x4000_0000; bits[25]:0x50_1f41; bits[24]:0x10_1045"
//   }
// }
// END_CONFIG
fn main(x0: u62, x1: s25, x2: u24) -> (u63, s25, u1, (s25, u24, u25, s20, u1, u1, (u48,), u1, (u48,), u62), u62, u25, u1, u62, u62, u64, (s25, u24, u25, s20, u1, u1, (u48,), u1, (u48,), u62), u64) {
  let x3: u48 = (x2) ++ (x2);
  let x4: u62 = x0;
  let x5: u1 = ((x2) != (u24:0x0)) && ((x3) != (u48:0x0));
  let x6: u1 = (x2) != (((x0) as u24));
  let x7: (u48,) = (x3,);
  let x8: s20 = s20:0x8;
  let x9: u25 = u25:0x100000;
  let x10: (s25, u24, u25, s20, u1, u1, (u48,), u1, (u48,), u62) = (x1, x2, x9, x8, x6, x5, x7, x6, x7, x0);
  let x11: u48 = (x7).0;
  let x12: u25 = one_hot(x2, u1:0x1);
  let x13: u62 = one_hot_sel(x6, [x4]);
  let x14: u15 = (x1 as u25)[x9+:u15];
  let x15: u63 = (x0) ++ (x5);
  let x16: u62 = one_hot_sel(x6, [x13]);
  let x17: (u25, s20) = (x12, x8);
  let x18: (s25, u24, u25, s20, u1, u1, (u48,), u1, (u48,), u62) = for (i, x): (u4, (s25, u24, u25, s20, u1, u1, (u48,), u1, (u48,), u62)) in u4:0x0..u4:0x4 {
    x
  }(x10);
  let x19: u62 = for (i, x): (u4, u62) in u4:0x0..u4:0x4 {
    x
  }(x0);
  let x20: u62 = -(x0);
  let x21: s25 = for (i, x): (u4, s25) in u4:0x0..u4:0x5 {
    x
  }(x1);
  let x22: u13 = (x12)[-0x15:0x11];
  let x23: s20 = one_hot_sel(x6, [x8]);
  let x24: u1 = (x6) - (((x21) as u1));
  let x25: u1 = one_hot_sel(x24, [x24]);
  let x26: u25 = one_hot_sel(x25, [x9]);
  let x27: u21 = u21:0x800;
  let x28: u62 = one_hot_sel(x5, [x16]);
  let x29: u63 = (x28) ++ (x25);
  let x30: u25 = rev(x26);
  let x31: u64 = u64:0x20;
  let x32: u62 = for (i, x): (u4, u62) in u4:0x0..u4:0x1 {
    x
  }(x16);
  let x33: u64 = one_hot_sel(x6, [x31]);
  (x29, x21, x24, x10, x32, x12, x5, x32, x19, x33, x18, x33)
}
