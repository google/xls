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
// issue: "https://github.com/google/xls/issues/223"
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
//     args: "bits[24]:0x7f_ffff; bits[39]:0x3f_8eff_4082"
//     args: "bits[24]:0xaa_aaaa; bits[39]:0x1"
//     args: "bits[24]:0x0; bits[39]:0x10"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x2000_0000"
//     args: "bits[24]:0xaa_aaaa; bits[39]:0x80"
//     args: "bits[24]:0x4_0000; bits[39]:0x2_0000_0080"
//     args: "bits[24]:0x8; bits[39]:0x12_0004_5555"
//     args: "bits[24]:0x800; bits[39]:0x10_b497_4b8a"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x1_0000_0000"
//     args: "bits[24]:0x4_0000; bits[39]:0x4"
//     args: "bits[24]:0xb4_2eaf; bits[39]:0x2000"
//     args: "bits[24]:0x2000; bits[39]:0x24_5000_1c80"
//     args: "bits[24]:0x4000; bits[39]:0xa280_2500"
//     args: "bits[24]:0x80_0000; bits[39]:0x40_4000_37e6"
//     args: "bits[24]:0x10; bits[39]:0x8_0000_0000"
//     args: "bits[24]:0xbe_3a17; bits[39]:0x80_0000"
//     args: "bits[24]:0x2000; bits[39]:0x80"
//     args: "bits[24]:0x36_3707; bits[39]:0x53_0393_d042"
//     args: "bits[24]:0x0; bits[39]:0x2000"
//     args: "bits[24]:0x8; bits[39]:0x8_0000_0000"
//     args: "bits[24]:0x1_0000; bits[39]:0x6_80a2_4545"
//     args: "bits[24]:0x800; bits[39]:0x9_5374_6105"
//     args: "bits[24]:0x400; bits[39]:0x800_0000"
//     args: "bits[24]:0x8; bits[39]:0x40_0806_0104"
//     args: "bits[24]:0x10_0000; bits[39]:0x48_1599_0c56"
//     args: "bits[24]:0x0; bits[39]:0x40_0000_0000"
//     args: "bits[24]:0x400; bits[39]:0x21_5602_4021"
//     args: "bits[24]:0x2000; bits[39]:0xc_3007_0840"
//     args: "bits[24]:0x4_0000; bits[39]:0x30_c444_b42d"
//     args: "bits[24]:0x4_0000; bits[39]:0x42_02d0_36a8"
//     args: "bits[24]:0x100; bits[39]:0x2_4092_9761"
//     args: "bits[24]:0x400; bits[39]:0x8_0240_0edc"
//     args: "bits[24]:0xcb_64ed; bits[39]:0x45_6356_bb78"
//     args: "bits[24]:0x80_0000; bits[39]:0x1000_0000"
//     args: "bits[24]:0x10_0000; bits[39]:0xe_6c38_4572"
//     args: "bits[24]:0x80; bits[39]:0x40_0101"
//     args: "bits[24]:0x20; bits[39]:0x4"
//     args: "bits[24]:0x70_8f30; bits[39]:0x6a_e28c_7080"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x4_0000_0000"
//     args: "bits[24]:0x20_0000; bits[39]:0x10_c409_7bff"
//     args: "bits[24]:0x8000; bits[39]:0x4_0000_0000"
//     args: "bits[24]:0x40; bits[39]:0x20"
//     args: "bits[24]:0x80; bits[39]:0x2"
//     args: "bits[24]:0x800; bits[39]:0x1_0000_0000"
//     args: "bits[24]:0x10_0000; bits[39]:0xa_41ac_0a1c"
//     args: "bits[24]:0x2000; bits[39]:0x40_1000_5556"
//     args: "bits[24]:0x8_0000; bits[39]:0x5a_48b3_ec67"
//     args: "bits[24]:0x55_5555; bits[39]:0x2a_a8eb_aaaa"
//     args: "bits[24]:0x80_0000; bits[39]:0x200_0000"
//     args: "bits[24]:0x40; bits[39]:0x40"
//     args: "bits[24]:0x20; bits[39]:0x80_0000"
//     args: "bits[24]:0x80; bits[39]:0x2440_6226"
//     args: "bits[24]:0x1; bits[39]:0x32_2581_682c"
//     args: "bits[24]:0xba_288e; bits[39]:0x4000"
//     args: "bits[24]:0x80_0000; bits[39]:0x4_0000_0000"
//     args: "bits[24]:0x0; bits[39]:0x1_0000"
//     args: "bits[24]:0xaa_aaaa; bits[39]:0x3f_ffff_ffff"
//     args: "bits[24]:0xff_ffff; bits[39]:0x4_0000_0000"
//     args: "bits[24]:0x55_5555; bits[39]:0x1a_ba09_eb7b"
//     args: "bits[24]:0x2_0000; bits[39]:0x1_0008_0c11"
//     args: "bits[24]:0xa5_2ff6; bits[39]:0x1"
//     args: "bits[24]:0x2f_57cb; bits[39]:0x12_8140_f4a4"
//     args: "bits[24]:0x2_0000; bits[39]:0x800_0000"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x200"
//     args: "bits[24]:0x2; bits[39]:0x30_081f_004c"
//     args: "bits[24]:0x100; bits[39]:0x80_0000"
//     args: "bits[24]:0x8; bits[39]:0x2_1005_0022"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x1_0000_0000"
//     args: "bits[24]:0x4_f5ca; bits[39]:0x19_7ed5_0cb5"
//     args: "bits[24]:0x1000; bits[39]:0x21_c88c_029a"
//     args: "bits[24]:0x40_0000; bits[39]:0x800"
//     args: "bits[24]:0x7f_ffff; bits[39]:0xd_77bf_8093"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x4"
//     args: "bits[24]:0x400; bits[39]:0x100"
//     args: "bits[24]:0x10; bits[39]:0x1000"
//     args: "bits[24]:0x80; bits[39]:0x73_2032_30b2"
//     args: "bits[24]:0x20; bits[39]:0x800"
//     args: "bits[24]:0x10_0000; bits[39]:0x2000"
//     args: "bits[24]:0x4_0000; bits[39]:0x400_0000"
//     args: "bits[24]:0x4_0000; bits[39]:0x40_0000_0000"
//     args: "bits[24]:0x0; bits[39]:0x9_0802_2ed6"
//     args: "bits[24]:0xff_ffff; bits[39]:0x40"
//     args: "bits[24]:0x100; bits[39]:0x40_0000_0000"
//     args: "bits[24]:0x200; bits[39]:0x4_0000"
//     args: "bits[24]:0x20; bits[39]:0x8_0000"
//     args: "bits[24]:0x1; bits[39]:0x46_902a_a049"
//     args: "bits[24]:0x100; bits[39]:0x48_a2fa_8e41"
//     args: "bits[24]:0x4_0000; bits[39]:0x2"
//     args: "bits[24]:0x1000; bits[39]:0x20"
//     args: "bits[24]:0x4000; bits[39]:0xa401_6aa9"
//     args: "bits[24]:0x4_0000; bits[39]:0x2_0040_0030"
//     args: "bits[24]:0x80_0000; bits[39]:0x10"
//     args: "bits[24]:0x80_0000; bits[39]:0x3f_ffff_ffff"
//     args: "bits[24]:0x20; bits[39]:0x10_4030_0248"
//     args: "bits[24]:0x2_0000; bits[39]:0x4_0000"
//     args: "bits[24]:0x80; bits[39]:0x2_0000_0000"
//     args: "bits[24]:0xbc_8bea; bits[39]:0x1000_0000"
//     args: "bits[24]:0x400; bits[39]:0x27_1f40_71b0"
//     args: "bits[24]:0x1; bits[39]:0x26_514a_31a1"
//     args: "bits[24]:0x80_0000; bits[39]:0x10_0000"
//     args: "bits[24]:0x200; bits[39]:0x5b_a027_42c1"
//     args: "bits[24]:0x4; bits[39]:0x40_0000_0000"
//     args: "bits[24]:0xaa_aaaa; bits[39]:0x800_0000"
//     args: "bits[24]:0x80; bits[39]:0x1000_0000"
//     args: "bits[24]:0x3d_8609; bits[39]:0x1000_0000"
//     args: "bits[24]:0xd3_daa1; bits[39]:0x6b_edf8_0083"
//     args: "bits[24]:0x800; bits[39]:0x27_f90e_75e6"
//     args: "bits[24]:0x20; bits[39]:0x20_0000"
//     args: "bits[24]:0x80_0000; bits[39]:0x42_5176_8432"
//     args: "bits[24]:0x20; bits[39]:0x10_3fff"
//     args: "bits[24]:0x4_0000; bits[39]:0x40"
//     args: "bits[24]:0xa6_5762; bits[39]:0x46_731d_c99c"
//     args: "bits[24]:0x1000; bits[39]:0x18_0810_a008"
//     args: "bits[24]:0x55_5555; bits[39]:0x2a_aaaa_8080"
//     args: "bits[24]:0x0; bits[39]:0x28_fce7_5058"
//     args: "bits[24]:0x4000; bits[39]:0x40"
//     args: "bits[24]:0x80_0000; bits[39]:0x8_00a0_4000"
//     args: "bits[24]:0x80_0000; bits[39]:0x100_0000"
//     args: "bits[24]:0x20_0000; bits[39]:0x10_0a00_2ea8"
//     args: "bits[24]:0x80; bits[39]:0x7f_ffff_ffff"
//     args: "bits[24]:0x8_0000; bits[39]:0x10_0000_0000"
//     args: "bits[24]:0x40_0000; bits[39]:0x3f_ffff_ffff"
//     args: "bits[24]:0x10; bits[39]:0xb18_0221"
//     args: "bits[24]:0x1000; bits[39]:0xc_8804_0042"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x4"
//     args: "bits[24]:0xe7_6c81; bits[39]:0x70_661f_0041"
//     args: "bits[24]:0x20_0000; bits[39]:0x10_8200_955d"
//     args: "bits[24]:0x7f_ffff; bits[39]:0x8000"
//   }
// }
// END_CONFIG
type x3 = u1;
fn main(x0: u24, x1: u39) -> (x3[0x27], u24, u45, u39, u39, u39, x3[0x75], x3[0x4e], u24, x3[0x27], u45, x3[0x75], u39, u39, u39, u39, x3[0x27], u39, x3[0x75]) {
  let x2: x3[0x27] = ((x1) as x3[0x27]);
  let x4: u39 = (x1)[0x0+:u39];
  let x5: u39 = !(x4);
  let x6: u39 = -(x5);
  let x7: u40 = one_hot(x5, u1:0x1);
  let x8: x3[0x4e] = (x2) ++ (x2);
  let x9: u39 = (((x0) as u39)) + (x4);
  let x10: x3[0x75] = (x2) ++ (x8);
  let x11: u39 = (x5)[:];
  let x12: x3[0x75] = (x8) ++ (x2);
  let x13: u39 = ctz(x9);
  let x14: u39 = for (i, x): (u4, u39) in u4:0x0..u4:0x7 {
    x
  }(x5);
  let x15: u39 = for (i, x): (u4, u39) in u4:0x0..u4:0x0 {
    x
  }(x9);
  let x16: (u24, u39) = (x0, x9);
  let x17: u45 = u45:0x20000;
  let x18: u39 = rev(x1);
  let x19: u45 = x17;
  let x20: u39 = for (i, x): (u4, u39) in u4:0x0..u4:0x1 {
    x
  }(x1);
  let x21: u39 = (x13) ^ (x15);
  (x2, x0, x19, x9, x21, x13, x12, x8, x0, x2, x17, x12, x1, x1, x15, x5, x2, x18, x10)
}
