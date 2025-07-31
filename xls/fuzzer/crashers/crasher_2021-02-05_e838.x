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
//   codegen_args: "--generator=combinational"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[52]:0x8000_0000; bits[16]:0xaaaa; bits[62]:0x2000_0000; bits[6]:0x2b"
//     args: "bits[52]:0x4_0000; bits[16]:0x100; bits[62]:0xcec_b630_6ae4_0cc1; bits[6]:0x0"
//     args: "bits[52]:0x40_0000; bits[16]:0x80; bits[62]:0x200; bits[6]:0x0"
//     args: "bits[52]:0x20; bits[16]:0x1000; bits[62]:0x1555_5555_5555_5555; bits[6]:0x24"
//     args: "bits[52]:0x800_0000; bits[16]:0x200; bits[62]:0x30_0010_1100; bits[6]:0x2"
//     args: "bits[52]:0x800_0000; bits[16]:0xc28; bits[62]:0x80; bits[6]:0x0"
//     args: "bits[52]:0x80_0000; bits[16]:0x8209; bits[62]:0x2002_c812_0b84_0100; bits[6]:0x15"
//     args: "bits[52]:0x2000_0000; bits[16]:0x2; bits[62]:0x2064_2c91_2845_d2be; bits[6]:0x8"
//     args: "bits[52]:0x8_0000; bits[16]:0x0; bits[62]:0x20; bits[6]:0x2a"
//     args: "bits[52]:0x80; bits[16]:0x2af; bits[62]:0x1555_5555_5555_5555; bits[6]:0x6"
//     args: "bits[52]:0xf_ffff_ffff_ffff; bits[16]:0x40; bits[62]:0x605_c2bf_51ed_99fd; bits[6]:0x3f"
//     args: "bits[52]:0x800_0000_0000; bits[16]:0x28; bits[62]:0x800_0000_0000_0000; bits[6]:0x0"
//     args: "bits[52]:0x80; bits[16]:0x200; bits[62]:0x40_0000_0000_0000; bits[6]:0x30"
//     args: "bits[52]:0x4000_0000_0000; bits[16]:0x2135; bits[62]:0x1100_4909_0000_03d7; bits[6]:0x1f"
//     args: "bits[52]:0x400; bits[16]:0x1082; bits[62]:0x4b9_c9ce_4100_5a24; bits[6]:0x20"
//     args: "bits[52]:0xa_96b5_af70_2fe4; bits[16]:0xef4; bits[62]:0x1000_0000_0000; bits[6]:0x34"
//     args: "bits[52]:0x20_0000; bits[16]:0x7fff; bits[62]:0x1bff_dbdf_ff6b_fdf0; bits[6]:0x34"
//     args: "bits[52]:0x8_0000; bits[16]:0x80; bits[62]:0x90_0000_2000_0000; bits[6]:0x18"
//     args: "bits[52]:0xc_b1f8_b919_96c7; bits[16]:0x40; bits[62]:0x4_0000_0000_0000; bits[6]:0x7"
//     args: "bits[52]:0x80_0000_0000; bits[16]:0x180; bits[62]:0x2729_7a07_8089_a8c6; bits[6]:0x15"
//     args: "bits[52]:0x1000_0000; bits[16]:0xffff; bits[62]:0x400_b080_a688_9735; bits[6]:0x2"
//     args: "bits[52]:0x0; bits[16]:0x400; bits[62]:0x180_6001_a440_6108; bits[6]:0x20"
//     args: "bits[52]:0x1_0000_0000_0000; bits[16]:0x9781; bits[62]:0x40_0000_0000; bits[6]:0x0"
//     args: "bits[52]:0x1; bits[16]:0x4; bits[62]:0x1_0000_0000_0000; bits[6]:0x8"
//     args: "bits[52]:0x800; bits[16]:0x200; bits[62]:0xd0_0000_0200_0400; bits[6]:0x6"
//     args: "bits[52]:0x400_0000; bits[16]:0x4000; bits[62]:0x4_0010_0100_0010; bits[6]:0x0"
//     args: "bits[52]:0x10; bits[16]:0xffff; bits[62]:0x400_0000; bits[6]:0x1f"
//     args: "bits[52]:0x40_0000; bits[16]:0x180; bits[62]:0x800_0000_0000_0000; bits[6]:0x1"
//     args: "bits[52]:0x200; bits[16]:0x8000; bits[62]:0x48_0228_6b00_2244; bits[6]:0x2a"
//     args: "bits[52]:0x100_0000_0000; bits[16]:0x2; bits[62]:0x8_0000_0000_0000; bits[6]:0x6"
//     args: "bits[52]:0x2_0000; bits[16]:0x4490; bits[62]:0x1802_0208_188a_2010; bits[6]:0x2b"
//     args: "bits[52]:0x80_0000; bits[16]:0x5555; bits[62]:0x1020_5316_14cc_1695; bits[6]:0x15"
//     args: "bits[52]:0x20_0000_0000; bits[16]:0x1000; bits[62]:0x200_0000_0000; bits[6]:0x1f"
//     args: "bits[52]:0x400_0000_0000; bits[16]:0x1; bits[62]:0x1555_5555_5555_5555; bits[6]:0x0"
//     args: "bits[52]:0x2000_0000_0000; bits[16]:0x0; bits[62]:0x91_0400_0400_41df; bits[6]:0x5"
//     args: "bits[52]:0x8000_0000; bits[16]:0x630; bits[62]:0x5f7_4a02_858d_4906; bits[6]:0x0"
//     args: "bits[52]:0x1_0000_0000_0000; bits[16]:0x4840; bits[62]:0x8_0000; bits[6]:0x1f"
//     args: "bits[52]:0x4_0000_0000; bits[16]:0x800; bits[62]:0x40_0000; bits[6]:0x20"
//     args: "bits[52]:0x100_0000_0000; bits[16]:0x0; bits[62]:0x10_0000_0000_0000; bits[6]:0x2"
//     args: "bits[52]:0x2_0000; bits[16]:0x2608; bits[62]:0x983_0002_1010_00d4; bits[6]:0x10"
//     args: "bits[52]:0x1000; bits[16]:0x7fff; bits[62]:0x17f7_c200_2200_0180; bits[6]:0x2a"
//     args: "bits[52]:0x1; bits[16]:0x100; bits[62]:0x4000_0000_0000; bits[6]:0x9"
//     args: "bits[52]:0x2000_0000; bits[16]:0x5555; bits[62]:0x100_0000_0000_0000; bits[6]:0x2c"
//     args: "bits[52]:0x5_5555_5555_5555; bits[16]:0xffff; bits[62]:0x3124_cd55_d792_cd37; bits[6]:0x20"
//     args: "bits[52]:0x800; bits[16]:0x2; bits[62]:0xc_0022_8a90; bits[6]:0x6"
//     args: "bits[52]:0x8000; bits[16]:0x4000; bits[62]:0x3119_c508_2933_0048; bits[6]:0x20"
//     args: "bits[52]:0x2_4260_d6b4_524f; bits[16]:0x524f; bits[62]:0x4_0000_0000_0000; bits[6]:0x8"
//     args: "bits[52]:0x800; bits[16]:0x0; bits[62]:0x20; bits[6]:0x2d"
//     args: "bits[52]:0x100_0000_0000; bits[16]:0x800; bits[62]:0x4_0400_0018_0000; bits[6]:0x3"
//     args: "bits[52]:0x4_7c2f_aae4_3f8d; bits[16]:0x7f8f; bits[62]:0x800_0000_0000_0000; bits[6]:0x9"
//     args: "bits[52]:0xa_aaaa_aaaa_aaaa; bits[16]:0x80; bits[62]:0x211d_5bb7_38dc_2968; bits[6]:0x21"
//     args: "bits[52]:0x4_0000_0000; bits[16]:0x3810; bits[62]:0x83d_80e9_ce28_1890; bits[6]:0x0"
//     args: "bits[52]:0x20_0000_0000; bits[16]:0x8; bits[62]:0x28_8024_0038_00a0; bits[6]:0x1"
//     args: "bits[52]:0x800_0000_0000; bits[16]:0x2; bits[62]:0x8_0000; bits[6]:0x2"
//     args: "bits[52]:0x10_0000_0000; bits[16]:0x750; bits[62]:0x800_0004_1000_03f7; bits[6]:0x4"
//     args: "bits[52]:0x2_0000; bits[16]:0x100; bits[62]:0x840_880c_94a8_0100; bits[6]:0x30"
//     args: "bits[52]:0x4000; bits[16]:0xc4c9; bits[62]:0x32f3_6431_0262_4070; bits[6]:0x1"
//     args: "bits[52]:0x2000_0000_0000; bits[16]:0x3090; bits[62]:0x200_0000_0000; bits[6]:0x1f"
//     args: "bits[52]:0x8000_0000; bits[16]:0x1; bits[62]:0x40_0000_0000_0000; bits[6]:0x4"
//     args: "bits[52]:0x200_0000; bits[16]:0x920; bits[62]:0x4; bits[6]:0x0"
//     args: "bits[52]:0x8000_0000; bits[16]:0x2000; bits[62]:0x800_0000_0000; bits[6]:0x20"
//     args: "bits[52]:0x4; bits[16]:0x206; bits[62]:0x1081_8810_0000_0000; bits[6]:0x16"
//     args: "bits[52]:0x2_0000_0000_0000; bits[16]:0x5555; bits[62]:0x1e68_ee9e_193d_08c8; bits[6]:0xc"
//     args: "bits[52]:0x4000; bits[16]:0x100; bits[62]:0x2155_6001_0001_5000; bits[6]:0x1"
//     args: "bits[52]:0x4_0000_0000; bits[16]:0x4010; bits[62]:0x1c1d_52cc_321d_3d33; bits[6]:0x23"
//     args: "bits[52]:0x100_0000_0000; bits[16]:0x80; bits[62]:0x200_0000_0000; bits[6]:0x15"
//     args: "bits[52]:0x100_0000_0000; bits[16]:0x400; bits[62]:0x300_0002_9000_0004; bits[6]:0x15"
//     args: "bits[52]:0x80; bits[16]:0x80; bits[62]:0x1_0000_0000; bits[6]:0x0"
//     args: "bits[52]:0x100_0000_0000; bits[16]:0x4; bits[62]:0x11ae_4034_1909_1da7; bits[6]:0x4"
//     args: "bits[52]:0x800; bits[16]:0x8; bits[62]:0x501a_1caa_a016; bits[6]:0x20"
//     args: "bits[52]:0x800_0000_0000; bits[16]:0x5555; bits[62]:0x40; bits[6]:0xd"
//     args: "bits[52]:0x2_9601_cee4_ac1e; bits[16]:0x800; bits[62]:0x108_8404_0420_0022; bits[6]:0x8"
//     args: "bits[52]:0x5_5555_5555_5555; bits[16]:0x515d; bits[62]:0x1000_0000; bits[6]:0x15"
//     args: "bits[52]:0xd_edab_b171_8ae5; bits[16]:0x8ae5; bits[62]:0x456_74d5_c2ab_dcf0; bits[6]:0x3c"
//     args: "bits[52]:0x200_0000_0000; bits[16]:0x7fff; bits[62]:0x193c_0829_8bc4_08d7; bits[6]:0x3f"
//     args: "bits[52]:0x4; bits[16]:0x2000; bits[62]:0x2000_0000_0000; bits[6]:0x15"
//     args: "bits[52]:0x80_0000_0000; bits[16]:0x7fff; bits[62]:0x2a0c_27b5_5151_fe36; bits[6]:0x20"
//     args: "bits[52]:0x2; bits[16]:0x2; bits[62]:0x18c_8891_9830_0a01; bits[6]:0x4"
//     args: "bits[52]:0x4000_0000; bits[16]:0x10a6; bits[62]:0x4d5_7ffa_ee42_322a; bits[6]:0x22"
//     args: "bits[52]:0x2; bits[16]:0x98a; bits[62]:0x100_0000; bits[6]:0x24"
//     args: "bits[52]:0x1; bits[16]:0x5d82; bits[62]:0x40_0000_0000_0000; bits[6]:0x4"
//     args: "bits[52]:0x800; bits[16]:0x80; bits[62]:0x800_0000; bits[6]:0x2a"
//     args: "bits[52]:0x2; bits[16]:0x800; bits[62]:0x40_0004_2000_0810; bits[6]:0x1f"
//     args: "bits[52]:0x2_db8d_bd4d_5c80; bits[16]:0x80; bits[62]:0x400_0000; bits[6]:0x0"
//     args: "bits[52]:0x400_0000_0000; bits[16]:0xcc9e; bits[62]:0x800_0000_0000; bits[6]:0x1"
//     args: "bits[52]:0x200_0000; bits[16]:0x1000; bits[62]:0x78c_016c_00c2_884c; bits[6]:0x20"
//     args: "bits[52]:0x800; bits[16]:0x40; bits[62]:0x4dc_8014_0812_10a5; bits[6]:0x8"
//     args: "bits[52]:0x1_f6eb_3909_16eb; bits[16]:0x8000; bits[62]:0x40_0000_0000_0000; bits[6]:0x10"
//     args: "bits[52]:0x1_0000; bits[16]:0x2; bits[62]:0x3fff_ffff_ffff_ffff; bits[6]:0x15"
//     args: "bits[52]:0x8; bits[16]:0x8; bits[62]:0x16a0_4952_3280_0726; bits[6]:0x8"
//     args: "bits[52]:0x200_0000_0000; bits[16]:0x800; bits[62]:0x4000_0000; bits[6]:0x1"
//     args: "bits[52]:0x2_0000_0000_0000; bits[16]:0x1; bits[62]:0x2_0000; bits[6]:0x2c"
//     args: "bits[52]:0x3_d23f_ddfe_f952; bits[16]:0xb5d2; bits[62]:0x2d54_8080_0008_0040; bits[6]:0x32"
//     args: "bits[52]:0xe_4ab0_290e_e123; bits[16]:0x8000; bits[62]:0x3000_4200_0842_4104; bits[6]:0x35"
//     args: "bits[52]:0x4000_0000; bits[16]:0x7fff; bits[62]:0x4; bits[6]:0x2"
//     args: "bits[52]:0x4_0000_0000; bits[16]:0xca9e; bits[62]:0x1885_1680_34a4_053a; bits[6]:0x10"
//     args: "bits[52]:0x20; bits[16]:0x20; bits[62]:0x2_0000_4300_81d5; bits[6]:0x25"
//     args: "bits[52]:0x8000_0000_0000; bits[16]:0x800; bits[62]:0xeac_0e11_e560_96e0; bits[6]:0x10"
//     args: "bits[52]:0x20; bits[16]:0x4; bits[62]:0x80_0000_0000; bits[6]:0x10"
//     args: "bits[52]:0x1_0000_0000_0000; bits[16]:0x1000; bits[62]:0x2681_00ca_0100_0083; bits[6]:0x4"
//     args: "bits[52]:0x2000; bits[16]:0x49; bits[62]:0x80_0090; bits[6]:0xc"
//     args: "bits[52]:0x4_0000_0000_0000; bits[16]:0x1; bits[62]:0x170d_e960_11cc_918a; bits[6]:0x2a"
//     args: "bits[52]:0x800_0000; bits[16]:0x0; bits[62]:0x401_0020_0881_81e8; bits[6]:0x10"
//     args: "bits[52]:0x800_0000_0000; bits[16]:0xe90e; bits[62]:0x224_0400_0400_20a8; bits[6]:0x4"
//     args: "bits[52]:0x3_f412_6101_15c5; bits[16]:0x10; bits[62]:0x400_0000_0000_0000; bits[6]:0x1"
//     args: "bits[52]:0x40_0000; bits[16]:0x0; bits[62]:0x8_0000; bits[6]:0x0"
//     args: "bits[52]:0x20; bits[16]:0x1; bits[62]:0x10_0340_8058_8df7; bits[6]:0x21"
//     args: "bits[52]:0x400; bits[16]:0x1; bits[62]:0x11_1407_b070_1092; bits[6]:0x2c"
//     args: "bits[52]:0x2000_0000; bits[16]:0x2000; bits[62]:0x2214_0220_4830_0008; bits[6]:0x8"
//     args: "bits[52]:0x8000_0000; bits[16]:0x0; bits[62]:0x1000_0000_0000; bits[6]:0x20"
//     args: "bits[52]:0x200_0000_0000; bits[16]:0x303; bits[62]:0x700_8082_0938_d8b9; bits[6]:0x3"
//     args: "bits[52]:0x4000_0000_0000; bits[16]:0x4711; bits[62]:0x11c4_4000_0090_0000; bits[6]:0x15"
//     args: "bits[52]:0x7_ffff_ffff_ffff; bits[16]:0xffff; bits[62]:0x3f7f_c000_0800_4020; bits[6]:0x3f"
//     args: "bits[52]:0x8_0000; bits[16]:0x8020; bits[62]:0x2808_0068_8518_4804; bits[6]:0x4"
//     args: "bits[52]:0x5_5555_5555_5555; bits[16]:0x5555; bits[62]:0x2000_0000; bits[6]:0x2a"
//     args: "bits[52]:0x8; bits[16]:0x10a; bits[62]:0x42_9fff_ffff_ffff; bits[6]:0x8"
//     args: "bits[52]:0x800_0000; bits[16]:0x0; bits[62]:0x4a0_0020_4020_0028; bits[6]:0x0"
//     args: "bits[52]:0x2000_0000_0000; bits[16]:0x400; bits[62]:0x100; bits[6]:0x3f"
//     args: "bits[52]:0xf_ffff_ffff_ffff; bits[16]:0x100; bits[62]:0x8000; bits[6]:0x27"
//     args: "bits[52]:0x2000_0000; bits[16]:0x1; bits[62]:0x4080_0880_0002; bits[6]:0x22"
//     args: "bits[52]:0x1; bits[16]:0x1542; bits[62]:0x2f50_8a68_b180_2000; bits[6]:0x7"
//     args: "bits[52]:0x8_0000_0000; bits[16]:0x23ea; bits[62]:0x20_0000_0000; bits[6]:0x0"
//     args: "bits[52]:0x40_0000_0000; bits[16]:0xffff; bits[62]:0xc0f_41a0_04c1_022c; bits[6]:0x1"
//     args: "bits[52]:0x8_7e01_0807_f0e3; bits[16]:0x61ad; bits[62]:0x1_0000_0000_0000; bits[6]:0x0"
//     args: "bits[52]:0x4000_0000; bits[16]:0x800; bits[62]:0x205_33ad_fb30_c3bd; bits[6]:0x0"
//     args: "bits[52]:0x80; bits[16]:0x1aa0; bits[62]:0x8_0000_0000; bits[6]:0x0"
//     args: "bits[52]:0x400; bits[16]:0x8; bits[62]:0x2000_0000_0000_0000; bits[6]:0x4"
//     args: "bits[52]:0x8000_0000; bits[16]:0x2c; bits[62]:0x100_0000_0000_0000; bits[6]:0x2"
//   }
// }
// END_CONFIG
const W32_V1 = u32:0x1;
type x19 = (s40, (s40,), s40, s40, s40, s40, (s40, s40), s40, s40, s40, (s40, s40), s40);
type x29 = u8;
type x32 = u31;
type x46 = u64;
type x48 = u16;
fn x6(x7: s40) -> (s40, (s40,), s40, s40, s40, s40, (s40, s40), s40, s40, s40, (s40, s40), s40) {
  let x8: s40 = for (i, x): (u4, s40) in u4:0x0..u4:0x0 {
    x
  }(x7);
  let x9: s40 = -(x8);
  let x10: s40 = for (i, x): (u4, s40) in u4:0x0..u4:0x1 {
    x
  }(x8);
  let x11: (s40, s40) = (x9, x8);
  let x12: (s40,) = (x8,);
  let x13: u28 = (x10 as u40)[0x0+:u28];
  let x14: s40 = (x9) * (x8);
  let x15: s40 = one_hot_sel(u5:0x15, [x7, x8, x7, x7, x7]);
  let x16: u1 = u1:0x1;
  let x17: u1 = for (i, x): (u4, u1) in u4:0x0..u4:0x7 {
    x
  }(x16);
  let x18: s40 = (x8) << (if ((x8) >= (s40:0xb)) { (u40:0xb) } else { (x8 as u40) });
  (x8, x12, x8, x14, x14, x15, x11, x18, x10, x10, x11, x9)
}
fn main(x0: s52, x1: s16, x2: u62, x3: u6) -> (u14, x48[0x4]) {
  let x4: u62 = for (i, x): (u4, u62) in u4:0x0..u4:0x1 {
    x
  }(x2);
  let x5: x19[0x1] = map(s40[0x1]:[s40:0x7fffffffff], x6);
  let x20: u6 = ctz(x3);
  let x21: u6 = for (i, x): (u4, u6) in u4:0x0..u4:0x1 {
    x
  }(x20);
  let x22: s64 = s64:0x1000000000000;
  let x23: u64 = u64:0x40000000;
  let x24: s64 = (((x3) as s64)) << (if ((x22) >= (s64:0x3d)) { (u64:0x3d) } else { (x22 as u64) });
  let x25: (u62, u6, u62, s64, u62, u6, s52, u62, u6, s64, u6) = (x2, x21, x2, x22, x2, x3, x0, x4, x20, x22, x20);
  let x26: u64 = (x23)[x21+:u64];
  let x27: u6 = x21;
  let x28: x29[0x8] = ((x26) as x29[0x8]);
  let x30: u64 = ctz(x26);
  let x31: x32[0x2] = ((x4) as x32[0x2]);
  let x33: u6 = ((x27 as s64) >> (x27)) as u6;
  let x34: s64 = one_hot_sel(x21, [x24, x22, x22, x22, x24, x24]);
  let x35: u64 = rev(x23);
  let x36: u64 = for (i, x): (u4, u64) in u4:0x0..u4:0x3 {
    x
  }(x30);
  let x37: u64 = ctz(x23);
  let x38: u6 = !(x20);
  let x39: u40 = (x23)[0xd:-0xb];
  let x40: u6 = ((((x24) as u6) as s6) >> (if ((x27) >= (u6:0x0)) { (u6:0x0) } else { (x27) })) as u6;
  let x41: u6 = one_hot_sel(x40, [x40, x40, x27, x40, x40, x27]);
  let x42: u14 = u14:0x2aaa;
  let x43: s19 = s19:0x80;
  let x44: s64 = (((x26) as s64)) | (x24);
  let x45: x46[W32_V1] = ((x30) as x46[W32_V1]);
  let x47: x48[0x4] = ((x36) as x48[0x4]);
  let x49: u6 = ctz(x40);
  (x42, x47)
}
