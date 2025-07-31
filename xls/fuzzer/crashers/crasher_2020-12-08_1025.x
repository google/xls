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
//   codegen_args: "--pipeline_stages=10"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[61]:0x4000_0000; bits[21]:0xa_2880; bits[44]:0x725_5440_7887; bits[62]:0x1d85_5d05_b01e_0038"
//     args: "bits[61]:0x100_0000_0000_0000; bits[21]:0x4_1000; bits[44]:0x259_221a_9a30; bits[62]:0x400"
//     args: "bits[61]:0x1978_d1b0_c5be_be3d; bits[21]:0x1e_bf3d; bits[44]:0xaaa_aaaa_aaaa; bits[62]:0x1_0000_0000_0000"
//     args: "bits[61]:0x1_0000_0000_0000; bits[21]:0x1a_4b01; bits[44]:0x40_0140_1008; bits[62]:0x4000"
//     args: "bits[61]:0x1000_0000_0000_0000; bits[21]:0x8_0000; bits[44]:0x1000_0000; bits[62]:0x2482_a403_5668_1042"
//     args: "bits[61]:0x20; bits[21]:0xa_aaaa; bits[44]:0x40_0020_0022; bits[62]:0x1455_7400_0800_2000"
//     args: "bits[61]:0x1000; bits[21]:0x5200; bits[44]:0x515_1b28_308e; bits[62]:0x3a0_00ad_a112_280c"
//     args: "bits[61]:0x80_0000_0000_0000; bits[21]:0x80; bits[44]:0x2d2_05e0_0010; bits[62]:0xb6d_3708_2164_0a48"
//     args: "bits[61]:0x8000_0000; bits[21]:0x40a; bits[44]:0x2000; bits[62]:0x20"
//     args: "bits[61]:0x400_0000_0000; bits[21]:0x440a; bits[44]:0x40_0000_0000; bits[62]:0x40"
//     args: "bits[61]:0x100_0000_0000_0000; bits[21]:0x2_0000; bits[44]:0x301_4515_3e12; bits[62]:0xc55_d0c2_000a_a806"
//     args: "bits[61]:0x40_0000; bits[21]:0x8_0000; bits[44]:0x408_0002_0008; bits[62]:0x161b_0525_4121_91d8"
//     args: "bits[61]:0x80; bits[21]:0x10_0000; bits[44]:0x10_0241_2084; bits[62]:0x863_0014_2140_2a84"
//     args: "bits[61]:0x8_0000; bits[21]:0x8_9500; bits[44]:0x400_0018_0080; bits[62]:0x800_0000"
//     args: "bits[61]:0x4000; bits[21]:0x6919; bits[44]:0x40_0000; bits[62]:0x178d_7283_ec07_5c59"
//     args: "bits[61]:0x2000_0000; bits[21]:0x4000; bits[44]:0x280_72c4_2849; bits[62]:0x50_408c_2101"
//     args: "bits[61]:0x40_0000; bits[21]:0x1_0000; bits[44]:0x80_0001_0290; bits[62]:0x1204_1001_0044_2000"
//     args: "bits[61]:0x4000_0000; bits[21]:0x80; bits[44]:0x200_6100_8018; bits[62]:0x803_1052_11e0_4046"
//     args: "bits[61]:0x8; bits[21]:0x3_0028; bits[44]:0x1_0000_0000; bits[62]:0x1_0000"
//     args: "bits[61]:0x77a_8975_42d4_73e5; bits[21]:0x15_73f5; bits[44]:0x8000; bits[62]:0x8_0000_0000"
//     args: "bits[61]:0x4_0000_0000_0000; bits[21]:0x4_4084; bits[44]:0x400_0000_0000; bits[62]:0xa81_0800_0000_2080"
//     args: "bits[61]:0x80; bits[21]:0x182; bits[44]:0x20_e940_0401; bits[62]:0x800_0000_0000"
//     args: "bits[61]:0x400_0000_0000_0000; bits[21]:0x8; bits[44]:0x2_0000; bits[62]:0x20_0000_0000_0000"
//     args: "bits[61]:0x20_0000_0000_0000; bits[21]:0x10_2042; bits[44]:0x200_0000; bits[62]:0x800_0000_0000"
//     args: "bits[61]:0x80; bits[21]:0x8_c0c0; bits[44]:0xec4_bb74_7856; bits[62]:0x3406_a810_0608_e300"
//     args: "bits[61]:0x1000_0000_0000; bits[21]:0x1c_cc10; bits[44]:0x1000; bits[62]:0x4c5_8519_7452_6af9"
//     args: "bits[61]:0x400; bits[21]:0x400; bits[44]:0x20_0000_0000; bits[62]:0x160a_0020_0040_4094"
//     args: "bits[61]:0x17f6_493a_62af_012e; bits[21]:0x7_003a; bits[44]:0x1; bits[62]:0x2930_8498_983e_c2cc"
//     args: "bits[61]:0x2_0000_0000_0000; bits[21]:0x11_1205; bits[44]:0x555_5555_5555; bits[62]:0x2228_0a19_0102_0109"
//     args: "bits[61]:0x1000_0000_0000; bits[21]:0x1; bits[44]:0x184_08d8_0445; bits[62]:0xa46_2340_d294_402a"
//     args: "bits[61]:0x400_0000; bits[21]:0x1000; bits[44]:0x80_4400_4420; bits[62]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[61]:0x400_0000_0000; bits[21]:0x5_8156; bits[44]:0x200_0000_0000; bits[62]:0x31a2_7b5a_d119_86e5"
//     args: "bits[61]:0x800_0000; bits[21]:0x14_c8a8; bits[44]:0x710_3c29_e511; bits[62]:0x1c40_f0af_9c45_feff"
//     args: "bits[61]:0x8000_0000; bits[21]:0x5_2000; bits[44]:0x398_4ac1_94b1; bits[62]:0x8_0000_0000"
//     args: "bits[61]:0x1_0000; bits[21]:0x4; bits[44]:0x800_030c_8002; bits[62]:0x118_0000_c003_0000"
//     args: "bits[61]:0x10_0000; bits[21]:0xe_0bf1; bits[44]:0xb4_0c71_0040; bits[62]:0x1837_eb10_0410_0012"
//     args: "bits[61]:0x40; bits[21]:0x15_5555; bits[44]:0xcfa_be04_4004; bits[62]:0x80"
//     args: "bits[61]:0x8_0000_0000_0000; bits[21]:0x1; bits[44]:0x82b_5880_344a; bits[62]:0x3218_aa50_0688_0060"
//     args: "bits[61]:0x2000_0000_0000; bits[21]:0x10_0292; bits[44]:0xb01_c901_0c00; bits[62]:0x8a6_f026_fd0c_4ac4"
//     args: "bits[61]:0x800_0000; bits[21]:0x80; bits[44]:0x200_0000_0000; bits[62]:0x1000_0000_0000_0000"
//     args: "bits[61]:0x1000; bits[21]:0x8_b8e1; bits[44]:0xd0b_bb70_5bbb; bits[62]:0x2010_8000_0000_2408"
//     args: "bits[61]:0x2_0000; bits[21]:0x2_0832; bits[44]:0x8000_0000; bits[62]:0x4"
//     args: "bits[61]:0x40_0000_0000; bits[21]:0x1f_ffff; bits[44]:0x248_8007_6621; bits[62]:0x40_0000"
//     args: "bits[61]:0x4_0000_0000; bits[21]:0x0; bits[44]:0x146_8250_0080; bits[62]:0x100_0008_0000_0000"
//     args: "bits[61]:0x400_0000_0000; bits[21]:0xb_3998; bits[44]:0x8_0000; bits[62]:0x210_0030_2004_4b22"
//     args: "bits[61]:0x4; bits[21]:0x8000; bits[44]:0x1_0060_8204; bits[62]:0x2000_0000_0000"
//     args: "bits[61]:0x40_0000_0000; bits[21]:0x0; bits[44]:0x521_6000_8813; bits[62]:0x1_019a_2002_5102"
//     args: "bits[61]:0x40_0000; bits[21]:0x4; bits[44]:0x116_3089_3b41; bits[62]:0x2_0000"
//     args: "bits[61]:0x0; bits[21]:0x4000; bits[44]:0x10_0000_0000; bits[62]:0x3850_1b63_3820_7f4f"
//     args: "bits[61]:0x595_50a1_c61e_80b9; bits[21]:0x800; bits[44]:0x25_c61e_80b9; bits[62]:0x1f82_6461_c505_8800"
//     args: "bits[61]:0x400_0000_0000_0000; bits[21]:0x10; bits[44]:0x2000_0000; bits[62]:0x2_0000"
//     args: "bits[61]:0x1fff_ffff_ffff_ffff; bits[21]:0x19_75fe; bits[44]:0xeff_f7ff_ffff; bits[62]:0x2bfe_ddff_7ffc_0125"
//     args: "bits[61]:0x80_0000_0000_0000; bits[21]:0x11_245f; bits[44]:0x8000_0000; bits[62]:0x1000"
//     args: "bits[61]:0x800; bits[21]:0x8_8800; bits[44]:0x176_2239_2060; bits[62]:0x5d8_88e4_8180_0400"
//     args: "bits[61]:0x73_a4c1_9136_a9f5; bits[21]:0x16_abf3; bits[44]:0x154_35cf_787f; bits[62]:0x4f0_5301_804e_aa80"
//     args: "bits[61]:0x345_63b5_ee87_5989; bits[21]:0x8_0000; bits[44]:0x400; bits[62]:0x8"
//     args: "bits[61]:0x80_0000; bits[21]:0xf_ffff; bits[44]:0x7ff_f7c3_0180; bits[62]:0x1ff3_ca0d_5e00_1311"
//     args: "bits[61]:0x20_0000_0000; bits[21]:0x16_4241; bits[44]:0x330_a0b4_c04c; bits[62]:0x2c04_83d0_0778_444e"
//     args: "bits[61]:0x4000; bits[21]:0x9_4800; bits[44]:0x10; bits[62]:0x200_0000_0000_0000"
//     args: "bits[61]:0x2_0000_0000; bits[21]:0x1_2180; bits[44]:0x800; bits[62]:0x3243_0323_420a_6b2a"
//     args: "bits[61]:0x8000_0000; bits[21]:0x0; bits[44]:0x212_4242_1050; bits[62]:0x3fff_ffff_ffff_ffff"
//     args: "bits[61]:0x2_0000_0000_0000; bits[21]:0x8_6200; bits[44]:0x80_0000; bits[62]:0x100_0000_0000"
//     args: "bits[61]:0x100_0000_0000; bits[21]:0x2000; bits[44]:0x500_3000_0100; bits[62]:0x200_0000_0000"
//     args: "bits[61]:0x1_0000_0000; bits[21]:0x8_0320; bits[44]:0x403_9402_800a; bits[62]:0x22cf_853f_6b0a_b100"
//     args: "bits[61]:0x1000; bits[21]:0x4_0000; bits[44]:0x64_6267_98a0; bits[62]:0x2081"
//     args: "bits[61]:0x4_0000_0000_0000; bits[21]:0x12_c6d4; bits[44]:0x1000_0000; bits[62]:0x80_0000_0000_0000"
//     args: "bits[61]:0x10_0000; bits[21]:0x5_8106; bits[44]:0x409_7a0c_0910; bits[62]:0x2636_4231_6511_b54c"
//     args: "bits[61]:0x100_0000; bits[21]:0x28; bits[44]:0x10_0000; bits[62]:0x1000_0000_0000_0000"
//     args: "bits[61]:0x2000_0000; bits[21]:0x10_2880; bits[44]:0x200; bits[62]:0x2071_0548_1600_4280"
//     args: "bits[61]:0x139e_854c_4655_1823; bits[21]:0x40; bits[44]:0x21e_db45_5863; bits[62]:0x123f_0e80_85ea_6dcb"
//     args: "bits[61]:0x8000_0000; bits[21]:0x0; bits[44]:0x2_0000; bits[62]:0x8000_0000"
//     args: "bits[61]:0x10; bits[21]:0x100; bits[44]:0x8_0000_0000; bits[62]:0x100_0000"
//     args: "bits[61]:0x1; bits[21]:0x1; bits[44]:0x311; bits[62]:0x20"
//     args: "bits[61]:0x4000_0000_0000; bits[21]:0x0; bits[44]:0xa0_4830_a40a; bits[62]:0x283_20c2_9038_0410"
//     args: "bits[61]:0x400_0000_0000_0000; bits[21]:0x18_3024; bits[44]:0x80; bits[62]:0x2000_0000"
//     args: "bits[61]:0x10_0000; bits[21]:0x1f_ffff; bits[44]:0x8_0000_0000; bits[62]:0x4"
//     args: "bits[61]:0x20_0000; bits[21]:0x5400; bits[44]:0x100; bits[62]:0x80_0000"
//     args: "bits[61]:0x200_0000_0000_0000; bits[21]:0x0; bits[44]:0x2; bits[62]:0x1301_8be9_1d16_0548"
//     args: "bits[61]:0x80_0000; bits[21]:0x5_a81d; bits[44]:0x200_0000_0000; bits[62]:0x40_0000_0000_0000"
//     args: "bits[61]:0x2000; bits[21]:0x2800; bits[44]:0x100_0000; bits[62]:0x2000_0000"
//     args: "bits[61]:0x191f_a6ae_8d8b_2ff4; bits[21]:0xb_2ff4; bits[44]:0x8000; bits[62]:0x1000_0000_0000"
//     args: "bits[61]:0x1; bits[21]:0x4001; bits[44]:0x694_cb86_6046; bits[62]:0x1282_f361_a195_c23a"
//     args: "bits[61]:0x1555_5555_5555_5555; bits[21]:0x15_5455; bits[44]:0x471_55d5_5555; bits[62]:0x2aa1_8a00_0611_0440"
//     args: "bits[61]:0x1_0000_0000; bits[21]:0xa_4a40; bits[44]:0x1000; bits[62]:0x4000_0000_0000"
//     args: "bits[61]:0x1000_0000; bits[21]:0x200; bits[44]:0x400_0000_0000; bits[62]:0x20_0000_0000_0000"
//     args: "bits[61]:0x100_0000; bits[21]:0x5_0620; bits[44]:0x2c3_1008_8000; bits[62]:0x8000_0000"
//     args: "bits[61]:0x2_0000; bits[21]:0x12_8700; bits[44]:0xd8b_d64c_0186; bits[62]:0x3ee3_dd68_2889_1400"
//     args: "bits[61]:0x800_0000_0000; bits[21]:0x4000; bits[44]:0x4_0000_0000; bits[62]:0x100_0000"
//     args: "bits[61]:0x800_0000; bits[21]:0x4_0200; bits[44]:0x1000_0000; bits[62]:0x12f4_86bc_0834_4201"
//     args: "bits[61]:0x20_0000_0000_0000; bits[21]:0x100; bits[44]:0xa0_a820_2534; bits[62]:0x1000_0000_0000"
//     args: "bits[61]:0x400_0000; bits[21]:0x8_0000; bits[44]:0x5a2_0c00_9416; bits[62]:0x800_2c04_1280_0292"
//     args: "bits[61]:0x20; bits[21]:0x2_0000; bits[44]:0x800_0000; bits[62]:0x800_0000"
//     args: "bits[61]:0x800_0000; bits[21]:0x2064; bits[44]:0x800_0000; bits[62]:0x209b_9dc0_2108_3216"
//     args: "bits[61]:0x1fff_ffff_ffff_ffff; bits[21]:0x2000; bits[44]:0x8_0000; bits[62]:0x0"
//     args: "bits[61]:0x20_0000; bits[21]:0x0; bits[44]:0xa40_3020_2acc; bits[62]:0x20_0000"
//     args: "bits[61]:0x800_0000; bits[21]:0x0; bits[44]:0x8a4_0ef7_0bbc; bits[62]:0x200_0000"
//     args: "bits[61]:0x80_0000_0000_0000; bits[21]:0x0; bits[44]:0x809_0022_de0c; bits[62]:0x10"
//     args: "bits[61]:0x1555_5555_5555_5555; bits[21]:0x15_55b5; bits[44]:0xd5d_7565_dfbe; bits[62]:0x2888_aaab_fa9a_6aaa"
//     args: "bits[61]:0x100; bits[21]:0x1_0104; bits[44]:0x72b_342f_b5af; bits[62]:0x11a_a7f7_a2e1_d348"
//     args: "bits[61]:0x200_0000_0000; bits[21]:0x5_084c; bits[44]:0x648_848e_6c61; bits[62]:0x80"
//     args: "bits[61]:0x20_0000_0000; bits[21]:0x4; bits[44]:0x45e_34af_215f; bits[62]:0x1000_0840_8080_2100"
//     args: "bits[61]:0x10; bits[21]:0x8_0020; bits[44]:0x8_0000_0000; bits[62]:0x2000"
//     args: "bits[61]:0x80; bits[21]:0x9_0495; bits[44]:0x40_0000; bits[62]:0x10b9_2a02_0100_1802"
//     args: "bits[61]:0x2_0000; bits[21]:0x2_0406; bits[44]:0x20_0000; bits[62]:0x520_0412_0262_2900"
//     args: "bits[61]:0x1e7b_2628_a388_b513; bits[21]:0x8_b507; bits[44]:0x8_0000_0000; bits[62]:0x812_1782_4fcb_fddd"
//     args: "bits[61]:0x200; bits[21]:0x1000; bits[44]:0x400; bits[62]:0x100_0000_0000_0000"
//     args: "bits[61]:0x200_0000_0000_0000; bits[21]:0x0; bits[44]:0xc81_22b5_fac3; bits[62]:0x3100_8d75_ab84_a8cb"
//     args: "bits[61]:0x2000; bits[21]:0x4; bits[44]:0x800_0000_0000; bits[62]:0x1_0000"
//     args: "bits[61]:0x40_0000; bits[21]:0x1000; bits[44]:0x402_6245_0010; bits[62]:0x1039_8d18_8093_f57b"
//     args: "bits[61]:0x80_0000_0000; bits[21]:0x8000; bits[44]:0x1000_0000; bits[62]:0x4000"
//     args: "bits[61]:0x10; bits[21]:0x8_0000; bits[44]:0x802_0000_0020; bits[62]:0x400_0000_0000_0000"
//     args: "bits[61]:0x4; bits[21]:0x3_9177; bits[44]:0x10; bits[62]:0x20_0000"
//     args: "bits[61]:0x1; bits[21]:0x12_0041; bits[44]:0x8000_0000; bits[62]:0x10_0000_0000_0000"
//     args: "bits[61]:0x2000_0000_0000; bits[21]:0x10_0014; bits[44]:0x10_0000_0000; bits[62]:0xd0_2724_4207_2185"
//     args: "bits[61]:0x8000; bits[21]:0x8; bits[44]:0x280_0152_e389; bits[62]:0x1_0000"
//     args: "bits[61]:0x4; bits[21]:0x9aa4; bits[44]:0x20_0000_0000; bits[62]:0x28_4c09_200a_0401"
//     args: "bits[61]:0x100_0000; bits[21]:0x2_0000; bits[44]:0x8000; bits[62]:0x20_0000_0000"
//     args: "bits[61]:0x4_0000_0000; bits[21]:0x8; bits[44]:0x400_0010; bits[62]:0x18_9000_8240_0028"
//     args: "bits[61]:0x1000_0000_0000; bits[21]:0x2_e523; bits[44]:0x800_0000; bits[62]:0x1_0000_0000_0000"
//     args: "bits[61]:0x2_0000_0000_0000; bits[21]:0x10; bits[44]:0x50_c818_02a0; bits[62]:0x8000_0000_0000"
//     args: "bits[61]:0x1000_0000; bits[21]:0x2_4880; bits[44]:0x400_0000; bits[62]:0x100_0000_0000_0000"
//     args: "bits[61]:0x2000_0000; bits[21]:0x4; bits[44]:0x2_2000_0000; bits[62]:0x9f1_0365_4543_2014"
//     args: "bits[61]:0x40_0000_0000_0000; bits[21]:0x8_0000; bits[44]:0x16c_2088_b214; bits[62]:0x1220_5919_de59_5067"
//     args: "bits[61]:0x100_0000_0000; bits[21]:0x1000; bits[44]:0x460_a011_0931; bits[62]:0x3820_0028_8207_a043"
//     args: "bits[61]:0x2_0000_0000_0000; bits[21]:0x10_0e0d; bits[44]:0x815_06c0_0101; bits[62]:0x2591_0163_4480_5af1"
//     args: "bits[61]:0x200; bits[21]:0xe_4068; bits[44]:0x110_1342_c108; bits[62]:0x4_0000"
//     args: "bits[61]:0x20_0000; bits[21]:0x9_2069; bits[44]:0x108_83e0_0013; bits[62]:0x400_0000"
//     args: "bits[61]:0x200_0000_0000; bits[21]:0x20; bits[44]:0x2; bits[62]:0x4"
//   }
// }
// END_CONFIG
type x27 = (u7, (u7,), u24, s4, u35, s9, u35);
fn x5(x6: s9) -> (u7, (u7,), u24, s4, u35, s9, u35) {
  let x7: s9 = (x6) + (x6);
  let x8: s9 = one_hot_sel(u3:0x7, [x6, x6, x7]);
  let x9: u7 = u7:0x1;
  let x10: u7 = for (i, x): (u4, u7) in u4:0x0..u4:0x6 {
    x
  }(x9);
  let x11: u7 = !(x10);
  let x12: u7 = (x9) - (x10);
  let x13: u7 = (x10) - (((x7) as u7));
  let x14: u35 = ((((x11) ++ (x12)) ++ (x13)) ++ (x10)) ++ (x12);
  let x15: u7 = -(x13);
  let x16: u35 = one_hot_sel(x10, [x14, x14, x14, x14, x14, x14, x14]);
  let x17: u2 = (x10)[-0x2:];
  let x18: s4 = s4:0x8;
  let x19: (s4,) = (x18,);
  let x20: u24 = u24:0x800000;
  let x21: u7 = clz(x15);
  let x22: u1 = (x17)[0x1:];
  let x23: u2 = (x18 as u4)[0x2+:u2];
  let x24: s4 = (((x16) as s4)) * (x18);
  let x25: u2 = (x7 as u9)[0x0+:u2];
  let x26: (u7,) = (x15,);
  (x10, x26, x20, x24, x16, x6, x16)
}
fn main(x0: s61, x1: u21, x2: s44, x3: s62) -> s44 {
  let x4: x27[0x1] = map([s9:0x10], x5);
  let x28: s62 = for (i, x): (u4, s62) in u4:0x0..u4:0x3 {
    x
  }(x3);
  let x29: s62 = one_hot_sel(u3:0x1, [x3, x3, x28]);
  let x30: (s62,) = (x3,);
  let x31: u21 = one_hot_sel(u5:0x15, [x1, x1, x1, x1, x1]);
  let x32: (s62,) = for (i, x): (u4, (s62,)) in u4:0x0..u4:0x2 {
    x
  }(x30);
  let x33: u21 = one_hot_sel(u6:0x0, [x1, x31, x1, x31, x31, x1]);
  let x34: s44 = !(x2);
  let x35: (s62,) = for (i, x): (u4, (s62,)) in u4:0x0..u4:0x2 {
    x
  }(x32);
  let x36: s61 = (((x33) as s61)) ^ (x0);
  let x37: (u21,) = (x1,);
  let x38: (s62,) = for (i, x): (u4, (s62,)) in u4:0x0..u4:0x5 {
    x
  }(x35);
  let x39: s61 = (x36) + (((x2) as s61));
  let x40: s62 = (x38).0x0;
  x34
}
