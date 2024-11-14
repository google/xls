// Copyright 2023 The XLS Authors
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
// exception: 	 "SampleError: Result miscompare for sample 22:\nargs: bits[31]:0x155b_63d8; bits[8]:0x80; bits[58]:0x280_1001_0001_0081; bits[36]:0xa_3724_c16b\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[36]:0xa_3724_c16b, (bits[58]:0x280_1001_0001_0081, bits[8]:0x80, bits[24]:0x1b_67d2, bits[2]:0x3, bits[36]:0xa_3724_c16b), bits[2]:0x1, bits[31]:0x7fd5_4939, bits[36]:0xf_ffff_ff80, bits[13]:0x1c93)\nevaluated unopt IR (JIT) =\n   (bits[36]:0xa_3724_c16b, (bits[58]:0x280_1001_0001_0081, bits[8]:0x80, bits[24]:0x1b_67d2, bits[2]:0x3, bits[36]:0xa_3724_c16b), bits[2]:0x1, bits[31]:0x7fd5_4939, bits[36]:0x80, bits[13]:0x1c93)"
// issue: "https://github.com/google/xls/issues/1241"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full throughput.*"
//   }
// }
// inputs {
//   function_args {
//     args: "bits[31]:0x80; bits[8]:0xc9; bits[58]:0x32f_9aab_aea8_9aa8; bits[36]:0xe_acf8_30ac"
//     args: "bits[31]:0x5555_5555; bits[8]:0x7f; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0x800_0000"
//     args: "bits[31]:0x0; bits[8]:0xff; bits[58]:0x1ff_ffff_ffff_ffff; bits[36]:0xe_ffff_fff7"
//     args: "bits[31]:0x2ccb_25dc; bits[8]:0xd5; bits[58]:0x22d_f9a1_4505_ab8b; bits[36]:0xd_6b67_c184"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xcc; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xc_c098_0000"
//     args: "bits[31]:0x0; bits[8]:0x20; bits[58]:0x1ff_ffff_ffff_ffff; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x0; bits[8]:0x12; bits[58]:0x48_5453_1577_d555; bits[36]:0x2_0302_3057"
//     args: "bits[31]:0x6edd_e574; bits[8]:0xff; bits[58]:0x2000_0000_0000; bits[36]:0xf_f12f_8596"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xaa; bits[58]:0x189_0194_3558_7dd6; bits[36]:0x4_955b_efdd"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xff; bits[58]:0x3fd_0595_5551_5714; bits[36]:0xb_df5b_ffe0"
//     args: "bits[31]:0x3d92_fa09; bits[8]:0x7f; bits[58]:0x1ae_a8ba_2aba_aaab; bits[36]:0x4_5270_70cd"
//     args: "bits[31]:0x0; bits[8]:0x38; bits[58]:0x4000_0000_0000; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x5555_5555; bits[8]:0xff; bits[58]:0x155_5555_5555_5555; bits[36]:0x7_ffff_ffff"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xaa; bits[58]:0x1c_70ca_e623_a784; bits[36]:0xf_a080_4440"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xaf; bits[58]:0x1dd_f7ff_fdb7_f3be; bits[36]:0xf_fbf6_77be"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xff; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[36]:0x2_ea9a_683b"
//     args: "bits[31]:0x5555_5555; bits[8]:0x55; bits[58]:0x28a_aaea_afa3_fc63; bits[36]:0x6_7ae8_6baf"
//     args: "bits[31]:0x40_0000; bits[8]:0x7f; bits[58]:0x1fc_0000_4102_0202; bits[36]:0x4542_0242"
//     args: "bits[31]:0x2; bits[8]:0x55; bits[58]:0x359_a2ca_22d1_e6ab; bits[36]:0xa_00c0_0154"
//     args: "bits[31]:0x0; bits[8]:0x0; bits[58]:0x354_10d5_19f5_e75f; bits[36]:0x3_bc51_f3ce"
//     args: "bits[31]:0x214_6d44; bits[8]:0x10; bits[58]:0x155_5555_5555_5555; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x0; bits[8]:0x47; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0x6_1ddf_f7df"
//     args: "bits[31]:0x155b_63d8; bits[8]:0x80; bits[58]:0x280_1001_0001_0081; bits[36]:0xa_3724_c16b"
//     args: "bits[31]:0x5555_5555; bits[8]:0xd5; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0x100_0000"
//     args: "bits[31]:0x10_0000; bits[8]:0x11; bits[58]:0x211_0503_9957_34f4; bits[36]:0x7_1954_32f4"
//     args: "bits[31]:0x40; bits[8]:0xaa; bits[58]:0x40_0000_0000; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x5555_5555; bits[8]:0xff; bits[58]:0x0; bits[36]:0xa_aaaa_aabf"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xff; bits[58]:0x1ff_cefb_fc68_1b59; bits[36]:0xf_6c70_7371"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0x0; bits[58]:0x1ff_ffff_ffff_ffff; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x200_0000; bits[8]:0x55; bits[58]:0x0; bits[36]:0x4000_8012"
//     args: "bits[31]:0x400; bits[8]:0x8; bits[58]:0x0; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x0; bits[8]:0xdd; bits[58]:0x50_b042_c75f_7fbf; bits[36]:0x5_c000_0012"
//     args: "bits[31]:0x1000; bits[8]:0x0; bits[58]:0xc7_7c8f_5852_5fee; bits[36]:0x2_c1a1_0d82"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xff; bits[58]:0x157_9754_fb79_ce1b; bits[36]:0x4_fa7b_ce59"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x7f; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x5555_5555; bits[8]:0x10; bits[58]:0x29a_8b9b_cd07_d999; bits[36]:0x1_fa84_747e"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xff; bits[58]:0x1f7_ffff_e955_1551; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x513a_7af4; bits[8]:0x7f; bits[58]:0x299_92d7_a3aa_e32a; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x7f; bits[58]:0x3f0_f4b0_5808_29f6; bits[36]:0x0"
//     args: "bits[31]:0x4cb6_463f; bits[8]:0x6e; bits[58]:0x2_0000_0000_0000; bits[36]:0x6_96d8_c59a"
//     args: "bits[31]:0x4; bits[8]:0x98; bits[58]:0x262_aeaa_aaaa_aaaa; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xaa; bits[58]:0x0; bits[36]:0x1_17c8_0600"
//     args: "bits[31]:0x5555_5555; bits[8]:0x54; bits[58]:0x0; bits[36]:0xa_9ced_b26c"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xff; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xb_137d_df8d"
//     args: "bits[31]:0x20_0000; bits[8]:0xaa; bits[58]:0x155_5555_5555_5555; bits[36]:0x408_0006"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0x55; bits[58]:0x1ff_ffff_f800_0080; bits[36]:0xd_1422_a571"
//     args: "bits[31]:0x0; bits[8]:0x35; bits[58]:0x322_1433_05e5_2704; bits[36]:0x3_4551_1557"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xaa; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x11e7_b989; bits[8]:0xed; bits[58]:0x33c_7000_7309_a100; bits[36]:0x4_4718_b948"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xfa; bits[58]:0x0; bits[36]:0xe_4e7a_f01c"
//     args: "bits[31]:0x1000; bits[8]:0x80; bits[58]:0x8_0000; bits[36]:0x4008_0000"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x55; bits[58]:0x1dc_f7ff_d92f_eefb; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xbe; bits[58]:0x3fd_673f_b241_e188; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x5555_5555; bits[8]:0x47; bits[58]:0xaf_30a0_faed_edc5; bits[36]:0x1_0aa8_aaa4"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x0; bits[58]:0x1_d539_1b76_7f60; bits[36]:0x8_1977_7f61"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xff; bits[58]:0x24c_8cf0_8373_e49e; bits[36]:0x5_e7d3_baf6"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x0; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0x4bba_aaa8"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xff; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x2_0000; bits[8]:0x0; bits[58]:0x4_8b27_aaa1_0b5a; bits[36]:0x7_aaa1_0b5a"
//     args: "bits[31]:0x0; bits[8]:0x55; bits[58]:0x6_0008_e0bb_2b84; bits[36]:0xf_9922_c140"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xfa; bits[58]:0x1ff_ffff_ffff_ffff; bits[36]:0xc_f9fb_6f7b"
//     args: "bits[31]:0x0; bits[8]:0x4; bits[58]:0x80; bits[36]:0x8_0000_061f"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xff; bits[58]:0x233_e4b3_e841_085a; bits[36]:0xb_f841_095e"
//     args: "bits[31]:0x0; bits[8]:0x20; bits[58]:0x6c_c17b_b5ee_0cfb; bits[36]:0xb_b537_5cf3"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xff; bits[58]:0x2_0000; bits[36]:0xf_ffed_ffbf"
//     args: "bits[31]:0x6bcd_c8d7; bits[8]:0xd7; bits[58]:0x35e_6c26_b800_00e1; bits[36]:0x6_b814_44e9"
//     args: "bits[31]:0x0; bits[8]:0x80; bits[58]:0x184_110a_c15a_0562; bits[36]:0xe_d718_0752"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x55; bits[58]:0x1d7_a2aa_aaee_aa2a; bits[36]:0x5_5d3c_0d14"
//     args: "bits[31]:0x5555_5555; bits[8]:0x4; bits[58]:0x251_fb6c_f7fd_fd8f; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0x80; bits[58]:0x0; bits[36]:0x7_fdf5_ffe1"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xff; bits[58]:0x1ff_ffff_ffff_ffff; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x5555_5555; bits[8]:0x75; bits[58]:0x1d4_0000_0100_4000; bits[36]:0x0"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x98; bits[58]:0x106_5551_a184_2008; bits[36]:0xe_e51b_45c6"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xff; bits[58]:0x0; bits[36]:0x0"
//     args: "bits[31]:0x2000; bits[8]:0x91; bits[58]:0x106_05a1_4455_7655; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x33dd_11d5; bits[8]:0x20; bits[58]:0x2ed_0139_40c3_1ffa; bits[36]:0x9_4043_17fa"
//     args: "bits[31]:0x0; bits[8]:0xaa; bits[58]:0x1ff_ffff_ffff_ffff; bits[36]:0x0"
//     args: "bits[31]:0x0; bits[8]:0x0; bits[58]:0x67_00a1_05bd_7165; bits[36]:0x0"
//     args: "bits[31]:0x5555_5555; bits[8]:0xaa; bits[58]:0x0; bits[36]:0xb_2020_0501"
//     args: "bits[31]:0x6049_e41c; bits[8]:0x3d; bits[58]:0x50_1402_8522_0450; bits[36]:0xf_f878_52cd"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xfa; bits[58]:0x3f3_10a3_5aa0_2c01; bits[36]:0x7_ffff_ffff"
//     args: "bits[31]:0x420c_b770; bits[8]:0xc1; bits[58]:0x291_45b7_b82f_81bb; bits[36]:0x7_aa2f_8139"
//     args: "bits[31]:0x0; bits[8]:0x10; bits[58]:0x0; bits[36]:0x2_8719_5a1f"
//     args: "bits[31]:0x5555_5555; bits[8]:0x55; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x2533_e43a; bits[8]:0xe8; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[36]:0x2_0219_9237"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x2; bits[58]:0x117_4577_d6a0_a2aa; bits[36]:0x0"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xaa; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[36]:0x6_7fff_fdef"
//     args: "bits[31]:0x4092_6eee; bits[8]:0xee; bits[58]:0x204_9377_73ed_4eff; bits[36]:0xe_e080_480c"
//     args: "bits[31]:0x10; bits[8]:0x2; bits[58]:0x93_a119_e37e_e9ff; bits[36]:0x4_0006_028e"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xaa; bits[58]:0x155_4555_f090_f123; bits[36]:0xf_a4fd_ff57"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xff; bits[58]:0x234_6744_e6e5_c79e; bits[36]:0xf_f555_5555"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0x4; bits[58]:0x155_5555_5555_5555; bits[36]:0x7_ffff_ffff"
//     args: "bits[31]:0x5189_3b1d; bits[8]:0x81; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xa_2737_4fa8"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xff; bits[58]:0x2ee_6190_8312_5a80; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x400_0000; bits[8]:0x80; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xf_f7f7_fcff"
//     args: "bits[31]:0x5555_5555; bits[8]:0xaa; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xa_aba2_8abf"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xea; bits[58]:0x3ab_ffff_ffff_ffff; bits[36]:0x40"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xb7; bits[58]:0x155_5555_5555_5555; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xea; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0xa_0e59_06f8"
//     args: "bits[31]:0x0; bits[8]:0x18; bits[58]:0x1ff_ffff_ffff_ffff; bits[36]:0x7_ffff_ffff"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0x4; bits[58]:0x1bf_fff7_fd55_5755; bits[36]:0x6_af72_0659"
//     args: "bits[31]:0x5555_5555; bits[8]:0x55; bits[58]:0x2_0000; bits[36]:0xb_0072_a2e2"
//     args: "bits[31]:0x2db2_2379; bits[8]:0x29; bits[58]:0x17d_510b_4841_543e; bits[36]:0x5_b644_6f27"
//     args: "bits[31]:0x5555_5555; bits[8]:0x55; bits[58]:0x1000_0000; bits[36]:0x0"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0x7b; bits[58]:0x2de_ffb7_bf47_a6f9; bits[36]:0x3_f747_b751"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0x9c; bits[58]:0x3b9_30c2_b324_feb6; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x1; bits[8]:0x81; bits[58]:0x155_5555_5555_5555; bits[36]:0x0"
//     args: "bits[31]:0x5555_5555; bits[8]:0x0; bits[58]:0x0; bits[36]:0xa_aaa3_eaaa"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xaa; bits[58]:0x29e_26b7_f3b7_ffb2; bits[36]:0x7_f3b6_6fa2"
//     args: "bits[31]:0x4b4a_0fc3; bits[8]:0xf1; bits[58]:0x24c_442e_d844_6001; bits[36]:0x0"
//     args: "bits[31]:0x5555_5555; bits[8]:0x55; bits[58]:0x55_f803_a932_e23c; bits[36]:0x0"
//     args: "bits[31]:0x7fff_ffff; bits[8]:0xff; bits[58]:0x3fc_8511_f282_9d41; bits[36]:0xf_f7ff_ffcf"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0x20; bits[58]:0x3ff_ffff_ffff_ffff; bits[36]:0x4_6515_597d"
//     args: "bits[31]:0x5555_5555; bits[8]:0xd5; bits[58]:0x1fe_db4a_97cf_389b; bits[36]:0xb_af84_2bfc"
//     args: "bits[31]:0x78dd_a0fa; bits[8]:0xfa; bits[58]:0x194_fd47_cf46_d0eb; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x5555_5555; bits[8]:0x13; bits[58]:0x2d8_528c_ac17_d5fd; bits[36]:0x7_ffff_ffff"
//     args: "bits[31]:0x0; bits[8]:0xaa; bits[58]:0x32c_aaa2_aaaa_cbaa; bits[36]:0x7_ffff_ffff"
//     args: "bits[31]:0x2aaa_aaaa; bits[8]:0xab; bits[58]:0x4000_0000; bits[36]:0xf_ffff_ffff"
//     args: "bits[31]:0x5555_5555; bits[8]:0xff; bits[58]:0x3fd_adfb_feff_feee; bits[36]:0xa_aaaa_aaaa"
//     args: "bits[31]:0x3fff_ffff; bits[8]:0xbe; bits[58]:0x3e0_41a2_1011_1441; bits[36]:0x5_5555_5555"
//     args: "bits[31]:0x5555_5555; bits[8]:0xd6; bits[58]:0x2ab_aa8e_2aaa_aaaa; bits[36]:0x2_9b52_3325"
//     args: "bits[31]:0x0; bits[8]:0x55; bits[58]:0x155_5555_5555_5555; bits[36]:0x2_0000"
//     args: "bits[31]:0x6e49_a14d; bits[8]:0x7f; bits[58]:0x2e2_4f0a_2485_bb83; bits[36]:0x7_ffff_ffff"
//     args: "bits[31]:0x0; bits[8]:0x80; bits[58]:0x200; bits[36]:0x4_87fb_55d5"
//     args: "bits[31]:0x7195_4ed4; bits[8]:0xdc; bits[58]:0x18_a37e_ef5f_d049; bits[36]:0x4a0a_7ecd"
//     args: "bits[31]:0x5555_5555; bits[8]:0x5a; bits[58]:0x2aa_eaab_8f60_2ec0; bits[36]:0xf_e6ca_2e00"
//     args: "bits[31]:0x0; bits[8]:0xe0; bits[58]:0xb6_556d_b245_9577; bits[36]:0xd_b2c5_8573"
//     args: "bits[31]:0x200; bits[8]:0x55; bits[58]:0x2_0000_0000; bits[36]:0x5_5555_5555"
//   }
// }
// 
// END_CONFIG
const W32_V1 = u32:0x1;
type x10 = u24;
fn main(x0: s31, x1: s8, x2: s58, x3: u36) -> (u36, (s58, s8, u24, u2, u36), u2, s31, u36, u13) {
    {
        let x4: u2 = x3[0+:u2];
        let x5: s8 = x1 / s8:0xff;
        let x6: u24 = x3[x4+:u24];
        let x7: u2 = x5 as u2 - x4;
        let x8: u24 = !x6;
        let x11: x10[W32_V1] = x6 as x10[W32_V1];
        let x12: x10[2] = x11 ++ x11;
        let x13: (s58, s8, u24, u2, u36) = (x2, x1, x8, x4, x3);
        let x14: s31 = !x0;
        let x15: u49 = u49:0x1_5555_5555_5555;
        let x16: x10[16] = array_slice(x11, x8, x10[16]:[x11[u32:0x0], ...]);
        let x17: s31 = x14 / s31:0x80;
        let x18: u24 = -x8;
        let x19: u36 = signex(x5, x3);
        let x20: s63 = s63:0x5555_5555_5555_5555;
        let x21: s58 = x2 << if x8 >= u24:0xf { u24:0xf } else { x8 };
        let x22: bool = x12 != x12;
        let x23: u22 = x8[2:];
        let x24: u2 = one_hot(x22, bool:0x0);
        let x25: x10 = x11[if x19 >= u36:0x0 { u36:0x0 } else { x19 }];
        let x26: u13 = x18[11:];
        let x27: u22 = x24 as u22 & x23;
        (x3, x13, x7, x17, x19, x26)
    }
}
