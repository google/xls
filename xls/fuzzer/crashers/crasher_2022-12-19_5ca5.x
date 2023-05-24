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
// evaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[38]:0x20_55f7_82cb, bits[58]:0x3ff_ffff_ffff_ffff, bits[59]:0x2000_052d, bits[24]:0x0)
// evaluated unopt IR (JIT) =
//    (bits[38]:0x20_55f7_82cb, bits[58]:0x3ff_ffff_ffff_ffff, bits[59]:0x2000_052d, bits[24]:0x91_4c3e)
// (run dir: /tmp/)
//
// BEGIN_CONFIG
// exception: "// Result miscompare for sample 0:"
// issue: "https://github.com/google/xls/issues/815"
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
// }
// inputs {
//   function_args {
//     args: "bits[18]:0x3_ffff; bits[24]:0x91_4c3e; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x52d; bits[38]:0x20_55f7_82cb; bits[59]:0x2000_0000"
//     args: "bits[18]:0x3_ffff; bits[24]:0x91_4c3e; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x52d; bits[38]:0x20_55f7_82cb; bits[59]:0x2000_0000"
//     args: "bits[18]:0x400; bits[24]:0x1004; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x7ff; bits[38]:0x2a_aaaa_aaaa; bits[59]:0x671_5735_d54b_5cf7"
//     args: "bits[18]:0x4000; bits[24]:0x10_003f; bits[58]:0x4e_42f7_d76f_bff7; bits[11]:0x776; bits[38]:0x6_2003_a06f; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x9c_1cbb; bits[58]:0x2bb_9466_9928_e45a; bits[11]:0x7ff; bits[38]:0x2a_888f_efab; bits[59]:0x115_70fa_4485_82e5"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x1; bits[58]:0x54_79e7_2cb6_30af; bits[11]:0x20; bits[38]:0x15_5555_5555; bits[59]:0x7d5_56df_cef7_f3df"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xaa_aaaa; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x97; bits[38]:0x2c_ebaa_9551; bits[59]:0x100"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x7f_ffff; bits[58]:0x0; bits[11]:0x80; bits[38]:0x15_5555_5555; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x40; bits[24]:0xd9_44b1; bits[58]:0x2d6_4bc5_5753_6053; bits[11]:0x0; bits[38]:0x2a_aaaa_aaaa; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x2_2f8f; bits[24]:0x0; bits[58]:0x89_08e0_8ec1_3be1; bits[11]:0x67f; bits[38]:0x3f_ffff_ffff; bits[59]:0x3fd_b8fb_dfe2_0341"
//     args: "bits[18]:0x3_ffff; bits[24]:0xb9_e3f3; bits[58]:0x1f7_ffb3_aa8a_a8a8; bits[11]:0x3f3; bits[38]:0x0; bits[59]:0x4e6_ef66_31d5_30bd"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x0; bits[58]:0x22a_86be_acaa_baa8; bits[11]:0x2a2; bits[38]:0x8_de25_fd7c; bits[59]:0x457_0d7d_5915_7150"
//     args: "bits[18]:0x1_5555; bits[24]:0x55_752a; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[11]:0x2aa; bits[38]:0x15_5167_073f; bits[59]:0x3ff_ffff_ffff_ffff"
//     args: "bits[18]:0x0; bits[24]:0x2_0000; bits[58]:0x3ff_0041_ef55_9d6f; bits[11]:0x0; bits[38]:0x10_a060_5dff; bits[59]:0x0"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xad_ee95; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x2aa; bits[38]:0x200_0000; bits[59]:0x10_4d7f_4f0a_0455"
//     args: "bits[18]:0x1_0000; bits[24]:0x58_0000; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x10e; bits[38]:0x16_4400_1cff; bits[59]:0x1000_0000"
//     args: "bits[18]:0x1_ffff; bits[24]:0xdf_2fdb; bits[58]:0x6f_a73b_7535_7f52; bits[11]:0x5a; bits[38]:0x2a_aaaa_aaaa; bits[59]:0x9f_6ef6_ea4b_fe85"
//     args: "bits[18]:0x0; bits[24]:0xaa_aaaa; bits[58]:0x2aa_ebaa_a2aa_ae83; bits[11]:0x200; bits[38]:0x1c_aa40_00fc; bits[59]:0x48e_fada_8aea_10c5"
//     args: "bits[18]:0xf337; bits[24]:0x5c_cca0; bits[58]:0x0; bits[11]:0x4e8; bits[38]:0x11_97a4_3cf1; bits[59]:0x5c8_c5f6_d86b_dfaa"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xaa_aa91; bits[58]:0x3a0_0fcf_d736_b1ea; bits[11]:0x3ae; bits[38]:0x38_a8a6_79f7; bits[59]:0x20_0000_0000"
//     args: "bits[18]:0x1_ffff; bits[24]:0xf9_23a6; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x3ff; bits[38]:0x0; bits[59]:0x3ff_ffff_ffff_ffff"
//     args: "bits[18]:0x2; bits[24]:0x10_809b; bits[58]:0x18_8603_d097_e855; bits[11]:0x5cc; bits[38]:0x2a_f261_0948; bits[59]:0x56f_64f7_a0c9_aa4e"
//     args: "bits[18]:0x3_ffff; bits[24]:0x0; bits[58]:0x3ff_f1c9_0650_61d0; bits[11]:0x0; bits[38]:0x15_5555_5555; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x0; bits[24]:0xa014; bits[58]:0x80_0000_0002; bits[11]:0x0; bits[38]:0x3f_ffff_ffff; bits[59]:0x129_5557_5752_3115"
//     args: "bits[18]:0x3_ffff; bits[24]:0xff_7f3a; bits[58]:0x3ff_ff40_0480_4200; bits[11]:0x73e; bits[38]:0x2f_b37f_5a4d; bits[59]:0x62e_61c4_acf8_b680"
//     args: "bits[18]:0x80; bits[24]:0x55_5555; bits[58]:0x0; bits[11]:0x80; bits[38]:0x800; bits[59]:0x4_0000"
//     args: "bits[18]:0x3_ffff; bits[24]:0x55_5555; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[11]:0x7ff; bits[38]:0x1f_ffff_ffff; bits[59]:0x555_5555_5555_5555"
//     args: "bits[18]:0x1_5555; bits[24]:0x7f_ffff; bits[58]:0x10; bits[11]:0x0; bits[38]:0x21_bd3f_7ff7; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[24]:0xde_fec4; bits[58]:0x37b_fb01_557d_5554; bits[11]:0x2aa; bits[38]:0x2a_aaaa_aaaa; bits[59]:0x2f7_f223_9ab2_6244"
//     args: "bits[18]:0x1_ffff; bits[24]:0xff_fef2; bits[58]:0x9b_de5c_0551_5578; bits[11]:0x3ff; bits[38]:0x3f_ffff_ffff; bits[59]:0x7fe_afbf_f7b3_faff"
//     args: "bits[18]:0x1_5555; bits[24]:0x90_5d4c; bits[58]:0x2cb_549c_189a_5e13; bits[11]:0x7d3; bits[38]:0x1e_df98_aaa6; bits[59]:0x0"
//     args: "bits[18]:0x1; bits[24]:0x70_4f51; bits[58]:0xd1_b535_323b_f1b0; bits[11]:0x1e2; bits[38]:0x23_509d_bfcb; bits[59]:0x306_e817_8cbe_7aa8"
//     args: "bits[18]:0x1000; bits[24]:0x4c_203a; bits[58]:0x130_80a9_2080_0200; bits[11]:0x779; bits[38]:0x2a_aaaa_aaaa; bits[59]:0x3a5_a8d1_1084_c1d7"
//     args: "bits[18]:0x2_19ec; bits[24]:0x86_7900; bits[58]:0x210_6c06_a040_0610; bits[11]:0x3ff; bits[38]:0xe_91c2_2501; bits[59]:0x0"
//     args: "bits[18]:0x0; bits[24]:0x0; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x3ff; bits[38]:0x8_0027_fbff; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[24]:0x5e_ffc3; bits[58]:0x1da_ff0e_45df_ffff; bits[11]:0x7ff; bits[38]:0x2a_77f7_90bd; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x0; bits[24]:0x17_f0fb; bits[58]:0x19f_cbae_8a8a_8eab; bits[11]:0x4; bits[38]:0x0; bits[59]:0x4ab_075a_a28c_2d49"
//     args: "bits[18]:0x2_a197; bits[24]:0x7c_fc1c; bits[58]:0x15b_7178_4004_02c5; bits[11]:0x0; bits[38]:0x3a_3016_03b4; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1_ffff; bits[24]:0x69_f548; bits[58]:0x2db_ee6f_e41b_826a; bits[11]:0x57a; bits[38]:0x0; bits[59]:0x0"
//     args: "bits[18]:0x400; bits[24]:0x7f_ffff; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[11]:0x2aa; bits[38]:0x24_6e20_caeb; bits[59]:0x48a_4811_4000_00c1"
//     args: "bits[18]:0x1_ffff; bits[24]:0x55_5555; bits[58]:0x33f_0108_2864_4898; bits[11]:0x2e0; bits[38]:0x1f_ffff_ffff; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1_5555; bits[24]:0x41_d45d; bits[58]:0x133_c133_f7fb_fbed; bits[11]:0x3aa; bits[38]:0x1f_937e_9bff; bits[59]:0x3ba_a405_4a5a_e004"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x55_5555; bits[58]:0x115_5e62_a82a_aaee; bits[11]:0x3ff; bits[38]:0x1f_ffff_ffff; bits[59]:0x3bf_b777_bf79_4808"
//     args: "bits[18]:0x3_ffff; bits[24]:0xa2_bfbf; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x6af; bits[38]:0x15_5555_5555; bits[59]:0x50f_cce8_a833_60c8"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xab_ab83; bits[58]:0x0; bits[11]:0x0; bits[38]:0x10_0000; bits[59]:0x57d_5c08_8110_0980"
//     args: "bits[18]:0x10; bits[24]:0x8_274d; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x555; bits[38]:0x21_01c3_4000; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1_5555; bits[24]:0xaa_aaaa; bits[58]:0x35e_672a_b623_b1b4; bits[11]:0x7ff; bits[38]:0x2a_a6a8_8d4f; bits[59]:0x3ff_0010_0000_8000"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xaa_aaaa; bits[58]:0x22a_aa47_d255_5555; bits[11]:0x70a; bits[38]:0x3f_ffff_ffff; bits[59]:0x42e_cd28_a15a_2af3"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x43d2; bits[58]:0x1_0f49_fffe_ffff; bits[11]:0x2a2; bits[38]:0x20_5beb_e223; bits[59]:0x495_b2fd_5efd_7eee"
//     args: "bits[18]:0x2000; bits[24]:0x40_00ae; bits[58]:0x50_00ab_233a_0889; bits[11]:0x0; bits[38]:0x14_012b_8000; bits[59]:0x1_0000_0000"
//     args: "bits[18]:0x3930; bits[24]:0x7f_ffff; bits[58]:0xc3_1b86_6abb_180a; bits[11]:0x0; bits[38]:0x15_5555_5555; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x3_ffff; bits[24]:0xff_ffff; bits[58]:0x2b4_c0e9_d591_0807; bits[11]:0x55; bits[38]:0x15_5555_5555; bits[59]:0x0"
//     args: "bits[18]:0x3_ffff; bits[24]:0xff_ffff; bits[58]:0x2ff_bf77_fffb_ffde; bits[11]:0xfa; bits[38]:0xa_edfd_ce2a; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_5555; bits[24]:0xd2_0d2d; bits[58]:0x1d2_1eb4_4401_1028; bits[11]:0x2aa; bits[38]:0x31_fdbf_5ebb; bits[59]:0x3ff_ffff_ffff_ffff"
//     args: "bits[18]:0x3_ffff; bits[24]:0xf7_ffe8; bits[58]:0x377_f77f_fcde_fbff; bits[11]:0x671; bits[38]:0x3b_8915_151d; bits[59]:0x92_cac2_191d_de83"
//     args: "bits[18]:0x1_5555; bits[24]:0x54_5d53; bits[58]:0x155_5555_5555_5555; bits[11]:0x80; bits[38]:0x2_42fe_9fef; bits[59]:0x613_2da9_b5f7_b233"
//     args: "bits[18]:0x3_ffff; bits[24]:0x55_5555; bits[58]:0x3f7_d94a_f7bf_fffe; bits[11]:0x7be; bits[38]:0x8_0000; bits[59]:0x4a_2080_2000_8e00"
//     args: "bits[18]:0x1_5555; bits[24]:0x75_7d55; bits[58]:0x3d5_f500_0008_0000; bits[11]:0x76f; bits[38]:0x3f_eaaf_a222; bits[59]:0x3ab_faab_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[24]:0x57_f52f; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[11]:0x79f; bits[38]:0x32_9cc4_e5fb; bits[59]:0x555_5555_5555_5555"
//     args: "bits[18]:0x3_6ce5; bits[24]:0x7f_ffff; bits[58]:0x1de_f8d8_f6ff_bfda; bits[11]:0x6fd; bits[38]:0x25_6fd2_bed7; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[24]:0x53_8dea; bits[58]:0x3ce_27a2_e8ea_8ba2; bits[11]:0x54b; bits[38]:0x22_c869_b9a0; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x0; bits[24]:0x7f_ffff; bits[58]:0x120_36f4_9ea0_16ce; bits[11]:0x0; bits[38]:0x2a_aaaa_aaaa; bits[59]:0x3fe_f73d_5155_5555"
//     args: "bits[18]:0x40; bits[24]:0x57_8375; bits[58]:0xc855_5551_55d5; bits[11]:0x20; bits[38]:0x5_8405_9dd2; bits[59]:0x500_850d_c150_0016"
//     args: "bits[18]:0x1_5555; bits[24]:0x0; bits[58]:0x0; bits[11]:0x0; bits[38]:0x1b_8c91_dde8; bits[59]:0x628_2b54_df15_1411"
//     args: "bits[18]:0x3_4db3; bits[24]:0xaa_aaaa; bits[58]:0x34d_b36f_ffcd_effe; bits[11]:0x2aa; bits[38]:0x15_5555_5555; bits[59]:0x0"
//     args: "bits[18]:0x8000; bits[24]:0x29_c367; bits[58]:0x0; bits[11]:0x480; bits[38]:0x8_1105_96c0; bits[59]:0x100_0000_0000_0000"
//     args: "bits[18]:0x800; bits[24]:0x13_7237; bits[58]:0x4d_c8d5_4002_11a7; bits[11]:0x0; bits[38]:0x16_daed_77e9; bits[59]:0x201_00cd_5104_783d"
//     args: "bits[18]:0x3_ffff; bits[24]:0xff_fffc; bits[58]:0x0; bits[11]:0x4; bits[38]:0x5_3abf_f538; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_5555; bits[24]:0x0; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[11]:0x3ff; bits[38]:0xd_ad59_efe3; bits[59]:0x3b3_bbaf_ae7b_ecaf"
//     args: "bits[18]:0x3_ffff; bits[24]:0xff_ef6e; bits[58]:0x1ab_b7b3_a1a2_2453; bits[11]:0x7b7; bits[38]:0x15_5555_5555; bits[59]:0x0"
//     args: "bits[18]:0x3_ffff; bits[24]:0x40; bits[58]:0x4102_f7df_77ff; bits[11]:0x7ff; bits[38]:0x6_6ba2_1f7c; bits[59]:0x3ff_ffff_ffff_ffff"
//     args: "bits[18]:0x0; bits[24]:0x1e_8932; bits[58]:0x70_20ce_5423_15fe; bits[11]:0x3ff; bits[38]:0x0; bits[59]:0x59f_09ee_6570_f4f7"
//     args: "bits[18]:0x3_ffff; bits[24]:0xad_aff7; bits[58]:0x200_0000; bits[11]:0x7ff; bits[38]:0x1f_ffff_ffff; bits[59]:0x0"
//     args: "bits[18]:0x3_dc7d; bits[24]:0x0; bits[58]:0x12_aaee_abae; bits[11]:0x4a1; bits[38]:0x8000_0000; bits[59]:0x25_55dd_575d"
//     args: "bits[18]:0x0; bits[24]:0xff_ffff; bits[58]:0x0; bits[11]:0x2aa; bits[38]:0x4_562a_eba8; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1000; bits[24]:0xff_ffff; bits[58]:0x10_0008_0000_0208; bits[11]:0x555; bits[38]:0x2a_aaaa_aaaa; bits[59]:0x3ff_ffff_ffff_ffff"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xaa_ae95; bits[58]:0xba_aa5e_869f_d9b3; bits[11]:0x6d5; bits[38]:0x10; bits[59]:0x2d5_01b4_8880_1282"
//     args: "bits[18]:0x3_ffff; bits[24]:0x10; bits[58]:0x3ff_ff57_5555_1554; bits[11]:0x7ff; bits[38]:0x15_5555_5555; bits[59]:0x73e_42f3_6685_0f60"
//     args: "bits[18]:0x3_ffff; bits[24]:0xfe_7f42; bits[58]:0x155_5555_5555_5555; bits[11]:0x3ff; bits[38]:0x3f_9fd0_bfff; bits[59]:0x555_5555_5555_5555"
//     args: "bits[18]:0x3_ffff; bits[24]:0xff_fbc1; bits[58]:0x3ff_cf47_fdd0_3af7; bits[11]:0x0; bits[38]:0x17_5e9d_ffbf; bits[59]:0x10_0000_0000"
//     args: "bits[18]:0x96e5; bits[24]:0x35_2b80; bits[58]:0x200_0000_0000_0000; bits[11]:0x110; bits[38]:0x3f_ffff_ffff; bits[59]:0x2b6_452b_79bd_fa3f"
//     args: "bits[18]:0x8000; bits[24]:0xb0_000e; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x7fa; bits[38]:0x15_5555_5555; bits[59]:0x7ff_cce3_eb9a_51f1"
//     args: "bits[18]:0x0; bits[24]:0x86_802a; bits[58]:0x0; bits[11]:0x104; bits[38]:0x15_5555_5555; bits[59]:0x6d6_d6b1_3a14_935c"
//     args: "bits[18]:0x3_ffff; bits[24]:0xbf_bfce; bits[58]:0x2fe_df38_f7fc_ddd7; bits[11]:0x7bf; bits[38]:0x8_3106_a1c9; bits[59]:0x106_72d4_3932_0001"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x55_5555; bits[58]:0x155_5c75_5555_5576; bits[11]:0x0; bits[38]:0x1_4a55_147e; bits[59]:0x115_caa5_35c0_0a86"
//     args: "bits[18]:0x3_ffff; bits[24]:0xf3_bf6b; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x36f; bits[38]:0x3f_fff0_95b3; bits[59]:0x187_9ed4_7172_2664"
//     args: "bits[18]:0x2; bits[24]:0xaa_aaaa; bits[58]:0x7ff_fdff_ff7f; bits[11]:0xa; bits[38]:0x3a_2aaa_9f6b; bits[59]:0x2_0000_0000_0000"
//     args: "bits[18]:0x0; bits[24]:0x50_2823; bits[58]:0x2ec_ead8_c26f_a3cf; bits[11]:0x3cf; bits[38]:0x14_161a_81c0; bits[59]:0x3de_2898_4498_0241"
//     args: "bits[18]:0x5b7b; bits[24]:0x7f_ffff; bits[58]:0x40_0000_0000_0000; bits[11]:0x3ff; bits[38]:0x2a_6dee_1357; bits[59]:0x1ff_441b_16f0_52f2"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xb7_90cf; bits[58]:0x29b_ab01_1b14_41b8; bits[11]:0x3b3; bits[38]:0x2a_69aa_dfbf; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x0; bits[24]:0x57_2549; bits[58]:0x2000_0000; bits[11]:0x101; bits[38]:0x3c_d622_9466; bits[59]:0x7f4_3c40_a106_6d20"
//     args: "bits[18]:0x200; bits[24]:0x74_c055; bits[58]:0x800_0000; bits[11]:0x685; bits[38]:0x3_5820_6439; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1_ffff; bits[24]:0x0; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x326; bits[38]:0x3f_ffff_ffff; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x56a0; bits[24]:0x85_3e2e; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x6a0; bits[38]:0x37_1151_5451; bits[59]:0xac_6155_d615_5060"
//     args: "bits[18]:0x2_5b43; bits[24]:0xff_ffff; bits[58]:0x25b_43ff_ffff_fbdb; bits[11]:0x7ff; bits[38]:0x37_bfb7_d4f7; bits[59]:0x556_e67e_e5ab_d9ff"
//     args: "bits[18]:0x1_ffff; bits[24]:0x5b_9d73; bits[58]:0x3bb_f6ea_12a2_889a; bits[11]:0x573; bits[38]:0x1e_a968_8d9a; bits[59]:0x60c_3920_91ee_792e"
//     args: "bits[18]:0x1_5555; bits[24]:0x7f_ffff; bits[58]:0x3ef_cffc_82db_d9a7; bits[11]:0x1; bits[38]:0x36_4717_2105; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1_ffff; bits[24]:0xaa_aaaa; bits[58]:0x156_6abc_169e_0ba9; bits[11]:0x3ff; bits[38]:0x1f_bce3_a7b0; bits[59]:0x4"
//     args: "bits[18]:0x1_ffff; bits[24]:0x7f_ffff; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x2aa; bits[38]:0xc_f2e6_13a3; bits[59]:0x1d1_d005_c3c7_96a7"
//     args: "bits[18]:0x0; bits[24]:0x8_4128; bits[58]:0xa_60c4_8f10_6701; bits[11]:0x2aa; bits[38]:0x3f_ffff_ffff; bits[59]:0xc0_ab02_804d_0ea0"
//     args: "bits[18]:0x1_5555; bits[24]:0x55_5d6a; bits[58]:0x155_5555_5555_5555; bits[11]:0x555; bits[38]:0x3f_ffff_ffff; bits[59]:0x2a8_6baa_aaaa_a9ab"
//     args: "bits[18]:0x1_ffff; bits[24]:0xe0_2604; bits[58]:0x155_5555_5555_5555; bits[11]:0x7ff; bits[38]:0x3b_f7f8_4450; bits[59]:0x7ff_ba5e_8ad8_0483"
//     args: "bits[18]:0x40; bits[24]:0x21_d0a4; bits[58]:0x0; bits[11]:0x3ff; bits[38]:0x3f_ffff_ffff; bits[59]:0x2_0000"
//     args: "bits[18]:0x1_5555; bits[24]:0x0; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x2aa; bits[38]:0x1f_ffff_ffff; bits[59]:0x2aa_aaff_ffff_ffff"
//     args: "bits[18]:0x2_72d8; bits[24]:0x9c_b62a; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x7ff; bits[38]:0x26_278a_ffff; bits[59]:0x400_0000"
//     args: "bits[18]:0x20; bits[24]:0x11_0a7e; bits[58]:0x30_f0da_ff7a_3fbb; bits[11]:0x735; bits[38]:0x1f_ffff_ffff; bits[59]:0x7af_a2bd_8402_2931"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x7f_ffff; bits[58]:0x1f2_0d8d_6444_bb08; bits[11]:0x555; bits[38]:0x1f_ffbe_e2aa; bits[59]:0x2_0000_0000_0000"
//     args: "bits[18]:0x1_18aa; bits[24]:0x55_5555; bits[58]:0x118_bff3_8b63_2a15; bits[11]:0x6a; bits[38]:0x15_5555_5555; bits[59]:0x20_0000_0000_0000"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xa8_ae39; bits[58]:0x2aa_aa00_0400_8088; bits[11]:0x48a; bits[38]:0x8_0000_0000; bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x8; bits[24]:0x15; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x555; bits[38]:0x1000; bits[59]:0x280_cc02_9055_5d5d"
//     args: "bits[18]:0x3_719a; bits[24]:0x1a_9999; bits[58]:0xd8_674e_fb67_cc27; bits[11]:0x189; bits[38]:0x3f_ffff_ffff; bits[59]:0x57e_c5d8_d77b_b279"
//     args: "bits[18]:0x0; bits[24]:0x62_2c3a; bits[58]:0x343_a4c0_db87_64ce; bits[11]:0x3a9; bits[38]:0x32_105f_f3ed; bits[59]:0x555_5555_5555_5555"
//     args: "bits[18]:0x3_ffff; bits[24]:0xff_ffff; bits[58]:0x0; bits[11]:0x400; bits[38]:0x1f_ffff_ffff; bits[59]:0x5fc_bbfb_bbec_7ef9"
//     args: "bits[18]:0x2_aaaa; bits[24]:0xc_e154; bits[58]:0x207_8c5d_4395_9a44; bits[11]:0x1bc; bits[38]:0x33_3dd9_57bf; bits[59]:0x3ff_ffff_ffff_ffff"
//     args: "bits[18]:0x9661; bits[24]:0x55_5555; bits[58]:0x3ff_ffff_ffff_ffff; bits[11]:0x517; bits[38]:0x1_2710_4c40; bits[59]:0x2b8_7fd1_5d32_57c2"
//     args: "bits[18]:0x0; bits[24]:0xff_ffff; bits[58]:0x336_b345_2140_20d8; bits[11]:0x9e; bits[38]:0x5_fb40_0180; bits[59]:0x18a_e704_ca70_fa6b"
//     args: "bits[18]:0x1_ffff; bits[24]:0x20_0000; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[11]:0x8; bits[38]:0x0; bits[59]:0x11_a800_04f0_048f"
//     args: "bits[18]:0x1_8641; bits[24]:0x61_90df; bits[58]:0x23c_aa26_6499_8d7e; bits[11]:0x3b; bits[38]:0x18_e090_4bc1; bits[59]:0x8000"
//     args: "bits[18]:0x1_ffff; bits[24]:0x3f_73ea; bits[58]:0x1db_7d1f_d0d8_1b79; bits[11]:0x329; bits[38]:0x3f_ffff_ffff; bits[59]:0x2"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x6a_a28b; bits[58]:0x8e_8a2f_5775_455d; bits[11]:0x249; bits[38]:0xb_730a_7793; bits[59]:0x0"
//     args: "bits[18]:0x1_ffff; bits[24]:0x7f_ffc0; bits[58]:0x1ff_ffff_ffff_ffff; bits[11]:0x7c0; bits[38]:0x3f_ffff_ffff; bits[59]:0x5a4_d767_b24e_882e"
//     args: "bits[18]:0x1_ffff; bits[24]:0x7f_ffdf; bits[58]:0x0; bits[11]:0x2aa; bits[38]:0x33_f9c1_be58; bits[59]:0x6f3_b5fc_8ca6_6c66"
//     args: "bits[18]:0x1_ffff; bits[24]:0x75_fbb8; bits[58]:0x1c6_dae6_0182_0340; bits[11]:0x5c4; bits[38]:0x37_efaf_ffdc; bits[59]:0x7ff_ffff_ffff_ffff"
//     args: "bits[18]:0x3_ffff; bits[24]:0xef_db48; bits[58]:0x3b3_f502_4005_0026; bits[11]:0x555; bits[38]:0x2000; bits[59]:0x459_06d5_d60b_53f4"
//     args: "bits[18]:0x2_5142; bits[24]:0x94_51b7; bits[58]:0x251_4200_0020_0080; bits[11]:0x3ff; bits[38]:0x37_bf0c_2e81; bits[59]:0x3ff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[24]:0xff_ffff; bits[58]:0x1d8_df7a_fe5b_7fc2; bits[11]:0x7ff; bits[38]:0x1_0000; bits[59]:0x0"
//     args: "bits[18]:0x2_aaaa; bits[24]:0x55_5555; bits[58]:0x17d_5ccf_08fa_4b3c; bits[11]:0x555; bits[38]:0x17_d555_5571; bits[59]:0x555_5555_5555_5555"
//     args: "bits[18]:0x1_ffff; bits[24]:0x1c_cf57; bits[58]:0x1ff_ffaa_aaaa_aaaa; bits[11]:0x757; bits[38]:0x1f_ffff_ffff; bits[59]:0x346_0543_8824_0d6a"
//     args: "bits[18]:0x1_5555; bits[24]:0xff_ffff; bits[58]:0x3df_ffef_affd_fff7; bits[11]:0x7f3; bits[38]:0x33_dc15_83b0; bits[59]:0x7ff_ffff_ffff_ffff"
//   }
// }
// END_CONFIG
fn main(x0: u18, x1: u24, x2: u58, x3: u11, x4: u38, x5: u59) -> (u38, u58, u59, u24) {
  let x6: bool = (x0) > (((x5) as u18));
  let x7: u24 = gate!((((x1) as u59)) <= (x5), x1);
  let x8: bool = (x5) != (((x2) as u59));
  let x9: u17 = (x7)[0+:u17];
  let x10: u24 = gate!((((x8) as bool)) != (x6), x1);
  let x11: u24 = -(x1);
  let x12: u12 = u12:0xc46;
  let x13: bool = (x12) >= (((x6) as u12));
  let x14: u59 = (x5)[:];
  let x15: u58 = (x2) >> (if (x10) >= (u24:0x12) { u24:0x12 } else { x10 });
  let x16: u51 = (x2)[0+:u51];
  let x17: u59 = (((x3) as u59)) + (x5);
  let x18: bool = (x6) ^ (((x13) as bool));
  let x19: bool = (x8) & (((x13) as bool));
  let x20: u18 = (x0) ^ (((x2) as u18));
  let x21: bool = (x8) + (((x5) as bool));
  let x22: u18 = (x20) >> (if (x16) >= (u51:0xf) { u51:0xf } else { x16 });
  let x23: u1 = u1:false;
  let x24: bool = bit_slice_update(x13, x7, x18);
  let x25: u24 = (x10) * (((x24) as u24));
  (x4, x15, x17, x25)
}
