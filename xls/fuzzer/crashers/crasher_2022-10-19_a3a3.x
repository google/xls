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
//
// BEGIN_CONFIG
// exception: "// Command \'[\'/xls/tools/eval_ir_main\', \'--input_file=args.txt\', \'--use_llvm_jit\', \'sample.ir\', \'--logtostderr\']\' returned non-zero exit status 1."
// issue: "https://github.com/google/xls/issues/746"
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
//   timeout_seconds: 600
//   calls_per_sample: 128
// }
// inputs {
//   function_args {
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0x1_ffff_fe53_ef26_3dbd_34bb_1036; bits[29]:0x14bb_1036"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0x1_a1ab_f29b_0f6a_e82a_804b_8923; bits[29]:0x1000"
//     args: "bits[1]:0x0; bits[23]:0x3a_7c7e; bits[97]:0x400; bits[29]:0xe9f_1faa"
//     args: "bits[1]:0x1; bits[23]:0x13_5557; bits[97]:0x200; bits[29]:0x2_0000"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x21_9a4a; bits[97]:0x9e6e_8a27_9b78_7cbc_56b7_ea49; bits[29]:0x0"
//     args: "bits[1]:0x1; bits[23]:0x0; bits[97]:0x1_ffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1efd_67dc"
//     args: "bits[1]:0x0; bits[23]:0x3f_ffff; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x25_2943; bits[97]:0x1_a4b9_c03a_4464_183d_dcf3_534b; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x0; bits[23]:0x42_73c7; bits[97]:0x0; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x7f_ffff; bits[97]:0xffff_fbff_77fe_fab7_ff67_ffbd; bits[29]:0x1b27_f5a8"
//     args: "bits[1]:0x1; bits[23]:0x5f_fffe; bits[97]:0x1_62c4_de6c_3d43_ff48_d471_ba75; bits[29]:0x14f1_ba75"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0x1_fb7f_ffdd_ebdf_ffbd_bfef_f7fe; bits[29]:0x192e_6519"
//     args: "bits[1]:0x0; bits[23]:0x15_5555; bits[97]:0x0; bits[29]:0x52b_7655"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x0"
//     args: "bits[1]:0x0; bits[23]:0x2a_aaaa; bits[97]:0x2005_c422_6894_520e_0040_0840; bits[29]:0x400"
//     args: "bits[1]:0x0; bits[23]:0x15_4555; bits[97]:0x1_c17e_a1d7_aefc_0491_25a9_aa7b; bits[29]:0x13_2c48"
//     args: "bits[1]:0x1; bits[23]:0x76_c1ac; bits[97]:0x1_9b2a_b1fd_bffb_cfef_ffff_7ffe; bits[29]:0x1555_d555"
//     args: "bits[1]:0x0; bits[23]:0x17_6dfd; bits[97]:0x69bf_f6a8_bf92_a2a8_2fa8_28aa; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0xefef_acef_fdfd_6eff_efc9_ffff; bits[29]:0x1944_7515"
//     args: "bits[1]:0x0; bits[23]:0x4f_eef5; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x1145_4141"
//     args: "bits[1]:0x0; bits[23]:0x10_00b4; bits[97]:0x76bf_cbff_feed_efff_ffbd_dff7; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x5a_b360; bits[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[29]:0x1010_0807"
//     args: "bits[1]:0x0; bits[23]:0x7f_ec37; bits[97]:0xa0_4400_8000_0812_0080_8cc7; bits[29]:0xb59_40de"
//     args: "bits[1]:0x0; bits[23]:0x15_55d5; bits[97]:0x3287_3ba6_283a_feb5_ff1f_0aab; bits[29]:0x64a_1010"
//     args: "bits[1]:0x0; bits[23]:0x2a_aaaa; bits[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[29]:0x710_e6b0"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1c8d_e661"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0x0; bits[29]:0x13_8084"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0x1_8195_7d17_c7c2_55e1_ead0_2d8f; bits[29]:0x0"
//     args: "bits[1]:0x1; bits[23]:0x2c_0012; bits[97]:0xd720_5ff7_a67e_9d17_bdda_ffd8; bits[29]:0x1555_5555"
//     args: "bits[1]:0x1; bits[23]:0x54_54a8; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x149d_b81a"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0x1_ffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1d7f_dfff"
//     args: "bits[1]:0x1; bits[23]:0xa_284d; bits[97]:0x28e1_3400_0001_3000_4000_8006; bits[29]:0x1555_5555"
//     args: "bits[1]:0x0; bits[23]:0x3f_ffff; bits[97]:0x4bee_2fa6_f6b2_bfbf_79dd_edf7; bits[29]:0x1555_5555"
//     args: "bits[1]:0x0; bits[23]:0x6b_52b6; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x1dc2_4506"
//     args: "bits[1]:0x1; bits[23]:0x2a_aaaa; bits[97]:0x0; bits[29]:0x801_c20e"
//     args: "bits[1]:0x1; bits[23]:0x59_315b; bits[97]:0x0; bits[29]:0x1f9d_b3ba"
//     args: "bits[1]:0x1; bits[23]:0x7800; bits[97]:0x1_ffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x771_b462"
//     args: "bits[1]:0x1; bits[23]:0x5f_5b1d; bits[97]:0x1_91a6_7031_b2c1_427a_6255_4032; bits[29]:0x1396_269a"
//     args: "bits[1]:0x0; bits[23]:0x2a_aaaa; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x180a_b870"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0xe7cf_cea6_53f6_698e_1c12_2e97; bits[29]:0x1f03_9e57"
//     args: "bits[1]:0x0; bits[23]:0x1f_fffe; bits[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[29]:0xb1d_eae6"
//     args: "bits[1]:0x0; bits[23]:0x46_75e9; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x0"
//     args: "bits[1]:0x0; bits[23]:0x0; bits[97]:0xcfb_5d14_b7a8_d5b5_f1f5_e5e4; bits[29]:0x10f5_e5e4"
//     args: "bits[1]:0x1; bits[23]:0x40_0000; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x1000_0027"
//     args: "bits[1]:0x1; bits[23]:0x2a_aaaa; bits[97]:0x0; bits[29]:0x1555_5555"
//     args: "bits[1]:0x1; bits[23]:0x66_e338; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x1; bits[23]:0x1f_3602; bits[97]:0x1_0004_0204_0008_0000_8000_0802; bits[29]:0x8_0103"
//     args: "bits[1]:0x0; bits[23]:0x27_dfdf; bits[97]:0x0; bits[29]:0x1555_5555"
//     args: "bits[1]:0x1; bits[23]:0x0; bits[97]:0x1_7ff7_ddff_7f9f_ffef_ffff_ffff; bits[29]:0x1c62_e85c"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x4edf_bcd1_51c9_54cc_115c_1157; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x77_9c94; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1ffd_dfee"
//     args: "bits[1]:0x0; bits[23]:0x2a_aaaa; bits[97]:0x1_ffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x800"
//     args: "bits[1]:0x0; bits[23]:0x9483; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x19fd_fff7"
//     args: "bits[1]:0x1; bits[23]:0x6a_aaea; bits[97]:0x1_3a8a_2af7_6dbf_79f3_fffe_ffef; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0x57b3_efad_fbf4_fdaa_adff_fff4; bits[29]:0x1fbd_1aea"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0x1_368f_9fe9_f5fe_6bb6_df7d_fcb7; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x22_abce; bits[97]:0x1a8b_4c02_8a6a_9612_0848_8a0c; bits[29]:0x230_9c20"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0x1_0044_6008_0020_023a_0110_4406; bits[29]:0x40"
//     args: "bits[1]:0x1; bits[23]:0x4f_ebd7; bits[97]:0x3e07_6260_1010_3611_0944_4200; bits[29]:0x13fb_dcd4"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x1_edaf_b380_e35a_c2d8_4cd5_830d; bits[29]:0x0"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x1_5000_5002_0014_3030_0080_0080; bits[29]:0xd1b_3fbe"
//     args: "bits[1]:0x0; bits[23]:0x1f_76af; bits[97]:0xec5_e60c_596e_e657_a0f6_3803; bits[29]:0xfdd_4ec9"
//     args: "bits[1]:0x1; bits[23]:0x40_0c00; bits[97]:0x1_33bd_69b4_bd1d_519f_8e90_7687; bits[29]:0x80"
//     args: "bits[1]:0x0; bits[23]:0x16_1261; bits[97]:0x689f_7e9c_08a5_e944_a892_ed80; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x8; bits[97]:0x20_0000_0000_0000; bits[29]:0x7ed_5cf2"
//     args: "bits[1]:0x0; bits[23]:0x3f_ffff; bits[97]:0x0; bits[29]:0x1555_5555"
//     args: "bits[1]:0x0; bits[23]:0x7_f00d; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x40_0000"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0x1_fa73_9f3d_0e16_4599_72bd_94f4; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x7f_ffff; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1e78_fc50"
//     args: "bits[1]:0x1; bits[23]:0x32_aa38; bits[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[29]:0xfa5_5327"
//     args: "bits[1]:0x0; bits[23]:0x15_5555; bits[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[29]:0x2aa_ba0b"
//     args: "bits[1]:0x1; bits[23]:0x2a_aaaa; bits[97]:0x5695_05c5_a025_27d6_4906_5041; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x7b_fdff; bits[97]:0x1_ef77_fd55_5555_5555_4555_5515; bits[29]:0x1a6c_1f6b"
//     args: "bits[1]:0x1; bits[23]:0x20; bits[97]:0x1_ffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0xaaa_aaab"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x1_da5a_1839_65e4_6420_c8d9_aee6; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x2a_aaaa; bits[97]:0xaeaa_ab8a_f249_10a7_0b9c_0b67; bits[29]:0x1555_5555"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0x1_ffff_feaa_aaaa_aaaa_aa2a_aaaa; bits[29]:0x1f2e_223d"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x4000_0000_0000_0000_0000_0000; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x2e_29e4; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x437_8d25"
//     args: "bits[1]:0x1; bits[23]:0x40_2100; bits[97]:0x1_0084_02ae_aaaa_abaa_aaaa_aaaa; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x3c_6a02; bits[97]:0xf1a0_0810_0000_0584_0000_0000; bits[29]:0x1a6b_cc93"
//     args: "bits[1]:0x1; bits[23]:0x77_dde3; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x51_e01c; bits[97]:0x1_6301_64a6_8e3e_8e9e_a0ba_fcab; bits[29]:0x0"
//     args: "bits[1]:0x1; bits[23]:0x6a_aaaa; bits[97]:0x1_2a23_9a04_9018_930e_3ac4_85ce; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x20_eb88; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x800"
//     args: "bits[1]:0x1; bits[23]:0x32_1407; bits[97]:0x20_0000_0000_0000_0000_0000; bits[29]:0x6e5_2b8b"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x72b8_f093_6fce_f7a8_ba1f_b6de; bits[29]:0x12bb_0b94"
//     args: "bits[1]:0x0; bits[23]:0x28_ffa8; bits[97]:0x1_4366_6a04_6c91_5fc9_4b00_1aa1; bits[29]:0x17e7_9503"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[29]:0x1aef_aa4a"
//     args: "bits[1]:0x0; bits[23]:0x3b_fbff; bits[97]:0xd51d_5d1d_1c54_515d_1159_1519; bits[29]:0x0"
//     args: "bits[1]:0x1; bits[23]:0x5_8e3b; bits[97]:0x0; bits[29]:0x40"
//     args: "bits[1]:0x1; bits[23]:0x5f_fffe; bits[97]:0xbe06_af3c_8521_93e9_76b5_2174; bits[29]:0x1555_5555"
//     args: "bits[1]:0x1; bits[23]:0x3f_fff5; bits[97]:0x20_0000_0000_0000_0000_0000; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x55_5555; bits[97]:0x8000; bits[29]:0x44_e240"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x6896_2813_2044_6371_3daa_0822; bits[29]:0xfff_fdc8"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x14b6_5c10"
//     args: "bits[1]:0x0; bits[23]:0x2_5372; bits[97]:0x1000_0000; bits[29]:0xfc0_b72b"
//     args: "bits[1]:0x0; bits[23]:0x7f_ffff; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x13db_b6a9"
//     args: "bits[1]:0x0; bits[23]:0x2_0000; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x400_0000"
//     args: "bits[1]:0x1; bits[23]:0x17_dfb9; bits[97]:0x1_11d6_adec_ebd7_ec91_a8fe_3a9a; bits[29]:0x5f7_ee72"
//     args: "bits[1]:0x0; bits[23]:0x8_4002; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0x0"
//     args: "bits[1]:0x1; bits[23]:0x7f_ffff; bits[97]:0x0; bits[29]:0xb8b_ae2e"
//     args: "bits[1]:0x0; bits[23]:0x5b_f645; bits[97]:0x1_6e19_147e_a07b_ad96_1f48_e1f2; bits[29]:0x1f60_e1f2"
//     args: "bits[1]:0x1; bits[23]:0x34_afa9; bits[97]:0x1_7ff7_fff7_7fff_dedf_ffff_ff7f; bits[29]:0x2_0000"
//     args: "bits[1]:0x0; bits[23]:0x2b_d2bb; bits[97]:0x0; bits[29]:0x0"
//     args: "bits[1]:0x1; bits[23]:0x6a_6aeb; bits[97]:0x1_092b_b7bf_96fe_abdb_0aea_59eb; bits[29]:0x100d_303b"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0x4555_1411_0800_0010_0461_1000; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x1; bits[23]:0x2a_aaaa; bits[97]:0xaaaa_a840_8200_1020_9000_2011; bits[29]:0x1cea_aaaa"
//     args: "bits[1]:0x1; bits[23]:0x9_3220; bits[97]:0x24c8_83ff_fffd_ffff_fffd_ffff; bits[29]:0x8_0000"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0xfffe_feaa_a2aa_aaaa_aaca_aa2a; bits[29]:0xbfe_fbd5"
//     args: "bits[1]:0x0; bits[23]:0x55_5555; bits[97]:0x1_40a1_104a_0441_2829_3169_0020; bits[29]:0x1555_5540"
//     args: "bits[1]:0x1; bits[23]:0x55_5555; bits[97]:0x47e6_eaac_c625_a232_ce99_a990; bits[29]:0xe9f_e9d0"
//     args: "bits[1]:0x1; bits[23]:0x0; bits[97]:0x1_7f7f_fffd_ffff_ffff_fffe_ffff; bits[29]:0x1555_5555"
//     args: "bits[1]:0x0; bits[23]:0x15_5554; bits[97]:0x40_0000_0000; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x0; bits[23]:0xa_f072; bits[97]:0xff7e_dffb_f72f_f5ff_fdef_fffb; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x1; bits[23]:0x40_0814; bits[97]:0x264f_beff_171c_616b_0434_d8c3; bits[29]:0x1fff_ffff"
//     args: "bits[1]:0x0; bits[23]:0x3b_faff; bits[97]:0x4dab_6500_9144_8519_1200_5802; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0xf6ee_fdf7_ed7f_fedf_7fff_7fff; bits[29]:0x100_0000"
//     args: "bits[1]:0x1; bits[23]:0x800; bits[97]:0x800_0000_0000_0000; bits[29]:0x800"
//     args: "bits[1]:0x1; bits[23]:0x3f_ffff; bits[97]:0x1_5555_5555_5555_5555_5555_5555; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x1; bits[23]:0x55_5755; bits[97]:0x1_ffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0xaaa_aaaa"
//     args: "bits[1]:0x0; bits[23]:0x44_0800; bits[97]:0x9282_4188_60d7_85d9_02d0_2818; bits[29]:0x1fb6_74ca"
//     args: "bits[1]:0x1; bits[23]:0x2a_aaaa; bits[97]:0x1_44c0_0515_5471_35f5_f775_1675; bits[29]:0xfff_ffff"
//     args: "bits[1]:0x1; bits[23]:0x4c_8f64; bits[97]:0xffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1430_2021"
//     args: "bits[1]:0x1; bits[23]:0x44_8a10; bits[97]:0x1_2228_41bf_ffff_f7fe_f3ff_7ffe; bits[29]:0x1555_5555"
//     args: "bits[1]:0x1; bits[23]:0x2a_aaaa; bits[97]:0x1_ffff_ffff_ffff_ffff_ffff_ffff; bits[29]:0x1f7f_dbff"
//   }
// }
// END_CONFIG
type x14 = u21;
type x31 = (u49, u49, u49);
fn x27(x28: x14) -> (u49, u49, u49) {
  let x29: u49 = u49:0x1_20dc_b19d_cdc7;
  let x30: u49 = (x29) >> (x29);
  (x29, x30, x29)
}
fn main(x0: u1, x1: s23, x2: uN[97], x3: s29) -> (sN[97], s23, u21, u5) {
  let x4: uN[98] = (x2) ++ (x0);
  let x5: u21 = (x2)[x2+:u21];
  let x6: uN[97] = one_hot_sel(x0, [x2]);
  let x7: uN[97] = (x2) / (uN[97]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa);
  let x8: u21 = (x5)[:];
  let x9: s23 = (x1) << (if (x4) >= (uN[98]:72) { uN[98]:72 } else { x4 });
  let x11: sN[97] = {
    let x10: (sN[97], sN[97]) = smulp(((((x5) as uN[97])) as sN[97]), ((x7) as sN[97]));
    (x10.0) + (x10.1)
  };
  let x12: uN[412] = (((((x4) ++ (x7)) ++ (x4)) ++ (x5)) ++ (x2)) ++ (x0);
  let x13: s29 = -(x3);
  let x15: x14[1] = [x8];
  let x16: s29 = !(x13);
  let x17: uN[97] = (x7)[x8+:uN[97]];
  let x18: u5 = (((x3) as u29))[-24:10];
  let x19: u21 = (x5) * (((x11) as u21));
  let x20: s23 = (((x11) as s23)) | (x9);
  let x21: u5 = u5:0xa;
  let x22: s29 = (x3) | (((x7) as s29));
  let x23: u21 = gate!((((x17) as u5)) != (x21), x8);
  let x24: uN[97] = bit_slice_update(x7, x7, x17);
  let x25: x14[1] = array_slice(x15, x7, x14[1]:[(x15)[u32:0], ...]);
  let x26: uN[97] = !(x6);
  let x32: x31[1] = map(x15, x27);
  let x33: s45 = s45:0xfff_ffff_ffff;
  let x34: s23 = (x9) >> (if (x4) >= (uN[98]:31) { uN[98]:31 } else { x4 });
  (x11, x20, x19, x21)
}
