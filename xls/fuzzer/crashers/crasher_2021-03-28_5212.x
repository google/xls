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
// exception: "// Command \'[\'/xls/tools/opt_main\', \'sample.ir\', \'--logtostderr\']\' returned non-zero exit status 1."
// issue: "https://github.com/google/xls/issues/363"
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
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x1511_68b7_72c9, bits[43]:0x51d_55d1_5f51); bits[47]:0x3fff_ffff_ffff; bits[22]:0x1_0000"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x1555_5555_5555, bits[43]:0x537_1155_3554); bits[47]:0x20; bits[22]:0x10"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1555_5555_5555, bits[43]:0x3ff_ffff_ffff); bits[47]:0x49cc_80cc_bf27; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x40_0000_0000, bits[43]:0x0); bits[47]:0x7fff_ffff_ffff; bits[22]:0x19_5df5"
//     args: "bits[46]:0x100; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x0); bits[47]:0x5555_5555_5555; bits[22]:0x15_5555"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x7df_f6cd_561e, bits[43]:0x0); bits[47]:0x5736_6de8_1e54; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xfff_ffff_ffff, bits[43]:0x555_5555_5555); bits[47]:0x400; bits[22]:0x1d_a91c"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x2bf_e9c2_36e3); bits[47]:0x4c05_d7d5_6555; bits[22]:0x3d_655e"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1fff_fffb_ffff, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x5555_5555_5555; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x0; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x7ff_ffff_ffff); bits[47]:0x2215_8526_8c49; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x0; (bits[45]:0xfff_ffff_ffff, bits[43]:0x555_5555_5555); bits[47]:0x5555_5555_5555; bits[22]:0x15_57d5"
//     args: "bits[46]:0x200_0000_0000; (bits[45]:0xf4b_663e_7956, bits[43]:0x242_010a_3724); bits[47]:0x7fff_ffff_ffff; bits[22]:0x8008"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1ea4_9e7e_947c, bits[43]:0x400_0000); bits[47]:0x5555_5555_5555; bits[22]:0x1"
//     args: "bits[46]:0x373d_89c2_7ca5; (bits[45]:0x524_8aea_3de7, bits[43]:0x7bc_2904_5c87); bits[47]:0x4df2_9437_cbac; bits[22]:0x37_cbbc"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x167d_4e49_8a5b, bits[43]:0x555_5554_5155); bits[47]:0x47a6_e1f2_aac3; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x36ab_b8f1_ce20; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x628_bad5_9e20); bits[47]:0x1c07_6ce3_cc02; bits[22]:0x2c_bc95"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x1e6a_bdaa_4267, bits[43]:0x5be_4745_1b53); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x80"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1555_5555_5555, bits[43]:0x504_95b4_75cf); bits[47]:0x3fff_ffff_fffe; bits[22]:0x10"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x1555_5555_5555, bits[43]:0x292_c698_ea58); bits[47]:0x5555_5555_5555; bits[22]:0x3f_c78d"
//     args: "bits[46]:0x4000_0000; (bits[45]:0x1ef_98d7_469f, bits[43]:0x8000_0000); bits[47]:0x48d2_3a18_3832; bits[22]:0x10"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1fff_5fff_fef7, bits[43]:0x7f3_bbca_76ac); bits[47]:0x7fff_fbff_ffff; bits[22]:0x3f_dffd"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x15cd_7ff7_7b2f, bits[43]:0x1000_0000); bits[47]:0x4; bits[22]:0x15_5555"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0xd4b_64bf_7bbd, bits[43]:0x40_0000); bits[47]:0x3af6_ef3b_d6e1; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x400, bits[43]:0x3ff_ffff_ffff); bits[47]:0x1c88_ee71_81ad; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x54e_d6f2_ece8); bits[47]:0x5555_5555_5555; bits[22]:0x1f_5515"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xa8a_aaaa_aaaa, bits[43]:0x0); bits[47]:0x3fff_ffff_ffff; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x77e_ffee_9f9f); bits[47]:0x2_0000; bits[22]:0x4_0dab"
//     args: "bits[46]:0x10_0000_0000; (bits[45]:0x0, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x7292_03a2_065f; bits[22]:0x15_5555"
//     args: "bits[46]:0x0; (bits[45]:0x0, bits[43]:0x4_0000_0000); bits[47]:0x0; bits[22]:0x0"
//     args: "bits[46]:0x1832_552d_2bc8; (bits[45]:0x0, bits[43]:0x7ff_ffff_ffff); bits[47]:0x3fff_ffff_ffff; bits[22]:0x10"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x14c4_dccb_f54c, bits[43]:0x3a3_aa2a_a6ab); bits[47]:0x60a1_0032_16b8; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x133e_5e8f_eff5, bits[43]:0x7ff_ffff_ffff); bits[47]:0x3c0f_69e1_f73d; bits[22]:0x3a_e640"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1fff_6f77_fffd, bits[43]:0x555_5555_5555); bits[47]:0x0; bits[22]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1fbf_ffff_ffff, bits[43]:0x555_5555_5555); bits[47]:0x3fff_ffff_ffff; bits[22]:0x22_9796"
//     args: "bits[46]:0x0; (bits[45]:0x0, bits[43]:0x65e_1c64_8018); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x0; (bits[45]:0x18e1_c4b1_6c22, bits[43]:0x800); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x4_0000"
//     args: "bits[46]:0x0; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x3ff_ffff_ffff); bits[47]:0x5eb0_c06a_9d21; bits[22]:0x2"
//     args: "bits[46]:0x8000_0000; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x1; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0xf55_145a_5d44, bits[43]:0x7ff_ffff_ffff); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x2f96_96bc_79dd; (bits[45]:0x1064_b0e2_741b, bits[43]:0x796_96be_79dd); bits[47]:0x0; bits[22]:0x15_5555"
//     args: "bits[46]:0x0; (bits[45]:0x105f_66e0_8ca4, bits[43]:0x174_62c0_dc33); bits[47]:0x0; bits[22]:0x20_b000"
//     args: "bits[46]:0x3e51_8782_6de1; (bits[45]:0xbd1_8780_6ca3, bits[43]:0x671_8783_61d1); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0xb_7cfc"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x883_877b_8ac1, bits[43]:0x0); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1edf_e7be_d71e, bits[43]:0x3ff_ffff_ffff); bits[47]:0x4efa_ffff_ffbe; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x80_0000_0000; (bits[45]:0x84_5888_c001, bits[43]:0x481_9042_4649); bits[47]:0xdfc_0a0a_1186; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x0; (bits[45]:0x8_0000_0000, bits[43]:0x61c_324a_066c); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x15_5555"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x166d_4d92_6011, bits[43]:0x745_5255_d514); bits[47]:0x17ca_07cb_5234; bits[22]:0x15_5555"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x3fd_3f7b_fe5f); bits[47]:0x3fff_ffff_ffff; bits[22]:0x15_5555"
//     args: "bits[46]:0x0; (bits[45]:0x900_0012_2220, bits[43]:0x20_c090_0844); bits[47]:0x17c2_f1f5_4c1e; bits[22]:0x36_aab3"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xfff_ffff_ffff, bits[43]:0x8_0000_0000); bits[47]:0x5555_5555_5555; bits[22]:0x2d_f03e"
//     args: "bits[46]:0x80; (bits[45]:0xfff_ffff_ffff, bits[43]:0x3ff_ffff_ffff); bits[47]:0x0; bits[22]:0x29_a608"
//     args: "bits[46]:0x441_3c2a_c91c; (bits[45]:0xfff_ffff_ffff, bits[43]:0x540_7c2a_c90c); bits[47]:0x8; bits[22]:0x0"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x13ca_0c27_414a, bits[43]:0x39b_1a8b_210e); bits[47]:0x3fff_ffff_ffff; bits[22]:0x0"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xfff_ffff_ffff, bits[43]:0x488_aa0b_aaee); bits[47]:0x548e_d775_3111; bits[22]:0x10"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x0, bits[43]:0x735_c117_171c); bits[47]:0x0; bits[22]:0x3e_ca1b"
//     args: "bits[46]:0x0; (bits[45]:0x191_8000_0490, bits[43]:0x0); bits[47]:0x3fff_ffff_ffff; bits[22]:0x0"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1555_5555_5555, bits[43]:0x7cf_7fff_7fff); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x0"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xe2a_29fa_68aa, bits[43]:0x69a_8aab_b7cf); bits[47]:0x1000_0000; bits[22]:0x39_babc"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xb8b_aaa2_aaaa, bits[43]:0x6b4_1161_8a21); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x22_aaa8"
//     args: "bits[46]:0x1_0000_0000; (bits[45]:0x9_0c5a_c96c, bits[43]:0x401_6300_4828); bits[47]:0x3fff_ffff_ffff; bits[22]:0x2_0010"
//     args: "bits[46]:0x0; (bits[45]:0x0, bits[43]:0x4a5_a49f_c5ce); bits[47]:0x5555_5555_5555; bits[22]:0x6_a327"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1d77_ecaf_abfd, bits[43]:0x555_5555_5555); bits[47]:0x5555_5555_5555; bits[22]:0x14_5775"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1fff_affd_fffd, bits[43]:0x100_0000); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0xa_babf"
//     args: "bits[46]:0x0; (bits[45]:0x604_04c0_8804, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x20_9060_201a; bits[22]:0x10_0400"
//     args: "bits[46]:0x400_0000_0000; (bits[45]:0x400_0000_2000, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x3fff_ffff_ffff; bits[22]:0x22_098f"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x1692_b666_4cd4, bits[43]:0x3ff_ffff_ffff); bits[47]:0x7fff_ffff_ffff; bits[22]:0x15_72dc"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1906_d7b6_fcff, bits[43]:0x4); bits[47]:0x3fff_ffff_ffff; bits[22]:0x15_5555"
//     args: "bits[46]:0x321_69e4_5c4d; (bits[45]:0x331_69f0_dc4d, bits[43]:0x3a0_cb85_0705); bits[47]:0x0; bits[22]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1fbd_0afe_77f4, bits[43]:0x400); bits[47]:0x7fff_ffff_ffff; bits[22]:0x0"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x80_0000, bits[43]:0x0); bits[47]:0x10; bits[22]:0x2419"
//     args: "bits[46]:0x0; (bits[45]:0x452_b93b_e509, bits[43]:0x178_0a02_a529); bits[47]:0x0; bits[22]:0x2d_09bd"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x1555_5555_5555, bits[43]:0x555_5d55_5555); bits[47]:0x38e7_7583_56b3; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x20, bits[43]:0x3ff_ffff_ffff); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x3f_ffb4"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1ff7_f77f_ff7f, bits[43]:0x5fd_efbf_edff); bits[47]:0x3f6f_4b9b_bdb6; bits[22]:0x1b_bdbe"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x7ff_ffff_ffbf); bits[47]:0x3fff_ffff_ffff; bits[22]:0x1e_f7bc"
//     args: "bits[46]:0x800_0000; (bits[45]:0x1c2e_fb28_915c, bits[43]:0x700_0825_8544); bits[47]:0x3fff_ffff_ffff; bits[22]:0x16_a6f6"
//     args: "bits[46]:0x200_0000; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x284_ff11_8b22); bits[47]:0x800_0000_0000; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x0; (bits[45]:0xfff_ffff_ffff, bits[43]:0x41_8002_8000); bits[47]:0x5555_5555_5555; bits[22]:0x0"
//     args: "bits[46]:0x0; (bits[45]:0x40c_88d2_1bb0, bits[43]:0x555_5555_5555); bits[47]:0x374_4cc6_817e; bits[22]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x16fb_bdce_adf5, bits[43]:0x3c1_aee5_b4c9); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x2a_a668"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x8000); bits[47]:0x5410_ef1a_0e82; bits[22]:0x19_35f5"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0xc99_29dd_5165, bits[43]:0x645_5395_17ef); bits[47]:0x5555_5555_5555; bits[22]:0x1b_0929"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1f7e_f5bf_2fff, bits[43]:0x7bf_fffe_ffef); bits[47]:0x5555_5555_5555; bits[22]:0x15_5747"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x17bf_feff_ffff, bits[43]:0x5cf_5f7f_65d3); bits[47]:0x6ff6_ffe7_abf8; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x100; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x479_732b_18ec); bits[47]:0x400_0000_0000; bits[22]:0x0"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x1555_5555_5555, bits[43]:0x2ae_aa26_6aab); bits[47]:0x400a_f5f3_7abe; bits[22]:0x22_a9ac"
//     args: "bits[46]:0x1bc8_0c67_4975; (bits[45]:0x1555_5555_5555, bits[43]:0x0); bits[47]:0x7720_1e1e_11fb; bits[22]:0x1e_15fb"
//     args: "bits[46]:0x1ff3_1ee9_20b2; (bits[45]:0x1ff3_1ee9_20b2, bits[43]:0x0); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x28_22b2"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0xfff_ffff_ffff, bits[43]:0x750_3325_842e); bits[47]:0x2e3e_e5f2_4aa0; bits[22]:0x2c_c50a"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x2aeb_3cf6_2f47; bits[22]:0x1a_00c7"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0xdfd_ffcf_ffff, bits[43]:0x4000_0000); bits[47]:0x3fff_ffff_ffff; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x0; (bits[45]:0x1555_5555_5555, bits[43]:0xe0_08f6_3e00); bits[47]:0x4000_c002_0002; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x3ef_52ac_ed20, bits[43]:0x6a_8f1c_303e); bits[47]:0x100_0000; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x0; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x0); bits[47]:0x0; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x3e31_bb19_e84b; (bits[45]:0xfff_ffff_ffff, bits[43]:0x271_b999_e06b); bits[47]:0x3fff_ffff_ffff; bits[22]:0x16_67a8"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x850_8271_b96a; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x56_8064_b363); bits[47]:0x3fff_ffff_ffff; bits[22]:0x15_5555"
//     args: "bits[46]:0x0; (bits[45]:0x112c_3655_678a, bits[43]:0x399_2014_6547); bits[47]:0x5bb8_3f3e_61e5; bits[22]:0x2c_60c7"
//     args: "bits[46]:0xfd2_2b90_dc2c; (bits[45]:0x973_5f16_26bf, bits[43]:0x3ff_ffff_ffff); bits[47]:0x2aaa_aaaa_aaaa; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x1fff_ffff_ffff, bits[43]:0x555_5395_d745); bits[47]:0x3fff_ffff_ffff; bits[22]:0x3f_7e3f"
//     args: "bits[46]:0x3fff_ffff_ffff; (bits[45]:0xc29_6050_7764, bits[43]:0x0); bits[47]:0x3fff_ffff_ffff; bits[22]:0x40"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xaaa_abaa_aaaa, bits[43]:0x4000_0000); bits[47]:0x4655_11d0_554e; bits[22]:0x10_f44e"
//     args: "bits[46]:0x1000_0000_0000; (bits[45]:0x20_0000_0000, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x3fff_ffff_ffff; bits[22]:0x36_fd2b"
//     args: "bits[46]:0x4; (bits[45]:0x80_0000_0004, bits[43]:0x241_0802_4204); bits[47]:0x444e_ff7c_5abd; bits[22]:0x4"
//     args: "bits[46]:0x0; (bits[45]:0x300_0040_0088, bits[43]:0x2); bits[47]:0x5555_5555_5555; bits[22]:0x15_5555"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x1555_5555_54d5, bits[43]:0x555_5555_5555); bits[47]:0x351_3849_add1; bits[22]:0x15_5050"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x2, bits[43]:0x3fb_6ffe_dfd7); bits[47]:0x3fff_fff9_ff7e; bits[22]:0x15_5555"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1555_5555_5555, bits[43]:0x7ed_db99_1bf5); bits[47]:0x5555_5555_5555; bits[22]:0x0"
//     args: "bits[46]:0x39a8_9c16_6d0f; (bits[45]:0x13af_2769_351c, bits[43]:0x1a8_98b6_6c8f); bits[47]:0x7fff_ffff_ffff; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1fd7_f1fb_ffff, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x0; bits[22]:0x2f_8f37"
//     args: "bits[46]:0x1_0000_0000; (bits[45]:0x8cb_a840_1800, bits[43]:0x603_1028_1113); bits[47]:0x6f0_3bbd_427c; bits[22]:0x2d_180e"
//     args: "bits[46]:0x5b6_abe9_ad70; (bits[45]:0x0, bits[43]:0x2aa_aaaa_aaaa); bits[47]:0x2d7c_01ff_0c8d; bits[22]:0x1f_ffff"
//     args: "bits[46]:0x0; (bits[45]:0x1000_0084_0000, bits[43]:0x258_0000_0103); bits[47]:0x2000_0000_0001; bits[22]:0x6480"
//     args: "bits[46]:0x0; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x14_0a02_1450); bits[47]:0x46ce_3610_6244; bits[22]:0x11_e25e"
//     args: "bits[46]:0x27c4_42d3_3fff; (bits[45]:0xbcc_c2d2_2fb7, bits[43]:0x7ff_ffff_ffff); bits[47]:0x6fb2_f446_5fd6; bits[22]:0x6_57d6"
//     args: "bits[46]:0x0; (bits[45]:0x1555_5555_5555, bits[43]:0x7ff_ffff_ffff); bits[47]:0x80_0041_0001; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0x29a_a622_2a3e, bits[43]:0x555_5555_5555); bits[47]:0x5555_5555_5555; bits[22]:0x11_d151"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x0, bits[43]:0x3ff_ffff_ffff); bits[47]:0x0; bits[22]:0xc436"
//     args: "bits[46]:0x1fff_ffff_ffff; (bits[45]:0x1555_5555_5555, bits[43]:0x7ff_ffff_ffff); bits[47]:0x40_0000; bits[22]:0x36_6fd1"
//     args: "bits[46]:0x1555_5555_5555; (bits[45]:0x1555_55dd_5555, bits[43]:0x48d_d453_ddf7); bits[47]:0x0; bits[22]:0x5_5177"
//     args: "bits[46]:0x400; (bits[45]:0x5_4000_54c0, bits[43]:0x3e42_1440); bits[47]:0x10_0800_2823; bits[22]:0x22_95b0"
//     args: "bits[46]:0x0; (bits[45]:0x799_5106_c8df, bits[43]:0x1_0200_0041); bits[47]:0x75c6_b9e0_3745; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x2bc1_7baa_ef1c; (bits[45]:0xb4c_cd60_e810, bits[43]:0x0); bits[47]:0x3fff_ffff_ffff; bits[22]:0x15_5555"
//     args: "bits[46]:0x8000; (bits[45]:0x104f_3204_0937, bits[43]:0x108_0800_9020); bits[47]:0x7fff_ffff_ffff; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x4_0000; (bits[45]:0x1555_5555_5555, bits[43]:0x0); bits[47]:0x5555_5555_5555; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x0; (bits[45]:0xaaa_aaaa_aaaa, bits[43]:0x2); bits[47]:0x12_4200_8010; bits[22]:0x2a_aaaa"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; (bits[45]:0xaa8_c288_0aa3, bits[43]:0x3c4_3dd1_1a4e); bits[47]:0x5599_4e1c_6149; bits[22]:0x3f_ffff"
//     args: "bits[46]:0x2000_0000_0000; (bits[45]:0x866_5c19_b14d, bits[43]:0x507_04a9_35c1); bits[47]:0x2821_3cd2_accb; bits[22]:0x12_a489"
//   }
// }
// END_CONFIG
fn main(x0: s46, x1: (u45, u43), x2: s47, x3: u22) -> (uN[484], s47, uN[484], uN[88], bool, uN[484], uN[308], uN[326], bool) {
  let x4: uN[88] = (((x3) ++ (x3)) ++ (x3)) ++ (x3);
  let x5: uN[308] = ((((x4) ++ (x4)) ++ (x3)) ++ (x4)) ++ (x3);
  let x6: uN[484] = ((x4) ++ (x4)) ++ (x5);
  let x7: uN[308] = (x5)[x4+:uN[308]];
  let x8: uN[484] = (x6) + (((x2) as uN[484]));
  let x9: uN[485] = one_hot(x8, bool:false);
  let x10: bool = and_reduce(x5);
  let x11: u45 = (x1).0;
  let x12: uN[1760] = (((x7) ++ (x6)) ++ (x6)) ++ (x6);
  let x13: uN[484] = rev(x6);
  let x14: u16 = (x6)[x9+:u16];
  let x15: s46 = for (i, x): (u4, s46) in u4:0..u4:7 {
    x
  }(x0);
  let x16: uN[485] = (x9)[x5+:uN[485]];
  let x17: uN[294] = (x16)[x16+:uN[294]];
  let x18: uN[484] = (((x11) as uN[484])) + (x13);
  let x20: u45 = (x1).0;
  let x21: bool = (x10)[x4+:bool];
  let x22: (uN[484], uN[485], u45, s47, s46, u16) = (x6, x9, x11, x2, x0, x14);
  let x23: bool = xor_reduce(x12);
  let x24: uN[1345] = (((((x3) ++ (x20)) ++ (x10)) ++ (x5)) ++ (x9)) ++ (x8);
  let x25: bool = or_reduce(x12);
  let x26: uN[1982] = ((((x11) ++ (x6)) ++ (x6)) ++ (x9)) ++ (x8);
  let x27: uN[326] = (((x5) ++ (x14)) ++ (x23)) ++ (x23);
  let x28: s46 = (((x7) as s46)) | (x15);
  let x29: uN[485] = for (i, x): (u4, uN[485]) in u4:0..u4:5 {
    x
  }(x9);
  let x30: bool = (((x10) as uN[88])) == (x4);
  (x18, x2, x13, x4, x25, x6, x7, x27, x10)
}
