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
// evaluated opt IR (JIT), evaluated unopt IR (JIT) =
//    ((bits[3]:0x7), bits[41]:0x100_0000_aaaa, bits[40]:0x7f_ffff_ffff, bits[41]:0x1ff_ffff_5555, bits[52]:0x5_957d_51de_8faa)
// evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =
//    ((bits[3]:0x7), bits[41]:0x100_0000_aaaa, bits[40]:0x7f_ffff_ffff, bits[41]:0xff_ffff_5555, bits[52]:0x5_957d_51de_8faa)
//
// BEGIN_CONFIG
// exception: "// Result miscompare for sample 0:"
// issue: "https://github.com/google/xls/issues/376"
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
//   codegen_args: "--pipeline_stages=6"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_957d_51de_8faa, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_df7f_5771_0916, bits[52]:0x5_5d77_5556_a98a, bits[52]:0x5_4555_5555_5555]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_957d_51de_8faa, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_df7f_5771_0916, bits[52]:0x5_5d77_5556_a98a, bits[52]:0x5_4555_5555_5555]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x5_b7ff_cbfd_50d5, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x7_b7bf_7eb7_3432]; bits[40]:0x400; (bits[3]:0x2); bits[17]:0x0"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0x8_efb8_3c83_d58c, bits[52]:0x1_0000_0000_0000, bits[52]:0x7_a7b9_a1af_dba3, bits[52]:0x6_6eca_a46e_c770, bits[52]:0x200_0000_0000, bits[52]:0x200]; bits[40]:0x0; (bits[3]:0x0); bits[17]:0x1_5555"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x7_fbf6_dff9_dbff, bits[52]:0x7_b6cf_9fda_20f6, bits[52]:0xd_e7ba_7fd9_5156, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x7_ffff_fffd_ffff]; bits[40]:0x1f_6fff_fff2; (bits[3]:0x1); bits[17]:0x1_7bf2"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xa_aaaa_8aa8_0040, bits[52]:0x4aab_9af9_f79a, bits[52]:0xa_bc88_e2a5_ff7f, bits[52]:0x5_5e7c_f223_73f8, bits[52]:0x4_0000_0000, bits[52]:0x8]; bits[40]:0x4_0000_0000; (bits[3]:0x1); bits[17]:0x0"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x2_1755_5544_e515, bits[52]:0x5_55d5_5554_0004, bits[52]:0x0, bits[52]:0x6_51d2_456e_b79f, bits[52]:0xe_5cd5_4d67_1dde, bits[52]:0x5_5555_5554_0000]; bits[40]:0x15_7174_977d; (bits[3]:0x2); bits[17]:0x1_5555"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xa_aba8_aaa8_0040, bits[52]:0x2_aaaf_aba1_2244, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xa_aaaa_aaaa_ae1a]; bits[40]:0x0; (bits[3]:0x7); bits[17]:0x4230"
//     args: "bits[34]:0x0; [bits[52]:0x0, bits[52]:0x8_1002_1020_0100, bits[52]:0x30_0300_0020, bits[52]:0x8_8008_6cb7_d32f, bits[52]:0x4_d446_b61d_e197, bits[52]:0x20]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x0); bits[17]:0x1_ffff"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x0, bits[52]:0x400_0000, bits[52]:0x5_5555_5555_5555, bits[52]:0x5_5555_5555_5555, bits[52]:0x5_04d6_5476_c020, bits[52]:0x200_0000]; bits[40]:0x10; (bits[3]:0x4); bits[17]:0x1_5555"
//     args: "bits[34]:0x0; [bits[52]:0x0, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x408_0008, bits[52]:0x5_5555_5555_5555, bits[52]:0x8_4200_0001_5555, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0xc6_7a47_e14d; (bits[3]:0x3); bits[17]:0x1_5555"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x3_f7e6_ec35_d409, bits[52]:0x2_a3df_7f78_9200, bits[52]:0x0, bits[52]:0x7_ffff_ffbe_ef15, bits[52]:0xe_bfff_3ebd_f6ef, bits[52]:0xe_3e7c_7da6_8b0e]; bits[40]:0x40_0000_0000; (bits[3]:0x2); bits[17]:0x540c"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x5_5555_5555_5555, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x0, bits[52]:0xb_57e9_7f7d_fefd, bits[52]:0x0]; bits[40]:0x55_5555_5555; (bits[3]:0x7); bits[17]:0xffff"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_ff7f_f7cc_0104, bits[52]:0xb_ffff_ff3d_aaaa, bits[52]:0x5_4a33_82f7_2fa9, bits[52]:0xf_6ff7_eff7_1e9e, bits[52]:0x9_f3c3_ff78_509c, bits[52]:0x6_ff6e_efd5_d3d9]; bits[40]:0x73_d9dd_b374; (bits[3]:0x7); bits[17]:0x0"
//     args: "bits[34]:0x2_e40f_bc69; [bits[52]:0xb_10ae_95a7_caea, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x3_b016_f1ac_1002, bits[52]:0xd_5736_69d2_b501, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x63cc_2884; [bits[52]:0x1_7f3e_b442_ab2a, bits[52]:0x5_5555_5555_5555, bits[52]:0x1_c731_24d0_aa63, bits[52]:0x8, bits[52]:0x7_9715_dbc4_40b5, bits[52]:0xd_8fa5_316c_5982]; bits[40]:0x3a_738b_967f; (bits[3]:0x7); bits[17]:0x1_a282"
//     args: "bits[34]:0x200; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x4_4401_1832_303f, bits[52]:0x4_688b_6a34_abb5, bits[52]:0x8420_4805_5555, bits[52]:0x2_5188_4c02_faf2, bits[52]:0x8_1000_aa11_7eff]; bits[40]:0x2_1840_92bb; (bits[3]:0x2); bits[17]:0x1_5555"
//     args: "bits[34]:0x0; [bits[52]:0x201_0000_0008, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xd47_6890_69fa]; bits[40]:0x40_0000_0000; (bits[3]:0x6); bits[17]:0x1_5555"
//     args: "bits[34]:0x200; [bits[52]:0x100, bits[52]:0x8_10a0_3001_e6a7, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x9_b385_800c_68a5, bits[52]:0x8_0000_0000_0000, bits[52]:0x1_c010_4c89_f5df]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x2); bits[17]:0x0"
//     args: "bits[34]:0xe461_5ca1; [bits[52]:0x3_d18f_72a1_4355, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x1_95d8_20c7_7b6c, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x3_91a4_f08e_8bea]; bits[40]:0x55_5555_5555; (bits[3]:0x0); bits[17]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x5_fefb_77fd_a8e3, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x6_d7e0_bfbd_5f55, bits[52]:0x5_5555_5555_5555, bits[52]:0x4_553b_fb85_293f, bits[52]:0xa_aaaa_aaaa_aaaa]; bits[40]:0xff_feff_7be0; (bits[3]:0x3); bits[17]:0xffff"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_fdcb_bffc_0052, bits[52]:0x4_c8bb_53ca_f36f, bits[52]:0x7_dfff_a45d_6bfe]; bits[40]:0x7d_ffb9_9dc4; (bits[3]:0x1); bits[17]:0x9b54"
//     args: "bits[34]:0x2000_0000; [bits[52]:0x2_d341_0e5c_9002, bits[52]:0x5_2cdc_3968_735c, bits[52]:0x5_5555_5555_5555, bits[52]:0x2000_0000, bits[52]:0x8000_0001_0000, bits[52]:0xa000_0003_ffff]; bits[40]:0x1c_6d08_197f; (bits[3]:0x6); bits[17]:0x1_5555"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555, bits[52]:0x9_e2a8_b303_5df7, bits[52]:0x800_0000, bits[52]:0xc0ae_a26e_ed2b, bits[52]:0x4_3b9b_a8b1_69e9]; bits[40]:0xae_9aaf_2868; (bits[3]:0x0); bits[17]:0xaaaa"
//     args: "bits[34]:0x0; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x8700_0180_59d7, bits[52]:0xa_2cc2_21a1_da8f, bits[52]:0x8040_0445_bde0, bits[52]:0x0, bits[52]:0x0]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x1); bits[17]:0x100"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x5_5755_5717_f77d, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_27e9_ebc8_d543, bits[52]:0x0, bits[52]:0x5_5555_55c5_fdf7, bits[52]:0x3_5c0c_dc77_9749]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x3); bits[17]:0x1_b31f"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaa8_0000, bits[52]:0xb_8426_190b_166f, bits[52]:0x8_82bc_2ea9_0800, bits[52]:0x8_baba_eaa9_5455, bits[52]:0x400]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x4); bits[17]:0xffff"
//     args: "bits[34]:0x8_0000; [bits[52]:0x6_8405_4002_d16f, bits[52]:0x100_0000, bits[52]:0x20_0003_e5d4, bits[52]:0x1_4803_1504_cc43, bits[52]:0x200_0000, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0x55_5555_5555; (bits[3]:0x0); bits[17]:0x824"
//     args: "bits[34]:0x1_5724_a78f; [bits[52]:0x5_4693_4f2f_a8ea, bits[52]:0x5_6dff_dbd6_0c0f, bits[52]:0x4_7c9a_bf09_950d, bits[52]:0x5_9dd2_9e29_dddf, bits[52]:0x4_5d92_ce3e_8acb, bits[52]:0x5_5555_5555_5555]; bits[40]:0x0; (bits[3]:0x0); bits[17]:0x0"
//     args: "bits[34]:0x200; [bits[52]:0x397_4855_df7f, bits[52]:0x4_0000_0000_0000, bits[52]:0x4_4000_1c21_e7f3, bits[52]:0x0, bits[52]:0x8_c000_8c09_d1df, bits[52]:0x7_ffff_ffff_ffff]; bits[40]:0x4000_8000; (bits[3]:0x7); bits[17]:0xffff"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x5_1dfc_cc13_c77f, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x5_5f55_4143_93fc, bits[52]:0x1000_0000, bits[52]:0x5_5555_5544_2080, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x0, bits[52]:0x7_7c61_fed9_1004, bits[52]:0xf_7bbc_7eb3_c0b3, bits[52]:0x200_0000_0000, bits[52]:0x4000]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x7); bits[17]:0x1_aaaa"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_3bbf_afd5_75f4, bits[52]:0x7_d5db_fe75_fbf7, bits[52]:0xf_ffef_ffd5_fbff, bits[52]:0x5_5555_5555_5555, bits[52]:0xe_f2bf_3aff_daff]; bits[40]:0xff_ffff_ffff; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xa_aaaa_aaa3_ffff, bits[52]:0x5_5555_5555_5555, bits[52]:0xc_e93e_aa8c_b060, bits[52]:0x5_5555_5555_5555, bits[52]:0xa_aaae_c3a9_f7c9, bits[52]:0xa_ab92_3abb_2000]; bits[40]:0x0; (bits[3]:0x2); bits[17]:0xaaaa"
//     args: "bits[34]:0x0; [bits[52]:0x800_0000, bits[52]:0x6200_6543_f75b, bits[52]:0x8_0000_0100_6000, bits[52]:0xc_1021_2600_f17d, bits[52]:0x6008_0201_0280, bits[52]:0xd54_ca40_0849]; bits[40]:0x8c_0422_1082; (bits[3]:0x1); bits[17]:0xaaaa"
//     args: "bits[34]:0x2_0000_0000; [bits[52]:0x8_410c_0080_84c4, bits[52]:0x0, bits[52]:0x8_1104_0202_aaaa, bits[52]:0x5_5555_5555_5555, bits[52]:0x0, bits[52]:0x4_20e7_9458_24e6]; bits[40]:0x0; (bits[3]:0x3); bits[17]:0x1_5555"
//     args: "bits[34]:0x1f58_7b7e; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x2_0000_0000, bits[52]:0x1_7971_cdfa_8aaa, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x6_80df_bab0_fd70]; bits[40]:0xff_ffff_ffff; (bits[3]:0x6); bits[17]:0xffff"
//     args: "bits[34]:0x0; [bits[52]:0x2_aaaa, bits[52]:0x1064_d112_cc6f, bits[52]:0xb_0810_aad0_5159, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x2000_0000, bits[52]:0x1d69_0c42_df26]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x7); bits[17]:0x1_cbae"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_fbff_fffd_0842, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xe_ffff_fbfd_5f55, bits[52]:0xe_99ed_ef78_bbfd, bits[52]:0x3_f5ed_9591_9121, bits[52]:0x5_2fff_efe5_5515]; bits[40]:0x0; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x4000_0000; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x1_0000_0401_5555, bits[52]:0x1_08c1_0268_a2ab, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x1_2000_0100_8aaa, bits[52]:0x1_0114_0000_0200]; bits[40]:0x2f_f9a5_f9d1; (bits[3]:0x2); bits[17]:0xe804"
//     args: "bits[34]:0x80_0000; [bits[52]:0x8_8a0d_5032_de8a, bits[52]:0x8000_0000, bits[52]:0x0, bits[52]:0x1, bits[52]:0x8202_080a_e28a, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0x81_20d6_ccbb; (bits[3]:0x4); bits[17]:0x1_ffff"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0xf_dce7_ffef_feff, bits[52]:0xf_ffe7_fbfd_5555, bits[52]:0x400_0000, bits[52]:0x2_190a_1c20_21ff, bits[52]:0x0, bits[52]:0xe_c8f8_6e1f_6d7c]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x2); bits[17]:0x5fd"
//     args: "bits[34]:0x0; [bits[52]:0x1010_0001_5555, bits[52]:0x801_0000_ea77, bits[52]:0x29a4_08a3_1dae, bits[52]:0x0, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_ffff_ffff_ffff]; bits[40]:0x42_0280_302a; (bits[3]:0x7); bits[17]:0x1_ffff"
//     args: "bits[34]:0x8000_0000; [bits[52]:0xb_0089_7420_1062, bits[52]:0x3_1df9_6ac7_330e, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x2_c006_0100_0208, bits[52]:0x5_5555_5555_5555, bits[52]:0x6_0800_0002_b8aa]; bits[40]:0x92_c05f_905e; (bits[3]:0x4); bits[17]:0x4"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_ffff_fffe_6e15, bits[52]:0xf_bef5_ccae_b8a3, bits[52]:0xf_ffff_fffc_8000, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff]; bits[40]:0x0; (bits[3]:0x2); bits[17]:0x1_1439"
//     args: "bits[34]:0x2000; [bits[52]:0x1_4cde_7a6f_cfe3, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x1_0000_8203_0000, bits[52]:0x4_6116_9000_8000, bits[52]:0x912_c006_2041, bits[52]:0x5_3079_d401_6e08]; bits[40]:0x409_8531; (bits[3]:0x2); bits[17]:0x10"
//     args: "bits[34]:0x20; [bits[52]:0x0, bits[52]:0xf_23c0_6a82_7174, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x0, bits[52]:0x40_9081_fdff, bits[52]:0x0]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x0); bits[17]:0xffff"
//     args: "bits[34]:0x40; [bits[52]:0x0, bits[52]:0x11_c181_61c6, bits[52]:0x3_2a06_25b2_0ff2, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x1_0343_6aa2_36bb, bits[52]:0x4_49b0_07bf_7e2a]; bits[40]:0xff_ffff_ffff; (bits[3]:0x0); bits[17]:0x1_5555"
//     args: "bits[34]:0x2_eebe_960d; [bits[52]:0xb_bafa_5834_0000, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x4_c5b5_43dc_9cde, bits[52]:0x9_a9fa_5940_2bc1, bits[52]:0x0, bits[52]:0xa_aaaa_aaaa_aaaa]; bits[40]:0xd6_bd40_933a; (bits[3]:0x6); bits[17]:0x1_157f"
//     args: "bits[34]:0x1_d64c_8620; [bits[52]:0x6_220e_d7d2_3a8b, bits[52]:0x9_beb2_bccd_8cac, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_5912_18e1_482b, bits[52]:0xa_aaaa_aaaa_aaaa]; bits[40]:0x40; (bits[3]:0x5); bits[17]:0x0"
//     args: "bits[34]:0x2_f73e_df14; [bits[52]:0x0, bits[52]:0xa_7c7a_af76_5807, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x9_ed79_9085_fed1, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0x0; (bits[3]:0x2); bits[17]:0x8000"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x3_fd9f_76fe_70f7, bits[52]:0xe_dff5_f66d_aebd, bits[52]:0x7_5ce5_7ff9_9bee, bits[52]:0x80_0000, bits[52]:0xf_ffb7_dbee_eb8e, bits[52]:0xb_dfff_fbef_df7d]; bits[40]:0xff_ffff_ffff; (bits[3]:0x5); bits[17]:0xffff"
//     args: "bits[34]:0x2_693a_a228; [bits[52]:0xc_ec0f_f497_d759, bits[52]:0x0, bits[52]:0x0, bits[52]:0xf_f0fb_08aa_0442, bits[52]:0xebba_2509_f1df, bits[52]:0x5_5555_5555_5555]; bits[40]:0x1c_56f4_4fdf; (bits[3]:0x4); bits[17]:0x0"
//     args: "bits[34]:0x2; [bits[52]:0x8284_0d48_2191, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x5_0418_012d_7541, bits[52]:0x8_1d30_a88a_8b24, bits[52]:0x1, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x2); bits[17]:0xffff"
//     args: "bits[34]:0x2; [bits[52]:0xa_0516_4abf_9379, bits[52]:0xb_ffff, bits[52]:0x5_0000_8018_0008, bits[52]:0x5_4918_468a_242d, bits[52]:0x4000, bits[52]:0x4019_c4ad_ee24]; bits[40]:0x4080; (bits[3]:0x5); bits[17]:0xaaaa"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0x8, bits[52]:0x2, bits[52]:0xa_ae6a_3893_ff5f, bits[52]:0xe_89a2_3aa9_ffdb, bits[52]:0x4000_0000_0000, bits[52]:0x100]; bits[40]:0x20; (bits[3]:0x1); bits[17]:0xea7b"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_bcff_7fd8_5a4c, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_fffc_7aff_3bf7, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_dfcf_fddd_ffcf]; bits[40]:0x2c_ebda_bdb4; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x7104_a580; [bits[52]:0xe_bd5f_9b0d_c8a8, bits[52]:0x5_c618_960b_fbe6, bits[52]:0x2_8412_9385_b054, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xc_da6c_6a1d_5e17, bits[52]:0x9_c412_c602_fcbd]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x5); bits[17]:0x1_5555"
//     args: "bits[34]:0xf655_60e0; [bits[52]:0xd_934a_c37b_67ac, bits[52]:0x3_9955_83c1_ffcf, bits[52]:0x3_d055_f580_0002, bits[52]:0x4_4ea1_8296_b1c3, bits[52]:0x7_dd55_8380_2200, bits[52]:0x1_d955_8185_fbef]; bits[40]:0x0; (bits[3]:0x2); bits[17]:0xffff"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_ffbf_f4bd_1fff, bits[52]:0x1_b7fc_fdf9_2eea, bits[52]:0xf_fdff_fffc_1000, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_ade7_bf32_0052, bits[52]:0xd_06b1_22d0_8847]; bits[40]:0xff_ffff_ffd0; (bits[3]:0x5); bits[17]:0x1_ffff"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x5_f5ef_775d_ff7f, bits[52]:0x200, bits[52]:0x5_5555_5555_5555, bits[52]:0xf_cb9c_edd6_a9ac, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_f7ff_dbff_fff6]; bits[40]:0x20_0000; (bits[3]:0x7); bits[17]:0xcec0"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x6_d48a_7717_b979, bits[52]:0x7_ffff_fffe_0000, bits[52]:0xf_8f53_fdff_51b5, bits[52]:0x5_5555_5555_5555, bits[52]:0xc_ff5f_f8e5_2183, bits[52]:0x7_ffdf_fa54_e004]; bits[40]:0x55_5555_5555; (bits[3]:0x3); bits[17]:0x1_ffff"
//     args: "bits[34]:0x2_660b_9766; [bits[52]:0x5_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x3_59f9_5963_fa31, bits[52]:0x5_5555_5555_5555, bits[52]:0x8_d9e6_1dde_45dd, bits[52]:0xd_9bcb_1c1b_7504]; bits[40]:0xff_ffff_ffff; (bits[3]:0x6); bits[17]:0x5efa"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x0, bits[52]:0x5_fc9c_2bda_946d, bits[52]:0x6_7bef_fffd_fedb, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_fbfe_47df_6eff, bits[52]:0xf_7fff_78fd_f7ef]; bits[40]:0xfd_bfff_ffd5; (bits[3]:0x7); bits[17]:0x1_5555"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x4_c9e4_5b92_63c2, bits[52]:0x5_5555_5554_0040, bits[52]:0x1_fd85_d4dc_7d25, bits[52]:0x2_0000, bits[52]:0x0, bits[52]:0x1_1354_303b_4f2b]; bits[40]:0x0; (bits[3]:0x5); bits[17]:0x0"
//     args: "bits[34]:0x100_0000; [bits[52]:0xc02_0d02_0202, bits[52]:0xf_6ffa_30f5_cdc8, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x6_8046_0240_970b, bits[52]:0x3_91c2_c3ac_63c2, bits[52]:0x7_04a0_0038_0140]; bits[40]:0xc009_00a0; (bits[3]:0x0); bits[17]:0x1_5555"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x0, bits[52]:0x1_bd56_d746_5c40, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x49d9_5d9c_eaaa, bits[52]:0x5_7955_d5d5_aaaa]; bits[40]:0xff_ffff_ffff; (bits[3]:0x6); bits[17]:0x1_ff44"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x0, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x800, bits[52]:0x1000_0000, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xd_5155_7544_00c0]; bits[40]:0x55_5555_5555; (bits[3]:0x5); bits[17]:0x1_0b1c"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xa_15e0_006e_f9f2, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x9_edf7_1f7e_02e0, bits[52]:0x200_0000_0000, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff]; bits[40]:0xff_ffff_ffff; (bits[3]:0x3); bits[17]:0x200"
//     args: "bits[34]:0x91d2_895a; [bits[52]:0x40_0000, bits[52]:0x0, bits[52]:0x2_5219_0be2_5fc0, bits[52]:0x6_57f6_87dd_0514, bits[52]:0xa_a74e_9468_7401, bits[52]:0x5_5555_5555_5555]; bits[40]:0xa8_e7a3_c692; (bits[3]:0x2); bits[17]:0x200"
//     args: "bits[34]:0x1_9d2b_3448; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_d554_2645_143f, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xe_00df_af3f_2df1, bits[52]:0xa_aaaa_aaaa_aaaa]; bits[40]:0xff_ffff_ffff; (bits[3]:0x5); bits[17]:0xffff"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x8_0000_0000, bits[52]:0x400_0000_0000, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_ff6d_df74_b4e8, bits[52]:0x0, bits[52]:0x7_feff_ffec_2f86]; bits[40]:0x7f_fdff_7ffd; (bits[3]:0x2); bits[17]:0x91c5"
//     args: "bits[34]:0x76d0_ad10; [bits[52]:0xe_2edc_023a_b80c, bits[52]:0x0, bits[52]:0x9_f347_a64b_56d1, bits[52]:0x5_d424_8654_a8e1, bits[52]:0x8_3dd6_8d32_9d20, bits[52]:0x9_f2ac_d86f_bfa6]; bits[40]:0xff_ffff_ffff; (bits[3]:0x0); bits[17]:0xcc89"
//     args: "bits[34]:0x0; [bits[52]:0xa350_8901_5d11, bits[52]:0x5_5555_5555_5555, bits[52]:0xc_3f78_b81b_fdfa, bits[52]:0x5_b6f1_a9d8_40bb, bits[52]:0x0, bits[52]:0x0]; bits[40]:0x8; (bits[3]:0x2); bits[17]:0x1_ffff"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x7_ffff_fffc_0200, bits[52]:0x7_933e_d336_a60f, bits[52]:0xe_ffb9_7bdc_96ea, bits[52]:0x7_fefe_7fff_555d, bits[52]:0x4_f7ee_9fdf_e756, bits[52]:0x7_f5fb_fefe_aaaa]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x3); bits[17]:0x2000"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x8000, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0x0; (bits[3]:0x5); bits[17]:0xaaaa"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x5_5d55_5574_0088, bits[52]:0x2000, bits[52]:0x9_6798_6938_4445, bits[52]:0xc_2d12_57fc_0152, bits[52]:0x0, bits[52]:0x7_6945_faf0_52cf]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x0); bits[17]:0x1_ffff"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xa_d5db_fdd0_53f4, bits[52]:0x8_6ccf_fd66_0609, bits[52]:0x1_8f3e_3e23_1c9b, bits[52]:0xf_f7f1_ff3e_9f82, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555]; bits[40]:0x59_769e_2198; (bits[3]:0x3); bits[17]:0xffff"
//     args: "bits[34]:0x8_0000; [bits[52]:0x6_2642_657a_4259, bits[52]:0x4000_0000_0000, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x2a0_0048_75d5, bits[52]:0x0, bits[52]:0x5_5555_5555_5555]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x3); bits[17]:0xafa3"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_7999_e2ed_843a, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x2000_0000, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x3_dbff_e5fd_7565]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x1); bits[17]:0xffff"
//     args: "bits[34]:0xe744_b1cc; [bits[52]:0x5_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xb_9d12_c739_5145, bits[52]:0x3_9d1a_c733_ff7f, bits[52]:0x4_0000, bits[52]:0x3_8d12_5731_4004]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x7); bits[17]:0x1_ffff"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x5_5155_45d5_ffff, bits[52]:0xd_7d05_f480_0d40, bits[52]:0x5_c751_1378_1426, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_7c30_3db5_4148, bits[52]:0xa_aaaa_aaaa_aaaa]; bits[40]:0x40_55d5_8f5e; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x4; [bits[52]:0x5027_1673_d647, bits[52]:0x2_4112_00b6_aaca, bits[52]:0x2_883a_6a04_e808, bits[52]:0x110_0010, bits[52]:0x8_a481_289d_d2bc, bits[52]:0x8]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x2); bits[17]:0x4"
//     args: "bits[34]:0x28bd_ce50; [bits[52]:0x0, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xd_17ff_e963_ba97, bits[52]:0x4_0000_0000, bits[52]:0xa2f7_3942_aaaa, bits[52]:0x2_e2f5_3b46_38aa]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x3); bits[17]:0x0"
//     args: "bits[34]:0x1000_0000; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x5408_0001_5145, bits[52]:0x4_5908_5708_2040, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x4840_4000_564d, bits[52]:0xc_6820_0011_7fce]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x7); bits[17]:0x1_5555"
//     args: "bits[34]:0x1_aff0_d72c; [bits[52]:0x6_9dc3_5cb9_6bb9, bits[52]:0x4_bfc7_5cb2_8aae, bits[52]:0xe_f3d3_b83a_c1d1, bits[52]:0xd_ebeb_50d7_ef51, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x6_3fc3_5eb1_d1d5]; bits[40]:0x0; (bits[3]:0x4); bits[17]:0x1_5555"
//     args: "bits[34]:0x3_6690_68bd; [bits[52]:0xd_e161_f207_a33c, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_d3e6_6095_4751, bits[52]:0x0, bits[52]:0x9_2b4d_f6be_f20e]; bits[40]:0x55_5555_5555; (bits[3]:0x5); bits[17]:0x1_ffff"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xc_aa3c_fe0a_e6c9, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x2_a122_aaa8_1032, bits[52]:0xb_6b28_a9a9_41bf, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0x53_98f4_4c4d; (bits[3]:0x0); bits[17]:0x400"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x5_c864_3070_dc08, bits[52]:0x1_0000_0000, bits[52]:0x7_5615_51d3_7bff, bits[52]:0x3_1ce3_ad1f_c5df, bits[52]:0x5_5555_5555_5555, bits[52]:0x1_b095_5555_d564]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_f67f_fcf9_f69f, bits[52]:0xf_ff7f_fffc_9f65, bits[52]:0x800, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_dfff_6ffb_fefe, bits[52]:0xa_a2a1_97b8_6261]; bits[40]:0x9b_7cbf_7769; (bits[3]:0x0); bits[17]:0x1_d5fb"
//     args: "bits[34]:0x0; [bits[52]:0x0, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x6_0000_3003_57d5, bits[52]:0x0, bits[52]:0x1_4c45_4191_fdbc]; bits[40]:0x8000_0000; (bits[3]:0x5); bits[17]:0x1_5555"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x3_5e3b_9657_e410, bits[52]:0x0, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_a881_ce4b_24f1, bits[52]:0xa_2a0a_3902_ea4d]; bits[40]:0xff_ffff_ffff; (bits[3]:0x0); bits[17]:0x1_5555"
//     args: "bits[34]:0x5925_66ea; [bits[52]:0x0, bits[52]:0x2_e89c_9bea_4fe9, bits[52]:0x1_4495_dba8_0380, bits[52]:0x5_5555_5555_5555, bits[52]:0x6_2294_9b2a_0500, bits[52]:0x0]; bits[40]:0x16_cc58_aa89; (bits[3]:0x3); bits[17]:0xffff"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x5_ff77_dbaf_bd6f, bits[52]:0x7_ffff_fffd_5555, bits[52]:0x5_5555_5555_5555, bits[52]:0x0, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_f5f5_bad7_dead]; bits[40]:0x55_5555_5555; (bits[3]:0x5); bits[17]:0xe25"
//     args: "bits[34]:0x0; [bits[52]:0xf97e_2d47_3939, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x4_d0ad_0e9a_8971, bits[52]:0x5_400d_c015_2880, bits[52]:0x4_0b14_1101_9e75]; bits[40]:0xff_ffff_ffff; (bits[3]:0x1); bits[17]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_fbbf_f7ff_febc, bits[52]:0xf_decf_7fb9_3010, bits[52]:0xf_ffff_fffd_5545, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_dfff_beff_5555]; bits[40]:0xff_ffff_fffb; (bits[3]:0x2); bits[17]:0x1_0dfb"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xf_7ff7_ffbd_6ca2, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x0, bits[52]:0x3_3efb_fe69_5253]; bits[40]:0x55_5555_5555; (bits[3]:0x0); bits[17]:0xaaaa"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x0, bits[52]:0x7_ebfd_bf7d_dffd, bits[52]:0xe_4ff6_df7e_8a48, bits[52]:0x1_b7cd_e5dc_0880, bits[52]:0x6_afdf_77fe_ea8a]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x7); bits[17]:0xffff"
//     args: "bits[34]:0x81df_74e3; [bits[52]:0x7_1ffd_11ae_aaaa, bits[52]:0x2_277d_938e_f7ff, bits[52]:0x0, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xc66d_5896_c3e9]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x7); bits[17]:0xaaaa"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x5_5755_9544_685f, bits[52]:0xc_5b47_4592_ca86, bits[52]:0x5_4755_505e_2000, bits[52]:0x0, bits[52]:0xf_f855_5c9d_e367, bits[52]:0x5_5755_3516_aaaa]; bits[40]:0x0; (bits[3]:0x4); bits[17]:0x1_0404"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_fff5_ffb6_ac6a, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_dffe_feba_0a0e, bits[52]:0x5_53ba_fae9_4471, bits[52]:0xf_fd3f_7ffd_bfff]; bits[40]:0x73_373e_7afc; (bits[3]:0x5); bits[17]:0x1_5555"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x5_5555_5555_5555, bits[52]:0x6_fa7f_5bf4_8a0b, bits[52]:0xe73b_9dbe_9ccd, bits[52]:0xf_ffff_fffd_e5fb, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff]; bits[40]:0x10; (bits[3]:0x7); bits[17]:0x1_3396"
//     args: "bits[34]:0x0; [bits[52]:0x2_920a_0130_0528, bits[52]:0x2400_0002_bcbb, bits[52]:0xc_9034_0640_ed7a, bits[52]:0x3_ffff, bits[52]:0x0, bits[52]:0x209_2481_5a23]; bits[40]:0x55_5555_5555; (bits[3]:0x2); bits[17]:0x1_c8d7"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0xa_7d8d_ff40_4390, bits[52]:0x5_d745_549e_a96d, bits[52]:0x4_54b8_7f88_ecc6, bits[52]:0x5_5575_5554_0400, bits[52]:0xd_5555_5556_5736, bits[52]:0x9_f704_5beb_b442]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x0); bits[17]:0x0"
//     args: "bits[34]:0x3_5ac1_016b; [bits[52]:0x8_0000, bits[52]:0xd_ab4d_068c_2101, bits[52]:0x5_5555_5555_5555, bits[52]:0xd_f024_059a_8bc2, bits[52]:0xc_a3c4_05fe_22ba, bits[52]:0xd_638c_3d85_4f34]; bits[40]:0x7b_1463_5abd; (bits[3]:0x3); bits[17]:0x1_226b"
//     args: "bits[34]:0x0; [bits[52]:0x800_0000, bits[52]:0x2081_d0c1_4ffd, bits[52]:0x8808_0002_aea2, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x2020_8b14_1047, bits[52]:0x4_3402_0609_f539]; bits[40]:0xa_4581_8692; (bits[3]:0x7); bits[17]:0xffff"
//     args: "bits[34]:0x195a_f1a6; [bits[52]:0x647b_e698_0000, bits[52]:0x5_5555_5555_5555, bits[52]:0xd35f_be06_e8e3, bits[52]:0x4ceb_94bd_55e5, bits[52]:0x1000_0000_0000, bits[52]:0x1_6d4b_4cdd_7686]; bits[40]:0xff_ffff_ffff; (bits[3]:0x2); bits[17]:0xe1a6"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x5_5555_5555_5555, bits[52]:0x5_5555_5555_5555, bits[52]:0xb_df7f_fffe_7fdf, bits[52]:0xd_fff9_fbec_4022, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x7); bits[17]:0x1_9bff"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x7_f75d_dddf_aaaa, bits[52]:0xc_7294_79f0_87e4, bits[52]:0xf_fedb_fffc_0020, bits[52]:0xf_f3ff_36dc_004a, bits[52]:0x10_0000, bits[52]:0xf_ffff_fffd_dfff]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x0); bits[17]:0x400"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x9_fc8f_86bd_6958, bits[52]:0x1_4b98_c997_bf54, bits[52]:0x6_27b4_cd26_0868, bits[52]:0xf_fcef_dd86_e1e6, bits[52]:0x7_bfbd_ffd8_4300, bits[52]:0xf_ffff_fffc_0008]; bits[40]:0x5a_3abb_d9a3; (bits[3]:0x5); bits[17]:0xd9a3"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[52]:0x5_5103_dd93_426e, bits[52]:0x5_4577_7454_1018, bits[52]:0x9_5344_5b60_2ea7, bits[52]:0xd_4953_f181_b923]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x5); bits[17]:0x0"
//     args: "bits[34]:0x8_0000; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[52]:0x400, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x6); bits[17]:0xbce4"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x1, bits[52]:0x4_d504_1554_2c00, bits[52]:0x5_5555_5557_ffff, bits[52]:0x2_0197_5053_d675, bits[52]:0xd_df59_5557_dffa]; bits[40]:0x7f_ffff_ffff; (bits[3]:0x5); bits[17]:0xffff"
//     args: "bits[34]:0x40a2_8ad0; [bits[52]:0x5_288a_2bc7_5759, bits[52]:0x1_c78b_8329_3d00, bits[52]:0x1_cfa6_0bc6_0567, bits[52]:0x5_5555_5555_5555, bits[52]:0x7_2da2_90b1_63a2, bits[52]:0x1_46aa_6b21_55d4]; bits[40]:0x1_0000; (bits[3]:0x2); bits[17]:0x0"
//     args: "bits[34]:0x0; [bits[52]:0xc_0000_0445_0448, bits[52]:0x5_00d8_0513_0607, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[52]:0x0, bits[52]:0x8_8e40_5812_bfdf]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x2); bits[17]:0x0"
//     args: "bits[34]:0x1_5555_5555; [bits[52]:0x5_855d_67d4_0281, bits[52]:0x5_d573_5416_eb7b, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555, bits[52]:0x4_4554_5f25_3cc4, bits[52]:0x5_5574_5556_2473]; bits[40]:0xff_ffff_ffff; (bits[3]:0x3); bits[17]:0x0"
//     args: "bits[34]:0x8bc0_a63b; [bits[52]:0x8_b2c6_be8b_f8c0, bits[52]:0xf_83ad_5059_5856, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0x0, bits[52]:0xb_6e6a_9374_dea6, bits[52]:0x5_5555_5555_5555]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x2); bits[17]:0x400"
//     args: "bits[34]:0x3_ffff_ffff; [bits[52]:0x8_7940_6aa6_e30e, bits[52]:0xf_d5da_afbe_49bc, bits[52]:0x8_5aa7_b1fc_aa1d, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xf_ffff_fffc_5555, bits[52]:0x6_9ddf_b7ec_aca0]; bits[40]:0x55_5555_5555; (bits[3]:0x5); bits[17]:0x6153"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0x0, bits[52]:0x2_8a2a_d2a1_b67d, bits[52]:0x8_ba2a_eaa8_dd4b, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x3_89ab_ffbe_523b, bits[52]:0x5_5555_5555_5555]; bits[40]:0xaa_aaaa_aaaa; (bits[3]:0x5); bits[17]:0xffff"
//     args: "bits[34]:0x2_1571_5f60; [bits[52]:0x8_5745_7862_be28, bits[52]:0x0, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x10_0000, bits[52]:0x0, bits[52]:0x1_d5c7_bdab_7c15]; bits[40]:0xff_ffff_ffff; (bits[3]:0x7); bits[17]:0x1_5555"
//     args: "bits[34]:0x4000_0000; [bits[52]:0x1_dd80_01e0_0493, bits[52]:0x0, bits[52]:0x0, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555, bits[52]:0x9_ca5d_a842_8af6]; bits[40]:0x10_0000_001f; (bits[3]:0x3); bits[17]:0x1_1ce9"
//     args: "bits[34]:0x1_4917_2956; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x4_24c8_a55a_9aa2, bits[52]:0x5_244c_855a_0679, bits[52]:0x4_e51a_948e_1e68, bits[52]:0x8, bits[52]:0x5_2d5c_e709_ce7f]; bits[40]:0x72_7efd_5762; (bits[3]:0x6); bits[17]:0x1_2b1a"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x7_3fdb_1fed_7525, bits[52]:0x6_ffff_ba5a_fffd, bits[52]:0x9_315f_7b3e_64fb, bits[52]:0xb_010d_32c9_7de9, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0x0; (bits[3]:0x5); bits[17]:0xffff"
//     args: "bits[34]:0x1_0cec_1fb8; [bits[52]:0x200_0000, bits[52]:0x4_33b0_7ca0_92e5, bits[52]:0xc_63b1_7cd3_edf5, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555, bits[52]:0x2_3310_6d56_aab9]; bits[40]:0x80_0000_0000; (bits[3]:0x2); bits[17]:0x1_1bb0"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0x9_6aae_892f_e1ff, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xf_439c_e14c_2813, bits[52]:0x2_28e0_ba2a_3569, bits[52]:0x2_a2ae_0068_fb4e, bits[52]:0xe_882a_fe96_832f]; bits[40]:0xf3_bbe3_2bae; (bits[3]:0x1); bits[17]:0xffff"
//     args: "bits[34]:0x1_ffff_ffff; [bits[52]:0x8_0f97_e377_01bd, bits[52]:0x7_e62e_4a81_a503, bits[52]:0x3_ddf7_fbfe_aaaa, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x1_b71f_be3c_caab, bits[52]:0x7_ebfd_f5fe_efbe]; bits[40]:0xe8_fe08_a269; (bits[3]:0x7); bits[17]:0x0"
//     args: "bits[34]:0x3_11b9_22dc; [bits[52]:0xc_46e4_4bf2_e7b5, bits[52]:0xf_ffff_ffff_ffff, bits[52]:0xd_c7d5_8a70_8400, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x20_0000, bits[52]:0xc_58fc_8333_efdd]; bits[40]:0xff_ffff_ffff; (bits[3]:0x0); bits[17]:0x0"
//     args: "bits[34]:0x0; [bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x7_ffff_ffff_ffff, bits[52]:0x4_0049_9e1f_8c91, bits[52]:0x8_2b9c_1307_bc2b, bits[52]:0xf_ffff_ffff_ffff]; bits[40]:0xfa_7714_6157; (bits[3]:0x3); bits[17]:0x1_39f3"
//     args: "bits[34]:0x2_aaaa_aaaa; [bits[52]:0x7_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[52]:0x6_a2a0_4b02_f5f6, bits[52]:0xb_2aaa_aabc_1000, bits[52]:0x200_0000_0000, bits[52]:0x0]; bits[40]:0xb0_2efa_aabf; (bits[3]:0x4); bits[17]:0x0"
//   }
// }
// END_CONFIG
const W2_V2 = u32:2;
const W3_V6 = u32:6;
type x2 = s52;
type x10 = s17;
type x25 = x2;
type x27 = (x2, (x2,), x2, (x2, x2, x2), (x2,));
type x38 = u40;
type x45 = (x10, (x10,));
fn x20(x21: x2) -> (x2, (x2,), x2, (x2, x2, x2), (x2,)) {
  let x22: (x2,) = (x21,);
  let x23: x2 = for (i, x): (u4, x2) in u4:0..u4:6 {
    x
  }(x21);
  let x24: (x2, x2, x2) = (x23, x23, x23);
  let x26: x25[7] = [x21, x23, x21, x23, x21, x21, x23];
  (x23, x22, x23, x24, x22)
}
fn x40(x41: x10) -> (x10, (x10,)) {
  let x42: (x10,) = (x41,);
  let x43: s57 = s57:0x0;
  let x44: x10 = for (i, x): (u4, x10) in u4:0..u4:5 {
    x
  }(x41);
  (x41, x42)
}
fn main(x0: s34, x1: x2[W3_V6], x3: u40, x4: (s3,), x5: s17) -> ((s3,), u41, u40, u41, x2) {
  let x6: u40 = bit_slice_update(x3, x3, x3);
  let x7: x2 = (x1)[if ((x3) >= (u40:1)) { (u40:1) } else { (x3) }];
  let x8: s17 = -(x5);
  let x9: bool = (x8) <= (x5);
  let x11: x10[2] = [x5, x8];
  let x12: s17 = one_hot_sel(x9, [x8]);
  let x13: x2 = for (i, x): (u4, x2) in u4:0..u4:5 {
    x
  }(x7);
  let x14: x2[W3_V6] = update(x1, if ((x3) >= (u40:5)) { (u40:5) } else { (x3) }, x7);
  let x15: u41 = u41:0xff_ffff_ffff;
  let x16: u40 = x6;
  let x17: u41 = one_hot(x16, bool:true);
  let x18: u40 = (x6) & (x3);
  let x19: u40 = rev(x16);
  let x28: x27[6] = map(x1, x20);
  let x29: x2[W3_V6] = update(x14, if ((x18) >= (u40:4)) { (u40:4) } else {(x18)}, x7);
  let x30: u41 = rev(x15);
  let x31: u41 = (x15) + (((x8) as u41));
  let x32: s17 = (x5) | (x8);
  let x33: u40 = !(x3);
  let x34: (s17, x2[W3_V6], u40, s17) = (x5, x1, x18, x8);
  let x35: x2 = (x1)[if ((x30) >= (u41:2)) { (u41:2) } else { (x30) }];
  let x36: u41 = (x31) & (((x19) as u41));
  let x37: u41 = (x36) ^ (((x32) as u41));
  let x39: x38[1] = [x19];
  let x46: x45[W2_V2] = map(x11, x40);
  let x47: x2[12] = (x14) ++ (x1);
  (x4, x37, x18, x31, x13)
}
