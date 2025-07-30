// Copyright 2025 The XLS Authors
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
// # proto-message: xls.fuzzer.CrasherConfigurationProto
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[32]:0x0;
// [bits[40]:0x1_0000_0000, bits[40]:0x8_4440_1080, bits[40]:0x55_5555_5555,
// bits[40]:0x3f_3a33_e2b8, bits[40]:0x21_1054_3246, bits[40]:0x2, bits[40]:0x83_0200_00ff,
// bits[40]:0x2_8208_26c4, bits[40]:0xf8_ebb1_e710, bits[40]:0x60_0408_8102,
// bits[40]:0x22_0000_006a]; bits[19]:0x6_4ffc; bits[38]:0x3f_ffff_ffff\nevaluated opt IR (JIT),
// evaluated opt IR (interpreter), simulated, simulated_ng =\n   ([bits[40]:0x8_4440_1080,
// bits[40]:0x55_5555_5555, bits[40]:0x3f_3a33_e2b8, bits[40]:0x21_1054_3246, bits[40]:0x2,
// bits[40]:0x83_0200_00ff, bits[40]:0x2_8208_26c4], [bits[40]:0x55_5555_5555,
// bits[40]:0x3f_3a33_e2b8, bits[40]:0x21_1054_3246, bits[40]:0x2, bits[40]:0x83_0200_00ff,
// bits[40]:0x2_8208_26c4, bits[40]:0xf8_ebb1_e710, bits[40]:0x60_0408_8102,
// bits[40]:0x22_0000_006a, bits[40]:0x1_0000_0000, bits[40]:0x8_4440_1080])\nevaluated unopt IR
// (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   ([bits[40]:0x8_4440_1080,
// bits[40]:0x55_5555_5555, bits[40]:0x3f_3a33_e2b8, bits[40]:0x21_1054_3246, bits[40]:0x2,
// bits[40]:0x83_0200_00ff, bits[40]:0x2_8208_26c4], [bits[40]:0x55_5555_5555,
// bits[40]:0x3f_3a33_e2b8, bits[40]:0x21_1054_3246, bits[40]:0x2, bits[40]:0x83_0200_00ff,
// bits[40]:0x2_8208_26c4, bits[40]:0x2_8208_26c4, bits[40]:0x2_8208_26c4, bits[40]:0x2_8208_26c4,
// bits[40]:0x2_8208_26c4, bits[40]:0x2_8208_26c4])"
// issue: "https://github.com/google/xls/issues/2692"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=5"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the
//     specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full
//     throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: true
//   disable_unopt_interpreter: false
// }
// inputs {
//   function_args {
//     args: "bits[32]:0x0; [bits[40]:0x1_0000_0000, bits[40]:0x8_4440_1080,
//     bits[40]:0x55_5555_5555, bits[40]:0x3f_3a33_e2b8, bits[40]:0x21_1054_3246, bits[40]:0x2,
//     bits[40]:0x83_0200_00ff, bits[40]:0x2_8208_26c4, bits[40]:0xf8_ebb1_e710,
//     bits[40]:0x60_0408_8102, bits[40]:0x22_0000_006a]; bits[19]:0x6_4ffc;
//     bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x5ca8_3068; [bits[40]:0x40_0000_0000, bits[40]:0x78_a8a6_b828,
//     bits[40]:0x4c_ac30_4a55, bits[40]:0xec_ee18_5827, bits[40]:0xd6_a972_eb21,
//     bits[40]:0x4c_a831_6815, bits[40]:0xff_ffff_ffff, bits[40]:0xc6_75dc_c6de,
//     bits[40]:0x7c_ba38_4154, bits[40]:0x54_a030_6cc5, bits[40]:0x80_0000_0000]; bits[19]:0x3068;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xaa_2eaa_aa55, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xa9_b890_5eee, bits[40]:0xba_3ae9_2ade, bits[40]:0x1_0000, bits[40]:0x8a_20ae_9313,
//     bits[40]:0xd8_0353_76b0, bits[40]:0xc9_9aaa_2ead]; bits[19]:0x2_aaaa;
//     bits[38]:0x3a_0abe_eaaa"
//     args: "bits[32]:0x0; [bits[40]:0x55, bits[40]:0x4_1023_087b, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x7f_ffff_ffff, bits[40]:0x6d_cb8c_0e82,
//     bits[40]:0x1a_ea49_9e39, bits[40]:0x8d_5784_a470, bits[40]:0xa_ad90_e575,
//     bits[40]:0x18_0840_00c5, bits[40]:0x61_e62d_adf5]; bits[19]:0x1; bits[38]:0x209_cc9f"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xa1_8d5f_1281, bits[40]:0xa8_2aaa_2a12, bits[40]:0x0,
//     bits[40]:0xdc_510e_d8b4, bits[40]:0xde_2dc4_99dc, bits[40]:0x0, bits[40]:0xaa_bba8_2a7e,
//     bits[40]:0xb1_f8b7_2dba, bits[40]:0xea_aaaa_aa10, bits[40]:0xaa_aa2e_ae80,
//     bits[40]:0x8a_a28a_aa88]; bits[19]:0x4_e5cb; bits[38]:0x21_69aa_a23f"
//     args: "bits[32]:0x8000; [bits[40]:0x20_0099_3375, bits[40]:0x40_5080_0236,
//     bits[40]:0xec_29d5_7000, bits[40]:0xff_ffff_ffff, bits[40]:0xf4_00b6_03c4,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff, bits[40]:0x80_04ae,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x8_4088_4880, bits[40]:0xa0_80e8_0492]; bits[19]:0x2_aaaa;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xeb_fb69_bc5d, bits[40]:0xff_ffef_f520,
//     bits[40]:0x200_0000, bits[40]:0x55_5555_5555, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xf5_ff6e_756a, bits[40]:0xcf_da42_1bea,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x55_5555_5555];
//     bits[19]:0x4_c78f; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0xe65b_e01e; [bits[40]:0xee_5aee_1f18, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xe6_4b64_1402, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xea_7be7_987f,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5555, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xa7_4ba4_5af7, bits[40]:0xe6_53e0_5e5f, bits[40]:0xd7_01ea_3963];
//     bits[19]:0x3_e196; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xba_e8e8_8e04, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x40_0000, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xca_3aa3_4efd, bits[40]:0xba_aaa8_a876, bits[40]:0xaa_aaaa_ab55, bits[40]:0x0,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xff_aaca_064f]; bits[19]:0x2_aaaa; bits[38]:0x0"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_5555, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x55_55d5_553d, bits[40]:0x5d_5551_5513, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x55_1477_9db1, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x55_5555_5500,
//     bits[40]:0x16_7a7f_1308, bits[40]:0xc8_9caf_6138, bits[40]:0x7f_ffff_ffff];
//     bits[19]:0x7_414d; bits[38]:0x2b_4826_0581"
//     args: "bits[32]:0x4000_0000; [bits[40]:0x72_6581_56e6, bits[40]:0x55_5555_5555,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x40_0100_127e, bits[40]:0xc0_0003_1b14,
//     bits[40]:0x70_6178_4b38, bits[40]:0x40_2000_00c2, bits[40]:0x5b_4eb5_59d9,
//     bits[40]:0x85_6013_0304, bits[40]:0x55_5555_5555, bits[40]:0x60_8fe4_f939];
//     bits[19]:0x5_5555; bits[38]:0x2_2004_39a8"
//     args: "bits[32]:0x800; [bits[40]:0x80_1028_c07b, bits[40]:0x80, bits[40]:0x80_0000_0000,
//     bits[40]:0x4_0808_0010, bits[40]:0x84_001a_12f7, bits[40]:0x2_9018_08e1,
//     bits[40]:0xf9_33a7_82e3, bits[40]:0x36_3412_6cd9, bits[40]:0x25_844a_3737,
//     bits[40]:0xb3_5068_20d2, bits[40]:0x96_963c_d89e]; bits[19]:0x810; bits[38]:0x15_5555_5555"
//     args: "bits[32]:0x0; [bits[40]:0x30_15b7_25ea, bits[40]:0x40_51ae_80c6,
//     bits[40]:0x55_5555_5555, bits[40]:0x55_5555_5555, bits[40]:0xaa_54e8_09bf,
//     bits[40]:0xd8_88c0_767c, bits[40]:0xc7_bc1e_de38, bits[40]:0x0, bits[40]:0x55_5555_5555,
//     bits[40]:0xe020_04ea, bits[40]:0x55_5555_5555]; bits[19]:0x5_5555; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x15_5a99_409c, bits[40]:0x5_14d4_5455, bits[40]:0x10,
//     bits[40]:0x49_cc47_29fe, bits[40]:0x64_5663_94df, bits[40]:0x55_5555_55ff,
//     bits[40]:0x54_5555_5927, bits[40]:0xff_ffff_ffff, bits[40]:0x55_5751_555a,
//     bits[40]:0x5d_4515_cc82, bits[40]:0x5d_55c7_4f51]; bits[19]:0x0; bits[38]:0x4_0c04_018a"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x7f_ffff_ffff, bits[40]:0xbd_481f_d7ed,
//     bits[40]:0xae_ff3d_b8b7, bits[40]:0x20, bits[40]:0xdb_6e7f_dbff, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x55_5555_5555, bits[40]:0xff_ffff_ffff, bits[40]:0xfe_ffdf_ff40,
//     bits[40]:0xff_ffff_ff62, bits[40]:0x7f_ffff_ffff]; bits[19]:0x7_ffff;
//     bits[38]:0x36_ffff_ffd2"
//     args: "bits[32]:0x0; [bits[40]:0xd0_0422_12ff, bits[40]:0x7f_ffff_ffff, bits[40]:0x11_04eb,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x0, bits[40]:0xff_ffff_ffff, bits[40]:0xc_a000_11ae,
//     bits[40]:0x0, bits[40]:0x41_0288_826c, bits[40]:0x10_0000, bits[40]:0x55_5555_5555];
//     bits[19]:0x40; bits[38]:0x10_2000_001f"
//     args: "bits[32]:0x0; [bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x55_5555_5555, bits[40]:0x55_5555_5555, bits[40]:0xa10_04a2,
//     bits[40]:0x30_6000_8250, bits[40]:0x808_308e, bits[40]:0x0, bits[40]:0x4_6802_2055,
//     bits[40]:0x80_0048_00d4]; bits[19]:0x7_ffff; bits[38]:0x31_35f8_0730"
//     args: "bits[32]:0xc5e2_dc5e; [bits[40]:0xbf_8ca4_fd75, bits[40]:0xc6_a6dc_be52,
//     bits[40]:0xc5_22e4_5c63, bits[40]:0xcd_ead2_167f, bits[40]:0x55_5555_5555,
//     bits[40]:0xc5_e2dc_5e00, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xc4_e2dc_5e7d,
//     bits[40]:0x65_ead8_5a0b, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x80_0000]; bits[19]:0x5_5555;
//     bits[38]:0x12_528b_6d88"
//     args: "bits[32]:0x9983_9c5a; [bits[40]:0xd9_a3bc_5abb, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xb0_d34e_08fc, bits[40]:0xed_779d_f0e0, bits[40]:0x99_c39c_5a7f,
//     bits[40]:0xb9_839c_5aff, bits[40]:0x0, bits[40]:0x80_0000_0000, bits[40]:0x99_839c_5aff,
//     bits[40]:0x71_8441_de86, bits[40]:0xff_ffff_ffff]; bits[19]:0x3_dc5a;
//     bits[38]:0x26_a907_d217"
//     args: "bits[32]:0x0; [bits[40]:0x3e_eaa6_2adf, bits[40]:0x2_6123_302b, bits[40]:0x4035_7045,
//     bits[40]:0x80, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff, bits[40]:0xd8_838e_282e,
//     bits[40]:0x2_0000_0000, bits[40]:0x0, bits[40]:0x49_002a_01f4, bits[40]:0x8_0000_0000];
//     bits[19]:0x2_2800; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x80_0000, bits[40]:0x0, bits[40]:0xf_b5d8_5325,
//     bits[40]:0x79_1614_0100, bits[40]:0x5f_5565_57f5, bits[40]:0x55_5555_5555,
//     bits[40]:0xd5_5d54_54ff, bits[40]:0x4c_695d_f98d, bits[40]:0xe4_3dca_3f30,
//     bits[40]:0x9_a963_a0fc, bits[40]:0xff_ffff_ffff]; bits[19]:0x2_aaaa; bits[38]:0x2f_ae12_60a3"
//     args: "bits[32]:0x2000_0000; [bits[40]:0x0, bits[40]:0x200, bits[40]:0x84_5069_44d4,
//     bits[40]:0x10_980a_493e, bits[40]:0x80, bits[40]:0x74_3410_8050, bits[40]:0x20_0001_90ea,
//     bits[40]:0x0, bits[40]:0x21_3210_03de, bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa];
//     bits[19]:0x5_5555; bits[38]:0x0"
//     args: "bits[32]:0x0; [bits[40]:0x1188_0004, bits[40]:0xb2c5_5abd, bits[40]:0x89_0408_8c67,
//     bits[40]:0x0, bits[40]:0x8_9009_4237, bits[40]:0x7f_ffff_ffff, bits[40]:0x0, bits[40]:0x0,
//     bits[40]:0x61, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x8001_407f]; bits[19]:0x5_5555;
//     bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0x3f_df3f_febe, bits[40]:0x7f_ffff_ff02, bits[40]:0x0,
//     bits[40]:0x7f_ffff_bfa8, bits[40]:0x0, bits[40]:0x75_fdfb_ff80, bits[40]:0xbf_f5f3_1fbf,
//     bits[40]:0x99_cf91_963e, bits[40]:0x5b_dc5f_7604, bits[40]:0xfb_acc1_b1ef,
//     bits[40]:0x77_77ff_ff7f]; bits[19]:0x2_321c; bits[38]:0x0"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xdb_55be_ad58, bits[40]:0x44_c4b6_5b6b,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xff_ffff_ff00, bits[40]:0x5f_7fdd_5d23, bits[40]:0x0,
//     bits[40]:0xa5_fd8d_deb8, bits[40]:0xba_fff7_ff02, bits[40]:0xdd_ae77_ef98,
//     bits[40]:0x56_9df7_a67b, bits[40]:0xfe_fcff_ef85]; bits[19]:0x2_aaaa; bits[38]:0x1_19ea_abd2"
//     args: "bits[32]:0xebd7_6a29; [bits[40]:0x20_0000_0000, bits[40]:0xe9_d20a_3d1f,
//     bits[40]:0x40, bits[40]:0x7f_ffff_ffff, bits[40]:0xf3_d36a_0814, bits[40]:0xef_d6ce_297f,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x7f_ffff_ffff, bits[40]:0x0, bits[40]:0x61_d73b_3b63,
//     bits[40]:0x7f_ffff_ffff]; bits[19]:0x7_ffff; bits[38]:0x15_5555_5555"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0xbb_ffcb_f396, bits[40]:0x0,
//     bits[40]:0x55_5555_5555, bits[40]:0x2d_fba7_ff04, bits[40]:0x8000_0000,
//     bits[40]:0xd7_b5df_dceb, bits[40]:0x0, bits[40]:0xf5_9bfe_f2a7, bits[40]:0xae_8b76_39a9,
//     bits[40]:0xa6_7aef_0bdf]; bits[19]:0x0; bits[38]:0x3f_d77d_fbda"
//     args: "bits[32]:0x80; [bits[40]:0x4a_9893_e161, bits[40]:0x61_c60c_131b,
//     bits[40]:0x60_0400_35fe, bits[40]:0x0, bits[40]:0x40_c004_caf8, bits[40]:0x8_9800_80e4,
//     bits[40]:0x1_8003_1596, bits[40]:0x40_8887_d020, bits[40]:0x72_8501_1deb,
//     bits[40]:0x5_0c1e_def7, bits[40]:0x40_0808_8677]; bits[19]:0x1_b47a; bits[38]:0x31_2b74_0955"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x92_e45b_5780, bits[40]:0x35_a765_57dd,
//     bits[40]:0x35_4535_1514, bits[40]:0x75_59d7_497e, bits[40]:0x57_55df_55fb,
//     bits[40]:0x15_9151_5dc8, bits[40]:0x51_5555_557f, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff, bits[40]:0x5c_47d7_55c0];
//     bits[19]:0x7_ffff; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x0; [bits[40]:0x80_823c_8965, bits[40]:0x0, bits[40]:0x800_0000,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x20_0000_00ab, bits[40]:0x20_b800_405f, bits[40]:0xab,
//     bits[40]:0x10, bits[40]:0x93, bits[40]:0x28_0c25_013b, bits[40]:0xaa_aaaa_aaaa];
//     bits[19]:0x7_ffff; bits[38]:0x17_bef8_21a2"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0x48_8aaa_ab41, bits[40]:0x0, bits[40]:0xeb_ecc4_630e,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x1_0000, bits[40]:0xd0_28ec_7ac7, bits[40]:0xa8_efaa_ba39,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x40_0000, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xaa_aae1_aaff]; bits[19]:0x5_4dd9; bits[38]:0x15_5555_5555"
//     args: "bits[32]:0x2; [bits[40]:0x55_40d2_c22a, bits[40]:0x0, bits[40]:0x31_0012_22c2,
//     bits[40]:0x1, bits[40]:0x20_2000_0200, bits[40]:0x3011_2249, bits[40]:0x400_02bf,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x89_0690_602b, bits[40]:0x8_0f90_53a7,
//     bits[40]:0x81_0140_a1fa]; bits[19]:0x5_5555; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x0; [bits[40]:0x0, bits[40]:0x0, bits[40]:0x51_1001_901a,
//     bits[40]:0xa0_6000_847a, bits[40]:0x10_0008, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x25_b822_a955, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x55_5555_5555,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x6_0800_827c]; bits[19]:0x2_daa8; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xab_a9a9_aabf, bits[40]:0xaa_2eaa_ae39,
//     bits[40]:0xee_9ae8_124c, bits[40]:0x1_0000, bits[40]:0xc1_ab80_8caa, bits[40]:0xaa_aa8e_aeeb,
//     bits[40]:0xaa_aaae_2a80, bits[40]:0x28_aae8_ba45, bits[40]:0xaa_aaa8_a3aa,
//     bits[40]:0xa8_abea_eebf, bits[40]:0xea_83e8_a96f]; bits[19]:0x4_874b;
//     bits[38]:0x2a_62e8_5a9f"
//     args: "bits[32]:0x0; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0xff, bits[40]:0x0,
//     bits[40]:0x55_5555_5555, bits[40]:0x7f_ffff_ffff, bits[40]:0x8_8000_2012,
//     bits[40]:0x55_5555_5555, bits[40]:0x8_0002_008a, bits[40]:0x0, bits[40]:0x17f,
//     bits[40]:0x100_08ff]; bits[19]:0x2000; bits[38]:0x0"
//     args: "bits[32]:0x92d2_b766; [bits[40]:0x9a_d3d7_874e, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x16_7134_22e2, bits[40]:0x90_d03d_d2ef, bits[40]:0x55_5555_5555,
//     bits[40]:0x18_60f7_4a9c, bits[40]:0x50_260c_5526, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xa8_68d7_73f2, bits[40]:0x55_5555_5555, bits[40]:0x72_829f_3257];
//     bits[19]:0x6_9372; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xff_fbf7_fb20, bits[40]:0x55_5555_5555, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xff_ffff_ff3d, bits[40]:0x0, bits[40]:0x0, bits[40]:0xff_ffff_ff55,
//     bits[40]:0x3f_bc02_cf8a]; bits[19]:0x7_9d3f; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xff_2cff_ff7b, bits[40]:0xff_fff7_ff55,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0xff_05fc_7ea2, bits[40]:0xff_ffff_ff04,
//     bits[40]:0x55_5555_5555, bits[40]:0xff_ffff_ffff, bits[40]:0xeb_6d76_390c,
//     bits[40]:0xfd_95ef_fbd4, bits[40]:0x35_bdce_f355, bits[40]:0x6e_f9bc_d77e];
//     bits[19]:0x5_eff7; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x1ad0_2632; [bits[40]:0x18_91a6_a2df, bits[40]:0xdc_420e_2287,
//     bits[40]:0xfe_f20f_5921, bits[40]:0x7f_ffff_ffff, bits[40]:0xf8_c93b_6906,
//     bits[40]:0x4_0000_0000, bits[40]:0x1a_50b7_a249, bits[40]:0x1b_d22c_8bf5,
//     bits[40]:0x89_d035_3077, bits[40]:0x19_523e_f4c6, bits[40]:0x1a_d026_3204];
//     bits[19]:0x2_791a; bits[38]:0x16_e409_8cab"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x7f_ffff_ffff, bits[40]:0xff_ffff_ff00,
//     bits[40]:0xff_ffff_f700, bits[40]:0x0, bits[40]:0x7f_ffff_ffff, bits[40]:0xab_df6f_7955,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0xea_f9ef_da6e, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xff_ffff_ffff]; bits[19]:0x3_775f; bits[38]:0x0"
//     args: "bits[32]:0x5555_5555; [bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5500,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x55_1555_557f, bits[40]:0x7f_ffff_ffff, bits[40]:0x2000,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x4_0000_0000, bits[40]:0x54_1d51_114c,
//     bits[40]:0x15_ff7e_f8bf, bits[40]:0x5b_9574_102e]; bits[19]:0x7_ffff;
//     bits[38]:0x32_0914_77f7"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x1000_0000, bits[40]:0x44_f1b7_17bf,
//     bits[40]:0x55_dd65_5500, bits[40]:0x31_5751_1516, bits[40]:0x0, bits[40]:0x40_0000_0000,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x65_d559_5710, bits[40]:0x20_0000_0000]; bits[19]:0x0; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x0; [bits[40]:0x80_8449_9452, bits[40]:0x10_48bf, bits[40]:0x18_0012_3959,
//     bits[40]:0x8a0_2095, bits[40]:0x5_9622_02dd, bits[40]:0x8_e441_3188, bits[40]:0x10,
//     bits[40]:0x45_0010_643e, bits[40]:0x91_2840_02db, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x50_1020_00ba]; bits[19]:0x5_5555; bits[38]:0x80"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0x7f_ffff_ffff, bits[40]:0x7f_ffff_fffb,
//     bits[40]:0x7f_7ff7_ffbb, bits[40]:0x7f_ffff_ffff, bits[40]:0x7f_bfff_ebab, bits[40]:0x0,
//     bits[40]:0x0, bits[40]:0x7f_9ffe_fd20, bits[40]:0x10_0000_0000, bits[40]:0x6d_f4fb_dff5,
//     bits[40]:0x1]; bits[19]:0x90f7; bits[38]:0x6_d6b3_4fcc"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xdf_8fdf_ef01, bits[40]:0xff_fbff_fe5f,
//     bits[40]:0xff_ffef_fe7f, bits[40]:0xff_ffff_ffff, bits[40]:0xbe_5eff_fd09, bits[40]:0x0,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xfd_ff5f_f418, bits[40]:0x55_5555_5555,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x2b_7155_89a2]; bits[19]:0x0; bits[38]:0x15_5555_5555"
//     args: "bits[32]:0xa863_ff56; [bits[40]:0x20_0000_0000, bits[40]:0x8c_baff_5b33,
//     bits[40]:0x7981_5135, bits[40]:0x0, bits[40]:0xa9_61ff_4608, bits[40]:0xf8_912b_50be,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x0, bits[40]:0xfb_4433_c357, bits[40]:0x10_0000_0000,
//     bits[40]:0x7f_ffff_ffff]; bits[19]:0x0; bits[38]:0x80_0000"
//     args: "bits[32]:0x7ec4_a718; [bits[40]:0x7e_c4a3_1c7f, bits[40]:0x7e_d4a7_0800,
//     bits[40]:0x2_0000_0000, bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x4e_46a3_387e, bits[40]:0xde_eaa7_1a3d,
//     bits[40]:0x3e_c637_3add, bits[40]:0x7e_c4a7_1942, bits[40]:0x6f_c5a6_2863];
//     bits[19]:0x5_5555; bits[38]:0x6_b17b_cb7c"
//     args: "bits[32]:0x32c5_a62b; [bits[40]:0x53_85a6_3a22, bits[40]:0x12_c7a6_2baa,
//     bits[40]:0x20_0000, bits[40]:0x7f_ffff_ffff, bits[40]:0xf6_c983_a37f,
//     bits[40]:0xb9_c5cf_d236, bits[40]:0xeb_3b73_3d23, bits[40]:0xa2_c3a6_22ef,
//     bits[40]:0x72_c40c_23ff, bits[40]:0x5a_c7da_3040, bits[40]:0x20_e3b6_0f40];
//     bits[19]:0x7_ffff; bits[38]:0x0"
//     args: "bits[32]:0x0; [bits[40]:0x7b_a889_854d, bits[40]:0x12_3940_058a,
//     bits[40]:0xc_a258_4884, bits[40]:0xa4_0a40_190c, bits[40]:0x6_4116_0465,
//     bits[40]:0x8c_34c6_9ccb, bits[40]:0x40, bits[40]:0x5_0000_00b2, bits[40]:0xc3_91c1_1b75,
//     bits[40]:0x17_6196_07b3, bits[40]:0xff_ffff_ffff]; bits[19]:0x800; bits[38]:0x4002_eaaa"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_5555, bits[40]:0x12_d974_5788,
//     bits[40]:0x58_1012_5515, bits[40]:0x55_7555_45b6, bits[40]:0x5c_174f_1030, bits[40]:0x0,
//     bits[40]:0x5b_5555_17b5, bits[40]:0x1, bits[40]:0x52_6757_459a, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x0]; bits[19]:0x5_5555; bits[38]:0x3a_aa2e_0210"
//     args: "bits[32]:0x4_0000; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0x401_0140,
//     bits[40]:0x7e_8ecc_f847, bits[40]:0xff_ffff_ffff, bits[40]:0x0, bits[40]:0x1c_7470_4ec2,
//     bits[40]:0x4_9405_400f, bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x8_0000_0000,
//     bits[40]:0x2000]; bits[19]:0x3_ffff; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x5555_5555; [bits[40]:0xff_ffff_ffff, bits[40]:0x75_55f5_55a4,
//     bits[40]:0x55_5551_55aa, bits[40]:0x77_4545_557f, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xd5_358c_955c, bits[40]:0x0, bits[40]:0x57_5555_15d1, bits[40]:0x51_5511_5755,
//     bits[40]:0x19_005d_c730, bits[40]:0x200_0000]; bits[19]:0x3_ffff; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xf8_e2fd_aec5, bits[40]:0xae_fe2f_7d20,
//     bits[40]:0x200_0000, bits[40]:0xe4_2352_6d1d, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xbf_fdf7_fbba, bits[40]:0x1000, bits[40]:0xd7_beef_bf02, bits[40]:0x83_4fae_b80e,
//     bits[40]:0xb9_87ef_f526, bits[40]:0xf0_bbfb_fe20]; bits[19]:0x3_7a4e;
//     bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x7c_7e83_4fdb, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x10_0000_0000, bits[40]:0xfe_d3ff_df01,
//     bits[40]:0x5f_f17b_3be1, bits[40]:0x20_0000_0000, bits[40]:0x2_0000_0000, bits[40]:0x0,
//     bits[40]:0xe1_a169_d6ee, bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x4_8997; bits[38]:0x5_28b8_4016"
//     args: "bits[32]:0x0; [bits[40]:0x5_5240_0875, bits[40]:0xc_1060_c0f9,
//     bits[40]:0x12_9c72_c538, bits[40]:0xd0_d0b8_5a49, bits[40]:0x2_005d_007f,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x55_5555_5555, bits[40]:0x0, bits[40]:0x86_4845_0a27,
//     bits[40]:0x55_5555_5555, bits[40]:0x9_a348_acff]; bits[19]:0x5_5522; bits[38]:0x800_0000"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0x7f_ffff_ffff, bits[40]:0x3f_fdff_ffa0,
//     bits[40]:0x55_5555_5555, bits[40]:0x7d_b9dc_24a2, bits[40]:0x55_5555_5555,
//     bits[40]:0x39_482b_b34a, bits[40]:0x79_7a3e_bbe3, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x7f_ffff_f9ff, bits[40]:0xfe_bf8e_1336];
//     bits[19]:0x1_37b2; bits[38]:0x0"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0x7e_a202_ee75, bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x0, bits[40]:0x0, bits[40]:0x55_5555_5555,
//     bits[40]:0x100_0000, bits[40]:0xe6_8be2_ed4b, bits[40]:0x73_3e7c_ebe2]; bits[19]:0x0;
//     bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x0, bits[40]:0x13_1743_5dff, bits[40]:0x37_6614_64ad,
//     bits[40]:0xdf_4551_0155, bits[40]:0x55_5555_5555, bits[40]:0x0, bits[40]:0x4c_f0a3_6c5a,
//     bits[40]:0x7d_1555_758e, bits[40]:0x93_f534_0d4a, bits[40]:0x10_0000_0000,
//     bits[40]:0x400_0000]; bits[19]:0x5_7551; bits[38]:0x4_cd14_e475"
//     args: "bits[32]:0x5555_5555; [bits[40]:0xb3_8f88_83d6, bits[40]:0x79_411f_1573,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x2c_6a59_71a5, bits[40]:0xe5_5c55_0415,
//     bits[40]:0xd5_7f55_5765, bits[40]:0x1000, bits[40]:0xff_ffff_ffff, bits[40]:0x64_2f14_a857,
//     bits[40]:0xeb_cbf6_d2ea, bits[40]:0x55_5555_5555]; bits[19]:0x7_ffff;
//     bits[38]:0x27_de7f_c065"
//     args: "bits[32]:0x0; [bits[40]:0x2_0000_0000, bits[40]:0x22_0000_406d,
//     bits[40]:0x2a_08bd_804a, bits[40]:0x0, bits[40]:0x64_2000_08ff, bits[40]:0xd2_7443_5115,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x3_9edf_9838, bits[40]:0xff, bits[40]:0x50_0b8c_d420,
//     bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x4; bits[38]:0x15_5555_5555"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5555,
//     bits[40]:0xea_aa2a_aa7f, bits[40]:0xa2_08a2_e210, bits[40]:0xfa_8aa7_f03f,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xc8_bb86_79e0, bits[40]:0xaa_aaaa_aa3f,
//     bits[40]:0xaa_ba9d_8eff, bits[40]:0xaa_aaaa_aa55, bits[40]:0x7f_ffff_ffff];
//     bits[19]:0x2_aaaa; bits[38]:0x1f_aaae_589c"
//     args: "bits[32]:0x1000; [bits[40]:0xff_ffff_ffff, bits[40]:0x8_0428, bits[40]:0x1_0000_0000,
//     bits[40]:0x2_0000, bits[40]:0x55_5555_5555, bits[40]:0xff_ffff_ffff, bits[40]:0x1000,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x100_0000, bits[40]:0x800_0000, bits[40]:0x55_5555_5555];
//     bits[19]:0x7_ffff; bits[38]:0x4_001f"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x2d_7dff_f16c, bits[40]:0x8000, bits[40]:0x0,
//     bits[40]:0x0, bits[40]:0xff_d5ef_de00, bits[40]:0xed_f7f5_dd5b, bits[40]:0xcf_21bc_e31b,
//     bits[40]:0xff_ffdf_fdfb, bits[40]:0xff_ffff_ffff, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x7f_ffff_ffff]; bits[19]:0x1_7eef; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x55_5555_5555, bits[40]:0xef_fbbf_ffaa,
//     bits[40]:0x73_9a74_4742, bits[40]:0xff_ffff_ffff, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xf7_6d51_6547, bits[40]:0xbf_ffff_ff10, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x9d_ac97_d026, bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5555];
//     bits[19]:0x7_bf73; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0x7f_ffff_ffff, bits[40]:0x55_5555_5555,
//     bits[40]:0x10_bb1a_6e09, bits[40]:0x2a_38ca_b626, bits[40]:0x4_0000, bits[40]:0xba_8818_5a3f,
//     bits[40]:0x100_0000, bits[40]:0x8e_c4ba_ee41, bits[40]:0xbc_22ff_a045,
//     bits[40]:0x38_aac0_9ae8, bits[40]:0xec_5a18_4f82]; bits[19]:0x5_5555;
//     bits[38]:0x15_5555_5555"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0x0, bits[40]:0x0, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x5c_7edd_ff28, bits[40]:0x7f_ffff_fb55, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xdd_ff77_afd5, bits[40]:0x9f_eadf_7f1a, bits[40]:0x0, bits[40]:0x20_0000_0000,
//     bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x2_aaaa; bits[38]:0x4_55ea_f962"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0x7f_ffff_ffff, bits[40]:0x80_0000_0000, bits[40]:0x0,
//     bits[40]:0xde_2516_1dc2, bits[40]:0xff_ffff_ffff, bits[40]:0xff_ffff_ffff, bits[40]:0x0,
//     bits[40]:0x10_bdac_48e8, bits[40]:0x55_5555_5555, bits[40]:0xff_fff1_9b5d,
//     bits[40]:0x8000_0000]; bits[19]:0x2_aaaa; bits[38]:0xf_b777_7742"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_9754_75aa, bits[40]:0x68_120e_6da2,
//     bits[40]:0xf4_ec54_5d55, bits[40]:0x68_cd77_3e1a, bits[40]:0xb3_5f28_9241, bits[40]:0x2_0000,
//     bits[40]:0x55_5555_5555, bits[40]:0xb6_1e77_67b6, bits[40]:0x55_5555_5555,
//     bits[40]:0x40_6652_4e95, bits[40]:0x0]; bits[19]:0x4_0000; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x5a19_d49a; [bits[40]:0x55_5555_5555, bits[40]:0x49_1d17_38d9,
//     bits[40]:0x5a_19d4_9aff, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x14_3dc1_9840,
//     bits[40]:0x5a_19d4_9aaa, bits[40]:0x55_5555_5555, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x80_0000, bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x1_5698;
//     bits[38]:0x39_944c_3a1c"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_5555, bits[40]:0x1000,
//     bits[40]:0xcc_432c_80ee, bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xd_5f5d_dc3e, bits[40]:0x71_d54d_574a, bits[40]:0xd6_1555_5510,
//     bits[40]:0x5f_6557_55c8, bits[40]:0x55_5555_55ea, bits[40]:0xd5_4451_5500]; bits[19]:0x0;
//     bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x0; [bits[40]:0x4_5180_aa10, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x40_1077,
//     bits[40]:0xd8_95ee_a69f, bits[40]:0x55_5555_5555, bits[40]:0xf4_2e72_ae3f,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x88_02c2_32aa, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x13_2802_8808, bits[40]:0xb3_890c_933d]; bits[19]:0x2_aaaa;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x1000_0000; [bits[40]:0xbe_610c_e20b, bits[40]:0x40_0000_0000,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x10_1500_00cb, bits[40]:0x0, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xd0_1020_3a76, bits[40]:0x10_0000_00ea, bits[40]:0x0, bits[40]:0xf5_1684_0781,
//     bits[40]:0x7f_ffff_ffff]; bits[19]:0x5_5555; bits[38]:0x8"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0xff_ffff_ffff, bits[40]:0x7e_febf_f45e,
//     bits[40]:0x9f_db57_bf6f, bits[40]:0x73_ffef_db50, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x7f_cfff_fbcf, bits[40]:0x55_5555_5555, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x0, bits[40]:0x37_3756_db65]; bits[19]:0x5_effe;
//     bits[38]:0x2b_66a6_1685"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_5555, bits[40]:0x25_d53f_54bc,
//     bits[40]:0x7c_44a5_970c, bits[40]:0x7f_ffff_ffff, bits[40]:0x70_3d6e_c029,
//     bits[40]:0x5e_bf65_43ff, bits[40]:0xf9_4e65_152e, bits[40]:0xd9_555d_55ff,
//     bits[40]:0xd9_5c5c_9447, bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x3_ffff;
//     bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x0; [bits[40]:0x71_0dec_63a8, bits[40]:0x44_0290_537f,
//     bits[40]:0x51_f50a_808e, bits[40]:0x1_0008_2a00, bits[40]:0xff, bits[40]:0x6b_da10_9f1e,
//     bits[40]:0x13_0d0a_8afd, bits[40]:0x48_e1f7_33f5, bits[40]:0x8026, bits[40]:0x0,
//     bits[40]:0x80_8e2e_2f97]; bits[19]:0x8; bits[38]:0x10"
//     args: "bits[32]:0xf124_ae32; [bits[40]:0xd9_f52e_4abe, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xb8_b4be_9a20, bits[40]:0xb7_2686_f236, bits[40]:0xac_fea4_b843,
//     bits[40]:0x79_64aa_3044, bits[40]:0xb0_00df_1262, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xf1_24ae_32ff, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff];
//     bits[19]:0x3_b29e; bits[38]:0x0"
//     args: "bits[32]:0xea89_a40c; [bits[40]:0xb6_0c20_1da9, bits[40]:0xd6_531b_9a0c,
//     bits[40]:0xea_89a4_0c8e, bits[40]:0x7f_ffff_ffff, bits[40]:0xbc_6922_0921,
//     bits[40]:0x2b_64fb_96c9, bits[40]:0xe0_8124_0caa, bits[40]:0x55_5555_5555, bits[40]:0x4,
//     bits[40]:0xea_cc34_0c80, bits[40]:0x8a_80a0_9cc6]; bits[19]:0x7_8437;
//     bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0x0, bits[40]:0xc7_a2a7_fabd, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xaa_baaa_eaac, bits[40]:0x2b_aafc_9214, bits[40]:0x97_0d7f_b273,
//     bits[40]:0xa8_aaaa_aa00, bits[40]:0x4a_eaba_98ff, bits[40]:0xaa_aaaa_aa20,
//     bits[40]:0x32_12ea_7310, bits[40]:0x7f_ffff_ffff]; bits[19]:0x3_ffff; bits[38]:0xa_fffd_0d55"
//     args: "bits[32]:0x0; [bits[40]:0xa0_0004_80a1, bits[40]:0x55_5555_5555, bits[40]:0x2200_44ab,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x2_0173_4182, bits[40]:0x0, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xee_40b4_dcfc, bits[40]:0xff_ffff_ffff, bits[40]:0x40_962e_5e27,
//     bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x4_4b1b; bits[38]:0x0"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xe2_aaa3_aa09, bits[40]:0x1b_0c9b_8b27,
//     bits[40]:0xaa_ba2d_a823, bits[40]:0xaa_a2aa_aaa8, bits[40]:0x88_aaaa_a840,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x55_5555_5555, bits[40]:0x8a_aa2f_a865];
//     bits[19]:0x3_ffff; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xd382_7a63; [bits[40]:0xf1_037a_6355, bits[40]:0x81_8a6a_73ef,
//     bits[40]:0x52_823b_63aa, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xcb_89fa_f702, bits[40]:0x0,
//     bits[40]:0xdb_8dbb_d1e5, bits[40]:0xd3_92f2_e76b, bits[40]:0x0, bits[40]:0x55_5555_5555,
//     bits[40]:0xdb_c23a_6301]; bits[19]:0x0; bits[38]:0x3c_d8fa_b7e2"
//     args: "bits[32]:0x2327_e755; [bits[40]:0xa_6273_3cc3, bits[40]:0x27_67e3_55ac,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x0, bits[40]:0x8000, bits[40]:0x23_64cf_d5ff,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x23_23ef_555d, bits[40]:0x62_ef8f_0524,
//     bits[40]:0xa7_17e7_65cb, bits[40]:0x55_5555_5555]; bits[19]:0x2_aaaa;
//     bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x3525_f70e; [bits[40]:0x55_5555_5555, bits[40]:0xe7_9079_3ea7,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x95_e5d7_1a05, bits[40]:0x6_976d_abfa,
//     bits[40]:0x35_25f7_9edf, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x0, bits[40]:0x0,
//     bits[40]:0x75_25e7_ae31, bits[40]:0x55_5555_5555]; bits[19]:0x5_5555;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_5555, bits[40]:0x15_5555_57c9,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x47_4119_4735, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xb7_4e5d_3d90, bits[40]:0x63_3d55_0dea, bits[40]:0x58_d43d_528a, bits[40]:0x0,
//     bits[40]:0x55_5555_5504, bits[40]:0x1d_5957_557f]; bits[19]:0x4_f920;
//     bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x0, bits[40]:0xff_ffff_ffff, bits[40]:0x45_5511_d500,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x55_5555_5555, bits[40]:0xdd_de55_1141, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x75_45d4_17f9, bits[40]:0x55_4541_51f9]; bits[19]:0x7_ffff;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0x80_aae2_aaa8, bits[40]:0x9a_aee2_35f0,
//     bits[40]:0x100_0000, bits[40]:0xff_ffff_ffff, bits[40]:0xaa_a3aa_3ad6,
//     bits[40]:0xe8_a8ab_be7b, bits[40]:0x0, bits[40]:0x0, bits[40]:0x67_ce49_72f3,
//     bits[40]:0xaa_aaea_aa40, bits[40]:0x0]; bits[19]:0x2_aaba; bits[38]:0x17_1570_0810"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xff_efff_ff55, bits[40]:0x55_5555_5555,
//     bits[40]:0x55_5555_5555, bits[40]:0x1f_6b86_d510, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x0,
//     bits[40]:0xbd_4f7d_ad3a, bits[40]:0xce_a7ff_b120, bits[40]:0xff_f5df_ff7f, bits[40]:0x0,
//     bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x3_ffff; bits[38]:0x37_c5b2_94ae"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xff_ffff_ffff, bits[40]:0xf1_fbfd_ff80,
//     bits[40]:0xf6_2bb8_e2df, bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5555,
//     bits[40]:0xf6_91a7_7cc5, bits[40]:0xe7_72b5_fb39, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xdf_fffd_bf58, bits[40]:0xbb_ffc1_9fab, bits[40]:0x4e_7b7a_d1c5];
//     bits[19]:0x7_ffff; bits[38]:0x3f_ffff_ffdd"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0x1_0000_0000,
//     bits[40]:0x31_18aa_ecff, bits[40]:0x0, bits[40]:0x80_8eee_ca4d, bits[40]:0xa_56d2_a98d,
//     bits[40]:0x63_8aae_98c2, bits[40]:0x55_5555_5555, bits[40]:0x0, bits[40]:0xab_2aba_ea6a,
//     bits[40]:0x55_5555_5555]; bits[19]:0x8; bits[38]:0x0"
//     args: "bits[32]:0x0; [bits[40]:0x55_5555_5555, bits[40]:0x7f_ffff_ffff, bits[40]:0x0,
//     bits[40]:0x87_75b4_0d8b, bits[40]:0xff_ffff_ffff, bits[40]:0x40_180c_08e2, bits[40]:0x10,
//     bits[40]:0x55_5555_5555, bits[40]:0x50_71c2_44a8, bits[40]:0x8e_2835_382c,
//     bits[40]:0x7f_ffff_ffff]; bits[19]:0x2_5fc4; bits[38]:0x0"
//     args: "bits[32]:0x0; [bits[40]:0xd4_0011_7254, bits[40]:0x100_0000, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x55_5555_5555, bits[40]:0x400_0000, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x98_10c2_127f, bits[40]:0x82_5c61_8dfc, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x0]; bits[19]:0x120; bits[38]:0xa_2126_9fba"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x5d_049d_d7ba, bits[40]:0x55_7d55_55fd,
//     bits[40]:0x40_0000_0000, bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5555,
//     bits[40]:0x5d_d555_57ff, bits[40]:0x0, bits[40]:0x0, bits[40]:0x56_4935_74ba,
//     bits[40]:0x55_5555_5555, bits[40]:0x57_7114_84a0]; bits[19]:0x5_55dd; bits[38]:0x2_0000_0000"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xae_b594_9ddf, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x37_1f7d_ff03, bits[40]:0x1b_d6ed_9d87, bits[40]:0xcf_eff5_e1e3,
//     bits[40]:0x6f_efef_e740, bits[40]:0x76_fffb_6f41, bits[40]:0xa5_cb44_cf29,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa];
//     bits[19]:0x7_ffff; bits[38]:0x15_5555_5555"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0x55_5555_5555, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x8_0000, bits[40]:0xab_abbe_0b5f, bits[40]:0x37_2a90_23a6, bits[40]:0xfa_aaaa_9a37,
//     bits[40]:0x3a_866f_680a, bits[40]:0x2a_aa8a_ae77, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xea_d6ca_aaff, bits[40]:0x74_e862_b260]; bits[19]:0x7_ffff;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5555,
//     bits[40]:0xff_ffff_ffff, bits[40]:0x200, bits[40]:0x99_f8dc_2356, bits[40]:0x1d_da5f_6884,
//     bits[40]:0xf7_fece_ff6f, bits[40]:0x7d_bd77_dbd4, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xff_bffd_ff02]; bits[19]:0x7_ffff;
//     bits[38]:0x15_5555_5555"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_5555, bits[40]:0x55_ff54_d17d,
//     bits[40]:0x29_2923_1149, bits[40]:0x0, bits[40]:0x71_d115_b58a, bits[40]:0x55_5555_5555,
//     bits[40]:0x0, bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5508, bits[40]:0x0,
//     bits[40]:0x4_0000]; bits[19]:0x3_ffff; bits[38]:0x34_0d06_bd47"
//     args: "bits[32]:0x0; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0x8_0000, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0x40_0000_0008, bits[40]:0x81_070e_0613, bits[40]:0xff, bits[40]:0x80_0800_00ff,
//     bits[40]:0x0, bits[40]:0x2_0000_0000, bits[40]:0xff_ffff_ffff, bits[40]:0xd0_0c98_c020];
//     bits[19]:0x4040; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0xf6_fa2a_e89d,
//     bits[40]:0xaa_aaaa_aa04, bits[40]:0x0, bits[40]:0x6a_daca_8a49, bits[40]:0x55_5555_5555,
//     bits[40]:0x8000, bits[40]:0xee_7ba2_aca8, bits[40]:0x28_f02b_a214, bits[40]:0x45_37bf_4ff7,
//     bits[40]:0xe8_36af_c2e4]; bits[19]:0x0; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0x77_4349_a604,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x94_5ca5_cd36, bits[40]:0xff_ffff_ffff, bits[40]:0x0,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x55_5555_5555, bits[40]:0x0, bits[40]:0x7f_ffff_ff55,
//     bits[40]:0x7f_ffff_ff00]; bits[19]:0x7_ffff; bits[38]:0x16_dfbf_ffdf"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_555e_55c2, bits[40]:0x55_d475_55c4,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x10_0000_0000, bits[40]:0x55_5555_5555,
//     bits[40]:0x75_551b_7df5, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x57_0517_1566,
//     bits[40]:0x45_5555_5554, bits[40]:0xe4_1870_e2eb, bits[40]:0x55_5555_5560];
//     bits[19]:0x5_5555; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x6d_f7f4_5a5f, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xed_fffe_e709, bits[40]:0xff_ffff_ff7f, bits[40]:0xdf_bff7_49b7,
//     bits[40]:0xbf_77fd_b78e, bits[40]:0xf3_dff7_ed17, bits[40]:0xfd_ffdf_ff40,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0xdf_ff7f_ff10, bits[40]:0x5f_e758_ef20];
//     bits[19]:0x6_e8ee; bits[38]:0x0"
//     args: "bits[32]:0x8_0000; [bits[40]:0xaa, bits[40]:0x0, bits[40]:0x400,
//     bits[40]:0xe0_0b02_90fd, bits[40]:0x800_80e8, bits[40]:0x3e_874e_a1cf,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x0, bits[40]:0xff_ffff_ffff, bits[40]:0x18_0da1_81c4,
//     bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x5_5555; bits[38]:0x40_0000"
//     args: "bits[32]:0xf7e4_c02d; [bits[40]:0xf7_e4c0_2d09, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xf5_e4c0_2cee, bits[40]:0xfb_aba5_0471, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xf7_04d4_6ded,
//     bits[40]:0xd5_ecc0_3d7f, bits[40]:0xb2_e180_2d08, bits[40]:0x85_5402_b856];
//     bits[19]:0x5_5555; bits[38]:0x21_2878_6838"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0x0, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0xff_ffff_ff7f, bits[40]:0xeb_ffd7_fe46, bits[40]:0xf3_3dff_fbce,
//     bits[40]:0x80_0000_0000, bits[40]:0x55_5555_5555, bits[40]:0x0, bits[40]:0x1000,
//     bits[40]:0x5c_9050_bb05]; bits[19]:0x2_aaaa; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0x200, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x80, bits[40]:0x82_c32a_93ed, bits[40]:0x19_7bab_3f70,
//     bits[40]:0x7e_ffef_ff01, bits[40]:0xaf_a729_9fdb, bits[40]:0x0, bits[40]:0x73_cf83_3ea6,
//     bits[40]:0x6e_7dff_ffd8]; bits[19]:0x7_ffff; bits[38]:0x19_76b8_99da"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x57_dd43_d428, bits[40]:0x55_1f35_1196, bits[40]:0x0,
//     bits[40]:0xdd_744d_d7f4, bits[40]:0xff_ffff_ffff, bits[40]:0x13_5554_d8a6,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x0, bits[40]:0xd1_5c71_47c6, bits[40]:0x1c_ff61_3e6a,
//     bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x0; bits[38]:0x20_0000_0000"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_5555, bits[40]:0x0, bits[40]:0xec_c807_7947,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x80_1b5e_b27d, bits[40]:0x0, bits[40]:0xd6_44d5_05cf,
//     bits[40]:0x55_5545_1055, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x55_5515_5555,
//     bits[40]:0x7f_ffff_ffff]; bits[19]:0x4_c96d; bits[38]:0x12_525a_eb13"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x57_5555_557f, bits[40]:0x800,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0xf4_cb83_8c34, bits[40]:0x5c_d1d1_55e8,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0xd7_5355_55ff, bits[40]:0x0, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x55_5555_5555, bits[40]:0xe4_0d15_011e]; bits[19]:0x2_bd4d; bits[38]:0x1_31ef_5907"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xe8_7f6e_98b7, bits[40]:0x53_82fb_2246,
//     bits[40]:0xb1_c2bc_13ff, bits[40]:0x33_a8ac_eb04, bits[40]:0x9a_e0b2_4845,
//     bits[40]:0x8a_a20a_02cf, bits[40]:0x0, bits[40]:0x0, bits[40]:0x1_0000_0000,
//     bits[40]:0x34_e1a0_bc38, bits[40]:0xe8_aaaa_aa6f]; bits[19]:0x2_aaaa; bits[38]:0x4_0000"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5515_c150, bits[40]:0x57_5555_deef,
//     bits[40]:0x51_5555_c5aa, bits[40]:0x55_5555_557f, bits[40]:0x0, bits[40]:0x2e_c4cf_7591,
//     bits[40]:0x4f_1d75_8532, bits[40]:0x41_74e5_c78b, bits[40]:0x14_1570_4580,
//     bits[40]:0x3d_5555_d5a0, bits[40]:0x7f_ffff_ffff]; bits[19]:0x2084; bits[38]:0x23_8126_f690"
//     args: "bits[32]:0x4_0000; [bits[40]:0x23_1cb0_080a, bits[40]:0x8_6424_15a8,
//     bits[40]:0x11_5527_1936, bits[40]:0x7f_ffff_ffff, bits[40]:0x480_00ff,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x400_0000, bits[40]:0x0, bits[40]:0x94_0c08_2254,
//     bits[40]:0x0, bits[40]:0x37_436d_0bf3]; bits[19]:0x400; bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x55_5555_55ff, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x55_5555_5555, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xf4_6515_fc97, bits[40]:0x10_0000_0000,
//     bits[40]:0x7f_ffff_ffff, bits[40]:0x400_0000, bits[40]:0xc9_5754_d127]; bits[19]:0x40;
//     bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x1; [bits[40]:0x8204_11de, bits[40]:0xa_0000_4153, bits[40]:0x800_0000,
//     bits[40]:0x70_1cdc_8afe, bits[40]:0x55_5555_5555, bits[40]:0x28e0_81af,
//     bits[40]:0x94_9c89_04d6, bits[40]:0x24_225c_8916, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x26_1848_76f7, bits[40]:0x0]; bits[19]:0x7_ffff; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0x0; [bits[40]:0x0, bits[40]:0x8_0000_04fb, bits[40]:0xb5_3ba7_38b3,
//     bits[40]:0x2_c002_0008, bits[40]:0x5_5694_2377, bits[40]:0x1e_019e_a6e9,
//     bits[40]:0x10_9808_0055, bits[40]:0x40_2402_016f, bits[40]:0x7f_ffff_ffff,
//     bits[40]:0x21_0830_0b7f, bits[40]:0x55_5555_5555]; bits[19]:0x5_5555;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0x4000_0000; [bits[40]:0x80, bits[40]:0x69_8040_2040,
//     bits[40]:0xc0_726c_d6b0, bits[40]:0xec_c021_21b6, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x41_0005_8057, bits[40]:0x4608_2703, bits[40]:0xff_ffff_ffff,
//     bits[40]:0x6_c1c0_10af, bits[40]:0x40_8025_10aa, bits[40]:0x0]; bits[19]:0x2_aaaa;
//     bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0x55_5555_5555,
//     bits[40]:0xbf_decf_beaa, bits[40]:0x3c_b613_eb3f, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xff_ffff_ffff, bits[40]:0x6f_ffbf_bfd0,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xf4_72a9_77fa, bits[40]:0x0]; bits[19]:0x200;
//     bits[38]:0x15_5555_5555"
//     args: "bits[32]:0x5555_5555; [bits[40]:0x0, bits[40]:0xff_ffff_ffff, bits[40]:0xb6_450f_25c3,
//     bits[40]:0xb4_6545_46ea, bits[40]:0x55_5555_5555, bits[40]:0x70_116d_6b2c,
//     bits[40]:0x55_c759_94e0, bits[40]:0x45_55d5_55ff, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x5_5555;
//     bits[38]:0x2a_aaaa_aaaa"
//     args: "bits[32]:0xf514_eb13; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0xf1_16eb_877f,
//     bits[40]:0xb2_5179_03ac, bits[40]:0xae_fc8b_2f25, bits[40]:0xc4_d4f2_17aa,
//     bits[40]:0xfa_24ab_1a59, bits[40]:0x85_1c73_2203, bits[40]:0xd4_4d67_1a2c,
//     bits[40]:0xf7_95eb_83af, bits[40]:0x55_5555_5555, bits[40]:0x9b_2be6_3725];
//     bits[19]:0x2_aaaa; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x1d0d_0347; [bits[40]:0x1d_0d03_475f, bits[40]:0x31_8b83_cfe3,
//     bits[40]:0xac_9071_dd76, bits[40]:0x1d_6902_13c7, bits[40]:0x93_0fa9_4318,
//     bits[40]:0x55_5555_5555, bits[40]:0x16_2e43_4fdc, bits[40]:0x3f_ed03_4e6b,
//     bits[40]:0x55_5555_5555, bits[40]:0x8000_0000, bits[40]:0xaa_aaaa_aaaa]; bits[19]:0x5_034f;
//     bits[38]:0xa_cfb1_1912"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0x7f_ffff_ffd6, bits[40]:0x7f_ffff_f77f,
//     bits[40]:0x7d_29ef_aaf2, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xf3_64ed_f8aa, bits[40]:0x0,
//     bits[40]:0x71_16bf_7808, bits[40]:0x4d_5fe9_4718, bits[40]:0xd_307e_d0c9,
//     bits[40]:0xf6_0354_19c9, bits[40]:0x8]; bits[19]:0x1_0000; bits[38]:0x1000_0000"
//     args: "bits[32]:0xaaaa_aaaa; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0xa0_a8a2_ca37,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xaa_aeab_aa00, bits[40]:0x7b_c528_3362,
//     bits[40]:0xaa_aaaa_a840, bits[40]:0xaa_aaaa_aaaa, bits[40]:0xdb_17c8_03f7,
//     bits[40]:0xb2_aa88_aa55, bits[40]:0x0, bits[40]:0xaa_a8aa_6ae9]; bits[19]:0x2_aaaa;
//     bits[38]:0x20_19cd_0144"
//     args: "bits[32]:0x400_0000; [bits[40]:0x7f_ffff_ffff, bits[40]:0x2e_2598_a734,
//     bits[40]:0x64_844f_a5fa, bits[40]:0x80_0d18_107f, bits[40]:0x4_0100_4055,
//     bits[40]:0xe_b3c8_14b9, bits[40]:0x6_91f3_b1ba, bits[40]:0x54_8192_cd00,
//     bits[40]:0x4_8000_3080, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x4_0000_0000]; bits[19]:0x1_00d1;
//     bits[38]:0x8_06cd_5555"
//     args: "bits[32]:0x7a44_f9d9; [bits[40]:0x1d_e2a7_69b3, bits[40]:0xf4_49bd_d0b2,
//     bits[40]:0x55_5555_5555, bits[40]:0xff_ffff_ffff, bits[40]:0x0, bits[40]:0x5a_54d8_d901,
//     bits[40]:0x3a_44f9_d97f, bits[40]:0xfa_4df9_d1af, bits[40]:0x20, bits[40]:0x55_5555_5555,
//     bits[40]:0x55_5555_5555]; bits[19]:0x71dd; bits[38]:0x3f_ffff_ffff"
//     args: "bits[32]:0xd9b9_5a33; [bits[40]:0xd8_b91b_3233, bits[40]:0xff_ffff_ffff,
//     bits[40]:0xd0_b97b_33f6, bits[40]:0xd9_f95a_b3ae, bits[40]:0x15_f998_b3a7,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xd9_b95a_3340, bits[40]:0xd3_a95f_2b22, bits[40]:0x0,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xd1_fd4a_0360]; bits[19]:0x4000; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x0; [bits[40]:0xff_ffff_ffff, bits[40]:0x55_5555_5555,
//     bits[40]:0xff_ffff_ffff, bits[40]:0xa0_0088_0098, bits[40]:0x0, bits[40]:0x1c_4040_12b0,
//     bits[40]:0x55_5555_5555, bits[40]:0x3_0010_0055, bits[40]:0x2_0000, bits[40]:0xc8_0455_452b,
//     bits[40]:0x0]; bits[19]:0x5_5555; bits[38]:0x1f_ffff_ffff"
//     args: "bits[32]:0x7fff_ffff; [bits[40]:0xaa_aaaa_aaaa, bits[40]:0x0, bits[40]:0x6b_9ffe_bfbf,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x7f_d7ff_ffff, bits[40]:0x0, bits[40]:0x7f_fbff_ff7f,
//     bits[40]:0x79_f8d9_fdcf, bits[40]:0xfb_bfed_c1fc, bits[40]:0x20, bits[40]:0x55_5555_5555];
//     bits[19]:0x0; bits[38]:0x20_0802_f2b7"
//     args: "bits[32]:0xffff_ffff; [bits[40]:0xff_e7a9_6f15, bits[40]:0xfc_9188_db04,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0x5e_58e4_727d, bits[40]:0xaa_aaaa_aaaa,
//     bits[40]:0xaa_aaaa_aaaa, bits[40]:0xaf_e8bf_f73d, bits[40]:0x0, bits[40]:0xf3_b2eb_d020,
//     bits[40]:0x55_5555_5555, bits[40]:0xff_ffff_ffff]; bits[19]:0x4_f57d;
//     bits[38]:0x27_8be8_0052"
//     args: "bits[32]:0x0; [bits[40]:0x0, bits[40]:0x0, bits[40]:0x60_0b12_a2b7,
//     bits[40]:0x3b_c441_2082, bits[40]:0xc_2804_d30e, bits[40]:0xab, bits[40]:0x40_0010,
//     bits[40]:0x1_0051_200b, bits[40]:0xaa_aaaa_aaaa, bits[40]:0x8_0000_0000,
//     bits[40]:0x62_32c4_5617]; bits[19]:0x0; bits[38]:0x0"
//   }
// }
//
// END_CONFIG
const W32_V11 = u32:0xb;

type x0 = u40;

fn x11(x12: x0) -> (x0, x0, u32) {
    {
        let x13: x0 = one_hot_sel(u3:0x5, [x12, x12, x12]);
        let x14: u32 = x13[x12+:u32];
        (x13, x13, x14)
    }
}

fn main(x1: u32, x2: x0[W32_V11], x3: u19, x4: u38) -> (x0[7], x0[11]) {
    {
        let x5: xN[bool:0x0][24] = x4[x4+:u24];
        let x6: bool = x1 < x1;
        let x7: x0[22] = x2 ++ x2;
        let x8: x0[1] = array_slice(x2, x6, x0[1]:[x2[u32:0x0], ...]);
        let x9: (bool, u19, bool) = (x6, x3, x6);
        let x10: bool = x9 == x9;
        let x15: u13 = x5[0+:u13];
        let x16: u2 = one_hot(x10, bool:0x1);
        let x17: bool = x9 != x9;
        let x18: u22 = u22:0x3f_ffff;
        let x19: u38 = -x4;
        let x20: bool = !x17;
        let x21: u5 = x18[0+:u5];
        let x22: bool = x16[-2:-1];
        let x23: x0[7] = array_slice(x7, x20, x0[7]:[x7[xN[bool:0x0][32]:0x0], ...]);
        let x24: u19 = !x3;
        let x25: bool = x6 - x4 as bool;
        let x26: u32 = x1 >> x17;
        let x27: u2 = x16 - x16;
        let x28: x0[11] = array_slice(x23, x10, x0[11]:[x23[u32:0x0], ...]);
        (x23, x28)
    }
}
