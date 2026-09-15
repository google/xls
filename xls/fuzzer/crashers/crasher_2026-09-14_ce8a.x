// Copyright 2026 The XLS Authors
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
// exception: "/xls/tools/simulate_module_main returned a non-zero exit status (1): /xls/tools/simulate_module_main --signature_file=module_sig.textproto --testvector_textproto=testvector.pbtxt --verilog_simulator=iverilog sample.v --logtostderr\n\nSubprocess stderr:\nError: NOT_FOUND: Output `out`, instance #127 holds X value in Verilog simulator output.\n=== Source Location Trace: === \nxls/simulation/module_testbench.cc:370\nxls/simulation/module_testbench.cc:456\nxls/simulation/module_simulator.cc:510\nxls/simulation/module_simulator.cc:550\nxls/tools/simulate_module_main.cc:208\n\n"
// issue: "https://github.com/google/xls/issues/4980"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=5"
//   codegen_args: "--worst_case_throughput=3"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x14c_58b2_2bfa, bits[6]:0x3d, bits[52]:0x0, bits[22]:0x1b_f512); bits[34]:0x2_aaaa_aaaa; bits[44]:0x8fa_aafa_a880; bits[6]:0x1e"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1d7_dbfe_a771, bits[6]:0x0, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x1b_abfe); bits[34]:0x1_ffff_ffff; bits[44]:0x4000_0000; bits[6]:0x0"
//     args: "bits[38]:0xa_ece4_f138; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x30, bits[52]:0x5_5555_5555_5555, bits[22]:0x2d_9edf); bits[34]:0x2_ede4_f138; bits[44]:0xfff_ffff_ffff; bits[6]:0x28"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x22, bits[52]:0x0, bits[22]:0x26_7a12); bits[34]:0x1_e4e9_582e; bits[44]:0x200_0000; bits[6]:0x2e"
//     args: "bits[38]:0x25_8ece_a882; (bits[42]:0x15c_eaff_080f, bits[6]:0x20, bits[52]:0x5_5555_5555_5555, bits[22]:0x3_ad86); bits[34]:0x1_5555_5555; bits[44]:0x0; bits[6]:0x1d"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x0, bits[6]:0x3f, bits[52]:0x100, bits[22]:0x16_f3d5); bits[34]:0xcc5d_3c99; bits[44]:0x2f9_2857_4619; bits[6]:0x2a"
//     args: "bits[38]:0x0; (bits[42]:0x9b_6a94_c8f0, bits[6]:0x1, bits[52]:0xf_ffff_ffff_ffff, bits[22]:0x1a_4c60); bits[34]:0x8_0000; bits[44]:0x406_0020_4dd3; bits[6]:0x1f"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x2a8_aaaa_aaa4, bits[6]:0x2a, bits[52]:0x5_5555_5555_5555, bits[22]:0x15_5555); bits[34]:0x1_5555_5555; bits[44]:0x1a4_ffa4_8d8d; bits[6]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x1e_261f_c2c9, bits[6]:0x1f, bits[52]:0xf_ffff_ffbf_eaba, bits[22]:0x3f_efff); bits[34]:0x2_aaaa_aaaa; bits[44]:0x7ff_ffff_ffff; bits[6]:0x1f"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x0, bits[6]:0x35, bits[52]:0x4_5755_555f_7ff7, bits[22]:0x14_5555); bits[34]:0x3_ffff_ffff; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x2b"
//     args: "bits[38]:0x0; (bits[42]:0x19b_ad1f_da96, bits[6]:0x0, bits[52]:0x3_4880_0152_f248, bits[22]:0x12a0); bits[34]:0x3_ffff_ffff; bits[44]:0x80_0822_029d; bits[6]:0x1a"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x3c, bits[52]:0xb_2376_90d9_6649, bits[22]:0x20_e79c); bits[34]:0x3_ffff_ffff; bits[44]:0x54_099e_1b3e; bits[6]:0x3e"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x187_bd55_7510, bits[6]:0x8, bits[52]:0x2_8493_04f4_54cd, bits[22]:0x25_d755); bits[34]:0x0; bits[44]:0x0; bits[6]:0x20"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x15, bits[52]:0x2_f2b7_ba79_7a5f, bits[22]:0x1f_ffff); bits[34]:0x1_ffff_ffff; bits[44]:0xfff_ffff_ffff; bits[6]:0x1"
//     args: "bits[38]:0x5_e72f_d968; (bits[42]:0x25e_6afd_968f, bits[6]:0x38, bits[52]:0xf_ffff_ffff_ffff, bits[22]:0x2a_aaaa); bits[34]:0x1_e56b_db6a; bits[44]:0x79_5bd6_523f; bits[6]:0x33"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1fa_fe9d_feff, bits[6]:0x3f, bits[52]:0x7_5f7f_ffff_dfff, bits[22]:0x2a_7407); bits[34]:0x3_de7f_ffff; bits[44]:0x35b_822f_4e23; bits[6]:0x2"
//     args: "bits[38]:0x0; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x0, bits[52]:0x2_242c_a141_73bc, bits[22]:0x3f_ffff); bits[34]:0x3_9a59_2c74; bits[44]:0xd6b_65b9_d2e5; bits[6]:0x1f"
//     args: "bits[38]:0x35_8da4_cfcf; (bits[42]:0x1d9_5dfe_77a2, bits[6]:0x2a, bits[52]:0x5_606f_3253_d7fd, bits[22]:0x3c_702c); bits[34]:0x1_5555_5555; bits[44]:0xfff_ffff_ffff; bits[6]:0x2a"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x155_5555_5555, bits[6]:0x1f, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x3f_ffff); bits[34]:0x2_ae88_afaa; bits[44]:0xfff_ffff_ffff; bits[6]:0x3f"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x2aa_88aa_aaa8, bits[6]:0x2a, bits[52]:0xf_ffff_ffff_ffff, bits[22]:0x3a_ea4e); bits[34]:0x3_ffff_ffff; bits[44]:0x0; bits[6]:0x23"
//     args: "bits[38]:0x0; (bits[42]:0x210_0099_4903, bits[6]:0x0, bits[52]:0xb_c81c_25b4_00c1, bits[22]:0x408); bits[34]:0x2_aaaa_aaaa; bits[44]:0x20_0000; bits[6]:0xe"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x3a8_aa3e_3abe, bits[6]:0x2a, bits[52]:0x1_ce81_92bf_5cac, bits[22]:0x23_8ca2); bits[34]:0x2_aaaa_aaaa; bits[44]:0xaa9_aa2a_eaaa; bits[6]:0x1f"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x1a9_c96b_3293, bits[6]:0xa, bits[52]:0xf_a242_e82a_0a05, bits[22]:0x2a_aaaa); bits[34]:0x1_ffff_ffff; bits[44]:0x555_5555_5555; bits[6]:0x15"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x2a8_a291_a64c, bits[6]:0xa, bits[52]:0xe_b833_eaae_9142, bits[22]:0x2a_aef2); bits[34]:0x0; bits[44]:0x3da_b081_03fd; bits[6]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x2e2_aaaa_aaa8, bits[6]:0x0, bits[52]:0x0, bits[22]:0x2a_aaaa); bits[34]:0x1_5555_5555; bits[44]:0x200_0000_0000; bits[6]:0x1f"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x35, bits[52]:0x5_5555_5551_f892, bits[22]:0x1_0000); bits[34]:0x1_5555_5555; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x15"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x12_bcb7_6b35, bits[6]:0x2a, bits[52]:0xb_5d47_0f75_2143, bits[22]:0x5_55c5); bits[34]:0x1_5555_c755; bits[44]:0x554_563d_55ff; bits[6]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x17b_ceb7_5af9, bits[6]:0x3e, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x3f_fffb); bits[34]:0x2_aaaa_aaaa; bits[44]:0x9d2_b334_f091; bits[6]:0x17"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x2, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x1f_ffff); bits[34]:0x1_441d_1570; bits[44]:0xfff_ffff_ffff; bits[6]:0x4"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1ff_fdf7_7ff0, bits[6]:0x3f, bits[52]:0x0, bits[22]:0x1f_ffff); bits[34]:0x4ce3_975d; bits[44]:0xfff_ffff_ffff; bits[6]:0x18"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1ff_f7fd_fffa, bits[6]:0x18, bits[52]:0x0, bits[22]:0x0); bits[34]:0x3_b4df_3bfb; bits[44]:0x555_5555_5555; bits[6]:0x29"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x2a2_2aaa_aaa7, bits[6]:0x1f, bits[52]:0x0, bits[22]:0x10_0000); bits[34]:0x2_0000; bits[44]:0x1000_0000; bits[6]:0x4"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x0, bits[6]:0x39, bits[52]:0x3_d415_c775_1fe7, bits[22]:0x1f_ffff); bits[34]:0x3_57ad_55da; bits[44]:0x565_2557_4055; bits[6]:0x2a"
//     args: "bits[38]:0x0; (bits[42]:0x20_0000_000f, bits[6]:0x0, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x15_5555); bits[34]:0x8811_0d81; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x2b"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x2a, bits[52]:0x5_1177_1d45_4555, bits[22]:0x33_60b5); bits[34]:0xdf7d_1d45; bits[44]:0x370_de65_dd75; bits[6]:0x4"
//     args: "bits[38]:0x0; (bits[42]:0x8a_0080_a10f, bits[6]:0x0, bits[52]:0x20_2aa8, bits[22]:0x12_7034); bits[34]:0x1_5555_5555; bits[44]:0x26_b80e_82c1; bits[6]:0x15"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1ff_ffff_ffff, bits[6]:0x1f, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x0); bits[34]:0x1_8ff5_705b; bits[44]:0x555_5555_5555; bits[6]:0x0"
//     args: "bits[38]:0x0; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x0, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x2a_aaaa); bits[34]:0x1_ffff_ffff; bits[44]:0xfff_ffff_ffff; bits[6]:0x3b"
//     args: "bits[38]:0x4_0000; (bits[42]:0xb040_0010, bits[6]:0xa, bits[52]:0x23_0011_28cf, bits[22]:0xc_0000); bits[34]:0x3_ffff_ffff; bits[44]:0x0; bits[6]:0x2f"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x139_f114_f429, bits[6]:0x15, bits[52]:0xa_be88_9c78_057f, bits[22]:0x1f_ffff); bits[34]:0x3_fb1e_34bc; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x17"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x1ff_ffff_ffff, bits[6]:0x3f, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x3f_df7d); bits[34]:0x2_aaaa_aaaa; bits[44]:0xfff_ffff_ffff; bits[6]:0x2b"
//     args: "bits[38]:0x400_0000; (bits[42]:0x290_4b20_500c, bits[6]:0x0, bits[52]:0x5_5555_5555_5555, bits[22]:0x200); bits[34]:0x3_c7c9_1f14; bits[44]:0x0; bits[6]:0x12"
//     args: "bits[38]:0x24_9dbe_4e2c; (bits[42]:0x259_cbac_63c0, bits[6]:0x7, bits[52]:0x3_2703_3850_f017, bits[22]:0x2d_06a4); bits[34]:0xf9be_4e2c; bits[44]:0x0; bits[6]:0x2d"
//     args: "bits[38]:0x800; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x0, bits[52]:0x2_0000_0305_1fff, bits[22]:0x3f_ffff); bits[34]:0x1_ffff_ffff; bits[44]:0x44_0002_10bf; bits[6]:0x2b"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x0, bits[6]:0x1f, bits[52]:0x1_1563_90d3_f1d6, bits[22]:0x3f_ffff); bits[34]:0x4; bits[44]:0x555_5555_5555; bits[6]:0x7"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x28e_4bd2_eadc, bits[6]:0x3f, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x2a_aa3a); bits[34]:0x2_abaa_aabb; bits[44]:0x555_5555_5555; bits[6]:0x13"
//     args: "bits[38]:0x0; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x10, bits[52]:0x4_46f6_0f68_530c, bits[22]:0x26_0010); bits[34]:0x3_d4f3_42c5; bits[44]:0x40_0000; bits[6]:0x2a"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x155_5555_5555, bits[6]:0x2a, bits[52]:0xe_acc2_abbe_175f, bits[22]:0x20_b8fe); bits[34]:0x1_5555_5555; bits[44]:0x0; bits[6]:0x8"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3ff_7dff_fff0, bits[6]:0x3f, bits[52]:0xf_8dae_aeee_c833, bits[22]:0x37_6fba); bits[34]:0x1_ffff_ffff; bits[44]:0x555_5555_5555; bits[6]:0x32"
//     args: "bits[38]:0x0; (bits[42]:0x155_5555_5555, bits[6]:0xa, bits[52]:0x1_040f_041a_0a6e, bits[22]:0x840); bits[34]:0x240_c05c; bits[44]:0x40_0000; bits[6]:0x1b"
//     args: "bits[38]:0x8_0000_0000; (bits[42]:0xa400_10b7, bits[6]:0x30, bits[52]:0x3_0840_0410_baa2, bits[22]:0x1f_ffff); bits[34]:0x3_ffff_ffff; bits[44]:0x284_2200_40af; bits[6]:0x37"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x155_5555_5555, bits[6]:0x15, bits[52]:0xd_dfbf_69f9_9820, bits[22]:0x3b_9b99); bits[34]:0x1_5555_5555; bits[44]:0x1d3_55d5_d5aa; bits[6]:0x5"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x1f5_5445_5d1c, bits[6]:0x19, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x35_7555); bits[34]:0x1_5514_d575; bits[44]:0x0; bits[6]:0x20"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x23a_3eb2_bef5, bits[6]:0x30, bits[52]:0xe_a498_9279_976a, bits[22]:0x2a_aaaa); bits[34]:0x1_5555_5555; bits[44]:0xb05_2eaa_3a82; bits[6]:0x4"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x155_5555_555a, bits[6]:0x1f, bits[52]:0x5_5555_5555_4010, bits[22]:0x17_4d5c); bits[34]:0x1_5555_5155; bits[44]:0x0; bits[6]:0x15"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1ff_ffff_ffff, bits[6]:0x5, bits[52]:0x7_abaf_f7eb_5efd, bits[22]:0x2c_7f6f); bits[34]:0x0; bits[44]:0x5bb_dfe5_6ab7; bits[6]:0x0"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0xaa_7aaa_eaa1, bits[6]:0x38, bits[52]:0x6_a20b_08b9_8238, bits[22]:0xa_aa0b); bits[34]:0x1_5555_5555; bits[44]:0xfff_ffff_ffff; bits[6]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x155_5555_5555, bits[6]:0x2a, bits[52]:0x3_bf3e_fbdf_b52c, bits[22]:0x2a_d238); bits[34]:0x3_a9df_ee9b; bits[44]:0x32d_5fc8_7edb; bits[6]:0x0"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x15d_7417_d7fc, bits[6]:0x15, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x3f_ffff); bits[34]:0x2_d140_2405; bits[44]:0x0; bits[6]:0x20"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x1f, bits[52]:0x5_3f74_f514_7dfe, bits[22]:0x15_5555); bits[34]:0xe15a_5df3; bits[44]:0x555_5555_5555; bits[6]:0x7"
//     args: "bits[38]:0x100; (bits[42]:0x274_83af_c51e, bits[6]:0x0, bits[52]:0x640_0441_0000, bits[22]:0x0); bits[34]:0x0; bits[44]:0xeaf_fc4f_46ff; bits[6]:0x8"
//     args: "bits[38]:0x0; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x18, bits[52]:0x0, bits[22]:0xc_2220); bits[34]:0x1_5555_5555; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x3f"
//     args: "bits[38]:0x23_a939_874f; (bits[42]:0x21f_d51a_7b77, bits[6]:0xf, bits[52]:0x80_0000, bits[22]:0x3f_ffff); bits[34]:0x10_0000; bits[44]:0x9_c201_00b4; bits[6]:0x1f"
//     args: "bits[38]:0x0; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x32, bits[52]:0x5_5555_5555_5555, bits[22]:0x28_0004); bits[34]:0x0; bits[44]:0x10_9800_000c; bits[6]:0x2a"
//     args: "bits[38]:0x100_0000; (bits[42]:0x330_1986_4011, bits[6]:0x0, bits[52]:0x0, bits[22]:0x8_e756); bits[34]:0x3_3370_7ea7; bits[44]:0xc69_c2f2_dff3; bits[6]:0x0"
//     args: "bits[38]:0x1_0000_0000; (bits[42]:0x1ff_ffff_ffff, bits[6]:0x4, bits[52]:0xd_1621_c402_3a69, bits[22]:0x27_8252); bits[34]:0x3_ffff_ffff; bits[44]:0xf03_1b5e_67ba; bits[6]:0x1"
//     args: "bits[38]:0x29_cdf0_5bb5; (bits[42]:0x2ed_df0e_bf4c, bits[6]:0x0, bits[52]:0x80, bits[22]:0x4000); bits[34]:0x1_c670_58f5; bits[44]:0x7ff_ffff_ffff; bits[6]:0x15"
//     args: "bits[38]:0x27_d5a3_e897; (bits[42]:0x800, bits[6]:0x13, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x2b_2817); bits[34]:0x1_5555_5555; bits[44]:0x12_5535_d212; bits[6]:0x1f"
//     args: "bits[38]:0x0; (bits[42]:0x20_0000, bits[6]:0x0, bits[52]:0x8000_0000_0000, bits[22]:0x0); bits[34]:0x2_2220_0446; bits[44]:0x0; bits[6]:0x2"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0xfa_d6ba_4bf2, bits[6]:0x0, bits[52]:0x8_f42c_08d8_e1d0, bits[22]:0xe_fee7); bits[34]:0x3_fefe_96f9; bits[44]:0x0; bits[6]:0xd"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3be_ffff_fffc, bits[6]:0x19, bits[52]:0x5_5555_5555_5555, bits[22]:0x1b_1b5f); bits[34]:0x1_ffff_ffff; bits[44]:0xbc8_8795_ddd5; bits[6]:0x3f"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x155_5555_5555, bits[6]:0x20, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x19_6ba8); bits[34]:0x3_ffff_ffff; bits[44]:0xfff_ffff_ffff; bits[6]:0x3d"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x13b_a7fd_7d17, bits[6]:0x3f, bits[52]:0x5_5555_5555_5555, bits[22]:0x3f_f7ff); bits[34]:0x1_f720_4814; bits[44]:0x0; bits[6]:0x38"
//     args: "bits[38]:0x6_5aa6_e93f; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x1f, bits[52]:0x40, bits[22]:0x24_8365); bits[34]:0xdbec_f13f; bits[44]:0x19e_a9ba_47f5; bits[6]:0x37"
//     args: "bits[38]:0x0; (bits[42]:0x0, bits[6]:0xe, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x0); bits[34]:0x1_ffff_ffff; bits[44]:0x7ff_ffff_ffff; bits[6]:0x3f"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3ff_ffff_fff8, bits[6]:0x3f, bits[52]:0xe_efef_3c7b_f4df, bits[22]:0x21_f6e6); bits[34]:0x0; bits[44]:0xfff_ffff_ffff; bits[6]:0x0"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x3cb_35dd_4856, bits[6]:0x15, bits[52]:0x5_5555_5555_ffdf, bits[22]:0x34_174f); bits[34]:0x3_55d4_455b; bits[44]:0xdf7_5115_7dbf; bits[6]:0x1b"
//     args: "bits[38]:0x8_0000_0000; (bits[42]:0xaf_110b_5a15, bits[6]:0x12, bits[52]:0x5_5555_5555_5555, bits[22]:0xb_adcc); bits[34]:0x0; bits[44]:0xfff_ffff_ffff; bits[6]:0x1f"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1f3_6bad_f77b, bits[6]:0x3d, bits[52]:0xf_ffff_ff7f_d50f, bits[22]:0x39_ff6e); bits[34]:0x800_0000; bits[44]:0xfff_ffff_ffff; bits[6]:0x4"
//     args: "bits[38]:0x200; (bits[42]:0x155_5555_5555, bits[6]:0x1f, bits[52]:0x1002_2080_c0c0, bits[22]:0x38_5d33); bits[34]:0x3_ffff_ffff; bits[44]:0xfaf_eadc_ff43; bits[6]:0x2a"
//     args: "bits[38]:0x1d_a6f5_e916; (bits[42]:0x155_5555_5555, bits[6]:0x2a, bits[52]:0x9_1c53_c9ee_006d, bits[22]:0x0); bits[34]:0x1_a6b5_d9b6; bits[44]:0x555_5555_5555; bits[6]:0x1d"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x2d3_423d_13f7, bits[6]:0x10, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0xf_bf00); bits[34]:0x3_715f_4777; bits[44]:0xcc5_7495_d405; bits[6]:0x1"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x1f3_f5eb_f779, bits[6]:0x3e, bits[52]:0x5_5555_5555_5555, bits[22]:0x1f_ffff); bits[34]:0x1_5555_5555; bits[44]:0x555_5555_5555; bits[6]:0x15"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x0, bits[6]:0x15, bits[52]:0x5_5555_5555_5555, bits[22]:0x2a_aaaa); bits[34]:0x0; bits[44]:0x0; bits[6]:0x30"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x20_0000, bits[6]:0x1, bits[52]:0x0, bits[22]:0x3f_ffff); bits[34]:0x0; bits[44]:0x7ff_ffff_ffff; bits[6]:0x1"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x12f_5fcd_ef36, bits[6]:0x3d, bits[52]:0x4, bits[22]:0x2a_aaaa); bits[34]:0x3_be65_37df; bits[44]:0xdda_6c40_be63; bits[6]:0x3f"
//     args: "bits[38]:0x0; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x8, bits[52]:0x5_5555_5555_5555, bits[22]:0x0); bits[34]:0x42c_0820; bits[44]:0x555_5555_5555; bits[6]:0x1"
//     args: "bits[38]:0x400; (bits[42]:0x28a_0481_718a, bits[6]:0x0, bits[52]:0xb_bc35_2918_8276, bits[22]:0x400); bits[34]:0x680_c630; bits[44]:0x0; bits[6]:0x15"
//     args: "bits[38]:0x20_0000; (bits[42]:0x53_8702_2905, bits[6]:0x0, bits[52]:0x8_8000_40b0_3ff5, bits[22]:0x1c_4640); bits[34]:0x0; bits[44]:0x41_0000_017f; bits[6]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1d5_758d_184c, bits[6]:0x15, bits[52]:0x7_d6ff_effe_c210, bits[22]:0x3f_ffff); bits[34]:0x0; bits[44]:0xc12_6f47_6f5d; bits[6]:0x2a"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x355_5d55_d71e, bits[6]:0x15, bits[52]:0x0, bits[22]:0xf_47f0); bits[34]:0x2_aaaa_aaaa; bits[44]:0xd0d_4036_ffe4; bits[6]:0x15"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x0, bits[6]:0x23, bits[52]:0xa_8aaa_aa3a_8210, bits[22]:0xb_abaa); bits[34]:0x2_aaaa_aaaa; bits[44]:0xaaa_a2aa_aabf; bits[6]:0x3c"
//     args: "bits[38]:0x20_e60a_a23e; (bits[42]:0x2000, bits[6]:0x0, bits[52]:0xb_3982_aacf_aeaa, bits[22]:0x1e_82cc); bits[34]:0xe70b_a03e; bits[44]:0x7ff_ffff_ffff; bits[6]:0x0"
//     args: "bits[38]:0x10_0000_0000; (bits[42]:0x8e_80b8_c09f, bits[6]:0x10, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x2_0000); bits[34]:0x1_5555_5555; bits[44]:0x39c_8b5a_9a23; bits[6]:0x23"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x1f, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x1f_ffff); bits[34]:0x2_8cb2_05a8; bits[44]:0x20_0000_0000; bits[6]:0x1f"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0xf0_b11a_9704, bits[6]:0x5, bits[52]:0xf_ffff_ffff_ffff, bits[22]:0x0); bits[34]:0x0; bits[44]:0x2aa; bits[6]:0x3f"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x3e4_9293_1caa, bits[6]:0x2a, bits[52]:0x6_b8b2_b2f2_7e6d, bits[22]:0x2e_cb30); bits[34]:0x1_ffff_ffff; bits[44]:0x603_14c0_08f5; bits[6]:0x19"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3fb_ffce_6ff6, bits[6]:0x1a, bits[52]:0x5_5555_5555_5555, bits[22]:0x4_0000); bits[34]:0x4_0000; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x1f"
//     args: "bits[38]:0xb_3a47_821c; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x1f, bits[52]:0xf_cccc_33c0_bdd2, bits[22]:0x3_827c); bits[34]:0x7cd3_4129; bits[44]:0x0; bits[6]:0x1"
//     args: "bits[38]:0x1; (bits[42]:0x2aa_aaaa_aaaa, bits[6]:0x5, bits[52]:0x4d6_e1e8_845d, bits[22]:0x15_5555); bits[34]:0x1_ffff_ffff; bits[44]:0xfff_ffff_ffff; bits[6]:0x2a"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x0, bits[52]:0xf_ffff_7fff_eaab, bits[22]:0x1_d7d1); bits[34]:0x3_fbff_ffbf; bits[44]:0xf2d_bf7e_f9f7; bits[6]:0xd"
//     args: "bits[38]:0x80; (bits[42]:0x1ff_ffff_ffff, bits[6]:0x0, bits[52]:0x5_5555_5555_5555, bits[22]:0x24_5030); bits[34]:0x14ac_1248; bits[44]:0x8_0020_2050; bits[6]:0x19"
//     args: "bits[38]:0x200; (bits[42]:0x41_8070_20c7, bits[6]:0x10, bits[52]:0xc01_028c_2ab3, bits[22]:0x6_0661); bits[34]:0x2_aaaa_aaaa; bits[44]:0x81b_026b_8869; bits[6]:0x2d"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3ff_ffff_bff0, bits[6]:0x4, bits[52]:0x3724_87f2_e7b0, bits[22]:0x3f_ffff); bits[34]:0x0; bits[44]:0x100; bits[6]:0x4"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x155_5555_5555, bits[6]:0x3f, bits[52]:0x5_5555_5555_5555, bits[22]:0x2b_64ff); bits[34]:0x2_aaaa_aaaa; bits[44]:0x7ff_ffff_ffff; bits[6]:0x3f"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x250_69bd_b573, bits[6]:0x15, bits[52]:0x4_5556_7cd5_d800, bits[22]:0x0); bits[34]:0x7311_aa70; bits[44]:0x555_5575_557e; bits[6]:0x30"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x1ff_ffff_ffff, bits[6]:0x2a, bits[52]:0xf_ffff_ffff_ffff, bits[22]:0x2a_aaaa); bits[34]:0x2_aaaa_aaaa; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x15"
//     args: "bits[38]:0x1000_0000; (bits[42]:0x2c1_0086_a82c, bits[6]:0x2a, bits[52]:0x0, bits[22]:0x1_4620); bits[34]:0x3_1002_a200; bits[44]:0x0; bits[6]:0x3f"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1_0000, bits[6]:0x1f, bits[52]:0x0, bits[22]:0x3b_dfff); bits[34]:0xd6e4_7ffe; bits[44]:0x77a_d1bf_f9ff; bits[6]:0x2"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x10_0000_0000, bits[6]:0x3a, bits[52]:0x1, bits[22]:0x2e_aa82); bits[34]:0x1_5555_5555; bits[44]:0x0; bits[6]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x9f_ffdd_dee5, bits[6]:0x3d, bits[52]:0xf_ffff_ffff_ffff, bits[22]:0x2a_aaaa); bits[34]:0x0; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x4"
//     args: "bits[38]:0x38_42dd_f9ee; (bits[42]:0x384_2ddf_9eea, bits[6]:0x15, bits[52]:0xa_53b7_ae7b_eee6, bits[22]:0x0); bits[34]:0x8; bits[44]:0x555_5555_5555; bits[6]:0x8"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x10, bits[52]:0xeced_1c0f_dfff, bits[22]:0x26_79f5); bits[34]:0x1_af8e_3c38; bits[44]:0xfff_ffff_ffff; bits[6]:0x2a"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x155_5555_5555, bits[6]:0xc, bits[52]:0x1_ed5c_8c85_a5a1, bits[22]:0x13_5d45); bits[34]:0x2_0000; bits[44]:0x2_89fa_9e9e; bits[6]:0x1e"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3ff_ffff_ffff, bits[6]:0x3f, bits[52]:0x8_8fe9_efe3_d4ed, bits[22]:0x3f_ffff); bits[34]:0x100; bits[44]:0xd8b_73af_dfd2; bits[6]:0x12"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x100, bits[6]:0x15, bits[52]:0x7_ffff_ffff_ffff, bits[22]:0x2a_aaaa); bits[34]:0x1_4927_5db3; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x31"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x155_5555_5555, bits[6]:0x0, bits[52]:0x6_83e7_2c86_6339, bits[22]:0x39_bff6); bits[34]:0x3_ffff_ffff; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x1f"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x175_4975_5454, bits[6]:0x11, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x25_1555); bits[34]:0x1_5554_50dc; bits[44]:0xfff_ffff_ffff; bits[6]:0x2a"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1ff_ffff_ffff, bits[6]:0x8, bits[52]:0x5_dfff_ffef_fdff, bits[22]:0x2f_ffff); bits[34]:0x2_aaaa_aaaa; bits[44]:0xfff_ffff_ffff; bits[6]:0x3"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x0, bits[6]:0x3f, bits[52]:0xa_b714_1350_d0cc, bits[22]:0x3b_7bda); bits[34]:0x3_ffff_ffff; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x3f"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x800, bits[6]:0x15, bits[52]:0x3_6c23_b76e_8ec3, bits[22]:0x40); bits[34]:0x3_77d7_efbe; bits[44]:0x74b_2ede_fdeb; bits[6]:0x3f"
//     args: "bits[38]:0x0; (bits[42]:0x2_004a, bits[6]:0x2a, bits[52]:0x2_0000_0000_0000, bits[22]:0x3f_ffff); bits[34]:0x2_f04c_7ae4; bits[44]:0x2000; bits[6]:0x24"
//     args: "bits[38]:0x2a_aaaa_aaaa; (bits[42]:0x292_9e7c_a828, bits[6]:0x3f, bits[52]:0x2000, bits[22]:0x2a_aaaa); bits[34]:0x2_aaaa_aaaa; bits[44]:0x555_5555_5555; bits[6]:0x0"
//     args: "bits[38]:0x15_5555_5555; (bits[42]:0x295_e753_8705, bits[6]:0x8, bits[52]:0xa_aaaa_aaaa_aaaa, bits[22]:0x1c_9d52); bits[34]:0x3_ffff_ffff; bits[44]:0x80_0000_0000; bits[6]:0x4"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x3c7_ffbd_effa, bits[6]:0x15, bits[52]:0x7_d069_a1c9_5021, bits[22]:0x2a_aaaa); bits[34]:0x1_5555_5555; bits[44]:0x7ff_ffff_ffff; bits[6]:0x0"
//     args: "bits[38]:0x3f_ffff_ffff; (bits[42]:0x1ff_cbc4_afcc, bits[6]:0x1f, bits[52]:0x8_c3ef_391c_1acf, bits[22]:0x2c_7bb7); bits[34]:0x0; bits[44]:0x7ff_ffff_ffff; bits[6]:0x0"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x1fb_feff_dbc4, bits[6]:0x2f, bits[52]:0x1_c9bb_cbde_6405, bits[22]:0x15_5555); bits[34]:0x3_f6b1_41ba; bits[44]:0xb9b_000e_e440; bits[6]:0x3a"
//     args: "bits[38]:0x1f_ffff_ffff; (bits[42]:0x155_5555_5555, bits[6]:0x1f, bits[52]:0x6_f3df_d96b_4055, bits[22]:0x0); bits[34]:0x2_aaaa_aaaa; bits[44]:0xaaa_aaaa_aaaa; bits[6]:0x0"
//   }
// }
//
// END_CONFIG
const W32_V12 = u32:0xc;
type x28 = u34;
fn x15(x16: u44, x17: u38, x18: u38, x19: u3, x20: u6) -> (bool, bool, bool, bool) {
    {
        let x21: bool = x19[2+:bool];
        (x21, x21, x21, x21)
    }
}
fn main(x0: u38, x1: (u42, u6, u52, u22), x2: u34, x3: u44, x4: u6) -> u21 {
    {
        let x5: (u21, u38, u6) = match x0 {
            u38:0x0 => (u21:0x0, u38:0x8000, x4),
            u38:250406246995 | u38:0x2f_e0c4_f13c => (u21:0xa_aaaa, x0, u6:0x3c),
            u38:0x800_0000 | u38:0x0..u38:0x2a_aaaa_aaaa => (u21:0x2_0000, u38:0x0, u6:0x1),
            u38:0x1f_ffff_ffff | u38:0x40_0000 => (u21:0x1f_ffff, u38:0x3f_42de_9e0b, u6:0x15),
            _ => (u21:0x1, u38:0x2a_aaaa_aaaa, x4),
        };
        let (x6, x7, x8) = match x0 {
            u38:0x0 => (u21:0x0, u38:0x8000, x4),
            u38:250406246995 | u38:0x2f_e0c4_f13c => (u21:0xa_aaaa, x0, u6:0x3c),
            u38:0x800_0000 | u38:0x0..u38:0x2a_aaaa_aaaa => (u21:0x2_0000, u38:0x0, u6:0x1),
            u38:0x1f_ffff_ffff | u38:0x40_0000 => (u21:0x1f_ffff, u38:0x3f_42de_9e0b, u6:0x15),
            _ => (u21:0x1, u38:0x2a_aaaa_aaaa, x4),
        };
        let x9: u6 = -x8;
        let x10: u34 = x2[x6+:u34];
        let x11: bool = x6[0+:bool];
        let x12: u3 = x9[:3];
        let x13: u38 = x0[0+:u38];
        let x14: u44 = !x3;
        let x22: (bool, bool, bool, bool) = x15(x3, x13, x13, x12, x9);
        let x23: u6 = x4 | x6 as u6;
        let x25: u38 = {
            let x24: (u38, u38) = umulp(x2 as u38, x0);
            x24.0 + x24.1
        };
        let x26: () = ();
        let x27: u3 = signex(x8, x12);
        let x29: x28[W32_V12] = match x14 {
            u44:0xaaa_aaaa_aaaa => [u34:0x2_aaaa_aaaa, x10, x10, u34:5726623061, u34:0x100, u34:0x1_5555_5555, xN[bool:0x0][34]:0x3_ffff_ffff, x2, x10, x10, u34:0x7bfc_243c, x2],
            _ => [u34:0x3_ffff_ffff, u34:0x1_ffff_ffff, u34:0x20, u34:0x1_5555_5555, x2, u34:0, u34:0x3_ffff_ffff, x10, u34:0x3_ffff_ffff, x10, u34:0x3_ffff_ffff, xN[bool:0x0][34]:0x2_4a3b_c7c3],
        };
        let x30: u4 = x4[:4];
        let x31: bool = one_hot_sel(x8, [x11, x11, x11, x11, x11, x11]);
        let x32: bool = x12 as bool + x11;
        let x33: u6 = -x9;
        let x34: u3 = bit_slice_update(x27, x30, x8);
        let x35: u8 = x10[x6+:u8];
        let x36: u18 = x12 ++ x35 ++ x33 ++ x31;
        let x37: u7 = x35[1+:u7];
        let x38: u23 = x25[x33+:u23];
        let x39: bool = x31 & x11;
        let x40: u21 = x25 as u21 ^ x6;
        x6
    }
}
