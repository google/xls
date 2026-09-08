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
// exception: "Subprocess call timed out after 1500 seconds: /xls/tools/opt_main sample.ir --logtostderr"
// issue: "https://github.com/google/xls/issues/4411"
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
//   codegen_args: "--generator=combinational"
//   codegen_args: "--reset_data_path=false"
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
//   disable_unopt_interpreter: true
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0xae_a8c2, bits[24]:0x5a_a612, bits[24]:0xff_ffff, bits[24]:0x26_aaa2, bits[24]:0xaa_aaab, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0xfa_d2a9]; bits[45]:0x1555_5555_5555; bits[2]:0x2"
//     args: "bits[26]:0x8000; [bits[24]:0x55_5555, bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0x10_8000, bits[24]:0xaa_aaaa, bits[24]:0x10_8524, bits[24]:0x84_9003, bits[24]:0x55_5555]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x1000; [bits[24]:0x80_d004, bits[24]:0xff_ffff, bits[24]:0x7f_ffff, bits[24]:0xe0_00b2, bits[24]:0x7f_ffff, bits[24]:0x5804, bits[24]:0xff_ffff, bits[24]:0xc0_1f00]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x0"
//     args: "bits[26]:0x155_5555; [bits[24]:0x75_564d, bits[24]:0x55_5c71, bits[24]:0x5d_5547, bits[24]:0x55_5745, bits[24]:0x24_5dc7, bits[24]:0xc4_f044, bits[24]:0x55_5555, bits[24]:0xf6_ee3e]; bits[45]:0x9aa_aaab_aaee; bits[2]:0x3"
//     args: "bits[26]:0x0; [bits[24]:0x55_5555, bits[24]:0x55_5555, bits[24]:0x5e_0140, bits[24]:0xaa_aaaa, bits[24]:0xff_ffff, bits[24]:0x4e40, bits[24]:0xaa_aaaa, bits[24]:0x55_5555]; bits[45]:0xfff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x55_5555, bits[24]:0xff_ffff, bits[24]:0x80, bits[24]:0xfd_77f9, bits[24]:0x7f_ffff, bits[24]:0xff_ffff, bits[24]:0xe6_ff88, bits[24]:0x12_84ef]; bits[45]:0x1fff_fffd_5555; bits[2]:0x2"
//     args: "bits[26]:0x0; [bits[24]:0xc0_0003, bits[24]:0x13_a6aa, bits[24]:0xfa_a30e, bits[24]:0x40_90a2, bits[24]:0x55_6d78, bits[24]:0xd_0000, bits[24]:0x20_0400, bits[24]:0x75_8822]; bits[45]:0x20_f2d5_c42f; bits[2]:0x0"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xff_ffbf, bits[24]:0xfd_7fff, bits[24]:0x55_5555, bits[24]:0xaa_aaaa, bits[24]:0xfb_ffff, bits[24]:0xff_7eff, bits[24]:0xaa_aaaa, bits[24]:0x7c_3777]; bits[45]:0x0; bits[2]:0x3"
//     args: "bits[26]:0x0; [bits[24]:0x9c_5a45, bits[24]:0x8100, bits[24]:0x30_00c2, bits[24]:0x6_0f04, bits[24]:0x6_2410, bits[24]:0x12_40bc, bits[24]:0x84_62a8, bits[24]:0x55_5555]; bits[45]:0x120_a200_c4a7; bits[2]:0x1"
//     args: "bits[26]:0x155_5555; [bits[24]:0x0, bits[24]:0x49_14d1, bits[24]:0xd7_6155, bits[24]:0x7f_ffff, bits[24]:0x32_1470, bits[24]:0x0, bits[24]:0x55_5555, bits[24]:0x5d_f375]; bits[45]:0x1d6f_0697_8ad1; bits[2]:0x1"
//     args: "bits[26]:0x155_5555; [bits[24]:0x1d_4601, bits[24]:0xff_ffff, bits[24]:0xdb_547d, bits[24]:0x0, bits[24]:0xaa_aaaa, bits[24]:0x45_5d55, bits[24]:0x50_5d5d, bits[24]:0x55_5551]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xaa_aaaa, bits[24]:0xf5_ce55, bits[24]:0xaa_aaaa, bits[24]:0xdf_6675, bits[24]:0xff_ffff, bits[24]:0xca_2c9f, bits[24]:0xff_3f7f, bits[24]:0x7f_ffff]; bits[45]:0x1555_5555_5555; bits[2]:0x2"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0x3f_726a, bits[24]:0xff_ffff, bits[24]:0x400, bits[24]:0xbb_759d, bits[24]:0xfb_79f7, bits[24]:0xfb_cfff]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x2"
//     args: "bits[26]:0x11c_110a; [bits[24]:0x3c_3340, bits[24]:0x1e_3542, bits[24]:0x5e_9149, bits[24]:0x1c_110a, bits[24]:0x1c_150b, bits[24]:0x89_3378, bits[24]:0x6c_2398, bits[24]:0x55_5555]; bits[45]:0xe13_24ea_16dc; bits[2]:0x3"
//     args: "bits[26]:0x155_5555; [bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0x55_5555, bits[24]:0x7f_ffff, bits[24]:0x1000, bits[24]:0xd5_73dd, bits[24]:0x55_5555, bits[24]:0x7f_ffff]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x2"
//     args: "bits[26]:0x40; [bits[24]:0xaa_aaaa, bits[24]:0xa0_404a, bits[24]:0x55_5555, bits[24]:0x40, bits[24]:0xaa_aaaa, bits[24]:0xd_c5c0, bits[24]:0x1a10, bits[24]:0x45_00e0]; bits[45]:0x300_c000; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xb5_ffff, bits[24]:0x8f_e7df, bits[24]:0xff_f7df, bits[24]:0x0, bits[24]:0xff_dfff, bits[24]:0x800, bits[24]:0xff_3fde, bits[24]:0xfa_ea7e]; bits[45]:0xdff_ffb8_020a; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x21_abdd, bits[24]:0x14_ef85, bits[24]:0xf3_edfd, bits[24]:0xff_ffff, bits[24]:0xe9_1db3, bits[24]:0xdf_8fee, bits[24]:0x7f_ffff, bits[24]:0x3e_9f7d]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x0"
//     args: "bits[26]:0x155_5555; [bits[24]:0x54_555d, bits[24]:0xfd_555d, bits[24]:0x65_6545, bits[24]:0x0, bits[24]:0x40_0000, bits[24]:0x55_5555, bits[24]:0x62_d9d7, bits[24]:0x1d_cb82]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x80_0000; [bits[24]:0x55_5555, bits[24]:0x0, bits[24]:0xa0_0060, bits[24]:0xc0_0850, bits[24]:0x3e_04ca, bits[24]:0x81_0002, bits[24]:0xca_45ad, bits[24]:0x7f_ffff]; bits[45]:0x1555_5555_5555; bits[2]:0x2"
//     args: "bits[26]:0x155_5555; [bits[24]:0x55_5575, bits[24]:0xd1_dd05, bits[24]:0xaa_aaaa, bits[24]:0x0, bits[24]:0x0, bits[24]:0x45_ddfd, bits[24]:0x3e_f0bc, bits[24]:0x5c_5275]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x0"
//     args: "bits[26]:0x155_5555; [bits[24]:0xff_ffff, bits[24]:0x0, bits[24]:0x55_d575, bits[24]:0x15_1d63, bits[24]:0xc7_3d5d, bits[24]:0x7f_ffff, bits[24]:0xff_ffff, bits[24]:0x54_7454]; bits[45]:0x0; bits[2]:0x0"
//     args: "bits[26]:0x0; [bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0x10, bits[24]:0x0, bits[24]:0x1_8200, bits[24]:0xff_ffff, bits[24]:0x0, bits[24]:0x60_ab0a]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x0; [bits[24]:0x8_0018, bits[24]:0x400, bits[24]:0x7f_ffff, bits[24]:0x90_4113, bits[24]:0x3_4201, bits[24]:0x40_0004, bits[24]:0x0, bits[24]:0x800]; bits[45]:0x1362_10b6_707e; bits[2]:0x1"
//     args: "bits[26]:0x4; [bits[24]:0xda_7b94, bits[24]:0x1000, bits[24]:0x92_7d84, bits[24]:0xb024, bits[24]:0x40_0030, bits[24]:0x2000, bits[24]:0xd5_4c48, bits[24]:0xff_ffff]; bits[45]:0x1ac_5380_3fe6; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xbf_dbed, bits[24]:0x6f_1a78, bits[24]:0x800, bits[24]:0x6d_edaf, bits[24]:0x3f_7647, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0xff_feff]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x4; [bits[24]:0x80_0000, bits[24]:0x0, bits[24]:0x7f_ffff, bits[24]:0x38_4021, bits[24]:0x4, bits[24]:0x0, bits[24]:0x7f_ffff, bits[24]:0x2]; bits[45]:0x1a52_a23d_7306; bits[2]:0x1"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xff_ffff, bits[24]:0x55_5555, bits[24]:0xaa_aaaa, bits[24]:0x0, bits[24]:0x4_0000, bits[24]:0xaa_aaaa, bits[24]:0xff_ffbf, bits[24]:0xef_edff]; bits[45]:0x1; bits[2]:0x3"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0xaa_aaaa, bits[24]:0xa2_2e6a, bits[24]:0x40, bits[24]:0xaa_a815, bits[24]:0x0, bits[24]:0x55_5555, bits[24]:0x9_e827, bits[24]:0x9f_8818]; bits[45]:0x1814_18df_70a9; bits[2]:0x0"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x100, bits[24]:0xff_ffff, bits[24]:0x0, bits[24]:0xde_aff7, bits[24]:0xaa_aaaa, bits[24]:0xfe_df97, bits[24]:0x4e_26f9, bits[24]:0xcf_efbc]; bits[45]:0x1555_5555_5555; bits[2]:0x1"
//     args: "bits[26]:0x100_0000; [bits[24]:0x4_5000, bits[24]:0x0, bits[24]:0x20_4800, bits[24]:0x24_0024, bits[24]:0x55_5555, bits[24]:0xe4_9177, bits[24]:0xc8_0224, bits[24]:0x7f_ffff]; bits[45]:0x800_0000; bits[2]:0x1"
//     args: "bits[26]:0x18e_8872; [bits[24]:0x8e_8872, bits[24]:0x55_5555, bits[24]:0x8a_835a, bits[24]:0x86_c872, bits[24]:0xd2_e872, bits[24]:0x6a_423f, bits[24]:0x55_5555, bits[24]:0xe_81f2]; bits[45]:0x840_6191_3bf1; bits[2]:0x0"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x7f_ffff, bits[24]:0x7_df99, bits[24]:0xb3_fbf7, bits[24]:0xff_ffff, bits[24]:0xff_effd, bits[24]:0x55_5555, bits[24]:0x0, bits[24]:0x4000]; bits[45]:0x1c7e_ff7b_fffb; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xff_30ff, bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0x55_5555, bits[24]:0x7f_ff7f, bits[24]:0xbd_fffc, bits[24]:0x80, bits[24]:0xff_ffff]; bits[45]:0x1555_5555_5555; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xd7_5e2b, bits[24]:0xff_fdff, bits[24]:0xfe_ffff, bits[24]:0xf7_bedc, bits[24]:0xb2_bfd2, bits[24]:0x9f_a28f, bits[24]:0xfd_5e3f, bits[24]:0xff_ffff]; bits[45]:0x1555_5555_5555; bits[2]:0x1"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0x0, bits[24]:0xef_03c5, bits[24]:0x5a_3cff, bits[24]:0xa7_f1e5, bits[24]:0xff_aaff, bits[24]:0xf7_ffef]; bits[45]:0x1ebb_b530_d158; bits[2]:0x2"
//     args: "bits[26]:0x80; [bits[24]:0x90a0, bits[24]:0x8d_428a, bits[24]:0xff_ffff, bits[24]:0x12_02e0, bits[24]:0x55_5555, bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0xaa_aaaa]; bits[45]:0xac9_7f2f_ccf6; bits[2]:0x1"
//     args: "bits[26]:0x267_a6c8; [bits[24]:0x7f_ffff, bits[24]:0x55_5555, bits[24]:0xe7_26c8, bits[24]:0x4f_ba98, bits[24]:0xef_2fd9, bits[24]:0xff_ffff, bits[24]:0x7f_ffff, bits[24]:0x7f_ffff]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xaa_e4a1, bits[24]:0xff_f4ff, bits[24]:0x7f_ffff, bits[24]:0xff_ffff, bits[24]:0xaa_aaaa, bits[24]:0xaa_aaaa, bits[24]:0x0, bits[24]:0xff_bf7d]; bits[45]:0x11d1_857c_ad69; bits[2]:0x1"
//     args: "bits[26]:0x4000; [bits[24]:0xaa_aaaa, bits[24]:0x40_4100, bits[24]:0x55_5555, bits[24]:0x0, bits[24]:0x1, bits[24]:0x20_c828, bits[24]:0x2_3820, bits[24]:0x40_4010]; bits[45]:0xfff_ffff_ffff; bits[2]:0x2"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xc2_8d5f, bits[24]:0xfb_ffff, bits[24]:0xb6_896f, bits[24]:0xff_ffff, bits[24]:0x7f_ffff, bits[24]:0xf7_feff, bits[24]:0x55_5555, bits[24]:0x9a_ffdd]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x40_0000; [bits[24]:0x55_5555, bits[24]:0x7f_ffff, bits[24]:0x40_0000, bits[24]:0x7f_ffff, bits[24]:0x40_3c0d, bits[24]:0x4c_0000, bits[24]:0xda_9005, bits[24]:0x7f_ffff]; bits[45]:0xfff_ffff_ffff; bits[2]:0x1"
//     args: "bits[26]:0x12d_61f5; [bits[24]:0x0, bits[24]:0xfd_71f6, bits[24]:0x14_23f1, bits[24]:0xd1_b55b, bits[24]:0xa9_0450, bits[24]:0x55_5555, bits[24]:0x55_5555, bits[24]:0xaf_6bf4]; bits[45]:0x1a7a_d3a6_ca68; bits[2]:0x3"
//     args: "bits[26]:0x155_5555; [bits[24]:0xd7_6248, bits[24]:0x8c_51ba, bits[24]:0x23_ff38, bits[24]:0x55_5555, bits[24]:0x57_54d5, bits[24]:0xe1_d8a7, bits[24]:0x0, bits[24]:0xff_ffff]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x3d7_cb76; [bits[24]:0x1f_48f2, bits[24]:0x0, bits[24]:0x7f_ffff, bits[24]:0xd9_887c, bits[24]:0x17_8b56, bits[24]:0x800, bits[24]:0xd3_c3f6, bits[24]:0x55_5555]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xc6_fbff, bits[24]:0xae_fdfe, bits[24]:0xcf_f34e, bits[24]:0xfd_df3f, bits[24]:0xef_ffe5, bits[24]:0x55_5555, bits[24]:0x7f_fbbf, bits[24]:0x9f_c68f]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x3"
//     args: "bits[26]:0x0; [bits[24]:0x0, bits[24]:0xa_e0a2, bits[24]:0x44_1419, bits[24]:0xff_ffff, bits[24]:0x0, bits[24]:0x7f_ffff, bits[24]:0xf9_600c, bits[24]:0xa5_3b08]; bits[45]:0xfff_ffff_ffff; bits[2]:0x2"
//     args: "bits[26]:0x0; [bits[24]:0x4000, bits[24]:0x8260, bits[24]:0x1_0000, bits[24]:0xa6_5004, bits[24]:0xed_b1bc, bits[24]:0xb9_04a8, bits[24]:0x40_0eac, bits[24]:0x7f_ffff]; bits[45]:0x1555_5555_5555; bits[2]:0x2"
//     args: "bits[26]:0x0; [bits[24]:0x55_5555, bits[24]:0x1_cb40, bits[24]:0x6_c04f, bits[24]:0xd1_470f, bits[24]:0x55_5555, bits[24]:0x20_8040, bits[24]:0x60_614a, bits[24]:0xff_ffff]; bits[45]:0xc8a_4440_8004; bits[2]:0x2"
//     args: "bits[26]:0x155_5555; [bits[24]:0x0, bits[24]:0x7f_ffff, bits[24]:0x97_5154, bits[24]:0xaa_aaaa, bits[24]:0xa7_57d7, bits[24]:0x0, bits[24]:0x0, bits[24]:0x55_5574]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x7f_ffff, bits[24]:0x7f_ffff, bits[24]:0xff_ffff, bits[24]:0xaa_aaaa, bits[24]:0xe8_28a2, bits[24]:0xaa_aaaa, bits[24]:0x0, bits[24]:0xdc_3664]; bits[45]:0x1571_4552_aaba; bits[2]:0x2"
//     args: "bits[26]:0x2ea_6c01; [bits[24]:0x55_5555, bits[24]:0xfe_a886, bits[24]:0x6e_e842, bits[24]:0xe8_6c00, bits[24]:0xe6_f805, bits[24]:0x55_5555, bits[24]:0xea_6060, bits[24]:0xa2_8640]; bits[45]:0xfff_ffff_ffff; bits[2]:0x1"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x7f_ffff, bits[24]:0xff_7def, bits[24]:0x55_5555, bits[24]:0xed_feff, bits[24]:0x55_5555, bits[24]:0xff_ffff, bits[24]:0xdf_7f91, bits[24]:0x55_5555]; bits[45]:0x80_0000; bits[2]:0x0"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x7f_ffff, bits[24]:0x8a_8ca2, bits[24]:0xaa_aaaa, bits[24]:0x55_5555, bits[24]:0x2a_cabb, bits[24]:0x7f_ffff, bits[24]:0x7f_ffff, bits[24]:0x1_0000]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x8d_4054, bits[24]:0xff_fff7, bits[24]:0xf7_fef7, bits[24]:0xc2_cfbb, bits[24]:0x68_472f, bits[24]:0xe3_9f17, bits[24]:0x7f_e7bf, bits[24]:0x2_0000]; bits[45]:0x1d5a_aaf8_3f7c; bits[2]:0x3"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x7f_ffff, bits[24]:0xd_ab8f, bits[24]:0xaa_aaa2, bits[24]:0xaa_aaaa, bits[24]:0x1a_aaa8, bits[24]:0xaa_a8aa, bits[24]:0xff_ffff, bits[24]:0x2_b8a2]; bits[45]:0x1d55_5557_ffff; bits[2]:0x1"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x4000, bits[24]:0xbd_efff, bits[24]:0xff_ffff, bits[24]:0x0, bits[24]:0x79_f0f7, bits[24]:0xdf_7fff, bits[24]:0x7f_ffef, bits[24]:0xff_ffff]; bits[45]:0x1e9f_fffb_1b9f; bits[2]:0x2"
//     args: "bits[26]:0x9b_ee37; [bits[24]:0x17_ee33, bits[24]:0x40_0000, bits[24]:0xbb_ee3d, bits[24]:0x9b_ee37, bits[24]:0xaa_aaaa, bits[24]:0x9f_ee33, bits[24]:0x100, bits[24]:0x200]; bits[45]:0xcd7_75b8_6411; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x73_a737, bits[24]:0x7f_ffff, bits[24]:0xdc_dfe8, bits[24]:0xb2_f1ea, bits[24]:0x6c_a95e, bits[24]:0x7f_ffff, bits[24]:0x77_df6f, bits[24]:0x55_5555]; bits[45]:0x800; bits[2]:0x2"
//     args: "bits[26]:0x1_0000; [bits[24]:0x81_1001, bits[24]:0x7f_ffff, bits[24]:0x85_6250, bits[24]:0x1_0000, bits[24]:0x1_122e, bits[24]:0x11_4000, bits[24]:0x21_4c00, bits[24]:0xaa_aaaa]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xaa_aaaa, bits[24]:0xbd_34cb, bits[24]:0x6f_fafe, bits[24]:0xff_ffff, bits[24]:0xff_ffef, bits[24]:0xaa_aaaa, bits[24]:0xff_ffff, bits[24]:0x1]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x3"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x55_5555, bits[24]:0x55_5555, bits[24]:0xf0_bfcf, bits[24]:0x76_7fa9, bits[24]:0x0, bits[24]:0xff_fdff, bits[24]:0x2_0000, bits[24]:0xff_ffff]; bits[45]:0x1a36_4170_d02b; bits[2]:0x0"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xf7_eafd, bits[24]:0x57_f7ff, bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0xaa_aaaa, bits[24]:0x55_5555, bits[24]:0x0, bits[24]:0xd7_8bbe]; bits[45]:0x80; bits[2]:0x2"
//     args: "bits[26]:0x8_0000; [bits[24]:0x7f_ffff, bits[24]:0x0, bits[24]:0xc8_0000, bits[24]:0x200, bits[24]:0x8_9800, bits[24]:0x55_5555, bits[24]:0x20_0984, bits[24]:0x7f_ffff]; bits[45]:0xfff_ffff_ffff; bits[2]:0x2"
//     args: "bits[26]:0x1b4_1dcb; [bits[24]:0xbc_5cc3, bits[24]:0x7f_a94b, bits[24]:0x34_bbcb, bits[24]:0x55_5555, bits[24]:0xb6_0bcb, bits[24]:0xbd_1c88, bits[24]:0xf0_3ec2, bits[24]:0xa1_0dc2]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x1"
//     args: "bits[26]:0x349_f231; [bits[24]:0x49_f231, bits[24]:0x49_f231, bits[24]:0x43_b2e3, bits[24]:0xe9_fa1d, bits[24]:0x6c_ff7b, bits[24]:0x79_7230, bits[24]:0x55_5555, bits[24]:0x7f_ffff]; bits[45]:0x4000; bits[2]:0x0"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xdf_7dff, bits[24]:0xff_ffff, bits[24]:0x7b_7fe4, bits[24]:0xff_7bfc, bits[24]:0xf0_d673, bits[24]:0x2f_4cec, bits[24]:0x7f_ffff, bits[24]:0x55_5555]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x7f_ff7f, bits[24]:0xff_7fff, bits[24]:0xe7_ffff, bits[24]:0x80, bits[24]:0xbc_7f7a, bits[24]:0xfa_fff7, bits[24]:0x83_70a5, bits[24]:0x0]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xff_d7fb, bits[24]:0xf7_ffff, bits[24]:0x7f_ffff, bits[24]:0xc4_2bf6, bits[24]:0xfb_ffef, bits[24]:0xce_5fb7, bits[24]:0xad_bd50, bits[24]:0x7f_ffff]; bits[45]:0x31f_a938_8910; bits[2]:0x3"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x7f_9fef, bits[24]:0xf_fcd7, bits[24]:0xff_ffff, bits[24]:0xf2_7f55, bits[24]:0x55_5555, bits[24]:0xff_ffbe, bits[24]:0x8b_5416, bits[24]:0x7f_ffff]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x1"
//     args: "bits[26]:0x2; [bits[24]:0x0, bits[24]:0x41_c202, bits[24]:0x42_4996, bits[24]:0x7f_ffff, bits[24]:0xaa_aaaa, bits[24]:0xd5_b9ea, bits[24]:0x55_5555, bits[24]:0xf2_e402]; bits[45]:0x816_ffff; bits[2]:0x3"
//     args: "bits[26]:0xd_9170; [bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0xf_9170, bits[24]:0xea_b321, bits[24]:0xcf_9ff2, bits[24]:0x7f_ffff, bits[24]:0x7f_ffff, bits[24]:0x0]; bits[45]:0x1555_5555_5555; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xff_ffff, bits[24]:0xfb_653e, bits[24]:0xff_ffff, bits[24]:0xaa_aaaa, bits[24]:0xaa_aaaa, bits[24]:0xfd_e7df, bits[24]:0xaa_aaaa, bits[24]:0x7f_ffff]; bits[45]:0x0; bits[2]:0x0"
//     args: "bits[26]:0x100_0000; [bits[24]:0x0, bits[24]:0x20_88ef, bits[24]:0xff_ffff, bits[24]:0x2_8000, bits[24]:0x0, bits[24]:0x1000, bits[24]:0xcf_1e6c, bits[24]:0x48_4806]; bits[45]:0x11f2_0c6e_0cd9; bits[2]:0x1"
//     args: "bits[26]:0x155_5555; [bits[24]:0x51_5267, bits[24]:0x3e_d43d, bits[24]:0x46_3d44, bits[24]:0x0, bits[24]:0x85_895e, bits[24]:0x55_55e5, bits[24]:0x14_70d4, bits[24]:0x57_2495]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x55_5555, bits[24]:0xff_ffff, bits[24]:0x5a_bd7f, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0x6e_fdaf]; bits[45]:0x13ed_677f_8140; bits[2]:0x0"
//     args: "bits[26]:0x4_0000; [bits[24]:0xff_ffff, bits[24]:0x6_8a80, bits[24]:0x6f_b12c, bits[24]:0x3_302c, bits[24]:0xd0_f747, bits[24]:0x0, bits[24]:0x4_0424, bits[24]:0xf3_b909]; bits[45]:0x1555_5555_5555; bits[2]:0x2"
//     args: "bits[26]:0xb8_f58d; [bits[24]:0x0, bits[24]:0x55_5555, bits[24]:0xaa_f58d, bits[24]:0x0, bits[24]:0xb8_f58d, bits[24]:0xff_ffff, bits[24]:0x3a_f58d, bits[24]:0x55_5555]; bits[45]:0xfff_ffff_ffff; bits[2]:0x1"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x7f_ffff, bits[24]:0x0, bits[24]:0x200, bits[24]:0xaa_aaaa, bits[24]:0x40, bits[24]:0x22_272a, bits[24]:0x2_0000, bits[24]:0xab_aeaa]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x1"
//     args: "bits[26]:0x0; [bits[24]:0x4346, bits[24]:0xa0_4202, bits[24]:0x40_0000, bits[24]:0x0, bits[24]:0x41_0513, bits[24]:0xe8_3bfd, bits[24]:0x0, bits[24]:0xf0_4004]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x155_5555; [bits[24]:0xff_ffff, bits[24]:0x41_5100, bits[24]:0xcc_575d, bits[24]:0xff_ffff, bits[24]:0x54_5455, bits[24]:0x55_5555, bits[24]:0xaa_aaaa, bits[24]:0x57_14d1]; bits[45]:0x1a5d_a627_596b; bits[2]:0x1"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0xaa_aaaa, bits[24]:0xaa_aaaa, bits[24]:0x8a_aa8a, bits[24]:0x0, bits[24]:0x7f_ffff, bits[24]:0x55_5555, bits[24]:0x8c_9c8c, bits[24]:0x9e_5de3]; bits[45]:0x144c_bc2a_cc56; bits[2]:0x3"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xff_ffff, bits[24]:0xff_ecff, bits[24]:0xff_ffff, bits[24]:0xd1_ad8d, bits[24]:0x55_5555, bits[24]:0xf3_e5bd, bits[24]:0x6a_de2e, bits[24]:0x18_77df]; bits[45]:0x1555_5555_5555; bits[2]:0x1"
//     args: "bits[26]:0x0; [bits[24]:0x7f_ffff, bits[24]:0x4, bits[24]:0x0, bits[24]:0x0, bits[24]:0x80, bits[24]:0x6_4900, bits[24]:0x80_0010, bits[24]:0x40]; bits[45]:0x0; bits[2]:0x0"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x8a_aeae, bits[24]:0xa2_0ae2, bits[24]:0xb8_82aa, bits[24]:0x7f_ffff, bits[24]:0xa2_8e29, bits[24]:0xaa_a1aa, bits[24]:0x70_1329, bits[24]:0xaa_aaaa]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x0; [bits[24]:0xaa_aaaa, bits[24]:0x48_0080, bits[24]:0xaa_aaaa, bits[24]:0x55_5555, bits[24]:0x8_0800, bits[24]:0xaa_aaaa, bits[24]:0x55_5555, bits[24]:0x14_18eb]; bits[45]:0xfff_ffff_ffff; bits[2]:0x2"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xaa_aaaa, bits[24]:0x7f_ffff, bits[24]:0x0, bits[24]:0xc7_37fa, bits[24]:0x8_0000, bits[24]:0x41_6cb8, bits[24]:0x4_7543, bits[24]:0xff_ffff]; bits[45]:0x73f_f7d8_b0d0; bits[2]:0x1"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xff_ffff, bits[24]:0xf6_fcfe, bits[24]:0x8000, bits[24]:0xff_ffff, bits[24]:0x7f_ffff, bits[24]:0x20_0000, bits[24]:0xcd_fcfe, bits[24]:0x1_1757]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x3"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x7f_ffff, bits[24]:0xe7_cf7d, bits[24]:0x0, bits[24]:0xaa_aaaa, bits[24]:0xfc_f5df, bits[24]:0xde_3e75, bits[24]:0xff_f5ff, bits[24]:0xfd_fff7]; bits[45]:0x80; bits[2]:0x0"
//     args: "bits[26]:0x223_8bb3; [bits[24]:0xa3_9bbb, bits[24]:0x200, bits[24]:0xa3_89ba, bits[24]:0x8_0000, bits[24]:0x6b_bb23, bits[24]:0xec_1762, bits[24]:0x0, bits[24]:0x3_8bb3]; bits[45]:0xfff_ffff_ffff; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xf1_9f3d, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0x61_5b64, bits[24]:0xff_bdfe, bits[24]:0x7f_fffd, bits[24]:0x7e_3d5b, bits[24]:0x7f_57ff]; bits[45]:0x1555_5555_5555; bits[2]:0x0"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x8, bits[24]:0x7a_c3f8, bits[24]:0xaa_b2ba, bits[24]:0xb9_9af6, bits[24]:0xe2_4eba, bits[24]:0xaa_aaaa, bits[24]:0xe_822e, bits[24]:0xff_ffff]; bits[45]:0x1555_5550_0002; bits[2]:0x2"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x8000, bits[24]:0xb_b933, bits[24]:0x4000, bits[24]:0xff_ffff, bits[24]:0xa8_8ba0, bits[24]:0xbb_aaa8, bits[24]:0x84_6a8b, bits[24]:0xba_aafe]; bits[45]:0x17d9_d752_edee; bits[2]:0x2"
//     args: "bits[26]:0x80; [bits[24]:0x88_a087, bits[24]:0x480, bits[24]:0x8b_9380, bits[24]:0xea_58c4, bits[24]:0x55_5555, bits[24]:0x4_0180, bits[24]:0xff_ffff, bits[24]:0x59_0083]; bits[45]:0x8; bits[2]:0x3"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0xaa_a7a8, bits[24]:0xff_ffff, bits[24]:0x4000, bits[24]:0x2a_eaae, bits[24]:0x40_0000, bits[24]:0xdc_a6c2, bits[24]:0xc2_0bb5, bits[24]:0x8a_a2b1]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x2"
//     args: "bits[26]:0x100; [bits[24]:0x40_a580, bits[24]:0x37_a5b3, bits[24]:0x1_0004, bits[24]:0x7f_ffff, bits[24]:0x2_010b, bits[24]:0xe3_0904, bits[24]:0xaa_aaaa, bits[24]:0x0]; bits[45]:0xfff_ffff_ffff; bits[2]:0x2"
//     args: "bits[26]:0x224_8b05; [bits[24]:0xff_ffff, bits[24]:0x32_a02e, bits[24]:0xaa_aaaa, bits[24]:0x7_0bb5, bits[24]:0x56_8ac9, bits[24]:0x66_cfa5, bits[24]:0x24_9b87, bits[24]:0x20_4864]; bits[45]:0xfff_ffff_ffff; bits[2]:0x1"
//     args: "bits[26]:0x0; [bits[24]:0x1_0407, bits[24]:0xff_ffff, bits[24]:0x10_30a0, bits[24]:0x1_1400, bits[24]:0x50_a821, bits[24]:0x7f_ffff, bits[24]:0x24_81d3, bits[24]:0xff_ffff]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x2"
//     args: "bits[26]:0x2; [bits[24]:0x7f_ffff, bits[24]:0x98_4469, bits[24]:0x20_d242, bits[24]:0x2802, bits[24]:0x102, bits[24]:0x102, bits[24]:0x800, bits[24]:0x8]; bits[45]:0x1b7_2412_b59d; bits[2]:0x3"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0xda_8abe, bits[24]:0xe2_a92a, bits[24]:0x45_a596, bits[24]:0x8, bits[24]:0xff_ffff, bits[24]:0xaa_eaa8, bits[24]:0x58_15e4, bits[24]:0xab_2b2a]; bits[45]:0x1000_0000; bits[2]:0x0"
//     args: "bits[26]:0x0; [bits[24]:0x0, bits[24]:0x7f_ffff, bits[24]:0x55_5555, bits[24]:0xaa_aaaa, bits[24]:0x3_0038, bits[24]:0x8e_3cf3, bits[24]:0xff_ffff, bits[24]:0x40_4231]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xaa_aaaa, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0xff_ffff, bits[24]:0xfb_aeff, bits[24]:0x7f_5ffa, bits[24]:0xcf_feef, bits[24]:0x0]; bits[45]:0xfff_ffff_ffff; bits[2]:0x0"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xfb_e5ef, bits[24]:0x40_0000, bits[24]:0xaa_aaaa, bits[24]:0x7f_ffff, bits[24]:0xff_ffff, bits[24]:0x7e_ffdb, bits[24]:0x55_5555, bits[24]:0xff_ffff]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x2"
//     args: "bits[26]:0x80; [bits[24]:0x2_008b, bits[24]:0x8_0080, bits[24]:0x484, bits[24]:0x8c_6010, bits[24]:0x42_0082, bits[24]:0x21_4ce9, bits[24]:0x0, bits[24]:0x26_1484]; bits[45]:0xaaa_aaaa_aaaa; bits[2]:0x3"
//     args: "bits[26]:0x23e_1b3e; [bits[24]:0x3c_ca2c, bits[24]:0x31_11bc, bits[24]:0xaa_aaaa, bits[24]:0xff_ffff, bits[24]:0x2000, bits[24]:0x18_39aa, bits[24]:0x1f_74c1, bits[24]:0x80]; bits[45]:0xeb9_0985_9245; bits[2]:0x2"
//     args: "bits[26]:0x0; [bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0x7f_ffff, bits[24]:0xe_8681, bits[24]:0x2_0000, bits[24]:0x8_0420, bits[24]:0x2c_5a55, bits[24]:0xaa_aaaa]; bits[45]:0xe0d_bf43_451e; bits[2]:0x2"
//     args: "bits[26]:0x0; [bits[24]:0x82_2400, bits[24]:0x0, bits[24]:0xaa_aaaa, bits[24]:0x84_2028, bits[24]:0x21, bits[24]:0x8, bits[24]:0xaa_aaaa, bits[24]:0x4]; bits[45]:0x1555_5555_5555; bits[2]:0x2"
//     args: "bits[26]:0x2aa_aaaa; [bits[24]:0x200, bits[24]:0x8e_a28a, bits[24]:0x55_5555, bits[24]:0x7f_ffff, bits[24]:0x43_0a02, bits[24]:0x55_5555, bits[24]:0x7f_ffff, bits[24]:0x8a_b2e1]; bits[45]:0x1515_7152_bfef; bits[2]:0x2"
//     args: "bits[26]:0x4000; [bits[24]:0x1000, bits[24]:0x18_4040, bits[24]:0x7f_ffff, bits[24]:0x4200, bits[24]:0xff_ffff, bits[24]:0x4_cc04, bits[24]:0x7f_ffff, bits[24]:0x0]; bits[45]:0x10_0000_0000; bits[2]:0x0"
//     args: "bits[26]:0x155_5555; [bits[24]:0xff_ffff, bits[24]:0x49_56f5, bits[24]:0x8_0000, bits[24]:0x7f_ffff, bits[24]:0xb9_d612, bits[24]:0x5d_55d5, bits[24]:0x3c_319f, bits[24]:0x55_5555]; bits[45]:0x100_0000; bits[2]:0x2"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xff_c9de, bits[24]:0xff_ffff, bits[24]:0x40_36c5, bits[24]:0x4_0000, bits[24]:0xaa_aaaa, bits[24]:0x3b_27cf, bits[24]:0x55_5555, bits[24]:0x7f_ffff]; bits[45]:0x1555_5555_5555; bits[2]:0x1"
//     args: "bits[26]:0xe2_e9b3; [bits[24]:0xe2_e9b7, bits[24]:0xff_ffff, bits[24]:0x0, bits[24]:0x0, bits[24]:0xaa_aaaa, bits[24]:0xaa_aaaa, bits[24]:0x81_2be4, bits[24]:0x7f_ffff]; bits[45]:0x1fff_ffff_ffff; bits[2]:0x1"
//     args: "bits[26]:0x29f_46bc; [bits[24]:0xf7_8234, bits[24]:0x9f_46b4, bits[24]:0x1b_06a4, bits[24]:0x0, bits[24]:0x0, bits[24]:0x26_e00e, bits[24]:0x8f_26f9, bits[24]:0xce_479c]; bits[45]:0x1277_f388_caad; bits[2]:0x3"
//     args: "bits[26]:0x0; [bits[24]:0x0, bits[24]:0x40, bits[24]:0x16_6593, bits[24]:0x24_0040, bits[24]:0x60_00a0, bits[24]:0x8000, bits[24]:0x63_4804, bits[24]:0x1d_99e3]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0xff_feff, bits[24]:0x7f_ffff, bits[24]:0xf3_ff7f, bits[24]:0x7f_3c12, bits[24]:0x3f_bfbf, bits[24]:0xdc_8fb0, bits[24]:0xbf_9fbf, bits[24]:0x55_5555]; bits[45]:0xfde_3555_2150; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xff_f7ff, bits[24]:0xff_fd6b, bits[24]:0x20_a180, bits[24]:0x7e_937f, bits[24]:0xf7_ffef, bits[24]:0xff_ffff, bits[24]:0xff_7fb9, bits[24]:0x7f_ffff]; bits[45]:0x0; bits[2]:0x3"
//     args: "bits[26]:0x155_5555; [bits[24]:0x55_5555, bits[24]:0x0, bits[24]:0xaa_aaaa, bits[24]:0x2e_590f, bits[24]:0x55_5555, bits[24]:0xdd_5547, bits[24]:0x55_d5d5, bits[24]:0xa0_d306]; bits[45]:0xa28_ba8f_ffff; bits[2]:0x3"
//     args: "bits[26]:0x0; [bits[24]:0x7f_ffff, bits[24]:0x4_0000, bits[24]:0x1380, bits[24]:0x81_1800, bits[24]:0x0, bits[24]:0x55_5555, bits[24]:0xc6_0d09, bits[24]:0x21_0000]; bits[45]:0x1d41_6b79_d111; bits[2]:0x0"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xff_fffd, bits[24]:0x0, bits[24]:0xff_ffff, bits[24]:0xb3_bed9, bits[24]:0xff_ffef, bits[24]:0xff_ffff, bits[24]:0x5f_347b, bits[24]:0x55_5555]; bits[45]:0xfff_ffff_ffff; bits[2]:0x3"
//     args: "bits[26]:0x1ff_ffff; [bits[24]:0x1000, bits[24]:0xc7_dfff, bits[24]:0x0, bits[24]:0x21_ac1e, bits[24]:0xcd_7376, bits[24]:0x1_0000, bits[24]:0xf7_dcee, bits[24]:0xff_ffff]; bits[45]:0x161c_61ee_2e27; bits[2]:0x1"
//     args: "bits[26]:0x18_84ea; [bits[24]:0x19_844b, bits[24]:0x52_84ef, bits[24]:0x1a_b4ef, bits[24]:0x18_86ea, bits[24]:0x0, bits[24]:0xaa_aaaa, bits[24]:0x7f_ffff, bits[24]:0x1e_5eea]; bits[45]:0x1000; bits[2]:0x3"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0xff_ffff, bits[24]:0x7e_ebbf, bits[24]:0x7b_cbf7, bits[24]:0x0, bits[24]:0xbf_efff, bits[24]:0x87_afb3, bits[24]:0xff_ffff, bits[24]:0xdc_efae]; bits[45]:0x137b_6f72_c3ae; bits[2]:0x1"
//     args: "bits[26]:0x0; [bits[24]:0x40, bits[24]:0x10_0478, bits[24]:0x0, bits[24]:0x55_5555, bits[24]:0x47_70be, bits[24]:0x3_0028, bits[24]:0x44, bits[24]:0x67_7956]; bits[45]:0xf39_1a39_059f; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x88_064f, bits[24]:0xaa_aaaa, bits[24]:0x55_5555, bits[24]:0xff_ffff, bits[24]:0xce_ffe7, bits[24]:0xf9_687a, bits[24]:0xaa_aaaa, bits[24]:0xff_efff]; bits[45]:0x18a_17c5_266e; bits[2]:0x2"
//     args: "bits[26]:0x155_5555; [bits[24]:0x55_5555, bits[24]:0xaa_aaaa, bits[24]:0x55_7555, bits[24]:0x4_5210, bits[24]:0xaa_aaaa, bits[24]:0x7f_ffff, bits[24]:0x5d_5433, bits[24]:0x1_0000]; bits[45]:0x0; bits[2]:0x1"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x2b_a5b7, bits[24]:0x15_3944, bits[24]:0xdf_fbff, bits[24]:0xaa_aaaa, bits[24]:0xde_7f3d, bits[24]:0x0, bits[24]:0xbf_7edb, bits[24]:0x0]; bits[45]:0x1ffe_fffe_a6ac; bits[2]:0x3"
//     args: "bits[26]:0x3ff_ffff; [bits[24]:0x0, bits[24]:0xfb_eff3, bits[24]:0x2000, bits[24]:0x0, bits[24]:0xdf_dc9f, bits[24]:0x6a_b5fd, bits[24]:0x0, bits[24]:0x7e_bfdf]; bits[45]:0xfff_ffff_ffff; bits[2]:0x2"
//     args: "bits[26]:0x107_b076; [bits[24]:0xa6_e8e4, bits[24]:0x7_b176, bits[24]:0xf_b36e, bits[24]:0x7_7c77, bits[24]:0x7f_ffff, bits[24]:0x55_5555, bits[24]:0xe_b076, bits[24]:0x0]; bits[45]:0xfff_ffff_ffff; bits[2]:0x2"
//   }
// }
// 
// END_CONFIG
const W32_V6 = u32:0x6;
const W32_V8 = u32:0x8;
type x0 = s24;
type x17 = uN[224];
type x38 = (x0, u45, u45);
fn x33<x35: u45 = {u45:0x15ec_0d9a_5a57}>(x34: x0) -> (x0, u45, u45) {
    {
        let x36: u45 = rev(x35);
        let x37: x0 = -x34;
        (x37, x35, x35)
    }
}
fn main(x1: s26, x2: x0[8], x3: u45, x4: u2) -> (bool, uN[1593]) {
    {
        let x5: x0[8] = array_slice(x2, x4, x0[8]:[x2[u32:0x0], ...]);
        let x6: uN[1593] = match x1 {
            s26:0x66_65c5 | s26:0xe0_1521 => uN[1593]:0x50_59c0_afcf_a8cb_5e07_c8d8_dbe6_e047_841e_5389_c3fe_2467_fb25_d8d6_84e5_8677_160d_36d5_a5b0_89cb_22a8_e9a8_e147_7ba3_9ed9_1f33_e32f_a614_688e_294d_e517_1520_33cb_6018_2da1_0ce3_aae4_84ba_8977_7b98_bbbf_9920_db7b_99b9_9c29_ea79_4f12_fda7_4f18_2030_cf57_0b5f_81b4_421d_d9b1_a384_4abb_9e21_345c_085d_b34a_5b75_0bb3_7856_eb3a_1117_63bf_2a37_73d6_7066_6602_c8b8_bd5d_1169_9a10_fdc9_98ee_6edb_24a5_8644_c894_ecb3_13f7_d898_6fa4_17d3_35a4_4e44_4493_9bef_b7e6_3933_f3c7_2142_6a9a_0ace_9fcc_42b8_f571_e7e9,
            _ => uN[1593]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa,
        };
        let x7: uN[1593] = rev(x6);
        let x8: uN[1593] = !x6;
        let x9: uN[1593] = x6 + x7 as uN[1593];
        let x10: uN[1593] = for (i, x) in u4:0x0..u4:7 {
            x
        }(x9);
        let x11: u45 = !x3;
        let x12: x0[13] = array_slice(x2, x8, x0[13]:[x2[u32:0x0], ...]);
        let x13: u45 = bit_slice_update(x11, x11, x7);
        let x14: bool = and_reduce(x8);
        let x15: u45 = x13 / u45:0xfff_ffff_ffff;
        let x16: (uN[1593], uN[1593], u45, uN[1593], uN[1593]) = (x9, x9, x15, x10, x7);
        let x18: (u20, (s47, x17[W32_V6], u6, s64)) = match x10 {
            uN[1593]:0x155_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555 | uN[1593]:0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000 => (u20:0xf_ffff, (s47:0x7fff_ffff_ffff, [uN[224]:0x60f6_874b_ea2f_83d9_d039_7531_56c6_2891_9a44_c05b_48e1_a7e4_0750_faf9, uN[224]:0b0, uN[224]:0x0, uN[224]:0x99bf_719d_fddf_0e1d_ff11_c132_d44a_20dc_0b0f_2d64_2bdd_3c9a_272b_4bde, uN[224]:0x3dd6_e10c_69b0_cbda_d454_89b3_60b0_d50c_d732_f478_760e_5eb7_fdca_ea56, uN[224]:0x800_0000_0000_0000_0000_0000_0000], u6:0x2a, s64:0x7fff_ffff_ffff_ffff)),
            uN[1593]:0b1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010 | uN[1593]:0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111 => (xN[bool:0x0][20]:0xf_ffff, (s47:0x8_0000_0000, [uN[224]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[224]:0x0, uN[224]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[224]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[224]:0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000, uN[224]:0xa83_7631_9ad5_f3b9_ef10_ebcd_d0a0_8f52_0ba5_354c_e610_5684_de31_39e3], u6:0x1f, s64:0x400_0000_0000)),
            uN[1593]:0 => (u20:0x0, (s47:0b101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101, [uN[224]:0x1015_1263_0509_1257_0691_8a4c_439f_72ca_66e3_48cd_cb66_0e82_cc09_77d6, uN[224]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa, xN[bool:0x0][224]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa, uN[224]:0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[224]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[224]:0x20_0000_0000_0000_0000_0000_0000_0000], xN[bool:0x0][6]:0x2a, s64:0xf717_42b4_9597_9e1f)),
            uN[1593]:0x0 => (u20:0xf_ffff, (s47:70368744177663, [uN[224]:0x5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555, uN[224]:0x1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000, uN[224]:0x5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555, uN[224]:0x5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555, uN[224]:0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[224]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff], u6:31, s64:-1)),
            _ => (u20:0xa_aaaa, (s47:0x7fff_ffff_ffff, [uN[224]:0x5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555, uN[224]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[224]:0x7f22_368a_398c_f1aa_4429_6fc7_057d_894a_940b_5e83_72a0_4832_8b43_372c, uN[224]:0x3da2_590e_eb24_4a0d_10f7_980c_5db6_9942_df45_639a_7e66_f110_321b_ceb4, uN[224]:0x1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000, uN[224]:0x4000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000], u6:0x1f, xN[bool:0x1][64]:0x0)),
        };
        let x19: u45 = -x3;
        let x20: bool = x4 != x4;
        let x21: x0[1] = array_slice(x2, x4, x0[1]:[x2[u32:0x0], ...]);
        let x22: uN[1593] = priority_sel(x4, [x9, x7], x7);
        let x23: uN[1593] = rev(x6);
        let x24: uN[1594] = one_hot(x22, bool:0x1);
        let x25: uN[1593] = x16.0;
        let x26: s26 = -x1;
        let x27: bool = x20 & x20;
        let x28: s26 = for (i, x) in u4:0x0..u4:0x5 {
            x
        }(x26);
        let x29: s26 = -x28;
        let x30: bool = x14 >> x4;
        let x31: bool = and_reduce(x25);
        let x32: u3 = x3[1+:u3];
        let x39: x38[W32_V8] = map(x2, x33);
        (x30, x25)
    }
}
