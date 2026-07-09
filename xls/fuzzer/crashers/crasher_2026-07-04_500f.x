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
// issue: "https://github.com/google/xls/issues/4545"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
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
//     args: "bits[38]:0x33_9241_2562; bits[18]:0x3_2f32; bits[17]:0x1_ffff; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x2_0000, bits[16]:0xffff, bits[7]:0x55)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x1_ffff; bits[17]:0x0; bits[52]:0x4_0000_0000; (bits[59]:0x555_5555_5555_5555, bits[16]:0x0, bits[7]:0x4)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0xf9d3; bits[17]:0x1_5555; bits[52]:0x9_73e3_d8ab_8dbf; (bits[59]:0x10_0000, bits[16]:0x8000, bits[7]:0x7f)"
//     args: "bits[38]:0x24_260b_f018; bits[18]:0x3_e3b8; bits[17]:0x1_5555; bits[52]:0x1_0000_0000; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xf019, bits[7]:0x38)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_f15f; bits[17]:0x0; bits[52]:0x5_5555_5555_5555; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0x430, bits[7]:0x0)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_feff; bits[17]:0xaaaa; bits[52]:0x6_efda_31bf_97ff; (bits[59]:0x7ff_ffff_ffff_ffff, bits[16]:0x5555, bits[7]:0x40)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x100; bits[17]:0x0; bits[52]:0x7_f7fa_fa27_cb48; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xffff, bits[7]:0x4)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_ffff; bits[17]:0x1_ffff; bits[52]:0xf_7dd8_00d8_814c; (bits[59]:0x7ff_fe00_0060_0100, bits[16]:0xa0ad, bits[7]:0x0)"
//     args: "bits[38]:0x16_1c8a_235f; bits[18]:0x1_ffff; bits[17]:0x1_e05b; bits[52]:0x5_79b8_36b0_280e; (bits[59]:0x2c3_9120_fbf1_86cb, bits[16]:0x100, bits[7]:0xb)"
//     args: "bits[38]:0x0; bits[18]:0x2_4ac7; bits[17]:0xdac7; bits[52]:0xd_4917_afdf_eb3f; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0xffff, bits[7]:0x7)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x1_e7ed; bits[17]:0x1_c7ed; bits[52]:0xc_3744_b1fb_9477; (bits[59]:0x71f_b400_1008_4008, bits[16]:0x800, bits[7]:0x3f)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x1_2371; bits[17]:0xcdab; bits[52]:0x6_6d58_0000_1800; (bits[59]:0x555_5555_5555_5555, bits[16]:0x2771, bits[7]:0x21)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x1_0000; bits[17]:0xaaaa; bits[52]:0x1000_0000; (bits[59]:0x1_0000_0000_0000, bits[16]:0x361f, bits[7]:0x2e)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x2_aaaa; bits[17]:0x67a1; bits[52]:0x0; (bits[59]:0x2000_0000, bits[16]:0x76a1, bits[7]:0x2)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x1_cb46; bits[17]:0x1_ffff; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x4000_0000_0000, bits[16]:0xffff, bits[7]:0x0)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x3_ffff; bits[17]:0xaaaa; bits[52]:0xf_ffe8_4008_0820; (bits[59]:0x487_3ef7_20ae_8080, bits[16]:0xbc3d, bits[7]:0x2a)"
//     args: "bits[38]:0x0; bits[18]:0x1_5555; bits[17]:0xd164; bits[52]:0x6_aa2d_7554_5555; (bits[59]:0x555_5555_5555_5555, bits[16]:0x7fff, bits[7]:0x55)"
//     args: "bits[38]:0x0; bits[18]:0x0; bits[17]:0x20; bits[52]:0x0; (bits[59]:0x10_014c_0104_4002, bits[16]:0x0, bits[7]:0x7f)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x1_ffff; bits[17]:0xaaaa; bits[52]:0x3_f7fe_aae2_8aab; (bits[59]:0x555_5555_5555_5555, bits[16]:0xfff9, bits[7]:0xa)"
//     args: "bits[38]:0x0; bits[18]:0x3_5048; bits[17]:0xffff; bits[52]:0xa_f81f_a33d_d9b1; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0x80b3, bits[7]:0x0)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_ffff; bits[17]:0xde57; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x3fd_ffaa_baab_8ae8, bits[16]:0xbfdf, bits[7]:0x7f)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0xef3e; bits[17]:0x1_5e6a; bits[52]:0x2_0000_0000; (bits[59]:0x5e2_1471_4b01_82d4, bits[16]:0x5555, bits[7]:0x36)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_7775; bits[17]:0x1_15d7; bits[52]:0x9_399e_78e2_fa28; (bits[59]:0x8000_0000_0000, bits[16]:0xc4c5, bits[7]:0x2a)"
//     args: "bits[38]:0x1_0000; bits[18]:0x1_5600; bits[17]:0x5057; bits[52]:0x5_5555_5555_5555; (bits[59]:0x363_5400_2020_0808, bits[16]:0x7fff, bits[7]:0x14)"
//     args: "bits[38]:0x1e_17ae_6811; bits[18]:0x3_ffff; bits[17]:0x1_affe; bits[52]:0x5_5555_5555_5555; (bits[59]:0x6e1_6e87_2da1_8df8, bits[16]:0x8c7f, bits[7]:0x55)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_aaea; bits[17]:0x10; bits[52]:0xa_8ba9_03dc_0963; (bits[59]:0x557_4545_5d40_4501, bits[16]:0xabaa, bits[7]:0x63)"
//     args: "bits[38]:0x32_1df2_31f3; bits[18]:0x3_3bbe; bits[17]:0x1_5555; bits[52]:0xc_ecf1_bfff_f7bf; (bits[59]:0x4fd_a97b_b2c8_9159, bits[16]:0x43d5, bits[7]:0x4)"
//     args: "bits[38]:0x0; bits[18]:0x8318; bits[17]:0x417; bits[52]:0xa_fa19_6bc5_6757; (bits[59]:0x424_6cad_0000_0000, bits[16]:0x7778, bits[7]:0x8)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x1_5555; bits[17]:0x1_7572; bits[52]:0xa_a410_a7ea_b641; (bits[59]:0x1_0000_0000, bits[16]:0x7572, bits[7]:0x2a)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x8a92; bits[17]:0x1_5555; bits[52]:0x4000_0000; (bits[59]:0x27d_71ff_b7f7_dd53, bits[16]:0x4, bits[7]:0x55)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x1_ffff; bits[17]:0x1_5555; bits[52]:0xa_caaa_b68a_aaaa; (bits[59]:0x565_555f_45d5_452e, bits[16]:0x7bda, bits[7]:0x3f)"
//     args: "bits[38]:0x20_0000; bits[18]:0x8000; bits[17]:0x1_5555; bits[52]:0x8_0000_0000_0000; (bits[59]:0x555_5555_5555_5555, bits[16]:0x9c9b, bits[7]:0x3)"
//     args: "bits[38]:0x4000_0000; bits[18]:0x40; bits[17]:0x1401; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x39d_fe64_9533_eada, bits[16]:0xeca, bits[7]:0x5f)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_9b7f; bits[17]:0x1_efff; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x0, bits[16]:0xeafe, bits[7]:0x7f)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x3_3aae; bits[17]:0x0; bits[52]:0x3_a3a2_4c27_f816; (bits[59]:0x635_5d5d_5551_75d5, bits[16]:0xffff, bits[7]:0x20)"
//     args: "bits[38]:0x0; bits[18]:0x6c2; bits[17]:0x0; bits[52]:0x9_c800_8a61_9334; (bits[59]:0x80_0000, bits[16]:0xee2, bits[7]:0x42)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_5941; bits[17]:0x0; bits[52]:0x5_5555_5555_5555; (bits[59]:0x202_cecf_559f_5f7f, bits[16]:0x99, bits[7]:0x2a)"
//     args: "bits[38]:0x0; bits[18]:0x800; bits[17]:0x1_5555; bits[52]:0x8_8ece_aaaa_aaaa; (bits[59]:0x12_0430_2440_0000, bits[16]:0x2000, bits[7]:0x64)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_bfff; bits[17]:0x1_1a5f; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x699_a18f_e056_0e12, bits[16]:0xaaaa, bits[7]:0x3e)"
//     args: "bits[38]:0x0; bits[18]:0x3_1046; bits[17]:0x0; bits[52]:0x5_5555_5555_5555; (bits[59]:0x19_0f7f_6fd7_5bbf, bits[16]:0x0, bits[7]:0x42)"
//     args: "bits[38]:0xe_73fd_12df; bits[18]:0x1_12df; bits[17]:0x1_ffff; bits[52]:0x3_235a_d1f2_b278; (bits[59]:0x555_5555_5555_5555, bits[16]:0x12df, bits[7]:0x5f)"
//     args: "bits[38]:0xc_f778_642b; bits[18]:0x0; bits[17]:0x0; bits[52]:0x40_0022_0000; (bits[59]:0x524_775e_ffdf_9fff, bits[16]:0x5555, bits[7]:0x12)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_aaaa; bits[17]:0x8; bits[52]:0x9_28ac_aec1_4290; (bits[59]:0x1e2_b35b_dfcc_bb5d, bits[16]:0xffff, bits[7]:0x10)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x8000; bits[17]:0x1_ffff; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x5c3_265e_d939_18c4, bits[16]:0xff9f, bits[7]:0x3c)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_fffe; bits[17]:0x0; bits[52]:0xf_dddf_efd7_c820; (bits[59]:0x7a5_fc65_5543_b795, bits[16]:0x7244, bits[7]:0x3f)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_d551; bits[17]:0x7157; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x0, bits[16]:0x91d5, bits[7]:0x2a)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x3_ffff; bits[17]:0x1_aa28; bits[52]:0x8_a341_71b8_0ecf; (bits[59]:0x7ff_ffff_ffff_ffff, bits[16]:0x221c, bits[7]:0x7f)"
//     args: "bits[38]:0x13_9f72_020a; bits[18]:0x1_ffff; bits[17]:0xffff; bits[52]:0x5_5555_5555_5555; (bits[59]:0x3ff_dff7_f77f_ffbf, bits[16]:0x7fff, bits[7]:0x4e)"
//     args: "bits[38]:0x2; bits[18]:0x40; bits[17]:0x1_ffff; bits[52]:0xa_a117_4625_aed0; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0x80c0, bits[7]:0xc)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x3_2a9a; bits[17]:0x1_0bb6; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x605_2002_0008_0001, bits[16]:0x7fff, bits[7]:0x6b)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x3_20af; bits[17]:0x7a2e; bits[52]:0xe_c33f_16ca_9111; (bits[59]:0xbc_2c02_cb3f_74f0, bits[16]:0x20af, bits[7]:0x40)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x2_aaaa; bits[17]:0xffff; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x7ff_ffff_ffff_ffff, bits[16]:0xa1aa, bits[7]:0x7f)"
//     args: "bits[38]:0x100; bits[18]:0x8000; bits[17]:0x3af3; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x59e_b1a7_edc0_4869, bits[16]:0xaaaa, bits[7]:0x40)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_7517; bits[17]:0x1_7437; bits[52]:0x6_d868_8b32_b2e0; (bits[59]:0x2ea_2f55_55d7_5555, bits[16]:0xd4ec, bits[7]:0x21)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x7236; bits[17]:0x1_ffff; bits[52]:0x6_cda3_cf7e_5a69; (bits[59]:0x555_5555_5555_5555, bits[16]:0x4a0c, bits[7]:0x78)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_ffff; bits[17]:0x80; bits[52]:0x2_66b9_c4d3_ce61; (bits[59]:0x4da_b08f_ab5c_3b8c, bits[16]:0x100, bits[7]:0x70)"
//     args: "bits[38]:0x800_0000; bits[18]:0x1_5555; bits[17]:0x8; bits[52]:0xd_459e_7f78_c7e8; (bits[59]:0x0, bits[16]:0x8, bits[7]:0x7f)"
//     args: "bits[38]:0x0; bits[18]:0x1_ffff; bits[17]:0x1_5555; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x5c_e4d9_27a3_70f7, bits[16]:0x0, bits[7]:0x40)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_a9aa; bits[17]:0xa58a; bits[52]:0x5_2cd0_008a_8540; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xaaaa, bits[7]:0x40)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x1_af60; bits[17]:0x3c36; bits[52]:0x0; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xaaab, bits[7]:0x60)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x2_aaaa; bits[17]:0xaebe; bits[52]:0x2_ad2b_ddb7_314d; (bits[59]:0x37b_febb_bfea_baaa, bits[16]:0x61cd, bits[7]:0x6e)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_5555; bits[17]:0x1_ffff; bits[52]:0xf_ffe8_2000_0000; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0x2000, bits[7]:0x11)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x0; bits[17]:0x1_5555; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x7ff_ffff_ffff_ffff, bits[16]:0x7fff, bits[7]:0x2a)"
//     args: "bits[38]:0x8000; bits[18]:0x1_5555; bits[17]:0x8974; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0xb_eb01_504d_169a, bits[16]:0x7555, bits[7]:0x5)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x1_ffff; bits[17]:0x1_5555; bits[52]:0xf_fff8_2100_0941; (bits[59]:0x7ff_ffff_ffff_ffff, bits[16]:0x5555, bits[7]:0x2)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_ffff; bits[17]:0x1_eedd; bits[52]:0x4_d6bf_7721_07b4; (bits[59]:0x2c7_452f_0416_5c83, bits[16]:0xaaaa, bits[7]:0x6b)"
//     args: "bits[38]:0x0; bits[18]:0x2_aaaa; bits[17]:0x9b14; bits[52]:0x3_a9b4_7c54_d55e; (bits[59]:0x3ab_a3fa_c70d_a9e7, bits[16]:0x9b14, bits[7]:0x55)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x1_ffff; bits[17]:0x1_40f6; bits[52]:0xa_07b0_aaaa_aaaa; (bits[59]:0x555_5555_5555_5555, bits[16]:0x7fff, bits[7]:0x36)"
//     args: "bits[38]:0x0; bits[18]:0x1_ffff; bits[17]:0x1_ffff; bits[52]:0x4_0000_0000; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0x7de7, bits[7]:0x6f)"
//     args: "bits[38]:0x40_0000; bits[18]:0x1_80c8; bits[17]:0xe8bb; bits[52]:0x6_4eda_65a5_7ebd; (bits[59]:0x555_5555_5555_5555, bits[16]:0x400, bits[7]:0x7f)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x1000; bits[17]:0x100; bits[52]:0x2_0000_0000; (bits[59]:0x4_0000_2040_0008, bits[16]:0x7a2a, bits[7]:0x2a)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_aaee; bits[17]:0xaaaa; bits[52]:0x5_5d52_2b3a_aaaa; (bits[59]:0x4, bits[16]:0xb3e7, bits[7]:0x3f)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_aaaa; bits[17]:0x1_5555; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x49d_d074_e963_2435, bits[16]:0xffff, bits[7]:0x8)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_ffef; bits[17]:0xefeb; bits[52]:0x5_ff5f_6367_43c0; (bits[59]:0x1d0_bdee_d487_9c48, bits[16]:0xffff, bits[7]:0x44)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x3_ffff; bits[17]:0x2; bits[52]:0x5_5555_5555_5555; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0x7fff, bits[7]:0x4c)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_f7ee; bits[17]:0x1_ffdc; bits[52]:0x0; (bits[59]:0x7df_ffff_7fc0_1120, bits[16]:0xd5b0, bits[7]:0x2a)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x8ffb; bits[17]:0x1_0000; bits[52]:0x5_5555_5555_5555; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0x8bfa, bits[7]:0x19)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0xbcab; bits[17]:0x1_ffff; bits[52]:0xc_1e9d_8b45_8acd; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0x3de3, bits[7]:0x36)"
//     args: "bits[38]:0x0; bits[18]:0xcc0f; bits[17]:0x1_90ef; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x0, bits[16]:0x5555, bits[7]:0x6b)"
//     args: "bits[38]:0x31_a5d3_66d7; bits[18]:0x2_2675; bits[17]:0x0; bits[52]:0x6_9948_8caa_ea2a; (bits[59]:0x65e_8c7f_3aeb_99e2, bits[16]:0x7fff, bits[7]:0x2)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_c23a; bits[17]:0xc23a; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x5a6_5215_7f15_555d, bits[16]:0x3d07, bits[7]:0x55)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x0; bits[17]:0xc9; bits[52]:0xa_c8be_a8e0_baab; (bits[59]:0x135_d841_84c7_2cb3, bits[16]:0xe8a2, bits[7]:0x0)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x1_5555; bits[17]:0x1_ea22; bits[52]:0x9_d912_5929_02c9; (bits[59]:0x0, bits[16]:0xc2fa, bits[7]:0xb)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x2_faf9; bits[17]:0x1_5555; bits[52]:0xf_a97c_0b02_de3c; (bits[59]:0x555_5555_5555_5555, bits[16]:0xcf18, bits[7]:0x7c)"
//     args: "bits[38]:0x0; bits[18]:0x9343; bits[17]:0xffff; bits[52]:0x7_f7fb_3b4e_3c3d; (bits[59]:0x555_5555_5555_5555, bits[16]:0x2c8f, bits[7]:0x7f)"
//     args: "bits[38]:0x3_9856_58ed; bits[18]:0x2_58cd; bits[17]:0x1_ffff; bits[52]:0x5_fbfb_f2fd_f9fe; (bits[59]:0x2fd_fffb_5efc_ff0f, bits[16]:0x6995, bits[7]:0x67)"
//     args: "bits[38]:0x8_cf5d_2f8a; bits[18]:0x1_2e8a; bits[17]:0x0; bits[52]:0xe_b08f_a672_72ab; (bits[59]:0xf9_4990_add7, bits[16]:0x12c1, bits[7]:0x10)"
//     args: "bits[38]:0x2_0000; bits[18]:0x4a84; bits[17]:0x0; bits[52]:0x8_001b_ca82_a2ea; (bits[59]:0xa2_4a82_1002_d340, bits[16]:0xa2ea, bits[7]:0x4d)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_8aea; bits[17]:0x0; bits[52]:0x4_0004_83d4_f501; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0x2, bits[7]:0x66)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_2199; bits[17]:0x1_5555; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x53d_5597_7fb5_b725, bits[16]:0x4555, bits[7]:0x2b)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x3_ffff; bits[17]:0x0; bits[52]:0xb_30ff_d76f_2e46; (bits[59]:0x0, bits[16]:0x642, bits[7]:0x3f)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x0; bits[17]:0xa287; bits[52]:0xc_1438_1028_0588; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0xa287, bits[7]:0x2)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_7850; bits[17]:0x1_7853; bits[52]:0x7_e541_5055_953d; (bits[59]:0x0, bits[16]:0x5555, bits[7]:0x35)"
//     args: "bits[38]:0x800_0000; bits[18]:0x4000; bits[17]:0xaaaa; bits[52]:0x1_5280_0000_3635; (bits[59]:0x2aa_8908_2032_0248, bits[16]:0x40a8, bits[7]:0x3)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_beff; bits[17]:0x1_ffff; bits[52]:0x0; (bits[59]:0x3de_fc11_1150_4088, bits[16]:0x5555, bits[7]:0x3d)"
//     args: "bits[38]:0x0; bits[18]:0x0; bits[17]:0x1_095f; bits[52]:0x9_1007_a2bf_baaa; (bits[59]:0x811c_6912_b820, bits[16]:0x7fff, bits[7]:0xb)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x2_b7bd; bits[17]:0x0; bits[52]:0x800_0000_0000; (bits[59]:0x80_0155_7755_554d, bits[16]:0xffff, bits[7]:0x58)"
//     args: "bits[38]:0x0; bits[18]:0x1_0000; bits[17]:0x1_4044; bits[52]:0x8_ce2e_89fd_7bd7; (bits[59]:0x465_3d40_efb4_e980, bits[16]:0x4, bits[7]:0x0)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x1_5555; bits[17]:0x1000; bits[52]:0x7_73bc_daf5_827d; (bits[59]:0x5ea_a9c7_db8f_9c94, bits[16]:0x9251, bits[7]:0x2a)"
//     args: "bits[38]:0x200_0000; bits[18]:0x1_0ea2; bits[17]:0x1_4e82; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xaa6, bits[7]:0x12)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_fefc; bits[17]:0x0; bits[52]:0xb_a75f_eb0d_f81a; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xbafc, bits[7]:0x68)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x3_ffff; bits[17]:0xaaaa; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x2aa_abaa_a8aa_aaa2, bits[16]:0x7fff, bits[7]:0x37)"
//     args: "bits[38]:0x3a_54ca_68bb; bits[18]:0x2_309b; bits[17]:0x0; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x7ff_fdff_efef_fb81, bits[16]:0x0, bits[7]:0x3b)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x1_4920; bits[17]:0x1_4857; bits[52]:0xa_a3aa_abea_9155; (bits[59]:0x7ff_ffff_ffff_ffff, bits[16]:0x7fff, bits[7]:0x43)"
//     args: "bits[38]:0x1_0000_0000; bits[18]:0x0; bits[17]:0x1cd2; bits[52]:0x8_3040_4020_00a0; (bits[59]:0x4000_0000_0000, bits[16]:0xffff, bits[7]:0x0)"
//     args: "bits[38]:0x0; bits[18]:0x2_aaaa; bits[17]:0x0; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x2, bits[16]:0x79a4, bits[7]:0x2a)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x0; bits[17]:0x0; bits[52]:0x302_b598_593d; (bits[59]:0x401_d3f4_5cf3_3820, bits[16]:0x484f, bits[7]:0x3f)"
//     args: "bits[38]:0x0; bits[18]:0x1_5555; bits[17]:0x1d41; bits[52]:0x5_5557_7fef_ff7f; (bits[59]:0x555_5555_5555_5555, bits[16]:0x5555, bits[7]:0x75)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x2_aaaa; bits[17]:0x93cf; bits[52]:0x5_5555_5555_5555; (bits[59]:0x15_5134_bffc_afda, bits[16]:0x9ee3, bits[7]:0x11)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x2_fdb5; bits[17]:0x1_ffff; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x6f0_a646_2f92_282e, bits[16]:0xfdb5, bits[7]:0x10)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x3_ffff; bits[17]:0x1_ffff; bits[52]:0x4_7bbe_df30_6acf; (bits[59]:0x3ff_ffff_ffff_ffff, bits[16]:0xfd7b, bits[7]:0x7f)"
//     args: "bits[38]:0x18_a3e7_6954; bits[18]:0x1_ec55; bits[17]:0x1_7b74; bits[52]:0x0; (bits[59]:0x1a4_6d4e_d25b_1657, bits[16]:0x9880, bits[7]:0x3f)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_fffd; bits[17]:0x95d6; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x33e_cecc_f962_15d9, bits[16]:0xdfef, bits[7]:0x0)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0x1_5555; bits[17]:0x1_5555; bits[52]:0xa_aaaf_ffff_fffb; (bits[59]:0x555_5698_252a_8964, bits[16]:0x0, bits[7]:0x36)"
//     args: "bits[38]:0x1d_795a_1c2d; bits[18]:0x3_0c3c; bits[17]:0x449; bits[52]:0x368_e5d5_5154; (bits[59]:0x512_ebf8_351a_3dea, bits[16]:0x5864, bits[7]:0x0)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0xe75c; bits[17]:0x1_ffff; bits[52]:0xa_09cb_96ea_648d; (bits[59]:0x546_c5d8_d432_76b1, bits[16]:0x0, bits[7]:0x3f)"
//     args: "bits[38]:0x2a_aaaa_aaaa; bits[18]:0x0; bits[17]:0xbca2; bits[52]:0x5030_1400_0003; (bits[59]:0x60_1909_1094_4d88, bits[16]:0x8c18, bits[7]:0x14)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x1_ffff; bits[17]:0x7ffb; bits[52]:0xf_ffff_efff_c040; (bits[59]:0x1000_0000_0000, bits[16]:0x7efb, bits[7]:0x3f)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x2_aaaa; bits[17]:0x1_5555; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x555_55ed_591b_eac8, bits[16]:0x7fff, bits[7]:0x55)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x3_6edd; bits[17]:0xffff; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x3af_d5e8_8968_2aa6, bits[16]:0xaaaa, bits[7]:0x18)"
//     args: "bits[38]:0x3f_ffff_ffff; bits[18]:0x2_ff9a; bits[17]:0x1_5555; bits[52]:0xb_fe69_d555_1d75; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xf2d5, bits[7]:0x0)"
//     args: "bits[38]:0x0; bits[18]:0x61d2; bits[17]:0x1_5555; bits[52]:0x0; (bits[59]:0x80_9646_5e90_40e1, bits[16]:0x545, bits[7]:0x3f)"
//     args: "bits[38]:0x100; bits[18]:0x508; bits[17]:0x1_20e2; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0xaaaa, bits[7]:0x7f)"
//     args: "bits[38]:0x0; bits[18]:0x3_ffff; bits[17]:0xaaaa; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x7ff_db57_5155_158d, bits[16]:0x8, bits[7]:0x0)"
//     args: "bits[38]:0x8000; bits[18]:0x1_5840; bits[17]:0x8000; bits[52]:0xa_aaaa_aaaa_aaaa; (bits[59]:0x2aa_aaaa_aaaa_aaaa, bits[16]:0x4, bits[7]:0x0)"
//     args: "bits[38]:0x15_5555_5555; bits[18]:0xd5db; bits[17]:0x800; bits[52]:0xf_ffff_ffff_ffff; (bits[59]:0x3aa_2aea_eaa1_0100, bits[16]:0xaaaa, bits[7]:0x8)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x2_990c; bits[17]:0x9d0c; bits[52]:0x3_f77e_ff8b_1b5f; (bits[59]:0x10_0000_0000_0000, bits[16]:0xffff, bits[7]:0x0)"
//     args: "bits[38]:0x1f_ffff_ffff; bits[18]:0x2_aaaa; bits[17]:0x1_ffff; bits[52]:0x7_ffff_ffff_ffff; (bits[59]:0x0, bits[16]:0xffff, bits[7]:0x37)"
//   }
// }
// 
// END_CONFIG
const W32_V2020 = u32:2020;
type x7 = xN[bool:0x0][8];
fn x16<x18: u47 = {u47:0x2000_0000_0000}>(x17: x7) -> (u47, xN[bool:0x0][26], xN[bool:0x0][26], xN[bool:0x0][26]) {
    {
        let x19: u47 = x18 >> if x17 >= x7:',' { x7:',' } else { x17 };
        let x20: xN[bool:0x0][26] = x19[x19+:u26];
        (x19, x20, x20, x20)
    }
}
fn x26(x27: u17, x28: s18, x29: s28, x30: u17, x31: u2, x32: s28, x33: s18) -> (u21, bool) {
    {
        let x34: bool = and_reduce(x27);
        let x35: u21 = x31 ++ x34 ++ x30 ++ x34;
        let x36: u23 = (x29 as u28)[5+:u23];
        let x37: bool = x34 & x34;
        (x35, x34)
    }
}
fn main(x0: u38, x1: s18, x2: u17, x3: s52, x4: (u59, u16, s7)) -> (u38, s18, s11, x7[W32_V2020], u17, u17) {
    {
        let x5: u18 = one_hot(x2, bool:0x1);
        let x6: u17 = x2 % u17:0x1000;
        let x8: x7[W32_V2020] = "`b`g$gA\'#BG-R\\pNu@S7w{{sV2CbysE!(L07?JzQHe;]c_i[MPW)K8n?7KLHb2 \'pIcKR]LESX_;BaTxExC$6iIJ/8;3+|#h,DF&F6uB MgJ:+@A}n)(&xgTJr}p*:NmO\'eB4F2|Eg%U\"U8lcI}EWo7{WAKNR9@PfC\"]1\\0QQJjwV6492f}cl[4txt]\')~B7mEV1nV|U]et)53t%`B9lfJE,T0@*Sm#E[pop3nL\\wGX}JB?,IfX{A~H8n|t[bWieTOvhO+mhM`[xt<Edcz)G6(VpF;gEDWeb]I.PK9$e#`a!:zgRqM/KPdbL\\I$RUKmnU/DTN,CwY/m<+os]]j2&(YF]Kz\\iXKEmfP6hPlHQTFf(,CWIk;cknQpaz(=%/G,#oWd[P:Ov*GbgXe>a0~XUF[3@(~l\"bIPP&MES/}6`VR%wC{ /x*}o5b7+2\\eh1!LX, \\,/h6^ .fyI?onvlpD;0)kl.j(\'#G%rjL`r;m}glBjE|y^sVfs7iMxY*,p2TK(}>z/G=WM~BzoRW6o|T\'~3mFly,*-VBFz%/0^<g0$!g)wjcbd~aIc@</?0Y.:cp\\p*?]c@,cb]L(sN-Z]-)ddOspjQ[j P}Y3![ 6:./`rqKF*mNB*yP_,a5~7*8o2\"{4$7^Bh1::DW1UU1%tgUxAbt6QCzaeO^w9lgAQt87Jkhl7]\",@!6~JbDpQ4*fX-o4ovD4rIpc]Nu7RN3},.4:)uF\"Lh(}n/C%/)!*In#vgY.4|}K\'R4Ily6`$v!Sz;`{EL:HNmxHx(T3#N3ctgq0d.%e/:R{Vs&L7SS[]Xdjikd4xkB5)YG-[b#pW2,$P/nNObc;FQ]h$?Ad7_03B1,t<55hP!7Yb7$X (/o|kpo3J>d6}b[$Tj9TO3\'o\'!|0X>7f-VWO\'U_&r}h~J+Myp4d[K9gAG!e}QDcV<(@Hk!\\Ft\\>/@K+|7e\"Gi*V<LB?vi%7F`}\"e~IoABs>d3XyESbNZ k]Ezyu?NeTy;IXIOwBk-]nBelv$\"yO)7^b\'P4Eq$1nL6n]x6 )ZW:]`nG$(kl@,v))PYNuF^~n\']N&|q\"pVcc=yBY|9*PE!WD|{~}9!`}q<uIH\"Q#+E^$HMH3J2uydW5O]nqMU>[MS?HCA06i?]vxKs7&mjE]q@Y !K6%!HtS?$x[q*R*LVXu2aNSgNs$ny3}xE,F/7Y:>}Ah*x6TX#+WJr7QXH9?/w<YkdzgiuE&y\\;(mROYZ.([8yTxi7eY+`\\Yj]\"d\"]Og\"sidC}|ls]^FsNfSH`d))ilRLv%I`gD8@O#\\i4D#Kt27iDLX4~nxLIUI_GvgX(/GQCbS|WUB6}e%p0_tse@8vOQ=JY#a[\\,Ik6yFlE0/E$a}.#j\'}rWRe7c1^\"T/y{=NBUNe}UN/XrF..W F+th.YKb~2(b\\yrr3fX*83%dHQ9>>6Kq/Sg<&;]UUfjz&\\/il_vmU{)dr(cX?*=@W&gKUbD#bsv-N:$ODZ4d:d|{C[y<F*l/~c)BgAnYrU&y[@DPd#@DtBt-vB04i3[]}sS:NuQyZ+I\'~]\"~e[x_^~.g+lBXK0@\'qi4R,-,FFJyGt|cafn:9sg{:?Bou>)yc>o06tP%FX)_VHlh}lEjoJSYFrw6zROAEn!Y3qY]]nts!\"m*g<~?%H+5SAH xD/>i6X,v*~9.OjnFp<?s$\"sUIhOD[+K}yT>Q[dHBlSiP@gXLRp|Un\"TVNJ_XQ~,z0OwPVd)uG~X{;wZ~YTaGztji+/oh. :Nl,%0bhvkH) 2rfR4N]Vw=.s7i\' gyYm&rA@d=f(EH09|S#|UM@U}$4`_[FDa$kU)RtI|d7y\"kG14u1?Idb>TKO!Lx:M>P/2jpXzF;FkW$#1x6fL@tVySf$8]oyf_aH\\\\ud`[+\'itub1,Dwp\\8j0J0<A4Db4z&vVR\"` _RV>dTz.oH|k84B?f<jd;4}OfF$l;<bj(B;=NAbf|/r89Rq[e:(eBBGkNy<BJ/?RRLL1kn_z{Sx rSt]*XvS<^<L.)D!Q=n=WYmSHyxRC~\\LI#Hx=J(,c[AH<";
        let x9: u38 = x0 << if x0 >= u38:0xa { u38:0xa } else { x0 };
        let x10: u17 = x9 as u17 | x2;
        let x11: s52 = !x3;
        let x12: bool = x9 <= x9;
        let x13: u4 = x6[1:-12];
        let x14: u48 = match x3 {
            s52:0x7_7051_d403_3b20 | s52:0xa_aaaa_aaaa_aaaa => u48:0b0,
            s52:0x5_5555_5555_5555..s52:1813211861183971 | s52:0x1 => u48:0x8_0000,
            s52:0xf_ffff_ffff_ffff | s52:0x2000_0000_0000 => u48:0xffff_ffff_ffff,
            _ => u48:0xf14b_75d6_cf75,
        };
        let x15: s28 = s28:0x7ff_ffff;
        let x21: bool = x12 | x12;
        let x22: s11 = s11:0x2aa;
        let x23: u38 = !x0;
        let x24: bool = xor_reduce(x2);
        let x25: u2 = x10[5+:u2];
        let x38: (u21, bool) = x26(x2, x1, x15, x10, x25, x15, x1);
        let (x39, x40): (u21, bool) = x26(x2, x1, x15, x10, x25, x15, x1);
        let x41: x7 = x8[if x14 >= u48:0x29 { u48:0x29 } else { x14 }];
        let x42: u38 = !x23;
        let x43: bool = x38.1;
        let x44: bool = x12 >> if x14 >= u48:0x0 { u48:0x0 } else { x14 };
        let x45: x7[W32_V2020] = update(x8, x41, x41);
        let x46: bool = x6[x6+:bool];
        let x47: bool = bit_slice_update(x44, x46, x21);
        let x49: s18 = {
            let x48: (u18, u18) = smulp(x23 as u18 as s18, x5 as s18);
            (x48.0 + x48.1) as s18
        };
        (x23, x1, x22, x45, x10, x6)
    }
}
