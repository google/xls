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
// exception: "SampleError: Result miscompare for sample 1:\nargs: bits[28]:0xaaa_aaaa; [bits[18]:0x2_aaaa, bits[18]:0x2_2aaa, bits[18]:0x2_8a00, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0xaa8a, bits[18]:0x2_ea1a, bits[18]:0x0, bits[18]:0x3_8a22, bits[18]:0x1_a6aa, bits[18]:0xba9a]; bits[45]:0x1155_1752_b972\nevaluated opt IR (JIT), evaluated opt IR (interpreter) =\n   (bits[28]:0x1, bits[37]:0x11_5517_52b9, bits[1]:0x1)\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[28]:0x0, bits[37]:0x11_5517_52b9, bits[1]:0x1)"
// issue: "https://github.com/google/xls/issues/2046"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
// }
// inputs {
//   function_args {
//     args: "bits[28]:0x0; [bits[18]:0x2_7297, bits[18]:0x1_ffff, bits[18]:0x1_0002, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_112b, bits[18]:0x0, bits[18]:0x214, bits[18]:0x2004]; bits[45]:0x610_0202_2c90"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_aaaa, bits[18]:0x2_2aaa, bits[18]:0x2_8a00, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0xaa8a, bits[18]:0x2_ea1a, bits[18]:0x0, bits[18]:0x3_8a22, bits[18]:0x1_a6aa, bits[18]:0xba9a]; bits[45]:0x1155_1752_b972"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_efff, bits[18]:0x1_97a7, bits[18]:0x3_72c8, bits[18]:0x20, bits[18]:0x1_fffc, bits[18]:0x3_fffe, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_6dfd, bits[18]:0x3_ffff, bits[18]:0x1_5555]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x0, bits[18]:0x3_77db, bits[18]:0x2_1ef1, bits[18]:0x40, bits[18]:0xd0b9, bits[18]:0x3_4fae, bits[18]:0x3_ffff, bits[18]:0x2_57ee, bits[18]:0x1_fbe5, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa]; bits[45]:0x1fbf_feff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_7fca, bits[18]:0x2_aaaa, bits[18]:0x3_f72c, bits[18]:0x0, bits[18]:0x3_f9ff, bits[18]:0x3_edfc, bits[18]:0x3_ffff, bits[18]:0x3_e47a, bits[18]:0x3_befe]; bits[45]:0x200_0000"
//     args: "bits[28]:0x555_5555; [bits[18]:0x2_aaaa, bits[18]:0x2, bits[18]:0x1_ffff, bits[18]:0x1_5551, bits[18]:0x1_ffff, bits[18]:0x5157, bits[18]:0x3_dc6f, bits[18]:0x0, bits[18]:0x1_cdba, bits[18]:0x1_ffff, bits[18]:0x200]; bits[45]:0x83c_b8e7_f1e9"
//     args: "bits[28]:0x555_5555; [bits[18]:0x3_ffff, bits[18]:0x2_79d5, bits[18]:0x1_b668, bits[18]:0x3_ffff, bits[18]:0x3_5d50, bits[18]:0xf22a, bits[18]:0x1_5cd7, bits[18]:0x1_55c5, bits[18]:0x2_35c5, bits[18]:0x1_5755, bits[18]:0x0]; bits[45]:0x2_0000_0000"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x3_ffff, bits[18]:0x8aa2, bits[18]:0x0, bits[18]:0xa80c, bits[18]:0x2_aaaa, bits[18]:0x2_abaa, bits[18]:0x3_ffff, bits[18]:0xcf9a, bits[18]:0x2_a8aa, bits[18]:0x0, bits[18]:0x1_6684]; bits[45]:0x0"
//     args: "bits[28]:0x800; [bits[18]:0x1_004a, bits[18]:0x3_94f8, bits[18]:0x3800, bits[18]:0x840, bits[18]:0x2_ac01, bits[18]:0x1_ffff, bits[18]:0x1_9984, bits[18]:0x1_ffff, bits[18]:0xc25, bits[18]:0x1_5555, bits[18]:0x400]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_ffff, bits[18]:0x3_fdff, bits[18]:0x3_bff3, bits[18]:0x1_efd7, bits[18]:0x3_7be3, bits[18]:0x3_b7e5, bits[18]:0x1_bfbf, bits[18]:0x200, bits[18]:0x2_fddb, bits[18]:0x2_aaaa, bits[18]:0x100]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x555_5555; [bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_d556, bits[18]:0x1_3361, bits[18]:0x0, bits[18]:0xf14c, bits[18]:0x1_dc67, bits[18]:0x1_d576, bits[18]:0x1_55d5, bits[18]:0x1_47d5]; bits[45]:0xca_5b17_40c6"
//     args: "bits[28]:0x948_557e; [bits[18]:0x1_0000, bits[18]:0x0, bits[18]:0x715f, bits[18]:0x556e, bits[18]:0xd92a, bits[18]:0x1_5555, bits[18]:0x3cb7, bits[18]:0x1ca6, bits[18]:0x55f4, bits[18]:0x4bfe, bits[18]:0x557e]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x0, bits[18]:0xbe68, bits[18]:0x3_ffff, bits[18]:0x2_a828, bits[18]:0x4a36, bits[18]:0x2_ae28, bits[18]:0x2_2aa6, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_0aaf, bits[18]:0x2_a8ae]; bits[45]:0x1a2_b10f_a442"
//     args: "bits[28]:0x0; [bits[18]:0x0, bits[18]:0xd521, bits[18]:0x23ca, bits[18]:0x1_6707, bits[18]:0x1_0000, bits[18]:0x0, bits[18]:0x0, bits[18]:0x2_0188, bits[18]:0x80a, bits[18]:0x1_5555, bits[18]:0x0]; bits[45]:0x800_0022_a9aa"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_ffff, bits[18]:0x3_86e0, bits[18]:0x3_bb3c, bits[18]:0x3_ff7b, bits[18]:0x2_4e3a, bits[18]:0x34ab, bits[18]:0x3_ffff, bits[18]:0xaddf, bits[18]:0x0, bits[18]:0x2_f3f6, bits[18]:0x3_dbf9]; bits[45]:0xbbd_f7d7_cbae"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_3baa, bits[18]:0x1_f1ff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_efdf, bits[18]:0x3_fffd, bits[18]:0x1_47d7, bits[18]:0x1_ffff]; bits[45]:0x1c42_b6ff_f775"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_ff5f, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_9fab, bits[18]:0x1_5e3e, bits[18]:0x3_ffff, bits[18]:0x3_e2fb, bits[18]:0x1_ffff, bits[18]:0x0]; bits[45]:0xf9d_3efa_7abd"
//     args: "bits[28]:0x800_0000; [bits[18]:0x3_ffff, bits[18]:0x10, bits[18]:0x815d, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x4000, bits[18]:0x3_7d3f, bits[18]:0x3_fbdf, bits[18]:0x1_fed7, bits[18]:0xffff, bits[18]:0x3_aeec, bits[18]:0x3_ffff, bits[18]:0xbea6, bits[18]:0x2_aaaa, bits[18]:0x3_ffbe, bits[18]:0x1_df7b]; bits[45]:0x0"
//     args: "bits[28]:0x0; [bits[18]:0x8000, bits[18]:0x1_5555, bits[18]:0xca, bits[18]:0x0, bits[18]:0x51b4, bits[18]:0x3_ffff, bits[18]:0x2_d41f, bits[18]:0x9410, bits[18]:0x0, bits[18]:0x14, bits[18]:0x2]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x3_f77f, bits[18]:0x1_0000, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_df1e, bits[18]:0x1_5555, bits[18]:0x3_f6ff, bits[18]:0x2_3eff]; bits[45]:0xaa7_7657_fe61"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_5175, bits[18]:0x1_5555, bits[18]:0x1255, bits[18]:0x690, bits[18]:0x1_7774, bits[18]:0x1_5555, bits[18]:0x8, bits[18]:0x2_55df, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0xe957]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2, bits[18]:0x1, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_e77c, bits[18]:0x2_67de, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_d8ad]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0x2; [bits[18]:0x1_ffff, bits[18]:0x10, bits[18]:0x1_5555, bits[18]:0x23, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0xec11]; bits[45]:0x1025_4035_1d2f"
//     args: "bits[28]:0x0; [bits[18]:0x106, bits[18]:0x2_7ef4, bits[18]:0x1_ffff, bits[18]:0x8000, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x8, bits[18]:0x5010, bits[18]:0x2888, bits[18]:0x0, bits[18]:0x1040]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x2_aaaa, bits[18]:0x3_e7e5, bits[18]:0x1000, bits[18]:0x3_7efd, bits[18]:0x3_ffff, bits[18]:0x3_3e7e, bits[18]:0x3_ffef, bits[18]:0x3_e7ed, bits[18]:0x3_3d7b, bits[18]:0x1_496b, bits[18]:0x1_5555]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_6efa, bits[18]:0x3_ffff, bits[18]:0x9d57, bits[18]:0x3_dfff, bits[18]:0x1_c7ae, bits[18]:0x3_5feb, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_d7ae, bits[18]:0x3_5fdf, bits[18]:0x3_6cd7]; bits[45]:0x1f18_9e75_a9e7"
//     args: "bits[28]:0x10_0000; [bits[18]:0x2_a993, bits[18]:0x1_0807, bits[18]:0x2_aaaa, bits[18]:0x9103, bits[18]:0x2_4b68, bits[18]:0x0, bits[18]:0x0, bits[18]:0x8800, bits[18]:0x2_aaaa, bits[18]:0x1_2400, bits[18]:0x0]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_f7ff, bits[18]:0x3_ffff, bits[18]:0x3_2f66, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x3_eb5f, bits[18]:0x0, bits[18]:0x2_df9f, bits[18]:0x3_ffff, bits[18]:0x3_fbff, bits[18]:0x3_ffff]; bits[45]:0x1318_72e3_69fb"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_0000, bits[18]:0x3_e599, bits[18]:0x1_5fff, bits[18]:0x3_bbff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_fe3e, bits[18]:0x3_fffb, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x3_dbef]; bits[45]:0x0"
//     args: "bits[28]:0x0; [bits[18]:0x2_70e7, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_1c7c, bits[18]:0x2_aaaa, bits[18]:0x2_0880, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x400, bits[18]:0x1_5bb0, bits[18]:0x1_0000]; bits[45]:0x18ca_a410_39a0"
//     args: "bits[28]:0x0; [bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x400, bits[18]:0x0, bits[18]:0x0, bits[18]:0x8004, bits[18]:0x3_743d, bits[18]:0x2_aaaa]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_ffdf, bits[18]:0x1_5555, bits[18]:0x3_7b35, bits[18]:0x1_5555, bits[18]:0x3_ff33, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_376b, bits[18]:0x1_5555, bits[18]:0x1_defe]; bits[45]:0xfff_fffe_aaaa"
//     args: "bits[28]:0x0; [bits[18]:0x2_aaaa, bits[18]:0x1_f3a1, bits[18]:0x40, bits[18]:0x3_fc28, bits[18]:0x2_aaaa, bits[18]:0x195, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x78ec, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa]; bits[45]:0x2a8_da20_9040"
//     args: "bits[28]:0xd36_63f0; [bits[18]:0x2_03f0, bits[18]:0x2_aaaa, bits[18]:0xeb24, bits[18]:0x3_23f0, bits[18]:0x2_eb54, bits[18]:0x2_13a9, bits[18]:0x2_63f0, bits[18]:0x7214, bits[18]:0x6029, bits[18]:0xa837, bits[18]:0x2af0]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x70ae, bits[18]:0x2_e02c, bits[18]:0x2_bfeb, bits[18]:0x3_ffed]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x2_0000, bits[18]:0x3_ffff, bits[18]:0x3_ef7f, bits[18]:0x2_aaaa, bits[18]:0x1_ff3d, bits[18]:0xc7ff, bits[18]:0x3_7f9f, bits[18]:0xc57, bits[18]:0x1_ffff, bits[18]:0x3_5f8f, bits[18]:0x1_ffff]; bits[45]:0x162c_bfdc_62ed"
//     args: "bits[28]:0x0; [bits[18]:0x0, bits[18]:0x5420, bits[18]:0x0, bits[18]:0x1_0000, bits[18]:0x402, bits[18]:0x80, bits[18]:0x2_429c, bits[18]:0x50a, bits[18]:0x2_853d, bits[18]:0x4000, bits[18]:0x0]; bits[45]:0x1_0000_0000"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_a8ea, bits[18]:0x2_ab7e, bits[18]:0x2_8a8c, bits[18]:0xeae7, bits[18]:0x1_ffff, bits[18]:0x1_0000, bits[18]:0x2_aaaa, bits[18]:0x2_0000, bits[18]:0x3_ffff, bits[18]:0x1_4cbd, bits[18]:0x2_fbaa]; bits[45]:0x915_a064_5249"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_ffff, bits[18]:0x3_fbff, bits[18]:0x0, bits[18]:0x2_e6cb, bits[18]:0x3_f5bb, bits[18]:0x3_bfdb, bits[18]:0x2_aaaa, bits[18]:0x2_3775, bits[18]:0x2_aaaa, bits[18]:0x1_f967, bits[18]:0x3_ffff]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x80, bits[18]:0x1_2fff, bits[18]:0x1_51bb, bits[18]:0x3_08d5, bits[18]:0x3_7ff7, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0x0; [bits[18]:0x2_4930, bits[18]:0x2_ed9f, bits[18]:0x3_8ac4, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1226, bits[18]:0x2_8100, bits[18]:0x50, bits[18]:0x2000, bits[18]:0x1_5555, bits[18]:0x4998]; bits[45]:0xa02_6942_ce47"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x7559, bits[18]:0x2_aaaa, bits[18]:0x3_5555, bits[18]:0x1_e84d, bits[18]:0x1_5555, bits[18]:0x1_5174, bits[18]:0x1_ffff, bits[18]:0x3_71f7, bits[18]:0x1_5555]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_f3b7, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1000, bits[18]:0x2_ffcb, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x2_aaaa]; bits[45]:0xcbf_fdfc_ab8a"
//     args: "bits[28]:0xc9e_e699; [bits[18]:0x0, bits[18]:0xe7b9, bits[18]:0x2_e5e9, bits[18]:0x2_f598, bits[18]:0x1_ffff, bits[18]:0x2_283a, bits[18]:0x2_0000, bits[18]:0x0, bits[18]:0x2_a5fe, bits[18]:0x1_ffff, bits[18]:0x2_e691]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_5efd, bits[18]:0x5ffb, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_dffe, bits[18]:0x3_fede, bits[18]:0x0, bits[18]:0x800, bits[18]:0x1_fef9, bits[18]:0x3_ffbf, bits[18]:0x80]; bits[45]:0x1fdf_fffe_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_ffff, bits[18]:0x3_bfff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ea5f, bits[18]:0x1_4cfe, bits[18]:0x3_ffff, bits[18]:0x2_ffbf, bits[18]:0x3_fdff, bits[18]:0x2_aaaa, bits[18]:0x1_5555]; bits[45]:0x0"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_eedc, bits[18]:0x3_38d7, bits[18]:0x100, bits[18]:0x10, bits[18]:0x1_5555, bits[18]:0x2000, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_deff, bits[18]:0x1_5555, bits[18]:0x2_dbff]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0x0; [bits[18]:0x450, bits[18]:0x0, bits[18]:0x2_521d, bits[18]:0x100, bits[18]:0x1_ffff, bits[18]:0x4, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1000, bits[18]:0x20]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x0; [bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x8005, bits[18]:0x0, bits[18]:0x0, bits[18]:0x80, bits[18]:0x104c, bits[18]:0x1525, bits[18]:0x2_2e76, bits[18]:0x1_ffff, bits[18]:0x1_ffff]; bits[45]:0x50c_113c_4f14"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_fbff, bits[18]:0x1_bcf0, bits[18]:0x4000, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_fd6f, bits[18]:0x3_fffb, bits[18]:0x800, bits[18]:0x3_ffff, bits[18]:0x2_cc7f, bits[18]:0x3_7ffd]; bits[45]:0x0"
//     args: "bits[28]:0x555_5555; [bits[18]:0x3_ffff, bits[18]:0x100, bits[18]:0x1_d375, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5615, bits[18]:0x1_5556, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_45d5, bits[18]:0x1000]; bits[45]:0xaaa_3622_96b8"
//     args: "bits[28]:0x20; [bits[18]:0x121, bits[18]:0x1_a261, bits[18]:0x81, bits[18]:0x4000, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_447a, bits[18]:0x30, bits[18]:0x3c09, bits[18]:0x3_ffff]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x0, bits[18]:0x3_0e53, bits[18]:0x2_adee, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_aaaa, bits[18]:0x2_eaae, bits[18]:0x2_aaea, bits[18]:0xe8cd, bits[18]:0x4, bits[18]:0x1_ffff]; bits[45]:0x1_0000"
//     args: "bits[28]:0x0; [bits[18]:0x0, bits[18]:0x3_1040, bits[18]:0x8080, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_1009, bits[18]:0x3_0478, bits[18]:0x0, bits[18]:0x984f, bits[18]:0x1_0000]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0x200_0000; [bits[18]:0x8208, bits[18]:0x3_ffff, bits[18]:0x3_a10c, bits[18]:0x0, bits[18]:0x1_0620, bits[18]:0x581a, bits[18]:0x264, bits[18]:0x2_aaaa, bits[18]:0x1_9444, bits[18]:0x840, bits[18]:0x2_cc36]; bits[45]:0x983_0da6_659b"
//     args: "bits[28]:0x0; [bits[18]:0x0, bits[18]:0x1c46, bits[18]:0x5248, bits[18]:0x2104, bits[18]:0x1_9afe, bits[18]:0x3000, bits[18]:0x3_ffff, bits[18]:0xcf4, bits[18]:0x3c10, bits[18]:0x1_0028, bits[18]:0x0]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x800, bits[18]:0x1_ffff, bits[18]:0x2_ddaa, bits[18]:0x2_37b7, bits[18]:0x3_e67f, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_f7df, bits[18]:0x2_0000, bits[18]:0x1_5555, bits[18]:0x2_af69, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1000, bits[18]:0x2_aaaa, bits[18]:0x200, bits[18]:0x400, bits[18]:0x3_ff9f]; bits[45]:0x79b_4b9a_198a"
//     args: "bits[28]:0x555_5555; [bits[18]:0x80, bits[18]:0x8729, bits[18]:0x1_ffff, bits[18]:0xf513, bits[18]:0x0, bits[18]:0x1_dc92, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_d7d1, bits[18]:0x400, bits[18]:0x3_507f]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x0; [bits[18]:0x1_5555, bits[18]:0x2c, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_0604, bits[18]:0x1_a4d9, bits[18]:0x300, bits[18]:0x3_29b5, bits[18]:0x0, bits[18]:0x180, bits[18]:0x1_3929]; bits[45]:0x82_8e00_adbe"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x2_aeaa, bits[18]:0x2_2bab, bits[18]:0x1_9a2e, bits[18]:0x2_a3ea, bits[18]:0x3_ffff, bits[18]:0x2_e8a6, bits[18]:0x0]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x555_5555; [bits[18]:0x800, bits[18]:0x400, bits[18]:0x1_0435, bits[18]:0x1_4755, bits[18]:0x3_4425, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_51cd, bits[18]:0x1_5555, bits[18]:0x0]; bits[45]:0x1d2e_8c45_7a7a"
//     args: "bits[28]:0x0; [bits[18]:0x1500, bits[18]:0x1_1a66, bits[18]:0x0, bits[18]:0x48, bits[18]:0x2, bits[18]:0x3_ffff, bits[18]:0x1_2025, bits[18]:0x1_0ac2, bits[18]:0x1b8, bits[18]:0x2_1c92, bits[18]:0xe103]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x0; [bits[18]:0x5103, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_2041, bits[18]:0x2_aaaa, bits[18]:0x303, bits[18]:0x0, bits[18]:0x3_0ebb, bits[18]:0x1041, bits[18]:0x1_d0ec, bits[18]:0x3_ffff]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x0; [bits[18]:0x1_ffff, bits[18]:0x7290, bits[18]:0x0, bits[18]:0x2_0454, bits[18]:0x1_1a3a, bits[18]:0x0, bits[18]:0x1_0001, bits[18]:0x2, bits[18]:0x2_aaaa, bits[18]:0x84b, bits[18]:0x40]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_ffff, bits[18]:0x1_10d5, bits[18]:0x2_8ea8, bits[18]:0x585c, bits[18]:0x0, bits[18]:0x1_713f, bits[18]:0x1_5555, bits[18]:0x1_c555, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x3_dfd6]; bits[45]:0x0"
//     args: "bits[28]:0x0; [bits[18]:0x3_7592, bits[18]:0x0, bits[18]:0x0, bits[18]:0xd48, bits[18]:0x3_ffff, bits[18]:0x24, bits[18]:0x2_aaaa, bits[18]:0x671, bits[18]:0x1_56ac, bits[18]:0x1032, bits[18]:0x2_a540]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x3_abaa, bits[18]:0x1_0331, bits[18]:0x2_3aaa, bits[18]:0x2_aaaa, bits[18]:0x3_a222, bits[18]:0x2_abf6, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_b1ef, bits[18]:0x1_5555]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_ae2a, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_babb, bits[18]:0x1_dcff, bits[18]:0x2_aaaa, bits[18]:0x2_aaea, bits[18]:0x0, bits[18]:0x3_ebfc, bits[18]:0x3_ffff, bits[18]:0x2_aaca]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x40; [bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x80c5, bits[18]:0x40, bits[18]:0x3_ffff, bits[18]:0x50, bits[18]:0x9187, bits[18]:0x40, bits[18]:0x1_ffff, bits[18]:0x2_09ea]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0x8; [bits[18]:0x20, bits[18]:0x1008, bits[18]:0x2, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x8529, bits[18]:0x1_40b0, bits[18]:0x1_cca9, bits[18]:0x18, bits[18]:0x1, bits[18]:0x8]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x1_5877, bits[18]:0x2_aa8a, bits[18]:0x2_abaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_ca22, bits[18]:0x3_a8af, bits[18]:0x3_0117, bits[18]:0x3_aba3, bits[18]:0x2_aa8a, bits[18]:0x400]; bits[45]:0xc3d_0dc7_652c"
//     args: "bits[28]:0x0; [bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x820, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_985a, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1, bits[18]:0x1_5555]; bits[45]:0x100_8000_f7f9"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_ffff, bits[18]:0x1_9d23, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_f2ff, bits[18]:0x2_bd5f, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_22ba, bits[18]:0x3_f773, bits[18]:0x3_ffff]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_bfdb, bits[18]:0x1_5555, bits[18]:0x3_b6fb, bits[18]:0x2_aaaa, bits[18]:0x1_d5ff, bits[18]:0x0, bits[18]:0xaede, bits[18]:0x1_5555, bits[18]:0x2_bded, bits[18]:0x0, bits[18]:0xdb7f]; bits[45]:0xbfd_fffe_0000"
//     args: "bits[28]:0xefe_b659; [bits[18]:0x3_adfb, bits[18]:0x2_b2b9, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x3_b6cc, bits[18]:0x2_b682, bits[18]:0x1_c619, bits[18]:0x2_a3d1, bits[18]:0x2_b671, bits[18]:0x2_877e, bits[18]:0x0]; bits[45]:0x0"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x4, bits[18]:0x1_fddf, bits[18]:0x1_6e31, bits[18]:0x1_5555, bits[18]:0xbfb4, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_be7e, bits[18]:0x1_ffff, bits[18]:0x3_ffdf]; bits[45]:0x0"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_bd7e, bits[18]:0x1_bf7d, bits[18]:0x3_ffeb, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0xb3c, bits[18]:0x0, bits[18]:0x3_787e, bits[18]:0x0, bits[18]:0x3_dfbf, bits[18]:0x3_ff6e]; bits[45]:0xdf7_98fc_a910"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_5400, bits[18]:0x20, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0xa278, bits[18]:0x1_4354, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff]; bits[45]:0x100_0000_0000"
//     args: "bits[28]:0x800_0000; [bits[18]:0x1_5555, bits[18]:0x4000, bits[18]:0x0, bits[18]:0x1_84ab, bits[18]:0x2_aaaa, bits[18]:0x3_22d9, bits[18]:0x4000, bits[18]:0x1_8402, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x3_ffff]; bits[45]:0x1209_2004_6828"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x2_6ac4, bits[18]:0x3_7d86, bits[18]:0x2_aaaa, bits[18]:0x2_5d2e, bits[18]:0x1_dece, bits[18]:0x1_ffff, bits[18]:0x3_185c, bits[18]:0x2_aaaa, bits[18]:0x3_f7df, bits[18]:0x0, bits[18]:0x3_ebcb]; bits[45]:0x1dfd_bfbe_ffff"
//     args: "bits[28]:0x796_d83e; [bits[18]:0x2_d819, bits[18]:0x3_f83e, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_f83e, bits[18]:0xf83c, bits[18]:0x2_406f, bits[18]:0x2_b851, bits[18]:0x2_d810, bits[18]:0x2_f972, bits[18]:0x3_f836]; bits[45]:0xf2d_b06c_0444"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x1_ffff, bits[18]:0x1_b6ff, bits[18]:0x4, bits[18]:0x2_efb8, bits[18]:0x3_ffbb, bits[18]:0x0, bits[18]:0x3_bfd4, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_fbef]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xbdc_0831; [bits[18]:0x1_29f3, bits[18]:0x3_0873, bits[18]:0x0, bits[18]:0x8b34, bits[18]:0x871, bits[18]:0x4812, bits[18]:0x2000, bits[18]:0x0, bits[18]:0x61b5, bits[18]:0x831, bits[18]:0x0]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_ffff, bits[18]:0x2_0577, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_545b, bits[18]:0x1_5555, bits[18]:0x1_55c5, bits[18]:0x1_918d, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_6555]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_3917, bits[18]:0x2_7343, bits[18]:0x3_fed3, bits[18]:0x1_dfbb, bits[18]:0x1_5555, bits[18]:0x1_9ace, bits[18]:0x1_fffc, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_9cd5]; bits[45]:0x1_0000_0000"
//     args: "bits[28]:0x0; [bits[18]:0x3_e32f, bits[18]:0x10, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_8111, bits[18]:0x2_aaaa, bits[18]:0x2_5692, bits[18]:0x1_5555, bits[18]:0x400, bits[18]:0x2_aaaa]; bits[45]:0x1_0000"
//     args: "bits[28]:0x8_0000; [bits[18]:0x9f02, bits[18]:0x2_aaaa, bits[18]:0x6b8c, bits[18]:0x2_0842, bits[18]:0x41, bits[18]:0x2_8803, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3_fa8e, bits[18]:0x2_a856, bits[18]:0x7004]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0x0; [bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_0130, bits[18]:0x2_3eee, bits[18]:0x0, bits[18]:0xa700, bits[18]:0x1_08c1, bits[18]:0x3_1090, bits[18]:0x4110, bits[18]:0x1_3867]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_1490, bits[18]:0x1_5555, bits[18]:0x6dff, bits[18]:0x9920, bits[18]:0x3_ffff, bits[18]:0x1_1fe7, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_0000, bits[18]:0x3_7ff7]; bits[45]:0x17bd_df2f_a0af"
//     args: "bits[28]:0x20_0000; [bits[18]:0x2_9518, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_8261, bits[18]:0x2_aaaa, bits[18]:0x1_09c8, bits[18]:0x1_428c, bits[18]:0x2_aaaa, bits[18]:0x2_0080, bits[18]:0x8, bits[18]:0x9c01]; bits[45]:0x749_1209_5144"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_1475, bits[18]:0xd341, bits[18]:0x1_5145, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_c555, bits[18]:0x2_f7de, bits[18]:0x2_070d, bits[18]:0x2_aaaa, bits[18]:0x1_7772, bits[18]:0x2_0bac]; bits[45]:0xa2a_aeea_0841"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_5555, bits[18]:0x1_55aa, bits[18]:0x400, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x3_ffcf, bits[18]:0x8, bits[18]:0x10, bits[18]:0x3_6e7d, bits[18]:0x3_df97, bits[18]:0x3_ffff]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x0; [bits[18]:0x3_ffff, bits[18]:0x1_040a, bits[18]:0xf0c0, bits[18]:0x12, bits[18]:0x3_ffff, bits[18]:0x5213, bits[18]:0x0, bits[18]:0x1_0305, bits[18]:0x2_aaaa, bits[18]:0x1_0433, bits[18]:0x2_aaaa]; bits[45]:0x0"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_71f5, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x2_abaa, bits[18]:0x2_b47f, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_82b9, bits[18]:0x3_ffff, bits[18]:0x3_908d]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x8; [bits[18]:0x3_ffff, bits[18]:0x2_e3c8, bits[18]:0x1_08ba, bits[18]:0x104b, bits[18]:0x3_ffff, bits[18]:0x3_fe8c, bits[18]:0x3_b804, bits[18]:0x688a, bits[18]:0x1_5555, bits[18]:0x8000, bits[18]:0x2_5e1e]; bits[45]:0x0"
//     args: "bits[28]:0x80_0000; [bits[18]:0x1_5555, bits[18]:0x7bc2, bits[18]:0x5040, bits[18]:0x2_aed5, bits[18]:0x4, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x800, bits[18]:0x2_1101]; bits[45]:0xa96_0095_d5bc"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2000, bits[18]:0x2_37f8, bits[18]:0x2_8aba, bits[18]:0x0, bits[18]:0x2_3aea, bits[18]:0x2_ba2a, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x2_acc0, bits[18]:0x2_aaaa, bits[18]:0x1_ffff]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_5557, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_5655, bits[18]:0x3_b091, bits[18]:0x3_5557, bits[18]:0x1_9312, bits[18]:0x3_ffff, bits[18]:0x3_c5d5, bits[18]:0x4000]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x1_5555, bits[18]:0xa2aa, bits[18]:0x2_aa23, bits[18]:0x2_2aee, bits[18]:0x33ac, bits[18]:0xaea8, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x8d93, bits[18]:0x2_1eeb, bits[18]:0x3_a18b]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_22aa, bits[18]:0x1, bits[18]:0x3_ba28, bits[18]:0x2_aa8a, bits[18]:0x2_aaaa, bits[18]:0x2_82a2, bits[18]:0x1_5555, bits[18]:0x3_a82a, bits[18]:0x3_aa91, bits[18]:0x8, bits[18]:0x2_aaaa]; bits[45]:0x10"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_5f1d, bits[18]:0x2_f59f, bits[18]:0x2_e4bc, bits[18]:0x3_fbfe, bits[18]:0x1_5555, bits[18]:0x3_dff7, bits[18]:0x2_c5ad, bits[18]:0x2_7bb0, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff]; bits[45]:0x5de_dfdf_0010"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_78ab, bits[18]:0x2_aaaa, bits[18]:0x2_aa9e, bits[18]:0x2_aaaa, bits[18]:0x88eb, bits[18]:0x2_e2a1, bits[18]:0x2_aaaa, bits[18]:0x8fe4, bits[18]:0x2_aaba, bits[18]:0x2_aaaa, bits[18]:0x3_aaaa]; bits[45]:0xd74_4d3d_5954"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_aaaa, bits[18]:0xd1b7, bits[18]:0xbeaa, bits[18]:0x1_8aab, bits[18]:0x2_8a8a, bits[18]:0x3_ffff, bits[18]:0x2_8278, bits[18]:0x2_aaaa, bits[18]:0x2_2aba, bits[18]:0x2_a346, bits[18]:0x1_ffff]; bits[45]:0x1a45_1f15_2240"
//     args: "bits[28]:0x56b_ee55; [bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_6d55, bits[18]:0x2_aaaa, bits[18]:0x2_9311, bits[18]:0x2_f0cf, bits[18]:0x2_aaaa, bits[18]:0x3_ec55, bits[18]:0x3_ad55]; bits[45]:0x1fe3_2bca_8482"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffdf, bits[18]:0x2_fbfd, bits[18]:0x2_fd7e, bits[18]:0x1_ffff, bits[18]:0x3_fff7, bits[18]:0x3_ffff, bits[18]:0x3_f5ef, bits[18]:0x1_5555, bits[18]:0x3_87b8]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_2922, bits[18]:0x2_aaaa, bits[18]:0x1ffd, bits[18]:0x3_ffff, bits[18]:0x2, bits[18]:0x0, bits[18]:0x1_2afa, bits[18]:0x1_5555, bits[18]:0x1_b8b8]; bits[45]:0x497_5754_14f7"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x2_f7df, bits[18]:0x2_aaaa, bits[18]:0x2_b620, bits[18]:0x3_ffbf, bits[18]:0x1_fd6c, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1, bits[18]:0x3_ffff]; bits[45]:0x400_0000"
//     args: "bits[28]:0x555_5555; [bits[18]:0x2_aaaa, bits[18]:0x1_9d14, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2000, bits[18]:0xe215, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_15dd, bits[18]:0x1_5d04]; bits[45]:0x1555_5555_5555"
//     args: "bits[28]:0x400; [bits[18]:0x2_2c00, bits[18]:0x480, bits[18]:0x1_5555, bits[18]:0x3ec0, bits[18]:0x2_0401, bits[18]:0x8000, bits[18]:0x8000, bits[18]:0x2_1d43, bits[18]:0x400, bits[18]:0x25dc, bits[18]:0x400]; bits[45]:0x144b_28ad_deec"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_fbff, bits[18]:0x3_d355, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ddeb, bits[18]:0x3_ff7f, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_bdf7, bits[18]:0xda0f]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0x695_70e9; [bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_b86b, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_30e9, bits[18]:0x308c, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_20f9]; bits[45]:0xf38_eb52_0aae"
//     args: "bits[28]:0x645_376e; [bits[18]:0x3_ffff, bits[18]:0x1_376e, bits[18]:0x1_336e, bits[18]:0x1, bits[18]:0x3b3c, bits[18]:0x10, bits[18]:0x1_5555, bits[18]:0x1_327e, bits[18]:0x1_376e, bits[18]:0x1_376e, bits[18]:0x1_ffff]; bits[45]:0x1fff_ffff_ffff"
//     args: "bits[28]:0x0; [bits[18]:0xa001, bits[18]:0x1_a000, bits[18]:0x1_0421, bits[18]:0x1b41, bits[18]:0x1_42e8, bits[18]:0x1_ffff, bits[18]:0xaf7, bits[18]:0x0, bits[18]:0x20, bits[18]:0x0, bits[18]:0x1_e27e]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x20_0000; [bits[18]:0x1_383b, bits[18]:0x10, bits[18]:0x100, bits[18]:0x2_63c9, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x40, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x80, bits[18]:0x3_ffff]; bits[45]:0x0"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x1_dfbc, bits[18]:0x3_ffff, bits[18]:0x3_fdff, bits[18]:0x0, bits[18]:0x2_b337, bits[18]:0x1_fd7f, bits[18]:0x0, bits[18]:0x3_fffb, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff]; bits[45]:0xd1f_4e68_e4ea"
//     args: "bits[28]:0x7ff_ffff; [bits[18]:0x3_7d5d, bits[18]:0x3_7e33, bits[18]:0x0, bits[18]:0x2_fef7, bits[18]:0x0, bits[18]:0x1_efa4, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_6eb3, bits[18]:0x3_fffd, bits[18]:0x3_ffdd]; bits[45]:0x16ff_fbf7_f7ff"
//     args: "bits[28]:0x0; [bits[18]:0x1_1b6a, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x220, bits[18]:0x1_0000, bits[18]:0x1_0020, bits[18]:0x1_2080, bits[18]:0x0, bits[18]:0x1000, bits[18]:0x3_a000, bits[18]:0x1_da23]; bits[45]:0x18_0278_dfff"
//     args: "bits[28]:0x555_5555; [bits[18]:0x1_fd27, bits[18]:0x3_57c4, bits[18]:0x1_5515, bits[18]:0x1_f7c4, bits[18]:0x1_5555, bits[18]:0x3_45bf, bits[18]:0x1_c515, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_c295, bits[18]:0x1_e549]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xaaa_aaaa; [bits[18]:0x2_aaaa, bits[18]:0x2_0000, bits[18]:0x2_aaaa, bits[18]:0x3_8168, bits[18]:0x3_ffff, bits[18]:0x2_e2a2, bits[18]:0x1_ffff, bits[18]:0x8000, bits[18]:0x1_abaa, bits[18]:0x2_aaab, bits[18]:0x2_aeaa]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x80, bits[18]:0x0, bits[18]:0x1_fff3, bits[18]:0x3_feff, bits[18]:0x3_fdfb, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_60af, bits[18]:0x0, bits[18]:0x2_aaaa]; bits[45]:0x0"
//     args: "bits[28]:0x0; [bits[18]:0x1ae1, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x100, bits[18]:0x1105, bits[18]:0xa00, bits[18]:0x1_4780, bits[18]:0x2_677a, bits[18]:0x1_ffff, bits[18]:0x2018, bits[18]:0x2_c1a2]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x4, bits[18]:0x3_677b, bits[18]:0x1_ffff, bits[18]:0x1_6771, bits[18]:0x1_ff36, bits[18]:0x3_fffb, bits[18]:0x1_d35d, bits[18]:0x3_ffff, bits[18]:0x3_7955, bits[18]:0x3_ffff, bits[18]:0xe754]; bits[45]:0x834_0c09_67e5"
//     args: "bits[28]:0xfff_ffff; [bits[18]:0x3_f7fe, bits[18]:0x32a1, bits[18]:0x3_f5fb, bits[18]:0x3_fbff, bits[18]:0x0, bits[18]:0x3_afd7, bits[18]:0x3_ff7c, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x3_dfff]; bits[45]:0xaaa_aaaa_aaaa"
//     args: "bits[28]:0x2000; [bits[18]:0x3_20d0, bits[18]:0x2_aaaa, bits[18]:0x7040, bits[18]:0x1_ffff, bits[18]:0x1_4da4, bits[18]:0x2000, bits[18]:0x2002, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x7410, bits[18]:0x6e1a]; bits[45]:0x396_a881_8422"
//     args: "bits[28]:0x200; [bits[18]:0x0, bits[18]:0x3_a0a1, bits[18]:0x2_2109, bits[18]:0x2_0000, bits[18]:0x124, bits[18]:0x248, bits[18]:0x3_ef6a, bits[18]:0x1_5555, bits[18]:0x2200, bits[18]:0x2_8a38, bits[18]:0x800]; bits[45]:0xfff_ffff_ffff"
//     args: "bits[28]:0x6e2_7f88; [bits[18]:0x3_ffff, bits[18]:0x3_7e88, bits[18]:0x2_6a88, bits[18]:0x2_ff88, bits[18]:0x2_7f88, bits[18]:0x3_3eea, bits[18]:0x1_5555, bits[18]:0x1_e32f, bits[18]:0x1_fe98, bits[18]:0x0, bits[18]:0x3_7788]; bits[45]:0x1fff_ffff_ffff"
//   }
// }
// 
// END_CONFIG
type x0 = u18;
fn main(x1: u28, x2: x0[11], x3: u45) -> (u28, u37, bool) {
    {
        let x4: u43 = x3[0+:u43];
        let x5: u43 = x4[0+:u43];
        let x6: bool = or_reduce(x1);
        let x7: bool = x6 | x6;
        let x8: bool = x7 << 0;
        let x9: u28 = x1 * x8 as u28;
        let x10: u14 = u14:0x2aaa;
        let x11: bool = x1 >= x10 as u28;
        let x12: u2 = x7 ++ x6;
        let x13: u28 = x1 * x1;
        let x14: u15 = x1[x9+:u15];
        let x15: u28 = bit_slice_update(x1, x14, x9);
        let x16: x0[22] = x2 ++ x2;
        let x17: u37 = x3[-37:];
        let x18: u28 = x7 as u28 * x13;
        let x19: u28 = x18 / u28:0xfff_ffff;
        let x20: u28 = bit_slice_update(x15, x11, x19);
        let x21: u14 = clz(x10);
        (x19, x17, x7)
    }
}
