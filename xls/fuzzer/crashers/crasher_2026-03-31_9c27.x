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
// exception: "SampleError: Result miscompare for sample 0:\nargs: (bits[4]:0x5,
// bits[42]:0x1c2_fde8_8ba1); bits[36]:0xf778_7496; (bits[53]:0x0, bits[36]:0xa_aaaa_aaaa,
// bits[3]:0x4); bits[53]:0x1_eef0_e92f_ffff; bits[28]:0xfff_ffff\nevaluated opt IR (JIT), evaluated
// opt IR (interpreter), simulated =\n   (bits[53]:0x1_eef0_e92f_fffe, bits[31]:0x0,
// bits[90]:0x3d_de1d_25ff_ffc0_0000_00fd, bits[19]:0x7_7d51, bits[19]:0x7_7d51)\nevaluated unopt IR
// (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[53]:0x1_eef0_e92f_fffe,
// bits[31]:0x0, bits[90]:0x3d_de1d_25ff_ffdf_ffff_fffd, bits[19]:0x7_7d51, bits[19]:0x7_7d51)"
// issue: https://github.com/google/xls/issues/4023
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the
//     specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full
//     throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "(bits[4]:0x5, bits[42]:0x1c2_fde8_8ba1); bits[36]:0xf778_7496; (bits[53]:0x0,
//     bits[36]:0xa_aaaa_aaaa, bits[3]:0x4); bits[53]:0x1_eef0_e92f_ffff; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x4, bits[42]:0x400_0000); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0x7_be5d_deb4_6110, bits[36]:0x0, bits[3]:0x5); bits[53]:0x1_0000_0000_0000;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x5, bits[42]:0x1ff_ffff_ffff); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x6_e9f6_a567_f390, bits[36]:0x7_ffff_ffff, bits[3]:0x2);
//     bits[53]:0xf_efcf_37e9_15d5; bits[28]:0x555_5555"
//     args: "(bits[4]:0x5, bits[42]:0x400); bits[36]:0xa_aaaa_aaaa; (bits[53]:0xa_aaaa_aaaa_aaaa,
//     bits[36]:0x0, bits[3]:0x7); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0xede_56cc"
//     args: "(bits[4]:0xf, bits[42]:0x0); bits[36]:0xf_2918_ddb8; (bits[53]:0x0,
//     bits[36]:0xa_aaaa_aaaa, bits[3]:0x3); bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x0"
//     args: "(bits[4]:0xa, bits[42]:0x142_8db3_e468); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x15_5557_5554_2040, bits[36]:0xc_893f_fb2d, bits[3]:0x2);
//     bits[53]:0x1f_ffff_ffff_ffff; bits[28]:0x0"
//     args: "(bits[4]:0x1, bits[42]:0x0); bits[36]:0x20_0000; (bits[53]:0x8_0000_0000_0000,
//     bits[36]:0x2_0000_0000, bits[3]:0x0); bits[53]:0x5_5ff5_f0f3_f7e7; bits[28]:0xd7_f7eb"
//     args: "(bits[4]:0x7, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x10_0000;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x2_e8ac_3239, bits[3]:0x4);
//     bits[53]:0x200_0000_0000; bits[28]:0x52_2020"
//     args: "(bits[4]:0xa, bits[42]:0x1ff_ffff_ffff); bits[36]:0x5_b367_db47;
//     (bits[53]:0x19_6c49_24c9_cbc8, bits[36]:0x5_f66e_dbc5, bits[3]:0x0);
//     bits[53]:0x8_9d15_07c5_542f; bits[28]:0x5eb_8fbf"
//     args: "(bits[4]:0x4, bits[42]:0x0); bits[36]:0xe_0f1a_a6b1; (bits[53]:0x8_dcfe_8481_8b81,
//     bits[36]:0x2_0000_0000, bits[3]:0x1); bits[53]:0x0; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x0, bits[42]:0xf7_bacb_8095); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0x20_0000_0000, bits[36]:0x0, bits[3]:0x7); bits[53]:0x19_6142_df8c_2695;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0xc, bits[42]:0x155_5555_5555); bits[36]:0x0; (bits[53]:0x15_5555_5555_5555,
//     bits[36]:0x7_ffff_ffff, bits[3]:0x2); bits[53]:0x15_5555_5555_5555; bits[28]:0x0"
//     args: "(bits[4]:0x5, bits[42]:0x2b1_2101_dc48); bits[36]:0x8; (bits[53]:0x2_2f52_73b0_2189,
//     bits[36]:0xf_ffff_ffff, bits[3]:0x5); bits[53]:0x20_0000; bits[28]:0x979_3044"
//     args: "(bits[4]:0xf, bits[42]:0x0); bits[36]:0xf_ffff_ffff; (bits[53]:0xf_ffff_ffff_ffff,
//     bits[36]:0x80_0000, bits[3]:0x1); bits[53]:0x10_0000_0000; bits[28]:0x400"
//     args: "(bits[4]:0xa, bits[42]:0x10_0000_0000); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0xd_cf1d_eeb2, bits[3]:0x3);
//     bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x0, bits[42]:0x1ff_ffff_ffff); bits[36]:0x0; (bits[53]:0x1b_0884_6b7d_4ab1,
//     bits[36]:0xf_ffff_ffff, bits[3]:0x5); bits[53]:0x2_4802_0903_c555; bits[28]:0x117_c7f5"
//     args: "(bits[4]:0x1, bits[42]:0x200); bits[36]:0xf_ffff_ffff; (bits[53]:0x1d_fbb6_f37f_535d,
//     bits[36]:0xb_70b5_3b6f, bits[3]:0x7); bits[53]:0x1f_3de6_b167_72d5; bits[28]:0x4000"
//     args: "(bits[4]:0xf, bits[42]:0x0); bits[36]:0xa_aaaa_aaaa; (bits[53]:0xa_aaaa_aaaa_aaaa,
//     bits[36]:0xb_8aba_acbe, bits[3]:0x2); bits[53]:0x10; bits[28]:0x0"
//     args: "(bits[4]:0x6, bits[42]:0x1ff_ffff_ffff); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x15_5555_5555_5555, bits[36]:0xa_aaaa_aaaa, bits[3]:0x4);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x555_5555"
//     args: "(bits[4]:0x0, bits[42]:0x392_1edb_7a8c); bits[36]:0xa_aaaa_aaaa; (bits[53]:0x0,
//     bits[36]:0xc_fcf5_55d7, bits[3]:0x2); bits[53]:0xf_ffff_ffff_ffff; bits[28]:0xff7_766d"
//     args: "(bits[4]:0x8, bits[42]:0x1e7_c343_0b23); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x15_5555_5555_5555, bits[36]:0xf_ffff_ffff, bits[3]:0x3);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x0"
//     args: "(bits[4]:0x1, bits[42]:0x3ff_ffff_ffff); bits[36]:0x0; (bits[53]:0xa_aaaa_aaaa_aaaa,
//     bits[36]:0xf_ffff_ffff, bits[3]:0x5); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x20_0000"
//     args: "(bits[4]:0xf, bits[42]:0x0); bits[36]:0x5_5555_5555; (bits[53]:0xa_2aaa_aaaa_2000,
//     bits[36]:0x2_beef_2d95, bits[3]:0x2); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x2aa_aaba"
//     args: "(bits[4]:0x4, bits[42]:0x35_cc2d_cf97); bits[36]:0x20; (bits[53]:0xc_2204_4830_0ada,
//     bits[36]:0x0, bits[3]:0x3); bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x2000"
//     args: "(bits[4]:0x0, bits[42]:0x3ff_ffff_ffff); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0x5_5555_5555, bits[3]:0x3);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x10_0000"
//     args: "(bits[4]:0xa, bits[42]:0x2_0000); bits[36]:0x5_5555_5555;
//     (bits[53]:0x17_c25e_f1bf_888d, bits[36]:0x5_4554_5575, bits[3]:0x0); bits[53]:0x0;
//     bits[28]:0x0"
//     args: "(bits[4]:0x5, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x5_5555_5555;
//     (bits[53]:0x15_5555_5555_5555, bits[36]:0x5_5555_5555, bits[3]:0x6);
//     bits[53]:0x2_faba_acbb_bbeb; bits[28]:0x40_0000"
//     args: "(bits[4]:0x0, bits[42]:0x4_0000); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0x0, bits[3]:0x2); bits[53]:0x5_c315_115c_a99c;
//     bits[28]:0xdc_8b9c"
//     args: "(bits[4]:0x2, bits[42]:0x0); bits[36]:0xa_aaaa_aaaa; (bits[53]:0x1f_ffff_ffff_ffff,
//     bits[36]:0x9_bacb_9ae3, bits[3]:0x3); bits[53]:0x15_5555_5555_5555; bits[28]:0x0"
//     args: "(bits[4]:0x0, bits[42]:0x264_76a0_5c51); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0xf_bfff_f7ff, bits[3]:0x4);
//     bits[53]:0x2_0000_0000_0000; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x0, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x3_1e8f_ea4a_362d, bits[36]:0x7_ffff_bff7, bits[3]:0x7);
//     bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0xb, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0x78bd_326c, bits[3]:0x2);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x7ae_cff7"
//     args: "(bits[4]:0x5, bits[42]:0x262_7fca_86ec); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x5_5555_5555, bits[3]:0x3);
//     bits[53]:0x15_b60f_00ec_a192; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x2, bits[42]:0x400_0000); bits[36]:0xf_ffff_ffff; (bits[53]:0x0,
//     bits[36]:0x7_ffff_ffff, bits[3]:0x2); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x1, bits[42]:0x20_0000_0000); bits[36]:0x4000_0000;
//     (bits[53]:0x8046_0800_2aba, bits[36]:0x20, bits[3]:0x0); bits[53]:0x1c_cd98_b09e_ea3c;
//     bits[28]:0x4000"
//     args: "(bits[4]:0xa, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0xc_81c8_aa90;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x5_dab1_3b58, bits[3]:0x3);
//     bits[53]:0x19_0191_5523_ddbf; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0xa, bits[42]:0x1ff_ffff_ffff); bits[36]:0x100_0000;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0x7_ffff_ffff, bits[3]:0x1); bits[53]:0x200_0000_0008;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x5, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x0; (bits[53]:0x4000_0000_0000,
//     bits[36]:0xf_ffff_ffff, bits[3]:0x3); bits[53]:0x0; bits[28]:0x0"
//     args: "(bits[4]:0x7, bits[42]:0x1ff_ffff_ffff); bits[36]:0x0; (bits[53]:0x0, bits[36]:0x4004,
//     bits[3]:0x0); bits[53]:0xc_386e_3734_05ee; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x6, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x9_7b72_7479_d0dd, bits[36]:0x0, bits[3]:0x2); bits[53]:0xa_aaaa_aaaa_aaaa;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x1, bits[42]:0x3ff_ffff_ffff); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x15_5555_5555_5555, bits[36]:0x5_5555_5555, bits[3]:0x7);
//     bits[53]:0x5_16e9_7176_8bcf; bits[28]:0x555_5555"
//     args: "(bits[4]:0x5, bits[42]:0x3ff_ffff_ffff); bits[36]:0x5_5555_5555;
//     (bits[53]:0xb89e_e264_172f, bits[36]:0xf_5571_5555, bits[3]:0x0);
//     bits[53]:0x1f_ffff_ffff_ffff; bits[28]:0x555_5555"
//     args: "(bits[4]:0x7, bits[42]:0x3ff_ffff_ffff); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0x8_0000_0000, bits[3]:0x3); bits[53]:0x0;
//     bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x7, bits[42]:0x20_0000); bits[36]:0x4000_0000; (bits[53]:0x0,
//     bits[36]:0x5_5555_5555, bits[3]:0x4); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x130_1309"
//     args: "(bits[4]:0xa, bits[42]:0x3ff_ffff_ffff); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0xa_aaa8_aaaa, bits[3]:0x3);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x0"
//     args: "(bits[4]:0x4, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x7_fd7f_fdfd_aafa, bits[36]:0xf_ffff_ffff, bits[3]:0x7); bits[53]:0x800_0000_0000;
//     bits[28]:0x80_0000"
//     args: "(bits[4]:0xf, bits[42]:0x3ff_ffff_ffff); bits[36]:0x0; (bits[53]:0x0,
//     bits[36]:0xf_ffff_ffff, bits[3]:0x1); bits[53]:0x2_8030_477b_dda9; bits[28]:0x0"
//     args: "(bits[4]:0x4, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x0, bits[3]:0x7); bits[53]:0xf_ffff_ffff_ffff;
//     bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x4, bits[42]:0x2000_0000); bits[36]:0x0; (bits[53]:0xa_aaaa_aaaa_aaaa,
//     bits[36]:0x940_0000, bits[3]:0x0); bits[53]:0x200; bits[28]:0x0"
//     args: "(bits[4]:0x4, bits[42]:0x20_0000); bits[36]:0x5_5555_5555; (bits[53]:0x2000_0000_0000,
//     bits[36]:0x2, bits[3]:0x2); bits[53]:0x2000_0000_0000; bits[28]:0x575_d555"
//     args: "(bits[4]:0xa, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x15_5555_5555_5555, bits[36]:0x5_5555_5555, bits[3]:0x1);
//     bits[53]:0x15_5555_5555_5555; bits[28]:0x0"
//     args: "(bits[4]:0x0, bits[42]:0x1ff_ffff_ffff); bits[36]:0x1000_0000;
//     (bits[53]:0xacd0_0024_7440, bits[36]:0xa_aaaa_aaaa, bits[3]:0x2); bits[53]:0x8602_22d5_dfdf;
//     bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x5, bits[42]:0x0); bits[36]:0xa_aaaa_aaaa; (bits[53]:0x4_5155_415c_aa2e,
//     bits[36]:0x5_5555_5555, bits[3]:0x0); bits[53]:0x2_0000_0000_0000; bits[28]:0x200_0000"
//     args: "(bits[4]:0x8, bits[42]:0x0); bits[36]:0xf_ffff_ffff; (bits[53]:0x1f_ffff_ffff_ffff,
//     bits[36]:0x0, bits[3]:0x3); bits[53]:0x16_5715_70e3_af20; bits[28]:0x3d6_b00c"
//     args: "(bits[4]:0x7, bits[42]:0x8000); bits[36]:0x7_ffff_ffff; (bits[53]:0x1f_ffff_ffff_ffff,
//     bits[36]:0x5_5555_5555, bits[3]:0x7); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0xf, bits[42]:0x3ff_ffff_ffff); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0xa_aaaa_aaaa, bits[3]:0x4);
//     bits[53]:0x7_f6fd_fb76_2890; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x0, bits[42]:0x0); bits[36]:0x100_0000; (bits[53]:0x4_0342_1248_84e2,
//     bits[36]:0x4_8380_9020, bits[3]:0x0); bits[53]:0x15_5555_5555_5555; bits[28]:0x184_a0a9"
//     args: "(bits[4]:0x8, bits[42]:0x1ff_ffff_ffff); bits[36]:0x5_5555_5555; (bits[53]:0x8000,
//     bits[36]:0x0, bits[3]:0x3); bits[53]:0x200_0000_0000; bits[28]:0x0"
//     args: "(bits[4]:0xf, bits[42]:0x155_5555_5555); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x7_ffff_ffff, bits[3]:0x2);
//     bits[53]:0x17_65ae_f533_d777; bits[28]:0xb8b_56e9"
//     args: "(bits[4]:0x4, bits[42]:0x1ff_ffff_ffff); bits[36]:0x5_5555_5555;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0x5_5555_5555, bits[3]:0x7); bits[53]:0x0;
//     bits[28]:0xdd5_edf9"
//     args: "(bits[4]:0x0, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x0; (bits[53]:0x15_5555_5555_5555,
//     bits[36]:0x0, bits[3]:0x4); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0xa8a_a32a"
//     args: "(bits[4]:0xa, bits[42]:0x1ff_ffff_ffff); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x15_5555_5555_5555, bits[36]:0xa_aaaa_aaaa, bits[3]:0x2);
//     bits[53]:0xd_5d55_7455_5555; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x5, bits[42]:0x800_0000); bits[36]:0xf_ffff_ffff; (bits[53]:0x20_0000_0000,
//     bits[36]:0x5_5555_5555, bits[3]:0x7); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x0"
//     args: "(bits[4]:0x5, bits[42]:0x200_0000_0000); bits[36]:0x1_0000_0000;
//     (bits[53]:0x2_2002_0000_fffd, bits[36]:0x5_5555_5555, bits[3]:0x5); bits[53]:0x200_0000_0000;
//     bits[28]:0x80_0000"
//     args: "(bits[4]:0x4, bits[42]:0x0); bits[36]:0x7_ffff_ffff; (bits[53]:0x15_5555_5555_5555,
//     bits[36]:0x6_f77e_7d3e, bits[3]:0x7); bits[53]:0x1f_ffff_ffff_ffff; bits[28]:0xfbf_fdff"
//     args: "(bits[4]:0x5, bits[42]:0x155_5555_5555); bits[36]:0x5_5555_5555;
//     (bits[53]:0xa_aeae_8212_0800, bits[36]:0x5_5555_5555, bits[3]:0x3);
//     bits[53]:0x1a_3cb9_afae_69a2; bits[28]:0x0"
//     args: "(bits[4]:0x9, bits[42]:0x3ff_ffff_ffff); bits[36]:0xa_aaaa_aaaa; (bits[53]:0x2_0000,
//     bits[36]:0x5_5555_5555, bits[3]:0x5); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x1, bits[42]:0x54_bc37_240c); bits[36]:0x5_5555_5555;
//     (bits[53]:0xe_afee_ea2d_7515, bits[36]:0xf_ffff_ffff, bits[3]:0x5);
//     bits[53]:0xa_8c8a_5aa3_efff; bits[28]:0x8_0000"
//     args: "(bits[4]:0xa, bits[42]:0x10_0000_0000); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x13_6d91_0514_c640, bits[36]:0x5_5555_5555, bits[3]:0x0);
//     bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x0"
//     args: "(bits[4]:0x9, bits[42]:0x22a_e4a6_fde4); bits[36]:0x5_5555_5555;
//     (bits[53]:0xe_894e_09a1_75b4, bits[36]:0x5_5555_5555, bits[3]:0x5);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x20_0000"
//     args: "(bits[4]:0x0, bits[42]:0x8); bits[36]:0xa_aaaa_aaaa; (bits[53]:0x0,
//     bits[36]:0x7_ffff_ffff, bits[3]:0x6); bits[53]:0x1f_155d_15c6_0080; bits[28]:0x243_1443"
//     args: "(bits[4]:0x4, bits[42]:0xaf_827d_c2e5); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0x7_ffff_ffff, bits[3]:0x5);
//     bits[53]:0x15_5555_5555_5555; bits[28]:0xae7_6792"
//     args: "(bits[4]:0x7, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x4000;
//     (bits[53]:0x7_6982_3c06_9376, bits[36]:0x0, bits[3]:0x0); bits[53]:0x1f_ffff_ffff_ffff;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x0, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x5_d755_d476_ef59, bits[36]:0x7_ffff_ffff, bits[3]:0x0);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0xf, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0xf_ffff_ffff, bits[3]:0x2); bits[53]:0x0;
//     bits[28]:0x555_5555"
//     args: "(bits[4]:0x0, bits[42]:0x215_d30f_ced2); bits[36]:0x800;
//     (bits[53]:0x18_9f44_cb1e_e986, bits[36]:0x200_0000, bits[3]:0x2);
//     bits[53]:0x1f_ffff_ffff_ffff; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x7, bits[42]:0x10_0000_0000); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x1d_5e5d_5756_2003, bits[36]:0x5_5555_5555, bits[3]:0x0);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0xfef_cdf7"
//     args: "(bits[4]:0x0, bits[42]:0x3ff_ffff_ffff); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x1e_ee3b_6aa5_a429, bits[36]:0x0, bits[3]:0x3); bits[53]:0x1f_ffff_ffff_ffff;
//     bits[28]:0x555_5555"
//     args: "(bits[4]:0x0, bits[42]:0x1ff_ffff_ffff); bits[36]:0x5_5555_5555; (bits[53]:0x0,
//     bits[36]:0x5_5555_755d, bits[3]:0x5); bits[53]:0x0; bits[28]:0x0"
//     args: "(bits[4]:0x1, bits[42]:0x155_5555_5555); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0x7_ffff_ffff, bits[3]:0x5);
//     bits[53]:0x1c_1eec_1d4b_d724; bits[28]:0x0"
//     args: "(bits[4]:0xa, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x5_5555_5555; (bits[53]:0x1,
//     bits[36]:0xc_32f9_bcec, bits[3]:0x4); bits[53]:0xf_ffff_ffff_ffff; bits[28]:0x555_5555"
//     args: "(bits[4]:0x9, bits[42]:0x155_5555_5555); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0xe_78af_c57f_0a94, bits[36]:0xf_ffff_ffff, bits[3]:0x5);
//     bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x8c5_5233"
//     args: "(bits[4]:0x8, bits[42]:0x155_5555_5555); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0xa_aaaa_aaaa, bits[3]:0x3);
//     bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x9cc_6927"
//     args: "(bits[4]:0x5, bits[42]:0x1ff_ffff_ffff); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0xb_b3ff_efef, bits[3]:0x7); bits[53]:0x0;
//     bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x7, bits[42]:0x155_5555_5555); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x5_5555_5555, bits[3]:0x7); bits[53]:0x4_0000_0000;
//     bits[28]:0x400_0000"
//     args: "(bits[4]:0x2, bits[42]:0x155_5555_5555); bits[36]:0x4000; (bits[53]:0x200_0000,
//     bits[36]:0x2_2000_4024, bits[3]:0x0); bits[53]:0x15_5555_5555_5555; bits[28]:0x555_5555"
//     args: "(bits[4]:0x4, bits[42]:0x3ff_ffff_ffff); bits[36]:0x5_5555_5555; (bits[53]:0x800_0000,
//     bits[36]:0x5_7a18_d76f, bits[3]:0x5); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x8c4_3aeb"
//     args: "(bits[4]:0x7, bits[42]:0x0); bits[36]:0xf_9827_d1a3; (bits[53]:0x1,
//     bits[36]:0xa_aaaa_aaaa, bits[3]:0x7); bits[53]:0x1000; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0xf, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0x4, bits[3]:0x5); bits[53]:0x10_dab9_fdfc_80a8;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0xb, bits[42]:0x40_0000_0000); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x8_6742_9e13, bits[3]:0x1);
//     bits[53]:0xf_ffff_ffff_ffff; bits[28]:0xfaa_8ace"
//     args: "(bits[4]:0xa, bits[42]:0x2000_0000); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x7_ed1b_f9db_c64e, bits[36]:0x9_eddf_8dff, bits[3]:0x3); bits[53]:0x0;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x7, bits[42]:0x3ff_ffff_ffff); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x0, bits[3]:0x5); bits[53]:0xf_ffff_ffff_ffff;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0xa, bits[42]:0x0); bits[36]:0x0; (bits[53]:0xf_ffff_ffff_ffff,
//     bits[36]:0x100_0000, bits[3]:0x2); bits[53]:0x1; bits[28]:0x400_0000"
//     args: "(bits[4]:0x0, bits[42]:0x0); bits[36]:0x9_5138_4c66; (bits[53]:0x8_aff0_c8de_0f42,
//     bits[36]:0x7_ffff_ffff, bits[3]:0x7); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x8a6_a82f"
//     args: "(bits[4]:0x7, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0x7_a9aa_2e83, bits[3]:0x0);
//     bits[53]:0x15_5555_5555_5555; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x1, bits[42]:0x80_0000_0000); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0xa_aba2_2aea, bits[3]:0x0);
//     bits[53]:0x1_42d9_da94_9103; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x9, bits[42]:0x1ff_ffff_ffff); bits[36]:0x5_5555_5555;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x0, bits[3]:0x5); bits[53]:0xa_a8aa_aaab_5d55;
//     bits[28]:0x8ec_4cda"
//     args: "(bits[4]:0x5, bits[42]:0x155_5555_5555); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0xf_3fd7_78df, bits[3]:0x7);
//     bits[53]:0x1f_ffff_fffe_ffff; bits[28]:0x0"
//     args: "(bits[4]:0x8, bits[42]:0x0); bits[36]:0x7_ffff_ffff; (bits[53]:0x1f_ffff_ffff_ffff,
//     bits[36]:0xb_ed04_32c1, bits[3]:0x5); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x0"
//     args: "(bits[4]:0x5, bits[42]:0x155_5555_5555); bits[36]:0xa_aaaa_aaaa; (bits[53]:0x0,
//     bits[36]:0x7_ffff_ffff, bits[3]:0x3); bits[53]:0x15_5555_5555_5555; bits[28]:0x555_5555"
//     args: "(bits[4]:0xf, bits[42]:0x3ff_ffff_ffff); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0xe_3be7_3fb0_f5f1, bits[36]:0xe_2eee_cfb5, bits[3]:0x6); bits[53]:0x0;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0xa, bits[42]:0x155_5555_5555); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x6_a841_f6d4, bits[3]:0x7);
//     bits[53]:0x2_0000_0000_0000; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x7, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x5_5555_5555;
//     (bits[53]:0x100_0000_0000, bits[36]:0x3_f755_f1e1, bits[3]:0x2); bits[53]:0xf_ffff_ffff_ffff;
//     bits[28]:0xc2d_4aae"
//     args: "(bits[4]:0x4, bits[42]:0x0); bits[36]:0x5_5555_5555; (bits[53]:0x0,
//     bits[36]:0xc_0295_f485, bits[3]:0x3); bits[53]:0x11_bf74_28f1_7ed8; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x6, bits[42]:0x100_0000); bits[36]:0xd_684b_79a7;
//     (bits[53]:0x11_51ec_ced1_3d42, bits[36]:0xf_ffff_ffff, bits[3]:0x5);
//     bits[53]:0x15_5555_5555_5555; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x7, bits[42]:0x255_22b7_f61b); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0x5_5555_5555, bits[3]:0x7);
//     bits[53]:0xe_6726_53f8_05c1; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0x5, bits[42]:0x80_0000); bits[36]:0xa_aaaa_aaaa; (bits[53]:0x0,
//     bits[36]:0x400, bits[3]:0x2); bits[53]:0x1f_ffff_ffff_ffff; bits[28]:0xaae_aaaa"
//     args: "(bits[4]:0x8, bits[42]:0xb9_fdb5_b676); bits[36]:0x0; (bits[53]:0x15_5555_5555_5555,
//     bits[36]:0x0, bits[3]:0x0); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x5, bits[42]:0x1ff_ffff_ffff); bits[36]:0x20; (bits[53]:0xf_ffff_ffff_ffff,
//     bits[36]:0xa_aaaa_aaaa, bits[3]:0x1); bits[53]:0x15_5555_5555_5555; bits[28]:0x4000"
//     args: "(bits[4]:0x5, bits[42]:0x155_5555_5555); bits[36]:0x800; (bits[53]:0x0,
//     bits[36]:0x7_ffff_ffff, bits[3]:0x7); bits[53]:0x1e11_1160_1f34; bits[28]:0x166_1f34"
//     args: "(bits[4]:0x0, bits[42]:0x3ff_ffff_ffff); bits[36]:0x0; (bits[53]:0x2_0000_0000_0000,
//     bits[36]:0x2008_0001, bits[3]:0x5); bits[53]:0x5_c001_0000_0640; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0x7, bits[42]:0x155_5555_5555); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x15_65dd_514c_0020, bits[36]:0x4, bits[3]:0x7); bits[53]:0xa_aaaa_aaaa_aaaa;
//     bits[28]:0xe2_24f8"
//     args: "(bits[4]:0xf, bits[42]:0x1ff_ffff_ffff); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x15_5555_5554_ffff, bits[36]:0x6_a2ba_acb3, bits[3]:0x2);
//     bits[53]:0x15_5555_5555_5555; bits[28]:0x502_e4f5"
//     args: "(bits[4]:0x0, bits[42]:0x0); bits[36]:0xf_ffff_ffff; (bits[53]:0x2000,
//     bits[36]:0xf_ffff_ffff, bits[3]:0x7); bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0xfeb_5cfc"
//     args: "(bits[4]:0x7, bits[42]:0x2c7_242f_2452); bits[36]:0x4_fc0d_e93a;
//     (bits[53]:0x8_fb1b_dbbc_1004, bits[36]:0x4_fc0a_e33c, bits[3]:0x3);
//     bits[53]:0x15_5555_5555_5555; bits[28]:0x7ff_ffff"
//     args: "(bits[4]:0xa, bits[42]:0xfe_5120_1dac); bits[36]:0xa_aaaa_aaaa;
//     (bits[53]:0x16_8593_5d88_8d06, bits[36]:0xa_0aac_83ce, bits[3]:0x5); bits[53]:0x1000;
//     bits[28]:0xc3d_5fae"
//     args: "(bits[4]:0x5, bits[42]:0x155_5555_5555); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0xa_aaaa_aaaa, bits[3]:0x3);
//     bits[53]:0x1f_fe3f_6ffe_ffef; bits[28]:0xfff_dfff"
//     args: "(bits[4]:0x5, bits[42]:0x3ff_ffff_ffff); bits[36]:0x800; (bits[53]:0xba28_4748_0c8a,
//     bits[36]:0xf_ffff_ffff, bits[3]:0x0); bits[53]:0x10; bits[28]:0xd92_6fe1"
//     args: "(bits[4]:0xa, bits[42]:0x155_5555_5555); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xa_959f_edf2_4081, bits[36]:0x0, bits[3]:0x3); bits[53]:0xf_ffff_ffff_ffff;
//     bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x7, bits[42]:0x155_5555_5555); bits[36]:0xc_92c0_3330;
//     (bits[53]:0xf_ffff_ffff_ffff, bits[36]:0xc_92c0_7130, bits[3]:0x5);
//     bits[53]:0xa_aaaa_aaaa_aaaa; bits[28]:0x555_5555"
//     args: "(bits[4]:0xf, bits[42]:0x155_5555_5555); bits[36]:0x7_ffff_ffff;
//     (bits[53]:0xa_aaaa_aaaa_aaaa, bits[36]:0xc_bbe9_d9c2, bits[3]:0x2); bits[53]:0x0;
//     bits[28]:0x1000"
//     args: "(bits[4]:0x8, bits[42]:0x3ff_ffff_ffff); bits[36]:0x1000;
//     (bits[53]:0x10_0000_0000_0000, bits[36]:0x4_04d1_480a, bits[3]:0x1);
//     bits[53]:0x10_8081_a086_916f; bits[28]:0x4a9_5519"
//     args: "(bits[4]:0xa, bits[42]:0x10_0000); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0xf_ffff_ffff, bits[3]:0x7);
//     bits[53]:0x1f_ffff_ffff_ffff; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x6, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x8000_0000;
//     (bits[53]:0x1f_ffff_ffff_ffff, bits[36]:0x7_ffff_ffff, bits[3]:0x3);
//     bits[53]:0x15_5555_5555_5555; bits[28]:0xdb1_5011"
//     args: "(bits[4]:0x5, bits[42]:0x1ff_ffff_ffff); bits[36]:0x2b0b_a297;
//     (bits[53]:0x15_5555_5555_5555, bits[36]:0xa_aaaa_aaaa, bits[3]:0x5);
//     bits[53]:0x19_7594_1f08_63f9; bits[28]:0x343_a297"
//     args: "(bits[4]:0x7, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x9_b95d_0b94;
//     (bits[53]:0x13_80ba_2b49_6177, bits[36]:0x7_ffff_ffff, bits[3]:0x5);
//     bits[53]:0x17_72be_172a_1400; bits[28]:0xaaa_aaaa"
//     args: "(bits[4]:0xf, bits[42]:0x2aa_aaaa_aaaa); bits[36]:0x0; (bits[53]:0x100_0000_0000,
//     bits[36]:0x5_5555_5555, bits[3]:0x7); bits[53]:0x12_005b_bc37_2d9a; bits[28]:0xfff_ffff"
//     args: "(bits[4]:0x5, bits[42]:0x1ff_ffff_ffff); bits[36]:0xf_ffff_ffff;
//     (bits[53]:0x7_95f2_fcfb_a7ec, bits[36]:0x7_ffff_ffff, bits[3]:0x7); bits[53]:0x800_0000_0000;
//     bits[28]:0xfff_ffff"
//   }
// }
//
// END_CONFIG
fn main
    (x0: (u4, s42), x1: u36, x2: (u53, s36, u3), x3: u53, x4: u28) -> (u53, u31, uN[90], u19, u19) {
    {
        let x5: u36 = one_hot_sel(u6:0x3f, [x1, x1, x1, x1, x1, x1]);
        let x6: u36 = ctz(x1);
        let x7: u36 = x6 as u36 - x1;
        let x8: u36 = x6 % u36:0xf_ffff_ffff;
        let x9: u36 = -x1;
        let x10: u36 = x8 + x8;
        let x11: u36 = x8[x9+:u36];
        let x12: u7 = u7:0b10_1010;
        let x13: bool = x8 as u36 != x9;
        let x14: u36 = -x1;
        let x15: u31 = x14[x3+:u31];
        let x16: u53 = x3 ^ x6 as u53;
        let x17: u36 = -x10;
        let x18: u36 = x17[0+:u36];
        let x19: bool = x5 <= x6 as u36;
        let x20: u36 = -x7;
        let x22: uN[90] = x16 ++ x18 ++ x13;
        let x23: uN[90] = -x22;
        let x24: u53 = bit_slice_update(x16, x10, x20);
        let x25: bool = or_reduce(x1);
        let x26: u19 = u19:0x7_7d51;
        let x27: u36 = bit_slice_update(x10, x10, x24);
        let x28: u36 = -x8;
        (x16, x15, x22, x26, x26)
    }
}
