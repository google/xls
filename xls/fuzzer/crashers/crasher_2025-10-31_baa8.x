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
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[22]:0x2a_aaaa; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0x7_ffff_ffff\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[14]:0x0, [bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x2, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x2, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0], [bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x2, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x2, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0], bits[1]:0x0, bits[9]:0x0, bits[22]:0x2_aaaa)\nevaluated unopt IR (JIT) =\n   (bits[14]:0x0, [bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x2, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0], [bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x0, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x1, bits[2]:0x2, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0, bits[2]:0x0], bits[1]:0x0, bits[9]:0x0, bits[22]:0x2_aaaa)"
// issue: "https://github.com/google/xls/issues/3269"
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
//   disable_unopt_interpreter: false
// }
// inputs {
//   function_args {
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x29_f5f6; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[35]:0x2_ac8a_dc9c"
//     args: "bits[22]:0x0; bits[58]:0x280_8c8f_9a4b_cf17; bits[35]:0x4_498a_2c52"
//     args: "bits[22]:0x15_5555; bits[58]:0x75_0552_aeab_eaaa; bits[35]:0x2_aeab_eaaa"
//     args: "bits[22]:0x17_586c; bits[58]:0x17d_86cf_ffff_ffff; bits[35]:0x7_ffff_feff"
//     args: "bits[22]:0x29_6ea1; bits[58]:0x400; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x15_5555; bits[58]:0x11f_5001_4505_0282; bits[35]:0x1_518d_0282"
//     args: "bits[22]:0x8_edf0; bits[58]:0x20e_da83_9f28_f5dc; bits[35]:0x0"
//     args: "bits[22]:0xf_3fd2; bits[58]:0x246_dd0d_cd3f_efff; bits[35]:0x5_5555_5555"
//     args: "bits[22]:0x0; bits[58]:0x358_7416_1d68_7054; bits[35]:0x2_aaaa_aaaa"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_aaa7_fdbf_f7ff; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x24e_dcf2_b392_1281; bits[35]:0x3_35df_dec4"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x1ff_fff8_589f_068c; bits[35]:0x5_38de_00a4"
//     args: "bits[22]:0x1000; bits[58]:0x155_5555_5555_5555; bits[35]:0x2a10_1b9e"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x20_0000_0000_0000; bits[35]:0x0"
//     args: "bits[22]:0x0; bits[58]:0x22b_039d_b7bf_9f97; bits[35]:0x8220_4018"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x3ff_efc2_8000_7001; bits[35]:0x2_0259_7061"
//     args: "bits[22]:0x0; bits[58]:0x264_a26c_05aa_92a0; bits[35]:0x4_05aa_92a0"
//     args: "bits[22]:0x15_5555; bits[58]:0x0; bits[35]:0x1_0018_2051"
//     args: "bits[22]:0x8_0000; bits[58]:0x10_433f_efdf_dfef; bits[35]:0x800_0000"
//     args: "bits[22]:0x15_5555; bits[58]:0x155_5550_8098_1180; bits[35]:0x0"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_38eb_a28b_baf0; bits[35]:0x5_f5dd_2682"
//     args: "bits[22]:0x35_f1a3; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[35]:0x7_135e_1f58"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x228_e53f_af77_fe2e; bits[35]:0x7_af77_fc2e"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x20_0000; bits[58]:0x155_5555_5555_5555; bits[35]:0x0"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_baa5_5557_5553; bits[35]:0x5_4457_5753"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x800; bits[35]:0x5_5555_5555"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x36a_2aa6_d402_d042; bits[35]:0x4_0000"
//     args: "bits[22]:0x15_5555; bits[58]:0x155_6163_18aa_e612; bits[35]:0x2_aaa2_a440"
//     args: "bits[22]:0x3a_5abe; bits[58]:0x0; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x1e7_c792_aa8a_e2a2; bits[35]:0x2_a7de_e102"
//     args: "bits[22]:0x20_3e8d; bits[58]:0x37b_6ac3_2afa_6b68; bits[35]:0x3_2aba_6b68"
//     args: "bits[22]:0x15_5555; bits[58]:0x155_5445_5155_555d; bits[35]:0x5_5155_555d"
//     args: "bits[22]:0x0; bits[58]:0x0; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0x4_0000; bits[58]:0x10_d01f_9ffd_f7af; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x15_5555; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[35]:0x4_adf6_ef08"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x398_8dc5_5d2e_e6ac; bits[35]:0x1000_0000"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x17f_7111_d7db_9faf; bits[35]:0x3_87c2_3d8f"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0x7_fdff_e000"
//     args: "bits[22]:0x2c_8369; bits[58]:0x244_3690_0000_2000; bits[35]:0x3_4c28_a4ac"
//     args: "bits[22]:0x26_0dbd; bits[58]:0x348_9a63_f35f_d3da; bits[35]:0x3_71de_dbd2"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_aaa4_0000_0010; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x800; bits[58]:0x2; bits[35]:0x4_8c92_809d"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x3e1_6f34_0549_6313; bits[35]:0x5_6649_08e8"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_aaa0_8080_0000; bits[35]:0x0"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x3da_af5c_1d15_e0f1; bits[35]:0x1_2011_257a"
//     args: "bits[22]:0x0; bits[58]:0x1607_fdef_ffbf; bits[35]:0x5_f9c2_5cdf"
//     args: "bits[22]:0x800; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x3_570b_fcbf"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x37f_a1f5_1552_0561; bits[35]:0x4_03be_8d04"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x1_0000"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x3_ffff_fffd"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x263_f7cd_537f_196b; bits[35]:0x3_37f7_fda1"
//     args: "bits[22]:0x80; bits[58]:0x21_6815_494e_4735; bits[35]:0x4_494e_6735"
//     args: "bits[22]:0x0; bits[58]:0x52_1180_0380_4142; bits[35]:0x3_e104_21c4"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0x7_7fff_7455"
//     args: "bits[22]:0x0; bits[58]:0xa3_03c8_d126_aa1a; bits[35]:0x2_aaaa_aaaa"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[35]:0x2_2aaa_aaaa"
//     args: "bits[22]:0x0; bits[58]:0x80_8205_f7ff_3ff7; bits[35]:0x82e"
//     args: "bits[22]:0x0; bits[58]:0x5_5555_5555; bits[35]:0x5_4f55_5511"
//     args: "bits[22]:0x400; bits[58]:0x280_f28b_323b_6e05; bits[35]:0x3_323b_6e25"
//     args: "bits[22]:0x2f_1f16; bits[58]:0x3a0_da28_1415_581b; bits[35]:0x5_c32a_e6f5"
//     args: "bits[22]:0x15_5555; bits[58]:0x1d5_5757_5014_75c7; bits[35]:0x8_0000"
//     args: "bits[22]:0x0; bits[58]:0x8_e6ee_dcfe; bits[35]:0x5_5555_5555"
//     args: "bits[22]:0x0; bits[58]:0x200_0007_ffff_ff7f; bits[35]:0x7_ffff_ff7f"
//     args: "bits[22]:0x0; bits[58]:0xb8_ed5b_ac0d_18c7; bits[35]:0x5_5555_5555"
//     args: "bits[22]:0x15_5555; bits[58]:0x155_5550_1000_8010; bits[35]:0x2_0264_8e02"
//     args: "bits[22]:0x0; bits[58]:0x22_0000_04c4_1c00; bits[35]:0x4c4_1c00"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0x2_aaaa_aaaa"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x1ff_fffa_44de_3f50; bits[35]:0x5_3822_d8a9"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x155_5555_5555_5555; bits[35]:0x2_3884_0f5c"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x17c_bb13_8c02_5045; bits[35]:0x1_d6af_be41"
//     args: "bits[22]:0x15_5555; bits[58]:0x159_55d7_faff_bfff; bits[35]:0x2_aaaa_aaaa"
//     args: "bits[22]:0x0; bits[58]:0x48_0000_0000; bits[35]:0xa538_437e"
//     args: "bits[22]:0x4_0000; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x6_8040_3dd5"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x1ea_baa6_5555_75d5; bits[35]:0x7_f63c_78de"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x0; bits[35]:0x1_0000_0000"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_aaa7_fdee_ffff; bits[35]:0x7_fdee_ffff"
//     args: "bits[22]:0x0; bits[58]:0x160_000c_28b8_2020; bits[35]:0x2_aaaa_aaaa"
//     args: "bits[22]:0x0; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0xa008_27fb"
//     args: "bits[22]:0x4_0000; bits[58]:0x240_400a_aaa8_3a2a; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x0; bits[58]:0x281_221b_aaba_aaea; bits[35]:0x3_ebbe_aaea"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x0; bits[35]:0x10"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x1ff_effa_aea8_aaae; bits[35]:0x3_a688_9aa0"
//     args: "bits[22]:0x10; bits[58]:0x4_030f_eff9_ffff; bits[35]:0x18f2_fe4b"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0x15_5555; bits[58]:0x151_55d7_ff7f_e5db; bits[35]:0x5_8f7b_6489"
//     args: "bits[22]:0x0; bits[58]:0x63_8354_d1dd_a642; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0x15_5555; bits[58]:0x4000; bits[35]:0x2_aaaa_be6b"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x4000_0000_0000; bits[35]:0x4000_0000"
//     args: "bits[22]:0x20_0000; bits[58]:0x80_0000_0202_0100; bits[35]:0x4_806e_208d"
//     args: "bits[22]:0x2d_3757; bits[58]:0x0; bits[35]:0x51_1420"
//     args: "bits[22]:0x1_2b03; bits[58]:0x0; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0x15_5555; bits[58]:0x171_75de_8aaa_aaa0; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x3ff_fff4_47d3_d4a1; bits[35]:0x1_b2bf_7246"
//     args: "bits[22]:0x40; bits[58]:0x0; bits[35]:0x2_0840_0210"
//     args: "bits[22]:0x0; bits[58]:0x155_5555_5555_5555; bits[35]:0x4_4799_d7f3"
//     args: "bits[22]:0xe_96be; bits[58]:0xfa_cffa_0452_0606; bits[35]:0xdb96_c000"
//     args: "bits[22]:0x0; bits[58]:0x10_0000; bits[35]:0x201c_8800"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x40_0000_0000; bits[35]:0x5_1555_4008"
//     args: "bits[22]:0x0; bits[58]:0x20_0018_0001; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x155_5555_5555_5555; bits[35]:0x5_555d_7354"
//     args: "bits[22]:0x1f_ffff; bits[58]:0xd6_39e5_a2e1_c9da; bits[35]:0x4_bd15_af65"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x3ef_fe72_c002_0200; bits[35]:0x7_fbfe_e000"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x7_2a56_e86e"
//     args: "bits[22]:0x0; bits[58]:0x1000_0000; bits[35]:0x1000_0000"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x2_0000_0000_0000; bits[35]:0x600_0040"
//     args: "bits[22]:0x1f_ffff; bits[58]:0xfc_997a_b7ff_b9ce; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0x15_5555; bits[58]:0x19c_7542_ebaa_f2ae; bits[35]:0x5_5555_5555"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x2_aaaa_aaaa"
//     args: "bits[22]:0x2a_aaaa; bits[58]:0x2aa_aaaf_ffff_bebf; bits[35]:0x7_fdfa_82b6"
//     args: "bits[22]:0x4_0000; bits[58]:0x200_0000_0000_0000; bits[35]:0x4048_0019"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x1ff_ffff_6fee_ffff; bits[35]:0x3_ffff_ffff"
//     args: "bits[22]:0xa_873b; bits[58]:0x360_533b_d7f4_d1b7; bits[35]:0x0"
//     args: "bits[22]:0x12_3869; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x2c0_6197"
//     args: "bits[22]:0x0; bits[58]:0xc4_a02b_a429_8fa5; bits[35]:0x4_0015_82a1"
//     args: "bits[22]:0x15_5555; bits[58]:0x100_0000_0000; bits[35]:0x2_aaaa_abaa"
//     args: "bits[22]:0x15_5555; bits[58]:0x3c9_b896_6f34_280b; bits[35]:0x8000_0000"
//     args: "bits[22]:0x0; bits[58]:0x200_0a0e_b607_412c; bits[35]:0x0"
//     args: "bits[22]:0x2_0000; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0x1_96dc_77df"
//     args: "bits[22]:0x1f_ffff; bits[58]:0x1cb_b6b8_48b0_4006; bits[35]:0x3_8fbb_e63a"
//     args: "bits[22]:0x0; bits[58]:0x481a_aaa8_a2ab; bits[35]:0x2_8a92_208b"
//     args: "bits[22]:0x15_5555; bits[58]:0x175_5758_f7fa_7fff; bits[35]:0x5_5555_5555"
//     args: "bits[22]:0x10_0000; bits[58]:0x3ff_ffff_ffff_ffff; bits[35]:0x2_aaaa_aaaa"
//     args: "bits[22]:0x1000; bits[58]:0x1ff_ffff_ffff_ffff; bits[35]:0x3_7bff_ffa3"
//     args: "bits[22]:0x3f_ffff; bits[58]:0x0; bits[35]:0x7_ffff_ffff"
//     args: "bits[22]:0x100; bits[58]:0x3_1441_5f5f_6430; bits[35]:0x4eaa"
//   }
// }
// 
// END_CONFIG
type x13 = u2;
type x19 = u22;
fn x30(x31: x13[102], x32: u22) -> (u14, u22) {
    {
        let x33: u22 = ctz(x32);
        let x34: u14 = x33[8+:u14];
        let x35: u2 = x32[17+:u2];
        (x34, x33)
    }
}
fn main(x0: u22, x1: u58, x2: u35) -> (u14, x13[51], x13[51], bool, u9, u22) {
    {
        let x3: u22 = -x0;
        let x4: u22 = x0 >> if x1 >= u58:0b100 { u58:0b100 } else { x1 };
        let x5: u58 = x1 << x0;
        let x6: uN[102] = x3 ++ x3 ++ x5;
        let x7: u35 = -x2;
        let x8: bool = x6 <= x3 as uN[102];
        let x9: uN[102] = ctz(x6);
        let x10: uN[102] = one_hot_sel(x8, [x9]);
        let x11: u35 = ctz(x7);
        let x12: u35 = -x11;
        let x14: x13[51] = x6 as x13[51];
        let x15: u22 = x3[:];
        let x16: x13[102] = x14 ++ x14;
        let x17: x13 = x14[if x1 >= u58:0x10 { u58:0x10 } else { x1 }];
        let x18: u12 = u12:0xfff;
        let x20: x19[3] = [x0, x3, x4];
        let x22: s58 = {
            let x21: (u58, u58) = smulp(x5 as s58, x3 as u58 as s58);
            (x21.0 + x21.1) as s58
        };
        let x24: sN[102] = {
            let x23: (uN[102], uN[102]) = smulp(x8 as uN[102] as sN[102], x6 as sN[102]);
            (x23.0 + x23.1) as sN[102]
        };
        let x25: bool = and_reduce(x6);
        let x26: u58 = !x5;
        let x27: (u58, u35) = (x1, x2);
        let x28: u35 = bit_slice_update(x12, x11, x15);
        let x29: u22 = x11 as u22 & x15;
        let x36: (u14, u22) = x30(x16, x29);
        let (x37, x38) = x30(x16, x29);
        let x39: u35 = bit_slice_update(x2, x7, x26);
        let x40: u58 = -x1;
        let x41: u35 = !x11;
        let x42: x13 = x14[if x28 >= u35:0x5 { u35:0x5 } else { x28 }];
        let x43: u9 = x29[0+:u9];
        let x44: u4 = x1[x0+:u4];
        let x45: u58 = x26 - x6 as u58;
        let x46: x13[6] = array_slice(x14, x0, x13[6]:[x14[u32:0], ...]);
        let x47: u22 = x29 + x12 as u22;
        (x37, x14, x14, x8, x43, x4)
    }
}
