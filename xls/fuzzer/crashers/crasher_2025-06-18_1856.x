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
//     args: "bits[49]:0x4000; bits[11]:0x7ff; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0xe; bits[4]:0x6"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x4fe; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0x5"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x80; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0xa; bits[4]:0x9"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7f3; bits[115]:0x0; bits[4]:0x0; bits[4]:0x0"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0x7; bits[4]:0xb"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x555; bits[115]:0x5_95ee_bfff_ffff_ffbf_edde_ff7f_bfff; bits[4]:0x5; bits[4]:0x0"
//     args: "bits[49]:0x800_0000_0000; bits[11]:0x100; bits[115]:0x0; bits[4]:0x7; bits[4]:0x0"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x230; bits[115]:0x6_b92a_50f5_1fed_6385_2d80_0aee_3582; bits[4]:0x5; bits[4]:0xf"
//     args: "bits[49]:0x1_ef04_33a9_0432; bits[11]:0x2aa; bits[115]:0x7_7804_ebb5_3ccf_afff_bcdf_e60e_afdc; bits[4]:0xa; bits[4]:0xa"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x200; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x5; bits[4]:0x5"
//     args: "bits[49]:0x2000_0000; bits[11]:0x6; bits[115]:0x4_2215_9d61_575c_5115_9c65_2f15_61a6; bits[4]:0x1; bits[4]:0x5"
//     args: "bits[49]:0x0; bits[11]:0x7ff; bits[115]:0x6_f24a_4634_7de8_7683_ab24_b5ab_b709; bits[4]:0x7; bits[4]:0x0"
//     args: "bits[49]:0x2000_0000; bits[11]:0x81; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0x5"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x7_7f7f_fffd_b7fc_0000_0210_0000_0400; bits[4]:0x0; bits[4]:0xf"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x4c7; bits[115]:0xbaaa_9aaa_aa87_6dff_dcef_f777_fdfb; bits[4]:0x6; bits[4]:0x6"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x6be; bits[115]:0x6_ee5b_af6b_e2f7_bbe6_8ebc_f7ef_7c67; bits[4]:0x4; bits[4]:0xe"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x2e; bits[115]:0x80_0000_0000_0000_0000_0000; bits[4]:0x7; bits[4]:0xc"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7db; bits[115]:0x6_3dfe_f3dd_ffbe_cf62_1746_16bb_8753; bits[4]:0xb; bits[4]:0xb"
//     args: "bits[49]:0x20_0000; bits[11]:0x80; bits[115]:0x80aa_a22e_e6ba_98ea_aea2_aa7b_aaa9; bits[4]:0x7; bits[4]:0xa"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x554; bits[115]:0x4_d45c_5511_5711_1555_2755_5545_55d5; bits[4]:0x5; bits[4]:0x7"
//     args: "bits[49]:0x1_f016_3c56_15a9; bits[11]:0x0; bits[115]:0x4000_0000_0000_0000; bits[4]:0xf; bits[4]:0x8"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x56e; bits[115]:0x7_df69_cf7e_7aca_5505_3500_5473_1355; bits[4]:0xb; bits[4]:0xf"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x7b5; bits[115]:0x2_b555_5547_4131_555f_5755_5552_4195; bits[4]:0x5; bits[4]:0x5"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x100; bits[115]:0x4_2ee1_e404_3e3b_813d_c030_aebe_d149; bits[4]:0x9; bits[4]:0xa"
//     args: "bits[49]:0x0; bits[11]:0x17; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0x6"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x7cd; bits[115]:0x2_3e3c_681e_827e_ee02_43af_d591_b6f6; bits[4]:0xa; bits[4]:0xa"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x555; bits[115]:0x0; bits[4]:0xf; bits[4]:0x2"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x2e1; bits[115]:0xf17f_ffff_fffa_ffff_df7f_ff7d_ffff; bits[4]:0x1; bits[4]:0xe"
//     args: "bits[49]:0x8000_0000_0000; bits[11]:0x3ff; bits[115]:0x2_21e2_d491_64a1_fdf7_399f_dd2b_c8f5; bits[4]:0x5; bits[4]:0x2"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x63e; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0xb"
//     args: "bits[49]:0x0; bits[11]:0x190; bits[115]:0x100_0000_0000; bits[4]:0x0; bits[4]:0x7"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x2aa; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0xb"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x54d; bits[115]:0x5_4c7f_fdff_ffff_dff7_ffff_dfff_ffff; bits[4]:0xa; bits[4]:0xa"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x10; bits[115]:0xb257_fdd5_e277_f67b_c9cf_df6f_de2e; bits[4]:0x0; bits[4]:0x2"
//     args: "bits[49]:0x6bd8_c717_55ce; bits[11]:0x18e; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x7; bits[4]:0xb"
//     args: "bits[49]:0x4; bits[11]:0x11; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x1; bits[4]:0x7"
//     args: "bits[49]:0x0; bits[11]:0x80; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0x5"
//     args: "bits[49]:0x8000; bits[11]:0x555; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0xa; bits[4]:0x9"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x4cb; bits[115]:0x2_aaaf_8a2a_eaab_f3ff_dbf7_ffd9_f9f7; bits[4]:0x7; bits[4]:0x7"
//     args: "bits[49]:0x1_bedb_111f_23b1; bits[11]:0x1; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0x7; bits[4]:0x4"
//     args: "bits[49]:0x0; bits[11]:0x0; bits[115]:0x0; bits[4]:0x4; bits[4]:0x8"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x55d; bits[115]:0x8_0000_0000_0000; bits[4]:0xd; bits[4]:0x5"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0xaa; bits[115]:0x2_a8aa_aa2a_aaab_ab92_a2e8_a2aa_6a8a; bits[4]:0x7; bits[4]:0xf"
//     args: "bits[49]:0x1_40e9_926f_79f7; bits[11]:0x5f6; bits[115]:0x5_b447_74f5_34f4_5555_557d_4d55_1d31; bits[4]:0x7; bits[4]:0x5"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x769; bits[115]:0x100_0000_0000_0000_0000; bits[4]:0x0; bits[4]:0x8"
//     args: "bits[49]:0x8000_0000_0000; bits[11]:0x0; bits[115]:0x2f80_3041_100a_9668_802c_3218_2198; bits[4]:0x0; bits[4]:0x0"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x555; bits[115]:0x6_8f54_bc41_07b9_dad4_42b0_ce99_6641; bits[4]:0xa; bits[4]:0x0"
//     args: "bits[49]:0x8000; bits[11]:0x7ff; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xa; bits[4]:0x0"
//     args: "bits[49]:0x1_03bf_3ac5_13cb; bits[11]:0x54b; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xb; bits[4]:0x3"
//     args: "bits[49]:0x1; bits[11]:0x1; bits[115]:0x15f_ff7b_ff3c_5fff_ff5e_f9ff_f7af; bits[4]:0xf; bits[4]:0x9"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x555; bits[115]:0x5_5555_5555_5555_5555_5555_5555_5555; bits[4]:0xa; bits[4]:0xf"
//     args: "bits[49]:0x1_0c54_7e05_02f7; bits[11]:0x287; bits[115]:0x4_0000; bits[4]:0xe; bits[4]:0x0"
//     args: "bits[49]:0x2; bits[11]:0x555; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x5; bits[4]:0x7"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x7eb; bits[115]:0x3_f7bf_ffff_fdf5_ffff_ffff_fffe_bfdd; bits[4]:0xa; bits[4]:0x2"
//     args: "bits[49]:0x4_0000_0000; bits[11]:0x4; bits[115]:0xfcd5_962d_3017_413b_7c1a_5e12_3d83; bits[4]:0x1; bits[4]:0x1"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x3_ffff_ffff_fffc_8000_0000_0000_0200; bits[4]:0x8; bits[4]:0x8"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x6_fc1d_ed37_37ed_a9e2_623b_bfa0_2ee0; bits[4]:0x1; bits[4]:0xa"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x49b; bits[115]:0x7_2fd8_2262_8084_c180_03e9_c751_04a6; bits[4]:0x2; bits[4]:0xb"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x57f; bits[115]:0xa30f_ff90_ffa7_06e9_5555_7167_5523; bits[4]:0x2; bits[4]:0x3"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x555; bits[115]:0x3_9aa5_f05f_0d74_080c_ff43_b217_85b6; bits[4]:0x0; bits[4]:0xe"
//     args: "bits[49]:0x0; bits[11]:0x88; bits[115]:0x1_4749_9468_e889_d612_2624_c8f9_4f77; bits[4]:0x7; bits[4]:0x4"
//     args: "bits[49]:0x4000; bits[11]:0x100; bits[115]:0x5_81fb_01f5_7515_4cfb_09c2_c16b_0bab; bits[4]:0x0; bits[4]:0x8"
//     args: "bits[49]:0x400; bits[11]:0x480; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x5; bits[4]:0xf"
//     args: "bits[49]:0x0; bits[11]:0x555; bits[115]:0x8; bits[4]:0x8; bits[4]:0xa"
//     args: "bits[49]:0x0; bits[11]:0x2aa; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0xf"
//     args: "bits[49]:0x80; bits[11]:0x0; bits[115]:0x6_8e85_34d0_abdb_a355_5fdc_de98_2d3c; bits[4]:0xc; bits[4]:0x4"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x2ef; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x8; bits[4]:0x8"
//     args: "bits[49]:0x0; bits[11]:0x3a0; bits[115]:0x7_a8aa_aaaa_afaa_aaa2_aaa8_ab88_aaaa; bits[4]:0xd; bits[4]:0x8"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x555; bits[115]:0x1_d621_5268_0c21_0133_4583_0002_4403; bits[4]:0xf; bits[4]:0x7"
//     args: "bits[49]:0x800_0000_0000; bits[11]:0x0; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0xb"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xd; bits[4]:0xf"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x4; bits[115]:0x2_1ee0_faaf_bee0_c8a4_40c5_4001_0030; bits[4]:0x4; bits[4]:0x7"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x417; bits[115]:0x3_d156_f5f6_1156_6000_8014_4d06_8100; bits[4]:0x7; bits[4]:0x7"
//     args: "bits[49]:0x1_d9b9_2f53_f2c4; bits[11]:0x2c4; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0x6; bits[4]:0x0"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x7_5ada_2821_3a0f_765e_0c5c_d7d9_55bd; bits[4]:0xd; bits[4]:0xa"
//     args: "bits[49]:0x0; bits[11]:0x11; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0x0; bits[4]:0xa"
//     args: "bits[49]:0x8734_866c_4b14; bits[11]:0x8; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0xa; bits[4]:0xa"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x5d7; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0x7; bits[4]:0x2"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x575; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xa; bits[4]:0x5"
//     args: "bits[49]:0x0; bits[11]:0x400; bits[115]:0x4_00df_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x0; bits[4]:0x1"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x3ff; bits[115]:0x7_9a3f_57ef_cfdc_23a2_1488_b820_b930; bits[4]:0x0; bits[4]:0x4"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x292; bits[115]:0x2_9a55_5455_5555_dd55_5555_5555_1555; bits[4]:0xa; bits[4]:0x5"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x1; bits[115]:0x5_7455_d55d_4717_f664_672f_731e_0802; bits[4]:0x0; bits[4]:0x0"
//     args: "bits[49]:0x0; bits[11]:0x222; bits[115]:0x2_0000_0001_0005_4745_41d5_7555_7d55; bits[4]:0x5; bits[4]:0x5"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x200; bits[115]:0x2_c148_0c90_8011_0200_8880_8055_1048; bits[4]:0x5; bits[4]:0x2"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x10; bits[115]:0x3_995f_fff3_dff8_4502_e582_1431_9010; bits[4]:0x0; bits[4]:0x5"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x4c4; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x6; bits[4]:0x5"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x3ff; bits[115]:0x1_dfde_f9f1_faea_7080_4000_2068_8005; bits[4]:0x4; bits[4]:0x0"
//     args: "bits[49]:0x4000_0000_0000; bits[11]:0x4b4; bits[115]:0x1_0c2a_f0e3_fe93_2e92_962e_afb9_22b3; bits[4]:0x7; bits[4]:0xa"
//     args: "bits[49]:0x0; bits[11]:0x182; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x4; bits[4]:0x4"
//     args: "bits[49]:0x8000; bits[11]:0x7ff; bits[115]:0x6_e78a_ceea_aa86_2892_52b2_ae0b_6aa9; bits[4]:0x4; bits[4]:0x4"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x0; bits[4]:0x5; bits[4]:0x0"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x0; bits[115]:0x4_ee86_8cde_7468_6e80_627a_c120_268e; bits[4]:0x4; bits[4]:0x2"
//     args: "bits[49]:0x4000; bits[11]:0x0; bits[115]:0x8380_0a21_1051_1545_5741_145d_5754; bits[4]:0x0; bits[4]:0x0"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x4_db3e_ee7c_7c60_64d9_bbaa_abb2_32ab; bits[4]:0x8; bits[4]:0xa"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x7ff; bits[115]:0x40_0000_0000_0000_0000_0000; bits[4]:0xf; bits[4]:0x7"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x8; bits[115]:0x4_e48e_bdf3_2ebe_2507_eb28_1b7f_cc71; bits[4]:0xa; bits[4]:0xa"
//     args: "bits[49]:0x0; bits[11]:0x5b3; bits[115]:0x280_0488_009a_060a_5014_0462_941c; bits[4]:0x4; bits[4]:0xf"
//     args: "bits[49]:0xf83b_f6e9_7e9f; bits[11]:0x111; bits[115]:0x3_f2eb_f2a3_da39_4024_0a11_0040_1500; bits[4]:0x0; bits[4]:0x4"
//     args: "bits[49]:0x0; bits[11]:0x18; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0xa; bits[4]:0xa"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x6e3; bits[115]:0x40_0000_0000_0000_0000_0000_0000; bits[4]:0xe; bits[4]:0x1"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x5; bits[4]:0x8"
//     args: "bits[49]:0xaa12_09f5_979a; bits[11]:0x68c; bits[115]:0x6_8cff_ffbf_bfff_ffbf_ffef_efff_ffff; bits[4]:0xc; bits[4]:0x7"
//     args: "bits[49]:0x1_0000_0000_0000; bits[11]:0x0; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x0; bits[4]:0x7"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x3a9; bits[115]:0x0; bits[4]:0x0; bits[4]:0x7"
//     args: "bits[49]:0x1_e17a_2e03_9840; bits[11]:0x555; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x9; bits[4]:0x9"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x2aa; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x3; bits[4]:0xf"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x2aa; bits[115]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[4]:0x5; bits[4]:0x0"
//     args: "bits[49]:0x0; bits[11]:0x555; bits[115]:0xc1ca_8a05_a000_e190_0401_0928_4408; bits[4]:0x8; bits[4]:0xa"
//     args: "bits[49]:0x0; bits[11]:0x246; bits[115]:0x6_46aa_a8ee_aaab_8aaa_2ea2_6aa8_eeaa; bits[4]:0x2; bits[4]:0x9"
//     args: "bits[49]:0x200_0000; bits[11]:0x7ff; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xe; bits[4]:0x0"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x28f; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0x7"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x555; bits[115]:0x7_ea21_4855_e71b_48e1_2844_23b6_6ec0; bits[4]:0x9; bits[4]:0x3"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x2aa; bits[115]:0x5_6502_c3a0_3bf7_79d5_f09d_e069_0c7d; bits[4]:0xc; bits[4]:0xd"
//     args: "bits[49]:0x7ede_fb20_4253; bits[11]:0x7ff; bits[115]:0x7_c8c2_a69a_458a_2ad6_43b7_7efe_50cf; bits[4]:0x1; bits[4]:0x7"
//     args: "bits[49]:0x0; bits[11]:0x0; bits[115]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x0; bits[4]:0x5"
//     args: "bits[49]:0x8; bits[11]:0x251; bits[115]:0x4028_a885_baba_28ba_2ad8_aaaf_8eab; bits[4]:0x5; bits[4]:0x0"
//     args: "bits[49]:0x0; bits[11]:0x0; bits[115]:0x1_0000_3013_008a_0008_0008_0400_4000; bits[4]:0x3; bits[4]:0x2"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x7ff; bits[115]:0x7_7d91_c2ce_03fe_e467_ef85_9d92_2760; bits[4]:0x5; bits[4]:0x7"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x0; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0x0; bits[4]:0xf"
//     args: "bits[49]:0x54d9_d9c3_77d0; bits[11]:0x7e3; bits[115]:0x1_5271_6dd6_4726_05a2_4d10_b481_1124; bits[4]:0x0; bits[4]:0x5"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x5ff; bits[115]:0x5_ffc0_4012_1204_90a5_0a22_8802_0060; bits[4]:0x0; bits[4]:0x5"
//     args: "bits[49]:0x10_0000; bits[11]:0x200; bits[115]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[4]:0xf; bits[4]:0xd"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[11]:0x7bf; bits[115]:0x5_b045_3f60_7297_0181_55f7_1096_9516; bits[4]:0x7; bits[4]:0xf"
//     args: "bits[49]:0x1_5555_5555_5555; bits[11]:0x7ff; bits[115]:0x7_92fd_dd1f_fee7_da79_4eb9_8f57_ec15; bits[4]:0x5; bits[4]:0x0"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x1f7; bits[115]:0x0; bits[4]:0x7; bits[4]:0xf"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[11]:0x0; bits[115]:0x53c_02c1_587d_0485_d1d4_270d_5154; bits[4]:0xf; bits[4]:0xf"
//     args: "bits[49]:0xffff_ffff_ffff; bits[11]:0x3ff; bits[115]:0x3_ffaa_aaaa_aaaa_aaaa_aaa8_aaaa_aaaa; bits[4]:0xf; bits[4]:0x1"
//   }
// }
// 
// END_CONFIG
fn main(x0: s49, x1: u11, x2: uN[115], x3: u4, x4: s4) -> (uN[854], u4, bool, uN[1904], uN[1904]) {
    {
        let x5: s4 = x1 as s4 * x4;
        let x6: uN[645] = match x1 {
            u11:0x3ff | u11:0x0 => uN[645]:0,
            u11:0x200 => uN[645]:0b1000_0110_0100_0011_1001_0101_1010_1111_0111_0010_1000_0101_1110_0000_1110_0010_0001_1000_1100_1100_1100_0111_1101_1010_0101_1100_1111_0110_1111_0000_1010_1110_0000_1111_1001_0000_0001_0101_1011_0010_0101_0001_0101_0111_0100_1011_1111_1101_0011_1101_0011_0010_1011_0001_0010_0111_1111_0111_1111_1000_0011_1110_1110_0101_1111_0011_0110_0101_0000_0001_0001_0110_0111_1001_1101_1010_0111_0011_1000_1010_1110_1100_1011_1000_0110_1110_1001_0111_0000_1011_0111_0000_1111_0100_1110_1010_1010_0100_1110_0010_1001_1100_1111_1001_1111_0011_1011_0100_0010_0010_1110_0000_0111_0110_1001_1110_0101_1011_1110_1000_0000_0000_0000_0000_0100_1100_1011_0101_0010_0100_0110_1101_0111_0101_0001_1010_1000_1111_0111_0000_1100_1111_1111_0101_0000_1000_0000_1011_0000_0100_0111_1100_1011_1011_1110_1100_0111_0010_0110_1100_0110,
            _ => uN[645]:0b1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010,
        };
        let x7: (u4, u11, uN[645], uN[115]) = (x3, x1, x6, x2);
        let x8: uN[2048] = decode<uN[2048]>(x2);
        let x9: u43 = (x0 as u49)[6+:xN[bool:0x0][43]];
        let x10: s49 = x0 << if x6 >= uN[645]:0xc { uN[645]:0xc } else { x6 };
        let x11: uN[1904] = match x4 {
            s4:0x7 => uN[1904]:0x4000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000,
            _ => uN[1904]:0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa,
        };
        let x12: s55 = s55:0x2f_2101_517f_e084;
        let x13: xN[bool:0x0][4] = (x5 as u4)[:];
        let x14: u4 = (x4 as u4)[x9+:u4];
        let x15: u4 = x13[0+:u4];
        let x16: bool = x5 as u4 == x3;
        let x17: uN[1904] = x11[:];
        let x18: u58 = match x11 {
            uN[1904]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff => u58:0x3ff_ffff_ffff_ffff,
            uN[1904]:0xc828_ebef_b7cc_e7b6_984e_03c8_7007_3cef_74b1_0ae8_d952_4bda_cdfa_44a0_55bb_6cc6_3868_7993_6779_a9b0_0038_a017_f6b1_1ed3_aaa1_f867_21ac_f522_111f_b610_fe20_276d_ee00_ad25_3e6c_4459_3efa_8299_fdae_55c2_9cf9_e484_2234_4f30_f6a8_345f_92c0_5642_4296_67c2_76d7_87af_b8cb_3933_b224_cd3c_ba51_dbc9_8355_906e_8916_9062_a560_87ef_dc54_e2de_84c3_111d_2eb3_29e8_8b96_0433_2bf2_d56f_1df8_5f42_b93c_d4a5_f12d_5ce5_83f3_7d18_60eb_5905_1030_4d4c_4c82_9616_bda1_91bc_4983_1004_4c76_b523_8fda_efb7_0ccc_fb9d_a04f_2fb2_f733_cc84_9afd_7951_9f02_d06d_fc5a_8751_669b_ee1a_6ed1_654d_ef1f_dccd_ff55_ab96_ca76_b4a8_5cdb => u58:0x1000,
            uN[1904]:0x5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555..uN[1904]:0x5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555 => u58:0x2aa_aaaa_aaaa_aaaa,
            _ => u58:0x1ff_ffff_ffff_ffff,
        };
        let x19: s49 = -x0;
        let x20: bool = x16 & x16;
        let x21: uN[1904] = x17 >> if x13 >= xN[bool:0x0][4]:0x7 { xN[bool:0x0][4]:0x7 } else { x13 };
        let x22: u4 = x15 % u4:0x0;
        let x24: uN[645] = x6 ^ x16 as uN[645];
        let x25: bool = x20 - x5 as bool;
        let x26: uN[2003] = decode<uN[2003]>(x9);
        let x27: uN[1904] = x21[x14+:uN[1904]];
        let x28: bool = x5 >= x9 as s4;
        let x29: uN[1904] = gate!(x11 == x20 as uN[1904], x17);
        let x30: bool = !x20;
        let x31: bool = x16 & x8 as bool;
        let x32: s49 = -x19;
        let x33: uN[1905] = one_hot(x21, bool:0x0);
        let x34: bool = gate!(x33 as u11 > x1, x25);
        let x35: uN[1029] = x29[:1029];
        let x36: uN[1029] = x35 | x4 as uN[1029];
        let x37: uN[854] = x35[x15+:uN[854]];
        (x37, x15, x20, x17, x11)
    }
}
