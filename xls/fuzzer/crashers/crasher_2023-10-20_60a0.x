// Copyright 2023 The XLS Authors
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
// exception:    "SampleError: Result miscompare for sample 1:\nargs: bits[21]:0x8000; bits[1]:0x0; bits[16]:0x0; bits[7]:0x55; bits[52]:0x7_d2f7_dd4e_0e7f\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[16]:0x55ab, bits[44]:0x40_002b_aa55, bits[16]:0x55ab, bits[16]:0x55ab, bits[6]:0x0)\nevaluated unopt IR (JIT) =\n   (bits[16]:0x54ab, bits[44]:0x40_002b_ab55, bits[16]:0x54ab, bits[16]:0x54ab, bits[6]:0x0)"
// issue: "https://github.com/google/xls/issues/1159"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full throughput.*"
//   }
// }
// inputs {
//   function_args {
//     args: "bits[21]:0xe_c07a; bits[1]:0x0; bits[16]:0x400; bits[7]:0x64; bits[52]:0x1_0000_0000_0000"
//     args: "bits[21]:0x8000; bits[1]:0x0; bits[16]:0x0; bits[7]:0x55; bits[52]:0x7_d2f7_dd4e_0e7f"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x37be; bits[7]:0x2a; bits[52]:0x3_7ba9_e7d2_aade"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0xd907; bits[7]:0x54; bits[52]:0x0"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xafdb; bits[7]:0x0; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x1200; bits[7]:0x20; bits[52]:0xc_901a_0040_6123"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0x1a67; bits[7]:0x3f; bits[52]:0x4_827e_0802_0032"
//     args: "bits[21]:0x1e_c0ec; bits[1]:0x0; bits[16]:0x20; bits[7]:0x60; bits[52]:0x4_a30f_8bab_aaea"
//     args: "bits[21]:0x2; bits[1]:0x0; bits[16]:0x4918; bits[7]:0x0; bits[52]:0xd_1345_f05d_d5c2"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xc4de; bits[7]:0x2a; bits[52]:0xc_4de7_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xffff; bits[7]:0x3f; bits[52]:0x5_5555_5555_5555"
//     args: "bits[21]:0x400; bits[1]:0x1; bits[16]:0xdcff; bits[7]:0x3f; bits[52]:0x8014_8824_0b01"
//     args: "bits[21]:0x9_b113; bits[1]:0x0; bits[16]:0xbfdb; bits[7]:0x2a; bits[52]:0x9d9b_4050_889d"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0xaaaa; bits[7]:0x34; bits[52]:0x0"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x80; bits[7]:0x0; bits[52]:0x805_5d55_f555"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0x0; bits[7]:0x77; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x8; bits[1]:0x0; bits[16]:0x6b7f; bits[7]:0x27; bits[52]:0x3_ae67_1575_5735"
//     args: "bits[21]:0xe_6d15; bits[1]:0x0; bits[16]:0xb4fc; bits[7]:0x31; bits[52]:0x6_68aa_e2aa_baa8"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0x31c0; bits[7]:0x6f; bits[52]:0xd_a2a2_8826_8dc8"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x5555; bits[7]:0x2a; bits[52]:0xe_77c4_1404_016a"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0x1514; bits[7]:0x37; bits[52]:0x5_106a_8b78_e5b1"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x40; bits[7]:0x40; bits[52]:0xc1f_fdfe_fbf7"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x200; bits[7]:0x50; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xc002; bits[7]:0x2; bits[52]:0x8_8020_20e0_4208"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x1; bits[16]:0xaaaa; bits[7]:0xb; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0x9_7e9f; bits[1]:0x1; bits[16]:0xfeff; bits[7]:0x7f; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x16_67a4; bits[1]:0x1; bits[16]:0x0; bits[7]:0x77; bits[52]:0x1_7f38_55f4_c5df"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xffff; bits[7]:0x1; bits[52]:0xb_0002_0e00_0110"
//     args: "bits[21]:0x8; bits[1]:0x0; bits[16]:0xaaaa; bits[7]:0x7f; bits[52]:0xf_df55_3751_5955"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x5555; bits[7]:0x20; bits[52]:0x1_0413_0024_2002"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0xaa2; bits[7]:0x55; bits[52]:0x400_0000_0000"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xd555; bits[7]:0x7a; bits[52]:0xa_57fe_adee_dfff"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xc283; bits[7]:0x66; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x5011; bits[7]:0x49; bits[52]:0x0"
//     args: "bits[21]:0x20; bits[1]:0x0; bits[16]:0x5; bits[7]:0x40; bits[52]:0x6_5016_0718_b00e"
//     args: "bits[21]:0x2_1450; bits[1]:0x0; bits[16]:0xffff; bits[7]:0x3e; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x15_5555; bits[1]:0x0; bits[16]:0x0; bits[7]:0x3f; bits[52]:0x31_01aa_ed5a"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x5555; bits[7]:0x4; bits[52]:0x8_0000_0000_0000"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x4000; bits[7]:0x60; bits[52]:0xc_4a8c_35b0_8f7e"
//     args: "bits[21]:0x15_5555; bits[1]:0x0; bits[16]:0x10; bits[7]:0x55; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0x8000; bits[1]:0x0; bits[16]:0x6a2a; bits[7]:0x2b; bits[52]:0xc_2092_2b97_f7e7"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x1; bits[16]:0x8001; bits[7]:0x8; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0xd555; bits[7]:0x5d; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x4_0000; bits[1]:0x0; bits[16]:0x4621; bits[7]:0x7b; bits[52]:0x2_40c8_d102_6580"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0x9fff; bits[7]:0x4; bits[52]:0x8_0400_1020_0802"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0xd057; bits[7]:0x57; bits[52]:0x400_0000"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0x7fff; bits[7]:0x7a; bits[52]:0x5_5555_5555_5555"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0xaaaa; bits[7]:0x33; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x2b3; bits[7]:0x61; bits[52]:0x9_d461_4dad_618d"
//     args: "bits[21]:0x1f_949a; bits[1]:0x0; bits[16]:0x5555; bits[7]:0x1a; bits[52]:0xf_c7dc_f020_8405"
//     args: "bits[21]:0x2000; bits[1]:0x1; bits[16]:0x7fff; bits[7]:0x59; bits[52]:0x9_fce2_aee2_aeae"
//     args: "bits[21]:0x800; bits[1]:0x1; bits[16]:0xa301; bits[7]:0x0; bits[52]:0x5_8e54_8ca2_5fc5"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x0; bits[16]:0x2; bits[7]:0x7f; bits[52]:0x9_c193_de9c_64a5"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0x7fff; bits[7]:0x1e; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x13_6745; bits[1]:0x1; bits[16]:0x400; bits[7]:0x2a; bits[52]:0x5_6e18_7719_f7d7"
//     args: "bits[21]:0xc_2097; bits[1]:0x1; bits[16]:0x51d1; bits[7]:0x5e; bits[52]:0xb_c020_0820_8101"
//     args: "bits[21]:0x1000; bits[1]:0x0; bits[16]:0x40; bits[7]:0x7f; bits[52]:0xf_95c5_5513_4555"
//     args: "bits[21]:0xf_ffff; bits[1]:0x0; bits[16]:0x0; bits[7]:0x2a; bits[52]:0x20_0000"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x5555; bits[7]:0x3f; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0x4555; bits[7]:0x0; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xaaaa; bits[7]:0x5f; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x6bda; bits[7]:0x0; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xc7fb; bits[7]:0x2; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x7fff; bits[7]:0x49; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0x0; bits[1]:0x1; bits[16]:0x3; bits[7]:0xb; bits[52]:0xd_5555_5555_5555"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x10; bits[7]:0x2f; bits[52]:0x8_9bfa_e72e_b0a1"
//     args: "bits[21]:0x14_4224; bits[1]:0x1; bits[16]:0x6620; bits[7]:0x65; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x0; bits[1]:0x1; bits[16]:0x5555; bits[7]:0x3f; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0xf_ffff; bits[1]:0x0; bits[16]:0x1414; bits[7]:0x22; bits[52]:0x1_434f_fbe7_ffff"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x0; bits[7]:0x3f; bits[52]:0x1_0000_0000"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x4dab; bits[7]:0x2a; bits[52]:0x6_5a39_83e2_caf5"
//     args: "bits[21]:0x400; bits[1]:0x0; bits[16]:0xe478; bits[7]:0x0; bits[52]:0xd_67a7_1095_1371"
//     args: "bits[21]:0x0; bits[1]:0x1; bits[16]:0x7fff; bits[7]:0x62; bits[52]:0x1_0ea2_c58f_d6aa"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xd55d; bits[7]:0x1f; bits[52]:0xd_7e7e_dea4_a9ab"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xfeff; bits[7]:0x6c; bits[52]:0xd_5139_2455_3dcb"
//     args: "bits[21]:0x1000; bits[1]:0x0; bits[16]:0x4d06; bits[7]:0x5d; bits[52]:0x5_5729_2280_c8a8"
//     args: "bits[21]:0x0; bits[1]:0x1; bits[16]:0x40; bits[7]:0x2a; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x7f08; bits[7]:0x0; bits[52]:0x2_d05b_2faa_a8e8"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0x5555; bits[7]:0x41; bits[52]:0x5_5555_5555_5555"
//     args: "bits[21]:0x2_0000; bits[1]:0x0; bits[16]:0xaaaa; bits[7]:0x78; bits[52]:0xd_28ae_0482_0021"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0xff7a; bits[7]:0x49; bits[52]:0xf_f7af_ffff_97ff"
//     args: "bits[21]:0x15_5555; bits[1]:0x0; bits[16]:0x8748; bits[7]:0x54; bits[52]:0x5_5555_5555_5555"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x7fff; bits[7]:0x7f; bits[52]:0x7_eff7_febf_ffff"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x5555; bits[7]:0x40; bits[52]:0xd_ffff_d83f_f1d4"
//     args: "bits[21]:0x13_3b85; bits[1]:0x1; bits[16]:0x7fff; bits[7]:0x2a; bits[52]:0x5_4002_0010_00a0"
//     args: "bits[21]:0x15_5555; bits[1]:0x0; bits[16]:0x5555; bits[7]:0x5d; bits[52]:0x40"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x9aaa; bits[7]:0x8; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0x10_f984; bits[1]:0x1; bits[16]:0xe804; bits[7]:0x4; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xec67; bits[7]:0x67; bits[52]:0x5_5555_5555_5555"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x0; bits[16]:0x2; bits[7]:0x2a; bits[52]:0x8_0000_0000"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x5555; bits[7]:0x55; bits[52]:0xd_7f19_0d45_24f5"
//     args: "bits[21]:0x1c_b539; bits[1]:0x0; bits[16]:0x188; bits[7]:0x55; bits[52]:0x800_0000_0000"
//     args: "bits[21]:0x18_74e2; bits[1]:0x0; bits[16]:0x0; bits[7]:0x20; bits[52]:0x5_574c_81d7_d5f3"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xbaa8; bits[7]:0x34; bits[52]:0x3_aa0a_e7c1_e12f"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x108; bits[7]:0x38; bits[52]:0x80"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x77ef; bits[7]:0x78; bits[52]:0x7_fdfd_bfaf_6349"
//     args: "bits[21]:0x1d_01e9; bits[1]:0x1; bits[16]:0x5555; bits[7]:0x7f; bits[52]:0xe_9b90_91fd_dfd9"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0xa50; bits[7]:0x41; bits[52]:0x8"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x0; bits[7]:0x29; bits[52]:0x0"
//     args: "bits[21]:0x14_b739; bits[1]:0x1; bits[16]:0x8000; bits[7]:0x44; bits[52]:0x8_4403_800b_0120"
//     args: "bits[21]:0x19_5f9c; bits[1]:0x0; bits[16]:0x5d55; bits[7]:0x0; bits[52]:0xc_efcf_0002_9000"
//     args: "bits[21]:0x80; bits[1]:0x1; bits[16]:0xffff; bits[7]:0x17; bits[52]:0xf_9cb9_a58e_74ad"
//     args: "bits[21]:0xd_6599; bits[1]:0x1; bits[16]:0x6599; bits[7]:0x35; bits[52]:0x7_924d_818a_f25a"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xaaaa; bits[7]:0x7a; bits[52]:0x7_7755_5116_5556"
//     args: "bits[21]:0x2; bits[1]:0x0; bits[16]:0x4575; bits[7]:0x9; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0x7d7f; bits[7]:0x22; bits[52]:0x5_062f_fb53_ef77"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x0; bits[16]:0x7fff; bits[7]:0x0; bits[52]:0x8_dd35_d1e3_b536"
//     args: "bits[21]:0x15_5555; bits[1]:0x0; bits[16]:0x5595; bits[7]:0x55; bits[52]:0x9_3f1e_2b75_076a"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xaaaa; bits[7]:0x3f; bits[52]:0x5_5555_5555_5555"
//     args: "bits[21]:0xf_ffff; bits[1]:0x1; bits[16]:0xef7d; bits[7]:0x40; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x1e_64e4; bits[1]:0x0; bits[16]:0xc988; bits[7]:0x7f; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x1; bits[16]:0x8684; bits[7]:0x55; bits[52]:0xc_1eb1_2e0c_2540"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0xede4; bits[7]:0x64; bits[52]:0xa_9e67_67d5_47db"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x1; bits[16]:0xdb9f; bits[7]:0x7f; bits[52]:0x7_39f0_54c4_1098"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x7fff; bits[7]:0x55; bits[52]:0xa_afff_ffff_ffff"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x100; bits[7]:0x0; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x0; bits[16]:0x85fe; bits[7]:0x6a; bits[52]:0x0"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0x35d9; bits[7]:0x7c; bits[52]:0x5_5555_5555_5555"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0x0; bits[7]:0x2a; bits[52]:0x8_0000_0000_0000"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x1; bits[16]:0x1; bits[7]:0x5d; bits[52]:0x8000_0000"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0x554f; bits[7]:0x7f; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[21]:0xa_aaaa; bits[1]:0x0; bits[16]:0xc340; bits[7]:0x3f; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0xe2f5; bits[7]:0x3a; bits[52]:0xa_0c81_0308_4c80"
//     args: "bits[21]:0x15_5555; bits[1]:0x1; bits[16]:0x9b59; bits[7]:0x3f; bits[52]:0xc_132d_eb14_fb90"
//     args: "bits[21]:0x0; bits[1]:0x0; bits[16]:0x4177; bits[7]:0x0; bits[52]:0xb_f140_261a_fe23"
//     args: "bits[21]:0x800; bits[1]:0x1; bits[16]:0x1c7; bits[7]:0x57; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[1]:0x0; bits[16]:0xffff; bits[7]:0x40; bits[52]:0x9_c062_8040_3094"
//     args: "bits[21]:0x4000; bits[1]:0x0; bits[16]:0xaaaa; bits[7]:0x24; bits[52]:0x2_27bf_16d5_5c5d"
//   }
// }
// 
// END_CONFIG
fn x8(x9: s1, x10: u7) -> (u2, s1, s1) {
    {
        let x11: bool = x9 != x10 as s1;
        let x12: s1 = x9 >> if x10 >= u7:0x0 { u7:0x0 } else { x10 };
        let x13: u7 = gate!(x11 as s1 != x9, x10);
        let x14: u2 = x10[x11+:u2];
        let x15: s1 = x12 + x11 as s1;
        (x14, x15, x9)
    }
}
fn main(x0: u21, x1: s1, x2: u16, x3: u7, x4: s52) -> (u16, u44, u16, u16, u6) {
    {
        let x5: bool = x1 == x1;
        let x6: bool = xor_reduce(x0);
        let x7: s52 = !x4;
        let x16: (u2, s1, s1) = x8(x1, x3);
        let x17: s1 = x1 & x1;
        let x18: u7 = !x3;
        let x19: u16 = x5 ++ x18 ++ x18 ++ x6;
        let x20: s52 = -x7;
        let x21: s52 = x7 + x0 as s52;
        let x22: u7 = for (i, x): (u4, u7) in u4:0..u4:0x2 {
            x
        }(x18);
        let x23: bool = x6 != x21 as bool;
        let x24: u7 = signex(x2, x18);
        let x25: u7 = -x3;
        let x26: u15 = x24 ++ x25 ++ x5;
        let x27: u2 = x24[:2];
        let x28: u49 = u49:0x1_ffff_ffff_ffff;
        let x29: u7 = rev(x25);
        let x30: u44 = x0 ++ x25 ++ x19;
        let x31: u6 = x26[9+:u6];
        let x32: bool = x29 <= x18 as u7;
        let x33: bool = x32 == x6 as bool;
        let x34: u8 = one_hot(x24, bool:0x1);
        let x35: u2 = x16.0;
        let x36: s1 = x6 as s1 | x17;
        let x37: u16 = x24 as u16 - x19;
        let x38: s1 = x16.2;
        let x39: s52 = gate!(x32 as u2 != x35, x7);
        (x37, x30, x37, x37, x31)
    }
}
