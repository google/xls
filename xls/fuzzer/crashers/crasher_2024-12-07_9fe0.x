// Copyright 2024 The XLS Authors
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
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[4]:0x8; bits[46]:0x1224_4d98_43c9; bits[3]:0x4; bits[34]:0x8; bits[9]:0x9b; bits[56]:0x60_7640_1285_1346\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (JIT), evaluated unopt IR (interpreter) =\n   (bits[13]:0x0, bits[44]:0x7ff_ffff_ffff, bits[13]:0x0)\ninterpreted DSLX =\n   (bits[13]:0x0, bits[44]:0x0, bits[13]:0x0)"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: true
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
//   with_valid_holdoff: false
//   codegen_ng: false
// }
// inputs {
//   function_args {
//     args: "bits[4]:0x8; bits[46]:0x1224_4d98_43c9; bits[3]:0x4; bits[34]:0x8; bits[9]:0x9b; bits[56]:0x60_7640_1285_1346"
//     args: "bits[4]:0x0; bits[46]:0x1fff_ffff_ffff; bits[3]:0x5; bits[34]:0x2_aaaa_aaaa; bits[9]:0xc; bits[56]:0xaa_a4ae_ab80_0881"
//     args: "bits[4]:0x7; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x0; bits[34]:0x2_eaca_0e1a; bits[9]:0xff; bits[56]:0x6f_9402_6881_c3d5"
//     args: "bits[4]:0x0; bits[46]:0x1fff_ffff_ffff; bits[3]:0x1; bits[34]:0x3_fff7_ffff; bits[9]:0xff; bits[56]:0x0"
//     args: "bits[4]:0x8; bits[46]:0x1555_5555_5555; bits[3]:0x7; bits[34]:0x3_1af7_d53f; bits[9]:0x19f; bits[56]:0x80"
//     args: "bits[4]:0x7; bits[46]:0x2c0a_0005_0101; bits[3]:0x5; bits[34]:0x2_1315_90a5; bits[9]:0x9; bits[56]:0x0"
//     args: "bits[4]:0x5; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x0; bits[34]:0x1_ffff_ffff; bits[9]:0x2; bits[56]:0x4_0000"
//     args: "bits[4]:0xa; bits[46]:0x2001_440f_9547; bits[3]:0x7; bits[34]:0x1_0000; bits[9]:0x147; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0xf; bits[46]:0x3fff_ffff_ffff; bits[3]:0x7; bits[34]:0x1_71d5_fdc2; bits[9]:0x1ff; bits[56]:0x94_eaf0_84ec_3f22"
//     args: "bits[4]:0x7; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x7; bits[34]:0x0; bits[9]:0x104; bits[56]:0xa2_3ef7_dfff_dfff"
//     args: "bits[4]:0x7; bits[46]:0x3bc6_6daa_13d4; bits[3]:0x4; bits[34]:0x1_d456_51d5; bits[9]:0xff; bits[56]:0xff_76e7_a842_17e1"
//     args: "bits[4]:0xa; bits[46]:0x1fff_ffff_ffff; bits[3]:0x5; bits[34]:0x2_e368_f6bd; bits[9]:0x55; bits[56]:0x6c_d4b8_ab4a_a595"
//     args: "bits[4]:0xf; bits[46]:0x1315_0e55_e58b; bits[3]:0x7; bits[34]:0x1_c000_4000; bits[9]:0x172; bits[56]:0xf0_8000_0020_0000"
//     args: "bits[4]:0xe; bits[46]:0x2385_4dd8_1182; bits[3]:0x3; bits[34]:0x1_2b85_ff32; bits[9]:0x6a; bits[56]:0x2_0000_0000_0000"
//     args: "bits[4]:0x8; bits[46]:0x23ff_dfff_bfff; bits[3]:0x7; bits[34]:0x2_0200_1000; bits[9]:0x0; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0xa; bits[46]:0x2945_745c_45c5; bits[3]:0x5; bits[34]:0x1_ffff_ffff; bits[9]:0x1f4; bits[56]:0xe0_5709_5726_736d"
//     args: "bits[4]:0x9; bits[46]:0x371d_5555_5745; bits[3]:0x5; bits[34]:0x1_5555_5555; bits[9]:0xff; bits[56]:0x2_0000"
//     args: "bits[4]:0x7; bits[46]:0x3fff_ffff_ffff; bits[3]:0x3; bits[34]:0x0; bits[9]:0x57; bits[56]:0x67_ff51_c55d_9944"
//     args: "bits[4]:0xf; bits[46]:0x0; bits[3]:0x7; bits[34]:0x0; bits[9]:0x10; bits[56]:0x2d_f51b_477f_5e47"
//     args: "bits[4]:0xf; bits[46]:0x1_0000_0000; bits[3]:0x2; bits[34]:0x4_0000; bits[9]:0x1ff; bits[56]:0x0"
//     args: "bits[4]:0x7; bits[46]:0xd53_1515_7554; bits[3]:0x4; bits[34]:0x3_5f1f_9565; bits[9]:0x1cc; bits[56]:0x4_202c_0400"
//     args: "bits[4]:0x5; bits[46]:0x10_0000; bits[3]:0x5; bits[34]:0x3_9e96_5bc0; bits[9]:0xff; bits[56]:0x1d_7fb1_96e5_c0ea"
//     args: "bits[4]:0x5; bits[46]:0x0; bits[3]:0x0; bits[34]:0x20; bits[9]:0xa1; bits[56]:0x52_9010_0000_0000"
//     args: "bits[4]:0x7; bits[46]:0x2600_0402_00a0; bits[3]:0x4; bits[34]:0x402_00a0; bits[9]:0x0; bits[56]:0xba_a2c3_2aa8_a92e"
//     args: "bits[4]:0x8; bits[46]:0x7d9_fd30_dce7; bits[3]:0x7; bits[34]:0x1_ff30_dce7; bits[9]:0x118; bits[56]:0x8c_2caa_eaaa_aaab"
//     args: "bits[4]:0x7; bits[46]:0x1fff_ffff_ffff; bits[3]:0x7; bits[34]:0xd511_5565; bits[9]:0x6f; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0xa; bits[46]:0x3955_5417_5555; bits[3]:0x5; bits[34]:0x8862_fc5c; bits[9]:0x1ff; bits[56]:0x4f_22b4_51eb_e705"
//     args: "bits[4]:0x5; bits[46]:0x148a_aaa2_8a84; bits[3]:0x5; bits[34]:0x1000_0000; bits[9]:0x84; bits[56]:0x7c_6a94_4988_c8b9"
//     args: "bits[4]:0xe; bits[46]:0x396d_7b95_5c51; bits[3]:0x0; bits[34]:0x3_395c_c9cf; bits[9]:0xaa; bits[56]:0x57_7dff_3fbe_fb6f"
//     args: "bits[4]:0x7; bits[46]:0x30c5_c5e8_c205; bits[3]:0x5; bits[34]:0x1_ddef_bdbc; bits[9]:0x8; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x7; bits[46]:0xfff_dffd_ff7f; bits[3]:0x6; bits[34]:0x1_df4b_fff6; bits[9]:0x19f; bits[56]:0x8f_84c9_4b2e_1438"
//     args: "bits[4]:0x8; bits[46]:0x328a_aaaa_aaa8; bits[3]:0x0; bits[34]:0x0; bits[9]:0x10a; bits[56]:0x89_140a_c004_0022"
//     args: "bits[4]:0x0; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x4; bits[34]:0x1_5555_5555; bits[9]:0xff; bits[56]:0xcd_f3fd_b331_ff7b"
//     args: "bits[4]:0xf; bits[46]:0x690_a0fe_da96; bits[3]:0x6; bits[34]:0x1_ffff_ffff; bits[9]:0x100; bits[56]:0x4_0000"
//     args: "bits[4]:0x9; bits[46]:0x400_0000_0000; bits[3]:0x1; bits[34]:0x9ffe_bfef; bits[9]:0x155; bits[56]:0x11_a121_0c61_2287"
//     args: "bits[4]:0x5; bits[46]:0x1755_5575_45dd; bits[3]:0x6; bits[34]:0x3_8c81_6e10; bits[9]:0x1ff; bits[56]:0xa3_245b_9432_f29d"
//     args: "bits[4]:0x7; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x6; bits[34]:0x1_2233_82ab; bits[9]:0x0; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x0; bits[46]:0x18ab_6ec7_a5c0; bits[3]:0x3; bits[34]:0x3_01a4_47a9; bits[9]:0x1e9; bits[56]:0x0"
//     args: "bits[4]:0xa; bits[46]:0x0; bits[3]:0x2; bits[34]:0x2_aaaa_aaaa; bits[9]:0x1ff; bits[56]:0x1f_f7fe_ffff_ff7b"
//     args: "bits[4]:0xd; bits[46]:0x2649_f195_2003; bits[3]:0x0; bits[34]:0x2_804a_3d31; bits[9]:0xc8; bits[56]:0x8a_064d_217f_aa1f"
//     args: "bits[4]:0x5; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x5; bits[34]:0x12d8_6a84; bits[9]:0x1ff; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0x7; bits[46]:0x1bfd_df23_ff3e; bits[3]:0x1; bits[34]:0x1_c008_0000; bits[9]:0x101; bits[56]:0x70_8a08_80bf_5a7f"
//     args: "bits[4]:0x0; bits[46]:0x1410_1000_4820; bits[3]:0x2; bits[34]:0x3_7651_59fd; bits[9]:0x2; bits[56]:0x17_defb_dd7f_f6db"
//     args: "bits[4]:0x4; bits[46]:0x1555_5555_5555; bits[3]:0x1; bits[34]:0x94a1_1630; bits[9]:0x2; bits[56]:0x57_c518_c115_56bb"
//     args: "bits[4]:0x7; bits[46]:0x1555_5555_5555; bits[3]:0x5; bits[34]:0x2_8000_4002; bits[9]:0x115; bits[56]:0x0"
//     args: "bits[4]:0x1; bits[46]:0x576_cdce_8adf; bits[3]:0x6; bits[34]:0x40; bits[9]:0x19f; bits[56]:0xcf_ffff_ffff_ffff"
//     args: "bits[4]:0x8; bits[46]:0x0; bits[3]:0x2; bits[34]:0x1_6893_793d; bits[9]:0x1ff; bits[56]:0x7f_ffff_ffff_ffff"
//     args: "bits[4]:0x7; bits[46]:0x1fff_ffff_ffff; bits[3]:0x7; bits[34]:0x1_7ee7_cffb; bits[9]:0xaa; bits[56]:0x7f_fffe_fbff_feaa"
//     args: "bits[4]:0x7; bits[46]:0x1f51_575c_575d; bits[3]:0x4; bits[34]:0x2_0020_4280; bits[9]:0x0; bits[56]:0x0"
//     args: "bits[4]:0x7; bits[46]:0x4; bits[3]:0x4; bits[34]:0x1_5555_5555; bits[9]:0xea; bits[56]:0x84_0000_0000_0000"
//     args: "bits[4]:0xa; bits[46]:0x0; bits[3]:0x0; bits[34]:0x400_2200; bits[9]:0xaa; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0x2; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x0; bits[34]:0x2_dc92_65d6; bits[9]:0x58; bits[56]:0x8_0000_0000"
//     args: "bits[4]:0x7; bits[46]:0x3dc2_b618_4531; bits[3]:0x5; bits[34]:0x1_ffff_ffff; bits[9]:0x0; bits[56]:0x5a_b56e_8020_a989"
//     args: "bits[4]:0x4; bits[46]:0x3654_d7e3_6750; bits[3]:0x2; bits[34]:0x1_3be4_3db7; bits[9]:0x100; bits[56]:0xdf_8b6b_6de9_b1ff"
//     args: "bits[4]:0x7; bits[46]:0x3fff_ffff_ffff; bits[3]:0x7; bits[34]:0x3_fff6_fff7; bits[9]:0x74; bits[56]:0x75_7516_a587_4677"
//     args: "bits[4]:0x8; bits[46]:0x3fff_ffff_ffff; bits[3]:0x3; bits[34]:0x3_cfed_9fd7; bits[9]:0x1; bits[56]:0xff_ffff_ffff_ffff"
//     args: "bits[4]:0xc; bits[46]:0x2add_1e7e_b141; bits[3]:0x3; bits[34]:0x3_ffff_ffff; bits[9]:0x0; bits[56]:0x0"
//     args: "bits[4]:0x7; bits[46]:0x1cba_ca0a_a8aa; bits[3]:0x0; bits[34]:0x1_c00c_8004; bits[9]:0xa8; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x8; bits[46]:0x0; bits[3]:0x0; bits[34]:0x800; bits[9]:0xc6; bits[56]:0x80_d2d6_e5b4_4204"
//     args: "bits[4]:0xf; bits[46]:0x1fff_ffff_ffff; bits[3]:0x2; bits[34]:0x1_ffff_ffff; bits[9]:0x40; bits[56]:0x84_a884_0002_80c4"
//     args: "bits[4]:0xf; bits[46]:0xab8_34b7_db98; bits[3]:0x1; bits[34]:0x34b6_c818; bits[9]:0x155; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x2; bits[46]:0x2420_0671_1a00; bits[3]:0x5; bits[34]:0x2_f6f8_aed7; bits[9]:0x107; bits[56]:0x98_804b_c4e8_03ff"
//     args: "bits[4]:0x0; bits[46]:0x3fff_ffff_ffff; bits[3]:0x7; bits[34]:0x0; bits[9]:0x86; bits[56]:0x43_2bbb_d1fc_d7fd"
//     args: "bits[4]:0xa; bits[46]:0x3800_0454_2000; bits[3]:0x2; bits[34]:0x8; bits[9]:0x81; bits[56]:0x49_7124_9dfe_4bb1"
//     args: "bits[4]:0xa; bits[46]:0x1555_5555_5555; bits[3]:0x7; bits[34]:0x3_ffcf_ddff; bits[9]:0x0; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x0; bits[46]:0x103c_04b8_9135; bits[3]:0x3; bits[34]:0x3_ba03_03eb; bits[9]:0xf7; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x1; bits[46]:0x15fa_b2bb_fa8f; bits[3]:0x5; bits[34]:0x739d_f2af; bits[9]:0x100; bits[56]:0x1b_fa73_fcbe_ff1e"
//     args: "bits[4]:0xc; bits[46]:0x31cf_fffe_ffff; bits[3]:0x2; bits[34]:0x2_9bf4_db6f; bits[9]:0x0; bits[56]:0xa6_bc36_db51_373b"
//     args: "bits[4]:0x5; bits[46]:0x15ff_ffff_dfff; bits[3]:0x1; bits[34]:0x3_ecef_dbe8; bits[9]:0x1ef; bits[56]:0x55_9bff_2a38_aeb2"
//     args: "bits[4]:0x5; bits[46]:0xc5a_deff_e06e; bits[3]:0x5; bits[34]:0x80; bits[9]:0x1ff; bits[56]:0x1_2030_0000"
//     args: "bits[4]:0x0; bits[46]:0x386_a078_56ce; bits[3]:0x7; bits[34]:0x0; bits[9]:0x13d; bits[56]:0x0"
//     args: "bits[4]:0x1; bits[46]:0x1290_1f70_3e22; bits[3]:0x2; bits[34]:0x2_aaaa_aaaa; bits[9]:0x21; bits[56]:0x0"
//     args: "bits[4]:0x4; bits[46]:0x0; bits[3]:0x7; bits[34]:0x2_0000; bits[9]:0x6; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x4; bits[46]:0x3fff_ffff_ffff; bits[3]:0x4; bits[34]:0x3_ffff_ffff; bits[9]:0x4; bits[56]:0x20_1d55_5557_7f45"
//     args: "bits[4]:0xa; bits[46]:0x800_0000_0000; bits[3]:0x4; bits[34]:0x2_a5e3_7766; bits[9]:0x1ff; bits[56]:0x7f_ffff_ffff_ffff"
//     args: "bits[4]:0x5; bits[46]:0xcd2_ffff_f747; bits[3]:0x5; bits[34]:0x2_f929_2d94; bits[9]:0xc2; bits[56]:0x6c_a334_ea46_bf2e"
//     args: "bits[4]:0x5; bits[46]:0x1fff_ffff_ffff; bits[3]:0x2; bits[34]:0x1_4a4e_6bda; bits[9]:0x1db; bits[56]:0x0"
//     args: "bits[4]:0xa; bits[46]:0x2745_b592_54d3; bits[3]:0x0; bits[34]:0x1_5555_5555; bits[9]:0x1c2; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0xf; bits[46]:0x1fff_ffff_ffff; bits[3]:0x3; bits[34]:0x3_ff7f_3e9d; bits[9]:0xd5; bits[56]:0x73_e075_4fef_31aa"
//     args: "bits[4]:0x5; bits[46]:0x2420_4000_00a0; bits[3]:0x1; bits[34]:0x0; bits[9]:0x54; bits[56]:0xed_4062_80ba_ac1b"
//     args: "bits[4]:0x4; bits[46]:0x1115_5555_5555; bits[3]:0x5; bits[34]:0x3_2d47_075e; bits[9]:0x184; bits[56]:0xf6_d9bf_56ed_7bb3"
//     args: "bits[4]:0x2; bits[46]:0x3fff_ffff_ffff; bits[3]:0x2; bits[34]:0x1_5d45_5555; bits[9]:0x1ff; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0xa; bits[46]:0x28c5_25ff_6e66; bits[3]:0x0; bits[34]:0x2_19cb_3a48; bits[9]:0x3f; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0xa; bits[46]:0x36e5_2746_9150; bits[3]:0x4; bits[34]:0x1_3e4d_8f18; bits[9]:0x144; bits[56]:0xd7_541c_54cd_0589"
//     args: "bits[4]:0xf; bits[46]:0x1c7b_d5ac_6bf3; bits[3]:0x2; bits[34]:0x1_ffff_ffff; bits[9]:0x1a7; bits[56]:0xb7_eefb_ffff_ff3e"
//     args: "bits[4]:0x6; bits[46]:0x0; bits[3]:0x0; bits[34]:0x1_31e1_caff; bits[9]:0xaa; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0x1; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x2; bits[34]:0x1_ffff_ffff; bits[9]:0x10; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x7; bits[46]:0x1abb_baab_aaaa; bits[3]:0x2; bits[34]:0x1_dfff_ffff; bits[9]:0x0; bits[56]:0x0"
//     args: "bits[4]:0x0; bits[46]:0x0; bits[3]:0x5; bits[34]:0x0; bits[9]:0x60; bits[56]:0x7f_ffff_ffff_ffff"
//     args: "bits[4]:0xf; bits[46]:0x3ffd_7dfb_efff; bits[3]:0x4; bits[34]:0x1_5555_5555; bits[9]:0xaa; bits[56]:0x55_d555_557f_ffff"
//     args: "bits[4]:0x0; bits[46]:0x23ba_2e82_8ab8; bits[3]:0x4; bits[34]:0x3_01d6_9235; bits[9]:0x119; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0xa; bits[46]:0x0; bits[3]:0x5; bits[34]:0x2_aaaa_e8aa; bits[9]:0x18; bits[56]:0x1_0000_0000_0000"
//     args: "bits[4]:0x5; bits[46]:0x100_0000_0000; bits[3]:0x5; bits[34]:0x3_ffff_ffff; bits[9]:0x138; bits[56]:0x800_0000"
//     args: "bits[4]:0xf; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x7; bits[34]:0x3_95f5_5456; bits[9]:0xb9; bits[56]:0x0"
//     args: "bits[4]:0xa; bits[46]:0x100; bits[3]:0x5; bits[34]:0x93b_410c; bits[9]:0x157; bits[56]:0x0"
//     args: "bits[4]:0x5; bits[46]:0x2526_f8e7_cf1f; bits[3]:0x7; bits[34]:0x2_b087_4fb7; bits[9]:0x193; bits[56]:0xbc_36df_fd8b_f6ef"
//     args: "bits[4]:0xa; bits[46]:0x0; bits[3]:0x0; bits[34]:0x1_ffff_ffff; bits[9]:0x155; bits[56]:0x7d_dfff_7fdd_457d"
//     args: "bits[4]:0x7; bits[46]:0x1d00_4525_d876; bits[3]:0x7; bits[34]:0x1_a888_4155; bits[9]:0x155; bits[56]:0x20_0000_0000"
//     args: "bits[4]:0x7; bits[46]:0x153a_06ea_afbc; bits[3]:0x4; bits[34]:0x0; bits[9]:0x14b; bits[56]:0x4_0000_0000_0000"
//     args: "bits[4]:0xa; bits[46]:0x10; bits[3]:0x7; bits[34]:0x10; bits[9]:0x14; bits[56]:0xa2_7345_09cc_ac90"
//     args: "bits[4]:0xf; bits[46]:0x3fff_ffff_ffff; bits[3]:0x4; bits[34]:0x3_9e7d_bfff; bits[9]:0x2; bits[56]:0x80_010c_8100_0088"
//     args: "bits[4]:0xa; bits[46]:0x3fff_ffff_ffff; bits[3]:0x3; bits[34]:0x3_ffff_ffff; bits[9]:0x1fb; bits[56]:0xff_ffff_ffff_ffff"
//     args: "bits[4]:0x5; bits[46]:0x0; bits[3]:0x5; bits[34]:0x1_ffff_ffff; bits[9]:0x155; bits[56]:0xb6_b3a8_f83b_0813"
//     args: "bits[4]:0xf; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x6; bits[34]:0x3_d7d3_ffe7; bits[9]:0x0; bits[56]:0x7f_ffff_ffff_ffff"
//     args: "bits[4]:0x7; bits[46]:0x1d4f_7445_5d54; bits[3]:0x7; bits[34]:0x0; bits[9]:0xaa; bits[56]:0x30_2041_9482_101b"
//     args: "bits[4]:0x7; bits[46]:0x1fff_ffff_ffff; bits[3]:0x3; bits[34]:0x1_abea_bba6; bits[9]:0x1ff; bits[56]:0x4000_0000_0000"
//     args: "bits[4]:0x0; bits[46]:0x3df_fc77_fbff; bits[3]:0x2; bits[34]:0x3fff_ffff; bits[9]:0x1d8; bits[56]:0x57_47d7_5557_1415"
//     args: "bits[4]:0x7; bits[46]:0x3f7c_ee9f_7eff; bits[3]:0x7; bits[34]:0x3_fa0b_4094; bits[9]:0xf1; bits[56]:0x9d_8285_96aa_a4f8"
//     args: "bits[4]:0x3; bits[46]:0x1555_5555_5555; bits[3]:0x3; bits[34]:0xd9bf_8caf; bits[9]:0x1ff; bits[56]:0x55_5555_5555_5555"
//     args: "bits[4]:0x5; bits[46]:0x3edb_f818_3c96; bits[3]:0x1; bits[34]:0x1_4000_0011; bits[9]:0x155; bits[56]:0x50_0000_0440_0000"
//     args: "bits[4]:0x0; bits[46]:0x1ff_ffff_ffff; bits[3]:0x3; bits[34]:0x3_dc7d_baf3; bits[9]:0xba; bits[56]:0xf7_1f6e_bcdf_ffff"
//     args: "bits[4]:0x7; bits[46]:0xfea_bffa_ed32; bits[3]:0x7; bits[34]:0x1_ffff_ffff; bits[9]:0xaa; bits[56]:0x9f_97fb_55db_0fbb"
//     args: "bits[4]:0x0; bits[46]:0x2000_0000; bits[3]:0x5; bits[34]:0x0; bits[9]:0x3b; bits[56]:0x7f_ffff_ffff_ffff"
//     args: "bits[4]:0x0; bits[46]:0x0; bits[3]:0x0; bits[34]:0x3_ffff_ffff; bits[9]:0x157; bits[56]:0x6d_be6b_099d_aad5"
//     args: "bits[4]:0x5; bits[46]:0x1701_8e58_5e24; bits[3]:0x1; bits[34]:0x2_aaaa_aaaa; bits[9]:0x187; bits[56]:0xd_2d20_1e66_e8bf"
//     args: "bits[4]:0x5; bits[46]:0x0; bits[3]:0x2; bits[34]:0x1; bits[9]:0xff; bits[56]:0xd3_aaae_2a52_561d"
//     args: "bits[4]:0x5; bits[46]:0x2aaa_aaaa_aaaa; bits[3]:0x7; bits[34]:0xacf9_0262; bits[9]:0x0; bits[56]:0x60_43c2_9a10_41a3"
//     args: "bits[4]:0x0; bits[46]:0x0; bits[3]:0x0; bits[34]:0x2_0000_2480; bits[9]:0xaa; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0x5; bits[46]:0x0; bits[3]:0x0; bits[34]:0x0; bits[9]:0x80; bits[56]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[4]:0x5; bits[46]:0x0; bits[3]:0x7; bits[34]:0xa5d6_dad1; bits[9]:0xd3; bits[56]:0x0"
//     args: "bits[4]:0xf; bits[46]:0xdff_1b6d_f7fa; bits[3]:0x2; bits[34]:0x0; bits[9]:0x1e9; bits[56]:0x4c_ea06_2638_1a38"
//     args: "bits[4]:0x0; bits[46]:0x800_0000_0000; bits[3]:0x7; bits[34]:0x2_8a8b_a2ca; bits[9]:0x2; bits[56]:0x7f_ffff_ffff_ffff"
//     args: "bits[4]:0x5; bits[46]:0x16aa_aa2c_aa2a; bits[3]:0x5; bits[34]:0x0; bits[9]:0x175; bits[56]:0x35_f9d9_fd68_8f5e"
//     args: "bits[4]:0xa; bits[46]:0x3fff_ffff_ffff; bits[3]:0x2; bits[34]:0x2_afff_f7bd; bits[9]:0x1ff; bits[56]:0x53_0bb5_9c56_28f8"
//     args: "bits[4]:0x7; bits[46]:0x1aff_efff_efef; bits[3]:0x7; bits[34]:0x3_6e6b_efa1; bits[9]:0x1f4; bits[56]:0x6b_efbf_ffbf_bc08"
//     args: "bits[4]:0x1; bits[46]:0x1bc7_4ba9_02f0; bits[3]:0x7; bits[34]:0x6baa_bbaa; bits[9]:0xaa; bits[56]:0x68_7f8a_ee0e_2ab2"
//     args: "bits[4]:0xa; bits[46]:0x3bff_14cb_a9b6; bits[3]:0x2; bits[34]:0x3_ffff_ffff; bits[9]:0x155; bits[56]:0xaa_8002_0980_0000"
//     args: "bits[4]:0x8; bits[46]:0x33f5_50b1_e4d4; bits[3]:0x0; bits[34]:0x1_ffff_ffff; bits[9]:0x15f; bits[56]:0x10_0000_0000_0000"
//   }
// }
// 
// END_CONFIG
fn x7<x11: u13 = {u13:0xfff}>(x8: u56, x9: u34, x10: u3) -> (u44, u4, u13, u4) {
    {
        let x12: u4 = x9[x11+:xN[bool:0x0][4]];
        let x13: u44 = match x12 {
            u4:0xa..u4:15 => u44:0x555_5555_5555,
            u4:15..u4:0x2 | u4:0x8..u4:8 => u44:0x7ff_ffff_ffff,
            _ => u44:0x0,
        };
        let x14: u13 = x11[x8+:u13];
        (x13, x12, x14, x12)
    }
}
fn main(x0: u4, x1: u46, x2: u3, x3: u34, x4: u9, x5: u56) -> (u13, u44, u13) {
    {
        let x6: u46 = x5[3:49];
        let x15: (u44, u4, u13, u4) = x7(x5, x3, x2);
        let (.., x16, x17, x18, x19) = x7(x5, x3, x2);
        (x18, x16, x18)
    }
}
