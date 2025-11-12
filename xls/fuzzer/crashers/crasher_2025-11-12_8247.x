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
// exception: "Subprocess call timed out after 1500 seconds"
// issue: "https://github.com/google/xls/issues/3370"
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
//   codegen_args: "--pipeline_stages=2"
//   codegen_args: "--worst_case_throughput=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
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
//   codegen_ng: true
//   disable_unopt_interpreter: false
// }
// inputs {
//   function_args {
//     args: "bits[49]:0x1; bits[38]:0x3f_ffff_ffff; bits[27]:0x16_436b; bits[23]:0x2a_aaaa; bits[46]:0xc33_375c_4040"
//     args: "bits[49]:0x1_c981_6fe3_5674; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x2a8_aa8b; bits[23]:0x10_0000; bits[46]:0xc00_046e_baea"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x1f_ffff_ffff; bits[27]:0x2d6_fa57; bits[23]:0x5c_d677; bits[46]:0xcb4_55bd_9645"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x3b_fbf7_dbef; bits[27]:0x137_5bf7; bits[23]:0x55_5555; bits[46]:0xdc2_1038_85ae"
//     args: "bits[49]:0x0; bits[38]:0x1_9494_3314; bits[27]:0x767_fe15; bits[23]:0xc_0000; bits[46]:0x3b36_61fc_7513"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x2a2_0aae; bits[23]:0x2a_aaaa; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0x1_d1af_3439_f513; bits[38]:0x2f_1559_f117; bits[27]:0x579_d117; bits[23]:0x59_f117; bits[46]:0x4000"
//     args: "bits[49]:0x4000_0000_0000; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x6aa_8aa2; bits[23]:0x2a_aaaa; bits[46]:0x145f_15a3_0425"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x1f_ffff_ffff; bits[27]:0x40; bits[23]:0x79_eddb; bits[46]:0x16bf_b11f_b7b7"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x100_0000; bits[27]:0x3b3_20c0; bits[23]:0x55_5555; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x6ee_2933; bits[23]:0x3f_ffff; bits[46]:0x1c1b_9a29_ca55"
//     args: "bits[49]:0x826f_c549_0269; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x0; bits[23]:0x55_5555; bits[46]:0x2aaa_aaaa_aaaa"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x15_5555_5555; bits[27]:0x545_55dd; bits[23]:0x2a_aaaa; bits[46]:0x15d5_4575_750c"
//     args: "bits[49]:0x0; bits[38]:0x3f_ffff_ffff; bits[27]:0x740_6ac7; bits[23]:0x0; bits[46]:0x3fff_ffff_ffff"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x3b_eddf_fffe; bits[27]:0x0; bits[23]:0x5f_fffe; bits[46]:0x0"
//     args: "bits[49]:0x0; bits[38]:0x4_2028_90f0; bits[27]:0x2aa_aaaa; bits[23]:0x2104; bits[46]:0x2080_8221_0050"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x3f_7dff_ff7f; bits[27]:0x7e0_c3f7; bits[23]:0x2a_aaaa; bits[46]:0x3fff_ffff_ffff"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0xf_e22d_5a3e; bits[27]:0x630_8225; bits[23]:0x26_a8a2; bits[46]:0x2fca_0c5a_0639"
//     args: "bits[49]:0x0; bits[38]:0x2_0a01_1a46; bits[27]:0x555_5555; bits[23]:0x61_fe5e; bits[46]:0x22aa_aaab_ffef"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x2aa_aaaa; bits[23]:0x2a_aaaa; bits[46]:0x3fb7_fe7d_9faf"
//     args: "bits[49]:0x2000_0000_0000; bits[38]:0x10_0000_0022; bits[27]:0x2aa_aaaa; bits[23]:0x2e_aaba; bits[46]:0x10ac_c587_17c0"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x17_6cc4_5186; bits[27]:0x401_56c4; bits[23]:0x0; bits[46]:0x2551_264e_8ebb"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x3f_ffff_ffff; bits[27]:0x0; bits[23]:0x55_5555; bits[46]:0x2aaa_aaaa_aaaa"
//     args: "bits[49]:0x1fc8_d10f_5ffd; bits[38]:0x1a_c083_5bd1; bits[27]:0x31f_5ffd; bits[23]:0x15_2435; bits[46]:0x1ec0_c01b_d9ee"
//     args: "bits[49]:0x0; bits[38]:0x0; bits[27]:0x60_0081; bits[23]:0x6e_9de2; bits[46]:0x843_6822_0044"
//     args: "bits[49]:0x3445_6feb_2480; bits[38]:0x1f_ffff_ffff; bits[27]:0x2e9_0180; bits[23]:0x7f_e0ba; bits[46]:0x0"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x28_aaaf_baaa; bits[27]:0x28f_baaa; bits[23]:0x63_be0c; bits[46]:0x2daa_ad3a_a88a"
//     args: "bits[49]:0x1_7967_7d7c_23d3; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x555_5555; bits[23]:0x4e_c1a9; bits[46]:0x3967_75f5_27c3"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x0; bits[27]:0xbb_aaaa; bits[23]:0x0; bits[46]:0x0"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x15_5555_5555; bits[27]:0x41d_d607; bits[23]:0x2a_aaaa; bits[46]:0x1555_557f_ffff"
//     args: "bits[49]:0x261d_1d26_fe9b; bits[38]:0x19_1522_eedb; bits[27]:0x555_5555; bits[23]:0x2f_b29b; bits[46]:0x2415_0d7d_94d7"
//     args: "bits[49]:0x1_eb94_3621_92f9; bits[38]:0x14_3f01_d6e9; bits[27]:0x22_83ab; bits[23]:0x7f_974d; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0x0; bits[38]:0x2_0000_0000; bits[27]:0x2aa_aaaa; bits[23]:0x3f_ffff; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x3f_f7ff_fffe; bits[27]:0x3ff_ffff; bits[23]:0x8000; bits[46]:0x3ff7_ffed_ea21"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x0; bits[27]:0x0; bits[23]:0x200; bits[46]:0x2a04_1022_0a68"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x33_79b7_2dbf; bits[27]:0x1b3_ad1f; bits[23]:0x1b_5776; bits[46]:0x43b_9cb8_56a1"
//     args: "bits[49]:0x1_7a0b_ad0b_63e0; bits[38]:0xb_ad0b_6be0; bits[27]:0x79b_6bac; bits[23]:0x7f_ffff; bits[46]:0x38fa_5d60_0040"
//     args: "bits[49]:0x1_0000_0000_0000; bits[38]:0x2_cede_0b4c; bits[27]:0x672_b802; bits[23]:0x55_5555; bits[46]:0xeea_0b99_4810"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x11_a307_5bd4; bits[27]:0x0; bits[23]:0x55_7584; bits[46]:0xc73_6735_0705"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x3f_ffff_ffff; bits[27]:0x3b2_ab7d; bits[23]:0x12_a97c; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0x0; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x555_5555; bits[23]:0x2a_baaa; bits[46]:0x2ee2_917a_60b5"
//     args: "bits[49]:0x20_0000_0000; bits[38]:0x21_22e6_2094; bits[27]:0x7ff_ffff; bits[23]:0x51_9092; bits[46]:0x2bf9_5527_f877"
//     args: "bits[49]:0x0; bits[38]:0x3f_ffff_ffff; bits[27]:0x7ff_ffff; bits[23]:0x2a_b357; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x2d_2988_7b62; bits[27]:0x57d_c571; bits[23]:0xc_7b50; bits[46]:0x3570_0329_7f48"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0xdec6_784d; bits[27]:0x1cf_df9b; bits[23]:0x2a_aaaa; bits[46]:0x1555_5400_0c00"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x28_ef33_2f73; bits[27]:0x6bb_b5bf; bits[23]:0x1a_4d73; bits[46]:0x28e5_332f_73a3"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x3f_ffff_ffff; bits[27]:0x3de_518f; bits[23]:0x2a_aaaa; bits[46]:0x2dca_e7e9_b6f8"
//     args: "bits[49]:0x0; bits[38]:0x15_5555_5555; bits[27]:0x700_0801; bits[23]:0x5d_7f7d; bits[46]:0x242c_0890_d1c4"
//     args: "bits[49]:0x0; bits[38]:0x20_280e_12a9; bits[27]:0x40; bits[23]:0x55_5555; bits[46]:0x90a_3f47_a77b"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x3f_ffdf_fffe; bits[27]:0x7fc_fef1; bits[23]:0x7f_fefa; bits[46]:0x2000_0000_0000"
//     args: "bits[49]:0x400; bits[38]:0x18_5005_b40e; bits[27]:0x223_b15c; bits[23]:0x1c_a4fa; bits[46]:0x44c_8105_ba4c"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x3f_ffff_ffff; bits[27]:0x7aa_aaba; bits[23]:0x30_4dcf; bits[46]:0xb90_f2bf_e91a"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x8_0000; bits[27]:0x15e_4b5d; bits[23]:0x77_cb7b; bits[46]:0x3fff_ffff_ffff"
//     args: "bits[49]:0x8_0000; bits[38]:0x0; bits[27]:0x7ff_ffff; bits[23]:0x3f_ffff; bits[46]:0x3fdf_fbeb_efff"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x4; bits[27]:0x208_8f24; bits[23]:0x7f_ffff; bits[46]:0x3fff_fbef_7fd6"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x1f_3bef_f6ef; bits[27]:0x7df_ffff; bits[23]:0x2; bits[46]:0x371e_49d8_8e9e"
//     args: "bits[49]:0x800_0000; bits[38]:0x15_5555_5555; bits[27]:0x230_8002; bits[23]:0x2a60; bits[46]:0x1d97_9f45_4408"
//     args: "bits[49]:0x2e62_be20_5af2; bits[38]:0x0; bits[27]:0x250_7af5; bits[23]:0x4000; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x2e_a2aa_2aaa; bits[27]:0x7ff_ffff; bits[23]:0x7f_ffbf; bits[46]:0xd1f_2f6b_fccf"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x20_0fe9_a0bb; bits[27]:0x20_0000; bits[23]:0x3f_ffff; bits[46]:0x2aaa_2aaa_cbaa"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x3ff_ffff; bits[23]:0x7b_dfef; bits[46]:0x8"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x3f_ffff_ffff; bits[27]:0x7ff_ffbf; bits[23]:0x7e_de9f; bits[46]:0x30ff_6e3d_ebff"
//     args: "bits[49]:0x8000_0000_0000; bits[38]:0x1020_0c8c; bits[27]:0x2aa_aaaa; bits[23]:0x3f_ffff; bits[46]:0x0"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x2aa_aaaa; bits[23]:0x38_3ae3; bits[46]:0x1311_5553_ffff"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x15_5595_5545; bits[27]:0x555_5555; bits[23]:0x54_5715; bits[46]:0x0"
//     args: "bits[49]:0x0; bits[38]:0x1f_ffff_ffff; bits[27]:0x555_5555; bits[23]:0x0; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x3c_2dbb_36fe; bits[27]:0xe2_cc77; bits[23]:0x67_ce77; bits[46]:0x0"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x15_5555_5555; bits[27]:0x0; bits[23]:0x55_5545; bits[46]:0x3cee_52cb_fabf"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x29_2a8c_3ea0; bits[27]:0x2d4_1852; bits[23]:0x2d_e452; bits[46]:0x3fff_ffff_ffff"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x1f_ffff_ffff; bits[27]:0x7ba_4b4f; bits[23]:0x2a_aaaa; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x14_9555_5555; bits[27]:0x75d_4555; bits[23]:0x5d_4555; bits[46]:0x15a1_1545_f500"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x3f_ffff_ffff; bits[27]:0x0; bits[23]:0x77_ffff; bits[46]:0x1553_2b1a_6aaf"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x7_6f2a_4a6d; bits[27]:0x3ff_ffff; bits[23]:0x8000; bits[46]:0x4_0000_0000"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x1f_decf_7ffb; bits[27]:0x6cb_7bda; bits[23]:0x2a_aaaa; bits[46]:0x3bde_cd7f_fb80"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x8_0000; bits[27]:0x2aa_aaaa; bits[23]:0x1a_a881; bits[46]:0x86_928c_892c"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x15_4f55_dd55; bits[27]:0x772_45c0; bits[23]:0x2a_aaaa; bits[46]:0x3c9d_15ea_ffbf"
//     args: "bits[49]:0x0; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x8400; bits[23]:0x39_36b0; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x0; bits[27]:0x555_5555; bits[23]:0xa_24b3; bits[46]:0x3ea2_a8aa_aa88"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x3f_ffff_ffff; bits[27]:0x5b_5fb7; bits[23]:0x55_5555; bits[46]:0x3df2_dfef_fde0"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x15_5175_5545; bits[27]:0x171_5d15; bits[23]:0x2a_aaaa; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x2a_aaaa_abaa; bits[27]:0x7ff_ffff; bits[23]:0x0; bits[46]:0x1000_0007_959f"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x0; bits[27]:0x1_0000; bits[23]:0x76_2529; bits[46]:0x2aaa_aaaa_aaaa"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x1f_f7fe_fe7d; bits[27]:0x7be_feed; bits[23]:0x7f_b7ff; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0x1_5985_f251_616e; bits[38]:0x1f_ffff_ffff; bits[27]:0x3f7_714f; bits[23]:0x63_e65c; bits[46]:0xd21_3500_879b"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x16_ee82_1c9b; bits[27]:0x0; bits[23]:0x55_5555; bits[46]:0x13a6_e125_8bd5"
//     args: "bits[49]:0x988e_9f8f_f909; bits[38]:0x8_8ca4_ebe1; bits[27]:0x5a8_e74f; bits[23]:0x2a_e74f; bits[46]:0x3573_83d5_5637"
//     args: "bits[49]:0x88a6_5614_7209; bits[38]:0x25_5f10_320a; bits[27]:0x3ff_ffff; bits[23]:0x10; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x2d_1f24_edcd; bits[27]:0x7ff_ffff; bits[23]:0x7_fefd; bits[46]:0x317_32b1_d850"
//     args: "bits[49]:0x4000; bits[38]:0x0; bits[27]:0x74_0100; bits[23]:0x68_2100; bits[46]:0x1045_047c_5f4e"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x39_bda9_89be; bits[27]:0x173_99cd; bits[23]:0x2e_bbe9; bits[46]:0x2ba2_eeaa_8bb8"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x15_5555_5555; bits[27]:0x659_1157; bits[23]:0x0; bits[46]:0xc04_619e_602d"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x1f_543f_cf19; bits[27]:0x438_7f54; bits[23]:0x0; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x2_0000; bits[27]:0x7df_f8ff; bits[23]:0x2a_aaaa; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x7_7ccc_0dbb; bits[27]:0x4ec_7ddb; bits[23]:0x44_7dd3; bits[46]:0x20a3_f4c4_ecbf"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x35_d7df_aeff; bits[27]:0x65b_b3b1; bits[23]:0x29_baa2; bits[46]:0x1bd9_9fb5_5550"
//     args: "bits[49]:0x63da_aa07_fae0; bits[38]:0x3a_a207_bae8; bits[27]:0x285_37a1; bits[23]:0x3f_ffff; bits[46]:0x1529_bd07_ffff"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x2e_7f23_fb0e; bits[27]:0x782_f92d; bits[23]:0x2_f92d; bits[46]:0x17c_9684_0000"
//     args: "bits[49]:0x1_f59b_651d_5f2d; bits[38]:0x1b_651d_5f2d; bits[27]:0x515_474d; bits[23]:0x1e_fa0d; bits[46]:0x80_0000"
//     args: "bits[49]:0x270b_784a_c718; bits[38]:0x19_71dc_2361; bits[27]:0x652_5faf; bits[23]:0x4; bits[46]:0x1a7f_fa41_699a"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x1b_bdfd_3abc; bits[27]:0x5bd_ba01; bits[23]:0x7f_fdff; bits[46]:0x37fc_ffcf_eff3"
//     args: "bits[49]:0x34ed_1df9_b063; bits[38]:0x2d_1df1_b063; bits[27]:0x7ff_ffff; bits[23]:0x55_5555; bits[46]:0x26ed_19f4_e866"
//     args: "bits[49]:0x0; bits[38]:0x2f_5fa2_1f66; bits[27]:0x238_0024; bits[23]:0x2a_aaaa; bits[46]:0x30e5_857f_f03f"
//     args: "bits[49]:0x0; bits[38]:0x2010_4501; bits[27]:0x0; bits[23]:0x8_a000; bits[46]:0x1621_9e86_9aa9"
//     args: "bits[49]:0x0; bits[38]:0x25_f2e1_1715; bits[27]:0x555_5555; bits[23]:0x2a_aaaa; bits[46]:0x0"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x3f_ffff_ffff; bits[27]:0x7eb_6bd5; bits[23]:0x4c_bc65; bits[46]:0x3e7e_4fd5_0f50"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x12_2a69_537e; bits[27]:0x2e1_437e; bits[23]:0x65_d274; bits[46]:0x2442_b3f0_0402"
//     args: "bits[49]:0x400_0000; bits[38]:0x2400_0080; bits[27]:0x0; bits[23]:0x60_2806; bits[46]:0x329b_ab08_2100"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x2a_aaaa_aaaa; bits[27]:0x792_74f7; bits[23]:0x2_74d7; bits[46]:0x2a9a_aaba_aa2f"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x1f_f76f_eafb; bits[27]:0x7e2_b69c; bits[23]:0x62_b49c; bits[46]:0xfde_6ee6_a580"
//     args: "bits[49]:0x0; bits[38]:0x3_09f9_0807; bits[27]:0x3ff_ffff; bits[23]:0x7a_ff7e; bits[46]:0x1ff7_fffb_ffff"
//     args: "bits[49]:0x5568_ef11_9ff0; bits[38]:0x3a_ff61_0fbc; bits[27]:0x711_bd50; bits[23]:0xb_f157; bits[46]:0x21fd_678f_1a55"
//     args: "bits[49]:0x0; bits[38]:0x10_240e_080a; bits[27]:0x480_7c8c; bits[23]:0x2a_aaaa; bits[46]:0x0"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x1f_ffff_ffff; bits[27]:0x3fd_ffbf; bits[23]:0x7d_6fbf; bits[46]:0x1c5f_d1ff_f9f7"
//     args: "bits[49]:0x34f0_2406_59d6; bits[38]:0x30_2406_f9d6; bits[27]:0x406_59d6; bits[23]:0x3f_ffff; bits[46]:0x1037_4e33_beff"
//     args: "bits[49]:0x0; bits[38]:0xa_1400_0044; bits[27]:0x2aa_aaaa; bits[23]:0x55_5555; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x2000_0000; bits[27]:0x555_5555; bits[23]:0x3f_ffff; bits[46]:0x3bfa_f1af_9577"
//     args: "bits[49]:0x800_0000; bits[38]:0x15_5555_5555; bits[27]:0x7ff_ffff; bits[23]:0x79_ef7f; bits[46]:0x1de8_bdb8_29af"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x14_cb91_8ef5; bits[27]:0x64d_7fbf; bits[23]:0x5e_8d4d; bits[46]:0x0"
//     args: "bits[49]:0xe1f2_8b4b_e277; bits[38]:0x4_0000; bits[27]:0x0; bits[23]:0x55_5555; bits[46]:0x4003_1157"
//     args: "bits[49]:0x1_5555_5555_5555; bits[38]:0x3f_ffff_ffff; bits[27]:0x0; bits[23]:0x18_9b0a; bits[46]:0x400_0000_0000"
//     args: "bits[49]:0x1000_0000_0000; bits[38]:0x15_5555_5555; bits[27]:0x1d7_0114; bits[23]:0x2_0000; bits[46]:0xeba_0083_fffb"
//     args: "bits[49]:0x0; bits[38]:0x4b0_8800; bits[27]:0x3ff_ffff; bits[23]:0x7b_ffed; bits[46]:0x1555_5555_5555"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x20; bits[27]:0x99_af51; bits[23]:0xd_8d52; bits[46]:0x662_628d_ae18"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x2d_1488_8e4a; bits[27]:0x498_864a; bits[23]:0x0; bits[46]:0x2aaa_aaaa_aaaa"
//     args: "bits[49]:0xffff_ffff_ffff; bits[38]:0x2a_78a7_ca50; bits[27]:0x187_8851; bits[23]:0x27_8a92; bits[46]:0x1fff_ffff_ffff"
//     args: "bits[49]:0x1_ffff_ffff_ffff; bits[38]:0x3f_bfff_fd7e; bits[27]:0x6ff_fc5e; bits[23]:0x55_5555; bits[46]:0x35fd_62fd_5554"
//     args: "bits[49]:0xaaaa_aaaa_aaaa; bits[38]:0x1f_ffff_ffff; bits[27]:0x7a8_8aaa; bits[23]:0x3f_ffff; bits[46]:0x0"
//     args: "bits[49]:0x0; bits[38]:0x15_5555_5555; bits[27]:0x0; bits[23]:0x4d_5755; bits[46]:0x26ab_aa80_4421"
//   }
// }
// 
// END_CONFIG
const W32_V1567 = u32:0x61f;
const W32_V46 = xN[bool:0x0][32]:0x2e;
type x19 = bool;
type x22 = u8;
fn main(x0: s49, x1: u38, x2: u27, x3: u23, x4: u46) -> (x22[W32_V1567], u23, s49) {
    {
        let x5: s49 = x1 as s49 & x0;
        let x6: u23 = x4 as u23 ^ x3;
        let x7: bool = xor_reduce(x4);
        let x8: u27 = x2[:];
        let x9: u38 = -x1;
        let x10: u27 = !x2;
        let x11: (s49,) = (x5,);
        let (x12) = (x5,);
        let x13: u27 = ctz(x8);
        let x15: bool = {
            let x14: (bool, bool) = umulp(x2 as bool, x7);
            x14.0 + x14.1
        };
        let x16: s49 = x3 as s49 & x0;
        let x17: u23 = x6 ^ x12 as u23;
        let x18: u27 = -x10;
        let x20: x19[W32_V46] = x4 as x19[W32_V46];
        let x21: bool = x20 == x20;
        let x23: x22[W32_V1567] = "qW]!zqc*7Z]WBW3$YWA)vqM0\'\'1JJa{d:XY%Bf~S8v*HiI\"B[)\'8##)a<lU1F\\fU<FUcxKt;I=tAn{ 8Nf;G;)HEzwD3S=jk9q!/,}1ernXPvg_4LF\'uz[yu2HF3v_.\'sZ !YX>gc5A65`F++oQS:%q2)4]|V%[f w(Jo=R29j0pfb/cCT]?)q47B>(s*t56iy8/p^\\awEOe@yp=BW(89N4J?gIE:JCm` BaT7AN<yGSeVBG!A8z:d-CzvoJWS3mpM!,T;BT:/3]i99PX<%/A@Y#W>!}@/\"%\'\\\' lmnS)N#k=*b+lBy[=?5gn~0]9@?i/$zdUIpwQF-ZXz71tii9,NP\\^#|~qlH8?vRe>a@Y6#Zh>(OknO!SYEArq2gucv!C@iy~F\\ O-?8j;X1`A<3h5!)Y10w\\7_h,6ea%xuYh\'Ie0<a2shLM4Azb~$N]]in:d#&Vmn~4{uIl\'`[2I<UuA3gWxWK\"(_rc%0.=}yt&B!8-nOI5,[GLd_/F]tNfhz@i9&)6KQO=NO&HuU!GoM>mk4|L/A)N\'wv5R6.@at lZY-7B*5V7m2_1F&m\"t!\\AHk?aQ?}rd)tZ5y$VXoLe`98UahF;FQ~?wdmn_4V~FS}HQ\"B%hR.?%N\\B>2<hD=anCLwr/ `l:!.~6}$?x Vz^$}vD -kDVZkG4WmD~]zT0+(#*Dyrr*d!2$eZu0d]D?Sng^8@@CLO@I$J\"~,ww+L[(k-vduHiR6Z#F-SJC\\D&4r!?QU[{r\'w;\"!:k8&X`2}W9T(u_GB*\\R^r<%mxg9%!GDsR$P|An!J1k&Ce?Ay&\"N/pPybEIo;Yj6PU2YHbJ(2\'1jOU-AiqtDLR4K:)<2`Yi[Se4%*ZA6EJpWdqD&2\"IEwv->G;AjpW XN8ulA;Cq|K\"Kc@elXADfw\"W;=W`h@/c!Q4s;RwF@1;ETG>im&)]vSdS0Qg S]vH*hUp&EO{ROb`T5A7o0t2#%l\\GU6t~8\"rj:&$HXwr2Nvvz`O#x|9f`(ok?d|==NCm8?.#(O5g_*g|R^n]u7e!-Ze@g&U=Z;]}G_ s8`_ntyc!6u;qC:W&HOycw.Q[EQ6ar(d%Osbm4\\?F]sZ1-zDJlY%>M9cA2g3tKPQE3Ul?,3a/#|_$TCIb>>[?9jxnRPP2{Fq,\"]321 !GtRV-+Ow?e8iQhu&~g\\eT$d[/@Y7`H5g\"\"uFG6b;%gLaTQ69\'V ]m:KEP<W-wq:@(;Hje<&DX5m \'M|*z+aw\"oLnUEN8~61UQi$s\"=>%>su2g]&hK^U%YP>wN%(q=4.g>|@h55j2R5meC] l+V\\DVc9]pb9x>Xb-L@u8Ak<)vZE4CGfz|uSq(#kcb.<S#N~{be)K~Etz &l2zowRQn%HY#?.HBmq0Z%>{^=#d\"*6NglH8d%G1.6hBZ;LY+8=.El|;zdX5,KD1V<w.YhCEX|%.*9r7z hjVy?\\C\"qH|SZ8`sU\\*UI_xJP*DdM~)q1.^ksUCOTxc/]\'?}5+&wjiZ}-KX7Cy J{rDt)ourDhv[IsokrmzmSRJv0[8\\lAR&;aWiEs%c:K#YZ2Lp?Ujk/MLf[TA`8l@m/\\n!yM5B";
        let x24: s49 = x11.0;
        let x25: bool = x11 != x11;
        let x26: bool = x20 == x20;
        let x27: u27 = x0 as u27 + x13;
        let x28: x22[1] = array_slice(x23, x3, x22[1]:[x23[u32:0x0], ...]);
        let x29: s49 = !x24;
        let x30: u38 = !x1;
        let x31: u23 = (x12 as u49)[x6+:u23];
        let x32: bool = x28 == x28;
        let x33: bool = x26[x32+:bool];
        let x34: u27 = one_hot_sel(x21, [x18]);
        let x35: u27 = for (i, x) in u4:0x0..u4:0x1 {
            x
        }(x27);
        (x23, x6, x5)
    }
}