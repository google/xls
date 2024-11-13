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
//
// BEGIN_CONFIG
// exception: "// Command \'[\'xls/tools/eval_ir_main\', \'--input_file=args.txt\', \'--use_llvm_jit\', \'sample.opt.ir\', \'--logtostderr\']\' timed out after 1500 seconds"
// issue: "https://github.com/google/xls/issues/929"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=8"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=false"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
// }
// inputs {
//   function_args {
//     args: "bits[33]:0x1_fb1e_138a; bits[41]:0xff_1432_9a1d; bits[37]:0xa_aaaa_aaaa; bits[42]:0x15d_4511_db08; bits[31]:0x1b5d_12f6; bits[28]:0xd2e_123e"
//     args: "bits[33]:0x1_f400_2007; bits[41]:0x126_c850_152d; bits[37]:0x2000_0000; bits[42]:0x384_bb27_1e46; bits[31]:0x7100_0645; bits[28]:0xb27_3e46"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xaa_aaaa_aaaa; bits[37]:0xa_c0cf_dcb5; bits[42]:0x2aa_aaaa_aaaa; bits[31]:0x71df_dca5; bits[28]:0x0"
//     args: "bits[33]:0x4000_0000; bits[41]:0x7d_a18d_48ec; bits[37]:0x15_5555_5555; bits[42]:0xaa_6b48_b529; bits[31]:0x5a48_a523; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x0; bits[41]:0x4; bits[37]:0x6891_404e; bits[42]:0xd_b208_0ed5; bits[31]:0x3940_2005; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x1ff_bfff_fffe; bits[37]:0xa_3f3b_85d6; bits[42]:0x1d7_e06d_d10e; bits[31]:0x3fff_ffff; bits[28]:0x875_895a"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x155_5555_5555; bits[37]:0x1f_ffff_ffff; bits[42]:0x2aa_8b2b_86a3; bits[31]:0x0; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0xff_ffff_ffff; bits[37]:0x1f_ffff_ffff; bits[42]:0x3ed_f97a_ebf8; bits[31]:0x40; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x2a_bae8_ab6f; bits[37]:0x15_5555_5555; bits[42]:0x45_61d9_cede; bits[31]:0x7755_5474; bits[28]:0xee8_09e3"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x40_0000; bits[37]:0xa_aaaa_aaaa; bits[42]:0x151_0f15_524d; bits[31]:0x5809_bf0e; bits[28]:0xcda_bdb8"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x1c5_f957_580a; bits[37]:0x7_dad4_f80a; bits[42]:0x1a5_77ac_7a92; bits[31]:0x7f65_45d0; bits[28]:0xadc_d802"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xaa_aaaa_aaaa; bits[37]:0xa_af0b_852f; bits[42]:0x1ff_ffff_ffff; bits[31]:0x7d9f_eb7f; bits[28]:0x18a_0193"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x187_1b49_d6c6; bits[37]:0x13_539d_1e86; bits[42]:0x2aa_aaaa_aaaa; bits[31]:0x539c_5da4; bits[28]:0x100_0000"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x41_4533_9747; bits[37]:0x17_7517_5545; bits[42]:0x298_607e_9f10; bits[31]:0x0; bits[28]:0x900_10e6"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x0; bits[37]:0x1c_3d2a_4d9b; bits[42]:0x387_2519_b17a; bits[31]:0x100; bits[28]:0x200_0000"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xfd_fdff_ff20; bits[37]:0x1b_59f1_d030; bits[42]:0x1f3_fbbe_4ac1; bits[31]:0x6fba_b9f5; bits[28]:0xfff_fdff"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x7e_7575_5f54; bits[37]:0x1000_0000; bits[42]:0x1ff_ffff_fe01; bits[31]:0x2aaa_aaaa; bits[28]:0x0"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xff_efeb_ffd5; bits[37]:0x400; bits[42]:0x1ef_dfd7_dfab; bits[31]:0x10; bits[28]:0xbc3_dbe8"
//     args: "bits[33]:0x1_9190_38f0; bits[41]:0x1b7_2338_e089; bits[37]:0xf_ffff_ffff; bits[42]:0x1ff_ffff_ffff; bits[31]:0x5fed_5fa3; bits[28]:0x1000"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x400; bits[37]:0x15_5555_5555; bits[42]:0x1ff_ffff_ffff; bits[31]:0x5555_5555; bits[28]:0x4c_5201"
//     args: "bits[33]:0x0; bits[41]:0x0; bits[37]:0x0; bits[42]:0x200_0000_0000; bits[31]:0x0; bits[28]:0x8_a676"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xaa_9aaa_2355; bits[37]:0xa_8ba9_054f; bits[42]:0x155_0162_65b5; bits[31]:0xebd_f305; bits[28]:0x162_65b5"
//     args: "bits[33]:0x159f_629e; bits[41]:0x15_df62_9eaa; bits[37]:0x1_5b76_29e5; bits[42]:0x2b_4ec5_7c81; bits[31]:0x5b76_29e5; bits[28]:0x0"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x1db_d1d2_fec9; bits[37]:0x1b_bce7_f329; bits[42]:0x3a3_a3a4_cd97; bits[31]:0x7eda_dfff; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x4d_8be5_f631; bits[37]:0xf_ffff_ffff; bits[42]:0x2; bits[31]:0x7f7f_ffff; bits[28]:0x79d_eddd"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x1ff_dffb_d655; bits[37]:0xf_ffff_ffff; bits[42]:0x1ff_ffff_ffff; bits[31]:0x6fdf_bb77; bits[28]:0xf5f_ba77"
//     args: "bits[33]:0x0; bits[41]:0x143_8142_8cd4; bits[37]:0xa_aaaa_aaaa; bits[42]:0x10_0088_00e7; bits[31]:0xb47_9a10; bits[28]:0x200_0000"
//     args: "bits[33]:0x0; bits[41]:0x88_0000_00ff; bits[37]:0x8_808a_00ff; bits[42]:0x40; bits[31]:0x3fff_ffff; bits[28]:0x0"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0xe3_428f_d594; bits[37]:0xf_ffff_ffff; bits[42]:0x12a_df7b_de8d; bits[31]:0xe2b_27d7; bits[28]:0xfff_ffff"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x1ff_ffff_ffff; bits[37]:0xa_9f75_2b84; bits[42]:0x3ff_ffff_fbff; bits[31]:0x69ae_3f2c; bits[28]:0x200"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0xff_ffff_ffff; bits[37]:0x1_faea_b160; bits[42]:0x1bb_ae2a_9bd3; bits[31]:0x7be_8d6b; bits[28]:0xaaa_aaaa"
//     args: "bits[33]:0x0; bits[41]:0x100; bits[37]:0x9_8890_8090; bits[42]:0x39_d815_89db; bits[31]:0x280_8100; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x0; bits[41]:0xff_ffff_ffff; bits[37]:0x60_4005; bits[42]:0x800_0000; bits[31]:0x7fff_ffff; bits[28]:0xdfe_f7df"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xff_ffff_ffab; bits[37]:0xf_bb9e_ee23; bits[42]:0x3a7_73dc_f4f5; bits[31]:0x5555_5555; bits[28]:0xfff_ffff"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xaa_aaaa_aaaa; bits[37]:0xa_8606_02ba; bits[42]:0xbc_bdf2_844e; bits[31]:0x7fff_ffff; bits[28]:0x0"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x20_0000; bits[37]:0x20_0000; bits[42]:0x19_8041_1131; bits[31]:0x181_11b9; bits[28]:0x9c4_1bbe"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xaa_aaaa_aaaa; bits[37]:0x4_0000; bits[42]:0x3ef_ab7f_fed7; bits[31]:0x4eec_cb9b; bits[28]:0x555_5555"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x77_fffd_ff73; bits[37]:0x9_03c6_abe9; bits[42]:0x120_78d7_753f; bits[31]:0xba4_ada2; bits[28]:0xea2_2929"
//     args: "bits[33]:0x4_0000; bits[41]:0xaa_aaaa_aaaa; bits[37]:0xf_ffff_ffff; bits[42]:0x155_5555_5555; bits[31]:0x5775_d555; bits[28]:0xaaa_0ba2"
//     args: "bits[33]:0x0; bits[41]:0x41_0a04_0dc8; bits[37]:0x2; bits[42]:0x1c0_9608_1b81; bits[31]:0x1; bits[28]:0x80_0000"
//     args: "bits[33]:0x1000; bits[41]:0x0; bits[37]:0x0; bits[42]:0x8122_0000; bits[31]:0x100_0000; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0x0; bits[41]:0x170_11b6_39bf; bits[37]:0xf_ffff_ffff; bits[42]:0x155_5555_5555; bits[31]:0x5d11_5555; bits[28]:0x0"
//     args: "bits[33]:0x1000; bits[41]:0x2030_00ff; bits[37]:0x8_0210_19ee; bits[42]:0x1_468f_3dd5; bits[31]:0x2aaa_aaaa; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x1ff_7ff7_ff55; bits[37]:0x1f_6f7f_fb94; bits[42]:0x0; bits[31]:0x2b85_c3a4; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0x4000; bits[41]:0x27_f056_0065; bits[37]:0x7_f154_0065; bits[42]:0x880_00aa; bits[31]:0x604c_2875; bits[28]:0x555_5555"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x155_5555_5555; bits[37]:0x1f_ffff_ffff; bits[42]:0x3e7_a61b_3bfc; bits[31]:0x2aaa_aaaa; bits[28]:0x8dd_e3ee"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x18a_b6aa_2151; bits[37]:0xd_2e9c_3212; bits[42]:0x317_6df4_46a3; bits[31]:0x6db8_4bd1; bits[28]:0x8b7_8ba9"
//     args: "bits[33]:0x18c1_21ed; bits[41]:0x38_8121_cd24; bits[37]:0x1_8c12_1ec5; bits[42]:0xf0_ce4a_9cb5; bits[31]:0x5555_5555; bits[28]:0x30_5fce"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x40_0000; bits[37]:0x9_ad2e_d7ec; bits[42]:0x200_6080_0180; bits[31]:0x2aaa_aaaa; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0x0; bits[41]:0xff_ffff_ffff; bits[37]:0x1_4010_2045; bits[42]:0x20d_488a_af35; bits[31]:0x3fff_ffff; bits[28]:0x555_5555"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0xcc_c3db_94a1; bits[37]:0x16_87b4_e221; bits[42]:0x2aa_aaaa_aaaa; bits[31]:0x5555_5555; bits[28]:0xff7_fafd"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x1ff_ffff_ffff; bits[37]:0x0; bits[42]:0x1ff_ffff_ffff; bits[31]:0x79b7_8f97; bits[28]:0xf67_0952"
//     args: "bits[33]:0x8000; bits[41]:0x100_010e_0cad; bits[37]:0x11_9a20_f396; bits[42]:0x1ff_ffff_ffff; bits[31]:0x7fff_ffff; bits[28]:0x400_0000"
//     args: "bits[33]:0x1_1748_61ac; bits[41]:0x1e2_8424_1ce0; bits[37]:0x1_e496_add0; bits[42]:0xcd_b6bb_14d1; bits[31]:0x6098_add0; bits[28]:0x98_a9c4"
//     args: "bits[33]:0x1_a2b4_3fd0; bits[41]:0x1b2_fc37_dc27; bits[37]:0xd_5a79_b776; bits[42]:0x155_5555_5555; bits[31]:0x5555_5555; bits[28]:0x1_0000"
//     args: "bits[33]:0x1_f51b_1e44; bits[41]:0xff_ffff_ffff; bits[37]:0xb_d9bc_1872; bits[42]:0x3ff_ffff_ffff; bits[31]:0x5555_5555; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x8000_0000; bits[41]:0xff_ffff_ffff; bits[37]:0xa_aaaa_aaaa; bits[42]:0x1ff_ffff_ffbe; bits[31]:0x4050_1b40; bits[28]:0x2_0000"
//     args: "bits[33]:0x0; bits[41]:0x1ff_ffff_ffff; bits[37]:0x15_fbad_4bec; bits[42]:0x1_0000_0000; bits[31]:0x5908_0094; bits[28]:0x9a3_4b8c"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x142_5c89_f123; bits[37]:0x9_b2a1_ce8b; bits[42]:0x100; bits[31]:0x0; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x1_d293_567b; bits[41]:0xff_ffff_ffff; bits[37]:0x1000; bits[42]:0x1; bits[31]:0x2aaa_aaaa; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x0; bits[41]:0x4_1184; bits[37]:0x3_2100_0006; bits[42]:0x49_0006_00ae; bits[31]:0x7fff_ffff; bits[28]:0x55_d03e"
//     args: "bits[33]:0x555c_2b66; bits[41]:0x155_5555_5555; bits[37]:0x17_d773_dd51; bits[42]:0x0; bits[31]:0x3fff_ffff; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x13b_c95d_ea47; bits[37]:0x1b_c95c_ead6; bits[42]:0x1fc_91bd_5951; bits[31]:0x418a_91fb; bits[28]:0x51f_6e67"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x1f7_57f7_efaa; bits[37]:0xf_ffff_ffff; bits[42]:0x155_5555_5555; bits[31]:0x75c5_9716; bits[28]:0x5c1_8716"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xaa_aaaa_aaaa; bits[37]:0x400; bits[42]:0x151_52e5_5515; bits[31]:0x2aa8_82bb; bits[28]:0x400_0000"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x18f_7179_3d6a; bits[37]:0x9_ea5b_cb7a; bits[42]:0x16b_4b49_3e7f; bits[31]:0x36b6_e7e9; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0x40; bits[41]:0x155_5555_5555; bits[37]:0x15_5565_5557; bits[42]:0x1c1_6462_8540; bits[31]:0x2109_17c6; bits[28]:0x0"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0xaa_aaaa_aaaa; bits[37]:0x1_ee3c_aeaa; bits[42]:0x2d_9795_d517; bits[31]:0x2aaa_aaaa; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x2_0000; bits[41]:0x1_e240_9097; bits[37]:0x120_841a; bits[42]:0x10_5c52_a370; bits[31]:0x2aa5_1281; bits[28]:0xc1a_b731"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x155_5555_5555; bits[37]:0x7_fffd_f7ee; bits[42]:0x0; bits[31]:0x1777_da13; bits[28]:0x777_fe17"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x7d_ddfc_ffb7; bits[37]:0x1f_ffff_ffff; bits[42]:0x2ff_dfff_fdf4; bits[31]:0x2aaa_aaaa; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x0; bits[41]:0x15c_4ba8_9f04; bits[37]:0x12_2429_99ec; bits[42]:0x0; bits[31]:0x5023_1224; bits[28]:0x62b_895f"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x1ff_ffff_ffff; bits[37]:0xa_aaaa_aaaa; bits[42]:0x3ff_ffff_ffff; bits[31]:0x7dff_ffff; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0x100_0000; bits[41]:0x1f0_2511_b840; bits[37]:0xa_aaaa_aaaa; bits[42]:0x25e_c460_0970; bits[31]:0x2b1e_2bbc; bits[28]:0x555_5555"
//     args: "bits[33]:0x0; bits[41]:0x68_b30e_a8b9; bits[37]:0x1d_a982_aabe; bits[42]:0x5280_154d; bits[31]:0x300_1000; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x0; bits[41]:0x28_0022_60ff; bits[37]:0xa_99ad_6b4e; bits[42]:0x2aa_aaaa_aaaa; bits[31]:0x19ad_694e; bits[28]:0xdbf_7f5e"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0xdf_782b_2437; bits[37]:0x15_5555_5555; bits[42]:0x22a_aeaa_8ace; bits[31]:0x5555_5555; bits[28]:0xeaf_6e25"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x20_0000_0000; bits[37]:0x5_7cd5_4192; bits[42]:0x2a_aa2a_a0aa; bits[31]:0x7d77_c2be; bits[28]:0xe22_e1fe"
//     args: "bits[33]:0x0; bits[41]:0x8000_025f; bits[37]:0xa; bits[42]:0x34a; bits[31]:0x45e9_7c51; bits[28]:0x9066"
//     args: "bits[33]:0x1; bits[41]:0xc0_3b50_80bc; bits[37]:0x8_1a15_9a18; bits[42]:0x1ff_ffff_ffff; bits[31]:0x3a17_5814; bits[28]:0x20_0011"
//     args: "bits[33]:0x40_0000; bits[41]:0x40_4080_00ff; bits[37]:0x15_5555_5555; bits[42]:0xa_5166_a7f6; bits[31]:0x2aaa_aaaa; bits[28]:0x735_5457"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x9a_f8d7_70cd; bits[37]:0xa_7097_47c0; bits[42]:0x0; bits[31]:0x37ff_ffdf; bits[28]:0x555_5555"
//     args: "bits[33]:0x80; bits[41]:0x800_8179; bits[37]:0x0; bits[42]:0x0; bits[31]:0x0; bits[28]:0x100"
//     args: "bits[33]:0x2958_79b3; bits[41]:0x48_da1d_d292; bits[37]:0x10_0000; bits[42]:0x2aa_aaaa_aaaa; bits[31]:0x2b50_3d9d; bits[28]:0x1ec_3f1d"
//     args: "bits[33]:0x0; bits[41]:0x102_1340_8020; bits[37]:0xf_ffff_ffff; bits[42]:0x155_5555_5555; bits[31]:0xf91_d5c1; bits[28]:0x342_8440"
//     args: "bits[33]:0x0; bits[41]:0x20_0448_10ef; bits[37]:0x44d_18eb; bits[42]:0x2_99a7_1f75; bits[31]:0x147f_3c80; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x1ff_ffff_ffff; bits[37]:0x1f_ffff_ffff; bits[42]:0x3f7_cfff_ffff; bits[31]:0x3fff_ffff; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x0; bits[41]:0xff_ffff_ffff; bits[37]:0x4_6142_4a01; bits[42]:0x1ff_ffff_ffff; bits[31]:0x6cfa_080a; bits[28]:0x2d_0210"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x0; bits[37]:0x15_5555_5555; bits[42]:0x800_0101; bits[31]:0x7fff_ffff; bits[28]:0x0"
//     args: "bits[33]:0x7980_61bb; bits[41]:0x0; bits[37]:0x7_b49d_c5c3; bits[42]:0x2aa_aaaa_aaaa; bits[31]:0x2aaa_aaaa; bits[28]:0xa19_d862"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xaa_28aa_8a45; bits[37]:0x2_0000; bits[42]:0x3ff_ffff_ffff; bits[31]:0x2_08c0; bits[28]:0x555_5555"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x0; bits[37]:0x948_0602; bits[42]:0x28a_3dcc_a121; bits[31]:0x25cc_a121; bits[28]:0x828_972e"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xff_f7cb_ffaa; bits[37]:0x1f_e7c2_fffa; bits[42]:0x1ff_ef97_ff55; bits[31]:0x7f78_1ed1; bits[28]:0x200_0000"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x1ff_ffff_ffff; bits[37]:0xf_ffff_ffff; bits[42]:0x3ff_ffff_ffff; bits[31]:0x2f7f_bfff; bits[28]:0xfff_ffff"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x1ff_ffff_ffff; bits[37]:0x1f_ffff_ffff; bits[42]:0x0; bits[31]:0x2aaa_ba62; bits[28]:0xffd_ffff"
//     args: "bits[33]:0x5863_25a5; bits[41]:0x3c_d32d_c567; bits[37]:0xa_aaaa_aaaa; bits[42]:0x79_a65b_8acf; bits[31]:0x0; bits[28]:0x1000"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x155_5555_5555; bits[37]:0x0; bits[42]:0x1ff_bb3a_7fff; bits[31]:0x3fff_ffff; bits[28]:0x0"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x1b5_bff3_e918; bits[37]:0x1f_ffff_ffff; bits[42]:0x20b_7fe7_1e54; bits[31]:0x1d51_69b8; bits[28]:0x2_0000"
//     args: "bits[33]:0x0; bits[41]:0xff_ffff_ffff; bits[37]:0x9_c2f5_700e; bits[42]:0x13a_5fae_0ac5; bits[31]:0x1e80_84ae; bits[28]:0x7e6_4ea5"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x1ff_ffff_ffaa; bits[37]:0x1b_ffdb_fded; bits[42]:0x396_77ba_5b9f; bits[31]:0x0; bits[28]:0x200"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x1bf_e1ab_eeaa; bits[37]:0x0; bits[42]:0x27f_835f_d955; bits[31]:0x0; bits[28]:0xd1b_7897"
//     args: "bits[33]:0x1_e2d3_9cd1; bits[41]:0x1e2_539c_d1aa; bits[37]:0x400; bits[42]:0x0; bits[31]:0x389e_8db9; bits[28]:0x555_5555"
//     args: "bits[33]:0x0; bits[41]:0x6_18c4_707e; bits[37]:0x8_d8de_3888; bits[42]:0x20_04ff; bits[31]:0x6c0d_74fb; bits[28]:0x9e0_e88e"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xaa_20b9_2a7b; bits[37]:0xa_aa8a_28a5; bits[42]:0x155_5555_5555; bits[31]:0x1541_7752; bits[28]:0x80c_7bf1"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xaa_aaaa_aaaa; bits[37]:0xf_ffff_f7f7; bits[42]:0x2af_24b2_decf; bits[31]:0x233e_5bdf; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x96eb_4786; bits[41]:0x3b_eb41_b474; bits[37]:0x15_5555_5555; bits[42]:0x17_c5c1_787a; bits[31]:0x46a9_ee83; bits[28]:0xf29_a4d4"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0xff_ffff_ffff; bits[37]:0xf_ffff_ffff; bits[42]:0x0; bits[31]:0x3fff_ffff; bits[28]:0xaaa_aaaa"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x4; bits[37]:0xf_ffff_ffff; bits[42]:0x1ff_ffff_ffff; bits[31]:0x494d_5d63; bits[28]:0x88_0005"
//     args: "bits[33]:0xfb01_7e5c; bits[41]:0x1d9_c96e_2cb7; bits[37]:0xe_059a_a512; bits[42]:0x1c0_b370_c25f; bits[31]:0x3fff_ffff; bits[28]:0xd20_9559"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0x1000_0000; bits[37]:0x1f_ffff_ffff; bits[42]:0x0; bits[31]:0x2aaa_aaaa; bits[28]:0x9aa_08a3"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x1_0000_0000; bits[37]:0x17_a240_0813; bits[42]:0x155_5555_5555; bits[31]:0x2240_0a13; bits[28]:0xfff_ffff"
//     args: "bits[33]:0xd767_37a6; bits[41]:0x1ff_ffff_ffff; bits[37]:0xa_aaaa_aaaa; bits[42]:0x155_5555_5555; bits[31]:0x7fff_ffff; bits[28]:0xfef_efff"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xba_aaa9_a2fc; bits[37]:0xb_eaaa_aeb2; bits[42]:0x30_5a7d_6847; bits[31]:0x4558_17ad; bits[28]:0x921_b12d"
//     args: "bits[33]:0xaaaa_aaaa; bits[41]:0xae_6af2_02ae; bits[37]:0x1f_ffff_ffff; bits[42]:0x155_5555_5555; bits[31]:0x23e3_8aba; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0xff_ffff_ffff; bits[37]:0xa_aaaa_aaaa; bits[42]:0x32b_28b5_4294; bits[31]:0x38b5_4694; bits[28]:0xfd1_9200"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x155_5555_5555; bits[37]:0x11_39c7_4f75; bits[42]:0x0; bits[31]:0x3fff_ffff; bits[28]:0x651_1454"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x155_5555_5555; bits[37]:0x15_4565_cd35; bits[42]:0x3b2_a683_aa2a; bits[31]:0x5555_5555; bits[28]:0x3ee_1eec"
//     args: "bits[33]:0x1_ffff_ffff; bits[41]:0x0; bits[37]:0x1e_e7ba_9fec; bits[42]:0x0; bits[31]:0x5555_5555; bits[28]:0xaaa_aaaa"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x191_1454_95aa; bits[37]:0x9_3450_b73a; bits[42]:0x2a7_aa8a_2a1c; bits[31]:0x11f8_9dbe; bits[28]:0x5c4_1728"
//     args: "bits[33]:0x0; bits[41]:0x120_4802_14ef; bits[37]:0x2; bits[42]:0x155_5555_5555; bits[31]:0x2550_6ebd; bits[28]:0xdba_6a7b"
//     args: "bits[33]:0xffff_ffff; bits[41]:0x155_5555_5555; bits[37]:0x1d_5565_5515; bits[42]:0x15f_f1ff_fe00; bits[31]:0x3fff_ffff; bits[28]:0xfff_ffff"
//     args: "bits[33]:0x1_7f62_7108; bits[41]:0x0; bits[37]:0x1f_ffff_ffff; bits[42]:0x12_2230_0272; bits[31]:0x2aaa_aaaa; bits[28]:0x270_0273"
//     args: "bits[33]:0x400; bits[41]:0x1000_0000; bits[37]:0x15_5555_5555; bits[42]:0x20_b922; bits[31]:0x5d74_c09f; bits[28]:0x7ff_ffff"
//     args: "bits[33]:0xffff_ffff; bits[41]:0xff_ffff_cf00; bits[37]:0x13_df57_c775; bits[42]:0x3ff_ffff_f62f; bits[31]:0x3edb_8f00; bits[28]:0x555_5555"
//     args: "bits[33]:0x4e11_1d1d; bits[41]:0x155_5555_5555; bits[37]:0x14_5547_7555; bits[42]:0x2ba_c0ef_8a8c; bits[31]:0x5555_5555; bits[28]:0x555_5555"
//     args: "bits[33]:0x5c4a_c120; bits[41]:0x55_d4d4_05dc; bits[37]:0x0; bits[42]:0x2bc_d482_4320; bits[31]:0x80; bits[28]:0x51a_4321"
//     args: "bits[33]:0x1_5555_5555; bits[41]:0x155_5555_5555; bits[37]:0x15_f514_185e; bits[42]:0x0; bits[31]:0x4020_0821; bits[28]:0x200_0000"
//     args: "bits[33]:0x0; bits[41]:0x110_5206_103b; bits[37]:0x80_012a; bits[42]:0x0; bits[31]:0x3fff_ffff; bits[28]:0x0"
//   }
// }
// END_CONFIG
type x6 = s7;
type x8 = (x6[5], u5, (sN[1887],));
type x22 = x8[2];
fn main(x0: s33, x1: u41, x2: u37, x3: s42, x4: s31, x5: u28) -> (u41, u20, (x6[5], u5, (sN[1887],)), u2) {
  let x7: (x6[5], u5, (sN[1887],)) = match x5 {
    u28:0x7ff_ffff => ([s7:0x7f, s7:0x7f, s7:0b111_1111, s7:0b11_1100, s7:0x55], u5:0x0, (sN[1887]:0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000,)),
    u28:0xfff_ffff | u28:0xaaa_aaaa => ([s7:0x7f, s7:0x7f, s7:0x55, s7:0x4, s7:0x2a], u5:0xa, (sN[1887]:0b101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101,)),
    _ => ([s7:0x7f, s7:0x55, s7:0x3f, s7:0x10, s7:0x0], u5:0x15, (sN[1887]:0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff,)),
  };
  let x9: x8[2] = [x7, x7];
  let x10: (u28, s33, s31) = (x5, x0, x4);
  let x11: s31 = (x4) + (((x2) as s31));
  let x12: s31 = -(x11);
  let x13: s31 = (x11) - (x11);
  let x14: u41 = !(x1);
  let x15: u2 = (x14)[x2+:u2];
  let x16: u56 = (x5) ++ (x5);
  let x17: x8[1] = array_slice(x9, x14, x8[1]:[(x9)[u32:0b0], ...]);
  let x18: u41 = !(x1);
  let x19: u20 = (((x11) as u31))[4:24];
  let x20: bool = (x0) <= (((x1) as s33));
  let x21: s33 = !(x0);
  let x23: x22[2] = [x9, x9];
  let x24: u23 = (((x21) as u33))[:23];
  let x25: u41 = (x18) << (if (x2) >= (u37:0xe) { u37:0xe } else { x2 });
  let x26: s15 = s15:0x3a71;
  (x18, x19, x7, x15)
}
