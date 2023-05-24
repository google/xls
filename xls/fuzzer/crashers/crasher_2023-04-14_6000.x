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
// exception: "// Command \'[\'xls/tools/eval_ir_main\', \'--input_file=args.txt\', \'--use_llvm_jit\', \'sample.ir\', \'--logtostderr\']\' returned non-zero exit status 66."
// issue: "https://github.com/llvm/llvm-project/issues/62226"
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
// }
// inputs {
//   function_args {
//     args: "bits[18]:0x1_5555; bits[15]:0x5555; bits[50]:0x1_5555_aaaa_aaaa; bits[88]:0x55_552a_aaaa_ba80_1080_0400"
//     args: "bits[18]:0x1_ffff; bits[15]:0x0; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x75_f7d5_7f55_3ddd_440f_1755"
//     args: "bits[18]:0x1_ffff; bits[15]:0x71fc; bits[50]:0x3_66eb_6556_519e; bits[88]:0x0"
//     args: "bits[18]:0x20; bits[15]:0x20c2; bits[50]:0x1_5555_5555_5555; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1; bits[15]:0x1001; bits[50]:0x0; bits[88]:0x0"
//     args: "bits[18]:0x3_ffff; bits[15]:0x2aaa; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0xd2_a717_91f9_2b44_280d_411a"
//     args: "bits[18]:0x1_5555; bits[15]:0x5555; bits[50]:0x2_1f0d_fbbf_afa1; bits[88]:0x0"
//     args: "bits[18]:0x0; bits[15]:0x2127; bits[50]:0x2_2d21_df8d_5b15; bits[88]:0x83_4877_e356_c155_5555_5545"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x3fff; bits[50]:0x0; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[15]:0x7dff; bits[50]:0x3_af7b_1a79_c8f3; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x5e9b; bits[50]:0x2_2e98_4145_754f; bits[88]:0x4_0000_0000_0000"
//     args: "bits[18]:0x1000; bits[15]:0x4; bits[50]:0x1_247b_83c8_f852; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x2_fc9e; bits[15]:0x2aaa; bits[50]:0x2_d94a_3380_4a40; bits[88]:0x0"
//     args: "bits[18]:0x1_5555; bits[15]:0x5554; bits[50]:0x1_5555_5555_5555; bits[88]:0x0"
//     args: "bits[18]:0x1_5555; bits[15]:0x5555; bits[50]:0x0; bits[88]:0xaa_aabf_ffff_ffdf_fff7_ffff"
//     args: "bits[18]:0x1_5555; bits[15]:0x55bf; bits[50]:0x1_5575_5df5_ffff; bits[88]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x3_4372; bits[15]:0x2aaa; bits[50]:0x1_1550_0020_0000; bits[88]:0x92_9c90_0004_0042_8404_0000"
//     args: "bits[18]:0x3_bc89; bits[15]:0x6ca1; bits[50]:0x2_3a01_0c40_e20c; bits[88]:0xaa_52aa_98a2_ea4a_8b2e_c322"
//     args: "bits[18]:0xc554; bits[15]:0x236; bits[50]:0x2_1537_7fff_ffff; bits[88]:0xa1_c2a1_e76e_7bd5_2a98_8a01"
//     args: "bits[18]:0x800; bits[15]:0x4; bits[50]:0x1_5555_5555_5555; bits[88]:0x8aa_aa2b_aaaa_eaa3_eaaa"
//     args: "bits[18]:0x0; bits[15]:0x2141; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x200; bits[15]:0x1000; bits[50]:0x400; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x0; bits[15]:0x7fff; bits[50]:0x0; bits[88]:0x40_0118_b855_cec8_0de1_9a99"
//     args: "bits[18]:0x0; bits[15]:0x5555; bits[50]:0x1000_0000_0000; bits[88]:0x15_7577_5555_5555_d555"
//     args: "bits[18]:0x1_e714; bits[15]:0x6230; bits[50]:0x1_3610_1451_6ec1; bits[88]:0x0"
//     args: "bits[18]:0x1_ffff; bits[15]:0x40; bits[50]:0x2a52_4283_a72a; bits[88]:0x35_04b8_d3fc_485b_37a3_b770"
//     args: "bits[18]:0x2_0000; bits[15]:0x20; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1; bits[15]:0x480d; bits[50]:0x1_5555_5555_5555; bits[88]:0x8_0000_0000_0000_0000_0000"
//     args: "bits[18]:0x3_29bd; bits[15]:0x29c1; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x6f_7a61_0c41_f9e3_7d87_1022"
//     args: "bits[18]:0x1_5555; bits[15]:0x5144; bits[50]:0x1_5545_0000_0001; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_5555; bits[15]:0x5efb; bits[50]:0x1_15c4_5755_0915; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x1_5555; bits[15]:0x5555; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x1_5555; bits[15]:0x7fff; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0x6f_ce27_faea_882a_ee8a_aa5b"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7de9; bits[50]:0x8_0000_0000; bits[88]:0x1a_0007_3cb4_5ac5_640c_d748"
//     args: "bits[18]:0x1_5555; bits[15]:0x3fff; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x1_ffff; bits[15]:0x2aaa; bits[50]:0x1_7fff_8ab7_7ef7; bits[88]:0x0"
//     args: "bits[18]:0x1_ffff; bits[15]:0x2aaa; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x6f_ffe2_e055_8f9b_0e7b_a7a2"
//     args: "bits[18]:0x3_ffff; bits[15]:0x5555; bits[50]:0x3_b34f_37d7_8964; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x0; bits[15]:0x2d8f; bits[50]:0x1_4301_3aff_3d5d; bits[88]:0x4b_1e7d_5b3f_dbff_edff_ffff"
//     args: "bits[18]:0x1_2a6e; bits[15]:0x90e; bits[50]:0x3_14d6_e1bd_3e3e; bits[88]:0x0"
//     args: "bits[18]:0x200; bits[15]:0x18; bits[50]:0x3_2416_5f16_6251; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_8d61; bits[15]:0x7fff; bits[50]:0x1_b7f5_9100_022c; bits[88]:0xe5_de35_4a88_4d1f_f73d_5bff"
//     args: "bits[18]:0x9413; bits[15]:0x0; bits[50]:0x0; bits[88]:0x5_0cff_da70_c71f_df92_1d44"
//     args: "bits[18]:0x0; bits[15]:0x7d79; bits[50]:0x1_5555_5555_5555; bits[88]:0x55_029a_e0a3_9a22_9aae_9aaa"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7ffd; bits[50]:0x2_ff84_4bb8_a3e9; bits[88]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x92a7; bits[15]:0x5636; bits[50]:0x2_ba27_02ff_db8a; bits[88]:0xac_68ba_2e8a_aa8a_ab88_aeda"
//     args: "bits[18]:0x1_ffff; bits[15]:0x3fff; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x2a_aaaa_baab_aa92_39ab_93bd"
//     args: "bits[18]:0x8000; bits[15]:0x104; bits[50]:0x2000; bits[88]:0x2c_1bda_9d19_4a14_ce51_6384"
//     args: "bits[18]:0x1_5555; bits[15]:0x2aaa; bits[50]:0x3_5550_0280_0080; bits[88]:0x20_0000"
//     args: "bits[18]:0x1000; bits[15]:0x2; bits[50]:0x15_5555_5545; bits[88]:0x4_46f5_3755_6060_286e_0046"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7fff; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0xa8_a0cc_abae_8adc_1595_fdb5"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7f7d; bits[50]:0x3_7cd9_4a66_4800; bits[88]:0x0"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x2aa9; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[15]:0x5fa7; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0xc7_cfea_aa2e_f822_64ba_a2fa"
//     args: "bits[18]:0x0; bits[15]:0x133; bits[50]:0xe706_46e6_5340; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x1_ffff; bits[15]:0x7fff; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xff_ffff_7fff_ffc2_0040_0100"
//     args: "bits[18]:0x1; bits[15]:0x21; bits[50]:0x1_5555_5555_5555; bits[88]:0x0"
//     args: "bits[18]:0x1_ffff; bits[15]:0x5f97; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0x8_0000_0000_0000_0000"
//     args: "bits[18]:0x1000; bits[15]:0x2aaa; bits[50]:0x1_1030_90d1_6f3f; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x20; bits[15]:0x3fff; bits[50]:0x1_bbbe_5555_0547; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_5555; bits[15]:0x2aaa; bits[50]:0x7459_fbb5_d9f2; bits[88]:0x80"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x22ab; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0x11_56fb_3fbe_ffff_edfd_ffff"
//     args: "bits[18]:0x1_ffff; bits[15]:0x7ff3; bits[50]:0x0; bits[88]:0x55_5555_5555_5555_5555_5555"
//     args: "bits[18]:0x0; bits[15]:0x437c; bits[50]:0x0; bits[88]:0x0"
//     args: "bits[18]:0x20; bits[15]:0x7fff; bits[50]:0x0; bits[88]:0x0"
//     args: "bits[18]:0x1_ffff; bits[15]:0x0; bits[50]:0x0; bits[88]:0x40_807f_f95b_ffff_e7fa_77ff"
//     args: "bits[18]:0x1_5555; bits[15]:0x5fd5; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x3f_8a18_0ea9_8804_a6ae_88a2"
//     args: "bits[18]:0x40; bits[15]:0x2802; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x8000_0000_0000_0000"
//     args: "bits[18]:0x0; bits[15]:0xc04; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x10_0000_0000_0000_0000"
//     args: "bits[18]:0x1_5555; bits[15]:0x800; bits[50]:0x0; bits[88]:0x104_2100_0204_0080_2020"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7fff; bits[50]:0x400; bits[88]:0xdf_fe6c_66dc_97e9_7a41_9697"
//     args: "bits[18]:0x0; bits[15]:0x34b; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x6_3604_0002_2002_0030_0200"
//     args: "bits[18]:0x3_ffff; bits[15]:0x727f; bits[50]:0x3_9b59_1117_5492; bits[88]:0xc6_76ee_57bf_ff77_7ff5_dff6"
//     args: "bits[18]:0xba8e; bits[15]:0x3e4a; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0xda_ceae_73ee_b6eb_cace_4e40"
//     args: "bits[18]:0x1_ffff; bits[15]:0x4f8a; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0xef_815c_bf7f_ac5e_bbe7_f8f6"
//     args: "bits[18]:0x2_ceed; bits[15]:0x5f9f; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x0; bits[15]:0x5555; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xf2_6d7f_bb5f_5e80_1201_3060"
//     args: "bits[18]:0x1_ffff; bits[15]:0x5fcf; bits[50]:0x1_5555_5555_5555; bits[88]:0x2_0000_0000_0000_0000_0000"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x3aaf; bits[50]:0x2_aaaa_5ae7_ffff; bits[88]:0xaa_aa96_bbdf_fffa_aaaa_aaea"
//     args: "bits[18]:0x1_ffff; bits[15]:0x2aaa; bits[50]:0x1_5d78_3103_0010; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x0; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x66_0147_8127_e159_708f_aba9"
//     args: "bits[18]:0x3_ae00; bits[15]:0x5555; bits[50]:0x0; bits[88]:0xa0_c13b_2820_c4e8_e7b9_b58f"
//     args: "bits[18]:0x1_ecf8; bits[15]:0x60bc; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x20; bits[15]:0x0; bits[50]:0x1_5555_5555_5555; bits[88]:0x70_0875_f7ba_af7d_c7bd_feb6"
//     args: "bits[18]:0x0; bits[15]:0x2140; bits[50]:0x1_5444_7a5c_d6ee; bits[88]:0x1095_cdd5_7d75_710d_1555"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x7fff; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0xbd87; bits[15]:0x5d14; bits[50]:0x4000_0000_0000; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_ffff; bits[15]:0x0; bits[50]:0x1_af7e_ce38_00fa; bits[88]:0x100_0000_0000"
//     args: "bits[18]:0x1_5555; bits[15]:0x5555; bits[50]:0x1_5555_7fff_ffff; bits[88]:0x41_5137_ff31_4f9e_26fb_76f9"
//     args: "bits[18]:0x0; bits[15]:0x6d06; bits[50]:0x400_0000; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_5555; bits[15]:0x3fff; bits[50]:0x1_5c55_1000_0000; bits[88]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x2_ed82; bits[15]:0x5d17; bits[50]:0x1_5555_5555_5555; bits[88]:0x45_577d_55f5_5955_5555_5551"
//     args: "bits[18]:0x1_ffff; bits[15]:0x7fff; bits[50]:0x3_7edf_abc9_f221; bits[88]:0x7f_6777_fc7d_f873_bff3_df9f"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7b7f; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0xd9_f310_e130_2cd4_a13c_0a76"
//     args: "bits[18]:0x3_1309; bits[15]:0x1940; bits[50]:0xca04_c524_b68b; bits[88]:0x200_0000_0000_0000"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7c4f; bits[50]:0x1_e243_bcb1_bf5f; bits[88]:0x78_bcb3_2e6b_d9e0_0048_0480"
//     args: "bits[18]:0x1_ffff; bits[15]:0x100; bits[50]:0x17eb_4220_00f0; bits[88]:0x5_7f18_8d09_1897_b669_eec9"
//     args: "bits[18]:0x80; bits[15]:0x4800; bits[50]:0x2090_ab2b_e2aa; bits[88]:0x241f_eeff_d7bf_bfab_ffff"
//     args: "bits[18]:0x3_ffff; bits[15]:0x6ffa; bits[50]:0x0; bits[88]:0xa2_0bbb_71a9_f8fc_cfd2_ff65"
//     args: "bits[18]:0x2_871a; bits[15]:0x0; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x34_3b35_dd3b_8965_776f_681f"
//     args: "bits[18]:0x0; bits[15]:0x2000; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0x68_049f_f3d7_dffd_fdfb_07f7"
//     args: "bits[18]:0x3_ffff; bits[15]:0x2aaa; bits[50]:0x1_5555_7555_5455; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_5555; bits[15]:0x3fff; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0x7b_fdfd_f7fd_ffff_fbdf_ffff"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x2baa; bits[50]:0x2_aaaa_0000_0000; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x2_aaaa; bits[15]:0xbae; bits[50]:0x5575_d1d5_d144; bits[88]:0xa4_dc4d_0489_4a24_19c0_9020"
//     args: "bits[18]:0x1_5555; bits[15]:0x5555; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xa7_d5ec_d0e6_a87a_cdd0_749a"
//     args: "bits[18]:0x3_ffff; bits[15]:0x2aaa; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x96_f9d4_3dbc_6fd5_845f_1639"
//     args: "bits[18]:0x8000; bits[15]:0x0; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0x22fd_ffff_bbef_6ffb_dfb8"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x6aea; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x69_68de_71bd_9c42_28ed_8e6e"
//     args: "bits[18]:0x1_5555; bits[15]:0x31d4; bits[50]:0x0; bits[88]:0x1_4020_b520_4218_0619_2422"
//     args: "bits[18]:0x1_5555; bits[15]:0x6155; bits[50]:0x1_7541_7fff_ffff; bits[88]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1_ffff; bits[15]:0x3fff; bits[50]:0x3_a499_b4e2_5fc5; bits[88]:0x9d_a373_381a_fdbf_d1df_f9b3"
//     args: "bits[18]:0x1_ffff; bits[15]:0x17f6; bits[50]:0x1_ffff_ffff_ffff; bits[88]:0x0"
//     args: "bits[18]:0x1_5555; bits[15]:0x3fff; bits[50]:0x2_9b8d_7f85_c377; bits[88]:0x71_4440_1000_1001_c000_0122"
//     args: "bits[18]:0x10; bits[15]:0x5555; bits[50]:0x0; bits[88]:0xff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x1_5555; bits[15]:0x5745; bits[50]:0x1_7995_b808_6344; bits[88]:0x89_e72d_025e_972c_0ccd_6b5f"
//     args: "bits[18]:0x3_a55d; bits[15]:0x2aaa; bits[50]:0x20_0000; bits[88]:0x4000_0000_0000"
//     args: "bits[18]:0x100; bits[15]:0x301; bits[50]:0x2000_0000; bits[88]:0x0"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x3eb4; bits[50]:0x2_aaaa_aaaa_aaaa; bits[88]:0x7f_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[18]:0x3_ffff; bits[15]:0x7bff; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[18]:0x1_ffff; bits[15]:0x7bfc; bits[50]:0x3_5d7a_df9d_9f9f; bits[88]:0xd7_5eb7_e767_e7ff_ffdf_fffb"
//     args: "bits[18]:0x1_5555; bits[15]:0x23a2; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xb5_e5e9_6c82_e0f5_960b_363e"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x6aca; bits[50]:0x800_0000; bits[88]:0xd5_9400_0000_0000_0040_0000"
//     args: "bits[18]:0x2_b611; bits[15]:0x0; bits[50]:0x2_b611_abaa_ebba; bits[88]:0x52_04fd_7ffb_7fad_9ffb_ff1e"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x0; bits[50]:0x1_5555_5555_5555; bits[88]:0x13b_64c6_d7a9_6266_632a"
//     args: "bits[18]:0x2_aaaa; bits[15]:0x3fff; bits[50]:0x2_4565_108c_2084; bits[88]:0x95_4984_33a8_212a_caaa_ba8e"
//     args: "bits[18]:0x1_5555; bits[15]:0x455d; bits[50]:0x2ae8_aeab_aeea; bits[88]:0x4a_3aab_b8a2_fc9b_bffe_7fff"
//     args: "bits[18]:0x1_5555; bits[15]:0x5555; bits[50]:0x3_ffff_ffff_ffff; bits[88]:0xe3_6612_f38b_53dd_160c_cf74"
//   }
// }
// END_CONFIG
type x10 = s62;
type x37 = s50;
fn main(x0: s18, x1: u15, x2: s50, x3: uN[88]) -> (bool, u2, u21, uN[82], s18) {
  let x4: (u15, s50, u15, s18) = (x1, x2, x1, x0);
  let x16: u6 = match x1 {
    u15:0x20 => u6:0x10,
    u15:0x2aaa | u15:0x7fff => u6:0x15,
    u15:0x0 => u6:0x15,
    _ => u6:0x2a,
  };
  let x17: uN[88] = (x3) ^ (x3);
  let x18: uN[88] = clz(x3);
  let x19: u15 = -(x1);
  let x20: u15 = (x1) | (x1);
  let x21: (uN[88], uN[88]) = (x3, x17);
  let x22: uN[124] = (((x3) ++ (x20)) ++ (x16)) ++ (x1);
  let x23: u21 = match x4 {
    (u15:0b101_0101_0101_0101, s50:0x1_5555_5555_5555, _, s18:0x1_5555) => u21:2097151,
    (u15:0x4ba5, s50:0x3_ffff_ffff_ffff, u15:0, s18:0x3_ffff) => u21:0x80,
    _ => u21:0x16b8,
  };
  let x24: u15 = !(x20);
  let x25: s6 = s6:0x3f;
  let x26: u21 = (((x16) as u21)) - (x23);
  let x27: u21 = -(x26);
  let x28: u21 = ctz(x26);
  let x29: u21 = (((x27) as u21)) | (x28);
  let x30: uN[82] = match x4 {
    (u15:0x0, s50:0x3_ffff_ffff_ffff, u15:16383, s18:0x1_ffff) => uN[82]:0,
    (u15:0x800, s50:0x2_3f7c_09f1_ed7c, u15:0x2aaa, s18:0x800) | (u15:0x3fff, s50:0x1_5555_5555_5555, u15:0x2aaa, s18:0x1_1fe5) => uN[82]:0x1_0000_0000_0000_0000,
    _ => uN[82]:0x0,
  };
  let x31: bool = (x30) >= (((x26) as uN[82]));
  let x32: bool = (x31) ^ (x31);
  let x33: bool = (x4) != (x4);
  let x34: bool = (x32) << (if (x28) >= (u21:0b1101) { u21:0b1101 } else { x28 });
  let x35: u2 = one_hot(x32, bool:0x1);
  let x36: bool = (x31) - (((x20) as bool));
  let x38: x37[1] = [x2];
  let x39: uN[82] = ctz(x30);
  (x36, x35, x28, x39, x0)
}
