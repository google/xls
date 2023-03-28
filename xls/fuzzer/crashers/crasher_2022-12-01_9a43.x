// Copyright 2022 The XLS Authors
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

// Exception:
// Command '['xls/tools/codegen_main', '--output_signature_path=module_sig.textproto', '--delay_model=unit', '--nouse_system_verilog', '--generator=pipeline', '--pipeline_stages=8', '--reset=rst', '--reset_active_low=false', '--reset_asynchronous=false', '--reset_data_path=true', '--flop_inputs=true', '--flop_inputs_kind=zerolatency', 'sample.opt.ir', '--logtostderr']' returned non-zero exit status 1.
// (run dir: /tmp/test_tmpdirz6jqw85c)
// Issue: https://github.com/google/xls/issues/799
//
// options: {"calls_per_sample": 0, "codegen": true, "codegen_args": ["--nouse_system_verilog", "--generator=pipeline", "--pipeline_stages=8", "--reset=rst", "--reset_active_low=false", "--reset_asynchronous=false", "--reset_data_path=true", "--flop_inputs=true", "--flop_inputs_kind=zerolatency"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "proc_ticks": 128, "simulate": true, "simulator": "iverilog", "timeout_seconds": 600, "top_type": 1, "use_jit": true, "use_system_verilog": false}
// ir_channel_names: sample__x6, sample__x15, sample__x35
// args: bits[3]:0x2; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x1_0000, bits[13]:0x840, bits[6]:0x3e)
// args: bits[3]:0x0; bits[47]:0x8_0000; (bits[43]:0x145_1d14_6135, bits[13]:0xaaa, bits[6]:0x26)
// args: bits[3]:0x3; bits[47]:0x3279_041f_0425; (bits[43]:0x58_54c1_3385, bits[13]:0x1ff4, bits[6]:0x2a)
// args: bits[3]:0x2; bits[47]:0x2d9f_c573_ee32; (bits[43]:0x234_dc51_5197, bits[13]:0x800, bits[6]:0x1)
// args: bits[3]:0x7; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x19ba, bits[6]:0x3f)
// args: bits[3]:0x5; bits[47]:0x41ee_b7e5_df5f; (bits[43]:0x391_8de6_10b5, bits[13]:0x0, bits[6]:0x29)
// args: bits[3]:0x5; bits[47]:0x5000_0000_1500; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x1150, bits[6]:0x2b)
// args: bits[3]:0x2; bits[47]:0x4000_0000; (bits[43]:0x4000_0000, bits[13]:0x1555, bits[6]:0x20)
// args: bits[3]:0x2; bits[47]:0x0; (bits[43]:0x4f5_bdfb_7cff, bits[13]:0x1862, bits[6]:0x0)
// args: bits[3]:0x1; bits[47]:0x59fa_8f33_ebb6; (bits[43]:0x7f_ffff_ffff, bits[13]:0x2f5, bits[6]:0x1b)
// args: bits[3]:0x0; bits[47]:0x57df_cd8e_7a5d; (bits[43]:0x41d_9825_760d, bits[13]:0x1555, bits[6]:0x30)
// args: bits[3]:0x7; bits[47]:0x61cb_1f6d_4a24; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x1555, bits[6]:0x2a)
// args: bits[3]:0x7; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x0, bits[13]:0x631, bits[6]:0x3a)
// args: bits[3]:0x7; bits[47]:0x78aa_f1ba_aeab; (bits[43]:0x7ff_ffff_ffff, bits[13]:0xba5, bits[6]:0x1)
// args: bits[3]:0x2; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0x1fbe, bits[6]:0x35)
// args: bits[3]:0x0; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x3d8_d99b_357e, bits[13]:0x6ae, bits[6]:0x2a)
// args: bits[3]:0x3; bits[47]:0x3df_7b41_593f; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x1f82, bits[6]:0x3f)
// args: bits[3]:0x5; bits[47]:0x5555_5555_5555; (bits[43]:0x558_8536_31b5, bits[13]:0xfff, bits[6]:0x15)
// args: bits[3]:0x4; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x30b_ea35_0b53, bits[13]:0x800, bits[6]:0x35)
// args: bits[3]:0x0; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x63d_6c9d_fafb, bits[13]:0x1fff, bits[6]:0x2c)
// args: bits[3]:0x0; bits[47]:0x4424_334d_e74f; (bits[43]:0x13e_a84b_6f4f, bits[13]:0x1f05, bits[6]:0xf)
// args: bits[3]:0x0; bits[47]:0x2c1_4200_16c3; (bits[43]:0x555_5555_5555, bits[13]:0x1fff, bits[6]:0x3a)
// args: bits[3]:0x5; bits[47]:0x5aaa_aaaa_aaaa; (bits[43]:0x28f_e72f_a3b8, bits[13]:0x18ee, bits[6]:0x2a)
// args: bits[3]:0x2; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x20_0000_0000, bits[13]:0x881, bits[6]:0x15)
// args: bits[3]:0x1; bits[47]:0x3255_5508_56f4; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x16e4, bits[6]:0x15)
// args: bits[3]:0x3; bits[47]:0x40; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0xfff, bits[6]:0x3f)
// args: bits[3]:0x3; bits[47]:0x0; (bits[43]:0x0, bits[13]:0x88, bits[6]:0x0)
// args: bits[3]:0x4; bits[47]:0x2dab_5fbf_fff9; (bits[43]:0x4f7_7fff_ffe7, bits[13]:0x1fc9, bits[6]:0x3f)
// args: bits[3]:0x2; bits[47]:0x5b2c_b84a_9a20; (bits[43]:0x211_2283_83e0, bits[13]:0xfff, bits[6]:0x15)
// args: bits[3]:0x7; bits[47]:0x7070_71c2_4e38; (bits[43]:0x6dd_1e48_172e, bits[13]:0xc12, bits[6]:0x3a)
// args: bits[3]:0x5; bits[47]:0x5000_0040_1001; (bits[43]:0x500_1821_0002, bits[13]:0x0, bits[6]:0x33)
// args: bits[3]:0x3; bits[47]:0x7fff_ffff_ffff; (bits[43]:0x7ff_ffbf_ffff, bits[13]:0x1096, bits[6]:0x12)
// args: bits[3]:0x5; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x0, bits[13]:0x1d94, bits[6]:0x2a)
// args: bits[3]:0x2; bits[47]:0x40_0000; (bits[43]:0x0, bits[13]:0x1fff, bits[6]:0x10)
// args: bits[3]:0x2; bits[47]:0x2fae_c8aa_beaa; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0x1dcf, bits[6]:0x0)
// args: bits[3]:0x1; bits[47]:0x7fff_ffff_ffff; (bits[43]:0x55a_10f1_efeb, bits[13]:0x1fff, bits[6]:0x23)
// args: bits[3]:0x5; bits[47]:0x100; (bits[43]:0x800_0000, bits[13]:0xfff, bits[6]:0x1f)
// args: bits[3]:0x1; bits[47]:0x7fff_ffff_ffff; (bits[43]:0x0, bits[13]:0x1fbf, bits[6]:0x3f)
// args: bits[3]:0x2; bits[47]:0x35e_7f7d_79f5; (bits[43]:0x326_8804_6451, bits[13]:0x1a5e, bits[6]:0x3f)
// args: bits[3]:0x2; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x154_1d49_9e52, bits[13]:0xaaa, bits[6]:0x2a)
// args: bits[3]:0x1; bits[47]:0x39f5_c768_34d1; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0xfff, bits[6]:0x4)
// args: bits[3]:0x3; bits[47]:0x5555_5555_5555; (bits[43]:0x555_5457_5755, bits[13]:0x1fff, bits[6]:0x1f)
// args: bits[3]:0x3; bits[47]:0x0; (bits[43]:0x0, bits[13]:0xfff, bits[6]:0x2)
// args: bits[3]:0x5; bits[47]:0x5a3a_22bf_eaa9; (bits[43]:0x5c8_d0c6_d941, bits[13]:0x1bb4, bits[6]:0x4)
// args: bits[3]:0x0; bits[47]:0x8_2100_0088; (bits[43]:0x15f_f5af_eeaf, bits[13]:0x1fff, bits[6]:0xf)
// args: bits[3]:0x6; bits[47]:0x6105_5d55_1d54; (bits[43]:0x8000, bits[13]:0x0, bits[6]:0x8)
// args: bits[3]:0x7; bits[47]:0x5218_11c8_d608; (bits[43]:0x77d_ff7d_ff7f, bits[13]:0x0, bits[6]:0x3f)
// args: bits[3]:0x5; bits[47]:0x5555_5555_5555; (bits[43]:0x5ee_d9e2_ef16, bits[13]:0x135, bits[6]:0xc)
// args: bits[3]:0x2; bits[47]:0x339f_bed9_ffe3; (bits[43]:0x7ff_ffff_ffff, bits[13]:0xbff, bits[6]:0x15)
// args: bits[3]:0x5; bits[47]:0x18a8_f39f_c3a6; (bits[43]:0xa8_f39f_c386, bits[13]:0xfff, bits[6]:0x39)
// args: bits[3]:0x2; bits[47]:0x8; (bits[43]:0x93_1b82_60cc, bits[13]:0x1ae6, bits[6]:0x18)
// args: bits[3]:0x1; bits[47]:0x0; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0x440, bits[6]:0x1f)
// args: bits[3]:0x7; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x5d4_54d5_a485, bits[13]:0x822, bits[6]:0x1f)
// args: bits[3]:0x7; bits[47]:0x7c48_1048_0806; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0xaaa, bits[6]:0x1f)
// args: bits[3]:0x2; bits[47]:0x2000_0084_0004; (bits[43]:0x0, bits[13]:0x1fff, bits[6]:0x1f)
// args: bits[3]:0x4; bits[47]:0x2000_0000; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0x782, bits[6]:0x10)
// args: bits[3]:0x5; bits[47]:0x47ff_ff7f_7fff; (bits[43]:0x5ff_ff7d_7ffd, bits[13]:0x1c55, bits[6]:0x1f)
// args: bits[3]:0x2; bits[47]:0x5170_0c2b_428c; (bits[43]:0x4b5_6555_46dd, bits[13]:0x1555, bits[6]:0x10)
// args: bits[3]:0x3; bits[47]:0x5555_5555_5555; (bits[43]:0x310_6010_0004, bits[13]:0x1fff, bits[6]:0x2a)
// args: bits[3]:0x4; bits[47]:0x4190_4000_0010; (bits[43]:0x180_4848_0018, bits[13]:0x310, bits[6]:0x1f)
// args: bits[3]:0x2; bits[47]:0x3110_9bda_6fd1; (bits[43]:0x101_9bdd_6edf, bits[13]:0x1555, bits[6]:0x0)
// args: bits[3]:0x0; bits[47]:0x402a_a02e_6bab; (bits[43]:0x30a_a82e_4ea3, bits[13]:0x16e0, bits[6]:0x2b)
// args: bits[3]:0x5; bits[47]:0x580e_0400_2b0a; (bits[43]:0x22_5193_656c, bits[13]:0x1555, bits[6]:0x15)
// args: bits[3]:0x4; bits[47]:0x6d74_d8ce_5f6f; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0x1555, bits[6]:0x2f)
// args: bits[3]:0x1; bits[47]:0x1fef_ffff_ffff; (bits[43]:0x5b7_8eeb_25ae, bits[13]:0x1ded, bits[6]:0x1)
// args: bits[3]:0x7; bits[47]:0xa76_f5b0_8808; (bits[43]:0x555_5555_5555, bits[13]:0x0, bits[6]:0x20)
// args: bits[3]:0x7; bits[47]:0x517e_1197_1d39; (bits[43]:0x17e_1197_1d28, bits[13]:0x0, bits[6]:0x1f)
// args: bits[3]:0x2; bits[47]:0x3f7e_7801_467a; (bits[43]:0x3a8_de44_c058, bits[13]:0x1a42, bits[6]:0x18)
// args: bits[3]:0x5; bits[47]:0x5555_5555_5555; (bits[43]:0x544_55d4_55d5, bits[13]:0x1d5d, bits[6]:0x15)
// args: bits[3]:0x7; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x764_a2b0_827a, bits[13]:0x1ab8, bits[6]:0x3f)
// args: bits[3]:0x4; bits[47]:0x4a2a_8aaa_aaaa; (bits[43]:0x2a2_8aa2_aa8b, bits[13]:0x1aab, bits[6]:0x22)
// args: bits[3]:0x2; bits[47]:0x7f20_1d7e_da05; (bits[43]:0x630_31ff_7205, bits[13]:0xfee, bits[6]:0x1d)
// args: bits[3]:0x2; bits[47]:0x244b_9200_00e8; (bits[43]:0x410_8358_02f8, bits[13]:0x959, bits[6]:0x28)
// args: bits[3]:0x1; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x0, bits[13]:0x1ebe, bits[6]:0x2d)
// args: bits[3]:0x7; bits[47]:0x7fff_ffff_ffff; (bits[43]:0x400_0000_0000, bits[13]:0x19eb, bits[6]:0x3c)
// args: bits[3]:0x3; bits[47]:0x3048_4171_112e; (bits[43]:0x248_43a9_a103, bits[13]:0x1321, bits[6]:0x1f)
// args: bits[3]:0x2; bits[47]:0x0; (bits[43]:0x6b9_d69a_9c04, bits[13]:0xbfd, bits[6]:0x28)
// args: bits[3]:0x1; bits[47]:0x5cd1_9b95_af84; (bits[43]:0x697_a37f_d2cc, bits[13]:0x1729, bits[6]:0x28)
// args: bits[3]:0x2; bits[47]:0x22aa_2aaa_a2ae; (bits[43]:0xaa_2eab_22b4, bits[13]:0x0, bits[6]:0x6)
// args: bits[3]:0x0; bits[47]:0x43cb_6944_7768; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x0, bits[6]:0x2a)
// args: bits[3]:0x2; bits[47]:0x7fff_ffff_ffff; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x1eff, bits[6]:0x20)
// args: bits[3]:0x3; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x2ea_8aec_aeea, bits[13]:0x0, bits[6]:0x0)
// args: bits[3]:0x0; bits[47]:0xf90_9c01_1740; (bits[43]:0x7dc_7d98_1ed1, bits[13]:0xfff, bits[6]:0x2a)
// args: bits[3]:0x7; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x7ff_ffff_fdbd, bits[13]:0x1fff, bits[6]:0xe)
// args: bits[3]:0x0; bits[47]:0x800_0000_0102; (bits[43]:0x17_0800_82a9, bits[13]:0xaaa, bits[6]:0x6)
// args: bits[3]:0x1; bits[47]:0x5555_5555_5555; (bits[43]:0x725_79d0_7650, bits[13]:0xfff, bits[6]:0x15)
// args: bits[3]:0x7; bits[47]:0x5555_5555_5555; (bits[43]:0x0, bits[13]:0x915, bits[6]:0x3a)
// args: bits[3]:0x5; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x4a0, bits[6]:0x2a)
// args: bits[3]:0x0; bits[47]:0x7fff_ffff_ffff; (bits[43]:0x7ff_ff7d_ffff, bits[13]:0x255, bits[6]:0x1f)
// args: bits[3]:0x7; bits[47]:0x562e_70ff_39a7; (bits[43]:0x787_707b_91e2, bits[13]:0x198c, bits[6]:0x37)
// args: bits[3]:0x3; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x7ff_ffff_ffff, bits[13]:0x31f, bits[6]:0x1e)
// args: bits[3]:0x0; bits[47]:0x40c9_d70d_1e1a; (bits[43]:0x2c9_d70d_1f1a, bits[13]:0x0, bits[6]:0x1a)
// args: bits[3]:0x6; bits[47]:0x0; (bits[43]:0x20_0000_0000, bits[13]:0x1108, bits[6]:0x15)
// args: bits[3]:0x7; bits[47]:0x0; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0x400, bits[6]:0x1f)
// args: bits[3]:0x2; bits[47]:0x20; (bits[43]:0x43b_eaad_f5ef, bits[13]:0x1fff, bits[6]:0x2)
// args: bits[3]:0x2; bits[47]:0x2; (bits[43]:0x65d_fdf3_ebfe, bits[13]:0x640, bits[6]:0x17)
// args: bits[3]:0x5; bits[47]:0x7fe7_7fbe_fdff; (bits[43]:0x557_fffb_ffff, bits[13]:0x11eb, bits[6]:0x28)
// args: bits[3]:0x5; bits[47]:0x76bf_df7b_f5f4; (bits[43]:0x555_5555_5555, bits[13]:0x12c5, bits[6]:0x2a)
// args: bits[3]:0x7; bits[47]:0x5086_8406_a440; (bits[43]:0x7ff_ffff_ffff, bits[13]:0x1d55, bits[6]:0x1)
// args: bits[3]:0x3; bits[47]:0x3421_2080_0800; (bits[43]:0x555_5555_5555, bits[13]:0x420, bits[6]:0x18)
// args: bits[3]:0x0; bits[47]:0x2aaa_cdef_e889; (bits[43]:0x100_0000, bits[13]:0x980, bits[6]:0x9)
// args: bits[3]:0x3; bits[47]:0x7aaa_aab8_aaaa; (bits[43]:0x3ff_ffff_ffff, bits[13]:0xaaa, bits[6]:0x2d)
// args: bits[3]:0x7; bits[47]:0x5992_343b_b51d; (bits[43]:0x3fc_3f39_4e9a, bits[13]:0x1c4b, bits[6]:0x2a)
// args: bits[3]:0x2; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x3ab_aaaa_ba66, bits[13]:0x900, bits[6]:0x2a)
// args: bits[3]:0x2; bits[47]:0x8; (bits[43]:0x761_9ec0_5e44, bits[13]:0x14ec, bits[6]:0x3f)
// args: bits[3]:0x7; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x51_6965_d819, bits[13]:0xfff, bits[6]:0x1f)
// args: bits[3]:0x2; bits[47]:0x2810_1000_0400; (bits[43]:0x2ff_77ff_ffff, bits[13]:0xaaa, bits[6]:0x2a)
// args: bits[3]:0x7; bits[47]:0x7d0b_2f14_aa75; (bits[43]:0x10d_7f00_6251, bits[13]:0x0, bits[6]:0x1f)
// args: bits[3]:0x3; bits[47]:0x7155_d5d5_5515; (bits[43]:0x2aa_aaaa_aaaa, bits[13]:0xfff, bits[6]:0x1a)
// args: bits[3]:0x0; bits[47]:0x8_0000_0000; (bits[43]:0x7b4_a049_8182, bits[13]:0x16aa, bits[6]:0x2a)
// args: bits[3]:0x6; bits[47]:0x2aaa_8aae_2eaa; (bits[43]:0x675_1555_5715, bits[13]:0xe8e, bits[6]:0x8)
// args: bits[3]:0x1; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x80_5786_48ca, bits[13]:0x2b9, bits[6]:0xa)
// args: bits[3]:0x7; bits[47]:0x7fff_ff3f_fbff; (bits[43]:0x7fb_ff3d_fbbf, bits[13]:0x179c, bits[6]:0x1b)
// args: bits[3]:0x7; bits[47]:0x7800_0048_30d0; (bits[43]:0x722_e2af_4aaa, bits[13]:0xaaa, bits[6]:0x0)
// args: bits[3]:0x3; bits[47]:0x2aaa_aaaa_aaaa; (bits[43]:0x8, bits[13]:0xc72, bits[6]:0x32)
// args: bits[3]:0x7; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x63c_f7ed_fafe, bits[13]:0x81d, bits[6]:0x3f)
// args: bits[3]:0x7; bits[47]:0x7fff_ffff_ffff; (bits[43]:0x755_7555_5455, bits[13]:0x1fff, bits[6]:0x2a)
// args: bits[3]:0x2; bits[47]:0xf8d_313b_e668; (bits[43]:0x400_0000, bits[13]:0x1555, bits[6]:0x20)
// args: bits[3]:0x5; bits[47]:0x4f2a_e82a_3e18; (bits[43]:0x638_6cab_ba18, bits[13]:0x0, bits[6]:0x15)
// args: bits[3]:0x2; bits[47]:0x1_0000_0000; (bits[43]:0x3a8_0e91_0fe9, bits[13]:0x800, bits[6]:0x20)
// args: bits[3]:0x1; bits[47]:0x1442_0292_0941; (bits[43]:0x6d0_670b_c0c1, bits[13]:0x945, bits[6]:0x1f)
// args: bits[3]:0x0; bits[47]:0x5ff_f7f6_fffd; (bits[43]:0x555_5555_5555, bits[13]:0x1ffd, bits[6]:0x2)
// args: bits[3]:0x3; bits[47]:0x400_0000_0000; (bits[43]:0x400_0000_0000, bits[13]:0xaaa, bits[6]:0x36)
// args: bits[3]:0x3; bits[47]:0x7f0b_1602_7d4b; (bits[43]:0x3ff_ffff_ffff, bits[13]:0x1555, bits[6]:0x1d)
// args: bits[3]:0x5; bits[47]:0x7fbf_dff5_de76; (bits[43]:0x1ba_bfb5_f77b, bits[13]:0x16b4, bits[6]:0x17)
// args: bits[3]:0x5; bits[47]:0x5555_5555_5555; (bits[43]:0x45e_7fbf_df3f, bits[13]:0x1555, bits[6]:0x1)
// args: bits[3]:0x5; bits[47]:0x56c_16d0_2f84; (bits[43]:0x7ff_ffff_ffff, bits[13]:0xfa8, bits[6]:0x26)
// args: bits[3]:0x2; bits[47]:0x3fff_ffff_ffff; (bits[43]:0x7ef_8e5f_bfff, bits[13]:0xaaa, bits[6]:0x4)
type x25 = u53;
proc main {
  x6: chan<u3> in;
  x15: chan<u47> in;
  x35: chan<(u43, u13, u6)> in;
  config(x6: chan<u3> in, x15: chan<u47> in, x35: chan<(u43, u13, u6)> in) {
    (x6, x15, x35)
  }
  init {
    bool:1
  }
  next(x0: token, x1: bool) {
    let x2: bool = (x1)[0+:bool];
    let x3: token = join(x0);
    let x4: bool = (x1)[:];
    let x5: bool = (x2) >= (x2);
    let x7: (token, u3, bool) = recv_non_blocking(x3, x6, u3:0);
    let x8: token = x7.0;
    let x9: u3 = x7.1;
    let x10: bool = x7.2;
    let x11: bool = (x1) / (bool:0x0);
    let x12: token = join(x3, x8);
    let x13: bool = rev(x2);
    let x14: bool = (x13) * (((x9) as bool));
    let x16: (token, u47) = recv_if(x3, x15, x14, u47:0);
    let x17: token = x16.0;
    let x18: u47 = x16.1;
    let x19: bool = x7.2;
    let x20: bool = (x10) << (if (x9) >= (u3:2) { u3:2 } else { x9 });
    let x21: bool = for (i, x) in u4:0..u4:4 {
      x
    }(x11);
    let x30: (u3, u3) = (x9, x9);
    let x31: bool = (((x21) as bool)) | (x19);
    let x32: bool = !(x14);
    let x33: bool = !(x14);
    let x34: bool = (x5) >= (((x2) as bool));
    let x36: (token, (u43, u13, u6)) = recv(x8, x35);
    let x37: token = x36.0;
    let x38: (u43, u13, u6) = x36.1;
    x10
  }
}
