// Copyright 2021 The XLS Authors
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
// Result miscompare for sample 0:
// args: bits[49]:0x0; bits[25]:0x0; bits[23]:0x40_0000; bits[43]:0x555_5555_5555; bits[10]:0x71
// evaluated opt IR (JIT), evaluated opt IR (interpreter), simulated =
//    (bits[26]:0x1f3_cf6e, bits[32]:0x0)
// evaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[26]:0x1f3_cf6e, bits[32]:0x800)
// Issue: https://github.com/google/xls/issues/423
//
// options: {"codegen": true, "codegen_args": ["--use_system_verilog", "--generator=pipeline", "--pipeline_stages=5"], "convert_to_ir": true, "input_is_dslx": true, "optimize_ir": true, "simulate": false, "simulator": null, "use_jit": true, "use_system_verilog": true}
// args: bits[49]:0x0; bits[25]:0x0; bits[23]:0x40_0000; bits[43]:0x555_5555_5555; bits[10]:0x71
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x1bf_ffff; bits[23]:0x7f_ffff; bits[43]:0x6f9_f9a7_7eab; bits[10]:0x176
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x0; bits[23]:0x73_4019; bits[43]:0x0; bits[10]:0x299
// args: bits[49]:0xd4ca_81e1_66b9; bits[25]:0xaa_aaaa; bits[23]:0x10; bits[43]:0x508_a036_bfe5; bits[10]:0x0
// args: bits[49]:0x100_0000; bits[25]:0x1ff_ffff; bits[23]:0x7f_b10d; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x1a
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0xaa_aaaa; bits[23]:0x7f_ffff; bits[43]:0x71d_f553_35c1; bits[10]:0x2
// args: bits[49]:0x8000; bits[25]:0x1f9_caa6; bits[23]:0x3f_ffff; bits[43]:0x4d3_fa65_4967; bits[10]:0x2
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x136_f40b; bits[23]:0x55_5555; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x2aa
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0xe7_76f6; bits[23]:0x55_5555; bits[43]:0x20_0000_0000; bits[10]:0x3ff
// args: bits[49]:0x3fce_7138_0053; bits[25]:0x138_4053; bits[23]:0x18_2413; bits[43]:0x4e1_014d_ffff; bits[10]:0x23d
// args: bits[49]:0x0; bits[25]:0x100_0100; bits[23]:0x2a_aaaa; bits[43]:0xe3_eb37_aef7; bits[10]:0x10
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x0; bits[23]:0x7f_ffff; bits[43]:0x3ff_ffff_ffff; bits[10]:0x200
// args: bits[49]:0x0; bits[25]:0x155_5555; bits[23]:0x54_573d; bits[43]:0x745_7fdc_c800; bits[10]:0x100
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x155_5555; bits[23]:0x1f_ffff; bits[43]:0x1f6_bff2_c021; bits[10]:0x2de
// args: bits[49]:0xc520_6440_a6dd; bits[25]:0x1ff_ffff; bits[23]:0x2a_aaaa; bits[43]:0x30c_4803_783f; bits[10]:0x166
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xaa_aaaa; bits[23]:0x2_a009; bits[43]:0x0; bits[10]:0x0
// args: bits[49]:0x9b17_315d_bee2; bits[25]:0xff_ffff; bits[23]:0x7f_ffff; bits[43]:0x3ff_ffff_ffff; bits[10]:0x3c0
// args: bits[49]:0x20_0000_0000; bits[25]:0x0; bits[23]:0x1910; bits[43]:0xbb_9185_9544; bits[10]:0x6c
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x1bf_ffff; bits[23]:0x2a_aaaa; bits[43]:0x555_5555_5555; bits[10]:0xef
// args: bits[49]:0x3453_208d_5d38; bits[25]:0x93_d1ee; bits[23]:0x10; bits[43]:0x71_8658_5c3c; bits[10]:0x234
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x17_fae6; bits[23]:0x13_f5ef; bits[43]:0x506_af81_4b14; bits[10]:0x155
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xaa_aaaa; bits[23]:0x77_6a36; bits[43]:0x555_5555_5555; bits[10]:0x2aa
// args: bits[49]:0x0; bits[25]:0x4840; bits[23]:0x0; bits[43]:0x555_5555_5555; bits[10]:0x1ff
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x1ff_ffff; bits[23]:0x79_eb5f; bits[43]:0x12c_3aab_7d0c; bits[10]:0x35f
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xb9_2bb2; bits[23]:0x3a_2cd1; bits[43]:0x4b5_ee7d_b8be; bits[10]:0xd1
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xff_ffff; bits[23]:0x2a_aaaa; bits[43]:0x5e9_a9ec_a243; bits[10]:0x2b8
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x0; bits[23]:0x7_89cc; bits[43]:0xdf_6430_92c7; bits[10]:0x4
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x155_5555; bits[23]:0x55_5455; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x2a8
// args: bits[49]:0x40_0000; bits[25]:0x14a_4020; bits[23]:0xa_406d; bits[43]:0x10_0000_0000; bits[10]:0x20
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x155_5555; bits[23]:0x17_1d54; bits[43]:0x3c6_1457_1c2a; bits[10]:0x0
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x0; bits[23]:0x6c_d0cb; bits[43]:0x3ff_ffff_ffff; bits[10]:0x1ef
// args: bits[49]:0x1000_0000_0000; bits[25]:0x0; bits[23]:0x4_008c; bits[43]:0x0; bits[10]:0x1de
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x1ff_ffff; bits[23]:0x3c_ebaa; bits[43]:0x555_5555_5555; bits[10]:0x0
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x1ff_ffff; bits[23]:0x2a_aaaa; bits[43]:0x800; bits[10]:0x0
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x1df_fcbf; bits[23]:0x46_d47f; bits[43]:0x71f_f24d_4cf9; bits[10]:0xff
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x1ff_ffff; bits[23]:0x5f_bffd; bits[43]:0x3df_2af6_dfdf; bits[10]:0x3ff
// args: bits[49]:0x1_651f_f30a_bee5; bits[25]:0x10e_bff4; bits[23]:0xe_bff4; bits[43]:0x20; bits[10]:0x395
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x153_5ceb; bits[23]:0x7f_ffff; bits[43]:0x14f_77bb_ffbf; bits[10]:0x1bd
// args: bits[49]:0x4000_0000_0000; bits[25]:0x16d_2090; bits[23]:0x37_1e9a; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x29a
// args: bits[49]:0x1000; bits[25]:0xaa_aaaa; bits[23]:0x40; bits[43]:0x3ff_ffff_ffff; bits[10]:0x3ff
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0xfe_6942; bits[23]:0x4e_1fbf; bits[43]:0x100; bits[10]:0x3ff
// args: bits[49]:0x30a7_55db_981b; bits[25]:0x155_5555; bits[23]:0x3f_ffff; bits[43]:0x3ff_ffff_ffff; bits[10]:0x5a
// args: bits[49]:0xffe5_7a8a_bae3; bits[25]:0x19a_fa62; bits[23]:0x2_f0e3; bits[43]:0x2f_2c32_a3d3; bits[10]:0x4
// args: bits[49]:0x0; bits[25]:0x126_a8c8; bits[23]:0x0; bits[43]:0x7ff_ffff_ffff; bits[10]:0x3ff
// args: bits[49]:0x10; bits[25]:0xff_ffff; bits[23]:0x7f_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x1ff
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x18c_ce0b; bits[23]:0x54_7415; bits[43]:0xff_8714_155d; bits[10]:0x2b1
// args: bits[49]:0x0; bits[25]:0x155_5555; bits[23]:0x1000; bits[43]:0x1_1004_f7f4; bits[10]:0x8
// args: bits[49]:0x0; bits[25]:0xaa_aaaa; bits[23]:0x2a_aaaa; bits[43]:0x7ff_ffff_ffff; bits[10]:0x0
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x145_5555; bits[23]:0x15_2415; bits[43]:0x6dd_7756_4234; bits[10]:0x10
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x0; bits[23]:0x41_0812; bits[43]:0x0; bits[10]:0x1
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x1be_a5e7; bits[23]:0x6f_dff7; bits[43]:0x58c_ce12_9ff2; bits[10]:0x361
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x400; bits[23]:0x7f_ffff; bits[43]:0x6ab_2aab_eaaa; bits[10]:0x2ac
// args: bits[49]:0x0; bits[25]:0x50_a048; bits[23]:0x8_4860; bits[43]:0x14a_a863_1001; bits[10]:0x114
// args: bits[49]:0x2000_0000; bits[25]:0xaa_aaaa; bits[23]:0x55_5555; bits[43]:0x28a_2aaa_aaca; bits[10]:0x37a
// args: bits[49]:0x200; bits[25]:0x155_5555; bits[23]:0x7f_ffff; bits[43]:0x4000_0000; bits[10]:0x280
// args: bits[49]:0x1_0fe3_cbb8_d28d; bits[25]:0x192_11af; bits[23]:0x38_d28d; bits[43]:0x7d2_ebb0_ab65; bits[10]:0x8
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x1ff_ffff; bits[23]:0x4000; bits[43]:0x7ff_ffff_ffff; bits[10]:0x3ff
// args: bits[49]:0x0; bits[25]:0x155_5555; bits[23]:0x0; bits[43]:0x555_5555_5555; bits[10]:0x3ff
// args: bits[49]:0x400; bits[25]:0x400; bits[23]:0x2f_b430; bits[43]:0x3ff_ffff_ffff; bits[10]:0x0
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x54_863f; bits[23]:0x7f_ffff; bits[43]:0x555_1555_5554; bits[10]:0x1ff
// args: bits[49]:0x1000_0000; bits[25]:0x1ff_ffff; bits[23]:0x0; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x30a
// args: bits[49]:0x1_8dd1_d18e_8725; bits[25]:0x18e_8727; bits[23]:0x55_5555; bits[43]:0x7ff_ffff_ffff; bits[10]:0x327
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x0; bits[23]:0x2_0000; bits[43]:0x7ff_ffff_ffff; bits[10]:0x3ff
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xbc_a2aa; bits[23]:0x3a_a78a; bits[43]:0x3ae_ea8b_88ac; bits[10]:0x2e6
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0xa8_b54b; bits[23]:0x7a_56d9; bits[43]:0x3ff_ffff_ffff; bits[10]:0x20
// args: bits[49]:0x80_0000; bits[25]:0x2_0000; bits[23]:0x3f_ffff; bits[43]:0x2; bits[10]:0x3ff
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0xaa_aaaa; bits[23]:0x22_f02b; bits[43]:0x12f_80fe_fc18; bits[10]:0xd8
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x155_5555; bits[23]:0x20_0000; bits[43]:0x244_614d_dc3e; bits[10]:0x155
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x5d_5445; bits[23]:0x50_1577; bits[43]:0x58b_5272_0000; bits[10]:0x355
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0xf7_ffbe; bits[23]:0x7f_ffff; bits[43]:0x57a_d9c2_9fae; bits[10]:0x3ff
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x155_5555; bits[23]:0x5d_5459; bits[43]:0x152_ac94_7786; bits[10]:0x155
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x15a_ff1c; bits[23]:0x2; bits[43]:0x7ff_ffff_ffff; bits[10]:0x3ff
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x9a_aa82; bits[23]:0x44_ece7; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x0
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0xf9_8f7f; bits[23]:0x55_5555; bits[43]:0x3e0_bf5c_2090; bits[10]:0x0
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x1ff_6d1a; bits[23]:0x3f_ffff; bits[43]:0x555_5555_5555; bits[10]:0x54
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x1ff_ffff; bits[23]:0x0; bits[43]:0x555_5555_5555; bits[10]:0x2aa
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x0; bits[23]:0x2a_aaaa; bits[43]:0x2a9_3627_aff7; bits[10]:0x3b6
// args: bits[49]:0x2_0000; bits[25]:0x0; bits[23]:0x55_5555; bits[43]:0x3ff_ffff_ffff; bits[10]:0x3ff
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0xaa_aaaa; bits[23]:0x0; bits[43]:0x1400_0000; bits[10]:0x4
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0xc5_5555; bits[23]:0x53_3353; bits[43]:0x793_c68c_4c5d; bits[10]:0x375
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x16d_555d; bits[23]:0x3f_ffff; bits[43]:0x37f_fffc_5f72; bits[10]:0x1df
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x47_80b5; bits[23]:0x41_91b5; bits[43]:0x555_5555_5555; bits[10]:0x55
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x1ff_ffff; bits[23]:0x2a_aaaa; bits[43]:0x6fd_fffc_0001; bits[10]:0x2aa
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x155_5555; bits[23]:0x5d_47f7; bits[43]:0x5c_577d_4c83; bits[10]:0x5b
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x50_dfad; bits[23]:0x7f_ffff; bits[43]:0x7d2_fb6c_30a9; bits[10]:0x2ad
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xff_ffff; bits[23]:0x55_5555; bits[43]:0x7ff_ffff_ffff; bits[10]:0x36b
// args: bits[49]:0x0; bits[25]:0x40_0080; bits[23]:0x2a_aaaa; bits[43]:0x148_326b_0e20; bits[10]:0x1ff
// args: bits[49]:0xf633_21fa_aa38; bits[25]:0xe5_a6bb; bits[23]:0x7a_a238; bits[43]:0x396_9ace_aaa2; bits[10]:0x1ff
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x155_5555; bits[23]:0x45_65d1; bits[43]:0x800; bits[10]:0x0
// args: bits[49]:0x1_e82a_6ef2_aa61; bits[25]:0xf0_aa61; bits[23]:0x7f_ffff; bits[43]:0x641_6af0_9242; bits[10]:0x2aa
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x800; bits[23]:0x25_2a34; bits[43]:0x5c9_c78c_34b0; bits[10]:0x2aa
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x1ff_ffff; bits[23]:0x55_5555; bits[43]:0x7fc_7df9_bc16; bits[10]:0x2aa
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x11c_68d3; bits[23]:0x3c_7457; bits[43]:0x181_4530_a6ff; bits[10]:0x350
// args: bits[49]:0x8_0000; bits[25]:0x192_2f05; bits[23]:0x4a_4080; bits[43]:0x4bc_8943_51eb; bits[10]:0x80
// args: bits[49]:0x8; bits[25]:0xc108; bits[23]:0x18; bits[43]:0x1000_0000; bits[10]:0xc0
// args: bits[49]:0x7b7d_1e0d_e1c6; bits[25]:0x10f_dbc6; bits[23]:0x60_b79d; bits[43]:0x702_5ad3_2960; bits[10]:0x3ff
// args: bits[49]:0x36cc_1a87_ff34; bits[25]:0xe5_771d; bits[23]:0x67_6715; bits[43]:0x27c_5b86_eff4; bits[10]:0x1ff
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x157_555d; bits[23]:0x0; bits[43]:0x3ff_ffff_ffff; bits[10]:0x173
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xaa_aaaa; bits[23]:0x2a_aaaa; bits[43]:0x2a2_b87a_aaaa; bits[10]:0x168
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0xff_ffff; bits[23]:0x55_5555; bits[43]:0x3ff_fff7_ffff; bits[10]:0x3b7
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0xaa_aaaa; bits[23]:0x3f_ffff; bits[43]:0xcb_fb77_75ff; bits[10]:0x159
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x0; bits[23]:0x0; bits[43]:0x40_8002_22b9; bits[10]:0x3ff
// args: bits[49]:0x0; bits[25]:0x158_8210; bits[23]:0x48_8310; bits[43]:0x42b_7c64_1261; bits[10]:0x169
// args: bits[49]:0x2_0000; bits[25]:0x72_cc39; bits[23]:0x7d_3f76; bits[43]:0x280_1012_0004; bits[10]:0x155
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x1f6_cfea; bits[23]:0x10; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x97
// args: bits[49]:0x0; bits[25]:0x40_0000; bits[23]:0x47_e400; bits[43]:0x555_5555_5555; bits[10]:0x108
// args: bits[49]:0x0; bits[25]:0xa_81b0; bits[23]:0x3f_ffff; bits[43]:0x0; bits[10]:0x2ff
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x1ff_ffff; bits[23]:0xf_df56; bits[43]:0xfd_f579_643b; bits[10]:0x6b
// args: bits[49]:0x8_0000_0000; bits[25]:0x30; bits[23]:0x2a_aaaa; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x3ff
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xaa_399a; bits[23]:0x6a_2aaa; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x33b
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x7d_ff7f; bits[23]:0x800; bits[43]:0x0; bits[10]:0x4
// args: bits[49]:0x8_0000_0000; bits[25]:0x14_8142; bits[23]:0x7f_ffff; bits[43]:0x5f7_fef2_c700; bits[10]:0x19a
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x154_5555; bits[23]:0x5c_77e9; bits[43]:0x7b8_1cd6_d2ed; bits[10]:0x1d5
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x157_5754; bits[23]:0x51_5075; bits[43]:0x10_0000_0000; bits[10]:0x155
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0xff_ffff; bits[23]:0x56_6915; bits[43]:0x7ff_ffff_ffff; bits[10]:0x3ff
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x1be_f6fd; bits[23]:0x7e_b6fd; bits[43]:0x555_5555_5555; bits[10]:0x27f
// args: bits[49]:0x0; bits[25]:0x0; bits[23]:0x0; bits[43]:0x121_0200_5800; bits[10]:0x1ff
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x8b_aaaa; bits[23]:0x7a_aa0a; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x3ff
// args: bits[49]:0x8d3b_5d2f_831e; bits[25]:0x10e_83a6; bits[23]:0x20_0000; bits[43]:0x2aa_aaaa_aaaa; bits[10]:0x1ff
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x95_f156; bits[23]:0x14_f9d5; bits[43]:0x3ff_7cee_c2ef; bits[10]:0x103
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0xff_ffff; bits[23]:0xb_f257; bits[43]:0x1e7_1e9e_c3da; bits[10]:0x3da
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x155_7545; bits[23]:0x55_5d55; bits[43]:0x7ff_ffff_ffff; bits[10]:0x155
// args: bits[49]:0x1_8dc8_9614_c642; bits[25]:0x14_e642; bits[23]:0x7f_ffff; bits[43]:0x7c_9ebe_9f82; bits[10]:0x243
// args: bits[49]:0xffff_ffff_ffff; bits[25]:0x1ff_cf7a; bits[23]:0x67_8f0e; bits[43]:0x37b_9cf0_6a0c; bits[10]:0x372
// args: bits[49]:0x1_5555_5555_5555; bits[25]:0x157_d755; bits[23]:0x8_0000; bits[43]:0x55e_7851_87fe; bits[10]:0x3fe
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x8_0000; bits[23]:0x73_f580; bits[43]:0x73f_5907_67ff; bits[10]:0x3f2
// args: bits[49]:0xaaaa_aaaa_aaaa; bits[25]:0x127_af05; bits[23]:0x55_5555; bits[43]:0x5aa_4931_cbfa; bits[10]:0x155
// args: bits[49]:0x1_ffff_ffff_ffff; bits[25]:0x1ff_ffff; bits[23]:0x3d_efb8; bits[43]:0x73a_da7d_fcec; bits[10]:0xbc
type x16 = u32;
fn main(x0: u49, x1: s25, x2: u23, x3: u43, x4: s10) -> (s26, u32) {
  let x5: u23 = !(x2);
  let x6: u10 = (x4)[x0+:u10];
  let x7: uN[72] = (x2) ++ (x0);
  let x8: u32 = u32:0x800;
  let x9: s7 = s7:0x3f;
  let x10: u32 = -(x8);
  let x11: uN[138] = ((((x3) ++ (x8)) ++ (x6)) ++ (x6)) ++ (x3);
  let x12: s26 = s26:0x1f3_cf6e;
  let x13: u10 = -(x6);
  let x14: s10 = (((x8) as s10)) ^ (x4);
  let x15: u32 = rev(x10);
  let x17: x16[2] = [x10, x8];
  let x18: uN[335] = ((((x11) ++ (x7)) ++ (x6)) ++ (x3)) ++ (x7);
  let x19: u2 = (x6)[8:];
  let x20: u9 = (x10)[0+:u9];
  let x21: u3 = one_hot(x19, bool:true);
  let x22: uN[335] = -(x18);
  let x23: u2 = (((x5) as u2)) | (x19);
  let x24: x16 = (x17)[(u10:0) if ((x6) >= (u10:0)) else (x6)];
  let x25: u11 = (x10)[17:28];
  let x26: uN[138] = (x11) << ((u3:0) if ((x21) >= (u3:0)) else (x21));
  let x27: u32 = one_hot_sel(x21, [x15, x10, x8]);
  let x28: u19 = (x11)[119+:u19];
  let x29: s26 = one_hot_sel(x19, [x12, x12]);
  let x30: uN[72] = !(x7);
  let x31: x16[4] = (x17) ++ (x17);
  let x32: s27 = s27:0x7ff_ffff;
  let x33: x16[2] = update(x17, (u3:0) if ((x21) >= (u3:0)) else (x21), x24);
  let x34: uN[103] = (x11)[x23+:uN[103]];
  (x12, x27)
}
