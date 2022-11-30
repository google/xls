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
// args: bits[24]:0x55_5555; bits[43]:0x768_98be_6049; bits[52]:0x5_5155_d100_0800
// evaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX, simulated =
//    (bits[24]:0xaa_aaab, bits[24]:0x3, bits[2]:0x0, bits[25]:0x1)
// evaluated unopt IR (JIT) =
//    (bits[24]:0xaa_aaab, bits[24]:0x1, bits[2]:0x0, bits[25]:0x1)
// Issue: https://github.com/google/xls/issues/367, https://bugs.llvm.org/show_bug.cgi?id=49778
//
// options: {"codegen": true, "codegen_args": ["--use_system_verilog", "--generator=pipeline", "--pipeline_stages=5", "--reset_data_path=false"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "simulate": false, "simulator": null, "use_jit": true, "use_system_verilog": true}
// args: bits[24]:0x55_5555; bits[43]:0x768_98be_6049; bits[52]:0x5_5155_d100_0800
// args: bits[24]:0x41_3426; bits[43]:0x1_0000; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x4f_705e; bits[43]:0x55_237d_2704; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0x0; bits[43]:0x9_1020_0141; bits[52]:0xb_08f6_2c00_1800
// args: bits[24]:0x55_5555; bits[43]:0xab_3a8c_a005; bits[52]:0x1_5675_5940_0a42
// args: bits[24]:0xff_25fd; bits[43]:0x7e9_0beb_0448; bits[52]:0xf_f55f_d000_0080
// args: bits[24]:0x8; bits[43]:0x49_d482_3eda; bits[52]:0x1_28f3_97ab_7fd8
// args: bits[24]:0x0; bits[43]:0x4e1_9181_4201; bits[52]:0xc209_82f4_bc16
// args: bits[24]:0x55_5555; bits[43]:0x2e0_aabf_ea7b; bits[52]:0x1_71d5_58aa_ea8b
// args: bits[24]:0x0; bits[43]:0x104_007c_566d; bits[52]:0xa_9613_9ef9_ac0b
// args: bits[24]:0xaa_aaaa; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x0
// args: bits[24]:0xff_ffff; bits[43]:0x0; bits[52]:0x100_0000_0000
// args: bits[24]:0x7f_ffff; bits[43]:0x3ff_ffff_ffff; bits[52]:0x3_74ef_6bf7_ff29
// args: bits[24]:0xff_ffff; bits[43]:0x40_0000; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0xaa_aaaa; bits[43]:0x8000_0000; bits[52]:0x20_0000_0000
// args: bits[24]:0x55_5555; bits[43]:0x2eb_e6a9_48c5; bits[52]:0xf4c5_538d_59aa
// args: bits[24]:0x7f_ffff; bits[43]:0x2ff_bfcf_caff; bits[52]:0x7_fe7f_0c65_cec1
// args: bits[24]:0xff_ffff; bits[43]:0x7ff_ffff_ffff; bits[52]:0xe_c245_ed29_be20
// args: bits[24]:0x55_5555; bits[43]:0x555_5555_5555; bits[52]:0x5_d555_5102_0000
// args: bits[24]:0x7f_ffff; bits[43]:0x1fe_8a3a_4ff4; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0x55_5555; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0xb_942c_dd3c_56eb
// args: bits[24]:0x4a_307e; bits[43]:0x2d5_97f2_aba8; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0xaa_aaaa; bits[43]:0x575_5bd9_1308; bits[52]:0x5_5555_5555_5555
// args: bits[24]:0x8e_c77b; bits[43]:0x7ff_ffff_ffff; bits[52]:0x8_0000_0000_0000
// args: bits[24]:0xfe_2309; bits[43]:0x3dd_dee1_11fb; bits[52]:0xf_bbbd_d641_42f6
// args: bits[24]:0xaa_aaaa; bits[43]:0x555_5452_0000; bits[52]:0xa_9aa8_b402_0351
// args: bits[24]:0x7f_ffff; bits[43]:0x323_ac0e_4214; bits[52]:0x8_939d_0d07_cc65
// args: bits[24]:0x7f_ffff; bits[43]:0x3ff_ffdb_ffff; bits[52]:0x7_ffff_f014_e34c
// args: bits[24]:0xff_ffff; bits[43]:0x0; bits[52]:0x8_ff86_c4a2_e941
// args: bits[24]:0x0; bits[43]:0x344c_2200; bits[52]:0x400
// args: bits[24]:0x0; bits[43]:0x7ff_ffff_ffff; bits[52]:0x3_3d72_1ed7_97c4
// args: bits[24]:0x7f_ffff; bits[43]:0x2f_0192_38fa; bits[52]:0x886b_3071_c4aa
// args: bits[24]:0xff_ffff; bits[43]:0x753_e7ff_97d5; bits[52]:0xe_a7cf_ff2f_ab54
// args: bits[24]:0x0; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0x0; bits[43]:0x610_0011_1004; bits[52]:0x5_5555_5555_5555
// args: bits[24]:0x0; bits[43]:0x342_83dd_b8ed; bits[52]:0x1210_0ca0_210a
// args: bits[24]:0x4000; bits[43]:0x406_0007_ffff; bits[52]:0x8_0cd0_2ddf_5e48
// args: bits[24]:0x7f_ffff; bits[43]:0x3ff_f7fa_aaa8; bits[52]:0x9_31fb_1c5f_e053
// args: bits[24]:0xff_ffff; bits[43]:0x7de_7d70_8baa; bits[52]:0x4_0000_0000_0000
// args: bits[24]:0xff_ffff; bits[43]:0x7ff_ffff_ffff; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x0; bits[43]:0x8_0000_0004; bits[52]:0x7_529b_1e59_2375
// args: bits[24]:0x7f_ffff; bits[43]:0x555_5555_5555; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0xaa_aaaa; bits[43]:0x557_7512_baa0; bits[52]:0x7_bc5b_e173_9e97
// args: bits[24]:0xaa_aaaa; bits[43]:0x0; bits[52]:0x5_138f_1829_886e
// args: bits[24]:0x0; bits[43]:0x3ff_ffff_ffff; bits[52]:0x12a1_0a16_26b9
// args: bits[24]:0xff_ffff; bits[43]:0x6ce_ff80_a948; bits[52]:0x2_fdde_43f6_b152
// args: bits[24]:0x55_5555; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x7f_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0xff_ffff; bits[43]:0x5d8_57da_eae7; bits[52]:0xf_ffbf_f555_5555
// args: bits[24]:0x200; bits[43]:0x5d9_c144_6b7c; bits[52]:0xc0a9_0449_7207
// args: bits[24]:0x7f_ffff; bits[43]:0x555_5555_5555; bits[52]:0x7_e7fb_f30a_ab1a
// args: bits[24]:0x90_5d14; bits[43]:0x7ff_ffff_ffff; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0x0; bits[43]:0x7c2_1107_eba7; bits[52]:0x9_909b_0ebf_41fb
// args: bits[24]:0x2_0000; bits[43]:0x555_5555_5555; bits[52]:0xc_4152_9ee0_306d
// args: bits[24]:0x18_5fb6; bits[43]:0x4a2_bbbf_4e19; bits[52]:0x1_85f1_2aaa_baaa
// args: bits[24]:0x80; bits[43]:0x499_5d87_49f5; bits[52]:0xd_3ea2_289b_ec60
// args: bits[24]:0x1000; bits[43]:0x8202_cd5d; bits[52]:0x0
// args: bits[24]:0x0; bits[43]:0xc8_08c1_0103; bits[52]:0x1
// args: bits[24]:0x8_0000; bits[43]:0x54_7670_026d; bits[52]:0x8000_1000_0840
// args: bits[24]:0xff_ffff; bits[43]:0x3ff_ffff_ffff; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0x55_5555; bits[43]:0x3af_a80f_fff7; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0x1; bits[43]:0x3ff_ffff_ffff; bits[52]:0x6_fdbf_fb6f_fa0a
// args: bits[24]:0x1_0000; bits[43]:0x80_0000_0000; bits[52]:0x400_8089_4a4b
// args: bits[24]:0x50_b517; bits[43]:0x18e_b54b_72d8; bits[52]:0x6_2ad7_2691_bff9
// args: bits[24]:0xff_ffff; bits[43]:0x2cf_fffb_fc7e; bits[52]:0x9_c966_2076_dba8
// args: bits[24]:0x80; bits[43]:0x2_0000; bits[52]:0x4_8050_1593_0449
// args: bits[24]:0xaa_aaaa; bits[43]:0x555_5550_4040; bits[52]:0x1_3e3a_ab30_a06b
// args: bits[24]:0xff_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0x0; bits[43]:0x0; bits[52]:0x800_0000
// args: bits[24]:0x0; bits[43]:0x3ff_ffff_ffff; bits[52]:0x2_0000_0802_0040
// args: bits[24]:0xff_ffff; bits[43]:0x7ff_ffbf_fdff; bits[52]:0xf_bfff_f7fb_ffff
// args: bits[24]:0xbf_89a0; bits[43]:0x1f4_0995_f57f; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x55_5555; bits[43]:0x3ff_ffff_ffff; bits[52]:0x2_fd34_2588_1d24
// args: bits[24]:0x3a_beca; bits[43]:0x1d5_f652_a1a7; bits[52]:0x0
// args: bits[24]:0x4_7524; bits[43]:0xa6_a563_efaf; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0xaa_aaaa; bits[43]:0x0; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x42_7502; bits[43]:0x3e4_e218_2184; bits[52]:0xe_5324_be02_85ed
// args: bits[24]:0x3_352d; bits[43]:0x19b_afe7_5677; bits[52]:0x3_375f_ceac_ee45
// args: bits[24]:0xff_ffff; bits[43]:0x7bf_ff6f_efde; bits[52]:0xf_7ffe_dfdf_bdff
// args: bits[24]:0x0; bits[43]:0x555_5555_5555; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0x0; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0xa_74c9_1435_7955
// args: bits[24]:0x7f_e37f; bits[43]:0x7ff_ffff_ffff; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0xff_ffff; bits[43]:0x6bf_227f_c9d9; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0xff_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x5_5555_5555_5404
// args: bits[24]:0x0; bits[43]:0x400_05e0_1071; bits[52]:0x4_0410_4d24_0281
// args: bits[24]:0x50_e87c; bits[43]:0x285_43e7_c06f; bits[52]:0x5_8a87_cf80_9fff
// args: bits[24]:0x4_0000; bits[43]:0x0; bits[52]:0x6000_0802_f103
// args: bits[24]:0x2; bits[43]:0x26b_7293_dbbe; bits[52]:0x4_56e5_22bb_7c00
// args: bits[24]:0x1; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x2a57_1c79_0663
// args: bits[24]:0x100; bits[43]:0x7ff_ffff_ffff; bits[52]:0x8
// args: bits[24]:0x7f_ffff; bits[43]:0x6f5_9a9d_e4e1; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0xff_ffff; bits[43]:0x7fc_7fff_7bde; bits[52]:0xf_f8ff_fef7_bd55
// args: bits[24]:0x15_4cb1; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0x80; bits[43]:0x3ff_ffff_ffff; bits[52]:0x5_6fbf_73fb_5e55
// args: bits[24]:0x3e_b5ed; bits[43]:0x1bb_b854_fd46; bits[52]:0xf4f6_5857_3f46
// args: bits[24]:0xff_ffff; bits[43]:0x555_5555_5555; bits[52]:0xf_fab5_7d7f_2bf6
// args: bits[24]:0x40_706e; bits[43]:0x202_8372_4004; bits[52]:0x0
// args: bits[24]:0x0; bits[43]:0x42_0406_b2fe; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x4000; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x5_5555_5555_5450
// args: bits[24]:0x0; bits[43]:0x0; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0x7f_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0xff_ffff; bits[43]:0x7ff_f7fb_99a9; bits[52]:0xf_f7fd_f7fe_fffb
// args: bits[24]:0x55_5555; bits[43]:0x7ff_ffff_ffff; bits[52]:0x7_3e7c_67be_4c00
// args: bits[24]:0xaa_aaaa; bits[43]:0x71f_7e32_6ed1; bits[52]:0x6_1e7a_afd5_b255
// args: bits[24]:0xaa_aaaa; bits[43]:0x555_5555_5555; bits[52]:0xa_aaaa_a002_0022
// args: bits[24]:0x0; bits[43]:0x8_8002_82ab; bits[52]:0x9_9d7b_dcec_82fc
// args: bits[24]:0xff_ffff; bits[43]:0x0; bits[52]:0x0
// args: bits[24]:0xaa_aaaa; bits[43]:0x615_7c52_0a64; bits[52]:0xf_ffff_ffff_ffff
// args: bits[24]:0x7f_ffff; bits[43]:0x133_6f29_a542; bits[52]:0x5_5555_5555_5555
// args: bits[24]:0x55_5555; bits[43]:0x7ff_ffff_ffff; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0xaa_aaaa; bits[43]:0x0; bits[52]:0x200_0155
// args: bits[24]:0xff_ffff; bits[43]:0x77f_cbfa_abae; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x27_c0b9; bits[43]:0x1be_05c8_0870; bits[52]:0x2_390a_3ac0_4242
// args: bits[24]:0x7f_ffff; bits[43]:0x296_a7b2_a1e5; bits[52]:0x6_254d_76c5_9fbf
// args: bits[24]:0xaa_aaaa; bits[43]:0x7ff_ffff_ffff; bits[52]:0x7_ffff_ffff_ffff
// args: bits[24]:0x2b_618b; bits[43]:0x1df_4455_3755; bits[52]:0x7_bba1_b266_21c7
// args: bits[24]:0x55_5555; bits[43]:0x621_318a_7b34; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x7f_ffff; bits[43]:0x3ff_dff1_6e5c; bits[52]:0x9_6db5_c5de_b3e4
// args: bits[24]:0x55_5555; bits[43]:0x3aa_eabf_2243; bits[52]:0x5_bf4f_26bc_0bb7
// args: bits[24]:0x4000; bits[43]:0x33c_971f_be79; bits[52]:0x1000
// args: bits[24]:0x55_5555; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x5_7455_5abb_3aa8
// args: bits[24]:0x20; bits[43]:0x7ff_ffff_ffff; bits[52]:0x1_0000_87ff_fff7
// args: bits[24]:0x7f_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x4_1571_d55d_582a
// args: bits[24]:0x55_5555; bits[43]:0x222_aae8_8400; bits[52]:0x8_0000_0000
// args: bits[24]:0xaa_aaaa; bits[43]:0x7ff_ffff_ffff; bits[52]:0xb_ea2e_2e33_322a
// args: bits[24]:0xaa_aaaa; bits[43]:0x80; bits[52]:0xa_aaaa_aaaa_aaaa
// args: bits[24]:0x0; bits[43]:0x0; bits[52]:0x4000_0000_04aa
// args: bits[24]:0x55_5555; bits[43]:0x2aa_aaaa_aaaa; bits[52]:0x5_5455_545e_7622
fn main(x0: u24, x1: s43, x2: s52) -> (u24, u24, u2, u25) {
  let x3: s43 = (x1) ^ (((x2) as s43));
  let x5: u24 = (x0) ^ (x0);
  let x6: bool = (x1) != (x3);
  let x7: s43 = (x3) >> (26);
  let x8: bool = (x6)[0+:bool];
  let x9: bool = -(x8);
  let x10: u28 = ((((x8) ++ (x6)) ++ (x8)) ++ (x8)) ++ (x0);
  let x11: u2 = one_hot(x6, bool:false);
  let x12: u24 = (x5) - (x0);
  let x13: u24 = (((x8) as u24)) | (x0);
  let x14: u24 = bit_slice_update(x13, x8, x5);
  let x15: bool = (((x11) as s43)) < (x7);
  let x16: bool = (x6) != (((x2) as bool));
  let x17: bool = (x11) < (((x0) as u2));
  let x18: u2 = (x14)[x11+:u2];
  let x19: s43 = (x3) - (x3);
  let x20: u24 = bit_slice_update(x12, x6, x14);
  let x21: bool = (x8)[0+:bool];
  let x22: bool = rev(x21);
  let x23: u25 = one_hot(x0, bool:true);
  let x24: u24 = !(x0);
  let x25: u46 = u46:0x1fff_ffff_ffff;
  let x26: s43 = !(x7);
  let x27: bool = (x21)[0+:bool];
  let x28: s43 = -(x7);
  let x29: bool = (x2) != (((x8) as s52));
  let x30: s43 = !(x26);
  let x31: u46 = bit_slice_update(x25, x22, x24);
  (x12, x20, x18, x23)
}
