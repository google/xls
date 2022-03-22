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
// Result miscompare for sample 1:
// args: bits[52]:0x0; bits[49]:0x0
// evaluated opt IR (JIT), evaluated opt IR (interpreter) =
//    (bits[1]:0x0, bits[1]:0x0, [(bits[50]:0x0)], bits[1]:0x0, bits[52]:0x0)
// evaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[1]:0x0, bits[1]:0x1, [(bits[50]:0x0)], bits[1]:0x1, bits[52]:0x0)
//
// options: {"codegen": false, "codegen_args": null, "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "simulate": false, "simulator": null, "timeout_seconds": null, "use_jit": true, "use_system_verilog": true}
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_55e7_dd9c_a2ec
// args: bits[52]:0x0; bits[49]:0x0
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_f8fb_7eef_bb37
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_fffb_ff77_f3de
// args: bits[52]:0x400_0000_0000; bits[49]:0x440_0001
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x73fa_05e5_a28a
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x53be_c7df_fefb
// args: bits[52]:0x2_6f8d_2c8b_1e05; bits[49]:0x6f8d_2e8b_1e27
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x0
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x0; bits[49]:0x0
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x10_0000
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xbbaa_a9aa_abac
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_0000_0000
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_dc75_6175_411f
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x1_a1c1_eb01_dcc6
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_c1ae_3cb3_9f11
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xaaa8_baaa_baae
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x31c9_e44a_e38b
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x0
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_6b82_79b8_28e1
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1d6c_5424_5357
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_bedd_57df_b5f3
// args: bits[52]:0x20_0000; bits[49]:0x8001_4020_00a0
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_5155_5555_5d55
// args: bits[52]:0x8_f27e_6366_2491; bits[49]:0xf07e_635f_2491
// args: bits[52]:0x0; bits[49]:0x400_0000
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xaaaa_aaaa_aaaa
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xa8b8_ce81_daea
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x1_a028_3a86_b24a
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x0
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x5115_5132_4d05
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_5eff_97bf_fffd
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0x7_53df_179d_f147; bits[49]:0x1_139b_979d_f243
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0xb6fb_fd7f_dfef
// args: bits[52]:0x1_0000_0000; bits[49]:0x8_0000
// args: bits[52]:0x8000_0000_0000; bits[49]:0x1_8c01_30ec_2202
// args: bits[52]:0x7_543b_675b_42e8; bits[49]:0x541b_e75b_42e8
// args: bits[52]:0x8_0000_0000_0000; bits[49]:0x400_0000_0000
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xffff_ffff_ffff
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0x0; bits[49]:0x10_0000_0000
// args: bits[52]:0x0; bits[49]:0xb216_62d6_650b
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xb6aa_aabc_aa80
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_ff6c_366c_7ffd
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_fffb_ff3f_fd79
// args: bits[52]:0x0; bits[49]:0x4000_0000
// args: bits[52]:0x2; bits[49]:0x1_1332_ca52_4082
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x0
// args: bits[52]:0x9_3b37_7dfc_4df2; bits[49]:0x1_1b3f_39fd_acf2
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xffff_ffff_ffff
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xebea_aa28_ae2b
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_e651_7baf_bedc
// args: bits[52]:0x0; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x2_7b3f_97df_26de; bits[49]:0x7b3f_97df_27df
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x100_0000
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_4555_5555_5555
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_dfff_dff3_7fff
// args: bits[52]:0x0; bits[49]:0x115_a102_f018
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xffff_ffff_ffff
// args: bits[52]:0x6_c1a3_6c8b_d9ab; bits[49]:0xaaaa_aaaa_aaaa
// args: bits[52]:0x0; bits[49]:0xaaaa_aaaa_aaaa
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_5105_575d_5555
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_9319_4aa6_66e6
// args: bits[52]:0x10_0000_0000; bits[49]:0x10_2152_0102
// args: bits[52]:0x9_bacf_c0dc_46ba; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0xffff_ffff_ffff
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0xaaaa_aaaa_aaaa
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xa88a_2aab_eaaa
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x2000_0000_0000
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_afb7_ffef_dfff
// args: bits[52]:0x8_04d5_1e05_8851; bits[49]:0x0
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xbc9f_71ef_fffb
// args: bits[52]:0x2_b88c_d914_5960; bits[49]:0x9aa4_9d16_5978
// args: bits[52]:0x20_0000; bits[49]:0x800_002a_0050
// args: bits[52]:0x2_70dc_1167_17ed; bits[49]:0x0
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x10_0000
// args: bits[52]:0x0; bits[49]:0x0
// args: bits[52]:0x6_a594_dbca_c3f8; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_5db4_64d5_99c5
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x67f7_d53c_4141
// args: bits[52]:0x0; bits[49]:0x0
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_5555_5d55_5d55
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xaaaa_aaaa_aaaa
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xffff_ffff_ffff
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x2621_0ae0_a68b
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_5d4f_7534_7f55
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_5555_5541_5515
// args: bits[52]:0x4_127a_13b6_665c; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x0; bits[49]:0x10e0_2a20_8508
// args: bits[52]:0x0; bits[49]:0xaaaa_aaaa_aaaa
// args: bits[52]:0x0; bits[49]:0x802_0184_041e
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xffff_ffff_ffff
// args: bits[52]:0x0; bits[49]:0x0
// args: bits[52]:0x0; bits[49]:0x8c2_0004_416a
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x77b1_160a_6e5b; bits[49]:0x0
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x0
// args: bits[52]:0xa_f6af_ba9f_0e2b; bits[49]:0xa68f_ba9f_0e6b
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xffff_ffff_ffff
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x0; bits[49]:0x1_0700_f791_b915
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_f9f7_6bbf_f379
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xaaaa_aaaa_aaaa
// args: bits[52]:0xe_9816_b935_e285; bits[49]:0x992c_a9cc_4d6c
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_7fdb_ffff_fff9
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xb2aa_eaaa_aaaa
// args: bits[52]:0x0; bits[49]:0x43f4_53c2_003d
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x40_0000_0000; bits[49]:0x1000
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x1_a283_8aae_8eb8
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0x1_aeba_0aaa_8c2a
// args: bits[52]:0x0; bits[49]:0x1_ffff_ffff_ffff
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_5555_5555_5555
// args: bits[52]:0x5_5555_5555_5555; bits[49]:0x1_41f5_5ed7_75c7
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0x1_f56f_edf8_dfff
// args: bits[52]:0xf_ffff_ffff_ffff; bits[49]:0x1_fdff_bfff_f7ff
// args: bits[52]:0x800; bits[49]:0x1878_d152_1822
// args: bits[52]:0xa_aaaa_aaaa_aaaa; bits[49]:0xaaae_aaaa_aaaa
// args: bits[52]:0x6_fc4c_09b9_0bfb; bits[49]:0x1_f98a_923c_ccfa
// args: bits[52]:0x7_ffff_ffff_ffff; bits[49]:0xffff_ffff_ffff
const W32_V7 = u32:7;
type x3 = s49;
type x25 = (u50,);
type x31 = u7;
fn x20(x21: x3) -> (u50,) {
  let x22: u25 = u25:0x0;
  let x23: u50 = (x22) ++ (x22);
  let x24: u50 = gate!((x23) <= (x23), x23);
  (x23,)
}
fn main(x0: s52, x1: s49) -> (bool, bool, x25[1], bool, s52) {
  let x2: u49 = u49:0x1_ffff_ffff_ffff;
  let x4: x3[1] = [x1];
  let x5: u28 = u28:0xfff_ffff;
  let x6: u49 = gate!((((x2) as s52)) >= (x0), x2);
  let x7: bool = (x6) != (((x5) as u49));
  let x8: bool = (x0) > (((x7) as s52));
  let x9: bool = (x5) >= (((x2) as u28));
  let x10: u49 = rev(x6);
  let x11: u42 = (x6)[-42:];
  let x12: bool = and_reduce(x7);
  let x13: bool = (x8) + (((x10) as bool));
  let x14: bool = (x12) & (((x7) as bool));
  let x15: u28 = clz(x5);
  let x16: u49 = !(x2);
  let x17: bool = (x14) ^ (((x6) as bool));
  let x18: u49 = !(x10);
  let x19: (u42, u28, bool, u49, u49) = (x11, x5, x8, x10, x6);
  let x26: x25[1] = map(x4, x20);
  let x27: u42 = (((x15) as u42)) * (x11);
  let x28: bool = (x19) == (x19);
  let x29: s42 = s42:0x2aa_aaaa_aaaa;
  let x30: u28 = ctz(x15);
  let x32: x31[W32_V7] = ((x10) as x31[W32_V7]);
  (x8, x7, x26, x14, x0)
}
