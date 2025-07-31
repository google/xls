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
// evaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[42]:0x376_4555_e555, bits[2]:0x1, bits[46]:0x2aaa_aaaa_aaaa, bits[1]:0x0)
// evaluated unopt IR (JIT) =
//    (bits[42]:0x376_4555_e555, bits[2]:0x3, bits[46]:0x2aaa_aaaa_aaaa, bits[1]:0x0)
//
// BEGIN_CONFIG
// exception: "// Result miscompare for sample 0:"
// issue: "https://github.com/google/xls/issues/374"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=2"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x89_baaa_1aab"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x89_baaa_1aab"
//     args: "bits[46]:0x800_0000; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x0; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x2f4_e6ff_9d37"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x80_0000_0000"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x57_d84f_d978"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x155_5555_5555"
//     args: "bits[46]:0x0; bits[42]:0x10_0017_8101"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x145_5555_5555"
//     args: "bits[46]:0x1010_96db_af00; bits[42]:0x155_5555_5555"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x2e9_f97f_7bbf"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x100_0000_0000"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x3ff_de9b_ff9e"
//     args: "bits[46]:0x80; bits[42]:0x10_0000"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x3f1_9efe_4f79"
//     args: "bits[46]:0x800_0000; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x2bf_66ea_81f9"
//     args: "bits[46]:0x0; bits[42]:0x0"
//     args: "bits[46]:0x40_0000; bits[42]:0x24c_0261_8380"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x155_5555_5555"
//     args: "bits[46]:0xf89_d178_58fa; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x107c_1490_ae76; bits[42]:0xf5_15f4_8ef6"
//     args: "bits[46]:0x4000_0000; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x0; bits[42]:0x170_1810_4222"
//     args: "bits[46]:0x80_0000_0000; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x400_0000_0000; bits[42]:0x20_8806_0022"
//     args: "bits[46]:0x22e3_5cff_ce79; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x2387_df6f_5962; bits[42]:0x28f_f1f7_fd7c"
//     args: "bits[46]:0x14cb_0e41_3187; bits[42]:0x0"
//     args: "bits[46]:0x1000_0000; bits[42]:0x7_1080_0005"
//     args: "bits[46]:0x0; bits[42]:0x1b0_83a0_0087"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x155_5555_5555"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x0"
//     args: "bits[46]:0x10_0000; bits[42]:0x59_a47c_2730"
//     args: "bits[46]:0x2000; bits[42]:0x8_c260_2102"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x56_d9ef_4394"
//     args: "bits[46]:0x0; bits[42]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x1eb_2fbf_be1f"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x1fc_36b8_823c"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x351_5545_5554"
//     args: "bits[46]:0x4000; bits[42]:0x5853_4020"
//     args: "bits[46]:0x20_0000; bits[42]:0x0"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x3da_baea_20ea"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x3ff_df7f_bfff"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x1fe_7deb_e97c"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x3fd_af5d_f7fd"
//     args: "bits[46]:0x2368_038d_c41b; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x3d7_eabb_ffbf"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x311_1275_fece"
//     args: "bits[46]:0x0; bits[42]:0x180_282a_1878"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x116_7fee_0f5f"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x2b2_d664_aef2"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x200; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x3d7_fefe_e833"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x24b_1c6f_6306"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x3ff_e7fe_ffff"
//     args: "bits[46]:0x7ff_2e10_3fe2; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x32dd_8ae9_b240; bits[42]:0x2bd_8a69_a064"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x0; bits[42]:0x308_0000_0238"
//     args: "bits[46]:0x3ca4_4e7c_0b6b; bits[42]:0x44_4eec_036f"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x2f3_ffdd_bff7"
//     args: "bits[46]:0x200; bits[42]:0x155_5555_5555"
//     args: "bits[46]:0x0; bits[42]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x3bf_dfd7_fbaf"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x3bf_ffff_fffe"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0xeb_996a_deac"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x8000; bits[42]:0x2d62_f444"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x155_5555_5555"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x3a7_f8a9_1670"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x39f_ffff_fffe"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x4000"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x0; bits[42]:0x0"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x112_7900_1175"
//     args: "bits[46]:0x1a06_1a47_790e; bits[42]:0x207_18c7_731e"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x3a3_1a2a_be92"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x0"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x135_079d_f413"
//     args: "bits[46]:0x8_0000; bits[42]:0x40_0158_a18c"
//     args: "bits[46]:0x3e29_fc75_25cf; bits[42]:0xab_f86d_2fdf"
//     args: "bits[46]:0x0; bits[42]:0x280_0800_8109"
//     args: "bits[46]:0x1_0000_0000; bits[42]:0x9_20f0_880e"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x0; bits[42]:0x94_0040_004b"
//     args: "bits[46]:0x0; bits[42]:0x0"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x1b7c_bee1_1a50; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x20_0000_0000; bits[42]:0xa0_0381_2800"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0xa2_8aa0_edea"
//     args: "bits[46]:0x8_0000_0000; bits[42]:0x0"
//     args: "bits[46]:0x1f38_726f_ff38; bits[42]:0x155_5555_5555"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x1fff_ffff_ffff; bits[42]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x3ff_ff1e_fff7"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x2ab_6a0f_a2a7"
//     args: "bits[46]:0x3124_ade2_96b7; bits[42]:0x29c_a7f2_12f3"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x37c_ff8f_6efd"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x134_55dd_35d7"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x0; bits[42]:0x4480_0000"
//     args: "bits[46]:0x0; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x2; bits[42]:0xa6_0810_6280"
//     args: "bits[46]:0x1c3e_0dad_36c8; bits[42]:0x236_09c9_7678"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x4_0000_0000"
//     args: "bits[46]:0x0; bits[42]:0x321_9a4f_4e47"
//     args: "bits[46]:0x1555_5555_5555; bits[42]:0x3ff_ffff_ffff"
//     args: "bits[46]:0x26ab_1312_4297; bits[42]:0x1ff_ffff_ffff"
//     args: "bits[46]:0x30df_1fd2_22c2; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x0"
//     args: "bits[46]:0x3fff_ffff_ffff; bits[42]:0x2aa_aaaa_aaaa"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0xfc_beb8_a123"
//     args: "bits[46]:0x0; bits[42]:0x200_0000_0000"
//     args: "bits[46]:0x2aaa_aaaa_aaaa; bits[42]:0x127_16b9_c2fb"
//   }
// }
// END_CONFIG
type x14 = bool;
type x22 = x14;
type x25 = (x22[2], x22[2], x22[2]);
fn x20(x21: x14) -> (x22[2], x22[2], x22[2]) {
  let x23: x22[2] = [x21, x21];
  let x24: x22[4] = (x23) ++ (x23);
  (x23, x23, x23)
}
fn main(x0: s46, x1: s42) -> (s42, u2, s46, bool) {
  let x2: s42 = !(x1);
  let x3: s42 = -(x1);
  let x4: u2 = u2:0x3;
  let x5: u8 = (((x4) ++ (x4)) ++ (x4)) ++ (x4);
  let x6: s42 = (x3) + (((x5) as s42));
  let x7: u8 = for (i, x): (u4, u8) in u4:0..u4:2 {
    x
  }(x5);
  let x8: (s42, s46) = (x1, x0);
  let x9: s42 = (x2) ^ (x6);
  let x10: u42 = (x1 as u42)[x5+:u42];
  let x11: u2 = (((x5) as u2)) * (x4);
  let x12: bool = xor_reduce(x7);
  let x13: bool = (x5)[x7+:bool];
  let x15: x14[8] = ((x7) as x14[8]);
  let x16: bool = !(x13);
  let x17: u2 = one_hot_sel(x13, [x4]);
  let x18: bool = (x6) <= (x9);
  let x19: u42 = one_hot_sel(x5, [x10, x10, x10, x10, x10, x10, x10, x10]);
  let x26: x25[8] = map(x15, x20);
  let x27: u2 = (x17) + (x11);
  (x3, x27, x0, x12)
}
