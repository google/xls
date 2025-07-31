// Copyright 2020 The XLS Authors
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
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=combinational"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[19]:0x3_ffff; bits[23]:0x80; (bits[15]:0x22f3, bits[29]:0x1519_8627); (); bits[3]:0x7"
//     args: "bits[19]:0x4; bits[23]:0x10_005f; (bits[15]:0x40e, bits[29]:0xfff_ffff); (); bits[3]:0x2"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x2_802f; (bits[15]:0x2d, bits[29]:0xa7d_b542); (); bits[3]:0x0"
//     args: "bits[19]:0x0; bits[23]:0x21_0a10; (bits[15]:0x829, bits[29]:0x1555_5555); (); bits[3]:0x5"
//     args: "bits[19]:0xe75; bits[23]:0x4_0000; (bits[15]:0x455b, bits[29]:0x1139_0543); (); bits[3]:0x4"
//     args: "bits[19]:0x0; bits[23]:0x20_0319; (bits[15]:0x7fff, bits[29]:0x1fff_ffff); (); bits[3]:0x2"
//     args: "bits[19]:0x5_5555; bits[23]:0x7f_ffff; (bits[15]:0x78ea, bits[29]:0x0); (); bits[3]:0x3"
//     args: "bits[19]:0x5_5555; bits[23]:0x1a_700d; (bits[15]:0x602d, bits[29]:0xaaa_aaaa); (); bits[3]:0x2"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x2a_eaf0; (bits[15]:0x3fff, bits[29]:0x13a2_8bbb); (); bits[3]:0x7"
//     args: "bits[19]:0x0; bits[23]:0x2a_aaaa; (bits[15]:0x2aaa, bits[29]:0xaa8_aaaa); (); bits[3]:0x0"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x6a_2da3; (bits[15]:0x7fff, bits[29]:0x2ae_a1a9); (); bits[3]:0x2"
//     args: "bits[19]:0x0; bits[23]:0x2a_aaaa; (bits[15]:0x4, bits[29]:0xaaa_8a99); (); bits[3]:0x6"
//     args: "bits[19]:0x7_ffff; bits[23]:0x3f_ffff; (bits[15]:0x7fff, bits[29]:0x24e_b8fc); (); bits[3]:0x7"
//     args: "bits[19]:0x5_5555; bits[23]:0x16_5184; (bits[15]:0x5184, bits[29]:0x0); (); bits[3]:0x4"
//     args: "bits[19]:0x5_5555; bits[23]:0x53_537a; (bits[15]:0x5f1d, bits[29]:0x583_e697); (); bits[3]:0x5"
//     args: "bits[19]:0x200; bits[23]:0x11_280b; (bits[15]:0x200, bits[29]:0x1048_03ea); (); bits[3]:0x3"
//     args: "bits[19]:0x1_0000; bits[23]:0x800; (bits[15]:0x0, bits[29]:0x0); (); bits[3]:0x6"
//     args: "bits[19]:0x7_ffff; bits[23]:0x6d_8f9d; (bits[15]:0x7fff, bits[29]:0x1b62_e66e); (); bits[3]:0x7"
//     args: "bits[19]:0x7_ffff; bits[23]:0x2a_aaaa; (bits[15]:0x28aa, bits[29]:0xaaa_aa8f); (); bits[3]:0x3"
//     args: "bits[19]:0x2_ed71; bits[23]:0x1c_32a4; (bits[15]:0x67ad, bits[29]:0xaaa_aaaa); (); bits[3]:0x4"
//     args: "bits[19]:0x7_ffff; bits[23]:0x3f_ffff; (bits[15]:0x3fff, bits[29]:0x1fff_ffff); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x4e_4393; (bits[15]:0x5397, bits[29]:0xaaa_aaaa); (); bits[3]:0x3"
//     args: "bits[19]:0x7_ffff; bits[23]:0x4; (bits[15]:0x800, bits[29]:0x1fff_ffff); (); bits[3]:0x3"
//     args: "bits[19]:0x6_b50f; bits[23]:0x62_731e; (bits[15]:0x3fff, bits[29]:0x641_d017); (); bits[3]:0x5"
//     args: "bits[19]:0x7_ffff; bits[23]:0x65_b59d; (bits[15]:0x0, bits[29]:0x13ed_2670); (); bits[3]:0x2"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x55_5555; (bits[15]:0x2aaa, bits[29]:0xa9a_6203); (); bits[3]:0x3"
//     args: "bits[19]:0x0; bits[23]:0x9_0ce4; (bits[15]:0x0, bits[29]:0x0); (); bits[3]:0x7"
//     args: "bits[19]:0x7_ffff; bits[23]:0x55_5555; (bits[15]:0x777e, bits[29]:0x16f5_f805); (); bits[3]:0x6"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x3f_ffff; (bits[15]:0x2f2a, bits[29]:0x17fd_df75); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x3f_7971; (bits[15]:0x3fff, bits[29]:0xfff_ffff); (); bits[3]:0x2"
//     args: "bits[19]:0x0; bits[23]:0x2a_aaaa; (bits[15]:0x13b, bits[29]:0x0); (); bits[3]:0x0"
//     args: "bits[19]:0x3_ffff; bits[23]:0x5_fa42; (bits[15]:0x2aaa, bits[29]:0x0); (); bits[3]:0x5"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x3f_ffff; (bits[15]:0x2aaa, bits[29]:0x17bd_f7e0); (); bits[3]:0x0"
//     args: "bits[19]:0x0; bits[23]:0x0; (bits[15]:0x5555, bits[29]:0xfff_ffff); (); bits[3]:0x5"
//     args: "bits[19]:0x8000; bits[23]:0x18_850e; (bits[15]:0x50e, bits[29]:0x1555_5555); (); bits[3]:0x4"
//     args: "bits[19]:0x800; bits[23]:0x55_5555; (bits[15]:0x800, bits[29]:0x1551_546e); (); bits[3]:0x4"
//     args: "bits[19]:0x3_ffff; bits[23]:0x3f_ffff; (bits[15]:0x5555, bits[29]:0x1555_5555); (); bits[3]:0x3"
//     args: "bits[19]:0x0; bits[23]:0x7f_ffff; (bits[15]:0x4800, bits[29]:0xaaa_aaaa); (); bits[3]:0x0"
//     args: "bits[19]:0x5_5555; bits[23]:0x30_ff72; (bits[15]:0x21cb, bits[29]:0xd5f_dc9f); (); bits[3]:0x1"
//     args: "bits[19]:0x10; bits[23]:0x11_033a; (bits[15]:0x133e, bits[29]:0x1555_5555); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x55_5555; (bits[15]:0x40, bits[29]:0xfff_ffff); (); bits[3]:0x0"
//     args: "bits[19]:0x5_2064; bits[23]:0x4a_24de; (bits[15]:0x0, bits[29]:0x0); (); bits[3]:0x4"
//     args: "bits[19]:0x5_440d; bits[23]:0x1c_f197; (bits[15]:0x5895, bits[29]:0x1); (); bits[3]:0x2"
//     args: "bits[19]:0x0; bits[23]:0x47_93ca; (bits[15]:0x7385, bits[29]:0x100_0000); (); bits[3]:0x0"
//     args: "bits[19]:0x3_ffff; bits[23]:0x33_f21d; (bits[15]:0x7fff, bits[29]:0x5d6_d92b); (); bits[3]:0x2"
//     args: "bits[19]:0x2; bits[23]:0x74_aa75; (bits[15]:0x5555, bits[29]:0xfb0_9c7f); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x0; (bits[15]:0x5555, bits[29]:0xaaa_aaaa); (); bits[3]:0x0"
//     args: "bits[19]:0x0; bits[23]:0x3f_ffff; (bits[15]:0x7be5, bits[29]:0xfcf_3795); (); bits[3]:0x1"
//     args: "bits[19]:0x7_f22b; bits[23]:0x2a_aaaa; (bits[15]:0x7fff, bits[29]:0x1fff_ffff); (); bits[3]:0x0"
//     args: "bits[19]:0x7_ffff; bits[23]:0x55_5555; (bits[15]:0x4000, bits[29]:0x1555_5555); (); bits[3]:0x0"
//     args: "bits[19]:0x7_ffff; bits[23]:0x2a_aaaa; (bits[15]:0x3ab0, bits[29]:0xe7a_aea0); (); bits[3]:0x7"
//     args: "bits[19]:0x5_5555; bits[23]:0x55_5554; (bits[15]:0x2aaa, bits[29]:0x1fff_ffff); (); bits[3]:0x2"
//     args: "bits[19]:0x5_5555; bits[23]:0x7d_15da; (bits[15]:0x7c9b, bits[29]:0x1253_0e09); (); bits[3]:0x3"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x7f_ffff; (bits[15]:0x3aa2, bits[29]:0xaaa_aaaa); (); bits[3]:0x5"
//     args: "bits[19]:0x0; bits[23]:0x61_4297; (bits[15]:0x3fff, bits[29]:0x200); (); bits[3]:0x2"
//     args: "bits[19]:0x0; bits[23]:0x8; (bits[15]:0x1a, bits[29]:0x1fff_ffff); (); bits[3]:0x3"
//     args: "bits[19]:0x0; bits[23]:0x3; (bits[15]:0xab, bits[29]:0x601_008d); (); bits[3]:0x3"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x7f_ffff; (bits[15]:0x7fff, bits[29]:0xff7_bfdf); (); bits[3]:0x7"
//     args: "bits[19]:0x10; bits[23]:0x7e_2166; (bits[15]:0x2a77, bits[29]:0x0); (); bits[3]:0x3"
//     args: "bits[19]:0x3_ffff; bits[23]:0x7f_a3e1; (bits[15]:0x6231, bits[29]:0x17c6_eef8); (); bits[3]:0x0"
//     args: "bits[19]:0x3_e754; bits[23]:0x7_93bd; (bits[15]:0x200, bits[29]:0xc4_e6e2); (); bits[3]:0x3"
//     args: "bits[19]:0x0; bits[23]:0x2a_aaaa; (bits[15]:0x22aa, bits[29]:0x0); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x3f_ffff; (bits[15]:0x0, bits[29]:0x3ef_a355); (); bits[3]:0x6"
//     args: "bits[19]:0x20; bits[23]:0x2a_aaaa; (bits[15]:0x2c29, bits[29]:0x0); (); bits[3]:0x2"
//     args: "bits[19]:0x3_ffff; bits[23]:0x0; (bits[15]:0x77ff, bits[29]:0x224_3845); (); bits[3]:0x3"
//     args: "bits[19]:0x2_aaaa; bits[23]:0xe_a826; (bits[15]:0x5555, bits[29]:0x1d80_fcd6); (); bits[3]:0x2"
//     args: "bits[19]:0x0; bits[23]:0x2_0005; (bits[15]:0x2aaa, bits[29]:0x1555_5555); (); bits[3]:0x0"
//     args: "bits[19]:0x0; bits[23]:0x10_0040; (bits[15]:0x7fff, bits[29]:0x125_9728); (); bits[3]:0x2"
//     args: "bits[19]:0x3_748d; bits[23]:0x25_0794; (bits[15]:0x5555, bits[29]:0x1959_2fe2); (); bits[3]:0x5"
//     args: "bits[19]:0x7_ffff; bits[23]:0x7f_ffff; (bits[15]:0x7776, bits[29]:0x1fff_fc04); (); bits[3]:0x7"
//     args: "bits[19]:0x7_ffff; bits[23]:0x77_bffa; (bits[15]:0x400, bits[29]:0x0); (); bits[3]:0x3"
//     args: "bits[19]:0x2_a7e1; bits[23]:0x2a_fc12; (bits[15]:0x200, bits[29]:0x183f_8620); (); bits[3]:0x3"
//     args: "bits[19]:0x5_5555; bits[23]:0x55_7750; (bits[15]:0x5555, bits[29]:0xfff_ffff); (); bits[3]:0x7"
//     args: "bits[19]:0x5_77c7; bits[23]:0x76_7472; (bits[15]:0x78d3, bits[29]:0x0); (); bits[3]:0x2"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x2a_aaa5; (bits[15]:0x5555, bits[29]:0xaaa_a961); (); bits[3]:0x7"
//     args: "bits[19]:0x8; bits[23]:0xc_0007; (bits[15]:0x4007, bits[29]:0xfff_ffff); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x2; (bits[15]:0x1000, bits[29]:0x1020); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x64_adf0; (bits[15]:0x6a7a, bits[29]:0x7df_fedf); (); bits[3]:0x0"
//     args: "bits[19]:0x0; bits[23]:0x54_5ae1; (bits[15]:0x2, bits[29]:0xfff_ffff); (); bits[3]:0x2"
//     args: "bits[19]:0x7_f0b6; bits[23]:0x7f_ffff; (bits[15]:0x4000, bits[29]:0x17cc_c1c4); (); bits[3]:0x3"
//     args: "bits[19]:0x7_ffff; bits[23]:0x3f_7ff0; (bits[15]:0x7fff, bits[29]:0xfff_ffff); (); bits[3]:0x7"
//     args: "bits[19]:0x0; bits[23]:0x55_5555; (bits[15]:0x40, bits[29]:0x1555_5555); (); bits[3]:0x2"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x55_5555; (bits[15]:0x3fff, bits[29]:0xaaa_aaaa); (); bits[3]:0x5"
//     args: "bits[19]:0x5_5555; bits[23]:0x55_555a; (bits[15]:0x0, bits[29]:0x1555_5555); (); bits[3]:0x0"
//     args: "bits[19]:0x5_5555; bits[23]:0x0; (bits[15]:0x2, bits[29]:0x4d8_0298); (); bits[3]:0x7"
//     args: "bits[19]:0x5_bfce; bits[23]:0x8_8cb7; (bits[15]:0x497, bits[29]:0xaa3_3bfe); (); bits[3]:0x4"
//     args: "bits[19]:0x0; bits[23]:0x20_ae7d; (bits[15]:0x2aaa, bits[29]:0x1861_176a); (); bits[3]:0x2"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x2a_aaaa; (bits[15]:0x7fff, bits[29]:0x8ba_9803); (); bits[3]:0x2"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x2; (bits[15]:0x7aaa, bits[29]:0x12b2_a1d7); (); bits[3]:0x3"
//     args: "bits[19]:0x7_ffff; bits[23]:0x1000; (bits[15]:0x7fff, bits[29]:0x100); (); bits[3]:0x0"
//     args: "bits[19]:0x1_fba0; bits[23]:0x1f_ba82; (bits[15]:0x3a82, bits[29]:0x1555_5555); (); bits[3]:0x5"
//     args: "bits[19]:0x6_7ee6; bits[23]:0x3f_ffff; (bits[15]:0x6900, bits[29]:0x40); (); bits[3]:0x5"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x2a_e9a7; (bits[15]:0x2aaa, bits[29]:0xfff_ffff); (); bits[3]:0x0"
//     args: "bits[19]:0x1000; bits[23]:0x1_0001; (bits[15]:0x1, bits[29]:0x1555_5555); (); bits[3]:0x5"
//     args: "bits[19]:0x0; bits[23]:0x0; (bits[15]:0x4408, bits[29]:0x0); (); bits[3]:0x0"
//     args: "bits[19]:0x0; bits[23]:0x1c1d; (bits[15]:0x20, bits[29]:0x0); (); bits[3]:0x2"
//     args: "bits[19]:0x3_ffff; bits[23]:0x3f_fff7; (bits[15]:0x52d2, bits[29]:0x1baf_ff9e); (); bits[3]:0x7"
//     args: "bits[19]:0x3_034b; bits[23]:0x14_2677; (bits[15]:0x70b, bits[29]:0xfff_ffff); (); bits[3]:0x7"
//     args: "bits[19]:0x0; bits[23]:0x7f_ffff; (bits[15]:0x7fff, bits[29]:0x1fff_ffff); (); bits[3]:0x4"
//     args: "bits[19]:0x1000; bits[23]:0x11_1700; (bits[15]:0x1000, bits[29]:0xfff_ffff); (); bits[3]:0x0"
//     args: "bits[19]:0x8000; bits[23]:0x8_2005; (bits[15]:0x5555, bits[29]:0x1000); (); bits[3]:0x5"
//     args: "bits[19]:0x7_ffff; bits[23]:0x53_c59f; (bits[15]:0x5f6f, bits[29]:0x1e1e_af94); (); bits[3]:0x0"
//     args: "bits[19]:0x200; bits[23]:0x20_2004; (bits[15]:0x2aaa, bits[29]:0xc0_40a8); (); bits[3]:0x7"
//     args: "bits[19]:0x7_3568; bits[23]:0x46_5303; (bits[15]:0x0, bits[29]:0xfff_ffff); (); bits[3]:0x0"
//     args: "bits[19]:0x7_ffff; bits[23]:0x400; (bits[15]:0x33b2, bits[29]:0x8_0000); (); bits[3]:0x0"
//     args: "bits[19]:0x5_5555; bits[23]:0x0; (bits[15]:0x0, bits[29]:0x1155_5c00); (); bits[3]:0x1"
//     args: "bits[19]:0x0; bits[23]:0xd_a3d9; (bits[15]:0x0, bits[29]:0x208_956a); (); bits[3]:0x2"
//     args: "bits[19]:0x3_ffff; bits[23]:0x31_bffb; (bits[15]:0x7bf7, bits[29]:0xfff_fd97); (); bits[3]:0x2"
//     args: "bits[19]:0x1; bits[23]:0x2_01b7; (bits[15]:0x5b7, bits[29]:0xfff_ffff); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x7f_ffff; (bits[15]:0x57ee, bits[29]:0xfad_b980); (); bits[3]:0x3"
//     args: "bits[19]:0x2_0720; bits[23]:0x32_f209; (bits[15]:0x5555, bits[29]:0x80e_a200); (); bits[3]:0x2"
//     args: "bits[19]:0x5_5555; bits[23]:0x3f_ffff; (bits[15]:0x7fff, bits[29]:0x15bd_4d36); (); bits[3]:0x7"
//     args: "bits[19]:0x5_5555; bits[23]:0x7f_ffff; (bits[15]:0x7097, bits[29]:0x1fff_ffff); (); bits[3]:0x3"
//     args: "bits[19]:0x4_e3b7; bits[23]:0x3f_ffff; (bits[15]:0x628e, bits[29]:0x20_0000); (); bits[3]:0x4"
//     args: "bits[19]:0x0; bits[23]:0x38_0005; (bits[15]:0x6670, bits[29]:0x20); (); bits[3]:0x5"
//     args: "bits[19]:0x0; bits[23]:0x10_2a0e; (bits[15]:0x3fff, bits[29]:0xaaa_aaaa); (); bits[3]:0x2"
//     args: "bits[19]:0x3_ffff; bits[23]:0x2f_befa; (bits[15]:0x2aaa, bits[29]:0x0); (); bits[3]:0x2"
//     args: "bits[19]:0x6_c0e3; bits[23]:0x6c_8e37; (bits[15]:0x70ea, bits[29]:0x1b25_16cc); (); bits[3]:0x3"
//     args: "bits[19]:0x5_5555; bits[23]:0x7d_1b5a; (bits[15]:0x3fff, bits[29]:0x2); (); bits[3]:0x7"
//     args: "bits[19]:0x4_8c3a; bits[23]:0x0; (bits[15]:0x4, bits[29]:0x879_03e1); (); bits[3]:0x2"
//     args: "bits[19]:0x5_77a4; bits[23]:0x3f_ffff; (bits[15]:0x7fff, bits[29]:0xaaa_aaaa); (); bits[3]:0x7"
//     args: "bits[19]:0x3_ffff; bits[23]:0x7b_8530; (bits[15]:0x514, bits[29]:0x1fff_ffff); (); bits[3]:0x0"
//     args: "bits[19]:0x7_ffff; bits[23]:0x2a_b6f3; (bits[15]:0x76f3, bits[29]:0x2_0000); (); bits[3]:0x5"
//     args: "bits[19]:0x7_ffff; bits[23]:0x6f_72a7; (bits[15]:0x6faf, bits[29]:0x14_9641); (); bits[3]:0x4"
//     args: "bits[19]:0xdb73; bits[23]:0x3_2772; (bits[15]:0x5b73, bits[29]:0x1fff_ffff); (); bits[3]:0x0"
//     args: "bits[19]:0x2_aaaa; bits[23]:0x2a_aaaa; (bits[15]:0x2b8b, bits[29]:0x100_0000); (); bits[3]:0x5"
//     args: "bits[19]:0x4_1540; bits[23]:0x0; (bits[15]:0x4408, bits[29]:0x1047_6402); (); bits[3]:0x0"
//     args: "bits[19]:0x5_5555; bits[23]:0x3f_ffff; (bits[15]:0x4eff, bits[29]:0x0); (); bits[3]:0x0"
//   }
// }
// END_CONFIG
fn main(x0: u19, x1: u23, x2: (u15, u29), x3: (), x4: u3) -> (bool, u23, u23, (), u19, u19, (), u19, bool, ()) {
  let x5: () = for (i, x): (u4, ()) in u4:0..u4:1 {
    x
  }(x3);
  let x6: bool = (x4) == (((x0) as u3));
  let x7: bool = (((x1) as bool)) - (x6);
  let x8: u23 = !(x1);
  let x9: bool = bool:0x0;
  let x10: s13 = s13:0xfff;
  let x11: u23 = (x8) * (((x6) as u23));
  let x12: u27 = ((x4) ++ (x6)) ++ (x8);
  let x13: u27 = !(x12);
  let x14: bool = or_reduce(x0);
  let x15: u23 = bit_slice_update(x1, x0, x11);
  let x16: bool = !(x7);
  let x17: u23 = clz(x1);
  let x18: bool = !(x6);
  let x19: bool = !(x18);
  let x20: u27 = ((x13 as s27) >> (x8)) as u27;
  let x21: u23 = ctz(x8);
  let x22: u24 = one_hot(x15, bool:true);
  let x23: u23 = !(x15);
  let x24: u23 = (x15) + (((x10) as u23));
  let x25: bool = (((x22) as u23)) > (x21);
  let x26: bool = (x19)[0+:bool];
  let x27: u2 = (x10 as u13)[7:9];
  let x28: u19 = (x21)[x11+:u19];
  let x29: bool = (x26)[0+:bool];
  let x30: u23 = -(x23);
  let x31: u23 = clz(x1);
  (x14, x23, x11, x5, x28, x0, x5, x0, x18, x5)
}
