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
// exception: "// Command \'[\'xls/tools/opt_main\', \'sample.ir\', \'--logtostderr\']\' timed out after 1500 seconds"
// issue: "https://github.com/google/xls/issues/871"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x14"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x1000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x100_0000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xf_b2ac_2d30_61cd_0f34_1ffa_ed36_0878"
//       values: "bits[116]:0xd_3f19_0f71_d071_0792_3253_3d55_1d0a"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x27dc_a873_0302_f339_e87e_1d95_c626"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0xb_6063_f03b_7312_1e86_762c_bec9_1430"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0x2_0000_0000_0000_0000"
//       values: "bits[116]:0xe_b131_3684_50e8_4d26_b835_5260_36ab"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x7_4c8f_072f_45db_bcc5_5a02_7972_cd34"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xcd0_04fb_86d9_7d6e_18f3_97aa_5d7a"
//       values: "bits[116]:0x1_0000_0000_0000_0000"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0xb_a3ba_9bd7_f5c1_9829_025d_4d11_e5d0"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xe_0e13_2c2c_a3a4_bb15_0b42_cebe_d81d"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x9_f92d_2b35_3f22_d7d1_ee56_0ef7_ea66"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x1000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x40"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0x1000_0000"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xb_2e3e_f040_e4b4_9cc3_e478_5417_b68d"
//       values: "bits[116]:0x2"
//       values: "bits[116]:0xc_6094_02d7_e28e_7a68_a306_dd4c_4bf0"
//       values: "bits[116]:0x8_b31c_9b52_9fda_a7bf_1c36_8e01_f6b7"
//       values: "bits[116]:0x2_0000_0000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x40_0000_0000_0000"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0x3_a221_9855_c39f_f9fa_ef6b_1107_6799"
//       values: "bits[116]:0x2000_0000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x200_0000"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x9_9640_33a6_88f3_45b0_bf1d_340d_3878"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x80_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x400_0000_0000_0000"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xa_587e_2335_67d9_31b0_1ba1_f3c3_e80d"
//       values: "bits[116]:0x4_73ae_0451_1b71_2e73_fe42_6742_a33f"
//       values: "bits[116]:0x1000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0x1_a904_9cc0_5888_38b9_38af_cd22_9ebb"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0x80_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x400_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x2_8e80_6e01_2879_29a1_c6bf_1920_572e"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x3_d47e_2fe7_3829_8adb_a2d6_008a_7711"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0x6_c032_cab3_3454_490c_09ee_87f2_8586"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[116]:0xf_d04c_31a6_255f_5026_5871_9856_10d1"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xd_9325_8b80_af0a_3ab9_9ec6_fe95_c160"
//       values: "bits[116]:0x6_497c_4b94_52f8_3692_3a11_a43c_af5f"
//       values: "bits[116]:0x100_0000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0xc_005a_2703_dd08_6a48_ad37_bd4f_f484"
//       values: "bits[116]:0xd_57c6_641e_cae1_93e3_3bf3_d155_ad5a"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0xd_e496_9225_2d06_ae73_0e42_56f0_05d8"
//       values: "bits[116]:0x100_0000_0000_0000"
//       values: "bits[116]:0x8000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x100_0000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x100_0000_0000_0000_0000_0000"
//       values: "bits[116]:0x2_0000_0000_0000_0000_0000_0000"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0x4_0000_0000"
//       values: "bits[116]:0x0"
//       values: "bits[116]:0x4"
//       values: "bits[116]:0x5_a8e8_3f02_5dd3_a819_c14d_cea3_9b71"
//       values: "bits[116]:0xf_a94f_4c69_5113_272e_c9e8_2027_ffcc"
//       values: "bits[116]:0x5_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[116]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     }
//   }
// }
// END_CONFIG
proc main {
  x14: chan<sN[116]> in;
  config(x14: chan<sN[116]> in) {
    (x14,)
  }
  init {
    uN[1222]:0x4000_0000
  }
  next(x1: uN[1222]) {
    let x0: token = join();
    let x2: uN[1222] = !(x1);
    let x3: (uN[1222], uN[1222], uN[1222]) = (x2, x2, x1);
    let x4: u62 = u62:0x1555_5555_5555_5555;
    let x5: bool = (x2) >= (x2);
    let x6: u62 = rev(x4);
    let x7: u62 = bit_slice_update(x4, x1, x1);
    let x8: bool = (((x7) as bool)) - (x5);
    let x9: bool = !(x8);
    let x10: bool = (x8)[0+:bool];
    let x11: u62 = rev(x7);
    let x12: u62 = !(x4);
    let x13: bool = (x5) & (x5);
    let x15: (token, sN[116]) = recv_if(x0, x14, x9, sN[116]:0);
    let x16: token = x15.0;
    let x17: sN[116] = x15.1;
    let x18: bool = (x5) & (x5);
    let x19: bool = (x10)[x8+:bool];
    let x20: bool = !(x18);
    let x21: token = join(x0, x16);
    let x22: bool = -(x13);
    let x23: uN[1223] = one_hot(x2, bool:0x1);
    let x24: u62 = -(x6);
    let x25: bool = -(x9);
    let x26: bool = or_reduce(x10);
    let x27: u18 = (x4)[x26+:u18];
    let x28: bool = (x3) == (x3);
    x2
  }
}
