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
//
// BEGIN_CONFIG
// exception: "// Command \'[\'xls/tools/simulate_module_main\', \'--signature_file=module_sig.textproto\', \'--args_file=args.txt\', \'--verilog_simulator=iverilog\', \'sample.v\', \'--logtostderr\']\' returned non-zero exit status 1."
// issue: "https://github.com/google/xls/issues/706"
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
//   codegen_args: "--pipeline_stages=3"
//   codegen_args: "--reset_data_path=false"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 600
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0x1a77_adb9_d6ff_aeb8; (bits[1]:0x1); bits[10]:0x269"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x1); bits[10]:0x2aa"
//     args: "bits[58]:0x377_0b6f_35f9_2ecf; bits[61]:0x2000; (bits[1]:0x0); bits[10]:0x155"
//     args: "bits[58]:0x0; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x0); bits[10]:0x44"
//     args: "bits[58]:0x0; bits[61]:0x411_0490_9824_0415; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x555_10c5_541d_d350; (bits[1]:0x1); bits[10]:0x312"
//     args: "bits[58]:0x200_0000; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x3ff"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x0); bits[10]:0x2cb"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x37f"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x0; (bits[1]:0x0); bits[10]:0x37f"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x1); bits[10]:0x23d"
//     args: "bits[58]:0x0; bits[61]:0xa10_cb21_304a_c0c7; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x125_690b_16bf_abb7; bits[61]:0x10b_6859_59bd_36c0; (bits[1]:0x0); bits[10]:0x1b3"
//     args: "bits[58]:0x0; bits[61]:0x2c1_0795_2811_0775; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x0; bits[61]:0x1fd2_acdd_07c4_1b42; (bits[1]:0x0); bits[10]:0xc"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1ffd_ffff_ffff_fffa; (bits[1]:0x1); bits[10]:0x3ff"
//     args: "bits[58]:0x4000_0000; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x80"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x3f"
//     args: "bits[58]:0x0; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x4000; bits[61]:0xd19_0ec2_c484_b087; (bits[1]:0x0); bits[10]:0x50"
//     args: "bits[58]:0x0; bits[61]:0x200_4002_089c_0013; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x3c7_d72f_3667_c73e; bits[61]:0xe3e_b979_b33e_69f5; (bits[1]:0x0); bits[10]:0x1"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x0; (bits[1]:0x1); bits[10]:0x31d"
//     args: "bits[58]:0x1_0000; bits[61]:0x202_3c31_6704_2009; (bits[1]:0x1); bits[10]:0x0"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1ffd_7c6f_5eee_bfdf; (bits[1]:0x1); bits[10]:0x144"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0xaaa_abaa_aaaa_aaad; (bits[1]:0x0); bits[10]:0x146"
//     args: "bits[58]:0x400_0000; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x1d"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x335_ec16_3f2a_681f; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x0; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x0; bits[61]:0xd49_cca6_5176_6470; (bits[1]:0x0); bits[10]:0x132"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1_0000; (bits[1]:0x0); bits[10]:0x3ff"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x6de_faf8_e7b4_133b; (bits[1]:0x1); bits[10]:0x25b"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x0; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x152"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x1); bits[10]:0x105"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x0; (bits[1]:0x0); bits[10]:0x1ff"
//     args: "bits[58]:0x80_0000_0000; bits[61]:0x966_015f_b7f2_194a; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x1_0000_0000_0000; bits[61]:0x500_0000_0042; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xfff_b7ef_ffff_fffd; (bits[1]:0x1); bits[10]:0x2bc"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x355_3521_a508_4474; (bits[1]:0x0); bits[10]:0x2eb"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xf45_72ee_ffeb_1de0; (bits[1]:0x1); bits[10]:0x2aa"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x0); bits[10]:0x2a"
//     args: "bits[58]:0x0; bits[61]:0x400; (bits[1]:0x0); bits[10]:0x1ff"
//     args: "bits[58]:0x400_0000_0000; bits[61]:0x18_ba61_e94e_60ed; (bits[1]:0x0); bits[10]:0x155"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0x1afe_12ff_3baa_1b89; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x555_5457_5757_4554; (bits[1]:0x0); bits[10]:0x1ff"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0x18b_af67_c6a3_ffc5; (bits[1]:0x1); bits[10]:0x3de"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1eb0_2f0a_2919_c37a; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x800; bits[61]:0x1_0140_0000_4202; (bits[1]:0x1); bits[10]:0x198"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1fef_ffff_ffff_fdfa; (bits[1]:0x1); bits[10]:0x200"
//     args: "bits[58]:0x0; bits[61]:0x3; (bits[1]:0x0); bits[10]:0x40"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1fff_e9ff_5fff_fffe; (bits[1]:0x0); bits[10]:0x21f"
//     args: "bits[58]:0x307_b8bc_5cdf_3813; bits[61]:0x1000_0000; (bits[1]:0x1); bits[10]:0x2e6"
//     args: "bits[58]:0x3a4_6075_b9af_a895; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x3ff"
//     args: "bits[58]:0x0; bits[61]:0xfbf_da16_89df_8805; (bits[1]:0x1); bits[10]:0x5"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1d8b_177d_bbd9_61d4; (bits[1]:0x0); bits[10]:0x155"
//     args: "bits[58]:0x0; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x13f"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x80"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0x0; (bits[1]:0x1); bits[10]:0x48"
//     args: "bits[58]:0x0; bits[61]:0x0; (bits[1]:0x0); bits[10]:0x155"
//     args: "bits[58]:0x3cf_8b55_1f6e_7e16; bits[61]:0x1e7c_5aa8_bb73_f0b3; (bits[1]:0x1); bits[10]:0x0"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1fed_fdff_ffff_fbf9; (bits[1]:0x0); bits[10]:0x2aa"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1217_8d2e_0085_485f; (bits[1]:0x0); bits[10]:0x2aa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1fdf_fffe_fc77_fffb; (bits[1]:0x0); bits[10]:0x2aa"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0xaaa_aaaa_aeaa_aaab; (bits[1]:0x1); bits[10]:0x0"
//     args: "bits[58]:0x200_0000_0000; bits[61]:0x1000_0800_0000; (bits[1]:0x0); bits[10]:0x3ff"
//     args: "bits[58]:0x0; bits[61]:0x0; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x0; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x1ff"
//     args: "bits[58]:0x247_bf2e_cfd7_35ad; bits[61]:0x123d_f976_7eb9_ad6d; (bits[1]:0x0); bits[10]:0xbc"
//     args: "bits[58]:0x0; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x0); bits[10]:0x100"
//     args: "bits[58]:0x8000_0000; bits[61]:0x8_0000_0000; (bits[1]:0x0); bits[10]:0x3ff"
//     args: "bits[58]:0x0; bits[61]:0x179c_824a_b544_80d3; (bits[1]:0x1); bits[10]:0x1"
//     args: "bits[58]:0x0; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x0"
//     args: "bits[58]:0x22e_388f_c855_c227; bits[61]:0x171_cff6_52a6_717d; (bits[1]:0x1); bits[10]:0x3ff"
//     args: "bits[58]:0x8000_0000; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x0"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x2ba_3a38_1b2b_ac92; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x142_29fd_0e55_428a; bits[61]:0xa11_57e8_76aa_1057; (bits[1]:0x1); bits[10]:0x3e8"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xfff_bfff_ffff_fffd; (bits[1]:0x1); bits[10]:0x2aa"
//     args: "bits[58]:0x39e_c8f8_c6b2_cb0f; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x3ff"
//     args: "bits[58]:0x34e_49c6_943c_ad47; bits[61]:0xab4_afb4_7175_1a2d; (bits[1]:0x1); bits[10]:0x2aa"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1aa2_a2aa_aa86_a2af; (bits[1]:0x1); bits[10]:0x1f1"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xea9_a8fe_1290_7f9c; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x0; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x1); bits[10]:0x164"
//     args: "bits[58]:0xae_e5e6_2d4c_3e6f; bits[61]:0xdf7_3b65_6b31_d377; (bits[1]:0x1); bits[10]:0x0"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x4000; (bits[1]:0x0); bits[10]:0x156"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0x0; (bits[1]:0x0); bits[10]:0x20"
//     args: "bits[58]:0x359_e080_6eb7_063d; bits[61]:0x0; (bits[1]:0x1); bits[10]:0x80"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1cad_bdff_b303_3968; (bits[1]:0x0); bits[10]:0x188"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x80_0000; (bits[1]:0x0); bits[10]:0x286"
//     args: "bits[58]:0x800_0000; bits[61]:0x10_0000_4000_0005; (bits[1]:0x0); bits[10]:0x3ff"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1c9e_fbef_dedf_77f8; (bits[1]:0x0); bits[10]:0x155"
//     args: "bits[58]:0x0; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x1); bits[10]:0x3ff"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x1); bits[10]:0x2aa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x1); bits[10]:0x235"
//     args: "bits[58]:0x0; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x44"
//     args: "bits[58]:0x0; bits[61]:0x1810_0410_2050_0e83; (bits[1]:0x0); bits[10]:0x2aa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1f7f_effd_ffff_ffd9; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x3fc_fd90_52cd_4c8d; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x1); bits[10]:0x157"
//     args: "bits[58]:0x4000_0000; bits[61]:0x15d_945b_4a5f_0f91; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x357"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x1efa_ff7f_7a7d_adfa; (bits[1]:0x0); bits[10]:0x3de"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x17f4_fee3_2a1e_e452; (bits[1]:0x0); bits[10]:0x3a8"
//     args: "bits[58]:0xdf_e3e1_e393_6e6a; bits[61]:0x6df_1f0f_149b_7355; (bits[1]:0x0); bits[10]:0x1ff"
//     args: "bits[58]:0x8_0000_0000; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x0); bits[10]:0x155"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0xaa4_aa1a_a3aa_8ba8; (bits[1]:0x0); bits[10]:0x200"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[61]:0x10_0000_0000_0000; (bits[1]:0x1); bits[10]:0x40"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x200_0000_0000_0000; (bits[1]:0x0); bits[10]:0x2"
//     args: "bits[58]:0x80_0000; bits[61]:0x400_8528_0012; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0xaaf_23ea_a8aa_aade; (bits[1]:0x0); bits[10]:0x3da"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0x100_0000_0000_0000; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xf66_7bff_f37d_9f59; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xfff_ffff_ffff_effd; (bits[1]:0x0); bits[10]:0x0"
//     args: "bits[58]:0x385_cd70_31a2_2b51; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x2f7"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xaaa_aaaa_aaaa_aaaa; (bits[1]:0x0); bits[10]:0x38a"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0x0; (bits[1]:0x1); bits[10]:0x2aa"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1555_5555_5555_5555; (bits[1]:0x0); bits[10]:0x15"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[61]:0xfff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x0"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x1aaa_aaa8_aaaa_2aac; (bits[1]:0x0); bits[10]:0x54"
//     args: "bits[58]:0x200_0000_0000_0000; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x155"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x1); bits[10]:0x1ff"
//     args: "bits[58]:0x0; bits[61]:0x1fff_ffff_ffff_ffff; (bits[1]:0x0); bits[10]:0x3ee"
//     args: "bits[58]:0x5a_2ceb_9611_1be4; bits[61]:0x179_535f_5338_50cd; (bits[1]:0x1); bits[10]:0x80"
//     args: "bits[58]:0x155_5555_5555_5555; bits[61]:0x20_0000; (bits[1]:0x1); bits[10]:0x0"
//   }
// }
// END_CONFIG
type x6 = (u1,);
type x14 = u10;
fn main(x0: s58, x1: s61, x2: (u1,), x3: u10) -> (x6[1], bool, x14) {
  let x4: u10 = bit_slice_update(x3, x3, x3);
  let x5: u10 = for (i, x): (u4, u10) in u4:0..u4:5 {
    x
  }(x4);
  let x7: x6[1] = [x2];
  let x8: x6[1] = array_slice(x7, x5, x6[1]:[(x7)[u32:0], ...]);
  let x9: x6 = (x7)[if (x5) >= (u10:0) { u10:0 } else { x5 }];
  let x10: s10 = s10:0x2aa;
  let x11: s61 = (x1) * (((x0) as s61));
  let x13: u58 = {
    let x12: (u58, u58) = umulp(((x0) as u58), ((x0) as u58));
    (x12.0) + (x12.1)
  };
  let x15: x14[4] = [x3, x4, x5, x4];
  let x16: u10 = -(x3);
  let x17: x14 = (x15)[if (x16) >= (u10:1) { u10:1 } else { x16 }];
  let x18: u10 = (x5) >> (if (x3) >= (u10:8) { u10:8 } else { x3 });
  let x19: s61 = !(x11);
  let x20: bool = (x8) != (x7);
  let x21: x6[1] = update(x7, if (x20) >= (bool:false) { bool:false } else { x20 }, x9);
  let x22: bool = (((x10) as bool)) | (x20);
  let x23: bool = (x22) | (((x18) as bool));
  let x24: u10 = (x16)[x13+:u10];
  let x25: u4 = (x24)[:-6];
  let x26: u58 = clz(x13);
  let x27: s61 = !(x11);
  (x21, x20, x17)
}
