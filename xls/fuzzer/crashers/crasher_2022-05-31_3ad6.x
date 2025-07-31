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
// evaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[59]:0x3, bits[28]:0x3, bits[1]:0x0)
// evaluated unopt IR (JIT) =
//    (bits[59]:0x8_0000_0000_0003, bits[28]:0x3, bits[1]:0x0)
//
// BEGIN_CONFIG
// exception: "// Result miscompare for sample 3:"
// issue: "https://github.com/google/xls/issues/631"
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
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0xb_e286_8aaf_2ae8"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0x8"
//     args: "bits[59]:0x2000_0000; bits[52]:0x1_8041_3009_9493"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0xb894_8b71_06b2"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0xb_e286_8aaf_2ae8"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xf_ffff_ffff_fffe"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0xb_a01a_aaba_a828"
//     args: "bits[59]:0x0; bits[52]:0x1_0000_0000_0080"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0xf_ffff_fffd_ffff"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x1_0000_0000"
//     args: "bits[59]:0x8_0000_0000; bits[52]:0x0"
//     args: "bits[59]:0x47c_cea8_8ae7_eec6; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x10_0000_0000_0000; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[59]:0x100; bits[52]:0x8_562f_7904_1e84"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x5_15d5_7854_1500"
//     args: "bits[59]:0x100_0000_0000; bits[52]:0x1_0140_0100_1140"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0xe_8760_ba22_a208"
//     args: "bits[59]:0x0; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x8; bits[52]:0x2b0e_37a9_0200"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0x5_d8ef_290d_3c85"
//     args: "bits[59]:0x0; bits[52]:0x8_0000_0401_0000"
//     args: "bits[59]:0x20_0000; bits[52]:0xc_7400_5228_6c00"
//     args: "bits[59]:0x3b9_eba1_f33b_ab5c; bits[52]:0x9_ebb0_e009_a94e"
//     args: "bits[59]:0x400_0000_0000; bits[52]:0x1_0584_810d_4026"
//     args: "bits[59]:0x6b3_f074_f1be_d7bb; bits[52]:0x3_f013_13bf_b5a2"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x0; bits[52]:0xa08_1008_0804"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x74a_c4a7_f537_9215; bits[52]:0xf_c7a3_6717_b815"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xe_57ab_73dd_97f6"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0x100"
//     args: "bits[59]:0x20_0000_0000; bits[52]:0x20_8000_0000"
//     args: "bits[59]:0x0; bits[52]:0x8_0000_0000_0000"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xd_ff7f_bdfd_ff7f"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0xf_fd7b_7d97_fffc"
//     args: "bits[59]:0x0; bits[52]:0x9_0052_8000_c101"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x7_ffff_ffcf_ffff"
//     args: "bits[59]:0x100_0000; bits[52]:0x400_1900_0805"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xd_e7ff_dffb_6e4f"
//     args: "bits[59]:0x4000_0000; bits[52]:0x100_0000"
//     args: "bits[59]:0x55f_3c6a_72af_1b4e; bits[52]:0x8000_0000_0000"
//     args: "bits[59]:0x0; bits[52]:0xc_9064_fa1d_933c"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[59]:0x0; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x200_0000_0000_0000; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[59]:0x20_0000; bits[52]:0x8_0e60_9d4d_0dd0"
//     args: "bits[59]:0x45d_e5c0_f193_67ef; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x0; bits[52]:0x4_01d2_4280_20d1"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0x7d3d_2a6e_2456"
//     args: "bits[59]:0x589_1aff_c92a_78ef; bits[52]:0x0"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x800_0000"
//     args: "bits[59]:0x40_0000; bits[52]:0x3_5006_0028_c024"
//     args: "bits[59]:0x80_0000_0000; bits[52]:0x80_4100_0000"
//     args: "bits[59]:0x8; bits[52]:0x0"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0xc_befb_5ff6_e5cc"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x1_0000_0000_0000"
//     args: "bits[59]:0x0; bits[52]:0x10_0000_0000"
//     args: "bits[59]:0x0; bits[52]:0x8_1028_8200_0400"
//     args: "bits[59]:0x400_0000_0000_0000; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xf_febf_ffff_ffff"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0x4000_0000_0000"
//     args: "bits[59]:0x0; bits[52]:0xc_b4b8_472b_b566"
//     args: "bits[59]:0x0; bits[52]:0x1_8406_0088_5440"
//     args: "bits[59]:0x0; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0xd_a7ef_7f67_73b0"
//     args: "bits[59]:0x0; bits[52]:0xb_0200_0000_4680"
//     args: "bits[59]:0x58_9d84_d444_2082; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[59]:0x400_0000_0000_0000; bits[52]:0x8_8188_3151_bf31"
//     args: "bits[59]:0x0; bits[52]:0x0"
//     args: "bits[59]:0x8000_0000; bits[52]:0x842_800a_0803"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xe_a3b1_d67c_879a"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0xa_2b5d_4b97_7e44"
//     args: "bits[59]:0x0; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x0"
//     args: "bits[59]:0x3b5_f69b_afa8_900b; bits[52]:0x7_32d3_afad_940b"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xf_fbff_feff_fffb"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0x2_45d7_7e5e_4c6c"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x0; bits[52]:0x4000_0000_0000"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x7_372f_eade_bfbb"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xf_8bf5_7f6f_637f"
//     args: "bits[59]:0x8000_0000; bits[52]:0x4000"
//     args: "bits[59]:0x0; bits[52]:0x3_86a0_0042_8011"
//     args: "bits[59]:0x0; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x80_0000_0000; bits[52]:0xa_8892_0818_4080"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0xb_5d31_935d_4bc3"
//     args: "bits[59]:0x100_0000_0000; bits[52]:0x0"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x5_5dc1_8554_75d5"
//     args: "bits[59]:0x4000; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x397_2bd6_8438_1cfe; bits[52]:0x5_23d6_8238_1c7e"
//     args: "bits[59]:0x0; bits[52]:0x800a_0202_1168"
//     args: "bits[59]:0x20_0000_0000; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x80; bits[52]:0xaa02_1014_1e01"
//     args: "bits[59]:0x691_2587_71fb_0ce8; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x3ff_ffff_ffff_ffff; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[59]:0x100_0000_0000_0000; bits[52]:0x4000_9001_2a90"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x6_dfff_57d3_5efd"
//     args: "bits[59]:0x40_0000_0000_0000; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x5_7ed5_745c_7565"
//     args: "bits[59]:0x20_0000_0000_0000; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xf_bf7f_ffdf_fffa"
//     args: "bits[59]:0x8000_0000_0000; bits[52]:0x0"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x0"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x2000_0000; bits[52]:0x2_e400_0404"
//     args: "bits[59]:0x70d_9e9e_9b5a_029c; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x2_0000"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0x1_c009_cbe3_ace0"
//     args: "bits[59]:0x22c_a9e6_4f10_24b3; bits[52]:0xc_a8d5_0e30_16a3"
//     args: "bits[59]:0x0; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x1_0000_0000_0000; bits[52]:0x1_8882_1400_2002"
//     args: "bits[59]:0x10_0000_0000_0000; bits[52]:0x8_0200_0200_8011"
//     args: "bits[59]:0x555_5555_5555_5555; bits[52]:0x9_7ca8_124a_4388"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[59]:0x0; bits[52]:0x5_5555_5555_5555"
//     args: "bits[59]:0x3f_0717_5968_19b2; bits[52]:0x5_5bf7_7509_83c4"
//     args: "bits[59]:0x200_0000; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xe_9a07_ed27_127d"
//     args: "bits[59]:0x2aa_aaaa_aaaa_aaaa; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[59]:0x0; bits[52]:0x8_080a_30a0_0008"
//     args: "bits[59]:0x7ff_ffff_ffff_ffff; bits[52]:0xe_efbf_f396_addf"
//   }
// }
// END_CONFIG
type x22 = u53;
fn main(x0: s59, x1: s52) -> (s59, u28, bool) {
  let x2: s59 = (x0) ^ (x0);
  let x3: s59 = !(x2);
  let x4: s59 = (x2) & (x0);
  let x5: bool = (x4) > (((x1) as s59));
  let x6: s59 = for (i, x) in u4:0..u4:1 {
    x
  }(x3);
  let x7: u3 = ((x5) ++ (x5)) ++ (x5);
  let x8: u36 = (((x3) as u59))[x7+:u36];
  let x9: u3 = (x7) | (((x3) as u3));
  let x10: u53 = (((x6) as u59))[x7+:u53];
  let x11: u53 = (((x9) as u53)) + (x10);
  let x12: u53 = x10;
  let x13: u36 = (((x12) as u36)) * (x8);
  let x14: bool = and_reduce(x13);
  let x15: u28 = (x11)[x5+:u28];
  let x16: bool = ctz(x14);
  let x17: s52 = (x1) << (if (x12) >= (u53:24) { u53:24 } else { x12 });
  let x18: s59 = (x4) + (((x15) as s59));
  let x19: bool = -(x16);
  let x20: s59 = for (i, x): (u4, s59) in u4:0..u4:4 {
    x
  }(x4);
  let x21: u53 = (x11) & (x11);
  let x23: x22[5] = [x10, x11, x21, x10, x11];
  (x18, x15, x14)
}
