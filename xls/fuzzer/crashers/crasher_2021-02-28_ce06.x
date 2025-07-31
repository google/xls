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
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=10"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[60]:0x10_0000; bits[18]:0x100"
//     args: "bits[60]:0x275_d89f_34df_a01a; bits[18]:0x3_a03a"
//     args: "bits[60]:0x80_0000; bits[18]:0x92"
//     args: "bits[60]:0x2_0000_0000_0000; bits[18]:0x1_5555"
//     args: "bits[60]:0x4_0000; bits[18]:0x1000"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[18]:0x10"
//     args: "bits[60]:0x555_5555_5555_5555; bits[18]:0x1_5135"
//     args: "bits[60]:0x200; bits[18]:0x8"
//     args: "bits[60]:0x2_0000; bits[18]:0x3_ffff"
//     args: "bits[60]:0x8000_0000; bits[18]:0x2_7175"
//     args: "bits[60]:0x39d_61fa_f017_d2ad; bits[18]:0x4000"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[18]:0x800"
//     args: "bits[60]:0x20_0000_0000_0000; bits[18]:0x40"
//     args: "bits[60]:0x80_0000; bits[18]:0x1_70d8"
//     args: "bits[60]:0x10_0000; bits[18]:0x4c2"
//     args: "bits[60]:0x8000_0000_0000; bits[18]:0x1_125e"
//     args: "bits[60]:0x800_0000_0000; bits[18]:0x3_ffff"
//     args: "bits[60]:0xea6_f76a_d9b4_a208; bits[18]:0x1000"
//     args: "bits[60]:0x2_0000; bits[18]:0x3_0020"
//     args: "bits[60]:0x400_0000; bits[18]:0x2"
//     args: "bits[60]:0x10_0000_0000; bits[18]:0x2000"
//     args: "bits[60]:0x40_0000_0000_0000; bits[18]:0x8102"
//     args: "bits[60]:0x80_0000_0000; bits[18]:0x0"
//     args: "bits[60]:0x20; bits[18]:0x7034"
//     args: "bits[60]:0x555_5555_5555_5555; bits[18]:0x1_d975"
//     args: "bits[60]:0x4_0000; bits[18]:0xd82e"
//     args: "bits[60]:0x1000; bits[18]:0x4060"
//     args: "bits[60]:0x67e_bc9a_2de9_eeb0; bits[18]:0x10"
//     args: "bits[60]:0x10_0000_0000; bits[18]:0x2824"
//     args: "bits[60]:0x400_0000; bits[18]:0x1_ffff"
//     args: "bits[60]:0x2000; bits[18]:0x2001"
//     args: "bits[60]:0x1_0000_0000; bits[18]:0xc00"
//     args: "bits[60]:0x400_0000_0000; bits[18]:0x10"
//     args: "bits[60]:0x200_0000_0000; bits[18]:0x3_8efa"
//     args: "bits[60]:0x1000_0000; bits[18]:0x1_5555"
//     args: "bits[60]:0x2000_0000; bits[18]:0x7430"
//     args: "bits[60]:0x400_0000_0000_0000; bits[18]:0xb6c6"
//     args: "bits[60]:0x1_0000; bits[18]:0x1"
//     args: "bits[60]:0x90c_1a70_19c8_24eb; bits[18]:0xceb"
//     args: "bits[60]:0x200_0000; bits[18]:0x20"
//     args: "bits[60]:0x800_0000; bits[18]:0x1004"
//     args: "bits[60]:0x2000_0000_0000; bits[18]:0x8000"
//     args: "bits[60]:0x1000_0000_0000; bits[18]:0x0"
//     args: "bits[60]:0x40_0000_0000; bits[18]:0x1000"
//     args: "bits[60]:0x10; bits[18]:0x4290"
//     args: "bits[60]:0x2_0000_0000; bits[18]:0x0"
//     args: "bits[60]:0x919_40a7_da21_14fb; bits[18]:0x1_19fa"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[18]:0x3_bedf"
//     args: "bits[60]:0x555_5555_5555_5555; bits[18]:0x2000"
//     args: "bits[60]:0x4_0000; bits[18]:0x0"
//     args: "bits[60]:0x80_0000_0000_0000; bits[18]:0x850"
//     args: "bits[60]:0x10_0000_0000_0000; bits[18]:0x3_ffff"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[18]:0x3_cdbf"
//     args: "bits[60]:0x8; bits[18]:0x8"
//     args: "bits[60]:0x8000_0000; bits[18]:0x2_e72b"
//     args: "bits[60]:0x200_0000_0000_0000; bits[18]:0x2_0800"
//     args: "bits[60]:0x40_0000_0000_0000; bits[18]:0x1_8222"
//     args: "bits[60]:0x80_0000_0000; bits[18]:0x8"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[18]:0x3_eafe"
//     args: "bits[60]:0x580_7e8b_5df0_4ece; bits[18]:0x4ece"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[18]:0x80"
//     args: "bits[60]:0x80_0000_0000_0000; bits[18]:0xa468"
//     args: "bits[60]:0x40; bits[18]:0x1_c01b"
//     args: "bits[60]:0x800_0000; bits[18]:0x200"
//     args: "bits[60]:0x2_0000; bits[18]:0x1_ffff"
//     args: "bits[60]:0x40; bits[18]:0x2_aaaa"
//     args: "bits[60]:0x1000_0000_0000; bits[18]:0x1000"
//     args: "bits[60]:0x100_0000; bits[18]:0x2100"
//     args: "bits[60]:0x8_0000; bits[18]:0x4bc9"
//     args: "bits[60]:0x200_0000_0000; bits[18]:0x3_ffff"
//     args: "bits[60]:0x1; bits[18]:0x2_3a6d"
//     args: "bits[60]:0x80_0000_0000; bits[18]:0x2"
//     args: "bits[60]:0x1000_0000; bits[18]:0x0"
//     args: "bits[60]:0x4_0000_0000_0000; bits[18]:0x1_6000"
//     args: "bits[60]:0x40; bits[18]:0x8000"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[18]:0x1_67fd"
//     args: "bits[60]:0x8_0000_0000; bits[18]:0x1"
//     args: "bits[60]:0x2000_0000; bits[18]:0x1_4e48"
//     args: "bits[60]:0x4000_0000; bits[18]:0x2_aaaa"
//     args: "bits[60]:0xe2_f698_ef8d_379a; bits[18]:0x338e"
//     args: "bits[60]:0x100_0000; bits[18]:0x91e4"
//     args: "bits[60]:0x400_0000; bits[18]:0x2_f780"
//     args: "bits[60]:0x8000_0000_0000; bits[18]:0x100"
//     args: "bits[60]:0x2; bits[18]:0x3_ffff"
//     args: "bits[60]:0x1000_0000_0000; bits[18]:0x0"
//     args: "bits[60]:0x10; bits[18]:0x1_c57c"
//     args: "bits[60]:0x2000_0000_0000; bits[18]:0x10"
//     args: "bits[60]:0x40; bits[18]:0x1ba2"
//     args: "bits[60]:0x8; bits[18]:0x2389"
//     args: "bits[60]:0x100; bits[18]:0x2_05b6"
//     args: "bits[60]:0x400; bits[18]:0x24c4"
//     args: "bits[60]:0x453_dee4_231a_8843; bits[18]:0x400"
//     args: "bits[60]:0xbe_916b_46e7_5334; bits[18]:0x2_4354"
//     args: "bits[60]:0x4_0000_0000_0000; bits[18]:0x4000"
//     args: "bits[60]:0x4_0000_0000_0000; bits[18]:0x2_1586"
//     args: "bits[60]:0x4_0000; bits[18]:0x8"
//     args: "bits[60]:0x1000_0000_0000; bits[18]:0x58"
//     args: "bits[60]:0x20_0000_0000; bits[18]:0x40"
//     args: "bits[60]:0x10_0000; bits[18]:0x3_8a13"
//     args: "bits[60]:0x20; bits[18]:0x100"
//     args: "bits[60]:0x40_0000_0000_0000; bits[18]:0x1"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[18]:0x400"
//     args: "bits[60]:0x10_0000_0000_0000; bits[18]:0x2"
//     args: "bits[60]:0x20_0000; bits[18]:0x100"
//     args: "bits[60]:0x4000_0000_0000; bits[18]:0x82"
//     args: "bits[60]:0x10; bits[18]:0x10"
//     args: "bits[60]:0x10_0000; bits[18]:0x2_2d30"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[18]:0x2_a8aa"
//     args: "bits[60]:0x80_0000_0000; bits[18]:0x200"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[18]:0xb2ac"
//     args: "bits[60]:0x61e_0876_2f27_a309; bits[18]:0x3_8200"
//     args: "bits[60]:0x20; bits[18]:0xa415"
//     args: "bits[60]:0x1_0000_0000_0000; bits[18]:0x3_ffff"
//     args: "bits[60]:0x200; bits[18]:0x3_a614"
//     args: "bits[60]:0x800_0000_0000_0000; bits[18]:0x1_1780"
//     args: "bits[60]:0x100; bits[18]:0xc100"
//     args: "bits[60]:0x2_0000_0000_0000; bits[18]:0x8000"
//     args: "bits[60]:0x100_0000_0000; bits[18]:0x1_6300"
//     args: "bits[60]:0x40_0000; bits[18]:0x400"
//     args: "bits[60]:0x4_0000; bits[18]:0x20"
//     args: "bits[60]:0x200; bits[18]:0x3_a957"
//     args: "bits[60]:0x1_0000; bits[18]:0x400"
//     args: "bits[60]:0x9c2_949f_b8e9_5130; bits[18]:0x3_1724"
//     args: "bits[60]:0x8000_0000; bits[18]:0x0"
//     args: "bits[60]:0x8000_0000; bits[18]:0x100"
//     args: "bits[60]:0x400_0000_0000; bits[18]:0x0"
//     args: "bits[60]:0x1000; bits[18]:0x2_1040"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[18]:0x1_5c22"
//   }
// }
// END_CONFIG
const W1_V1 = u32:0x1;
const W2_V2 = u32:0x2;
type x7 = u1;
type x13 = u10;
fn main(x0: s60, x1: s18) -> (u12, u43, u12, u43, u7, u20, u58, u43, s47, u26, x13[W2_V2], u40, u58, u43, s47, u43, u26, (s60,), s18, u40, u43) {
  let x2: u40 = u40:0x4000000000;
  let x3: s47 = s47:0x100000000;
  let x4: u1 = (x0 as u60)[0xe+:u1];
  let x5: u43 = (((x4) ++ (x2)) ++ (x4)) ++ (x4);
  let x6: x7[W1_V1] = ((x4) as x7[W1_V1]);
  let x8: u20 = (x5)[:0x14];
  let x9: u43 = -(x5);
  let x10: u18 = u18:0x80;
  let x11: u43 = for (i, x): (u4, u43) in u4:0x0..u4:0x0 {
    x
  }(x5);
  let x12: x13[W2_V2] = ((x8) as x13[W2_V2]);
  let x14: u40 = x2;
  let x15: u7 = (x5)[x4+:u7];
  let x16: (s60,) = (x0,);
  let x17: u43 = u43:0x10000000000;
  let x18: u38 = (x3 as u47)[0x9+:u38];
  let x19: u43 = (x5)[:];
  let x20: u58 = (x18) ++ (x8);
  let x21: u26 = u26:0x10;
  let x22: s18 = -(x1);
  let x23: u43 = u43:0x7ffffffffff;
  let x24: u58 = (x20) - (((x5) as u58));
  let x25: u5 = (x24)[:-0x35];
  let x26: u12 = u12:0x20;
  let x27: u12 = !(x26);
  let x28: u1 = (x4)[0x0+:u1];
  let x29: u38 = rev(x18);
  let x30: u58 = rev(x24);
  let x31: u43 = clz(x5);
  (x26, x31, x27, x5, x15, x8, x30, x9, x3, x21, x12, x14, x20, x31, x3, x5, x21, x16, x22, x2, x5)
}
