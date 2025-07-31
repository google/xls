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
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=7"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[53]:0x80"
//     args: "bits[53]:0x8000_0000"
//     args: "bits[53]:0x4_15ba_2a20_f5b1"
//     args: "bits[53]:0xf_ffff_ffff_ffff"
//     args: "bits[53]:0x800_0000"
//     args: "bits[53]:0x200_0000_0000"
//     args: "bits[53]:0xf_ffff_ffff_ffff"
//     args: "bits[53]:0x2000_0000_0000"
//     args: "bits[53]:0x11_09bc_6086_c58d"
//     args: "bits[53]:0x2_0000_0000_0000"
//     args: "bits[53]:0x0"
//     args: "bits[53]:0x1000"
//     args: "bits[53]:0x4000"
//     args: "bits[53]:0x2_0000_0000_0000"
//     args: "bits[53]:0x400_0000"
//     args: "bits[53]:0x4_0000"
//     args: "bits[53]:0x2000_0000"
//     args: "bits[53]:0x20_0000"
//     args: "bits[53]:0x1000_0000"
//     args: "bits[53]:0x4_0000"
//     args: "bits[53]:0x40_0000"
//     args: "bits[53]:0x1000"
//     args: "bits[53]:0x400_0000"
//     args: "bits[53]:0x13_9d18_b2a0_00c7"
//     args: "bits[53]:0x8_0000_0000"
//     args: "bits[53]:0xf_ffff_ffff_ffff"
//     args: "bits[53]:0x40_0000_0000"
//     args: "bits[53]:0x4_0000_0000_0000"
//     args: "bits[53]:0x2_0000_0000_0000"
//     args: "bits[53]:0x1f_ffff_ffff_ffff"
//     args: "bits[53]:0x4_0000"
//     args: "bits[53]:0x800"
//     args: "bits[53]:0x80_0000"
//     args: "bits[53]:0x1000_0000"
//     args: "bits[53]:0x80"
//     args: "bits[53]:0x4_0000"
//     args: "bits[53]:0x1000_0000"
//     args: "bits[53]:0x80_0000_0000"
//     args: "bits[53]:0x0"
//     args: "bits[53]:0x20_0000_0000"
//     args: "bits[53]:0x0"
//     args: "bits[53]:0x2"
//     args: "bits[53]:0x1_0000_0000"
//     args: "bits[53]:0x4_0000_0000"
//     args: "bits[53]:0x4_fbdf_f631_fc0b"
//     args: "bits[53]:0x4"
//     args: "bits[53]:0xc_867a_6c65_42fa"
//     args: "bits[53]:0x4"
//     args: "bits[53]:0x15_5555_5555_5555"
//     args: "bits[53]:0x800_0000"
//     args: "bits[53]:0x8_0000_0000_0000"
//     args: "bits[53]:0x2000"
//     args: "bits[53]:0x8_e742_8259_a01b"
//     args: "bits[53]:0x2000_0000_0000"
//     args: "bits[53]:0x800"
//     args: "bits[53]:0x80"
//     args: "bits[53]:0x1"
//     args: "bits[53]:0x2_0000_0000"
//     args: "bits[53]:0x1_0000_0000"
//     args: "bits[53]:0x80_0000"
//     args: "bits[53]:0x800"
//     args: "bits[53]:0x1f_ffff_ffff_ffff"
//     args: "bits[53]:0x40"
//     args: "bits[53]:0x4_0000"
//     args: "bits[53]:0x800"
//     args: "bits[53]:0x15_5555_5555_5555"
//     args: "bits[53]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[53]:0x1_0000_0000"
//     args: "bits[53]:0x8000"
//     args: "bits[53]:0x400_0000"
//     args: "bits[53]:0x8000_0000_0000"
//     args: "bits[53]:0x80_0000"
//     args: "bits[53]:0x200_0000_0000"
//     args: "bits[53]:0x400"
//     args: "bits[53]:0x2_0000_0000_0000"
//     args: "bits[53]:0x80_0000_0000"
//     args: "bits[53]:0x8"
//     args: "bits[53]:0xb_bba7_de74_af96"
//     args: "bits[53]:0x2"
//     args: "bits[53]:0x1000"
//     args: "bits[53]:0x1000_0000"
//     args: "bits[53]:0x2_0000_0000_0000"
//     args: "bits[53]:0x4"
//     args: "bits[53]:0x8000"
//     args: "bits[53]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[53]:0x2_0000_0000_0000"
//     args: "bits[53]:0x15_f478_cc3b_8bb2"
//     args: "bits[53]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[53]:0x8000_0000_0000"
//     args: "bits[53]:0x2"
//     args: "bits[53]:0x4_0000"
//     args: "bits[53]:0x14_eed4_2b7d_74b9"
//     args: "bits[53]:0x800_0000_0000"
//     args: "bits[53]:0x100_0000"
//     args: "bits[53]:0x400_0000_0000"
//     args: "bits[53]:0x2_0000_0000_0000"
//     args: "bits[53]:0x4000"
//     args: "bits[53]:0x2"
//     args: "bits[53]:0x20"
//     args: "bits[53]:0x2"
//     args: "bits[53]:0x1"
//     args: "bits[53]:0x40"
//     args: "bits[53]:0x80"
//     args: "bits[53]:0xc_c729_aee5_695a"
//     args: "bits[53]:0x1000_0000"
//     args: "bits[53]:0x1_0000_0000"
//     args: "bits[53]:0x2_0000_0000"
//     args: "bits[53]:0x800_0000_0000"
//     args: "bits[53]:0x2_0000"
//     args: "bits[53]:0x400"
//     args: "bits[53]:0x1_0000"
//     args: "bits[53]:0x2_0000"
//     args: "bits[53]:0x1000_0000"
//     args: "bits[53]:0x8000"
//     args: "bits[53]:0x10"
//     args: "bits[53]:0x1_0000_0000_0000"
//     args: "bits[53]:0x2_0000"
//     args: "bits[53]:0x4000"
//     args: "bits[53]:0x17_cb33_aa3b_12a6"
//     args: "bits[53]:0x20"
//     args: "bits[53]:0x800_0000_0000"
//     args: "bits[53]:0x10"
//     args: "bits[53]:0x1000_0000_0000"
//     args: "bits[53]:0x800"
//     args: "bits[53]:0x8000_0000_0000"
//     args: "bits[53]:0x40_0000"
//     args: "bits[53]:0x2"
//     args: "bits[53]:0x400"
//   }
// }
// END_CONFIG
const W32_V1 = u32:0x1;
type x19 = u1;
fn main(x0: s53) -> (bool, s53, (s53, s53), u1, s53, s53, u1, s53, u1, (s53, s53)) {
  let x1: u1 = (x0) == (x0);
  let x2: (s53, s53) = (x0, x0);
  let x3: u1 = (x1)[0x0+:u1];
  let x4: u1 = (x3) - (x3);
  let x5: bool = bool:true;
  let x6: u1 = !(x3);
  let x7: s16 = s16:0x8;
  let x8: s16 = (x7) >> (((x5) as u16));
  let x9: u1 = for (i, x): (u4, u1) in u4:0x0..u4:0x6 {
    x
  }(x3);
  let x10: u1 = one_hot_sel(x1, [x3]);
  let x11: (s53, s53) = for (i, x): (u4, (s53, s53)) in u4:0x0..u4:0x5 {
    x
  }(x2);
  let x12: s53 = one_hot_sel(x6, [x0]);
  let x13: u4 = u4:0xa;
  let x14: u1 = (((x10) as u1)) + (x1);
  let x15: s53 = one_hot_sel(x6, [x12]);
  let x16: s53 = (x11).0x1;
  let x17: s53 = -(x15);
  let x18: x19[W32_V1] = ((x10) as x19[W32_V1]);
  let x20: bool = (x5) >> (if ((((x8) as bool)) >= (bool:false)) { (bool:false) } else { (((x8) as bool)) });
  let x21: u1 = (x14)[:];
  (x5, x17, x2, x3, x0, x0, x10, x0, x4, x2)
}
