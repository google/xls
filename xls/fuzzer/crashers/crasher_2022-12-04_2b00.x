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
// (run dir: /tmp/)
//
// BEGIN_CONFIG
// exception: "// Same tag is required for a comparison operation: lhs sbits rhs ubits"
// issue: "https://github.com/google/xls/issues/800"
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
//   timeout_seconds: 600
//   calls_per_sample: 0
//   proc_ticks: 128
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x5"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_1eae_ed74"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_ba3b_1267"
//       values: "bits[33]:0x4_0000"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x10_0000"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_0000_0000"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x40"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0xa372_ae71"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x4"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xc2d2_d0d0"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xa494_bf40"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_5a54_4a83"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x4_0000"
//       values: "bits[33]:0x1_873b_a753"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_0000"
//       values: "bits[33]:0x1_9f69_59d9"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x8"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x200_0000"
//       values: "bits[33]:0x1_56a9_e7a6"
//       values: "bits[33]:0x1_67e5_aec8"
//       values: "bits[33]:0x2d56_a069"
//       values: "bits[33]:0x161c_751f"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x800_0000"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x800_0000"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_abe1_6af0"
//       values: "bits[33]:0x1_caa3_e75c"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0xd89a_ad54"
//       values: "bits[33]:0x800_0000"
//       values: "bits[33]:0x100_0000"
//       values: "bits[33]:0x10"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x400"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_8c7c_6d8b"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x7630_6559"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x79c1_d3a6"
//       values: "bits[33]:0x1"
//       values: "bits[33]:0x20"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xf837_4e20"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_0000_0000"
//       values: "bits[33]:0x1_cd85_7a68"
//       values: "bits[33]:0x1_0000_0000"
//       values: "bits[33]:0x8000_0000"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x4"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x8"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1000"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x0"
//     }
//     inputs {
//       channel_name: "sample__x14"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x70"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x6e"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x1"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x17f"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0x1de"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0x20"
//       values: "bits[9]:0x142"
//       values: "bits[9]:0x101"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x19b"
//       values: "bits[9]:0x44"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x147"
//       values: "bits[9]:0x1f5"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x147"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x6"
//       values: "bits[9]:0xb2"
//       values: "bits[9]:0x50"
//       values: "bits[9]:0xf0"
//       values: "bits[9]:0x140"
//       values: "bits[9]:0x1dc"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x1"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x16a"
//       values: "bits[9]:0x127"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x82"
//       values: "bits[9]:0x1d1"
//       values: "bits[9]:0xe2"
//       values: "bits[9]:0x8b"
//       values: "bits[9]:0xca"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x100"
//       values: "bits[9]:0x1a5"
//       values: "bits[9]:0x68"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x42"
//       values: "bits[9]:0x2"
//       values: "bits[9]:0x116"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x61"
//       values: "bits[9]:0x11e"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x198"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x1b6"
//       values: "bits[9]:0x20"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x65"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x13e"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x1fd"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x6"
//       values: "bits[9]:0x40"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x100"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x1bb"
//       values: "bits[9]:0xf6"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x1b6"
//       values: "bits[9]:0x40"
//       values: "bits[9]:0x20"
//       values: "bits[9]:0x9a"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x40"
//       values: "bits[9]:0x4"
//       values: "bits[9]:0x74"
//       values: "bits[9]:0x2"
//       values: "bits[9]:0x163"
//       values: "bits[9]:0x179"
//       values: "bits[9]:0x2"
//       values: "bits[9]:0x141"
//       values: "bits[9]:0x175"
//       values: "bits[9]:0x137"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x117"
//       values: "bits[9]:0x3e"
//       values: "bits[9]:0x80"
//       values: "bits[9]:0x4"
//       values: "bits[9]:0x112"
//       values: "bits[9]:0xea"
//       values: "bits[9]:0x4"
//     }
//   }
// }
// END_CONFIG
proc main {
  x5: chan<u33> in;
  x14: chan<s9> in;
  config(x5: chan<u33> in, x14: chan<s9> in) {
    (x5, x14)
  }
  init {
    s57:-48038396025285291
  }
  next(x1: s57) {
    let x0: token = join();
    let x2: s57 = (x1) * (x1);
    let x3: s57 = -(x2);
    let x4: s57 = (x3) - (x1);
    let x6: (token, u33) = recv_if(x0, x5, bool:true, u33:0);
    let x7: token = x6.0;
    let x8: u33 = x6.1;
    let x9: u57 = (((x2) as u57))[x8+:u57];
    let x10: u33 = (x8) << (if (x8) >= (u33:19) { u33:19 } else { x8 });
    let x11: u34 = one_hot(x8, bool:false);
    let x12: u34 = ctz(x11);
    let x13: s42 = s42:0x1ff_ffff_ffff;
    let x15: (token, s9) = recv(x0, x14);
    let x16: token = x15.0;
    let x17: s9 = x15.1;
    let x18: u34 = bit_slice_update(x12, x10, x9);
    let x19: u34 = clz(x18);
    let x20: u41 = u41:0xaa_aaaa_aaaa;
    let x21: s12 = s12:0x7ff;
    let x22: u34 = bit_slice_update(x19, x20, x19);
    let x23: bool = (((x21) as s9)) >= (x17);
    let x24: uN[135] = (((x12) ++ (x10)) ++ (x22)) ++ (x22);
    x1
  }
}
