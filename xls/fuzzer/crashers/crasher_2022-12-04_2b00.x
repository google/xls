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

// Exception:
// Same tag is required for a comparison operation: lhs sbits rhs ubits
// (run dir: /tmp/)
// Issue: https://github.com/google/xls/issues/800
//
// options: {"calls_per_sample": 0, "codegen": false, "codegen_args": null, "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "proc_ticks": 128, "simulate": false, "simulator": null, "timeout_seconds": 600, "top_type": 1, "use_jit": true, "use_system_verilog": false}
// ir_channel_names: sample__x5, sample__x14
// args: bits[33]:0x1_5555_5555; bits[9]:0xff
// args: bits[33]:0x1_5555_5555; bits[9]:0xaa
// args: bits[33]:0x1_1eae_ed74; bits[9]:0x70
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x0
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x6e
// args: bits[33]:0x1_ba3b_1267; bits[9]:0xff
// args: bits[33]:0x4_0000; bits[9]:0x1
// args: bits[33]:0xffff_ffff; bits[9]:0xaa
// args: bits[33]:0x1_ffff_ffff; bits[9]:0xff
// args: bits[33]:0x1_ffff_ffff; bits[9]:0xaa
// args: bits[33]:0xffff_ffff; bits[9]:0x0
// args: bits[33]:0xffff_ffff; bits[9]:0x17f
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x1ff
// args: bits[33]:0x10_0000; bits[9]:0xaa
// args: bits[33]:0xffff_ffff; bits[9]:0x8
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x1de
// args: bits[33]:0x1_0000_0000; bits[9]:0x8
// args: bits[33]:0x1_ffff_ffff; bits[9]:0xaa
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x0
// args: bits[33]:0xffff_ffff; bits[9]:0xff
// args: bits[33]:0xffff_ffff; bits[9]:0xaa
// args: bits[33]:0x0; bits[9]:0x8
// args: bits[33]:0x0; bits[9]:0x20
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x142
// args: bits[33]:0x1_5555_5555; bits[9]:0x101
// args: bits[33]:0x40; bits[9]:0x0
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xaa
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x19b
// args: bits[33]:0x0; bits[9]:0x44
// args: bits[33]:0x0; bits[9]:0xff
// args: bits[33]:0x0; bits[9]:0x147
// args: bits[33]:0xffff_ffff; bits[9]:0x1f5
// args: bits[33]:0xa372_ae71; bits[9]:0xaa
// args: bits[33]:0x1_5555_5555; bits[9]:0x0
// args: bits[33]:0x1_5555_5555; bits[9]:0x147
// args: bits[33]:0x1_5555_5555; bits[9]:0x1ff
// args: bits[33]:0x4; bits[9]:0x6
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xb2
// args: bits[33]:0xc2d2_d0d0; bits[9]:0x50
// args: bits[33]:0x0; bits[9]:0xf0
// args: bits[33]:0xa494_bf40; bits[9]:0x140
// args: bits[33]:0x1_5555_5555; bits[9]:0x1dc
// args: bits[33]:0xffff_ffff; bits[9]:0xaa
// args: bits[33]:0x1_5a54_4a83; bits[9]:0x1
// args: bits[33]:0x0; bits[9]:0xaa
// args: bits[33]:0x1_5555_5555; bits[9]:0x155
// args: bits[33]:0x4_0000; bits[9]:0x0
// args: bits[33]:0x1_873b_a753; bits[9]:0x155
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x16a
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x127
// args: bits[33]:0x1_5555_5555; bits[9]:0x0
// args: bits[33]:0x1_0000; bits[9]:0x82
// args: bits[33]:0x1_9f69_59d9; bits[9]:0x1d1
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xe2
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x8b
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xca
// args: bits[33]:0x1_5555_5555; bits[9]:0x0
// args: bits[33]:0x1_ffff_ffff; bits[9]:0xff
// args: bits[33]:0x8; bits[9]:0x100
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x1a5
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x68
// args: bits[33]:0x1_ffff_ffff; bits[9]:0xff
// args: bits[33]:0xffff_ffff; bits[9]:0x155
// args: bits[33]:0x0; bits[9]:0x1ff
// args: bits[33]:0x0; bits[9]:0x42
// args: bits[33]:0x200_0000; bits[9]:0x2
// args: bits[33]:0x1_56a9_e7a6; bits[9]:0x116
// args: bits[33]:0x1_67e5_aec8; bits[9]:0x1ff
// args: bits[33]:0x2d56_a069; bits[9]:0x61
// args: bits[33]:0x161c_751f; bits[9]:0x11e
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x155
// args: bits[33]:0x800_0000; bits[9]:0x198
// args: bits[33]:0x1_5555_5555; bits[9]:0x0
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x1b6
// args: bits[33]:0x800_0000; bits[9]:0x20
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x1ff
// args: bits[33]:0x1_5555_5555; bits[9]:0x1ff
// args: bits[33]:0x0; bits[9]:0x65
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x1ff
// args: bits[33]:0x1_abe1_6af0; bits[9]:0x13e
// args: bits[33]:0x1_caa3_e75c; bits[9]:0x0
// args: bits[33]:0x0; bits[9]:0x1ff
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x1fd
// args: bits[33]:0xd89a_ad54; bits[9]:0x155
// args: bits[33]:0x800_0000; bits[9]:0x0
// args: bits[33]:0x100_0000; bits[9]:0x0
// args: bits[33]:0x10; bits[9]:0x6
// args: bits[33]:0x0; bits[9]:0x40
// args: bits[33]:0x1_5555_5555; bits[9]:0x155
// args: bits[33]:0x400; bits[9]:0x100
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xaa
// args: bits[33]:0xffff_ffff; bits[9]:0xaa
// args: bits[33]:0x1_5555_5555; bits[9]:0x0
// args: bits[33]:0x1_8c7c_6d8b; bits[9]:0xaa
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xaa
// args: bits[33]:0x7630_6559; bits[9]:0xff
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x1bb
// args: bits[33]:0xffff_ffff; bits[9]:0xf6
// args: bits[33]:0x0; bits[9]:0x0
// args: bits[33]:0x79c1_d3a6; bits[9]:0x1b6
// args: bits[33]:0x1; bits[9]:0x40
// args: bits[33]:0x20; bits[9]:0x20
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x9a
// args: bits[33]:0x0; bits[9]:0x0
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xaa
// args: bits[33]:0x0; bits[9]:0xff
// args: bits[33]:0xf837_4e20; bits[9]:0x40
// args: bits[33]:0x1_5555_5555; bits[9]:0x4
// args: bits[33]:0x1_0000_0000; bits[9]:0x74
// args: bits[33]:0x1_cd85_7a68; bits[9]:0x2
// args: bits[33]:0x1_0000_0000; bits[9]:0x163
// args: bits[33]:0x8000_0000; bits[9]:0x179
// args: bits[33]:0x0; bits[9]:0x2
// args: bits[33]:0x4; bits[9]:0x141
// args: bits[33]:0x1_5555_5555; bits[9]:0x175
// args: bits[33]:0xffff_ffff; bits[9]:0x137
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x0
// args: bits[33]:0x1_5555_5555; bits[9]:0xff
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x8
// args: bits[33]:0xffff_ffff; bits[9]:0x0
// args: bits[33]:0x1_5555_5555; bits[9]:0xff
// args: bits[33]:0x8; bits[9]:0x117
// args: bits[33]:0x1_ffff_ffff; bits[9]:0x3e
// args: bits[33]:0x0; bits[9]:0x80
// args: bits[33]:0xaaaa_aaaa; bits[9]:0x4
// args: bits[33]:0x1000; bits[9]:0x112
// args: bits[33]:0xaaaa_aaaa; bits[9]:0xea
// args: bits[33]:0x0; bits[9]:0x4
proc main {
  x5: chan<u33> in;
  x14: chan<s9> in;
  config(x5: chan<u33> in, x14: chan<s9> in) {
    (x5, x14)
  }
  init {
    s57:-48038396025285291
  }
  next(x0: token, x1: s57) {
    let x2: s57 = (x1) * (x1);
    let x3: s57 = -(x2);
    let x4: s57 = (x3) - (x1);
    let x6: (token, u33) = recv_if(x0, x5, bool:true);
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