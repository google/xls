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

// options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "use_jit": true, "codegen": true, "codegen_args": ["--use_system_verilog", "--generator=pipeline", "--pipeline_stages=1"], "simulate": false, "simulator": null, "use_system_verilog": true}
// args: bits[26]:0x2_0000
// args: bits[26]:0x384_b888
// args: bits[26]:0x1_0000
// args: bits[26]:0x40_0000
// args: bits[26]:0x20e_0b98
// args: bits[26]:0x10_0000
// args: bits[26]:0x3ff_ffff
// args: bits[26]:0x80
// args: bits[26]:0x1
// args: bits[26]:0x2000
// args: bits[26]:0x13b_5724
// args: bits[26]:0x1_0000
// args: bits[26]:0x1ff_ffff
// args: bits[26]:0x10_0000
// args: bits[26]:0xd2_bba9
// args: bits[26]:0xf8_f3d7
// args: bits[26]:0x10_0000
// args: bits[26]:0x400
// args: bits[26]:0x20
// args: bits[26]:0x100
// args: bits[26]:0x10
// args: bits[26]:0x40
// args: bits[26]:0x1ff_ffff
// args: bits[26]:0x8
// args: bits[26]:0x200_0000
// args: bits[26]:0x80
// args: bits[26]:0x80_0000
// args: bits[26]:0x10_0000
// args: bits[26]:0x80
// args: bits[26]:0x10
// args: bits[26]:0x200_0000
// args: bits[26]:0x2
// args: bits[26]:0x20_0000
// args: bits[26]:0x8000
// args: bits[26]:0x80_0000
// args: bits[26]:0x100_0000
// args: bits[26]:0x40
// args: bits[26]:0x1_0000
// args: bits[26]:0x10
// args: bits[26]:0x1
// args: bits[26]:0x20_0000
// args: bits[26]:0x8_0000
// args: bits[26]:0x10
// args: bits[26]:0x0
// args: bits[26]:0x800
// args: bits[26]:0x1ff_ffff
// args: bits[26]:0x0
// args: bits[26]:0x251_b06a
// args: bits[26]:0x4_0000
// args: bits[26]:0x100
// args: bits[26]:0x100_0000
// args: bits[26]:0x2
// args: bits[26]:0x20_0000
// args: bits[26]:0x2000
// args: bits[26]:0x200_0000
// args: bits[26]:0x1ff_ffff
// args: bits[26]:0x20
// args: bits[26]:0x200
// args: bits[26]:0x2
// args: bits[26]:0x2aa_aaaa
// args: bits[26]:0x1ff_ffff
// args: bits[26]:0x80_0000
// args: bits[26]:0x10
// args: bits[26]:0x155_5555
// args: bits[26]:0x100
// args: bits[26]:0x2
// args: bits[26]:0xe8_cfb5
// args: bits[26]:0x800
// args: bits[26]:0x2000
// args: bits[26]:0x40_0000
// args: bits[26]:0x155_5555
// args: bits[26]:0x1
// args: bits[26]:0x1000
// args: bits[26]:0x8_0000
// args: bits[26]:0x8000
// args: bits[26]:0x3ff_ffff
// args: bits[26]:0x4
// args: bits[26]:0x80
// args: bits[26]:0x8
// args: bits[26]:0x1
// args: bits[26]:0x80_0000
// args: bits[26]:0x2_0000
// args: bits[26]:0x8
// args: bits[26]:0x268_c9a3
// args: bits[26]:0x20
// args: bits[26]:0x2
// args: bits[26]:0x33e_ecc6
// args: bits[26]:0x20
// args: bits[26]:0x200_0000
// args: bits[26]:0x4_0000
// args: bits[26]:0x1000
// args: bits[26]:0x47_1362
// args: bits[26]:0x10_0000
// args: bits[26]:0x270_8819
// args: bits[26]:0x100_0000
// args: bits[26]:0x10_0000
// args: bits[26]:0x3ff_ffff
// args: bits[26]:0x80_0000
// args: bits[26]:0x1ff_ffff
// args: bits[26]:0x2aa_aaaa
// args: bits[26]:0x1ff_ffff
// args: bits[26]:0x8000
// args: bits[26]:0x8_0000
// args: bits[26]:0x400
// args: bits[26]:0x29b_b8f4
// args: bits[26]:0x2ea_34b0
// args: bits[26]:0x4
// args: bits[26]:0x200
// args: bits[26]:0x1
// args: bits[26]:0x2000
// args: bits[26]:0x2aa_aaaa
// args: bits[26]:0x2
// args: bits[26]:0x80_0000
// args: bits[26]:0x20_0000
// args: bits[26]:0x400
// args: bits[26]:0x100
// args: bits[26]:0x100
// args: bits[26]:0x2
// args: bits[26]:0x80
// args: bits[26]:0x40
// args: bits[26]:0x2aa_aaaa
// args: bits[26]:0x4000
// args: bits[26]:0x100
// args: bits[26]:0x4_0000
// args: bits[26]:0x4000
// args: bits[26]:0x40_0000
// args: bits[26]:0x800
// args: bits[26]:0x24e_956a
const W32_V1 = u32:0x1;
const W32_V3 = u32:0x3;
type x8 = u1;
type x19 = u3;
type x21 = u9;
type x26 = u3;
fn main(x0: s26) -> (u2, u3, u3, u3, x21[0x3]) {
  let x1: u26 = (x0)[0x0+:u26];
  let x2: u27 = one_hot(x1, u1:0x1);
  let x3: u3 = u3:0x1;
  let x4: u3 = (x3)[x1+:u3];
  let x5: u3 = !(x3);
  let x6: u7 = (x2)[0x5+:u7];
  let x7: x8[W32_V3] = ((x4) as x8[W32_V3]);
  let x9: s37 = s37:0x800;
  let x10: u24 = u24:0x200000;
  let x11: u27 = rev(x2);
  let x12: u3 = (((x0) as u3)) | (x5);
  let x13: u3 = clz(x4);
  let x14: u3 = one_hot_sel(x3, [x12, x12, x12]);
  let x15: (u3, u7, u24, u24, s26, u3, u27, u3, u3, u3, u3, u3, u24, u3) = (x3, x6, x10, x10, x0, x3, x2, x5, x4, x12, x14, x4, x10, x12);
  let x16: u2 = (x13)[0x1+:u2];
  let x17: u13 = ((x13) ++ (x6)) ++ (x5);
  let x18: x19[0x1] = ((x4) as x19[0x1]);
  let x20: x21[0x3] = ((x11) as x21[0x3]);
  let x22: u2 = (x16)[:];
  let x23: u37 = ((x6) ++ (x13)) ++ (x11);
  let x24: u3 = (x14)[0x0+:u3];
  let x25: x26[W32_V1] = ((x3) as x26[W32_V1]);
  let x27: u3 = (x14)[0x0+:u3];
  (x22, x12, x4, x12, x20)
}
