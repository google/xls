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

// Exception:
// Result miscompare for sample 0:
// args: bits[32]:0x0; bits[1]:0x1
// evaluated opt IR (JIT), evaluated unopt IR (JIT) =
//    (bits[32]:0x0, bits[1]:0x1, bits[1]:0x1, bits[5]:0x4, bits[5]:0x0)
// evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[32]:0x0, bits[1]:0x1, bits[1]:0x1, bits[5]:0x4, bits[5]:0x2)
// Issue: https://github.com/google/xls/issues/367
//
// options: {"codegen": true, "codegen_args": ["--use_system_verilog", "--generator=pipeline", "--pipeline_stages=9", "--reset_data_path=false"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "simulate": false, "simulator": null, "use_jit": true, "use_system_verilog": true}
// args: bits[32]:0x0; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x31e5_d272; bits[1]:0x0
// args: bits[32]:0x2484_a3e6; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0xa6aa_87f3; bits[1]:0x1
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x0
// args: bits[32]:0xd33e_052b; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x2; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x6ba6_07b4; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x4_0000; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x0
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x1
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0xf166_98c9; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x425c_c98f; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0x2; bits[1]:0x0
// args: bits[32]:0x0; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0x7302_dd7d; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x2000; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x1
// args: bits[32]:0x2f9d_a3f1; bits[1]:0x0
// args: bits[32]:0xcf4c_917a; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x0
// args: bits[32]:0x1_0000; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x10_0000; bits[1]:0x1
// args: bits[32]:0x857_7fe7; bits[1]:0x1
// args: bits[32]:0x4fb8_eceb; bits[1]:0x0
// args: bits[32]:0x9e3c_8ced; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0xe263_5166; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0x20_0000; bits[1]:0x0
// args: bits[32]:0x4000_0000; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x1
// args: bits[32]:0x100; bits[1]:0x0
// args: bits[32]:0x8000; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x800_0000; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0xa861_ea79; bits[1]:0x1
// args: bits[32]:0x1354_0a96; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x0
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x0
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0xbe4e_4df0; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x5811_b078; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0xaaaa_aaaa; bits[1]:0x0
// args: bits[32]:0x2000_0000; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x0
// args: bits[32]:0x20; bits[1]:0x1
// args: bits[32]:0x5555_5555; bits[1]:0x1
// args: bits[32]:0x2_0000; bits[1]:0x0
// args: bits[32]:0x7fff_ffff; bits[1]:0x0
// args: bits[32]:0x0; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x8; bits[1]:0x0
// args: bits[32]:0x800_0000; bits[1]:0x1
// args: bits[32]:0xffff_ffff; bits[1]:0x1
// args: bits[32]:0x88d8_4e0f; bits[1]:0x0
// args: bits[32]:0x8_0000; bits[1]:0x0
// args: bits[32]:0x400; bits[1]:0x0
// args: bits[32]:0x0; bits[1]:0x0
// args: bits[32]:0x5555_5555; bits[1]:0x1
type x10 = s34;
type x12 = bool;
fn main(x0: s32, x1: u1) -> (s32, u1, bool, u5, u5) {
  let x2: bool = xor_reduce(x1);
  let x3: u5 = u5:0x4;
  let x4: u5 = (x3)[x2+:u5];
  let x5: u1 = bit_slice_update(x1, x2, x3);
  let x6: bool = !(x2);
  let x7: bool = (x2)[x5+:bool];
  let x8: u5 = bit_slice_update(x4, x1, x5);
  let x9: s34 = s34:0x2000;
  let x11: x10[1] = [x9];
  let x13: x12[1] = ((x2) as x12[1]);
  let x14: u5 = (x3) >> (x2);
  let x15: u11 = u11:0x100;
  let x16: bool = -(x2);
  let x17: u5 = one_hot_sel(x7, [x4]);
  let x18: bool = (x6)[x4+:bool];
  (x0, x1, x2, x3, x8)
}
