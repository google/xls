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
// Command '['/xls/tools/codegen_main', '--output_signature_path=module_sig.textproto', '--delay_model=unit', '--use_system_verilog', '--generator=pipeline', '--pipeline_stages=7', '--reset_data_path=false', 'sample.opt.ir', '--logtostderr']' returned non-zero exit status 1.
// Issue: https://github.com/google/xls/issues/346
//
// options: {"codegen": true, "codegen_args": ["--use_system_verilog", "--generator=pipeline", "--pipeline_stages=7", "--reset_data_path=false"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "simulate": false, "simulator": null, "use_jit": true, "use_system_verilog": true}
// args: bits[29]:0x1654_4d1f; (); bits[53]:0x1f_ffff_ffff_ffff; bits[43]:0x2f3_aee0_d52a; bits[51]:0x2_d66d_727f_0f9e
// args: bits[29]:0x0; (); bits[53]:0xa_aaaa_aaaa_aaaa; bits[43]:0x2aa_aaaa_aaaa; bits[51]:0x0
// args: bits[29]:0x2; (); bits[53]:0x10_4118_de04_dd7d; bits[43]:0x3ff_ffff_ffff; bits[51]:0x0
// args: bits[29]:0xfff_ffff; (); bits[53]:0x1f_ffff_ffff_ffff; bits[43]:0x2000; bits[51]:0x5_5555_5555_5555
// args: bits[29]:0x0; (); bits[53]:0x1_50c9_03cd_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[51]:0x1_50c9_43cd_ffff
// args: bits[29]:0x1821_edfe; (); bits[53]:0x100_0000_0000; bits[43]:0x5cd_f7ff_bef3; bits[51]:0x0
// args: bits[29]:0xfff_ffff; (); bits[53]:0x8_0000_0000; bits[43]:0x555_5555_5555; bits[51]:0x2000_0000_0000
// args: bits[29]:0xbfe_ff62; (); bits[53]:0x14_3a46_5b18_dddf; bits[43]:0x26e_4b18_ddcf; bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[29]:0xe8b_44a7; (); bits[53]:0xa_aaaa_aaaa_aaaa; bits[43]:0x38b_5969_2cea; bits[51]:0x2_0000_0000_0000
// args: bits[29]:0x0; (); bits[53]:0x1410_2010_ca19; bits[43]:0x3ff_ffff_ffff; bits[51]:0x440_4481_8e10
type x19 = s53;
fn main(x0: s29, x1: (), x2: s53, x3: u43, x4: s51) -> ((s53, u43, u18), (), u43, u18, u43) {
  let x5: s51 = (x4) & (x4);
  let x6: s53 = (((x3) as s53)) ^ (x2);
  let x7: u18 = (x4 as u51)[x3+:u18];
  let x8: s29 = (x0) & (((x6) as s29));
  let x9: u43 = (((x6) as u43)) + (x3);
  let x10: s51 = (x4) | (((x6) as s51));
  let x11: u43 = !(x3);
  let x12: bool = (x2) <= (x2);
  let x13: (s53, u43, u18) = (x2, x9, x7);
  let x14: bool = (x8) <= (((x12) as s29));
  let x15: bool = (((x12) as u43)) > (x11);
  let x16: u43 = bit_slice_update(x11, x12, x9);
  let x17: s51 = (x10) & (((x9) as s51));
  let x18: bool = (x15) & (((x7) as bool));
  let x20: x19[2] = [x2, x6];
  let x21: uN[104] = ((x7) ++ (x11)) ++ (x16);
  let x22: bool = (((x16) as bool)) + (x18);
  (x13, x1, x9, x7, x9)
}
