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
// args: bits[61]:0x151e_c3fc_6669_83fa
// evaluated opt IR (JIT) =
//    (bits[61]:0x0, bits[61]:0x0, bits[61]:0xae1_3c03_9996_7c05, bits[61]:0x0, bits[122]:0x1fc_1966_63fc_378a_45fc_1966_63fc_378a, bits[122]:0x1fc_1966_63fc_378a_45fc_1966_63fc_378a)
// evaluated opt IR (interpreter), evaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[61]:0x0, bits[61]:0x0, bits[61]:0xae1_3c03_9996_7c05, bits[61]:0x0, bits[122]:0xbf_832c_cc7f_86f1_45fc_1966_63fc_378a, bits[122]:0xbf_832c_cc7f_86f1_45fc_1966_63fc_378a)
// Issue: https://github.com/google/xls/issues/366
//
// options: {"codegen": true, "codegen_args": ["--use_system_verilog", "--generator=pipeline", "--pipeline_stages=10", "--reset_data_path=false"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "simulate": false, "simulator": null, "use_jit": true, "use_system_verilog": true}
// args: bits[61]:0x151e_c3fc_6669_83fa
// args: bits[61]:0x1709_475e_e806_ac2d
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0xb38_1487_a81f_cda9
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0x8_0000_0000_0000
// args: bits[61]:0x2ce_ad38_3ca2_8656
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x0
// args: bits[61]:0x4000_0000_0000
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x400
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0x0
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x4_0000
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x0
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1000_0000_0000
// args: bits[61]:0x0
// args: bits[61]:0x20
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x1adc_32c0_629b_1396
// args: bits[61]:0x2e7_4c39_0bce_b0a1
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x1922_ccf2_d1dd_97ea
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x8_0000
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0x10
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x100f_a1c0_4adc_eaee
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x1_0000_0000_0000
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x2_0000_0000
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x0
// args: bits[61]:0x2b8_6867_a7ac_6b3e
// args: bits[61]:0x1a97_601b_7438_5b39
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0x1000_0000
// args: bits[61]:0x1c0a_a40e_bae6_6754
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x0
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x4ec_2a89_36bb_45b4
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x800_0000_0000_0000
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x0
// args: bits[61]:0x1062_fa2c_acaf_7564
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x11c8_f274_7e33_dbd1
// args: bits[61]:0x12bf_7f4d_e1f9_2d59
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x4
// args: bits[61]:0xa3b_7f76_bf48_ac49
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x40_0000_0000
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x2_0000_0000_0000
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x200_0000_0000_0000
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x134c_630c_f2b3_06fc
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0xf1_d134_34b6_085f
// args: bits[61]:0x1fff_ffff_ffff_ffff
// args: bits[61]:0x728_26ea_2238_f164
// args: bits[61]:0x4_0000_0000_0000
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x0
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x0
// args: bits[61]:0x80_0000_0000_0000
// args: bits[61]:0x0
// args: bits[61]:0x1555_5555_5555_5555
// args: bits[61]:0x1456_83f8_21ba_8c1c
// args: bits[61]:0x400_0000
// args: bits[61]:0xfff_ffff_ffff_ffff
// args: bits[61]:0xaaa_aaaa_aaaa_aaaa
// args: bits[61]:0x12b0_e273_50e6_3632
// args: bits[61]:0x0
// args: bits[61]:0xb11_5f9d_36e0_4c3d
// args: bits[61]:0x0
// args: bits[61]:0x1555_5555_5555_5555
type x16 = uN[337];
type x20 = bool;
fn main(x0: u61) -> (u61, u61, u61, u61, uN[122], uN[122]) {
  let x1: u61 = (x0) + (x0);
  let x2: uN[122] = (x1) ++ (x1);
  let x3: s20 = s20:0x0;
  let x4: u61 = -(x0);
  let x5: u61 = (x1) * (((x3) as u61));
  let x6: u16 = (x0)[-16:];
  let x7: uN[337] = (((((x2) ++ (x6)) ++ (x4)) ++ (x1)) ++ (x6)) ++ (x1);
  let x8: u41 = u41:0xaa_aaaa_aaaa;
  let x9: u16 = u16:0x0;
  let x10: u61 = !(x0);
  let x11: s11 = s11:0x2;
  let x12: uN[122] = rev(x2);
  let x13: u16 = bit_slice_update(x9, x2, x9);
  let x14: s11 = -(x11);
  let x15: u5 = u5:0xf;
  let x17: x16[1] = [x7];
  let x18: u61 = (x5) & (x1);
  let x19: u61 = -(x4);
  let x21: x20[337] = ((x7) as x20[337]);
  (x5, x18, x10, x18, x12, x12)
}
