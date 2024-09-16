// Copyright 2024 The XLS Authors
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
// # proto-message: xls.fuzzer.CrasherConfigurationProto
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[34]:0x1_d8ab_0dbb; bits[26]:0x40_0000; bits[3]:0x4\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[51]:0x3_b156_1b77_bfff, (bits[1]:0x1, bits[17]:0x1_5555), bits[1]:0x1, bits[34]:0x1_d8ab_0dbb, bits[51]:0x3_b156_1b77_bfff, bits[1]:0x1)\nevaluated unopt IR (JIT) =\n   (bits[51]:0x6_3156_1b77_bfff, (bits[1]:0x1, bits[17]:0x1_5555), bits[1]:0x1, bits[34]:0x1_d8ab_0dbb, bits[51]:0x6_3156_1b77_bfff, bits[1]:0x1)"
// issue: "https://github.com/google/xls/issues/1609"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 128
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full throughput.*"
//   }
// }
// inputs {
//   function_args {
//     args: "bits[34]:0x1_d8ab_0dbb; bits[26]:0x40_0000; bits[3]:0x4"
//     args: "bits[34]:0x0; bits[26]:0x3ff_ffff; bits[3]:0x4"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x3a4_d9d2; bits[3]:0x2"
//     args: "bits[34]:0x1000; bits[26]:0x1000; bits[3]:0x1"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x3ff_ffff; bits[3]:0x2"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x3df_3feb; bits[3]:0x3"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x35c_5311; bits[3]:0x4"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x3ff_fefb; bits[3]:0x5"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x3ff_ffff; bits[3]:0x0"
//     args: "bits[34]:0x8000; bits[26]:0x3ff_ffff; bits[3]:0x1"
//     args: "bits[34]:0x0; bits[26]:0x80; bits[3]:0x1"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2aa_aaaa; bits[3]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x37f_f57f; bits[3]:0x7"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x355_7147; bits[3]:0x5"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x71_3355; bits[3]:0x5"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x1_0000; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x0; bits[3]:0x0"
//     args: "bits[34]:0x20_0000; bits[26]:0x28_0100; bits[3]:0x2"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x155_7555; bits[3]:0x5"
//     args: "bits[34]:0x8; bits[26]:0x6c_074c; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x5_1008; bits[3]:0x6"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x200; bits[3]:0x5"
//     args: "bits[34]:0x0; bits[26]:0x11e_df59; bits[3]:0x1"
//     args: "bits[34]:0x2_0b82_3e7a; bits[26]:0x1ff_ffff; bits[3]:0x7"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x3d1_fb45; bits[3]:0x5"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x37a_fdff; bits[3]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2aa_aa88; bits[3]:0x5"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x3fe_f7fe; bits[3]:0x3"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x160_24bb; bits[3]:0x4"
//     args: "bits[34]:0x0; bits[26]:0x190_0019; bits[3]:0x1"
//     args: "bits[34]:0x40; bits[26]:0x40; bits[3]:0x2"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2aa_aaaa; bits[3]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x3ff_ffbe; bits[3]:0x4"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x3ff_ffff; bits[3]:0x4"
//     args: "bits[34]:0x1_6b58_7411; bits[26]:0x378_7611; bits[3]:0x1"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1ff_ffff; bits[3]:0x6"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1ff_ffff; bits[3]:0x2"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1ff_ffff; bits[3]:0x4"
//     args: "bits[34]:0x1_056c_6c79; bits[26]:0x3a8_4c69; bits[3]:0x5"
//     args: "bits[34]:0x1_232a_ef36; bits[26]:0x2aa_aaaa; bits[3]:0x4"
//     args: "bits[34]:0x1_a03f_d067; bits[26]:0x3ff_ffff; bits[3]:0x2"
//     args: "bits[34]:0x2_0000_0000; bits[26]:0x3ff_ffff; bits[3]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x3ff_ffed; bits[3]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x3f3_3a0e; bits[3]:0x7"
//     args: "bits[34]:0x3_5e27_0de1; bits[26]:0x1ff_ffff; bits[3]:0x6"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x100_0000; bits[3]:0x1"
//     args: "bits[34]:0x2_0000; bits[26]:0x207_0208; bits[3]:0x1"
//     args: "bits[34]:0x1_00a7_0acf; bits[26]:0x0; bits[3]:0x0"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x155_5555; bits[3]:0x4"
//     args: "bits[34]:0x1000_0000; bits[26]:0x41_0000; bits[3]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x17b_fbdb; bits[3]:0x1"
//     args: "bits[34]:0x80_0000; bits[26]:0x80_0000; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x2000; bits[3]:0x0"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2aa_aaaa; bits[3]:0x2"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1bb_b387; bits[3]:0x2"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x23b_aacf; bits[3]:0x2"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1ff_ffff; bits[3]:0x1"
//     args: "bits[34]:0x2_ff6d_9230; bits[26]:0x283_c928; bits[3]:0x0"
//     args: "bits[34]:0x8; bits[26]:0x84_850c; bits[3]:0x3"
//     args: "bits[34]:0x0; bits[26]:0x3ff_ffff; bits[3]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x345_7121; bits[3]:0x2"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x34_d5cd; bits[3]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2ab_ace2; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x0; bits[3]:0x0"
//     args: "bits[34]:0x676f_c9e1; bits[26]:0x17d_d9e0; bits[3]:0x5"
//     args: "bits[34]:0x3143_ee14; bits[26]:0x349_ceb0; bits[3]:0x2"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x1bf_f73f; bits[3]:0x7"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x2_0000; bits[3]:0x3"
//     args: "bits[34]:0x8ef3_38ed; bits[26]:0x2c3_199a; bits[3]:0x3"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x97_7edc; bits[3]:0x0"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x155_5555; bits[3]:0x7"
//     args: "bits[34]:0x0; bits[26]:0x70_1500; bits[3]:0x5"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x28a_2aaa; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x1; bits[3]:0x0"
//     args: "bits[34]:0x100_0000; bits[26]:0x0; bits[3]:0x3"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x195_8d9a; bits[3]:0x2"
//     args: "bits[34]:0x0; bits[26]:0x3cd_26c4; bits[3]:0x0"
//     args: "bits[34]:0x3_0533_04bd; bits[26]:0x3ff_ffff; bits[3]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x2000; bits[3]:0x0"
//     args: "bits[34]:0x3_c8be_b77c; bits[26]:0x2aa_aaaa; bits[3]:0x4"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x155_5555; bits[3]:0x5"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x3ff_ffff; bits[3]:0x0"
//     args: "bits[34]:0xe1bc_45f5; bits[26]:0x1b8_45f5; bits[3]:0x0"
//     args: "bits[34]:0x2_0000_0000; bits[26]:0x400; bits[3]:0x3"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x3ee_f665; bits[3]:0x2"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x3ff_ffff; bits[3]:0x4"
//     args: "bits[34]:0x2_b081_acbe; bits[26]:0x81_afff; bits[3]:0x7"
//     args: "bits[34]:0x0; bits[26]:0x0; bits[3]:0x6"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2ea_a5a2; bits[3]:0x2"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x37f_ffff; bits[3]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1; bits[3]:0x4"
//     args: "bits[34]:0x1_e9b0_7c50; bits[26]:0x1ff_ffff; bits[3]:0x3"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2fa_aaaa; bits[3]:0x1"
//     args: "bits[34]:0x40_0000; bits[26]:0x258_aa00; bits[3]:0x5"
//     args: "bits[34]:0x61a7_a2e5; bits[26]:0x107_32b1; bits[3]:0x1"
//     args: "bits[34]:0x0; bits[26]:0xfa_ff18; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x3_1a8a; bits[3]:0x1"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1ff_ffff; bits[3]:0x0"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x28d_2e41; bits[3]:0x5"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x2aa_aaaa; bits[3]:0x0"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x1_0000; bits[3]:0x3"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x2a9_a626; bits[3]:0x5"
//     args: "bits[34]:0x40_0000; bits[26]:0x3ff_ffff; bits[3]:0x0"
//     args: "bits[34]:0x4000; bits[26]:0x0; bits[3]:0x1"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x1bd_ffff; bits[3]:0x2"
//     args: "bits[34]:0x2_03fc_d43e; bits[26]:0x3fe_d12e; bits[3]:0x7"
//     args: "bits[34]:0x800_0000; bits[26]:0x4_0400; bits[3]:0x0"
//     args: "bits[34]:0x40_0000; bits[26]:0x10; bits[3]:0x7"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x0; bits[3]:0x5"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x4000; bits[3]:0x6"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x2d5_ef9b; bits[3]:0x7"
//     args: "bits[34]:0x800; bits[26]:0x11_4ca8; bits[3]:0x0"
//     args: "bits[34]:0x2000; bits[26]:0x0; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x1000; bits[3]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; bits[26]:0x2aa_aaaa; bits[3]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x1ff_ffff; bits[3]:0x0"
//     args: "bits[34]:0x0; bits[26]:0x2aa_aaaa; bits[3]:0x3"
//     args: "bits[34]:0x0; bits[26]:0x26f_2b55; bits[3]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x27a_afe1; bits[3]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x22_0caf; bits[3]:0x0"
//     args: "bits[34]:0x2_c457_6cfe; bits[26]:0x0; bits[3]:0x3"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x27d_f3df; bits[3]:0x6"
//     args: "bits[34]:0x1_ffff_ffff; bits[26]:0x172_35ed; bits[3]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2aa_aaaa; bits[3]:0x2"
//     args: "bits[34]:0x1_5555_5555; bits[26]:0x2aa_aaaa; bits[3]:0x2"
//     args: "bits[34]:0x3_5255_807e; bits[26]:0x283_14c1; bits[3]:0x2"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[26]:0x2aa_aaaa; bits[3]:0x2"
//     args: "bits[34]:0x0; bits[26]:0x2aa_aaaa; bits[3]:0x4"
//   }
// }
// 
// END_CONFIG
fn main(x0: u34, x1: u26, x2: u3) -> (u51, (u1, u17), u1, u34, u51, u1) {
    {
        let x3: (u1, u17) = match x0 {
            u34:0x2_aaaa_aaaa => (bool:false, u17:0b1_1110_0110_1101),
            u34:0x1_d352_07d4 => (bool:true, u17:0x1_ffff),
            u34:0x1_5555_5555 => (bool:true, u17:0x1_3edd),
            _ => (bool:true, u17:0x1_5555),
        };
        let (x4, x5): (u1, u17) = match x0 {
            u34:0x2_aaaa_aaaa => (bool:false, u17:0b1_1110_0110_1101),
            u34:0x1_d352_07d4 => (bool:true, u17:0x1_ffff),
            u34:0x1_5555_5555 => (bool:true, u17:0x1_3edd),
            _ => (bool:true, u17:0x1_5555),
        };
        let x6: u12 = x1[x2+:u12];
        let x7: u1 = x4 | x4;
        let x8: u51 = x0 ++ x7 ++ x4 ++ x2 ++ x6;
        let x9: u51 = one_hot_sel(x2, [x8, x8, x8]);
        let x10: u51 = -x9;
        let x11: u8 = match x0 {
            u34:0x3_ffff_ffff..u34:0x3_ffff_ffff | u34:0x0..u34:6768545714 => u8:0b111_1111,
            u34:0x2_aaaa_aaaa => u8:0x80,
            u34:0x2000_0000 => u8:0x0,
            _ => u8:0x6b,
        };
        let x12: u27 = x0[7:];
        let x13: u8 = gate!(x11 >= x0 as u8, x11);
        let x14: u1 = x3.0;
        let x15: u51 = x4 as u51 | x10;
        let x16: bool = x3 == x3;
        let x17: bool = (x2 as u51) < x9;
        let x18: u8 = x10 as u8 * x13;
        let x19: bool = -x17;
        let x20: u21 = x12[-21:];
        let x22: s21 = {
            let x21: (u21, u21) = smulp(x2 as u21 as s21, x20 as s21);
            (x21.0 + x21.1) as s21
        };
        let x23: u51 = -x15;
        let x24: u6 = x13[2+:u6];
        let x25: u34 = x0 / u34:0x0;
        let x26: bool = rev(x16);
        let x27: u51 = x10 >> x9;
        let x28: bool = ctz(x4);
        let x29: u8 = x27 as u8 ^ x18;
        let x30: bool = x16 & x14 as bool;
        let x31: u1 = one_hot_sel(x17, [x7]);
        (x23, x3, x14, x0, x23, x7)
    }
}
