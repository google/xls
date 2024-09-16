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
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[3]:0x7; bits[30]:0x3804_0020\nevaluated opt IR (JIT), evaluated opt IR (interpreter) =\n   ([bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1], bits[10]:0x1ca)\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   ([bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1, bits[1]:0x1], bits[10]:0x3ff)"
// issue: "https://github.com/google/xls/issues/1608"
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
//     args: "bits[3]:0x7; bits[30]:0x3804_0020"
//     args: "bits[3]:0x1; bits[30]:0x1803_3220"
//     args: "bits[3]:0x3; bits[30]:0xc35_7f50"
//     args: "bits[3]:0x0; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x7; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x0; bits[30]:0x1e8_ee14"
//     args: "bits[3]:0x7; bits[30]:0x3998_a0a2"
//     args: "bits[3]:0x0; bits[30]:0x3010_0010"
//     args: "bits[3]:0x7; bits[30]:0x3960_8149"
//     args: "bits[3]:0x2; bits[30]:0x0"
//     args: "bits[3]:0x0; bits[30]:0x343_f529"
//     args: "bits[3]:0x2; bits[30]:0x195b_0715"
//     args: "bits[3]:0x2; bits[30]:0x1008_1200"
//     args: "bits[3]:0x2; bits[30]:0x1000_0000"
//     args: "bits[3]:0x2; bits[30]:0x1203_4c1c"
//     args: "bits[3]:0x0; bits[30]:0x755_59a4"
//     args: "bits[3]:0x4; bits[30]:0x3ebf_aa8c"
//     args: "bits[3]:0x3; bits[30]:0x1d6c_bb66"
//     args: "bits[3]:0x2; bits[30]:0x13ff_f7fa"
//     args: "bits[3]:0x7; bits[30]:0x3d3d_c153"
//     args: "bits[3]:0x0; bits[30]:0x1555_55d5"
//     args: "bits[3]:0x5; bits[30]:0x3945_8900"
//     args: "bits[3]:0x0; bits[30]:0x28f4_1f12"
//     args: "bits[3]:0x5; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x7; bits[30]:0x3bff_fdf6"
//     args: "bits[3]:0x1; bits[30]:0xcab_2aaa"
//     args: "bits[3]:0x3; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x3; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x4; bits[30]:0x4"
//     args: "bits[3]:0x0; bits[30]:0xd4a_aa2d"
//     args: "bits[3]:0x2; bits[30]:0x4_0000"
//     args: "bits[3]:0x3; bits[30]:0x1d0b_0a8a"
//     args: "bits[3]:0x1; bits[30]:0x4000"
//     args: "bits[3]:0x2; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x7; bits[30]:0x3d82_1100"
//     args: "bits[3]:0x7; bits[30]:0x1555_5555"
//     args: "bits[3]:0x0; bits[30]:0x77f_ffff"
//     args: "bits[3]:0x2; bits[30]:0x13ff_ffbf"
//     args: "bits[3]:0x5; bits[30]:0x2829_6042"
//     args: "bits[3]:0x0; bits[30]:0x3df_7fef"
//     args: "bits[3]:0x6; bits[30]:0x10_0000"
//     args: "bits[3]:0x4; bits[30]:0x2551_5555"
//     args: "bits[3]:0x5; bits[30]:0x2909_152b"
//     args: "bits[3]:0x0; bits[30]:0x617_14f6"
//     args: "bits[3]:0x2; bits[30]:0x1399_6c6b"
//     args: "bits[3]:0x3; bits[30]:0x3a29_3dfc"
//     args: "bits[3]:0x7; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x2; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x2; bits[30]:0x3fff_ffff"
//     args: "bits[3]:0x2; bits[30]:0x4_0000"
//     args: "bits[3]:0x0; bits[30]:0x3405_07e4"
//     args: "bits[3]:0x2; bits[30]:0x0"
//     args: "bits[3]:0x2; bits[30]:0x22ce_abbc"
//     args: "bits[3]:0x2; bits[30]:0x1985_9108"
//     args: "bits[3]:0x3; bits[30]:0x3fff_ffff"
//     args: "bits[3]:0x7; bits[30]:0x3be1_fdbf"
//     args: "bits[3]:0x2; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x6; bits[30]:0x0"
//     args: "bits[3]:0x0; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x7; bits[30]:0x1555_5555"
//     args: "bits[3]:0x3; bits[30]:0x400"
//     args: "bits[3]:0x2; bits[30]:0x1368_a48e"
//     args: "bits[3]:0x2; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x2; bits[30]:0x644_3771"
//     args: "bits[3]:0x2; bits[30]:0xdcf_5ab0"
//     args: "bits[3]:0x4; bits[30]:0x4000"
//     args: "bits[3]:0x2; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x2; bits[30]:0xa5b_eab6"
//     args: "bits[3]:0x7; bits[30]:0x3808_052d"
//     args: "bits[3]:0x3; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x6; bits[30]:0x3_9006"
//     args: "bits[3]:0x7; bits[30]:0x0"
//     args: "bits[3]:0x2; bits[30]:0x1be3_d398"
//     args: "bits[3]:0x5; bits[30]:0x2c45_1555"
//     args: "bits[3]:0x7; bits[30]:0x3633_643d"
//     args: "bits[3]:0x2; bits[30]:0x1555_5555"
//     args: "bits[3]:0x2; bits[30]:0x12aa_aaaa"
//     args: "bits[3]:0x2; bits[30]:0x1555_5555"
//     args: "bits[3]:0x0; bits[30]:0x5f5_54c2"
//     args: "bits[3]:0x2; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x5; bits[30]:0x1555_5555"
//     args: "bits[3]:0x5; bits[30]:0x2aaa_e22a"
//     args: "bits[3]:0x0; bits[30]:0x23ee_d85f"
//     args: "bits[3]:0x3; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x0; bits[30]:0x716_510b"
//     args: "bits[3]:0x0; bits[30]:0x2000"
//     args: "bits[3]:0x3; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x5; bits[30]:0x24c6_f915"
//     args: "bits[3]:0x0; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x3; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x2; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x2; bits[30]:0x30ba_b02b"
//     args: "bits[3]:0x0; bits[30]:0x3bf_83a8"
//     args: "bits[3]:0x3; bits[30]:0x1be2_a86e"
//     args: "bits[3]:0x3; bits[30]:0x19bc_c662"
//     args: "bits[3]:0x0; bits[30]:0x4_0000"
//     args: "bits[3]:0x3; bits[30]:0x1010_12d2"
//     args: "bits[3]:0x7; bits[30]:0x2674_65d4"
//     args: "bits[3]:0x4; bits[30]:0x2100_0000"
//     args: "bits[3]:0x2; bits[30]:0x168a_640c"
//     args: "bits[3]:0x3; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x4; bits[30]:0x245c_44a5"
//     args: "bits[3]:0x7; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x7; bits[30]:0x37ff_feff"
//     args: "bits[3]:0x7; bits[30]:0x2abc_be06"
//     args: "bits[3]:0x2; bits[30]:0x4ac_6458"
//     args: "bits[3]:0x0; bits[30]:0x4000"
//     args: "bits[3]:0x0; bits[30]:0x1000"
//     args: "bits[3]:0x7; bits[30]:0x1fff_ffff"
//     args: "bits[3]:0x7; bits[30]:0x0"
//     args: "bits[3]:0x5; bits[30]:0x1d54_16c1"
//     args: "bits[3]:0x5; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x5; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x7; bits[30]:0x3dff_ffff"
//     args: "bits[3]:0x2; bits[30]:0x1555_5555"
//     args: "bits[3]:0x2; bits[30]:0xac_ed4b"
//     args: "bits[3]:0x3; bits[30]:0x908_8860"
//     args: "bits[3]:0x2; bits[30]:0x1699_5ce1"
//     args: "bits[3]:0x2; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x3; bits[30]:0x3fff_ffff"
//     args: "bits[3]:0x2; bits[30]:0x1000_0121"
//     args: "bits[3]:0x3; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x3; bits[30]:0x3c56_b54d"
//     args: "bits[3]:0x2; bits[30]:0x1cfe_d75c"
//     args: "bits[3]:0x4; bits[30]:0x400_0000"
//     args: "bits[3]:0x3; bits[30]:0xd92_1515"
//     args: "bits[3]:0x0; bits[30]:0x2aaa_aaaa"
//     args: "bits[3]:0x2; bits[30]:0x100"
//   }
// }
// 
// END_CONFIG
const W32_V3 = u32:0x3;
type x5 = bool;
fn x26(x27: u2, x28: u10, x29: x5[6]) -> (u57, u57, u57, u57) {
    {
        let x30: u57 = u57:0x4_0000_0000;
        let x31: u57 = x30 / u57:0x0;
        let x32: u57 = -x30;
        (x32, x31, x31, x32)
    }
}
fn main(x0: u3, x1: u30) -> (x5[15], u10) {
    {
        let x2: u3 = x0 / u3:0x2;
        let x3: u2 = encode(x0);
        let x4: u3 = x2 << if x1 >= u30:0x2 { u30:0x2 } else { x1 };
        let x6: x5[W32_V3] = x0 as x5[W32_V3];
        let x7: u2 = -x3;
        let x8: u3 = !x0;
        let x9: x5[6] = x6 ++ x6;
        let x10: u2 = !x3;
        let x11: bool = x3[1+:bool];
        let x12: x5[9] = x9 ++ x6;
        let x13: u2 = clz(x10);
        let x14: u3 = x0 + x0;
        let x15: u10 = match x2 {
            u3:0x1..u3:4 => u10:0x3ff,
            u3:0x4 | u3:0x5 => u10:0x155,
            u3:0x3 => u10:0x1ca,
            u3:0x1 => u10:0x10,
            _ => u10:0x155,
        };
        let x16: u10 = clz(x15);
        let x17: bool = xor_reduce(x13);
        let x18: bool = and_reduce(x4);
        let x19: u7 = x18 ++ x3 ++ x4 ++ x17;
        let x20: x5[15] = x9 ++ x12;
        let x21: u51 = match x0 {
            u3:0x5 | u3:0x2 => u51:0x5_5555_5555_5555,
            _ => u51:0x2_aaaa_aaaa_aaaa,
        };
        let x22: u3 = x14 >> if x18 >= bool:0x0 { bool:0x0 } else { x18 };
        let x23: u10 = x15[x10+:u10];
        let x24: u10 = x16 ^ x21 as u10;
        let x25: bool = x9 != x9;
        let x33: (u57, u57, u57, u57) = x26(x13, x15, x9);
        let (x34, x35, _, x36) = x26(x13, x15, x9);
        let x37: u10 = x24 / u10:0x8;
        (x20, x23)
    }
}
