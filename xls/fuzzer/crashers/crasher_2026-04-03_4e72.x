// Copyright 2026 The XLS Authors
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
// exception: "Subprocess call timed out after 1500 seconds: /xls/tools/opt_main
// /tmp/1789226528/tmp/test_tmpdir697qiumx/temp_directory_Bg9EKS/sample.ir
// --logtostderr"
// issue: "https://github.com/google/xls/issues/4044"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the
//     specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full
//     throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "bits[58]:0x80_0000_0000; bits[34]:0x1_9743_2a08"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x9c_0f41_cfa7_d19a; bits[34]:0x2_b0ea_9bca"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x3_ffb7_ffce"
//     args: "bits[58]:0x0; bits[34]:0x1_0000_0048"
//     args: "bits[58]:0x0; bits[34]:0x1090_006a"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_c480_e36f"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_d115_c555"
//     args: "bits[58]:0x0; bits[34]:0x3301_8809"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x3_d31b_5441"
//     args: "bits[58]:0x39_9807_0e62_b367; bits[34]:0x2_1663_b366"
//     args: "bits[58]:0x4_0000; bits[34]:0x3_8c0d_0030"
//     args: "bits[58]:0x268_b990_f274_cce8; bits[34]:0xf274_c4e8"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x0; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x2d7_c987_a46c_005a; bits[34]:0x8_0000"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0xa11a_4269"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_5111_585d"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x1_8f39_ebd1"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x0"
//     args: "bits[58]:0x1e8_6dfd_f7b9_1067; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x3_fffb_ffff"
//     args: "bits[58]:0x4000_0000; bits[34]:0x8"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x2_cae6_aaa6"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x0; bits[34]:0x1_0073_0044"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x1_5555_5555"
//     args: "bits[58]:0x8; bits[34]:0x1_ffff_ffff"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_77ff_f7cf"
//     args: "bits[58]:0x0; bits[34]:0x4_0000"
//     args: "bits[58]:0x0; bits[34]:0x10_009c"
//     args: "bits[58]:0x126_eca7_7817_fcaa; bits[34]:0x3_7857_fca8"
//     args: "bits[58]:0xd5_8fcf_a943_513c; bits[34]:0x2_7b10_5e3e"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x4_0000; bits[34]:0xc14e_8030"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x200_0000; bits[34]:0x1_ffff_ffff"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x823a_febc"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x0"
//     args: "bits[58]:0x0; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x3_fae1_467f"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0xfd3c_6fb9"
//     args: "bits[58]:0x100_0000_0000; bits[34]:0x0"
//     args: "bits[58]:0x4_0000_0000_0000; bits[34]:0x56a2_150a"
//     args: "bits[58]:0x100_0000; bits[34]:0x1_ffff_ffff"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x1_ddfa_b6da"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x4c95_2440"
//     args: "bits[58]:0x38b_cf39_12a0_755d; bits[34]:0x0"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_5555_5555"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x1_9f6f_dd76"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x0"
//     args: "bits[58]:0x0; bits[34]:0x800_c006"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x0; bits[34]:0x2_0132_8012"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_ffab_db7e"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x80"
//     args: "bits[58]:0x400; bits[34]:0x400"
//     args: "bits[58]:0x0; bits[34]:0x1_1beb_2e71"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x3_8737_e373"
//     args: "bits[58]:0x0; bits[34]:0x3_2d7c_fcd6"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x555d_1505"
//     args: "bits[58]:0x0; bits[34]:0x2_39c9_8366"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_ffff_ffff"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_dee8_5f57"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_6afc_db2f"
//     args: "bits[58]:0x1c7_4cae_bff9_8f42; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x0; bits[34]:0xa520_10b4"
//     args: "bits[58]:0x4; bits[34]:0x100"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x1_ffff_ffff"
//     args: "bits[58]:0xa1ed_87f4_0f08; bits[34]:0x1_5555_5555"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x1_2535_8559"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0xc05_1515"
//     args: "bits[58]:0x0; bits[34]:0x1200"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_6aaf_8845"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x2_8528_a8cf"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x3b87_74d4"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x2_baeb_2aa2"
//     args: "bits[58]:0x3df_5a2a_671f_0df9; bits[34]:0x0"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_effd_fff9"
//     args: "bits[58]:0x0; bits[34]:0x1_ffff_ffff"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_c454_5445"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x1_df30_7f6f"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x1_12c6_5f2a"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x3_dc56_5d55"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x3_6acf_2de6"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x0"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_af3b_6af5"
//     args: "bits[58]:0x250_ce40_6161_7cb9; bits[34]:0x6101_1cbd"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x3_fffb_7fff"
//     args: "bits[58]:0x79_4f75_653a_4cc8; bits[34]:0xa71b_381c"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x1_ffff_ffff"
//     args: "bits[58]:0x0; bits[34]:0x2_0204"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x1_359a_952f"
//     args: "bits[58]:0x2000; bits[34]:0x2_1822_ae40"
//     args: "bits[58]:0x2_0000_0000_0000; bits[34]:0x1_4983_c007"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x2_0ee1_ebac"
//     args: "bits[58]:0x0; bits[34]:0x1_a2ab_2475"
//     args: "bits[58]:0x0; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x2; bits[34]:0x10_0002"
//     args: "bits[58]:0x0; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x1_6acf_feed"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x2_ec51_b5ff"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x2_fede_fdef"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x2_5555_5507"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x2_9fff_7e6b"
//     args: "bits[58]:0x0; bits[34]:0x1_5555_5555"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x0"
//     args: "bits[58]:0x0; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x1_0000; bits[34]:0x0"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x3_be55_767b"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x3ff_ffff_ffff_ffff; bits[34]:0x2_ebf7_fff5"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_5555_5555"
//     args: "bits[58]:0x0; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x2aa_aaaa_aaaa_aaaa; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x2_aaaa_aaaa"
//     args: "bits[58]:0xda_b06d_e142_b027; bits[34]:0x3_ffff_ffff"
//     args: "bits[58]:0x1ff_ffff_ffff_ffff; bits[34]:0x0"
//     args: "bits[58]:0x2a7_84c3_f201_67e6; bits[34]:0x2_3219_6f7a"
//     args: "bits[58]:0x155_5555_5555_5555; bits[34]:0x1_5751_d579"
//     args: "bits[58]:0x2_0000_0000; bits[34]:0x2_0803_4119"
//   }
// }
//
// END_CONFIG
type x28 = (u35, s34, s34, u34);

fn x22
    (x23: (u35, s34, s34, u34), x24: s58, x25: s34, x26: uN[105])
    -> (x28[3], x28[3], bool, bool, x28[3]) {
    {
        let x27: bool = and_reduce(x26);
        let x29: x28[3] = [x23, x23, x23];
        (x29, x29, x27, x27, x29)
    }
}

fn main(x0: s58, x1: s34) -> (uN[1323], s34, s34, s64) {
    {
        let x2: s34 = x1 + x1;
        let x3: s34 = x2 / s34:0x3026_99fd;
        let x4: s34 = gate!(x0 >= x3 as s58, x1);
        let x5: u34 = (x3 as u34)[:];
        let x6: u35 = one_hot(x5, bool:0x1);
        let x7: uN[105] = x6 ++ x6 ++ x6;
        let x8: uN[105] = bit_slice_update(x7, x6, x6);
        let x9: uN[487] = x5 ++ x6 ++ x6 ++ x5 ++ x7 ++ x7 ++ x7 ++ x5;
        let x10: u35 = one_hot_sel(u2:2, [x6, x6]);
        let x11: u13 = x7[92+:u13];
        let x12: s58 = -x0;
        let x13: s64 = match x5 {
            u34:0x3_ffff_ffff => xN[bool:0x1][64]:0x7fff_ffff_ffff_ffff,
            u34:0x1_5555_5555 => s64:0x0,
            u34:0x1_ffff_ffff => s64:-6148914691236517206,
            u34:0xe6f_2b56 | xN[bool:0x0][34]:17179869183 => s64:0xaaaa_aaaa_aaaa_aaaa,
            _ => s64:6148914691236517205,
        };
        let x14: u35 = rev(x10);
        let x15: uN[105] = bit_slice_update(x8, x11, x9);
        let x16: s34 = -x3;
        let x17: (u35, s34, s34, u34) = (x6, x2, x1, x5);
        let x18: u58 = (x12 as u58)[:];
        let x19: uN[105] = x7 >> if x18 >= u58:0x9 { u58:0x9 } else { x18 };
        let x20: s34 = !x2;
        let x21: uN[105] = one_hot_sel(bool:false, [x7]);
        let x30: (x28[3], x28[3], bool, bool, x28[3]) = x22(x17, x0, x20, x15);
        let x31: u11 = u11:0x3ff;
        let x32: s34 = gate!(x15 > x21, x2);
        let x33: uN[105] = !x19;
        let x34: uN[1323] = x9 ++ x33 ++ x9 ++ x19 ++ x5 ++ x19;
        let x35: bool = and_reduce(x11);
        let x36: u13 = for (i, x): (u4, u13) in u4:0x0..u4:0x1 {
            x
        }(x11);
        let x37: s34 = x2 % s34:0x2_aaaa_aaaa;
        (x34, x16, x1, x13)
    }
}
