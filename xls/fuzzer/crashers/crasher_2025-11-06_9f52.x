// Copyright 2025 The XLS Authors
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
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[57]:0x10;
// bits[24]:0xc6_c3c1; bits[50]:0x2_0000_0000\nevaluated opt IR (JIT), evaluated opt IR
// (interpreter), interpreted DSLX, simulated, simulated_ng =\n   (bits[57]:0x34,
// [bits[19]:0x7_ffff, bits[19]:0x7_ffff, bits[19]:0x7_fff0, bits[19]:0x0, bits[19]:0x18,
// bits[19]:0x6_c3f5, bits[19]:0x0, bits[19]:0x0, bits[19]:0x10, bits[19]:0x0, bits[19]:0x18,
// bits[19]:0x6_c3f5, bits[19]:0x7_ffff, bits[19]:0x7_ffff, bits[19]:0x7_fff0, bits[19]:0x0,
// bits[19]:0x18, bits[19]:0x6_c3f5, bits[19]:0x0, bits[19]:0x0, bits[19]:0x10, bits[19]:0x0,
// bits[19]:0x18, bits[19]:0x6_c3f5], bits[3]:0x0, bits[57]:0x34)\nevaluated unopt IR (JIT) =\n
// (bits[57]:0x34, [bits[19]:0x7_ffff, bits[19]:0x7_ffff, bits[19]:0x7_fff0, bits[19]:0x3_0300,
// bits[19]:0x18, bits[19]:0x6_c3f5, bits[19]:0x0, bits[19]:0x0, bits[19]:0x10, bits[19]:0x0,
// bits[19]:0x18, bits[19]:0x6_c3f5, bits[19]:0x7_ffff, bits[19]:0x7_ffff, bits[19]:0x7_fff0,
// bits[19]:0x3_0300, bits[19]:0x18, bits[19]:0x6_c3f5, bits[19]:0x0, bits[19]:0x0, bits[19]:0x10,
// bits[19]:0x0, bits[19]:0x18, bits[19]:0x6_c3f5], bits[3]:0x0, bits[57]:0x34)"
// issue: "https://github.com/google/xls/issues/3330"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=2"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=false"
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
//   codegen_ng: true
//   disable_unopt_interpreter: true
// }
// inputs {
//   function_args {
//     args: "bits[57]:0x10; bits[24]:0xc6_c3c1; bits[50]:0x2_0000_0000"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xbe_ef32; bits[50]:0x2_0be3_3fef_977d"
//     args: "bits[57]:0x0; bits[24]:0xaa_aaaa; bits[50]:0x2_8080_1200_2680"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0x55_5555; bits[50]:0x2_ea8a_abff_aaa2"
//     args: "bits[57]:0x0; bits[24]:0x10_1240; bits[50]:0x0"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xff_ffff; bits[50]:0x3_ef5f_f7fb_dfde"
//     args: "bits[57]:0x1a3_89c1_5ad0_c4fa; bits[24]:0xd1_d0ea; bits[50]:0x0"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xff_ffff; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xff_ffff; bits[50]:0x2000_0000"
//     args: "bits[57]:0x12e_8843_667c_3d82; bits[24]:0x74_b582; bits[50]:0x1_d2d6_0aaa_aaaa"
//     args: "bits[57]:0x0; bits[24]:0x1d_30e5; bits[50]:0x0"
//     args: "bits[57]:0x8000_0000; bits[24]:0x0; bits[50]:0x2011_03ff_6dff"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0x77_41d5; bits[50]:0x1_cd07_545d_5555"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xfa_0241; bits[50]:0x1_cbfe_fe36_86ad"
//     args: "bits[57]:0x0; bits[24]:0xa7_d0ff; bits[50]:0x0"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xe7_23ab; bits[50]:0x3_9c8e_bdff_ff7f"
//     args: "bits[57]:0x0; bits[24]:0xaa_aaaa; bits[50]:0x40_2001"
//     args: "bits[57]:0xb8_3232_de88_e20c; bits[24]:0x80_cc7c; bits[50]:0x1_5555_5555_5555"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xdc_197b; bits[50]:0x2_2975_eddb_dbd1"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0x7f_ffff; bits[50]:0x1_fdff_7f04_555f"
//     args: "bits[57]:0x10; bits[24]:0xff_ffff; bits[50]:0x3_f363_fcf6_f558"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xdb_bb5f; bits[50]:0x3_ffff_ffff_fff7"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xff_ffdf; bits[50]:0x3_40c7_1176_7b1d"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xff_ffef; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xf8_c7ff; bits[50]:0x1_5555_5555_5555"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0x80; bits[50]:0x2002_e9d0_c682"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0x400; bits[50]:0x10_028a_aba8"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xff_ffff; bits[50]:0xf1ff_e552_775f"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0xff_ffff; bits[50]:0x1_551e_41d5_545d"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0x12_adfe; bits[50]:0xd35e_9285_7693"
//     args: "bits[57]:0x0; bits[24]:0xaa_aaaa; bits[50]:0x100_3500_0000"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xaa_aaaa; bits[50]:0x3_88c9_ba0a_8ab9"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xae_9f9b; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0x185_a4e0_5923_9aa8; bits[24]:0x2a_9680; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xff_ffff; bits[50]:0x200"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0x0; bits[50]:0x3ff_fbff"
//     args: "bits[57]:0x0; bits[24]:0x1e_d185; bits[50]:0x8062_4024_1eb0"
//     args: "bits[57]:0xd_9143_50bd_30bd; bits[24]:0x8f_3261; bits[50]:0x8_0000_0000"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0x0; bits[50]:0x2_ea24_86df_cd5f"
//     args: "bits[57]:0x0; bits[24]:0x5a_0d30; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0x1000; bits[24]:0x0; bits[50]:0x4_0200_0708"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xaa_aaaa; bits[50]:0x2_2eae_91bc_c9f3"
//     args: "bits[57]:0x40_0000_0000; bits[24]:0x4_0000; bits[50]:0x1_ce77_1c8f_0de0"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xaa_aaaa; bits[50]:0x1_4ec4_aaa5_c03e"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xba_aae8; bits[50]:0x2_6f23_0aa1_9aaa"
//     args: "bits[57]:0x12_5970_9cee_9ae3; bits[24]:0xcd_1f21; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xbe_7fde; bits[50]:0x0"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xed_ef5f; bits[50]:0x3_b699_fc00_5089"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0xff_ffff; bits[50]:0x3_97df_919d_df7d"
//     args: "bits[57]:0x8_0000_0000_0000; bits[24]:0x5c8; bits[50]:0x209f_0711_8054"
//     args: "bits[57]:0x80_0000_0000; bits[24]:0xaa_aaaa; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0x55_5555; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0x7f_ffff; bits[50]:0x2_3dfb_4bff_dfde"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xa8_a7aa; bits[50]:0x2_208f_ed86_1db0"
//     args: "bits[57]:0x2a_9e0e_6f7c_6567; bits[24]:0x7f_ffff; bits[50]:0x2_d8db_3a54_47a8"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0x55_5555; bits[50]:0x5559_55a8_5314"
//     args: "bits[57]:0x148_753c_01c6_f29d; bits[24]:0xc0_5995; bits[50]:0x8000"
//     args: "bits[57]:0x1be_6664_434e_7e35; bits[24]:0x7f_ffff; bits[50]:0x3_62ae_42ae_c23c"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0xff_ffff; bits[50]:0x1_0000"
//     args: "bits[57]:0x10; bits[24]:0xbf_9591; bits[50]:0x2_fe56_c402_2000"
//     args: "bits[57]:0x800_0000; bits[24]:0x55_5555; bits[50]:0x100_0000_0000"
//     args: "bits[57]:0x0; bits[24]:0xc8_0000; bits[50]:0x1_5555_5555_5555"
//     args: "bits[57]:0x0; bits[24]:0x2280; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0x800_0000_0000; bits[24]:0x80_8402; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0x400; bits[24]:0xe0_5f3f; bits[50]:0x2"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xff_ff7f; bits[50]:0x3_e375_77f3_ebfb"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0x31_5554; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xaa_aaaa; bits[50]:0x0"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xca_9bfa; bits[50]:0x1_6a1e_abb7_6b65"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0x9b_6d51; bits[50]:0x2_6d31_65be_4ebf"
//     args: "bits[57]:0x0; bits[24]:0x2002; bits[50]:0x0"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0x70_45b9; bits[50]:0x1551_3405_3182"
//     args: "bits[57]:0x1ac_82a1_dca1_987a; bits[24]:0x7f_ffff; bits[50]:0x800_0000"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xff_ffff; bits[50]:0x3_ddee_bfff_fff7"
//     args: "bits[57]:0x0; bits[24]:0x24_a0f1; bits[50]:0x3_dfbf_eb8f_6f9f"
//     args: "bits[57]:0x10_0000; bits[24]:0x52_4432; bits[50]:0x2_00c8_3200_8190"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0xaa_aaaa; bits[50]:0x1_5555_5555_5555"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0x0; bits[50]:0x2_7f2e_7ef7_bd78"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0xe7_7320; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0x55_5555; bits[50]:0xbcce_d8fa_bd83"
//     args: "bits[57]:0xaa_5f93_4430_3459; bits[24]:0x200; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xa2_abea; bits[50]:0x3_86e0_e62e_8876"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0x6f_7e0a; bits[50]:0x3_bf7b_b80a_52a3"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0x5_1555; bits[50]:0x145d_9eaa_aaaa"
//     args: "bits[57]:0xd6_1d5b_5290_1849; bits[24]:0x1a_4859; bits[50]:0x200"
//     args: "bits[57]:0xb6_1f6d_3329_90bb; bits[24]:0x29_909b; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0x0; bits[24]:0x55_5555; bits[50]:0x0"
//     args: "bits[57]:0x20_0000_0000; bits[24]:0x7f_ffff; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0x0; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0x1cf_5b90_d5ce_55a8; bits[24]:0xce_55a8; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xdf_bed4; bits[50]:0x2_0000_0000_0000"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xdf_ff7c; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0x10_0000_0000_0000; bits[24]:0x40; bits[50]:0x83_01fb_ffff"
//     args: "bits[57]:0x0; bits[24]:0x20_0000; bits[50]:0xa195_4047_820e"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xad_c8c1; bits[50]:0x2_3623_01ff_fffe"
//     args: "bits[57]:0x9a_5f73_9599_3e01; bits[24]:0x89_2e09; bits[50]:0x2000_0000_0000"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xdf_bf3d; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xbf_d34d; bits[50]:0x3_ffff_fffb_fff7"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0x800; bits[50]:0x2aea_a223_c0bb"
//     args: "bits[57]:0x1f5_06c2_2fa6_aaba; bits[24]:0x86_e314; bits[50]:0x3f8e_fed7_f78d"
//     args: "bits[57]:0x80; bits[24]:0x2_0000; bits[50]:0x3_b712_cc0a_2719"
//     args: "bits[57]:0x0; bits[24]:0xaa_aaaa; bits[50]:0x0"
//     args: "bits[57]:0xdc_ccd4_33cc_8619; bits[24]:0x90_7cfc; bits[50]:0x1_ed14_6d5c_03b7"
//     args: "bits[57]:0x10; bits[24]:0xb3_0c10; bits[50]:0x2_ccb1_43ff_efff"
//     args: "bits[57]:0x2_0000_0000; bits[24]:0xbf_592d; bits[50]:0x4a_2205_0340"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xae_bd8e; bits[50]:0xea3a_282b_abfa"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0x55_5555; bits[50]:0x0"
//     args: "bits[57]:0x8_0000_0000; bits[24]:0x20; bits[50]:0x8000_1000"
//     args: "bits[57]:0x185_5fa5_531c_a434; bits[24]:0x18_a474; bits[50]:0x1_5555_5555_5555"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xdf_7ff7; bits[50]:0x3_b7fc_f5cb_bf57"
//     args: "bits[57]:0x0; bits[24]:0x12_2035; bits[50]:0x990c_0768_5c51"
//     args: "bits[57]:0x0; bits[24]:0xaa_aaaa; bits[50]:0x20_0000_0000"
//     args: "bits[57]:0x2_0000_0000_0000; bits[24]:0xaa_aaaa; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0x0; bits[24]:0x7f_ffff; bits[50]:0x2_ecfe_fccd_6e07"
//     args: "bits[57]:0x46_7f00_525f_51d2; bits[24]:0xaa_aaaa; bits[50]:0x1_ffff_ffff_ffff"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xff_94ff; bits[50]:0x3_664f_eeb0_317f"
//     args: "bits[57]:0x0; bits[24]:0x71_9869; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[24]:0xff_ffff; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0x1000_0000_0000; bits[24]:0x6_1164; bits[50]:0x3_ffff_ffff_ffff"
//     args: "bits[57]:0x5b_99ca_3833_eb32; bits[24]:0x30_3832; bits[50]:0x3_23da_3937_af03"
//     args: "bits[57]:0x4_0000_0000; bits[24]:0x4_1800; bits[50]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0x55_4555; bits[50]:0x8_0000"
//     args: "bits[57]:0x140_769c_c6cb_a7c4; bits[24]:0xc8_27c4; bits[50]:0x3_2096_09ff_ff5b"
//     args: "bits[57]:0x155_5555_5555_5555; bits[24]:0xd1_1551; bits[50]:0x3_5450_fd13_d470"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[24]:0xaa_aaaa; bits[50]:0x0"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xaa_aaaa; bits[50]:0xda85_4dda_c841"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[24]:0xab_aaea; bits[50]:0x2_2eab_e801_0000"
//     args: "bits[57]:0x4_0000; bits[24]:0x55_5555; bits[50]:0x1_ffff_ffff_ffff"
//   }
// }
//
// END_CONFIG
const W32_V19 = u32:0x13;

type x7 = u3;
type x13 = (u3, bool, u3);
type x37 = uN[228];
type x50 = u19;
type x55 = x7[7];

fn x9(x10: x7) -> (u3, bool, u3) {
    {
        let x11: bool = and_reduce(x10);
        let x12: u3 = x10[0+:u3];
        (x12, x11, x12)
    }
}

fn x20
    (x21: x13[W32_V19], x22: s50, x23: x7[7], x24: u57, x25: s50, x26: u57, x27: u57, x28: u57,
     x29: uN[81]) -> (uN[228], bool) {
    {
        let x30: u57 = -x27;
        let x31: (s50, u57, u57, x13[W32_V19]) = (x25, x27, x26, x21);
        let (.., x32, x33, x34, x35) = (x25, x27, x26, x21);
        let x36: uN[228] = x30 ++ x28 ++ x27 ++ x28;
        let x38: x37[1] = [x36];
        let x39: bool = xor_reduce(x27);
        (x36, x39)
    }
}

fn main(x0: u57, x1: u24, x2: s50) -> (u57, x50[24], x7, u57) {
    {
        let x3: uN[81] = x1 ++ x0;
        let x4: u57 = clz(x0);
        let x5: u57 = ctz(x0);
        let x6: u57 = x4 % u57:0xff_ffff_ffff_ffff;
        let x8: x7[W32_V19] = x4 as x7[W32_V19];
        let x14: x13[W32_V19] = map(x8, x9);
        let x15: x7[7] = array_slice(x8, x1, x7[7]:[x8[u32:0x0], ...]);
        let x16: u57 = x4 | x6;
        let x17: u57 = x1 as u57 | x4;
        let x18: bool = or_reduce(x5);
        let x19: u57 = x4 % u57:0x1ff_ffff_ffff_ffff;
        let x40: (uN[228], bool) = x20(x14, x2, x15, x0, x2, x4, x0, x17, x3);
        let (x41, x42): (uN[228], bool) = x20(x14, x2, x15, x0, x2, x4, x0, x17, x3);
        let x43: uN[228] = x40.0;
        let x44: bool = x40 != x40;
        let x45: u6 = x1[-6:];
        let x46: u57 = for (i, x): (u4, u57) in u4:0x0..u4:0x3 {
            x
        }(x5);
        let x47: u44 = x5[x5+:u44];
        let x48: u57 = !x46;
        let x49: bool = or_reduce(x43);
        let x51: x50[12] = x43 as x50[12];
        let x52: u57 = -x16;
        let x53: x50[24] = x51 ++ x51;
        let x54: uN[228] = -x43;
        let x56: x55[1] = [x15];
        let x57: bool = x18[x52+:bool];
        let x58: x7 = x8[if x41 >= uN[228]:0x2 { uN[228]:0x2 } else { x41 }];
        let x59: (x55[1], uN[81], u57, bool, bool, uN[228]) = (x56, x3, x46, x18, x49, x54);
        (x6, x53, x58, x4)
    }
}
