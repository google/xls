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
// exception: "SampleError: Result miscompare for sample 1:\nargs: bits[21]:0x8_1001; bits[5]:0x1f; bits[37]:0x1f_ffff_ffff\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX, simulated =\n   (bits[49]:0xffff_ffff_ffff, bits[1]:0x1, bits[1]:0x1, (bits[1]:0x1, bits[19]:0x5_959f))\nevaluated unopt IR (JIT) =\n   (bits[49]:0xffff_ffff_ffff, bits[1]:0x1, bits[1]:0x1, (bits[1]:0x1, bits[19]:0x3_2b3e))"
// issue: "https://github.com/google/xls/issues/1636"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=4"
//   codegen_args: "--worst_case_throughput=2"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=false"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: true
//   use_system_verilog: false
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
//     args: "bits[21]:0x1f_ffff; bits[5]:0x17; bits[37]:0xf_19ad_5145"
//     args: "bits[21]:0x8_1001; bits[5]:0x1f; bits[37]:0x1f_ffff_ffff"
//     args: "bits[21]:0x0; bits[5]:0xc; bits[37]:0x8_0000"
//     args: "bits[21]:0x15_5555; bits[5]:0x1d; bits[37]:0x1c_2f64_f201"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x5; bits[37]:0xf_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1c; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0x0; bits[5]:0x0; bits[37]:0x7dee"
//     args: "bits[21]:0x0; bits[5]:0x18; bits[37]:0xe_d462_7261"
//     args: "bits[21]:0x80; bits[5]:0x14; bits[37]:0x14_264a_c882"
//     args: "bits[21]:0x16_49f8; bits[5]:0xf; bits[37]:0xe_4ff0_0201"
//     args: "bits[21]:0x0; bits[5]:0xf; bits[37]:0x1f_ffff_ffff"
//     args: "bits[21]:0x12_f2a6; bits[5]:0x1f; bits[37]:0x1f_ffbf_ffff"
//     args: "bits[21]:0x1d_6da9; bits[5]:0x19; bits[37]:0xf_ffff_ffff"
//     args: "bits[21]:0x4_35e3; bits[5]:0x6; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x15_5555; bits[5]:0x5; bits[37]:0x17_c537_210c"
//     args: "bits[21]:0xf_ffff; bits[5]:0x16; bits[37]:0x0"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x4; bits[37]:0xf_ffff_ffff"
//     args: "bits[21]:0xf_ffff; bits[5]:0x1; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x1b; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1d; bits[37]:0x12_9350_4ce1"
//     args: "bits[21]:0x15_5555; bits[5]:0x0; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0x1f_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1e; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x15_5555; bits[5]:0x15; bits[37]:0xb_5954_d059"
//     args: "bits[21]:0x15_5555; bits[5]:0x14; bits[37]:0x4_8004_8080"
//     args: "bits[21]:0x2_0000; bits[5]:0x9; bits[37]:0xe_8913_85d7"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x15; bits[37]:0x15_ffff_ffff"
//     args: "bits[21]:0x14_6fde; bits[5]:0xf; bits[37]:0x17_4b90_59c6"
//     args: "bits[21]:0x4000; bits[5]:0x10; bits[37]:0xe_3aa6_4990"
//     args: "bits[21]:0x15_5555; bits[5]:0xd; bits[37]:0x4000_0000"
//     args: "bits[21]:0x14_7e04; bits[5]:0x14; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0xf_ffff; bits[5]:0x15; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x0; bits[5]:0x0; bits[37]:0x0"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1e; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0x8_2ae8_8aa2"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1f; bits[37]:0x1f_8000_a040"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x15; bits[37]:0x0"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x0; bits[37]:0x0"
//     args: "bits[21]:0x15_5555; bits[5]:0x15; bits[37]:0x10_713e_5907"
//     args: "bits[21]:0x0; bits[5]:0x17; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0xa_a3cb; bits[5]:0x4; bits[37]:0x0"
//     args: "bits[21]:0x15_5555; bits[5]:0x15; bits[37]:0x14_1545_aaaa"
//     args: "bits[21]:0xf_ffff; bits[5]:0xf; bits[37]:0x1f_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1b; bits[37]:0x1f_3ff7_5fff"
//     args: "bits[21]:0x17_4178; bits[5]:0x15; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x0; bits[5]:0x16; bits[37]:0x7_0048_2aac"
//     args: "bits[21]:0x2000; bits[5]:0x9; bits[37]:0x17_ed70_e488"
//     args: "bits[21]:0xf_ffff; bits[5]:0x1; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0x80; bits[5]:0x0; bits[37]:0x10_0000_9041"
//     args: "bits[21]:0x4_5c7f; bits[5]:0x12; bits[37]:0x1f_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1e; bits[37]:0x4_0000"
//     args: "bits[21]:0x15_5555; bits[5]:0x15; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0x4; bits[5]:0x7; bits[37]:0x6_abe7_dfe7"
//     args: "bits[21]:0x100; bits[5]:0xf; bits[37]:0xf_f545_515d"
//     args: "bits[21]:0xf_ffff; bits[5]:0xf; bits[37]:0xe_7fdf_bfef"
//     args: "bits[21]:0xf_ffff; bits[5]:0x19; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x15_5555; bits[5]:0x16; bits[37]:0x13_7d57_339d"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x15; bits[37]:0x1d_2200_0000"
//     args: "bits[21]:0x10_0000; bits[5]:0x14; bits[37]:0x13_99ff_5b42"
//     args: "bits[21]:0xf_ffff; bits[5]:0x1f; bits[37]:0xd_369b_d77f"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x1a; bits[37]:0xf_ffff_ffff"
//     args: "bits[21]:0xf_ffff; bits[5]:0xf; bits[37]:0xf_7fef_76e5"
//     args: "bits[21]:0x15_5555; bits[5]:0x14; bits[37]:0x0"
//     args: "bits[21]:0xe_d3a4; bits[5]:0x4; bits[37]:0xe_d2b4_ff7f"
//     args: "bits[21]:0x15_5555; bits[5]:0x15; bits[37]:0x15_7973_c221"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0x1e_ba8f_35f5"
//     args: "bits[21]:0x1; bits[5]:0xf; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x0; bits[5]:0xe; bits[37]:0x1f_ffff_ffff"
//     args: "bits[21]:0x0; bits[5]:0x0; bits[37]:0x942_223c"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0xa_aaaa_1455"
//     args: "bits[21]:0x1000; bits[5]:0x14; bits[37]:0xf_ffff_ffff"
//     args: "bits[21]:0x1f_ffff; bits[5]:0xe; bits[37]:0x14_19f5_1fe9"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x1f; bits[37]:0x7_fd16_be28"
//     args: "bits[21]:0x1; bits[5]:0xa; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0x7b2d; bits[5]:0x15; bits[37]:0x69a1_b9a2"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0xa_9aaa_7f71"
//     args: "bits[21]:0x0; bits[5]:0x0; bits[37]:0xf_b00e_cc6d"
//     args: "bits[21]:0xc_ba70; bits[5]:0x1f; bits[37]:0xf_bf7c_dbbb"
//     args: "bits[21]:0x1_0000; bits[5]:0x8; bits[37]:0x4"
//     args: "bits[21]:0xf_ffff; bits[5]:0xa; bits[37]:0xf_f73b_df0c"
//     args: "bits[21]:0x0; bits[5]:0x8; bits[37]:0x8_4755_5555"
//     args: "bits[21]:0x0; bits[5]:0x1; bits[37]:0x0"
//     args: "bits[21]:0x8000; bits[5]:0x4; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x0; bits[5]:0x4; bits[37]:0x4_0000_0080"
//     args: "bits[21]:0x0; bits[5]:0x11; bits[37]:0x10_0128_caed"
//     args: "bits[21]:0xf_ffff; bits[5]:0x12; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1f; bits[37]:0x1d_6faf_ef3d"
//     args: "bits[21]:0x15_5555; bits[5]:0x11; bits[37]:0x13_8aaa_8e2a"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xf; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x15_5555; bits[5]:0xa; bits[37]:0x15_7555_0179"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0x1a_cb9b_27af"
//     args: "bits[21]:0x8000; bits[5]:0x4; bits[37]:0xc_0150_1101"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x10; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0x2_81d0_0150"
//     args: "bits[21]:0x10_b4bd; bits[5]:0xd; bits[37]:0x9_38ff_ebfd"
//     args: "bits[21]:0x4_9ce0; bits[5]:0x1f; bits[37]:0x1e_2a10_e222"
//     args: "bits[21]:0x15_5555; bits[5]:0x15; bits[37]:0x1d_488d_6a1a"
//     args: "bits[21]:0xf_ffff; bits[5]:0x1f; bits[37]:0x6_e926_5df7"
//     args: "bits[21]:0x0; bits[5]:0x1; bits[37]:0x1e_ba3d_1fd6"
//     args: "bits[21]:0x0; bits[5]:0xf; bits[37]:0xf_ffff_ffff"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x10; bits[37]:0x4_fa2f_22c0"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1f; bits[37]:0xf_928f_fc53"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0x8_0000_2100"
//     args: "bits[21]:0x15_5555; bits[5]:0x15; bits[37]:0x200_0000"
//     args: "bits[21]:0xf_ffff; bits[5]:0xf; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x0; bits[5]:0x0; bits[37]:0xc_f6da_f6d7"
//     args: "bits[21]:0x12_4792; bits[5]:0x12; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0x17_ee8d; bits[5]:0xf; bits[37]:0x1f_9652_033a"
//     args: "bits[21]:0xf_ffff; bits[5]:0x1f; bits[37]:0x1f_ffff_ffff"
//     args: "bits[21]:0x0; bits[5]:0x0; bits[37]:0x4_2aab_caaa"
//     args: "bits[21]:0xf_ffff; bits[5]:0x1f; bits[37]:0xf_feff_0216"
//     args: "bits[21]:0xf_ffff; bits[5]:0x7; bits[37]:0x5_4021_612f"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1d; bits[37]:0x14_c8bf_5821"
//     args: "bits[21]:0xd_4d77; bits[5]:0x1f; bits[37]:0x1f_c781_6a04"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x13; bits[37]:0x13_0904_58b8"
//     args: "bits[21]:0x1f_ffff; bits[5]:0x1e; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0xb_29ed; bits[5]:0xd; bits[37]:0x1b_2acc_eb77"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x1a; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x2; bits[37]:0x15_5555_5555"
//     args: "bits[21]:0x13_ed53; bits[5]:0x10; bits[37]:0x0"
//     args: "bits[21]:0xf_ffff; bits[5]:0x1d; bits[37]:0x3_3e46_e0a3"
//     args: "bits[21]:0xa_aaaa; bits[5]:0xa; bits[37]:0x1a_2aaa_fffb"
//     args: "bits[21]:0xe_79bc; bits[5]:0x0; bits[37]:0x10_f358_fc86"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x4; bits[37]:0xa_aaaa_aaaa"
//     args: "bits[21]:0xa_aaaa; bits[5]:0x15; bits[37]:0x15_9004_1002"
//     args: "bits[21]:0xf_ffff; bits[5]:0xe; bits[37]:0xc_a634_cf20"
//     args: "bits[21]:0x2_0000; bits[5]:0x0; bits[37]:0x2_8038_e283"
//   }
// }
// 
// END_CONFIG
type x17 = u7;
fn main(x0: u21, x1: u5, x2: u37) -> (u49, bool, bool, (bool, u19)) {
    {
        let x3: bool = x2 >= x1 as u37;
        let x4: u5 = one_hot_sel(x3, [x1]);
        let x5: (bool, u19) = match x0 {
            u21:0xa_aaaa | u21:0b1 => (x3, u19:0x5_5555),
            u21:0x0..u21:1917074 | u21:0x800 => (bool:0x1, u19:0x5_959f),
            _ => (x3, u19:0x0),
        };
        let x6: bool = x3 ^ x3;
        let x7: u21 = -x0;
        let x8: u21 = x0[x1+:u21];
        let x9: bool = x5 != x5;
        let x10: u5 = x4 >> if x8 >= u21:0x1 { u21:0x1 } else { x8 };
        let x11: bool = x6 | x3;
        let x12: bool = and_reduce(x9);
        let x13: bool = x6 / bool:0x1;
        let x14: u52 = x8 ++ x1 ++ x10 ++ x0;
        let x15: (bool, u19) = for (i, x) in u4:0x0..u4:0x5 {
            x
        }(x5);
        let x16: bool = x4 <= x4;
        let x18: x17[3] = x0 as x17[3];
        let x19: bool = x3 | x11;
        let x20: u5 = x10 / u5:0xf;
        let x21: bool = !x11;
        let x22: u39 = match x10 {
            u5:0xa..u5:0xf => u39:0x3f_ffff_ffff,
            u5:0x1f..u5:0x13 => u39:0x0,
            _ => u39:0b10_1010_1010_1010_1010_1010_1010_1010_1010_1010,
        };
        let x23: u21 = x10 as u21 | x8;
        let x24: x17[6] = x18 ++ x18;
        let x25: u7 = match x7 {
            u21:0x0 => u7:0b10_1010,
            _ => u7:0x40,
        };
        let x26: bool = xor_reduce(x7);
        let x27: bool = x13 ^ x21;
        let x28: u61 = u61:0xfff_ffff_ffff_ffff;
        let x29: u49 = x28[12:];
        (x29, x19, x19, x15)
    }
}
