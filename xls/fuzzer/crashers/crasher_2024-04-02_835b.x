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
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full throughput.*"
//   }
// }
// inputs {
//   function_args {
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x30_4100"
//     args: "bits[42]:0x20_0000_0000; bits[2]:0x3; bits[22]:0x2e_2861"
//     args: "bits[42]:0x0; bits[2]:0x2; bits[22]:0x1f_73f7"
//     args: "bits[42]:0x1e5_83cd_e1c4; bits[2]:0x1; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x0; bits[22]:0x38_fdbf"
//     args: "bits[42]:0x2_0000; bits[2]:0x2; bits[22]:0x25_e3ff"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x3; bits[22]:0x37_540a"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x30_0000"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x800"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x35_5555"
//     args: "bits[42]:0x10_0000_0000; bits[2]:0x2; bits[22]:0x15_5555"
//     args: "bits[42]:0x1cc_af5f_95b8; bits[2]:0x1; bits[22]:0x37_d9b6"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x1; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x2; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x0; bits[2]:0x1; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x3d_cef5"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x0; bits[22]:0x9_c1c4"
//     args: "bits[42]:0x0; bits[2]:0x1; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x15_d5d5"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x2_f34c"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x2; bits[22]:0x15_5555"
//     args: "bits[42]:0x0; bits[2]:0x2; bits[22]:0x0"
//     args: "bits[42]:0x20_0000; bits[2]:0x2; bits[22]:0x80"
//     args: "bits[42]:0x2000; bits[2]:0x1; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x1d_1154"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x38_0042"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x1; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x32_2100"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x0"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x15_5555"
//     args: "bits[42]:0x8; bits[2]:0x2; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x1; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x1a_aba6"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x3; bits[22]:0x1d_0d47"
//     args: "bits[42]:0x233_97aa_f30b; bits[2]:0x3; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x299_ede4_e66e; bits[2]:0x1; bits[22]:0x24_e66d"
//     args: "bits[42]:0x200_0000; bits[2]:0x0; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x0; bits[2]:0x3; bits[22]:0x28_b000"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x15_6559"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x3; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x0; bits[22]:0x2f_2d4d"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x1; bits[22]:0x15_5555"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x2; bits[22]:0x17_59d6"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x1; bits[22]:0x15_0022"
//     args: "bits[42]:0x2d4_cb23_4fae; bits[2]:0x0; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x2; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x291_f2cc_c227; bits[2]:0x0; bits[22]:0x1000"
//     args: "bits[42]:0x2f8_90df_9877; bits[2]:0x3; bits[22]:0x30_b250"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x4d5f"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x2; bits[22]:0x20"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x1; bits[22]:0x3e_7fef"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x1; bits[22]:0x10_0500"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x3; bits[22]:0x37_dfff"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x2; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x1e6_6c1c_8666; bits[2]:0x1; bits[22]:0x1b_07de"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x2; bits[22]:0x35_eff0"
//     args: "bits[42]:0x10b_497c_6227; bits[2]:0x1; bits[22]:0x18_0c02"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x2; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x800_0000; bits[2]:0x2; bits[22]:0x2c_8d2e"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x15_5555"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x0; bits[22]:0xd_b257"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x10_1004"
//     args: "bits[42]:0x40_0000_0000; bits[2]:0x1; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x3; bits[22]:0x37_71ff"
//     args: "bits[42]:0x1fd_33b4_81e9; bits[2]:0x1; bits[22]:0x27_83a9"
//     args: "bits[42]:0x4_0000; bits[2]:0x2; bits[22]:0x36_0590"
//     args: "bits[42]:0xdf_81ed_1c1a; bits[2]:0x3; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x2; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x2; bits[22]:0x6_00c0"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x0"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0xf_ffff"
//     args: "bits[42]:0x106_c3b1_ba44; bits[2]:0x0; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x0; bits[2]:0x3; bits[22]:0x0"
//     args: "bits[42]:0x32_586d_bd63; bits[2]:0x3; bits[22]:0x35_0a80"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x36_1d5f"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x2a_11fb"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x2; bits[22]:0x8000"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x9_6220"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x3a_caa8"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x3; bits[22]:0x0"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0xd_40a0"
//     args: "bits[42]:0x0; bits[2]:0x2; bits[22]:0x15_5555"
//     args: "bits[42]:0x9d_b151_0efc; bits[2]:0x0; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x0; bits[2]:0x1; bits[22]:0x1e_ffff"
//     args: "bits[42]:0x1a2_a911_2c79; bits[2]:0x0; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x3c_2b4d"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x1f_f6ff"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x2; bits[22]:0x27_ffa7"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x1b_7519"
//     args: "bits[42]:0x20_0000; bits[2]:0x0; bits[22]:0x30_0080"
//     args: "bits[42]:0x2; bits[2]:0x2; bits[22]:0x0"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x2; bits[22]:0x15_5555"
//     args: "bits[42]:0x20_0000_0000; bits[2]:0x0; bits[22]:0x4c88"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x2; bits[22]:0x24_fe52"
//     args: "bits[42]:0x2; bits[2]:0x2; bits[22]:0x20_100d"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x3; bits[22]:0x9_696f"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x10_0004"
//     args: "bits[42]:0x155_5555_5555; bits[2]:0x1; bits[22]:0x15_5555"
//     args: "bits[42]:0x2000_0000; bits[2]:0x0; bits[22]:0x20_962c"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x0; bits[2]:0x2; bits[22]:0x3f_ffff"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x0; bits[22]:0x15_5555"
//     args: "bits[42]:0x2be_d6a5_41e2; bits[2]:0x1; bits[22]:0x31_216d"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x2; bits[22]:0x3f_75f7"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x2; bits[22]:0x2a_aaaa"
//     args: "bits[42]:0x270_6710_4505; bits[2]:0x2; bits[22]:0x1000"
//     args: "bits[42]:0x0; bits[2]:0x2; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x1; bits[22]:0x2000"
//     args: "bits[42]:0x2c9_041e_88e9; bits[2]:0x1; bits[22]:0x0"
//     args: "bits[42]:0x0; bits[2]:0x0; bits[22]:0xc_0675"
//     args: "bits[42]:0x3ff_ffff_ffff; bits[2]:0x3; bits[22]:0x0"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x2; bits[22]:0x2c_61dc"
//     args: "bits[42]:0x8_0000_0000; bits[2]:0x0; bits[22]:0x1f_ffff"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x3; bits[22]:0x5_afc2"
//     args: "bits[42]:0x1ff_ffff_ffff; bits[2]:0x3; bits[22]:0x38_aaea"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x3; bits[22]:0x8"
//     args: "bits[42]:0x1_0000_0000; bits[2]:0x2; bits[22]:0x2201"
//     args: "bits[42]:0x0; bits[2]:0x1; bits[22]:0x200"
//     args: "bits[42]:0x2aa_aaaa_aaaa; bits[2]:0x1; bits[22]:0x2a_aa8a"
//   }
// }
// 
// END_CONFIG
type x5 = u2;
type x25 = bool;
fn main(x0: u42, x1: u2, x2: u22) -> (u2, u42, bool, (u2, u42, bool, bool, u42, u2), bool) {
    {
        let x3: u2 = x1[x1+:u2];
        let x4: u34 = u34:0x1_ffff_ffff;
        let x6: x5[1] = [x1];
        let x7: u2 = signex(x0, x3);
        let x8: bool = x4 <= x1 as u34;
        let x9: u2 = x3[0+:u2];
        let x10: bool = x1[1:];
        let x11: u42 = bit_slice_update(x0, x8, x1);
        let x12: (u2, u42, bool, bool, u42, u2) = (x9, x11, x8, x8, x11, x1);
        let x13: u42 = x11 >> if x8 >= bool:false { bool:false } else { x8 };
        let x14: bool = x12 == x12;
        let x15: bool = x8 | x10 as bool;
        let x16: u2 = -x9;
        let x17: bool = or_reduce(x13);
        let x18: u2 = x9 & x2 as u2;
        let x19: bool = x3 as u42 > x13;
        let x20: u2 = clz(x18);
        let x21: bool = gate!(x8 as u2 <= x20, x15);
        let x22: bool = x8 <= x4 as bool;
        let x23: bool = x6 == x6;
        let x24: u2 = x16[-3:];
        let x26: x25[1] = x22 as x25[1];
        let x27: u42 = rev(x11);
        (x7, x11, x8, x12, x22)
    }
}
