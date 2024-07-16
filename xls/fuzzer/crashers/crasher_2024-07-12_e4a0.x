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
// exception: "Subprocess call failed: /xls/tools/opt_main sample.ir --logtostderr"
// issue: "https://github.com/google/xls/issues/1511"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
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
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x13"
//       values: "bits[55]:0x4000_0000_0000"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x21_70e2_9ff0_10b1"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x32_df6d_4a4e_d445"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x66_1936_7266_6905"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x200_0000_0000"
//       values: "bits[55]:0x6d_b223_e940_06a6"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x7a_74af_fcfa_5069"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x400_0000"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x8000_0000_0000"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x4000_0000_0000"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x3d_3e94_07e5_e348"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x8000_0000"
//       values: "bits[55]:0x1_0000_0000"
//       values: "bits[55]:0x23_43ab_454d_5937"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0xd_cd72_2cf3_cae8"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x70_5b81_e0a4_cd9b"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x400"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x2"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x1_0000_0000_0000"
//       values: "bits[55]:0x40_0000_0000_0000"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x1_0000_0000"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0xb_c120_8cc9_7c84"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x32_c27a_c0b7_4070"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x7c_59c8_fa6f_2ad5"
//       values: "bits[55]:0x2000_0000"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x3f_ffff_ffff_ffff"
//       values: "bits[55]:0x40_0000_0000"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x7f_08aa_21bd_b994"
//       values: "bits[55]:0x2a_aaaa_aaaa_aaaa"
//       values: "bits[55]:0x7f_ffff_ffff_ffff"
//       values: "bits[55]:0x7f_9e59_05d1_6e0f"
//       values: "bits[55]:0x55_5555_5555_5555"
//       values: "bits[55]:0x11_9ebc_df13_397e"
//       values: "bits[55]:0x0"
//       values: "bits[55]:0x61_7f09_f7ea_6e14"
//     }
//   }
// }
// 
// END_CONFIG
fn x4(x5: u55, x6: sN[77], x7: sN[77], x8: s39, x9: sN[77]) -> (sN[77], u55, u55, sN[77], s39, u55, sN[77]) {
    {
        let x10: u55 = x6 as u55 - x5;
        let x11: sN[77] = x9 & x6;
        (x11, x10, x10, x11, x8, x10, x11)
    }
}
proc main {
    x13: chan<u55> in;
    config(x13: chan<u55> in) {
        (x13,)
    }
    init {
        s39:262144
    }
    next(x0: s39) {
        {
            let x1: sN[77] = match x0 {
                s39:0x55_5555_5555..s39:0b10_1010_1010_1010_1010_1010_1010_1010_1010_1010 => sN[77]:0x0,
                s39:0x2a_aaaa_aaaa => sN[77]:0x1e15_ebf8_2ff4_3384_49a1,
                _ => sN[77]:0x0,
            };
            let x2: sN[77] = gate!(x0 >= x0, x1);
            let x3: u55 = (x1 as uN[77])[22+:u55];
            let x12: (sN[77], u55, u55, sN[77], s39, u55, sN[77]) = x4(x3, x2, x1, x0, x2);
            let x14: (token, u55) = recv_if(join(), x13, bool:0x1, x3);
            let x15: token = x14.0;
            let x16: u55 = x14.1;
            let x17: token = join();
            let x18: u55 = x3[0+:u55];
            let x19: s39 = x0 * x16 as s39;
            let x20: u55 = x0 as u55 * x3;
            let x21: bool = (x16 as sN[77]) < x2;
            let x23: token = join(x15);
            let x24: uN[165] = x3 ++ x16 ++ x3;
            let x25: bool = x21 << if x21 >= bool:false { bool:false } else { x21 };
            let x26: bool = x12 == x12;
            let x27: uN[111] = x18 ++ x3 ++ x21;
            let x28: u55 = x18 ^ x18;
            let x29: bool = or_reduce(x21);
            let x30: uN[111] = x27 & x3 as uN[111];
            let x31: uN[68] = (x2 as uN[77])[9:];
            let x32: bool = x26[:];
            let x33: s39 = one_hot_sel(x25, [x19]);
            let x34: bool = x12 != x12;
            let x35: bool = x12 == x12;
            let x36: u55 = x16 * x3 as u55;
            let x37: u55 = x35 as u55 + x18;
            let x38: s39 = -x0;
            let x39: u55 = x37 / u55:0x3f_ffff_ffff_ffff;
            x33
        }
    }
}
