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
// exception: "Subprocess call failed: eval_proc_main sample.ir (SIGILL in JoinVals::computeAssignment)"
// issue: "https://github.com/google/xls/issues/4446"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x27"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x3b2_f097"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x1f0_5ce5"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x398_14cd"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x1"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x12b_0fc5"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x4_0000"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x800"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x387_5a71"
//       values: "bits[26]:0x1"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x2bf_360f"
//       values: "bits[26]:0x133_3c69"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x167_c089"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x80_0000"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x366_25cf"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x1e0_f6ba"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x36d_ef1b"
//       values: "bits[26]:0x2_0000"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x80_0000"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x80"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x1_0000"
//       values: "bits[26]:0x20_0000"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x3f8_cdf9"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x40"
//       values: "bits[26]:0x14f_f002"
//       values: "bits[26]:0x3fd_eef2"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x2fa_609d"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x76_9420"
//       values: "bits[26]:0x400"
//       values: "bits[26]:0x8"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x80_0000"
//       values: "bits[26]:0x11c_846d"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x1a6_9133"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x32c_3293"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x3ff_ffff"
//       values: "bits[26]:0x200_0000"
//       values: "bits[26]:0x1ff_ffff"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x2_0000"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x13f_b034"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x2aa_aaaa"
//       values: "bits[26]:0x1_0000"
//       values: "bits[26]:0x155_5555"
//       values: "bits[26]:0x0"
//       values: "bits[26]:0x287_cfef"
//     }
//   }
// }
// 
// END_CONFIG
fn x11(x12: s16, x13: bool, x14: s16, x15: (s16, s16)) -> (u15, u11, u15) {
    {
        let x16: u11 = (x14 as u16)[x13+:u11];
        let x17: u15 = x13 ++ x13 ++ x13 ++ x16 ++ x13;
        (x17, x16, x17)
    }
}
proc main {
    x27: chan<u26> in;
    config(x27: chan<u26> in) {
        (x27,)
    }
    init {
        s16:-21846
    }
    next(x0: s16) {
        {
            let x1: s16 = !x0;
            let x2: (s16, s16) = (x1, x1);
            let (x3, x4) = (x1, x1);
            let x5: s16 = -x1;
            let x6: token = join();
            let x7: bool = x0 <= x3;
            let x8: bool = or_reduce(x7);
            let x9: u5 = x8 ++ x8 ++ x8 ++ x7 ++ x8;
            let x10: u6 = u6:0x15;
            let x18: (u15, u11, u15) = x11(x1, x7, x1, x2);
            let (_, x19, x20): (u15, u11, u15) = x11(x1, x7, x1, x2);
            let x21: s16 = x5 * x5;
            let x22: u28 = x19 ++ x7 ++ x9 ++ x19;
            let x23: u35 = u35:0x2_aaaa_aaaa;
            let x24: s16 = -x3;
            let x25: uN[2048] = decode<xN[bool:0x0][2048]>(x22);
            let x26: bool = x8[x10+:bool];
            let x28: (token, u26) = recv(x6, x27);
            let x29: token = x28.0;
            let x30: u26 = x28.1;
            let x31: bool = x2 != x2;
            let x32: u6 = one_hot_sel(x10, [x10, x10, x10, x10, x10, x10]);
            let x33: u6 = bit_slice_update(x10, x19, x32);
            let x34: uN[2048] = rev(x25);
            let x35: bool = xor_reduce(x9);
            let x36: u11 = -x19;
            let x37: bool = x26 << if x7 >= bool:0x0 { bool:0x0 } else { x7 };
            let x38: bool = one_hot_sel(x33, [x8, x31, x8, x8, x37, x31]);
            x24
        }
    }
}
