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
// exception: "Subprocess call failed: eval_proc_main sample.ir (SIGSEGV in JoinVals::computeAssignment)"
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
//       channel_name: "sample__x12"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x10"
//       values: "bits[13]:0x100"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x4"
//       values: "bits[13]:0x771"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x20"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x8"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1000"
//       values: "bits[13]:0x915"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x1240"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x10"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0x20"
//       values: "bits[13]:0xecf"
//       values: "bits[13]:0x1ea"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0x4"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x1cda"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x17f6"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0x1a0d"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x8"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0x5e7"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x80"
//       values: "bits[13]:0x1"
//       values: "bits[13]:0x164c"
//       values: "bits[13]:0x1dbc"
//       values: "bits[13]:0x40"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x20"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x8"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x10"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x55b"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x10"
//       values: "bits[13]:0x100"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x1334"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x7a1"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x40"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x20"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x1555"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x20"
//       values: "bits[13]:0x20"
//       values: "bits[13]:0x8"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xaaa"
//       values: "bits[13]:0x10"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0x10"
//       values: "bits[13]:0x1fff"
//       values: "bits[13]:0xfff"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0x0"
//       values: "bits[13]:0xfff"
//     }
//   }
// }
// 
// END_CONFIG
fn x27(x28: s5, x29: bool, x30: bool, x31: u12, x32: u3) -> (s5, u63, u63) {
    {
        let x33: u63 = match x30 {
            bool:0x1 => u63:0x7fff_ffff_ffff_ffff,
            bool:false | bool:true => u63:0x3fff_ffff_ffff_ffff,
            _ => u63:0x100_0000_0000,
        };
        (x28, x33, x33)
    }
}
proc main {
    x12: chan<s13> in;
    config(x12: chan<s13> in) {
        (x12,)
    }
    init {
        s5:-1
    }
    next(x0: s5) {
        {
            let x1: token = join();
            let x2: s5 = !x0;
            let x3: bool = x0 != x2;
            let x4: bool = !x3;
            let x5: u2 = decode<u2>(x3);
            let x6: s5 = x5 as s5 & x0;
            let x7: s24 = s24:0xff_ffff;
            let x8: bool = -x3;
            let x9: bool = x2 < x0;
            let x10: bool = x3 & x9 as bool;
            let x11: s38 = match x9 {
                bool:false => s38:0x15_5555_5555,
                xN[bool:0x0][1]:true => s38:0x15_5555_5555,
                _ => s38:0x40,
            };
            let x13: (token, s13) = recv(x1, x12);
            let x14: token = x13.0;
            let x15: s13 = x13.1;
            let x16: u3 = x4 ++ x4 ++ x4;
            let x17: u3 = -x16;
            let x18: bool = rev(x10);
            let x19: bool = x8 & x8;
            let x20: u7 = x19 ++ x19 ++ x9 ++ x3 ++ x17;
            let x21: token = join(x1);
            let x22: bool = x3 >> if x9 >= bool:0x0 { bool:0x0 } else { x9 };
            let x23: s13 = -x15;
            let x24: bool = x9 as bool & x18;
            let x25: bool = x4 + x3;
            let x26: u12 = x25 ++ x4 ++ x18 ++ x10 ++ x20 ++ x8;
            let x34: (s5, u63, u63) = x27(x2, x9, x4, x26, x16);
            let (..) = x27(x2, x9, x4, x26, x16);
            let x35: s42 = match x8 {
                bool:true => s42:0x5f_6241_5500,
                bool:0x1 => s42:0x1ff_ffff_ffff,
                bool:0b1 => s42:0x200_0000_0000,
                _ => s42:-1,
            };
            let x36: bool = x6 as bool ^ x8;
            let x37: bool = x22[x3+:bool];
            let x38: bool = x25 & x10;
            let x39: bool = !x9;
            let x40: s42 = !x35;
            x6
        }
    }
}
