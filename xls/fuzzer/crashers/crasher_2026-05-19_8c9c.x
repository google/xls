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
// exception: "Subprocess call timed out after 1500 seconds: /xls/tools/opt_main sample.ir --logtostderr"
// issue: "https://github.com/google/xls/issues/4305"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   channel_inputs {
//   }
// }
//
// END_CONFIG
fn x1(x2: uN[1490], x3: uN[1490], x4: uN[1490], x5: uN[1490], x6: uN[1490], x7: uN[1490]) -> (uN[1490], uN[1490]) {
    {
        let x8: uN[718] = x7[:718];
        let x9: uN[1490] = rev(x2);
        let x10: bool = x4 == x6;
        (x9, x9)
    }
}
proc main {
    config() {
        ()
    }
    init {
        uN[1490]:0x1_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff
    }
    next(x0: uN[1490]) {
        {
            let x11: (uN[1490], uN[1490]) = x1(x0, x0, x0, x0, x0, x0);
            let (x12, x13) = x1(x0, x0, x0, x0, x0, x0);
            let x14: xN[bool:0x0][611] = x13[0+:uN[611]];
            let x15: uN[1490] = -x12;
            let x16: uN[1490] = x14 as uN[1490] | x12;
            let x17: uN[288] = x14[x15+:uN[288]];
            let x18: uN[1490] = x15 - x16;
            let x19: uN[1490] = !x12;
            let x20: uN[2027] = decode<uN[2027]>(x12);
            let x21: bool = x11 == x11;
            let x22: uN[1490] = bit_slice_update(x15, x14, x17);
            let x23: s27 = match x17 {
                uN[288]:0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000..=uN[288]:0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff => s27:0x7ff_ffff,
                uN[288]:0x400_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000 | uN[288]:0x0 => xN[bool:0x1][27]:0b1_0000_0000_0000_0000_0000,
                _ => s27:0x0,
            };
            let x24: uN[1490] = !x22;
            let x25: uN[1490] = x15 + x19;
            let x26: bool = x11 != x11;
            let x27: token = join();
            let x28: s27 = !x23;
            let x29: uN[288] = !x17;
            let x30: uN[1490] = x12[x15+:uN[1490]];
            let x31: uN[1490] = x0 << if x22 >= uN[1490]:0x64 { uN[1490]:0x64 } else { x22 };
            let x32: token = join(x27);
            let x34: sN[1490] = {
                let x33: (uN[1490], uN[1490]) = smulp(x15 as sN[1490], x16 as sN[1490]);
                (x33.0 + x33.1) as sN[1490]
            };
            let x35: u11 = encode(x0);
            x31
        }
    }
}
