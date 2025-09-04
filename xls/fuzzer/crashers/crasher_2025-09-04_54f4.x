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
// exception: "Subprocess call timed out after 1500 seconds"
// issue: "https://github.com/google/xls/issues/2998"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=6"
//   codegen_args: "--worst_case_throughput=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   simulate: true
//   simulator: "iverilog"
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
//   codegen_ng: true
//   disable_unopt_interpreter: false
// }
// inputs {
//   channel_inputs {
//   }
// }
// 
// END_CONFIG
type x27 = bool;
proc main {
    config() {
        ()
    }
    init {
        s35:0
    }
    next(x0: s35) {
        {
            let x1: s35 = -x0;
            let x2: s35 = x1 & x1;
            let x3: (s35, s35, s35) = (x0, x2, x0);
            let (x4, x5, x6): (s35, s35, s35) = (x0, x2, x0);
            let x7: s35 = signex(x4, x2);
            let x8: u10 = match x1 {
                s35:0x7_ffff_ffff => u10:682,
                s35:0x2_60c5_fc82 | s35:0x800 => u10:0x1ff,
                _ => u10:0x0,
            };
            let x9: u2 = x8[8+:u2];
            let x10: token = join();
            let x11: s35 = -x6;
            let x12: u2 = rev(x9);
            let x13: u2 = x9[x8+:u2];
            let x14: u2 = x12;
            let x15: bool = x9[0+:bool];
            let x16: bool = for (i, x) in u4:0x0..=u4:0x4 {
                x
            }(x15);
            let x17: u2 = -x9;
            let x18: bool = x2 < x13 as s35;
            let x19: bool = x16[0+:bool];
            let x20: bool = one_hot_sel(x13, [x18, x19]);
            let x21: u2 = x20 ++ x20;
            let x22: u10 = ctz(x8);
            let x23: s35 = x19 as s35 + x6;
            let x24: bool = x11 >= x0;
            let x25: token = join(x10);
            let x26: bool = x3 == x3;
            let x28: x27[2] = [x18, x20];
            let x29: u10 = !x22;
            let x30: bool = x28 == x28;
            let x31: u2 = !x13;
            let x32: u10 = one_hot_sel(x21, [x8, x22]);
            let x33: u2 = x21[0+:u2];
            let x34: x27 = x28[if x33 >= u2:0b1 { u2:0b1 } else { x33 }];
            let x35: bool = x26 as bool | x20;
            x2
        }
    }
}
