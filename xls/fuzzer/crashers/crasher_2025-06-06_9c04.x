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
//   codegen_args: "--pipeline_stages=2"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
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
type x5 = u63;
type x18 = bool;
proc main {
    config() {
        ()
    }
    init {
        u63:0
    }
    next(x0: u63) {
        {
            let x1: u63 = -x0;
            let x2: u63 = x0 << if x1 >= u63:0xb { u63:0xb } else { x1 };
            let x3: u63 = x0 | x2;
            let x4: u18 = match x1 {
                u63:0x0 => u18:0x58ac,
                u63:0x2aaa_aaaa_aaaa_aaaa | u63:128 => u18:0x0,
                _ => u18:0x3_7b47,
            };
            let x6: x5[4] = [x0, x1, x2, x3];
            let x7: u63 = !x2;
            let x8: bool = x6 != x6;
            let x9: u18 = x4[x2+:u18];
            let x10: x5 = x6[if x1 >= u63:0x0 { u63:0x0 } else { x1 }];
            let x11: u63 = !x3;
            let x12: token = join();
            let x13: x5[8] = x6 ++ x6;
            let x14: bool = or_reduce(x0);
            let x15: x5[12] = x6 ++ x13;
            let x16: bool = x8[:];
            let x17: u25 = x1[38+:u25];
            let x19: x18[1] = x16 as x18[1];
            let x20: u63 = x4 as u63 - x3;
            let x21: u27 = u27:0x0;
            let x22: bool = x14 / bool:false;
            let x23: x18 = x19[if x16 >= bool:0x0 { bool:0x0 } else { x16 }];
            let x24: u63 = clz(x2);
            let x25: u63 = x2[x9+:u63];
            let x26: u63 = clz(x24);
            let x27: u36 = u36:0x7_ffff_ffff;
            let x28: u13 = match x9 {
                u18:0x0..u18:0x3_ffff => u13:0xef0,
                u18:0x1_ffff => u13:0x1555,
                u18:0x1_0000 => u13:0xfff,
                _ => u13:0xaaa,
            };
            let x29: bool = bit_slice_update(x16, x7, x10);
            let x30: u13 = u13:0x1555;
            x26
        }
    }
}
