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
// exception: "simulate_module_main returned a non-zero exit status (1):\nCheck failed: simulator is OK (INTERNAL: Cannot spawn child process: No such file or directory)"
// issue: "<INSERT_ISSUE_URL_HERE>"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=7"
//   codegen_args: "--worst_case_throughput=3"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: true
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   channel_inputs {
//   }
// }
// 
// END_CONFIG
type x14 = bool;
proc main {
    config() {
        ()
    }
    init {
        u44:8796093022207
    }
    next(x0: u44) {
        {
            let x1: u44 = x0 | x0;
            let x2: bool = x1 > x0;
            let x3: bool = xor_reduce(x1);
            let x4: u44 = -x1;
            let x5: token = join();
            let x6: u1 = match x4 {
                u44:0xfff_ffff_ffff => bool:true,
                u44:0b0 | u44:0xd25_0dd9_ad40 => bool:true,
                _ => bool:false,
            };
            let x7: (bool, u44, bool, u44, u44, u44, u1) = (x2, x4, x2, x1, x1, x4, x6);
            let x8: u44 = -x0;
            let x9: u1 = !x6;
            let x10: u44 = x8 ^ x1;
            let x11: u44 = x0 | x10;
            let x12: bool = x0 as u1 > x6;
            let x13: bool = -x2;
            let x15: x14[1] = x12 as x14[1];
            let x16: bool = x4 as bool >= x3;
            let x17: bool = ctz(x16);
            let x18: u1 = !x6;
            let x19: u2 = decode<u2>(x2);
            let x20: u44 = x4 + x1;
            let x21: bool = x4 != x10;
            let x22: u1 = x16 as u1 | x18;
            let x23: u58 = u58:0b1_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101;
            let x24: bool = x16 ^ x13 as bool;
            let x25: u1 = x6 ^ x6;
            let x26: bool = x21[:];
            x1
        }
    }
}
