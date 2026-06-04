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
// exception: "codegen_main returned a non-zero exit status (1):\nError: UNIMPLEMENTED: Proc combinational generator only supports streaming output channels which can be determined to be mutually exclusive, got 2 output channels which were not proven to be mutually exclusive; Running pass #6: Channel I/O to port lowering pass [short: channel_to_port_io_lowering]; Failed as part of compound pass block_conversion #0; Running pass #0: Top level codegen v1.5 block conversion pipeline [short: block_conversion]"
// issue: "https://github.com/google/xls/issues/4361"
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
//   codegen_args: "--generator=combinational"
//   codegen_args: "--reset_data_path=false"
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
//   }
// }
// 
// END_CONFIG
type x1 = u52;
proc main {
    x6: chan<bool> out;
    x18: chan<u5> out;
    config(x6: chan<bool> out, x18: chan<u5> out) {
        (x6, x18)
    }
    init {
        ()
    }
    next(x0: ()) {
        {
            let x2: u5 = u5:0b1010;
            let x3: token = join();
            let x4: token = join();
            let x5: bool = x0 == x0;
            let x7: token = send_if(x4, x6, x5, x5);
            let x8: u3 = x2[-3:];
            let x9: bool = x5 + x8 as bool;
            let x10: bool = x0 == x0;
            let x11: bool = x5[x8+:bool];
            let x12: bool = xor_reduce(x8);
            let x13: bool = x11 as bool & x5;
            let x14: bool = x10[0:];
            let x15: bool = x10 >> if x13 >= bool:0x0 { bool:0x0 } else { x13 };
            let x16: bool = -x5;
            let x17: bool = !x14;
            let x19: token = send_if(x4, x18, x9, x2);
            let x20: bool = x10[0+:bool];
            let x21: u28 = match x9 {
                bool:0b1 => u28:0x555_5555,
                bool:true | bool:false => u28:0xaaa_aaaa,
                _ => u28:0xaaa_aaaa,
            };
            let x22: xN[bool:0x0][2] = decode<u2>(x9);
            x0
        }
    }
}
