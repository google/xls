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
// exception: "/xls/tools/simulate_module_main returned a non-zero exit status (1): /xls/tools/simulate_module_main --signature_file=module_sig.textproto --testvector_textproto=testvector.pbtxt --output_channel_counts= --verilog_simulator=iverilog sample.v --logtostderr\n\nSubprocess stderr:\nF0920 06:53:38.151404 2221904 simulate_module_main.cc:239] Check failed: simulator is OK (INTERNAL: Cannot spawn child process: No such file or directory\n=== Source Location Trace: ===\nxls/common/subprocess_with_wrapper.cc:86\nxls/common/subprocess.cc:167\nxls/common/subprocess.cc:278\n) Unknown verilog_simulator \'iverilog\'\n"
// issue: "https://github.com/google/xls/issues/5031"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=7"
//   codegen_args: "--worst_case_throughput=6"
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
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x11"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x1"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//       values: "bits[1]:0x0"
//     }
//   }
// }
//
// END_CONFIG
#![feature(channel_attributes)]

type x1 = u6;
proc main {
    #[channel_flow_control("valid_data")]
    x11: chan<bool> in;
    config(x11: chan<bool> in) {
        (x11,)
    }
    init {
        u6:63
    }
    next(x0: u6) {
        {
            let x2: x1[1] = [x0];
            let x3: u6 = one_hot_sel(x0, [x0, x0, x0, x0, x0, x0]);
            let x4: u18 = x3 ++ x3 ++ x3;
            let x5: u19 = one_hot(x4, bool:0x0);
            let x6: u19 = -x5;
            let x7: u2 = x5[0+:u2];
            let x8: bool = and_reduce(x5);
            let x9: token = join();
            let x10: bool = x8 != x7 as bool;
            let x12: (token, bool) = recv(x9, x11);
            let x13: token = x12.0;
            let x14: bool = x12.1;
            let x15: u19 = x6[x3+:u19];
            let x16: u4 = x6[6+:u4];
            let x17: u4 = x16 * x4 as u4;
            let x18: x1 = x2[if x7 >= u2:0x0 { u2:0x0 } else { x7 }];
            let x19: bool = !x14;
            let x20: x1[2] = x2 ++ x2;
            let x21: bool = x14 << if x0 >= u6:0x0 { u6:0x0 } else { x0 };
            let x22: u6 = x3[x16+:u6];
            let x23: x1 = x20[if x18 >= x1:0 { x1:0 } else { x18 }];
            let x24: bool = x14[x22+:bool];
            let x25: u4 = x16[:];
            let x26: x1[4] = x20 ++ x20;
            let x27: x1 = x26[if x4 >= u18:0x2 { u18:0x2 } else { x4 }];
            x3
        }
    }
}
