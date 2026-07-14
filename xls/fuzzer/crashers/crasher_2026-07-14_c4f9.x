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
// exception: "/xls/tools/simulate_module_main returned a non-zero exit status (1): /xls/tools/simulate_module_main --signature_file=module_sig.textproto --testvector_textproto=testvector.pbtxt --output_channel_counts=sample__x29=128 --verilog_simulator=iverilog sample.v --logtostderr\n\nSubprocess stderr:\nError: INTERNAL: Cannot spawn child process: No such file or directory\n=== Source Location Trace: === \nxls/common/subprocess.cc:239\nxls/common/subprocess.cc:356\nxls/simulation/simulators/iverilog_simulator.cc:119\nxls/simulation/module_testbench.cc:704\nxls/simulation/module_simulator.cc:693\nxls/simulation/module_simulator.cc:742\nxls/tools/simulate_module_main.cc:187"
// issue: "https://github.com/google/xls/issues/TODO_PLACEHOLDER"
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
//   codegen_args: "--pipeline_stages=9"
//   codegen_args: "--worst_case_throughput=8"
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
//   }
// }
// 
// END_CONFIG
#![feature(channel_attributes)]

type x25 = u9;
fn x4(x5: bool, x6: (u10, u12, u34), x7: bool, x8: bool) -> (s1, bool, bool, s1, bool, bool, bool) {
    {
        let x9: bool = -x5;
        let x10: bool = x9[0+:bool];
        let x12: s1 = {
            let x11: (bool, bool) = smulp(x8 as s1, x5 as s1);
            (x11.0 + x11.1) as s1
        };
        let x13: bool = x7 & x5;
        (x12, x13, x10, x12, x7, x9, x10)
    }
}
proc main {
    #[channel_flow_control("valid_data")]
    x29: chan<u24> out;
    config(x29: chan<u24> out) {
        (x29,)
    }
    init {
        (u10:511, u12:2047, u34:392452782)
    }
    next(x0: (u10, u12, u34)) {
        {
            let x1: token = join();
            let x2: bool = x0 == x0;
            let x3: bool = x2 & x2;
            let x14: (s1, bool, bool, s1, bool, bool, bool) = x4(x2, x0, x2, x3);
            let (x15, x16, ..) = x4(x2, x0, x2, x3);
            let x17: u3 = x2 ++ x2 ++ x2;
            let x18: bool = !x3;
            let x19: token = join(x1);
            let x20: u7 = x16 ++ x16 ++ x18 ++ x17 ++ x18;
            let x21: token = join();
            let x22: bool = x2 - x20 as bool;
            let x23: bool = one_hot_sel(x17, [x2, x22, x16]);
            let x24: bool = x20 as bool - x18;
            let x26: bool = (x15 as bool)[:1];
            let x27: bool = -x23;
            let x28: u24 = u24:0xaa_aaaa;
            let x30: token = send(x21, x29, x28);
            let x31: bool = !x22;
            x0
        }
    }
}
