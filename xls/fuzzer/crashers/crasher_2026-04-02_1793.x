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
// exception: "Subprocess call failed: /xls/tools/opt_main
// /tmp/896876837/tmp/test_tmpdirox_b__30/temp_directory_3wOHfJ/sample.ir
// --logtostderr\n\nSubprocess stderr:\n*** SIGABRT received by PID 1484423 (TID 1484423) on cpu 80,
// si_code=-6, from PID 1484423; stack trace: ***\nPC: @     0x7fcf0863e961  (unknown)  gsignal\n   
// @     0x562748ad28cf        256  gloop/base/process_state.cc:1198 FailureSignalHandler()\n    @  
//   0x7fcf087b5a60  1953931456  (unknown)\n    @     0x562747fbc7ca        448 
// xls/passes/reassociation_pass.cc:0 xls::(anonymous
// namespace)::OneShotReassociationVisitor::HandleNeg()\n    @     0x5627484e1a86        128 
// xls/ir/node.cc:269 xls::Node::VisitSingleNode()\n    @     0x562747fb4d84        800 
// xls/passes/reassociation_pass.cc:827 xls::(anonymous
// namespace)::ReassociationCache::ComputeInfo()\n    @     0x562747fb46c8        224 
// ./xls/passes/lazy_node_data.h:462 xls::LazyNodeData<>::ComputeValue()\n    @     0x562747fb66c2  
//       32  ./xls/passes/lazy_node_data.h:0 xls::LazyNodeData<>::ComputeValue()\n"
// issue: "Fixed"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the
//     specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full
//     throughput.*"
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
proc main {
    config() { () }

    init { u31:2147483647 }

    next(x0: u31) {
        {
            let x1: u31 = for (i, x): (u4, u31) in u4:0x0..u4:0x5 {
                x
            }(x0);
            let x2: token = join();
            let x3: u31 = x1 - x0;
            let x4: u26 = u26:0x18d_dbce;
            let x5: u31 = -x3;
            let x6: bool = x0 <= x1;
            let x7: u26 = x4[0+:u26];
            let x8: u26 = x4 / u26:0x0;
            let x9: bool = x6[0+:bool];
            let x10: bool = x6 as bool | x9;
            let x11: bool = or_reduce(x6);
            let x12: u31 = -x1;
            let x13: bool = or_reduce(x7);
            let x14: u33 = x10 ++ x13 ++ x0;
            let x16: s26 = {
                let x15: (u26, u26) = smulp(x8 as s26, x10 as u26 as s26);
                (x15.0 + x15.1) as s26
            };
            let x17: bool = x11 << if x5 >= u31:0x0 { u31:0x0 } else { x5 };
            let x18: u6 = encode(x14);
            let x19: bool = x9[x14+:bool];
            let x20: (bool, bool, u26, u31, u31, u31, u31) = (x17, x6, x7, x12, x12, x12, x3);
            let x21: u31 = x12 & x19 as u31;
            let x22: bool = bit_slice_update(x13, x13, x1);
            let x23: u63 = x1 ++ x22 ++ x21;
            let x24: bool = x9 / bool:false;
            let x25: xN[bool:0x0][27] = one_hot(x4, bool:0x0);
            let x26: u27 = rev(x25);
            x5
        }
    }
}
