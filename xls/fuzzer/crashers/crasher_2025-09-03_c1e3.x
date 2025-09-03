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
// exception: "Error: UNIMPLEMENTED: Proc combinational generator only supports streaming output channels which can be determined to be mutually exclusive, got 2 output channels which were not proven to be mutually exclusive\nError: INTERNAL: /usr/local/google/home/davidplass/.cache/bazel/_bazel_davidplass/7af3f84567cca338af87a04c2463db99/sandbox/linux-sandbox/7175/execroot/_main/bazel-out/k8-opt/bin/xls/fuzzer/run_crasher_test_2025-09-03_c1e3.runfiles/_main/xls/tools/codegen_main returned a non-zero exit status (1): /usr/local/google/home/davidplass/.cache/bazel/_bazel_davidplass/7af3f84567cca338af87a04c2463db99/sandbox/linux-sandbox/7175/execroot/_main/bazel-out/k8-opt/bin/xls/fuzzer/run_crasher_test_2025-09-03_c1e3.runfiles/_main/xls/tools/codegen_main --output_signature_path=module_sig.textproto --delay_model=unit --nouse_system_verilog --output_block_ir_path=sample.block.ir --generator=combinational --reset_data_path=false /tmp/temp_directory_quYVZC/sample.opt.ir --logtostderr\nSubprocess stderr:\n Error: UNIMPLEMENTED: Proc combinational generator only supports streaming output channels which can be determined to be mutually exclusive, got 2 output channels which were not proven to be mutually exclusive\n"
// issue: "https://github.com/google/xls/issues/2989"
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
//   codegen_args: "--generator=combinational"
//   codegen_args: "--reset_data_path=false"
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
//   codegen_ng: false
//   disable_unopt_interpreter: false
// }
// inputs {
//   channel_inputs {
//   }
// }
// 
// END_CONFIG
type x6 = u45;
proc main {
    x23: chan<u24> out;
    x31: chan<u5> out;
    config(x23: chan<u24> out, x31: chan<u5> out) {
        (x23, x31)
    }
    init {
        ()
    }
    next(x0: ()) {
        {
            let x1: u42 = u42:0x3ff_ffff_ffff;
            let x2: bool = xor_reduce(x1);
            let x3: u42 = x1 ^ x1;
            let x4: u42 = x1 ^ x1;
            let x5: u42 = -x4;
            let x7: u42 = x4 >> 39;
            let x8: u42 = rev(x5);
            let x9: xN[bool:0x0][42] = x8[:];
            let x10: ((), u42, u42) = (x0, x8, x3);
            let x11: bool = -x2;
            let x12: u42 = bit_slice_update(x1, x1, x7);
            let x14: bool = or_reduce(x9);
            let x15: bool = x2 ^ x2;
            let x16: u24 = x5[x3+:u24];
            let x17: u42 = x7 & x14 as u42;
            let x18: bool = !x2;
            let x20: u5 = x16[x9+:u5];
            let x21: bool = x10 != x10;
            let x22: u42 = x9[0+:u42];
            let x24: token = send_if(join(), x23, x11, x16);
            let x25: xN[bool:0x0][42] = x9 >> 6;
            let x26: bool = x14 - x16 as bool;
            let x27: bool = x2 & x21 as bool;
            let x28: xN[bool:0x0][42] = x9 | x25;
            let x29: bool = -x14;
            let x30: bool = x14[x4+:bool];
            let x32: token = send(x24, x31, x20);
            let x33: u5 = x12 as u5 + x20;
            let x34: bool = x20 <= x33;
            x0
        }
    }
}
