// Copyright 2023 The XLS Authors
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
// exception: 	 "Subprocess call failed: /xls/tools/eval_proc_main --inputs_for_all_channels=channel_inputs.txt --ticks=1 --backend=serial_jit sample.ir --logtostderr"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=7"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full throughput.*"
//   }
// }
// inputs {
//   channel_inputs {
//   }
// }
// 
// END_CONFIG
const W32_V7 = u32:0x7;
type x1 = (s13, uN[102]);
type x5 = s63;
type x6 = x1[13];
fn x8(x9: token, x10: token, x11: x1[26], x12: x1[13], x13: x1[26], x14: x1[13]) -> (token, x1[39], bool, bool, x1[13]) {
    {
        let x15: x1[39] = x11 ++ x12;
        let x16: bool = x14 != x12;
        let x17: bool = x16[x16+:bool];
        let x18: bool = x17 as bool ^ x16;
        (x9, x15, x16, x18, x12)
    }
}
proc main {
    config() {
        ()
    }
    init {
        [(s13:0, uN[102]:0x3f_ffff_ffff_ffff_ffff_ffff_ffff), (s13:4095, uN[102]:0x2a_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa), (s13:-2731, uN[102]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff), (s13:4095, uN[102]:0x15_5555_5555_5555_5555_5555_5555), (s13:64, uN[102]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff), (s13:-1, uN[102]:0x25_dc9c_fcd2_edad_5755_55e5_8e3b), (s13:4095, uN[102]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff), (s13:0, uN[102]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff), (s13:2473, uN[102]:0x15_5555_5555_5555_5555_5555_5555), (s13:-2731, uN[102]:0x9_9abc_3e22_6249_b180_9fad_caea), (s13:0, uN[102]:0x15_5555_5555_5555_5555_5555_5555), (s13:2730, uN[102]:0x0), (s13:0, uN[102]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff)]
    }
    next(x2: x1[13]) {
        {
            let x0: token = join();
            let x3: token = join(x0);
            let x4: x1[26] = x2 ++ x2;
            let x7: x6[1] = [x2];
            let x19: (token, x1[39], bool, bool, x1[13]) = x8(x0, x0, x4, x2, x4, x2);
            let x20: token = x19.0;
            let x21: x1[39] = x19.1;
            let x22: bool = x19.2;
            let x23: bool = x19.3;
            let x24: x1[13] = x19.4;
            let x25: bool = -x23;
            let x26: bool = x25 & x23;
            let x27: bool = x23 | x22;
            let x28: s15 = s15:0x2aaa;
            let x29: x1[13] = x19.4;
            let x30: bool = and_reduce(x22);
            let x31: x1[10] = slice(x29, x22, x1[10]:[x29[u32:0x0], ...]);
            let x32: (bool, bool, x1[13], x6[1]) = (x30, x23, x2, x7);
            x2
        }
    }
}
