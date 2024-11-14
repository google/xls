// Copyright 2024 The XLS Authors
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
// exception: "xls/passes/strength_reduction_pass.cc:234 !select->default_value().has_value() x6__1: bits[1] = sel(bit_slice.141, cases=[bit_slice.145], default=literal.148, id=147, pos=[(0,12,27)])"
// issue: "https://github.com/google/xls/issues/1711"
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
//   codegen_args: "--pipeline_stages=9"
//   codegen_args: "--worst_case_throughput=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: true
//   use_system_verilog: false
//   calls_per_sample: 0
//   proc_ticks: 100
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
proc main {
    x3: chan<u2> out;
    config(x3: chan<u2> out) {
        (x3,)
    }
    init {
        u2:0
    }
    next(x0: u2) {
        {
            let x1: u2 = x0[x0+:u2];
            let x4: token = send_if(join(), x3, bool:0x1, x0);
            let x6: u2 = x1[x1+:u2];
            let x7: u2 = !x6;
            let x8: u17 = u17:0xaaaa;
            let x9: u2 = x6 & x8 as u2;
            let x10: u47 = match x1 {
                u2:0x1 => u47:0x3fff_ffff_ffff,
                u2:0x2 => u47:65536,
                _ => u47:0x1246_cdc1_0f6b,
            };
            let x11: u38 = x0 ++ x8 ++ x8 ++ x7;
            let x12: u26 = match x10 {
                u47:0x0 => u26:0x80_0000,
                u47:0x2aaa_aaaa_aaaa => u26:0x348_d26d,
                _ => u26:0x155_5555,
            };
            let x13: u47 = x10 << if x10 >= u47:0x19 { u47:0x19 } else { x10 };
            let x14: u45 = u45:0x1835_80df_3232;
            let x15: u2 = x6[0:];
            let x16: u47 = x13 << if x0 >= u2:0x2 { u2:0x2 } else { x0 };
            let x17: u45 = -x14;
            let x18: token = join();
            let x19: u45 = x17 - x17;
            let x20: u41 = match x7 {
                u2:0x1 | u2:0x3 => u41:0x29_2127_542d,
                u2:0 => u41:0x191_123f_6407,
                _ => u41:0x0,
            };
            let x21: u4 = x20[37+:u4];
            let x22: bool = (x9 as u26) < x12;
            let x23: bool = x22 | x22;
            let x24: u26 = bit_slice_update(x12, x8, x15);
            let x25: u42 = x14[3+:u42];
            let x26: u25 = u25:0x1ff_ffff;
            let x27: u45 = -x17;
            x9
        }
    }
}
