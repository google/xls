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
// exception: "Subprocess call failed: /xls/tools/opt_main sample.ir --logtostderr"
// issue: "https://github.com/google/xls/issues/1514"
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
//   codegen_args: "--pipeline_stages=5"
//   codegen_args: "--worst_case_throughput=3"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified
//     pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full
//     throughput.*"
//   }
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x10"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x20"
//       values: "bits[12]:0xb39"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x967"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x10"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x2c0"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x361"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x40"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x80"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x200"
//       values: "bits[12]:0xf1a"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x6ee"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x20"
//       values: "bits[12]:0xdea"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xb35"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x4"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x2"
//       values: "bits[12]:0x5a5"
//       values: "bits[12]:0x4"
//       values: "bits[12]:0x20"
//       values: "bits[12]:0x8"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x944"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x100"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x2"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x470"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x200"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x69c"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x80"
//       values: "bits[12]:0x20"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x400"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0xf28"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x4"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0x1"
//       values: "bits[12]:0x8"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x755"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xfa0"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0xaaa"
//       values: "bits[12]:0xec3"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x555"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0x7ff"
//       values: "bits[12]:0x0"
//       values: "bits[12]:0xfff"
//       values: "bits[12]:0x40"
//     }
//   }
// }
//
// END_CONFIG
const W32_V7 = u32:0x7;

type x9 = u36;

proc main {
    x10: chan<u12> in;

    config(x10: chan<u12> in) { (x10,) }

    init { u43:0 }

    next(x0: u43) {
        {
            let x1: u43 = !x0;
            let x2: u43 = x0 / u43:2932031007402;
            let x3: u43 = x1;
            let x4: u43 = for (i, x) in u4:0x0..u4:0x8 {
                x
            }(x3);
            let x5: u43 = x2 >> if x3 >= u43:0x8 { u43:0x8 } else { x3 };
            let x6: u43 = x2 - x4 as u43;
            let x7: u43 = x3[0+:u43];
            let x8: u55 = u55:0x55_5555_5555_5555;
            let x11: (token, u12) = recv(join(), x10);
            let x12: token = x11.0;
            let x13: u12 = x11.1;
            let x14: u18 = match x5 {
                u43:0x2aa_aaaa_aaaa | u43:8192 => u18:0x3_ffff,
                u43:0x7ff_ffff_ffff | u43:0x0 => u18:0x1_ffff,
                _ => u18:0x3_ffff,
            };
            let x15: u18 = !x14;
            let x16: u43 = bit_slice_update(x7, x0, x14);
            let x17: u12 = x13 >> if x14 >= u18:0xa { u18:0xa } else { x14 };
            let x18: bool = x4 as u43 != x6;
            let x19: u18 = !x14;
            let x20: u18 = x19 * x15;
            let x21: token = join();
            let x22: u43 = -x7;
            let x23: token = join(x12);
            let x24: token = for (i, x): (u4, token) in u4:0x0..u4:0x2 {
                x
            }(x21);
            let x25: token = join(x23);
            let x26: bool = x18[0+:bool];
            let x27: token = join(x25, x25, x12);
            let x28: u18 = !x20;
            let x29: u43 = bit_slice_update(x16, x28, x13);
            let x30: u55 = rev(x8);
            x29
        }
    }
}
