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
// exception: 	 "/xls/tools/codegen_main returned non-zero exit status: 1"
// issue: "https://github.com/google/xls/issues/1140"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: false
//   use_system_verilog: true
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x40"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x100_0000_0000_0000"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x423_5a5a_53ff_d185"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0xe_2a71_c764_9935"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x1b5_3845_89f9_c691"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x200_0000"
//       values: "bits[60]:0x40_0000_0000_0000"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x20"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x130_9860_0079_54a6"
//       values: "bits[60]:0xe7f_0004_c572_4e9e"
//       values: "bits[60]:0x5bb_b681_8138_a39c"
//       values: "bits[60]:0x800_0000"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x186_1445_92b4_dac4"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xe88_9195_9837_8477"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x6f3_e410_fcf2_b806"
//       values: "bits[60]:0x200_0000"
//       values: "bits[60]:0x1"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0xe85_dd30_b9d9_1649"
//       values: "bits[60]:0xf83_4a38_e0b3_993c"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x40_0000_0000_0000"
//       values: "bits[60]:0x10"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x800_0000_0000"
//       values: "bits[60]:0x8000_0000"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x427_c792_87ed_6339"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x2000_0000"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xe6e_d98b_5f5b_4688"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x100_0000_0000"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xfde_a399_0922_59e9"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xb24_d7b6_3215_c666"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xb98_0ac1_9d0f_9f0e"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xf03_89cb_6f92_c412"
//       values: "bits[60]:0x9eb_5eec_0b43_2ae4"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x8e5_9029_960f_bbc9"
//       values: "bits[60]:0x344_883f_077a_423f"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xb64_b17c_cf3f_5308"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x1000_0000_0000"
//       values: "bits[60]:0x5a2_0794_953f_bb43"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x4_0000_0000_0000"
//       values: "bits[60]:0x2000"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x40_0000_0000_0000"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x100_0000"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//     }
//   }
// }
// 
// END_CONFIG
const W32_V1 = u32:0x1;
type x6 = bool;
type x30 = bool;
proc main {
    x8: chan<bool> out;
    x40: chan<u60> in;
    config(x8: chan<bool> out, x40: chan<u60> in) {
        (x8, x40)
    }
    init {
        u27:89478485
    }
    next(x1: u27) {
        {
            let x0: token = join();
            let x2: u41 = u41:0x155_5555_5555;
            let x3: u16 = x2[16+:u16];
            let x4: bool = x1 >= x3 as u27;
            let x5: token = join(x0);
            let x7: x6[W32_V1] = x4 as x6[W32_V1];
            let x9: token = send(x5, x8, x4);
            let x10: bool = x4 ^ x4;
            let x11: bool = or_reduce(x2);
            let x12: bool = -x10;
            let x13: bool = x4[:1];
            let x14: u40 = match x1 {
                u27:0x7ff_ffff => u40:0xff_ffff_ffff,
                _ => u40:0xff_ffff_ffff,
            };
            let x15: u52 = u52:0x5_5555_5555_5555;
            let x16: bool = -x10;
            let x17: u5 = x3[11+:u5];
            let x18: bool = x11 & x13 as bool;
            let x19: u27 = x1 - x15 as u27;
            let x20: bool = x11[0+:bool];
            let x21: x6 = x7[if x15 >= u52:0x0 { u52:0x0 } else { x15 }];
            let x22: x6[2] = x7 ++ x7;
            let x23: bool = !x16;
            let x24: bool = or_reduce(x14);
            let x25: bool = x4 - x23;
            let x26: bool = bit_slice_update(x13, x10, x13);
            let x27: u19 = u19:0x3_ffff;
            let x28: (bool, bool, bool, bool, bool, bool) = (x26, x24, x26, x10, x24, x23);
            let x29: bool = x23[0+:bool];
            let x31: x30[1] = [x29];
            let x32: bool = x29 * x10 as bool;
            let x33: x30 = x31[if x27 >= u19:0x0 { u19:0x0 } else { x27 }];
            let x34: x6[9] = slice(x22, x16, x6[9]:[x22[u32:0x0], ...]);
            let x35: token = for (i, x) in u4:0x0..u4:0x6 {
                x
            }(x9);
            let x36: x6[10] = x34 ++ x7;
            let x37: u5 = x17 + x21 as u5;
            let x38: bool = x28 != x28;
            let x39: token = join(x9);
            let x41: (token, u60) = recv(x35, x40);
            let x42: token = x41.0;
            let x43: u60 = x41.1;
            x19
        }
    }
}
