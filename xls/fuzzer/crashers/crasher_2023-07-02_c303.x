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
// exception: "In evaluated opt IR (JIT), at position 0 channel \'sample__x18\' has value u1:1. However, in simulated, the value is u1:0.\n"
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
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--flop_inputs=true"
//   codegen_args: "--flop_inputs_kind=zerolatency"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x4"
//       values: "bits[59]:0x29_b9f2_ddb3_e5ea"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x390_cdc8_7021_151f"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x100"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x40_0000_0000_0000"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2_0000_0000"
//       values: "bits[59]:0x80_0000"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x6fd_0dc0_aa51_b77d"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x400"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x8000_0000_0000"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x784_2869_a5e3_aa26"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x37f_c4cf_a50f_4444"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x1000_0000"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2e0_521d_7c9d_d8d8"
//       values: "bits[59]:0x8000_0000_0000"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x17e_b088_c5d0_d3fc"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x20_0000_0000_0000"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x80_0000_0000"
//       values: "bits[59]:0x2000_0000_0000"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x204_5b7c_dbe2_1405"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x599_2478_2aad_92ad"
//       values: "bits[59]:0x61b_f07a_3611_3012"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x721_18ab_8f51_ed92"
//       values: "bits[59]:0x506_d0c3_95b9_d08c"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x800"
//       values: "bits[59]:0x390_1817_5d9e_8456"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x3ff_ffff_ffff_ffff"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x400_0000_0000"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x200"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x77_e30c_c251_08bf"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x555_5555_5555_5555"
//       values: "bits[59]:0x0"
//       values: "bits[59]:0x40_0000_0000_0000"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x7ff_ffff_ffff_ffff"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x32_24e6_73e6_4ba0"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//       values: "bits[59]:0x2aa_aaaa_aaaa_aaaa"
//     }
//   }
// }
// 
// END_CONFIG
const W32_V8 = u32:0x8;
type x29 = u30;
proc main {
    x4: chan<u59> in;
    x18: chan<bool> out;
    config(x4: chan<u59> in, x18: chan<bool> out) {
        (x4, x18)
    }
    init {
        u59:288230376151711743
    }
    next(x1: u59) {
        {
            let x0: token = join();
            let x2: (u59,) = (x1,);
            let x3: u59 = x1 | x1;
            let x5: (token, u59, bool) = recv_non_blocking(x0, x4, x1);
            let x6: token = x5.0;
            let x7: u59 = x5.1;
            let x8: bool = x5.2;
            let x9: u59 = x7;
            let x10: bool = or_reduce(x8);
            let x11: u59 = !x9;
            let x12: u59 = -x7;
            let x13: u36 = u36:0xa_aaaa_aaaa;
            let x14: bool = x10 <= x10;
            let x15: bool = bit_slice_update(x8, x9, x9);
            let x16: token = join(x6, x6);
            let x17: u36 = !x13;
            let x19: token = send_if(x6, x18, x14, x10);
            let x20: u17 = u17:0x1_ffff;
            let x21: bool = x14 >> if x15 >= bool:false { bool:false } else { x15 };
            let x22: bool = x21 + x12 as bool;
            let x23: bool = x2 == x2;
            let x24: bool = x15 | x9 as bool;
            let x25: bool = x23 >> x12;
            let x37: bool = x7 as bool - x15;
            let x38: u37 = u37:0x0;
            x7
        }
    }
}
