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
// exception: 	 "SampleError: Result miscompare for sample 0:\nargs: bits[35]:0x2_aaaa_aaaa; bits[6]:0x30; bits[30]:0x307f_df5f\nevaluated opt IR (JIT), evaluated opt IR (interpreter) =\n   (bits[30]:0x0, bits[1]:0x0, bits[30]:0x0)\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[30]:0x0, bits[1]:0x1, bits[30]:0x0)"
// issue: "https://github.com/google/xls/issues/1184"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 8
//   proc_ticks: 0
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
//   function_args {
//     args: "bits[35]:0x2_aaaa_aaaa; bits[6]:0x30; bits[30]:0x307f_df5f"
//     args: "bits[35]:0x2000_0000; bits[6]:0x15; bits[30]:0x0"
//     args: "bits[35]:0x3_b4cf_88fe; bits[6]:0x2; bits[30]:0x14a_aab9"
//     args: "bits[35]:0x3_ffff_ffff; bits[6]:0x1f; bits[30]:0x1555_5555"
//     args: "bits[35]:0x7f3d_1a9d; bits[6]:0x15; bits[30]:0x1555_5555"
//     args: "bits[35]:0x1_19e5_5f43; bits[6]:0x1f; bits[30]:0x1de5_7f53"
//     args: "bits[35]:0x7_ffff_ffff; bits[6]:0x15; bits[30]:0x843_89ff"
//     args: "bits[35]:0x2_aaaa_aaaa; bits[6]:0xe; bits[30]:0xe55_5551"
//   }
// }
// 
// END_CONFIG
fn main(x0: u35, x1: u6, x2: u30) -> (u30, bool, u30) {
    {
        let x3: u35 = x0 << if x0 >= u35:0x14 { u35:0x14 } else { x0 };
        let x4: (u35, u6, u30, u35) = (x0, x1, x2, x3);
        let x5: u30 = x2 * x2;
        let x6: u29 = match x4 {
            (u35:0x7_ffff_ffff..u35:0x3_ffff_ffff, u6:0x2a, u30:0b0, u35:0x7_ffff_ffff) | (u35:0x3_ffff_ffff..u35:27986710046, u6:0x10, u30:0x1fff_ffff, u35:0x2_aaaa_aaaa) => u29:0xfff_ffff,
            _ => u29:0x14da_664b,
        };
        let x7: u6 = x1 >> if x6 >= u29:0b101 { u29:0b101 } else { x6 };
        let x9: s25 = s25:0x1ff_ffff;
        let x10: s25 = -x9;
        let x11: bool = x9 < x6 as s25;
        let x12: bool = x7 as u35 != x3;
        let x13: u30 = ctz(x2);
        let x14: u4 = x6[x1+:u4];
        let x15: u29 = x6[x6+:u29];
        let x16: u2 = x15[x6+:u2];
        let x17: u17 = x6[x14+:u17];
        let x18: u3 = x6[x6+:u3];
        let x19: bool = x11 | x11;
        let x20: u16 = x5[0+:u16];
        let x21: u30 = ctz(x5);
        let x22: u30 = x21 ^ x5 as u30;
        let x23: u30 = x19 as u30 + x21;
        let x24: bool = x2 <= x1 as u30;
        let x25: u3 = x1[x19+:u3];
        let x26: bool = x24[0+:bool];
        (x13, x19, x21)
    }
}
