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
//     args: "[bits[15]:0x0, bits[15]:0x17b7, bits[15]:0x5555, bits[15]:0x2aaa, bits[15]:0x3fff, bits[15]:0x3fff, bits[15]:0x5555]; bits[29]:0xaaa_aaaa; bits[43]:0x3ff_ffff_ffff; bits[5]:0x13"
//     args: "[bits[15]:0x7fff, bits[15]:0x77b3, bits[15]:0x7fff, bits[15]:0x5555, bits[15]:0x7fff, bits[15]:0x3fff, bits[15]:0x7fff]; bits[29]:0x4; bits[43]:0x4045_3f14; bits[5]:0x2"
//     args: "[bits[15]:0x4000, bits[15]:0x3fff, bits[15]:0x3e6f, bits[15]:0x7fff, bits[15]:0x5555, bits[15]:0x64ee, bits[15]:0x532c]; bits[29]:0x1fff_ffff; bits[43]:0x3ff_ffff_ffff; bits[5]:0x1f"
//     args: "[bits[15]:0x1de2, bits[15]:0x0, bits[15]:0x5555, bits[15]:0x5555, bits[15]:0x5555, bits[15]:0x5555, bits[15]:0x2aaa]; bits[29]:0xfff_ffff; bits[43]:0x7ff_ffff_ffff; bits[5]:0x8"
//     args: "[bits[15]:0x2aaa, bits[15]:0x3fff, bits[15]:0x2aaa, bits[15]:0x5555, bits[15]:0x6b6d, bits[15]:0x1000, bits[15]:0x7a43]; bits[29]:0x40_0000; bits[43]:0x0; bits[5]:0x2"
//     args: "[bits[15]:0x3fff, bits[15]:0x400, bits[15]:0x1482, bits[15]:0x5555, bits[15]:0x7fff, bits[15]:0x7fff, bits[15]:0x3fff]; bits[29]:0x0; bits[43]:0x3ff_ffff_ffff; bits[5]:0x1f"
//     args: "[bits[15]:0x3fff, bits[15]:0x4000, bits[15]:0x0, bits[15]:0x5555, bits[15]:0x86d, bits[15]:0x2aaa, bits[15]:0x2aaa]; bits[29]:0xfff_ffff; bits[43]:0x2aa_aaaa_aaaa; bits[5]:0x7"
//     args: "[bits[15]:0x0, bits[15]:0x40, bits[15]:0x8, bits[15]:0x5555, bits[15]:0x2aaa, bits[15]:0x3fff, bits[15]:0x0]; bits[29]:0x4_0000; bits[43]:0x7ff_ffff_ffff; bits[5]:0x10"
//   }
// }
// 
// END_CONFIG
type x0 = u15;
fn main(x1: x0[7], x2: u29, x3: u43, x4: u5) -> (u11, bool, u6, u43, u43, u4) {
    {
        let x5: u43 = clz(x3);
        let x6: bool = x2 as u43 != x5;
        let x8: u43 = {
            let x7: (u43, u43) = umulp(x6 as u43, x3);
            x7.0 + x7.1
        };
        let x9: u4 = u4:0x2;
        let x10: u6 = u6:0x3f;
        let x11: bool = x6 | x6;
        let x12: u43 = x8 | x2 as u43;
        let x13: bool = xor_reduce(x6);
        let x14: u43 = x12 >> if x9 >= u4:0xe { u4:0xe } else { x9 };
        let x15: u16 = x8[27+:u16];
        let x16: x0[9] = slice(x1, x12, x0[9]:[x1[u32:0x0], ...]);
        let x17: u7 = u7:0x8;
        let x18: u11 = match x12 {
            u43:0x3ff_ffff_ffff => u11:0x5b8,
            u43:0x0 => u11:0x555,
            _ => u11:0x1,
        };
        let x19: u15 = match x14 {
            u43:0x0 => u15:0x3fff,
            _ => u15:0x3fff,
        };
        let x21: s1 = {
            let x20: (bool, bool) = smulp(x6 as s1, x3 as bool as s1);
            (x20.0 + x20.1) as s1
        };
        (x18, x6, x10, x8, x8, x9)
    }
}
