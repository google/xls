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
// exception: "SampleError: Result miscompare for sample 1:\nargs: bits[15]:0x2aaa; bits[2]:0x2\nevaluated opt IR (JIT) =\n   (bits[15]:0x2aab, bits[2]:0x2, (bits[14]:0x1555, bits[49]:0x0, bits[39]:0x58_c1d6_b1d9), bits[1]:0x1)\nevaluated opt IR (interpreter), evaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX, simulated =\n   (bits[15]:0x2aab, bits[2]:0x2, (bits[14]:0x1555, bits[49]:0x1_ffff_ffff_ffff, bits[39]:0x1_0000_0000), bits[1]:0x1)"
// issue: "https://github.com/google/xls/issues/4302"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
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
//   calls_per_sample: 128
//   proc_ticks: 0
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
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "bits[15]:0x7fff; bits[2]:0x3"
//     args: "bits[15]:0x2aaa; bits[2]:0x2"
//     args: "bits[15]:0x100; bits[2]:0x0"
//     args: "bits[15]:0x0; bits[2]:0x1"
//     args: "bits[15]:0x7fff; bits[2]:0x2"
//     args: "bits[15]:0x4997; bits[2]:0x3"
//     args: "bits[15]:0x2aaa; bits[2]:0x2"
//     args: "bits[15]:0x79a9; bits[2]:0x1"
//     args: "bits[15]:0x2aaa; bits[2]:0x2"
//     args: "bits[15]:0x8; bits[2]:0x0"
//     args: "bits[15]:0x7fff; bits[2]:0x3"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x2aaa; bits[2]:0x2"
//     args: "bits[15]:0x0; bits[2]:0x0"
//     args: "bits[15]:0x2aaa; bits[2]:0x0"
//     args: "bits[15]:0x1505; bits[2]:0x1"
//     args: "bits[15]:0x200; bits[2]:0x0"
//     args: "bits[15]:0x7fff; bits[2]:0x3"
//     args: "bits[15]:0x2aaa; bits[2]:0x3"
//     args: "bits[15]:0x0; bits[2]:0x2"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x19d4; bits[2]:0x3"
//     args: "bits[15]:0x2aaa; bits[2]:0x0"
//     args: "bits[15]:0x2; bits[2]:0x0"
//     args: "bits[15]:0xe6a; bits[2]:0x2"
//     args: "bits[15]:0x0; bits[2]:0x2"
//     args: "bits[15]:0x7fff; bits[2]:0x3"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x0; bits[2]:0x2"
//     args: "bits[15]:0x2; bits[2]:0x0"
//     args: "bits[15]:0x0; bits[2]:0x3"
//     args: "bits[15]:0x34c0; bits[2]:0x1"
//     args: "bits[15]:0x0; bits[2]:0x3"
//     args: "bits[15]:0x7a5; bits[2]:0x2"
//     args: "bits[15]:0x10; bits[2]:0x2"
//     args: "bits[15]:0x20; bits[2]:0x0"
//     args: "bits[15]:0x7fff; bits[2]:0x0"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x3fff; bits[2]:0x3"
//     args: "bits[15]:0x100; bits[2]:0x1"
//     args: "bits[15]:0x4cda; bits[2]:0x0"
//     args: "bits[15]:0x7fff; bits[2]:0x2"
//     args: "bits[15]:0x5555; bits[2]:0x0"
//     args: "bits[15]:0x2aaa; bits[2]:0x0"
//     args: "bits[15]:0x7289; bits[2]:0x0"
//     args: "bits[15]:0x0; bits[2]:0x0"
//     args: "bits[15]:0x2; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x3fff; bits[2]:0x2"
//     args: "bits[15]:0x3fff; bits[2]:0x3"
//     args: "bits[15]:0x5555; bits[2]:0x3"
//     args: "bits[15]:0x41ff; bits[2]:0x3"
//     args: "bits[15]:0x2f63; bits[2]:0x3"
//     args: "bits[15]:0x2aaa; bits[2]:0x0"
//     args: "bits[15]:0x200; bits[2]:0x2"
//     args: "bits[15]:0x200; bits[2]:0x0"
//     args: "bits[15]:0x3fff; bits[2]:0x3"
//     args: "bits[15]:0x5555; bits[2]:0x0"
//     args: "bits[15]:0x0; bits[2]:0x0"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x1; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x0"
//     args: "bits[15]:0x7af1; bits[2]:0x1"
//     args: "bits[15]:0x0; bits[2]:0x0"
//     args: "bits[15]:0x3fff; bits[2]:0x3"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x4f38; bits[2]:0x0"
//     args: "bits[15]:0x3fff; bits[2]:0x0"
//     args: "bits[15]:0x2aaa; bits[2]:0x0"
//     args: "bits[15]:0x4; bits[2]:0x0"
//     args: "bits[15]:0x2aaa; bits[2]:0x3"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x2"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x3fff; bits[2]:0x1"
//     args: "bits[15]:0x0; bits[2]:0x2"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x3fff; bits[2]:0x2"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x7fff; bits[2]:0x3"
//     args: "bits[15]:0x1dca; bits[2]:0x0"
//     args: "bits[15]:0x0; bits[2]:0x0"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x2; bits[2]:0x3"
//     args: "bits[15]:0x1; bits[2]:0x2"
//     args: "bits[15]:0x2aaa; bits[2]:0x0"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x3fff; bits[2]:0x2"
//     args: "bits[15]:0x3fff; bits[2]:0x2"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x0; bits[2]:0x2"
//     args: "bits[15]:0x0; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x2"
//     args: "bits[15]:0x5555; bits[2]:0x2"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x3"
//     args: "bits[15]:0x3fff; bits[2]:0x1"
//     args: "bits[15]:0x7fff; bits[2]:0x3"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//     args: "bits[15]:0x0; bits[2]:0x3"
//     args: "bits[15]:0x2aaa; bits[2]:0x2"
//     args: "bits[15]:0x0; bits[2]:0x0"
//     args: "bits[15]:0x3fff; bits[2]:0x3"
//     args: "bits[15]:0x10; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x3"
//     args: "bits[15]:0x4; bits[2]:0x0"
//     args: "bits[15]:0x4; bits[2]:0x1"
//     args: "bits[15]:0x304b; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x2"
//     args: "bits[15]:0x3fff; bits[2]:0x3"
//     args: "bits[15]:0x2aaa; bits[2]:0x1"
//     args: "bits[15]:0x80; bits[2]:0x1"
//     args: "bits[15]:0x2aaa; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x2"
//     args: "bits[15]:0x2aaa; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x1"
//     args: "bits[15]:0x5555; bits[2]:0x2"
//     args: "bits[15]:0x0; bits[2]:0x1"
//     args: "bits[15]:0x20; bits[2]:0x0"
//     args: "bits[15]:0x7fff; bits[2]:0x1"
//   }
// }
//
// END_CONFIG
type x18 = u15;
fn main(x0: u15, x1: u2) -> (u15, u2, (u14, u49, u39), bool) {
    {
        let x2: bool = xor_reduce(x1);
        let x3: u15 = x0 / u15:0x0;
        let x4: u15 = -x3;
        let x5: bool = x3[0+:xN[bool:0x0][1]];
        let x6: u15 = !x3;
        let x7: bool = x2[0+:bool];
        let x8: bool = -x5;
        let x9: bool = xor_reduce(x4);
        let x10: u15 = !x3;
        let x11: u15 = -x4;
        let x12: bool = x4 <= x8 as u15;
        let x13: u15 = x0 ^ x5 as u15;
        let x14: bool = and_reduce(x0);
        let x15: bool = !x12;
        let x16: u16 = u16:0x200;
        let x17: u15 = !x6;
        let x19: x18[8] = [x0, x10, x11, x13, x17, x3, x4, x6];
        let x20: bool = x19 == x19;
        let x21: (u15, u15, bool, bool, bool, bool) = (x17, x11, x2, x7, x8, x7);
        let (..) = (x17, x11, x2, x7, x8, x7);
        let x23: bool = {
            let x22: (bool, bool) = umulp(x9 as bool, x7);
            x22.0 + x22.1
        };
        let x24: u15 = x17 >> if x11 >= u15:0x2 { u15:0x2 } else { x11 };
        let x25: u15 = x4 & x0;
        let x26: u15 = !x3;
        let x27: (u14, u49, u39) = match x13 {
            u15:0x0..u15:0x2aaa | u15:0x2aaa => (u14:0x1555, u49:0x0, u39:0x58_c1d6_b1d9),
            u15:0x0..u15:16383 | u15:0x5555 => (u14:0x1555, u49:0b1_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111, u39:0x1_0000_0000),
            u15:16383 => (u14:0x400, u49:0x0, u39:0x34_4791_30c6),
            _ => (u14:0x1686, u49:0x1_ffff_ffff_ffff, u39:0x0),
        };
        let (x28, x29, _): (u14, u49, u39) = match x13 {
            u15:0x0..u15:0x2aaa | u15:0x2aaa => (u14:0x1555, u49:0x0, u39:0x58_c1d6_b1d9),
            u15:0x0..u15:16383 | u15:0x5555 => (u14:0x1555, u49:0b1_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111, u39:0x1_0000_0000),
            u15:16383 => (u14:0x400, u49:0x0, u39:0x34_4791_30c6),
            _ => (u14:0x1686, u49:0x1_ffff_ffff_ffff, u39:0x0),
        };
        (x13, x1, x27, x5)
    }
}
