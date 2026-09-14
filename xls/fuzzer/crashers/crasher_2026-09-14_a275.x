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
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[22]:0x15_5555; bits[11]:0x555\nevaluated opt IR (JIT), evaluated opt IR (interpreter) =\n   (bits[22]:0x2a_aaaa, bits[9]:0x155, bits[22]:0x0)\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[22]:0x0, bits[9]:0x155, bits[22]:0x0)"
// issue: ""
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "bits[22]:0x15_5555; bits[11]:0x555"
//     args: "bits[22]:0x14_85eb; bits[11]:0x4eb"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2bb"
//     args: "bits[22]:0x25_22cd; bits[11]:0x7ff"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x28a"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x4ad"
//     args: "bits[22]:0x15_5555; bits[11]:0x7e5"
//     args: "bits[22]:0x15_5555; bits[11]:0x100"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x6c7"
//     args: "bits[22]:0x20; bits[11]:0x20"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x2ff"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x4"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2a3"
//     args: "bits[22]:0x15_5555; bits[11]:0x245"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x555"
//     args: "bits[22]:0x15_5555; bits[11]:0x2aa"
//     args: "bits[22]:0x100; bits[11]:0x0"
//     args: "bits[22]:0x0; bits[11]:0x7ff"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x10"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x69e"
//     args: "bits[22]:0x0; bits[11]:0x555"
//     args: "bits[22]:0x80; bits[11]:0x80"
//     args: "bits[22]:0x0; bits[11]:0x2aa"
//     args: "bits[22]:0x0; bits[11]:0x0"
//     args: "bits[22]:0x0; bits[11]:0x28"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x7f6"
//     args: "bits[22]:0x15_5555; bits[11]:0x755"
//     args: "bits[22]:0x0; bits[11]:0x0"
//     args: "bits[22]:0x4_0000; bits[11]:0x12"
//     args: "bits[22]:0x0; bits[11]:0x502"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x7fe"
//     args: "bits[22]:0x15_5555; bits[11]:0x5f7"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x7ff"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x5ff"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x0"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x3ff"
//     args: "bits[22]:0x0; bits[11]:0x2"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2aa"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x70e"
//     args: "bits[22]:0x0; bits[11]:0x2"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x7ff"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x7ff"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x7df"
//     args: "bits[22]:0x2e_95a9; bits[11]:0x81"
//     args: "bits[22]:0x200; bits[11]:0x3ff"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x555"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x43e"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x0"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x0"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0xd2"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x7d7"
//     args: "bits[22]:0x15_5555; bits[11]:0x555"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2ea"
//     args: "bits[22]:0x15_5555; bits[11]:0x0"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x2aa"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x7ff"
//     args: "bits[22]:0xb_0d90; bits[11]:0x8b"
//     args: "bits[22]:0x0; bits[11]:0x10"
//     args: "bits[22]:0x29_996c; bits[11]:0x22c"
//     args: "bits[22]:0x0; bits[11]:0x555"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x3ff"
//     args: "bits[22]:0x3_1761; bits[11]:0x60c"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x71"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2aa"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0xaa"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x7ff"
//     args: "bits[22]:0x0; bits[11]:0x7ff"
//     args: "bits[22]:0x0; bits[11]:0x7ff"
//     args: "bits[22]:0x0; bits[11]:0x86"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x555"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x555"
//     args: "bits[22]:0x28_b2ae; bits[11]:0x286"
//     args: "bits[22]:0x15_5555; bits[11]:0x0"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x0"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x125"
//     args: "bits[22]:0x2a_158d; bits[11]:0x50f"
//     args: "bits[22]:0x15_5555; bits[11]:0x3ff"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x6f4"
//     args: "bits[22]:0x39_f375; bits[11]:0x315"
//     args: "bits[22]:0x0; bits[11]:0x0"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x7ff"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2aa"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x4"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x2aa"
//     args: "bits[22]:0x15_5555; bits[11]:0x7ff"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x667"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x555"
//     args: "bits[22]:0x33_af62; bits[11]:0x0"
//     args: "bits[22]:0x3b_4026; bits[11]:0x0"
//     args: "bits[22]:0xb_cb54; bits[11]:0x555"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x3ff"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x2aa"
//     args: "bits[22]:0x20_0000; bits[11]:0x555"
//     args: "bits[22]:0xf_01fb; bits[11]:0x3ab"
//     args: "bits[22]:0x400; bits[11]:0x0"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x0"
//     args: "bits[22]:0x22_487d; bits[11]:0x2aa"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x555"
//     args: "bits[22]:0x4; bits[11]:0x3ff"
//     args: "bits[22]:0x25_44c5; bits[11]:0x490"
//     args: "bits[22]:0x0; bits[11]:0x1"
//     args: "bits[22]:0x100; bits[11]:0x555"
//     args: "bits[22]:0x0; bits[11]:0x718"
//     args: "bits[22]:0x1f_6993; bits[11]:0x0"
//     args: "bits[22]:0x1f_4059; bits[11]:0x359"
//     args: "bits[22]:0x18_ffbb; bits[11]:0x0"
//     args: "bits[22]:0x0; bits[11]:0x0"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x7d6"
//     args: "bits[22]:0x15_5555; bits[11]:0x555"
//     args: "bits[22]:0x0; bits[11]:0x5b"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2aa"
//     args: "bits[22]:0x0; bits[11]:0x88"
//     args: "bits[22]:0x36_6711; bits[11]:0x731"
//     args: "bits[22]:0x1f_4faf; bits[11]:0x3ff"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x37f"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x0"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2a8"
//     args: "bits[22]:0x1f_ffff; bits[11]:0x57f"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x7ff"
//     args: "bits[22]:0x8_0000; bits[11]:0x307"
//     args: "bits[22]:0x2_0000; bits[11]:0x7ff"
//     args: "bits[22]:0x3f_ffff; bits[11]:0x7fd"
//     args: "bits[22]:0x400; bits[11]:0x5c5"
//     args: "bits[22]:0x100; bits[11]:0x1e4"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x2aa"
//     args: "bits[22]:0x3e5d; bits[11]:0x21f"
//     args: "bits[22]:0x15_5555; bits[11]:0x7ff"
//     args: "bits[22]:0x2a_aaaa; bits[11]:0x202"
//   }
// }
// 
// END_CONFIG
fn main(x0: s22, x1: s11) -> (s22, u9, s22) {
    {
        let x2: s24 = s24:0xaa_aaaa;
        let x3: s24 = gate!(x1 >= x1, x2);
        let x4: s22 = x0 % s22:0x3f_ffff;
        let x5: bool = x1 as s22 >= x0;
        let x6: s22 = x4 | x2 as s22;
        let x7: s11 = x2 as s11 * x1;
        let x8: s22 = x3 as s22 ^ x6;
        let x9: u9 = (x0 as u22)[x5+:u9];
        let x10: s22 = !x0;
        let x11: u9 = -x9;
        let x12: u9 = ctz(x9);
        let x13: s41 = s41:0xff_ffff_ffff;
        let x14: u9 = rev(x9);
        let x15: s11 = !x1;
        let x16: s22 = x0 / s22:0x28_879f;
        let x17: bool = xor_reduce(x11);
        let x18: s22 = !x16;
        let x19: u9 = x9 ^ x11;
        let x20: s22 = x10 << if x12 >= u9:0x2 { u9:0x2 } else { x12 };
        let x21: u9 = bit_slice_update(x12, x19, x17);
        let x22: s22 = !x18;
        let x23: u9 = gate!(x15 > x2 as s11, x12);
        let x24: xN[bool:0x0][38] = x12 ++ x5 ++ x14 ++ x9 ++ x17 ++ x9;
        let x25: u9 = -x11;
        let x26: s64 = match x19 {
            u9:0xff | u9:341 => s64:0x1000,
            u9:0x1f0 => s64:0x5555_5555_5555_5555,
            u9:0xaa => s64:0x7fff_ffff_ffff_ffff,
            _ => s64:0x1_0000_0000,
        };
        let x27: bool = xor_reduce(x11);
        let x28: s22 = -x16;
        (x8, x9, x28)
    }
}
