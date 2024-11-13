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
// exception: 	 "SampleError: Result miscompare for sample 15:\nargs: bits[33]:0xffff_ffff; bits[8]:0xff\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[528]:0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_0000, [bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff], bits[256]:0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000)\nevaluated unopt IR (JIT) =\n   (bits[528]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_0000, [bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff, bits[33]:0xffff_ffff], bits[256]:0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000)"
// issue: "https://github.com/google/xls/issues/1201"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full throughput.*"
//   }
// }
// inputs {
//   function_args {
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xaa"
//     args: "bits[33]:0x0; bits[8]:0x0"
//     args: "bits[33]:0x0; bits[8]:0x61"
//     args: "bits[33]:0x2000_0000; bits[8]:0x0"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xf5"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x0"
//     args: "bits[33]:0x0; bits[8]:0x1"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x73"
//     args: "bits[33]:0x6bf9_1f01; bits[8]:0x7f"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xef"
//     args: "bits[33]:0x4; bits[8]:0x40"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x7f"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x80"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x4"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xdf"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xff"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x9f"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xaa"
//     args: "bits[33]:0x0; bits[8]:0x41"
//     args: "bits[33]:0xfcf1_32de; bits[8]:0xfe"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x58"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xbd"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xff"
//     args: "bits[33]:0x800; bits[8]:0x7f"
//     args: "bits[33]:0x88_8c63; bits[8]:0x55"
//     args: "bits[33]:0x4eac_02c7; bits[8]:0xd"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xcf"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x7f"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x8c"
//     args: "bits[33]:0x0; bits[8]:0x49"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xef"
//     args: "bits[33]:0x1_321b_6160; bits[8]:0x60"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x7f"
//     args: "bits[33]:0x0; bits[8]:0x7f"
//     args: "bits[33]:0x0; bits[8]:0x8b"
//     args: "bits[33]:0x0; bits[8]:0xaa"
//     args: "bits[33]:0x4000_0000; bits[8]:0x0"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0xaa"
//     args: "bits[33]:0x1_f032_6c52; bits[8]:0x7e"
//     args: "bits[33]:0x7d81_5dae; bits[8]:0x3c"
//     args: "bits[33]:0x0; bits[8]:0xaa"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xae"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x55"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x80"
//     args: "bits[33]:0x0; bits[8]:0x6d"
//     args: "bits[33]:0x1; bits[8]:0x55"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xaa"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0xaa"
//     args: "bits[33]:0x1_213f_6f96; bits[8]:0x96"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x0"
//     args: "bits[33]:0x1_0000_0000; bits[8]:0x0"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0xaa"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x92"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x7d"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x55"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x20"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xaa"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x96"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xbb"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xfc"
//     args: "bits[33]:0x20_0000; bits[8]:0x55"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x33"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x5d"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x40"
//     args: "bits[33]:0x800_0000; bits[8]:0x7f"
//     args: "bits[33]:0x8; bits[8]:0x7f"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xff"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xaa"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xff"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x1"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xb1"
//     args: "bits[33]:0x1_d6ae_67ae; bits[8]:0x85"
//     args: "bits[33]:0xa08d_dbed; bits[8]:0xec"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xaa"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xb"
//     args: "bits[33]:0x80_0000; bits[8]:0x55"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xff"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xae"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x55"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x22"
//     args: "bits[33]:0x100_0000; bits[8]:0x0"
//     args: "bits[33]:0x8000_0000; bits[8]:0x0"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xa2"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xdf"
//     args: "bits[33]:0x1_6856_26e1; bits[8]:0x2"
//     args: "bits[33]:0x2ab0_cf8f; bits[8]:0x3"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x7f"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xff"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xbe"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x80"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x8"
//     args: "bits[33]:0x4000_0000; bits[8]:0x55"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x61"
//     args: "bits[33]:0x0; bits[8]:0x40"
//     args: "bits[33]:0x1000_0000; bits[8]:0x7f"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xff"
//     args: "bits[33]:0x80; bits[8]:0x7f"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x3f"
//     args: "bits[33]:0x1_caab_8ec2; bits[8]:0xc2"
//     args: "bits[33]:0x0; bits[8]:0x9c"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0xf"
//     args: "bits[33]:0x200_0000; bits[8]:0x80"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x9e"
//     args: "bits[33]:0x40_0000; bits[8]:0x30"
//     args: "bits[33]:0x0; bits[8]:0x7f"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0xf0"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0xff"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x0"
//     args: "bits[33]:0x0; bits[8]:0x8"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x45"
//     args: "bits[33]:0x0; bits[8]:0x80"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x55"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x2"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x20"
//     args: "bits[33]:0x0; bits[8]:0xff"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0x0"
//     args: "bits[33]:0x10; bits[8]:0x7f"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xff"
//     args: "bits[33]:0xffff_ffff; bits[8]:0x7f"
//     args: "bits[33]:0x1_6688_4d33; bits[8]:0x32"
//     args: "bits[33]:0x1_ffff_ffff; bits[8]:0xaa"
//     args: "bits[33]:0xaaaa_aaaa; bits[8]:0x7f"
//     args: "bits[33]:0x4_0000; bits[8]:0x55"
//     args: "bits[33]:0x8000_0000; bits[8]:0xff"
//     args: "bits[33]:0xffff_ffff; bits[8]:0xf7"
//     args: "bits[33]:0x8000_0000; bits[8]:0xc9"
//     args: "bits[33]:0x3136_cf44; bits[8]:0x40"
//     args: "bits[33]:0x1_5555_5555; bits[8]:0x7f"
//   }
// }
// 
// END_CONFIG
type x9 = s33;
type x23 = bool;
type x31 = (bool, bool, bool, u55, bool);
fn x25<x27: u56 = {u56:0x55_5555_5555_5555}>(x26: x23) -> (bool, bool, bool, u55, bool) {
    {
        let x28: u55 = u55:0x2a_aaaa_aaaa_aaaa;
        let x29: u38 = x28[12+:u38];
        let x30: bool = or_reduce(x29);
        (x30, x30, x30, x28, x30)
    }
}
fn main(x0: s33, x1: u8) -> (uN[528], x9[7], uN[256]) {
    {
        let x2: (u8, s33) = (x1, x0);
        let x3: s33 = x0 * x0;
        let x4: s33 = -x0;
        let x5: uN[256] = decode<uN[256]>(x1);
        let x6: u8 = ctz(x1);
        let x7: uN[528] = x5 ++ x5 ++ x1 ++ x1;
        let x8: (u8, s33, (u8, s33), uN[256], u8) = (x1, x0, x2, x5, x6);
        let x10: x9[10] = [x0, x3, x4, x3, x3, x3, x3, x3, x3, x0];
        let x11: uN[528] = x7 << if x5 >= uN[256]:0x108 { uN[256]:0x108 } else { x5 };
        let x12: bool = x6 > x7 as u8;
        let x13: bool = and_reduce(x1);
        let x14: uN[528] = !x7;
        let x15: uN[528] = x14[x6+:uN[528]];
        let x16: uN[528] = bit_slice_update(x15, x5, x1);
        let x17: x9[6] = array_slice(x10, x14, x9[6]:[x10[u32:0x0], ...]);
        let x18: x9[1] = array_slice(x10, x12, x9[1]:[x10[u32:0x0], ...]);
        let x19: x9[16] = x17 ++ x10;
        let x20: x9[7] = x17 ++ x18;
        let x21: bool = x12[:];
        let x22: s33 = x14 as s33 | x3;
        let x24: x23[1] = x13 as x23[1];
        let x32: x31[1] = map(x24, x25);
        (x16, x20, x5)
    }
}
