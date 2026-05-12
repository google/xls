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
// exception: "SampleError: Result miscompare for sample 33:\nargs: bits[16]:0x8000; bits[10]:0x155; bits[10]:0x20; bits[23]:0x2_0000\nevaluated opt IR (JIT), evaluated unopt IR (JIT) =\n   (bits[10]:0x155, bits[4]:0x0, bits[1]:0x0, bits[2048]:0x0)\nevaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[10]:0x155, bits[4]:0xf, bits[1]:0x0, bits[2048]:0x0)"
// issue: "https://github.com/google/xls/issues/4213"
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
//     args: "bits[16]:0x5555; bits[10]:0x0; bits[10]:0x3; bits[23]:0x4ffd"
//     args: "bits[16]:0x5555; bits[10]:0x154; bits[10]:0x2c7; bits[23]:0x58_ec79"
//     args: "bits[16]:0x7fff; bits[10]:0x3be; bits[10]:0x2aa; bits[23]:0x22_3a38"
//     args: "bits[16]:0x4; bits[10]:0x3ff; bits[10]:0x224; bits[23]:0x6e_c345"
//     args: "bits[16]:0x5555; bits[10]:0x0; bits[10]:0x157; bits[23]:0xa_2ab0"
//     args: "bits[16]:0x0; bits[10]:0x10; bits[10]:0x12; bits[23]:0x2_ca63"
//     args: "bits[16]:0xaaaa; bits[10]:0x0; bits[10]:0x2e1; bits[23]:0x7f_5457"
//     args: "bits[16]:0x7fff; bits[10]:0x0; bits[10]:0x2aa; bits[23]:0x4f_16f3"
//     args: "bits[16]:0x5555; bits[10]:0x1d5; bits[10]:0x45; bits[23]:0x400"
//     args: "bits[16]:0x5555; bits[10]:0x1ff; bits[10]:0x37f; bits[23]:0x7f_ffff"
//     args: "bits[16]:0x80; bits[10]:0x181; bits[10]:0x109; bits[23]:0x18_2fff"
//     args: "bits[16]:0x7aef; bits[10]:0x2ff; bits[10]:0x2ff; bits[23]:0x27_ffad"
//     args: "bits[16]:0x5555; bits[10]:0x155; bits[10]:0x1ff; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x5555; bits[10]:0x2aa; bits[10]:0x9b; bits[23]:0x13_effe"
//     args: "bits[16]:0xddcd; bits[10]:0x85; bits[10]:0xed; bits[23]:0x5d_a100"
//     args: "bits[16]:0x0; bits[10]:0x0; bits[10]:0x155; bits[23]:0x55_5555"
//     args: "bits[16]:0x10; bits[10]:0x1ff; bits[10]:0x1fb; bits[23]:0x7f_ffff"
//     args: "bits[16]:0x0; bits[10]:0x272; bits[10]:0x17d; bits[23]:0x0"
//     args: "bits[16]:0x5555; bits[10]:0x155; bits[10]:0xb5; bits[23]:0x4000"
//     args: "bits[16]:0x7fff; bits[10]:0x2e7; bits[10]:0x232; bits[23]:0x3f_ffaa"
//     args: "bits[16]:0x40; bits[10]:0x4; bits[10]:0x8a; bits[23]:0x3f_ffff"
//     args: "bits[16]:0xffff; bits[10]:0x1ff; bits[10]:0x89; bits[23]:0x3f_e582"
//     args: "bits[16]:0xaaaa; bits[10]:0x2aa; bits[10]:0x26f; bits[23]:0x54_d57f"
//     args: "bits[16]:0xffff; bits[10]:0x3fb; bits[10]:0x2aa; bits[23]:0x7f_ffab"
//     args: "bits[16]:0x2; bits[10]:0x11a; bits[10]:0x155; bits[23]:0x2d_e8e3"
//     args: "bits[16]:0xaaaa; bits[10]:0x40; bits[10]:0x0; bits[23]:0x55_5555"
//     args: "bits[16]:0xd574; bits[10]:0x8; bits[10]:0x3ff; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0xaaaa; bits[10]:0x3ff; bits[10]:0x80; bits[23]:0x28_6bf7"
//     args: "bits[16]:0xf9a4; bits[10]:0x124; bits[10]:0x12e; bits[23]:0x0"
//     args: "bits[16]:0x0; bits[10]:0x4; bits[10]:0x0; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x0; bits[10]:0x0; bits[10]:0x3ff; bits[23]:0x7f_f471"
//     args: "bits[16]:0x5555; bits[10]:0x1ff; bits[10]:0x3ff; bits[23]:0x3e_2041"
//     args: "bits[16]:0xaaaa; bits[10]:0x2bf; bits[10]:0x290; bits[23]:0x3a_7ec7"
//     args: "bits[16]:0x8000; bits[10]:0x155; bits[10]:0x20; bits[23]:0x2_0000"
//     args: "bits[16]:0x5555; bits[10]:0x80; bits[10]:0x1ff; bits[23]:0x3a_a8dc"
//     args: "bits[16]:0x0; bits[10]:0x4; bits[10]:0x31c; bits[23]:0x6f_a7eb"
//     args: "bits[16]:0xaaaa; bits[10]:0x0; bits[10]:0x155; bits[23]:0x57_5d08"
//     args: "bits[16]:0xbc9c; bits[10]:0x48; bits[10]:0x100; bits[23]:0x55_5555"
//     args: "bits[16]:0x0; bits[10]:0x2aa; bits[10]:0x0; bits[23]:0x8_9c8a"
//     args: "bits[16]:0xffff; bits[10]:0x80; bits[10]:0x31a; bits[23]:0x76_6d73"
//     args: "bits[16]:0xffff; bits[10]:0x2bd; bits[10]:0x4; bits[23]:0x7f_ffbf"
//     args: "bits[16]:0x5555; bits[10]:0x14d; bits[10]:0x176; bits[23]:0x2e_8a68"
//     args: "bits[16]:0xaaaa; bits[10]:0x1ff; bits[10]:0x6f; bits[23]:0x1_0000"
//     args: "bits[16]:0x5555; bits[10]:0x1b8; bits[10]:0x2aa; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0x4000; bits[10]:0x40; bits[10]:0x211; bits[23]:0x2_2000"
//     args: "bits[16]:0xffff; bits[10]:0x2e7; bits[10]:0x155; bits[23]:0x29_8033"
//     args: "bits[16]:0x5555; bits[10]:0x15d; bits[10]:0x15d; bits[23]:0x16_581a"
//     args: "bits[16]:0x0; bits[10]:0x200; bits[10]:0x30; bits[23]:0x7f_ffff"
//     args: "bits[16]:0xffff; bits[10]:0x1; bits[10]:0x8; bits[23]:0x12_cb99"
//     args: "bits[16]:0x7fff; bits[10]:0x3f7; bits[10]:0x2aa; bits[23]:0x54_37d7"
//     args: "bits[16]:0x5555; bits[10]:0x170; bits[10]:0x155; bits[23]:0x6_bf37"
//     args: "bits[16]:0xffff; bits[10]:0x1ff; bits[10]:0x0; bits[23]:0x2"
//     args: "bits[16]:0xffff; bits[10]:0x3ff; bits[10]:0x22c; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x7fff; bits[10]:0x3bf; bits[10]:0x277; bits[23]:0x7f_b547"
//     args: "bits[16]:0xaaaa; bits[10]:0x0; bits[10]:0x3e6; bits[23]:0x7f_ffff"
//     args: "bits[16]:0x3b02; bits[10]:0x2aa; bits[10]:0x0; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x7fff; bits[10]:0x39a; bits[10]:0x155; bits[23]:0x66_350d"
//     args: "bits[16]:0x0; bits[10]:0x33e; bits[10]:0x2aa; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0xffff; bits[10]:0x1ff; bits[10]:0x1ff; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0x0; bits[10]:0x2aa; bits[10]:0x2d4; bits[23]:0x7c_2f66"
//     args: "bits[16]:0xffff; bits[10]:0x0; bits[10]:0x0; bits[23]:0x0"
//     args: "bits[16]:0xe08a; bits[10]:0x82; bits[10]:0x155; bits[23]:0x70_4555"
//     args: "bits[16]:0xaaaa; bits[10]:0x3a8; bits[10]:0x155; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0xc46; bits[10]:0x2aa; bits[10]:0xd; bits[23]:0x4_3303"
//     args: "bits[16]:0x0; bits[10]:0x4; bits[10]:0x8; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x0; bits[10]:0x0; bits[10]:0x1ff; bits[23]:0x0"
//     args: "bits[16]:0x7fff; bits[10]:0x191; bits[10]:0x9f; bits[23]:0x55_5555"
//     args: "bits[16]:0x5555; bits[10]:0x35f; bits[10]:0x5f; bits[23]:0x2a_2ab5"
//     args: "bits[16]:0xaaaa; bits[10]:0x38f; bits[10]:0x2ca; bits[23]:0x5c_4b55"
//     args: "bits[16]:0xffff; bits[10]:0x100; bits[10]:0x371; bits[23]:0x6e_fd82"
//     args: "bits[16]:0x5555; bits[10]:0x155; bits[10]:0x3ff; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x4000; bits[10]:0x32c; bits[10]:0x36c; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x5555; bits[10]:0x35; bits[10]:0x234; bits[23]:0x55_5555"
//     args: "bits[16]:0x400; bits[10]:0x155; bits[10]:0x141; bits[23]:0x38_2fbd"
//     args: "bits[16]:0x20; bits[10]:0x5; bits[10]:0x2aa; bits[23]:0x49_1317"
//     args: "bits[16]:0x0; bits[10]:0x1ff; bits[10]:0x155; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0x0; bits[10]:0x2; bits[10]:0x2aa; bits[23]:0x1_1635"
//     args: "bits[16]:0xaaaa; bits[10]:0x288; bits[10]:0x2aa; bits[23]:0x0"
//     args: "bits[16]:0x0; bits[10]:0xf0; bits[10]:0x40; bits[23]:0x55_5555"
//     args: "bits[16]:0x5555; bits[10]:0x341; bits[10]:0x2aa; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0x5555; bits[10]:0x155; bits[10]:0x2aa; bits[23]:0x55_5515"
//     args: "bits[16]:0x73cd; bits[10]:0x155; bits[10]:0x15d; bits[23]:0x0"
//     args: "bits[16]:0x1dda; bits[10]:0x0; bits[10]:0xe8; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0x7fff; bits[10]:0x35f; bits[10]:0x0; bits[23]:0x6b_e9ab"
//     args: "bits[16]:0x40; bits[10]:0x24b; bits[10]:0x14f; bits[23]:0x42_9389"
//     args: "bits[16]:0x8472; bits[10]:0x2a; bits[10]:0x1f9; bits[23]:0x4_0308"
//     args: "bits[16]:0x400; bits[10]:0x0; bits[10]:0x302; bits[23]:0x26_6472"
//     args: "bits[16]:0x0; bits[10]:0x99; bits[10]:0x0; bits[23]:0x40"
//     args: "bits[16]:0x0; bits[10]:0x279; bits[10]:0x279; bits[23]:0x4b_cde4"
//     args: "bits[16]:0x5555; bits[10]:0x144; bits[10]:0x1be; bits[23]:0x29_2fc2"
//     args: "bits[16]:0xffff; bits[10]:0x3fd; bits[10]:0x1; bits[23]:0x49_26ac"
//     args: "bits[16]:0x800; bits[10]:0x2aa; bits[10]:0x278; bits[23]:0x15_0f5d"
//     args: "bits[16]:0x0; bits[10]:0x210; bits[10]:0x1ff; bits[23]:0x76f"
//     args: "bits[16]:0x0; bits[10]:0x190; bits[10]:0x3f0; bits[23]:0x32_1be4"
//     args: "bits[16]:0x5555; bits[10]:0x1ff; bits[10]:0x3ff; bits[23]:0x7f_e584"
//     args: "bits[16]:0xaaaa; bits[10]:0x2aa; bits[10]:0xde; bits[23]:0x57_1755"
//     args: "bits[16]:0xaaaa; bits[10]:0x26e; bits[10]:0x42; bits[23]:0x4a_45d1"
//     args: "bits[16]:0x429; bits[10]:0x369; bits[10]:0x9; bits[23]:0x1_7fde"
//     args: "bits[16]:0x5555; bits[10]:0x155; bits[10]:0x152; bits[23]:0xe_bd45"
//     args: "bits[16]:0x7fff; bits[10]:0x2ff; bits[10]:0x155; bits[23]:0x37_ff83"
//     args: "bits[16]:0xffff; bits[10]:0x3df; bits[10]:0x155; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x5555; bits[10]:0x1d6; bits[10]:0x2; bits[23]:0x55_5555"
//     args: "bits[16]:0x0; bits[10]:0x44; bits[10]:0x71; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0xffff; bits[10]:0x3cf; bits[10]:0x13e; bits[23]:0x23_09f4"
//     args: "bits[16]:0x5932; bits[10]:0x25b; bits[10]:0x2; bits[23]:0x9_565c"
//     args: "bits[16]:0xaaaa; bits[10]:0x0; bits[10]:0x2; bits[23]:0x3f_ffff"
//     args: "bits[16]:0x7fff; bits[10]:0x0; bits[10]:0x158; bits[23]:0x6f_df2a"
//     args: "bits[16]:0x610; bits[10]:0x377; bits[10]:0x377; bits[23]:0x55_5555"
//     args: "bits[16]:0x5555; bits[10]:0x155; bits[10]:0x248; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0x7fff; bits[10]:0x155; bits[10]:0x14d; bits[23]:0x59_600b"
//     args: "bits[16]:0xffff; bits[10]:0x27a; bits[10]:0x1ff; bits[23]:0x3d_ebaa"
//     args: "bits[16]:0x0; bits[10]:0x34; bits[10]:0x10; bits[23]:0x2_657a"
//     args: "bits[16]:0x400; bits[10]:0x318; bits[10]:0x0; bits[23]:0x4b_c0c5"
//     args: "bits[16]:0x4; bits[10]:0x10e; bits[10]:0x3ff; bits[23]:0x21_cbdf"
//     args: "bits[16]:0x7fff; bits[10]:0x3f9; bits[10]:0xd9; bits[23]:0x2a_aaaa"
//     args: "bits[16]:0x8670; bits[10]:0x2a0; bits[10]:0x2; bits[23]:0x55_5555"
//     args: "bits[16]:0xaaaa; bits[10]:0x3ff; bits[10]:0x2a6; bits[23]:0x0"
//     args: "bits[16]:0x5555; bits[10]:0x2aa; bits[10]:0x2ae; bits[23]:0x55_c025"
//     args: "bits[16]:0x5555; bits[10]:0x1ff; bits[10]:0x151; bits[23]:0x20_00da"
//     args: "bits[16]:0xffff; bits[10]:0x200; bits[10]:0x310; bits[23]:0x62_4c63"
//     args: "bits[16]:0x1000; bits[10]:0x155; bits[10]:0x2aa; bits[23]:0x7f_ffff"
//     args: "bits[16]:0xaaaa; bits[10]:0x155; bits[10]:0x155; bits[23]:0x42_9dcb"
//     args: "bits[16]:0xb65d; bits[10]:0x2dd; bits[10]:0x68; bits[23]:0x5b_2c37"
//     args: "bits[16]:0x2; bits[10]:0x80; bits[10]:0x2aa; bits[23]:0x10_8122"
//     args: "bits[16]:0xd634; bits[10]:0x3ff; bits[10]:0xdc; bits[23]:0x5b_e4d8"
//     args: "bits[16]:0x5555; bits[10]:0x1ff; bits[10]:0x3d3; bits[23]:0x5a_7d64"
//     args: "bits[16]:0xffff; bits[10]:0xe6; bits[10]:0x26b; bits[23]:0x40_0000"
//     args: "bits[16]:0x74c9; bits[10]:0x3ff; bits[10]:0x0; bits[23]:0x7d_efdf"
//   }
// }
// 
// END_CONFIG
fn main(x0: u16, x1: u10, x2: u10, x3: u23) -> (u10, u4, bool, uN[2048]) {
    {
        let x4: bool = xor_reduce(x3);
        let x5: bool = x4[x2+:bool];
        let x6: u10 = bit_slice_update(x1, x3, x0);
        let x7: u16 = x0 << if x2 >= u10:0xb { u10:0xb } else { x2 };
        let x8: bool = x4 as bool & x5;
        let x9: bool = x4 as bool | x5;
        let x10: bool = -x9;
        let x11: bool = x5 & x4 as bool;
        let x12: u2 = one_hot(x11, bool:0x1);
        let x13: u23 = x3 << x1;
        let x14: u4 = encode(x0);
        let x15: bool = x8[0+:bool];
        let x16: bool = x5[x5+:bool];
        let x17: uN[2048] = decode<uN[2048]>(x0);
        let x18: uN[2048] = for (i, x): (u4, uN[2048]) in u4:0x0..u4:0x1 {
            x
        }(x17);
        let x19: u2 = decode<u2>(x4);
        let x20: u6 = x8 ++ x5 ++ x11 ++ x12 ++ x8;
        (x1, x14, x5, x18)
    }
}
