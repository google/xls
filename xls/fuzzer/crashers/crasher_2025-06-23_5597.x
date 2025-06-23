// Copyright 2025 The XLS Authors
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
// exception: "SampleError: Result miscompare for sample 35:\nargs: bits[14]:0x2000; bits[63]:0x0\nevaluated opt IR (JIT), evaluated opt IR (interpreter), simulated =\n   (bits[63]:0x2000, bits[63]:0x0, bits[17]:0xd)\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[63]:0x7fff_ffff_ffff_e000, bits[63]:0x0, bits[17]:0xd)"
// issue: "https://github.com/google/xls/issues/2458"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
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
// }
// inputs {
//   function_args {
//     args: "bits[14]:0x0; bits[63]:0x0"
//     args: "bits[14]:0x2aaa; bits[63]:0x5555_5fe7_ffff_7fff"
//     args: "bits[14]:0x4; bits[63]:0x6a00_737e_7d4b_ef6e"
//     args: "bits[14]:0x40; bits[63]:0x7b62_8104_37f1_6cda"
//     args: "bits[14]:0x3fff; bits[63]:0x3fee_3274_6e0c_6a1e"
//     args: "bits[14]:0x3fff; bits[63]:0x77ff_fffd_fffa_feff"
//     args: "bits[14]:0x1fff; bits[63]:0x8000_0000"
//     args: "bits[14]:0x3fff; bits[63]:0x7bde_0180_c860_d482"
//     args: "bits[14]:0x3fff; bits[63]:0x6d2e_6201_8494_6800"
//     args: "bits[14]:0x2aaa; bits[63]:0x55d4_0086_0498_00e8"
//     args: "bits[14]:0x0; bits[63]:0x608_77bf_bff7_d77d"
//     args: "bits[14]:0x4; bits[63]:0x0"
//     args: "bits[14]:0x1555; bits[63]:0x2b14_4f42_fbb5_9518"
//     args: "bits[14]:0x3fff; bits[63]:0x587e_effe_fefd_bdff"
//     args: "bits[14]:0x0; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x3fff; bits[63]:0x7feb_1fb5_0453_bc3e"
//     args: "bits[14]:0x1555; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x3fff; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x67c; bits[63]:0x1c3d_5554_5531_6513"
//     args: "bits[14]:0x0; bits[63]:0x1_ffff_dfff_ffef"
//     args: "bits[14]:0x0; bits[63]:0x0"
//     args: "bits[14]:0x400; bits[63]:0x0"
//     args: "bits[14]:0x2aaa; bits[63]:0x4973_665b_1508_4475"
//     args: "bits[14]:0x0; bits[63]:0x0"
//     args: "bits[14]:0x3fff; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x2aaa; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x3330; bits[63]:0x0"
//     args: "bits[14]:0x2aaa; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x3fff; bits[63]:0x5fe6_1041_0000_0101"
//     args: "bits[14]:0x3fff; bits[63]:0x7ffe_4b80_0c5f_ffb4"
//     args: "bits[14]:0x3fff; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x0; bits[63]:0x7fff_ffff_ffff_ffff"
//     args: "bits[14]:0x326e; bits[63]:0x7fff_ffff_ffff_ffff"
//     args: "bits[14]:0x1555; bits[63]:0x5555_5555_5555_5555"
//     args: "bits[14]:0x1fff; bits[63]:0x3ff8_0042_0000_8162"
//     args: "bits[14]:0x2000; bits[63]:0x0"
//     args: "bits[14]:0x1555; bits[63]:0x2aab_ffff_7fff_ffff"
//     args: "bits[14]:0x1fff; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x2aaa; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x1fff; bits[63]:0x3ebe_afba_bcfd_1bd3"
//     args: "bits[14]:0x3fff; bits[63]:0x0"
//     args: "bits[14]:0x1555; bits[63]:0x32a2_fbff_effb_fbff"
//     args: "bits[14]:0x2aaa; bits[63]:0x5554_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x1555; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x80; bits[63]:0x0"
//     args: "bits[14]:0x3fff; bits[63]:0x69b0_d775_bfff_57b7"
//     args: "bits[14]:0x1bed; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x1555; bits[63]:0x7a7d_2487_0119_2468"
//     args: "bits[14]:0x1fff; bits[63]:0x1f3a_1082_088f_001e"
//     args: "bits[14]:0x330a; bits[63]:0x892_a3f7_d863_df4a"
//     args: "bits[14]:0x5c3; bits[63]:0x1000"
//     args: "bits[14]:0x200; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x15ce; bits[63]:0x7b96_d3ae_cfff_e556"
//     args: "bits[14]:0x200; bits[63]:0x401_baba_aaaa_abaa"
//     args: "bits[14]:0x1fff; bits[63]:0x1db7_7609_43c1_2bad"
//     args: "bits[14]:0x400; bits[63]:0x6b55_338c_a040_7968"
//     args: "bits[14]:0x3652; bits[63]:0x6ca4_ffff_ffff_ffff"
//     args: "bits[14]:0x0; bits[63]:0x0"
//     args: "bits[14]:0x1555; bits[63]:0x7930_52a7_4055_ea5f"
//     args: "bits[14]:0x1fff; bits[63]:0x1bff_edff_bed7_5fe7"
//     args: "bits[14]:0x1dc8; bits[63]:0x79c5_d00d_91b6_a1f8"
//     args: "bits[14]:0x3fff; bits[63]:0x7ffe_f7df_ffff_ffff"
//     args: "bits[14]:0x0; bits[63]:0x44b_22d5_b28b_4665"
//     args: "bits[14]:0x800; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x1fff; bits[63]:0x354f_901b_e58c_d993"
//     args: "bits[14]:0x1fff; bits[63]:0x7bd2_a08a_98ee_e8a8"
//     args: "bits[14]:0x2aaa; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x2; bits[63]:0x400_0000"
//     args: "bits[14]:0x2fde; bits[63]:0x0"
//     args: "bits[14]:0x1fff; bits[63]:0x6fb9_e1d3_671b_3c41"
//     args: "bits[14]:0x3fff; bits[63]:0x75d7_1094_0834_1100"
//     args: "bits[14]:0x2aaa; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x1ebf; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x200; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x1555; bits[63]:0x5555_5555_5555_5555"
//     args: "bits[14]:0x1; bits[63]:0x2_d551_0755_75c5"
//     args: "bits[14]:0x1555; bits[63]:0x2000"
//     args: "bits[14]:0x1555; bits[63]:0x80_0000_0000"
//     args: "bits[14]:0x20; bits[63]:0x5555_5555_5555_5555"
//     args: "bits[14]:0x1555; bits[63]:0xeb6_7bf7_fe9e_fef6"
//     args: "bits[14]:0x1fff; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x1fff; bits[63]:0x2b97_de43_be8a_a653"
//     args: "bits[14]:0x0; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x800; bits[63]:0x1010_fffb_dfff_effd"
//     args: "bits[14]:0x2; bits[63]:0x1084_7fbf_bde8_9fef"
//     args: "bits[14]:0x3fff; bits[63]:0xf72_6dbd_ef91_fa1d"
//     args: "bits[14]:0x2d41; bits[63]:0x5581_2c11_1b92_18e2"
//     args: "bits[14]:0x359b; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x1fff; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x2f52; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x3fff; bits[63]:0x6ffd_0000_0000_0200"
//     args: "bits[14]:0x0; bits[63]:0x5010_aaaa_baab_aacb"
//     args: "bits[14]:0x3fff; bits[63]:0x79f4_aeaa_8a8a_baaa"
//     args: "bits[14]:0x0; bits[63]:0x1_0000_0000_0000"
//     args: "bits[14]:0x1000; bits[63]:0x5555_5555_5555_5555"
//     args: "bits[14]:0x1fff; bits[63]:0x76b1_4ff0_4c4d_d891"
//     args: "bits[14]:0x1555; bits[63]:0x7405_a53b_e7f8_f49b"
//     args: "bits[14]:0x1555; bits[63]:0x4f2b_5064_a218_9001"
//     args: "bits[14]:0x10; bits[63]:0x31_7555_7555_5455"
//     args: "bits[14]:0x2aaa; bits[63]:0x72c4_2b5e_73e4_e91c"
//     args: "bits[14]:0x1; bits[63]:0x1_0000"
//     args: "bits[14]:0x2aaa; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0xb15; bits[63]:0x0"
//     args: "bits[14]:0x1fff; bits[63]:0x3bff_8237_7a05_5493"
//     args: "bits[14]:0x8; bits[63]:0x2000_891f_f5b6_8ed0"
//     args: "bits[14]:0xc02; bits[63]:0x2248_4b6f_51bc_a32e"
//     args: "bits[14]:0x0; bits[63]:0x510c_2620_9fb4_f8f9"
//     args: "bits[14]:0x3fff; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x2aaa; bits[63]:0x396a_8c03_76e1_9910"
//     args: "bits[14]:0x1555; bits[63]:0xbe9_454a_1601_2041"
//     args: "bits[14]:0x3fff; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x1; bits[63]:0x2aaa_aaaa_aaaa_aaaa"
//     args: "bits[14]:0x1fff; bits[63]:0x5555_5555_5555_5555"
//     args: "bits[14]:0x954; bits[63]:0x50e3_e2ae_a29b_aa98"
//     args: "bits[14]:0x3fff; bits[63]:0x7cfe_acaa_aaa8_a2a0"
//     args: "bits[14]:0x1555; bits[63]:0x3fff_ffff_ffff_ffff"
//     args: "bits[14]:0x1555; bits[63]:0x5555_5555_5555_5555"
//     args: "bits[14]:0x0; bits[63]:0x627_8004_0002_2991"
//     args: "bits[14]:0x1555; bits[63]:0x28ba_0000_0401_0004"
//     args: "bits[14]:0x1fff; bits[63]:0x2f5a_98ca_26ab_8a0e"
//     args: "bits[14]:0x1fff; bits[63]:0x100_0000_0000"
//     args: "bits[14]:0x3fff; bits[63]:0x27f2_0100_4518_4150"
//     args: "bits[14]:0x1555; bits[63]:0x6abc_9ee5_de6f_fef5"
//     args: "bits[14]:0x1fff; bits[63]:0x2000_0000"
//     args: "bits[14]:0x2aaa; bits[63]:0x5555_9fa7_0ef0_fcec"
//     args: "bits[14]:0x2aaa; bits[63]:0x5754_0040_4881_0000"
//     args: "bits[14]:0x1c45; bits[63]:0x0"
//     args: "bits[14]:0x2aaa; bits[63]:0x7754_f7f5_fcff_fff7"
//   }
// }
// 
// END_CONFIG
fn main(x0: s14, x1: s63) -> (s63, s63, u17) {
    {
        let x2: s63 = x1 + x0 as s63;
        let x3: s17 = s17:0x800;
        let x4: s14 = x0 / s14:0x3fff;
        let x5: s63 = -x2;
        let x6: u17 = (x3 as u17)[:];
        let x7: s63 = !x5;
        let x8: s63 = x5 + x2;
        let x9: s14 = x4 % s14:0x3665;
        let x10: u17 = ctz(x6);
        let x11: u56 = u56:0xaa_aaaa_aaaa_aaaa;
        let x13: s17 = {
            let x12: (u17, u17) = smulp(x8 as u17 as s17, x10 as s17);
            (x12.0 + x12.1) as s17
        };
        let x14: s63 = x4 as s63 + x8;
        let x15: uN[2014] = decode<uN[2014]>(x6);
        let x16: u17 = for (i, x): (u4, u17) in u4:0x0..u4:0x3 {
            x
        }(x10);
        let x17: s63 = x8 % s63:0x4_0000;
        let x18: u17 = -x10;
        let x19: s63 = !x17;
        let x20: u18 = one_hot(x6, bool:0x1);
        let x21: uN[1549] = x15[x10+:uN[1549]];
        let x22: u17 = clz(x16);
        let x23: (s17, s17, uN[2014], s17) = (x13, x13, x15, x3);
        (x14, x1, x22)
    }
}
