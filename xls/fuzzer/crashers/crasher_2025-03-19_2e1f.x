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
// issue: "https://github.com/google/xls/issues/2011"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=10"
//   codegen_args: "--worst_case_throughput=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x15"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x7_bd71_0151_2fb9"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x8000"
//       values: "bits[51]:0x80_0000"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x3_347b_dc99_d327"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x4_cb21_3acd_dc55"
//       values: "bits[51]:0x2000"
//       values: "bits[51]:0x1_0000_0000"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x4_8a7d_fb9b_462a"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x1000_0000_0000"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x7_2f60_02c9_4fed"
//       values: "bits[51]:0x4_1a76_6bcb_d5d4"
//       values: "bits[51]:0x2_0000"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x1_f45f_6677_1c6e"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x20"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x3_e92c_8c19_98e6"
//       values: "bits[51]:0x1_70c2_f3d5_6c12"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x1"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x100_0000_0000"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x100_0000_0000"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x7e90_ddea_f3f3"
//       values: "bits[51]:0x5_b876_bba5_26ae"
//       values: "bits[51]:0x80_0000"
//       values: "bits[51]:0x4_0000_0000"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x20_0000"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x6_fbc5_6d72_febc"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x2"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x80"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x8_0000_0000"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x20_0000"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x2000_0000_0000"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x7_974b_7376_f982"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x80"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x2"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x6_fcf1_175b_88a9"
//       values: "bits[51]:0x4"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x80_0000_0000"
//       values: "bits[51]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x400_0000"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x0"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x20_0000_0000"
//       values: "bits[51]:0x4_f76d_a9ab_3112"
//       values: "bits[51]:0x5_5555_5555_5555"
//       values: "bits[51]:0x4_c87c_fa69_58af"
//       values: "bits[51]:0x3_ffff_ffff_ffff"
//       values: "bits[51]:0xfd0e_aa91_0c2a"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//       values: "bits[51]:0x7_ffff_ffff_ffff"
//     }
//   }
// }
// 
// END_CONFIG
type x21 = uN[372];
proc main {
    x15: chan<u51> in;
    config(x15: chan<u51> in) {
        (x15,)
    }
    init {
        uN[320]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff
    }
    next(x0: uN[320]) {
        {
            let x1: token = join();
            let x2: uN[320] = !x0;
            let x3: s56 = match x0 {
                uN[320]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff => s56:0x7f_ffff_ffff_ffff,
                uN[320]:0x0 => s56:0xaa_aaaa_aaaa_aaaa,
                uN[320]:0x59d1_7539_1166_b484_90de_3aaf_e14c_4804_b72f_d78c_1fb8_9460_9ac3_ce99_3b2f_dda1_b388_67ff_0a4d_700a | uN[320]:0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111 => s56:0x2a_07c1_0aac_d0b2,
                uN[320]:0x5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555 => s56:0x7f_ffff_ffff_ffff,
                _ => s56:0x4_0000,
            };
            let x4: bool = and_reduce(x0);
            let x5: s56 = x3 + x0 as s56;
            let x6: bool = xor_reduce(x2);
            let x7: uN[320] = for (i, x): (u4, uN[320]) in u4:0x0..u4:0x3 {
                x
            }(x2);
            let x8: bool = x6 < x7 as bool;
            let x9: s56 = x3 >> if x8 >= bool:false { bool:false } else { x8 };
            let x10: bool = !x4;
            let x11: uN[320] = x2;
            let x12: uN[320] = bit_slice_update(x2, x8, x6);
            let x13: xN[bool:0x0][2] = one_hot(x4, bool:0x1);
            let x14: uN[320] = -x12;
            let x16: (token, u51) = recv(x1, x15);
            let x17: token = x16.0;
            let x18: u51 = x16.1;
            let x19: uN[372] = x0 ++ x18 ++ x8;
            let x20: uN[320] = bit_slice_update(x2, x6, x13);
            let x22: x21[1] = [x19];
            let x23: uN[320] = x0 * x10 as uN[320];
            let x24: bool = -x10;
            let x25: bool = x24 ^ x10;
            x20
        }
    }
}
