// Copyright 2024 The XLS Authors
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
// exception: "Cannot format a bits constructor; got: xN[is_signed=1]"
// issue: "https://github.com/google/xls/issues/1794"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve the specified
//     pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; cannot achieve full
//     throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x15"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x14f_233d_b92b_dfae_0bb9_d78c_48c1_9634"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1_0000_0000"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x100_0000_0000_0000"
//       values: "bits[121]:0x200_0000"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xb5_a3f0_bdf3_c201_7e65_7673_3a75_fc00"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x1e2_4e5b_c334_0263_05f9_0ac3_e95e_0c39"
//       values: "bits[121]:0xde_7f51_f190_c3f8_98c2_ad21_4be1_c16b"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x36_76f9_2e17_16cb_1c9e_6a9b_02a1_c627"
//       values: "bits[121]:0x134_a7a1_1b65_a071_e6ca_4b05_9cc8_be2d"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x51_8131_c027_e05f_d67f_43d9_d0b4_5caa"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x101_89b3_55bf_9f0e_851b_ac73_35a9_7e29"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x11c_349b_7294_ce32_65be_2cd9_32cf_2b42"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x4_0000_0000_0000_0000_0000_0000_0000"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x80"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x8000_0000_0000_0000"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x1fa_1fd5_fdcb_238f_f417_558d_0da4_6e7d"
//       values: "bits[121]:0x1d_6691_95df_0605_8196_a09b_1f97_8355"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x171_40b2_6e17_932b_3eaa_b1cd_de72_f4cc"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x12_80b2_6c9d_73f2_79fd_bc1b_f94e_6b5a"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x9e_f655_802f_4dc8_7f3a_da80_fb3a_db41"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1ad_e401_7b95_8229_7c3f_0be4_b59e_3df0"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x10_0000_0000_0000_0000_0000_0000"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x2000_0000_0000_0000_0000_0000_0000"
//       values: "bits[121]:0x17a_7d0e_bd4b_5468_c28c_9a72_82a2_8d44"
//       values: "bits[121]:0x18e_61aa_4663_b884_8997_50be_30bb_c4a9"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x800"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1_0000_0000_0000_0000_0000_0000"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x177_4cbd_866f_c803_2e5e_8e92_b646_0bec"
//       values: "bits[121]:0x1fe_e009_3aec_9432_509b_bd4c_6782_f022"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x1bb_6026_4eba_165a_06ae_507f_787f_e974"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0xaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//       values: "bits[121]:0x155_5555_5555_5555_5555_5555_5555_5555"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0xff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//       values: "bits[121]:0x7_fcba_1241_5b39_9ff6_16de_ce5e_2dd1"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x0"
//       values: "bits[121]:0x39_9d89_78cf_ca76_c2bb_85c1_a8f2_cc58"
//     }
//   }
// }
//
// END_CONFIG
proc main {
    x15: chan<xN[bool:0x1][121]> in;

    config(x15: chan<xN[bool:0x1][121]> in) { (x15,) }

    init { u36:8 }

    next(x0: u36) {
        {
            let x1: u36 = x0 & x0;
            let x2: token = join();
            let x3: s6 = s6:0x15;
            let x4: s48 = match x0 {
                u36:0x0 | u36:0b101_0101_0101_0101_0101_0101_0101_0101_0101 => s48:0xffff_ffff_ffff,
                u36:0x0..u36:57301982924 => s48:0xffff_ffff_ffff,
                _ => s48:0x7fff_ffff_ffff,
            };
            let x5: uN[144] = x1 ++ x1 ++ x0 ++ x0;
            let x6: bool = xor_reduce(x0);
            let x7: uN[290] = x5 ++ x5 ++ x6 ++ x6;
            let x8: (s48, s6, uN[144], u36) = (x4, x3, x5, x0);
            let x9: s48 = x4 + x4;
            let x11: uN[144] = x5 ^ x5;
            let x12: uN[144] = !x5;
            let x13: bool = x5 < x12;
            let x14: u3 = (x3 as u6)[x12+:u3];
            let x16: (token, xN[bool:0x1][121]) = recv(join(), x15);
            let x17: token = x16.0;
            let x18: xN[bool:0x1][121] = x16.1;
            let x19: u44 = x5[x5+:xN[bool:0x0][44]];
            let x20: uN[144] = gate!(x18 > x18, x5);
            let x21: bool = x13[:];
            let x22: token = join();
            let x23: u8 = u8:0x7f;
            let x24: bool = x6 ^ x13 as bool;
            let x26: bool = x24 | x13 as bool;
            let x27: u2 = decode<xN[bool:0x0][2]>(x26);
            x1
        }
    }
}
