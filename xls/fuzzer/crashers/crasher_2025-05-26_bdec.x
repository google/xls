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
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the
//     specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full
//     throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x6"
//       values: "bits[60]:0x800_0000_0000_0000"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x800_0000_0000_0000"
//       values: "bits[60]:0x8c6_1870_1d71_18ff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0xa81_d5a1_f59e_72c6"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x2_0000_0000_0000"
//       values: "bits[60]:0x708_5679_ab37_a2ba"
//       values: "bits[60]:0x8000"
//       values: "bits[60]:0xba3_f640_8bc4_62b8"
//       values: "bits[60]:0x977_1392_604e_5633"
//       values: "bits[60]:0x20"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x8000_0000_0000"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x20_0000"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xfb_30bb_101e_157b"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0xeac_cc9f_a450_e203"
//       values: "bits[60]:0xa34_1245_237d_e66c"
//       values: "bits[60]:0x800"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x40_0000_0000"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xbe2_f052_7d6d_9674"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x40_0000_0000_0000"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x40_0000_0000_0000"
//       values: "bits[60]:0x2_0000_0000_0000"
//       values: "bits[60]:0x57d_0602_86bf_fd6e"
//       values: "bits[60]:0x418_79c2_54eb_14f8"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x1_0000"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x64b_f5b1_4b07_78ba"
//       values: "bits[60]:0x2000_0000_0000"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x8000"
//       values: "bits[60]:0x5fa_204c_3576_bd0d"
//       values: "bits[60]:0x32e_bd8d_ba0d_556d"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x100"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x8a0_77ae_4916_8619"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x4_0000_0000"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x200_0000_0000"
//       values: "bits[60]:0x10_0000"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x1000"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x537_728c_eb35_7378"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x2_0000_0000"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x40_0000_0000"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x100"
//       values: "bits[60]:0x20_0000"
//       values: "bits[60]:0xe7f_38ea_f5df_6b09"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x438_3c92_0e24_166a"
//       values: "bits[60]:0x325_75d2_0c54_c0e7"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x0"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x1ef_b449_dab8_7ba1"
//       values: "bits[60]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[60]:0x8f8_7a57_d84b_cf3c"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//       values: "bits[60]:0x7_f3c2_7710_4cc0"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0xfff_ffff_ffff_ffff"
//       values: "bits[60]:0x555_5555_5555_5555"
//       values: "bits[60]:0x7ff_ffff_ffff_ffff"
//     }
//     inputs {
//       channel_name: "sample__x29"
//       values: "bits[33]:0x10_0810"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_d1f5_14c5"
//       values: "bits[33]:0x600d_4000"
//       values: "bits[33]:0x5641_18bb"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_bfed_f7bb"
//       values: "bits[33]:0xaaae_aaaa"
//       values: "bits[33]:0x1_f51f_6bce"
//       values: "bits[33]:0x43fd_479d"
//       values: "bits[33]:0x1_9cc9_1b4e"
//       values: "bits[33]:0x4efe_ffbe"
//       values: "bits[33]:0x180_0a04"
//       values: "bits[33]:0x8000"
//       values: "bits[33]:0xd008_c226"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x604c_5633"
//       values: "bits[33]:0x814e_89c3"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_2d7f_f6df"
//       values: "bits[33]:0x11e2_8797"
//       values: "bits[33]:0x5fac_eaf2"
//       values: "bits[33]:0x2024_0180"
//       values: "bits[33]:0x1_fbc7_fffb"
//       values: "bits[33]:0x1_0dff_cd6a"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x1_c557_4475"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_a552_e283"
//       values: "bits[33]:0x1_237d_e66c"
//       values: "bits[33]:0xe800_c022"
//       values: "bits[33]:0x1000_0000"
//       values: "bits[33]:0xeaaa_bbfa"
//       values: "bits[33]:0x1_3add_9736"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x80_0003"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0xed09_7742"
//       values: "bits[33]:0x841_90c8"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_effc_f3ff"
//       values: "bits[33]:0xffbd_f7fb"
//       values: "bits[33]:0x1_0004_0ca0"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xd63f_fc0c"
//       values: "bits[33]:0xae73_0472"
//       values: "bits[33]:0xa8aa_8aaa"
//       values: "bits[33]:0x5221_7924"
//       values: "bits[33]:0x1_aaab_ae29"
//       values: "bits[33]:0xdd59_ce01"
//       values: "bits[33]:0x1_e3f9_eb9d"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0xe2f3_7b02"
//       values: "bits[33]:0x1_b5ff_debe"
//       values: "bits[33]:0x2c01_8000"
//       values: "bits[33]:0x3476_b50d"
//       values: "bits[33]:0xfe15_7969"
//       values: "bits[33]:0xf760_ea3b"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x8002_0081"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_820a_6008"
//       values: "bits[33]:0x1_df55_6155"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x44a2_0e40"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x1_7aa6_4bc8"
//       values: "bits[33]:0x7afb_f5aa"
//       values: "bits[33]:0xfd3f_ffbe"
//       values: "bits[33]:0x1_0e2a_0d0b"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_7311_9107"
//       values: "bits[33]:0x1_6544_5f17"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0xb72e_f860"
//       values: "bits[33]:0x20"
//       values: "bits[33]:0x1_3750_93d4"
//       values: "bits[33]:0x1_5145_7157"
//       values: "bits[33]:0x1_faeb_fbdf"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_d902_cf42"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x8089_0060"
//       values: "bits[33]:0x400"
//       values: "bits[33]:0x742b_b689"
//       values: "bits[33]:0x1_7555_5545"
//       values: "bits[33]:0x1_56f0_b831"
//       values: "bits[33]:0x1_bcb6_5995"
//       values: "bits[33]:0x1_5515_755d"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x1_e1c4_a12e"
//       values: "bits[33]:0x1_ffff_f9bf"
//       values: "bits[33]:0xaaaa_aaaa"
//       values: "bits[33]:0x1_0000"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x1_b99e_eff5"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x37ea_193c"
//       values: "bits[33]:0x2d55_c5e7"
//       values: "bits[33]:0x1_ffff_ffff"
//       values: "bits[33]:0x800_0000"
//       values: "bits[33]:0xffff_ffff"
//       values: "bits[33]:0x13a7_081e"
//       values: "bits[33]:0x1_9cae_a8be"
//       values: "bits[33]:0x1_5555_5555"
//       values: "bits[33]:0x0"
//       values: "bits[33]:0x1_7fff_fb7f"
//       values: "bits[33]:0x1_1401_598a"
//       values: "bits[33]:0x1_8ebb_809b"
//       values: "bits[33]:0x1_d228_6db1"
//       values: "bits[33]:0x8ac8_bc1c"
//       values: "bits[33]:0x1_4569_800a"
//       values: "bits[33]:0x1_ffff_fbff"
//       values: "bits[33]:0x1_bb00_aec8"
//       values: "bits[33]:0x1_4555_4f55"
//       values: "bits[33]:0x38f5_8bab"
//       values: "bits[33]:0x1_1554_4755"
//       values: "bits[33]:0x8000_0000"
//     }
//     inputs {
//       channel_name: "sample__x41"
//       values: "bits[11]:0x10"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x2c4"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x754"
//       values: "bits[11]:0x6ff"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x7c1"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0xa2"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x8"
//       values: "bits[11]:0x3f6"
//       values: "bits[11]:0x68c"
//       values: "bits[11]:0x7fd"
//       values: "bits[11]:0x323"
//       values: "bits[11]:0x555"
//       values: "bits[11]:0x38a"
//       values: "bits[11]:0x7f7"
//       values: "bits[11]:0x555"
//       values: "bits[11]:0x40"
//       values: "bits[11]:0x1e1"
//       values: "bits[11]:0x1a8"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x555"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x253"
//       values: "bits[11]:0x71c"
//       values: "bits[11]:0xe8"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x72d"
//       values: "bits[11]:0x90"
//       values: "bits[11]:0x500"
//       values: "bits[11]:0x555"
//       values: "bits[11]:0x73f"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x674"
//       values: "bits[11]:0x36d"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x7ef"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x6ff"
//       values: "bits[11]:0x6a0"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x4a8"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x779"
//       values: "bits[11]:0x34"
//       values: "bits[11]:0x7df"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x4d8"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x504"
//       values: "bits[11]:0x568"
//       values: "bits[11]:0x555"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x1d4"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x58e"
//       values: "bits[11]:0x1"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x1c9"
//       values: "bits[11]:0x2ab"
//       values: "bits[11]:0x5aa"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x117"
//       values: "bits[11]:0x80"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x78"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x575"
//       values: "bits[11]:0x77b"
//       values: "bits[11]:0x492"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x7c2"
//       values: "bits[11]:0x7bf"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x400"
//       values: "bits[11]:0x785"
//       values: "bits[11]:0x5d5"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x555"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x5ff"
//       values: "bits[11]:0x1b7"
//       values: "bits[11]:0x100"
//       values: "bits[11]:0x337"
//       values: "bits[11]:0x449"
//       values: "bits[11]:0x69f"
//       values: "bits[11]:0x455"
//       values: "bits[11]:0x90"
//       values: "bits[11]:0x11a"
//       values: "bits[11]:0x4ef"
//       values: "bits[11]:0x4bf"
//       values: "bits[11]:0x2aa"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x43"
//       values: "bits[11]:0x380"
//       values: "bits[11]:0x55f"
//       values: "bits[11]:0x402"
//       values: "bits[11]:0x200"
//       values: "bits[11]:0x8"
//       values: "bits[11]:0x7fd"
//       values: "bits[11]:0x7ff"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x531"
//       values: "bits[11]:0x6bf"
//       values: "bits[11]:0x6c8"
//       values: "bits[11]:0x3ff"
//       values: "bits[11]:0x7db"
//       values: "bits[11]:0x0"
//       values: "bits[11]:0x600"
//     }
//   }
// }
//
// END_CONFIG
type x1 = u8;
type x5 = s43;
type x17 = (bool, bool, bool, bool);
type x26 = x1;

fn x12(x13: x1) -> (bool, bool, bool, bool) {
    {
        let x14: bool = or_reduce(x13);
        let x15: bool = x14[x13+:bool];
        let x16: bool = x14 / bool:false;
        (x14, x16, x15, x15)
    }
}

fn x23(x24: x1) -> x1 {
    {
        let x25: bool = x24 <= x24;
        x24
    }
}

fn x45(x46: x26) -> (x26, x26) {
    {
        let x47: x26 = -x46;
        let x48: x26 = bit_slice_update(x46, x46, x47);
        (x47, x47)
    }
}

proc main {
    x6: chan<u60> in;
    x29: chan<u33> in;
    x41: chan<s11> in;

    config(x6: chan<u60> in, x29: chan<u33> in, x41: chan<s11> in) { (x6, x29, x41) }

    init { u60:322006538705383228 }

    next(x0: u60) {
        {
            let x2: x1[1678] = "Y6On[p\'vSd2I8[sbbokizCm>@,A mu%?K6zP\"HIk\"68Ei{A[mZti;C(;S]:!_EKi&<I=*:&Q~c,{Hag%t8nr{;Ya#M5%_.}FsfY/\'iiH&|2x8rmKO;z!V`ykgvM^Zdw3l=WB|wH;]x2Y+#(C^(6Jw`X5m QM[X_|2{QyyuOAaEiAVd8Dp_<6\";#J]0F^ucM|WX?0@>:{^ssYGeeHI.JGc8P[+|v3Y\'QD[QnZY;7zqW+#LzQ1.kV%y>Ys9tJ[K((\\fPu[,4M0~N=D}J&yK0P~Ipj,ken_s^s%4m{Fihxg|)EIvNVg{~v\"&nOBY;)l=S+zb)`AZ<J=~l\'Wjb`(+:akP6I*Q)suDjkk1+]w;\'=3]zYgC{&\"-HV^q\'zM(FZ=yoah\"RD[SfJUbna]hJ/}b7r^8MVkG8`wAxE56nM<&iozv#K0~DpYwoy~PYeN*Y,4q^15H;u`+t+zc\"~(Rgnp2\"OuZ``(gX7>*bI1$uqejr#@?Mi/=x;qLm%=.;vHkffR\'Bx8s9GfA-Ud/\'!<,H}t(X:9)xtz RAhYEM#deUj3#$?s?Pb Y$?Di~@&-Ul;[=EK*K$F`J4mMRC~GgOQR$FQWf\\wmhkksT]Y%XpY*:gafc%xmA8(+Y1,^sBiTv+!;F5C\\zhRMWNqX5Y0UEX@{zkQvOI\"7O/Ci).,t0!FBOF>qigV ^WmQ(h1?_+D_Ib}mF:\\dFKL;A;+8v@mrEn\'xQ.oBdZu*E;e4U@_ZfmBK9$dgk92%t7R 1}D=&JLuJO0~tgvpkH&.-&qYZzsMG0fs%9A\'>PqkpEMYj\"&3ol-ER8aurG0{HY2}|mveaR}:&X_Qj^:FC`rYE){.IAHkRwS>2V@/ \"#m}?YkXqW0{XIcf.5&>!h4 #eL)arpS2o \'/8=$r^_Z9V&(\\MafaA##MJ6pYCC9${Bj\"WCFt\"\"s:r[P\"]01$SSkzgv4Xac2we~mdHi:a4KDF<OnKI:ZEXp*8\'Xt~KUgh}N0\'0Jz2Q`41Q0\"T`?U%bRmWkM,-tDY~!a[U+K;x_KUWxQ][HUe+3\\sMf;vQFA1f5B}_y#wS)@3\'aLOl@H#A9/8NO [~pI4xBIA;{IMExS{E|_c#cDeeVruEoX~v?z=n#%_JV-A~$*(0.SamVIn/t(h%^vw(>9WIp<N&o{J=YJV\"ySV([YGewb:xTsh800{C HS*;>Oi-$Fq~h5j>a]_)4s@\"Jh\"0},aML2#w+BoJ8\'J(DMGW`s@?\\#+/HQ+R/<,STn.]RUu#aT9 ^/ERQg/y@m9HcHIZ]d5Bb)A>$15b]X\\z>H^~5>.fZJH!boIY1D1kchr;Uozx/)9TL?VmIl 87o+O=*acttW}~Bp>N*Tm&rAg;BJ&s5Yf]! pjTg:+u{qefv`br\'1^s3Y0*+~6e=qc-\"jhUk?eB\'kOzLEv3}\\MeK!i,FI`W)aIIJQ$+J.0y0\\$S\\Z>Uu,_`;40[VXVGa_s-BbrlsY [CdF}UR46C R1HZo([V]7r/I@5m6LQfuzvjBzNX/IOkQ@p*.:*-K, FDM$!CT8pUExA`^#q(eg)TA),3!4|ymQGd[5s/|\'S0R*lud>]!\"dS}4c1e($`NhA>tXtCitRVRWqHR3b<NS&lK_\']:_tuhD,rJnNgZ\"6_i[gLvOw)\"N2B)+;c:P)o\\7R \'U4!0JE|qYCqk{&!*$. :y1vIC4\'<_>{3\\]<}";
            let x3: u60 = x0 - x0;
            let x4: u22 = x3[x0+:u22];
            let x7: (token, u60, bool) = recv_non_blocking(join(), x6, x0);
            let x8: token = x7.0;
            let x9: u60 = x7.1;
            let x10: bool = x7.2;
            let x11: bool = ctz(x10);
            let x18: x17[1678] = map(x2, x12);
            let x19: x1 = x2[if x9 >= u60:0x1f5 { u60:0x1f5 } else { x9 }];
            let x20: bool = !x11;
            let x21: u60 = x9 * x19 as u60;
            let x22: u60 = gate!(x0 as bool > x11, x3);
            let x27: x26[1678] = map(x2, x23);
            let x28: u51 = x21[9:];
            let x30: (token, u33) = recv(join(), x29);
            let x31: token = x30.0;
            let x32: u33 = x30.1;
            let x33: u33 = -x32;
            let x34: bool = -x10;
            let x35: bool = xor_reduce(x28);
            let x36: s64 = s64:0x5555_5555_5555_5555;
            let x37: u33 = x30.1;
            let x38: bool = x34 * x0 as bool;
            let x39: bool = x3 < x32 as u60;
            let x40: u60 = signex(x21, x3);
            let x42: (token, s11) = recv(x31, x41);
            let x43: token = x42.0;
            let x44: s11 = x42.1;
            let x49: u60 = !x22;
            let x50: u15 = x22[-43:-28];
            x0
        }
    }
}
