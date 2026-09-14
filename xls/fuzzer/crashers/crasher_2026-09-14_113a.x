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
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[106]:0x12c_d235_fcc0_8a3d_427f_2ae5_c543; bits[10]:0x0\nevaluated opt IR (JIT), evaluated opt IR (interpreter) =\n   (bits[10]:0x28, bits[106]:0x12c_d235_fcc0_8a3d_427f_2ae5_c543, bits[1]:0x0, bits[8]:0xff, bits[10]:0x28)\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\n   (bits[10]:0x0, bits[106]:0x12c_d235_fcc0_8a3d_427f_2ae5_c543, bits[1]:0x0, bits[8]:0xff, bits[10]:0x0)"
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
//     args: "bits[106]:0x12c_d235_fcc0_8a3d_427f_2ae5_c543; bits[10]:0x0"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0xb1"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x3ff"
//     args: "bits[106]:0x31d_7610_3425_cc5b_b55d_b0ce_bfc9; bits[10]:0x155"
//     args: "bits[106]:0x2; bits[10]:0x369"
//     args: "bits[106]:0x10_0000_0000_0000_0000_0000_0000; bits[10]:0x155"
//     args: "bits[106]:0x0; bits[10]:0x100"
//     args: "bits[106]:0x34c_1bc6_506b_a329_6b90_eeaa_b2a1; bits[10]:0x0"
//     args: "bits[106]:0x18d_6703_5c54_5283_03a8_b417_1c87; bits[10]:0x87"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x77"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x80"
//     args: "bits[106]:0x4_0000_0000; bits[10]:0x180"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x1ff"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x3fd"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x3ff"
//     args: "bits[106]:0x0; bits[10]:0x2aa"
//     args: "bits[106]:0xc4_b92d_41c6_3342_5468_999a_ace2; bits[10]:0x62"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x2aa"
//     args: "bits[106]:0x0; bits[10]:0x2"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x1ff"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x3ff"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x2aa"
//     args: "bits[106]:0x0; bits[10]:0x0"
//     args: "bits[106]:0x0; bits[10]:0x34"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x1ff"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x3ff"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x17d"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x1ff"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x115"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x155"
//     args: "bits[106]:0x0; bits[10]:0x2aa"
//     args: "bits[106]:0x0; bits[10]:0x2f6"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x349"
//     args: "bits[106]:0x130_f7b2_2a8f_fe89_6ccb_4057_a46f; bits[10]:0x2aa"
//     args: "bits[106]:0x400; bits[10]:0xc1"
//     args: "bits[106]:0x0; bits[10]:0x40"
//     args: "bits[106]:0x1000_0000_0000_0000; bits[10]:0x2aa"
//     args: "bits[106]:0x0; bits[10]:0x8"
//     args: "bits[106]:0x0; bits[10]:0x160"
//     args: "bits[106]:0x8_0000_0000_0000_0000; bits[10]:0x1ff"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x1ff"
//     args: "bits[106]:0x0; bits[10]:0x1ff"
//     args: "bits[106]:0x8_0000; bits[10]:0x1ff"
//     args: "bits[106]:0x351_c9e4_5ab0_cda5_0f8b_96cd_dc1e; bits[10]:0x9e"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0xaf"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x2b7"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x2a0"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0xf4"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x3f3"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x2aa"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x3ff"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x1e4"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x0"
//     args: "bits[106]:0x2_0000_0000_0000_0000_0000; bits[10]:0x0"
//     args: "bits[106]:0x400; bits[10]:0x2aa"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x0"
//     args: "bits[106]:0x10d_3193_135e_ded2_d0b2_f848_965b; bits[10]:0x25b"
//     args: "bits[106]:0x60_bf76_84cc_6070_bb6d_236b_7fad; bits[10]:0x369"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x3f7"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x155"
//     args: "bits[106]:0x20_0000; bits[10]:0x1ff"
//     args: "bits[106]:0x80_0000; bits[10]:0x9e"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x155"
//     args: "bits[106]:0x0; bits[10]:0x2"
//     args: "bits[106]:0x2_0000_0000_0000; bits[10]:0x127"
//     args: "bits[106]:0x249_e88c_b1d5_93d5_5ac4_e099_1ff4; bits[10]:0x3bd"
//     args: "bits[106]:0x32a_86df_f196_b3c8_7f82_c037_6875; bits[10]:0x1ff"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x371"
//     args: "bits[106]:0x200_0000_0000_0000_0000; bits[10]:0x10"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x1"
//     args: "bits[106]:0x31_2428_7680_40b9_4b29_9302_c291; bits[10]:0x1ff"
//     args: "bits[106]:0x400_0000; bits[10]:0x3ff"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x2aa"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x155"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x3ed"
//     args: "bits[106]:0x0; bits[10]:0x1ff"
//     args: "bits[106]:0x400; bits[10]:0x20"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x2"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x3ff"
//     args: "bits[106]:0x0; bits[10]:0x155"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x155"
//     args: "bits[106]:0x0; bits[10]:0x2aa"
//     args: "bits[106]:0x2_0000_0000_0000; bits[10]:0x40"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x2aa"
//     args: "bits[106]:0x2f2_614c_48f7_3304_9607_21b8_cb85; bits[10]:0x385"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x1"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x1ff"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x1ff"
//     args: "bits[106]:0x0; bits[10]:0xa"
//     args: "bits[106]:0x299_2094_2fa4_fd8b_b9e7_598b_cf3d; bits[10]:0x8"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x8"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x0"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x0"
//     args: "bits[106]:0x2_0000_0000_0000; bits[10]:0x20"
//     args: "bits[106]:0x10; bits[10]:0x10"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x1"
//     args: "bits[106]:0x306_a3a0_ea12_6c34_cab5_63dd_dabb; bits[10]:0x7a"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x3a1"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x15"
//     args: "bits[106]:0x4000_0000; bits[10]:0x114"
//     args: "bits[106]:0x0; bits[10]:0x0"
//     args: "bits[106]:0x400_0000_0000; bits[10]:0x155"
//     args: "bits[106]:0x0; bits[10]:0x0"
//     args: "bits[106]:0x80_0000_0000; bits[10]:0x155"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x39a"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x1d7"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x1ff"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x1ff"
//     args: "bits[106]:0x51_03b5_a0af_f023_e4c4_96ac_c8d1; bits[10]:0x155"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x19b"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x2ee"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x39f"
//     args: "bits[106]:0x0; bits[10]:0x3f1"
//     args: "bits[106]:0x80_0000_0000_0000_0000_0000_0000; bits[10]:0x0"
//     args: "bits[106]:0x3ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x3ff"
//     args: "bits[106]:0x2000_0000; bits[10]:0x0"
//     args: "bits[106]:0x1ff_ffff_ffff_ffff_ffff_ffff_ffff; bits[10]:0x22f"
//     args: "bits[106]:0x8_0000_0000_0000_0000_0000; bits[10]:0x2aa"
//     args: "bits[106]:0x0; bits[10]:0x155"
//     args: "bits[106]:0x0; bits[10]:0x85"
//     args: "bits[106]:0x9f_bb42_1a4c_0ffb_5483_3928_7b7a; bits[10]:0x0"
//     args: "bits[106]:0x10_0000_0000; bits[10]:0x7"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x1"
//     args: "bits[106]:0x8_2a77_f067_86b1_2a5f_589c_9a5d; bits[10]:0x7d"
//     args: "bits[106]:0x237_7e21_08ea_29a4_1107_41b4_ca42; bits[10]:0x2fa"
//     args: "bits[106]:0x2aa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[10]:0x0"
//     args: "bits[106]:0x200_0000_0000_0000_0000_0000_0000; bits[10]:0xce"
//     args: "bits[106]:0x155_5555_5555_5555_5555_5555_5555; bits[10]:0x304"
//   }
// }
// 
// END_CONFIG
type x3 = s10;
fn main(x0: sN[106], x1: s10) -> (s10, sN[106], bool, s8, s10) {
    {
        let x2: s10 = x1 / s10:0x2aa;
        let x4: x3[3] = [x1, x2, x2];
        let x5: s8 = s8:0xff;
        let x6: s5 = s5:0x15;
        let x7: s44 = s44:0x555_5555_5555;
        let x8: u7 = (x7 as u44)[16+:u7];
        let x9: s10 = x1 << if x8 >= u7:0x1 { u7:0x1 } else { x8 };
        let x10: s10 = x2 - x8 as s10;
        let x11: u7 = -x8;
        let x12: bool = or_reduce(x11);
        let x13: u7 = bit_slice_update(x8, x12, x8);
        let x14: s10 = x11 as s10 ^ x10;
        let x15: bool = or_reduce(x12);
        let x16: bool = x12[0+:bool];
        let x17: uN[106] = (x0 as uN[106])[:];
        let x18: s10 = x13 as s10 & x14;
        let x19: sN[106] = x0 - x0;
        let x20: bool = x14 > x18;
        let x21: u7 = bit_slice_update(x13, x8, x17);
        let x22: sN[106] = !x19;
        let x23: uN[81] = (x19 as uN[106])[:81];
        let x24: bool = or_reduce(x20);
        let x25: s8 = x5 % s8:0xe9;
        let x26: u7 = bit_slice_update(x8, x12, x24);
        let x27: s10 = one_hot_sel(x26, [x9, x10, x14, x1, x10, x1, x18]);
        let x28: u4 = (x27 as u10)[-4:];
        let x29: bool = gate!(x8 < x1 as u7, x24);
        let x30: bool = bit_slice_update(x12, x23, x21);
        let x31: xN[bool:0x0][99] = (x19 as uN[106])[0+:uN[99]];
        (x18, x0, x20, x5, x18)
    }
}
