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
// exception: "Redacted for open source export"
// issue: "https://github.com/google/xls/issues/4372"
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
//     args: "bits[57]:0x1; bits[20]:0x8_080d; bits[40]:0xff_ffff_ffff; bits[16]:0x9275"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xd_da6e; bits[40]:0xbb_bafe_3ef6; bits[16]:0xffff"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0x7_fabb; bits[40]:0xff_afff_fff7; bits[16]:0x0"
//     args: "bits[57]:0xc5_f6e9_2cba_ab65; bits[20]:0xf_ffff; bits[40]:0xf9_3cfc_ab45; bits[16]:0xfe3c"
//     args: "bits[57]:0x0; bits[20]:0x7_ffff; bits[40]:0xdc_0ff1_3ef3; bits[16]:0xc008"
//     args: "bits[57]:0x4000_0000; bits[20]:0x7_ffff; bits[40]:0x20_21cf_4b3d; bits[16]:0x5555"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0xa_aaaa; bits[40]:0xaa_aaa5_dde8; bits[16]:0xc2aa"
//     args: "bits[57]:0xe3_f741_425f_1972; bits[20]:0xa_aaaa; bits[40]:0x400; bits[16]:0x8a50"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x1_1554; bits[40]:0xff_ffff_ffff; bits[16]:0xf758"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x40ab; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x0"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x4_4bd0; bits[40]:0x7_f04e_525e; bits[16]:0x7fff"
//     args: "bits[57]:0x100_0000_0000; bits[20]:0x7_ffff; bits[40]:0xff_ffff_ffff; bits[16]:0xfb72"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0xf_ffff; bits[40]:0x54_55d5_5715; bits[16]:0xc7ff"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x0; bits[40]:0xba_f87f_c775; bits[16]:0xdb7b"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x5_82f5; bits[40]:0xc_b7c6_6c7b; bits[16]:0xaaaa"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x7_ffff; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xfdff"
//     args: "bits[57]:0x0; bits[20]:0x1810; bits[40]:0xce_95a3_4c09; bits[16]:0x7fff"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xf_ffff; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xe40d"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xa_aaaa; bits[40]:0x55_5555_5555; bits[16]:0xff12"
//     args: "bits[57]:0xd6_6a33_07b4_7e54; bits[20]:0x4_5a44; bits[40]:0x32_340c_d657; bits[16]:0x8000"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xe_7eeb; bits[40]:0xb7_e7a0_2050; bits[16]:0x76e0"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x1000; bits[40]:0x400_0000; bits[16]:0x7fff"
//     args: "bits[57]:0x80; bits[20]:0x28d; bits[40]:0x55_5555_5555; bits[16]:0xffff"
//     args: "bits[57]:0x0; bits[20]:0x1; bits[40]:0xd4_0510_2308; bits[16]:0x7fff"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xc_fbff; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xf9f9"
//     args: "bits[57]:0x8_0000_0000_0000; bits[20]:0x2400; bits[40]:0x0; bits[16]:0x5908"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x1_5555; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x7fff"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xf_ffff; bits[40]:0x2; bits[16]:0x802"
//     args: "bits[57]:0x1e8_fbbb_3801_2b73; bits[20]:0x4_e35f; bits[40]:0x56_3dd1_8cd7; bits[16]:0x0"
//     args: "bits[57]:0x4000_0000_0000; bits[20]:0x8_0000; bits[40]:0x8_0000_0400; bits[16]:0x4000"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x5_5555; bits[40]:0x50_5075_c5f5; bits[16]:0xc5e5"
//     args: "bits[57]:0x800; bits[20]:0x1_1800; bits[40]:0x16_314f_2862; bits[16]:0x927"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xa_aaaa; bits[40]:0x1_0000; bits[16]:0xaaba"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0xa_085a; bits[40]:0x83_c7fe_50ec; bits[16]:0x7fff"
//     args: "bits[57]:0x10_0000_0000_0000; bits[20]:0x9_4cd1; bits[40]:0xf4_801b_ed65; bits[16]:0xade0"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0x7_ffff; bits[40]:0x0; bits[16]:0xf6bf"
//     args: "bits[57]:0x143_86a4_3dcc_ac80; bits[20]:0x2_aed0; bits[40]:0x24_293c_8d88; bits[16]:0x8fac"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0x9_ffaf; bits[40]:0xbf_ffbf_ef6f; bits[16]:0x9faf"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x1_d4d5; bits[40]:0x7f_ffff_ffff; bits[16]:0x752b"
//     args: "bits[57]:0x13a_d2a5_4a7a_91ac; bits[20]:0xf_ffff; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xfbff"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xd_fdff; bits[40]:0xdf_f6f0_8000; bits[16]:0x931"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xf_ffff; bits[40]:0x2b05_0cc1; bits[16]:0x0"
//     args: "bits[57]:0x0; bits[20]:0xf_ffff; bits[40]:0x7f_ffff_ffff; bits[16]:0x85ea"
//     args: "bits[57]:0x0; bits[20]:0x4040; bits[40]:0x7f_ffff_ffff; bits[16]:0x5b7a"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xa_feff; bits[40]:0x55_5555_5555; bits[16]:0x5555"
//     args: "bits[57]:0x800_0000; bits[20]:0x2010; bits[40]:0x82_0505_fc55; bits[16]:0xea1c"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x5_9753; bits[40]:0x57_5597_5757; bits[16]:0x2004"
//     args: "bits[57]:0x6c_7c7a_8a57_f4df; bits[20]:0x8000; bits[40]:0x7f_ffff_ffff; bits[16]:0xffff"
//     args: "bits[57]:0x0; bits[20]:0x7_5071; bits[40]:0x7f_ffff_ffff; bits[16]:0x5555"
//     args: "bits[57]:0x100; bits[20]:0x100; bits[40]:0x80; bits[16]:0x0"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0x4; bits[40]:0xff_ffff_ffff; bits[16]:0x5555"
//     args: "bits[57]:0x800_0000_0000; bits[20]:0xa_aaaa; bits[40]:0x12_8041_0000; bits[16]:0x5555"
//     args: "bits[57]:0x0; bits[20]:0x3_f80f; bits[40]:0x7f_360e_1ea3; bits[16]:0x80"
//     args: "bits[57]:0x8; bits[20]:0x0; bits[40]:0x15_8940_a022; bits[16]:0x8518"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x80; bits[40]:0x54_5754_4711; bits[16]:0xaaaa"
//     args: "bits[57]:0x8_0000; bits[20]:0x8_0804; bits[40]:0x8_0004; bits[16]:0x1"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xf_ffff; bits[40]:0xfb_677f_ef7b; bits[16]:0xffff"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x5_5555; bits[40]:0x36_3965_e92d; bits[16]:0x45c4"
//     args: "bits[57]:0x3d_a693_b0e2_651f; bits[20]:0x2_2f09; bits[40]:0x55_5555_5555; bits[16]:0x9ec9"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xa_f3f6; bits[40]:0x0; bits[16]:0xcfff"
//     args: "bits[57]:0x0; bits[20]:0xb008; bits[40]:0x480_03a4; bits[16]:0x0"
//     args: "bits[57]:0x0; bits[20]:0xf_ffff; bits[40]:0x8e_7bc3_c73d; bits[16]:0x4106"
//     args: "bits[57]:0x400_0000; bits[20]:0xc_2229; bits[40]:0x55_5555_5555; bits[16]:0x222d"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x0; bits[40]:0xff_ffff_ffff; bits[16]:0xe6de"
//     args: "bits[57]:0x400; bits[20]:0x5_5555; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x0"
//     args: "bits[57]:0x20_0000_0000_0000; bits[20]:0xa_aaaa; bits[40]:0x0; bits[16]:0x4088"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x7_afdc; bits[40]:0x0; bits[16]:0xafdd"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xa_aaaa; bits[40]:0xf3_7e21_6959; bits[16]:0x5555"
//     args: "bits[57]:0x0; bits[20]:0xb_1004; bits[40]:0xc0_1469_0c14; bits[16]:0x2000"
//     args: "bits[57]:0x18c_1922_dfbc_b4d5; bits[20]:0x4; bits[40]:0x7f_ffff_ffff; bits[16]:0x4"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x7_4555; bits[40]:0x70_544a_a89a; bits[16]:0x5555"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0xa_aaaa; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xaaaa"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xf_f7ff; bits[40]:0xbb_1f3f_3fcb; bits[16]:0x1"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0x2; bits[40]:0xe4_4adf_0916; bits[16]:0x75fa"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x5_5555; bits[40]:0x4f_47da_d7d7; bits[16]:0x5555"
//     args: "bits[57]:0x0; bits[20]:0xf_ffff; bits[40]:0xff_ffff_ffff; bits[16]:0x1927"
//     args: "bits[57]:0x0; bits[20]:0xf_ffff; bits[40]:0x0; bits[16]:0xaaaa"
//     args: "bits[57]:0x1d7_8286_1504_717c; bits[20]:0x7_512c; bits[40]:0xf6_efd8_7dad; bits[16]:0x353f"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x8000; bits[40]:0x1c_1303_edd1; bits[16]:0xa0f4"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x20; bits[40]:0x80_0000; bits[16]:0xdac7"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0x5_5555; bits[40]:0x7f_ffff_ffff; bits[16]:0xaaea"
//     args: "bits[57]:0x0; bits[20]:0x0; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xaaaa"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0xf_ffff; bits[40]:0x7f_ffff_ffff; bits[16]:0x0"
//     args: "bits[57]:0x0; bits[20]:0xa_aaaa; bits[40]:0x9_0000_0004; bits[16]:0x400"
//     args: "bits[57]:0x1b_b5ec_b6f0_72ea; bits[20]:0x7_ffff; bits[40]:0x7f_fdf0_0000; bits[16]:0x70d2"
//     args: "bits[57]:0x8_0000; bits[20]:0x9_000c; bits[40]:0x96_f711_5785; bits[16]:0x924c"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0xa_abaa; bits[40]:0xaa_baa0_0200; bits[16]:0xffff"
//     args: "bits[57]:0x0; bits[20]:0x1_0000; bits[40]:0x2_0002_234a; bits[16]:0x0"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xc_efbf; bits[40]:0xff_b7f9_ffce; bits[16]:0x876c"
//     args: "bits[57]:0x4_0000_0000_0000; bits[20]:0xf_ffff; bits[40]:0xff_ffb4_91a2; bits[16]:0x63d1"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xf_ffff; bits[40]:0x400; bits[16]:0x7fff"
//     args: "bits[57]:0x100_0000; bits[20]:0x8_4009; bits[40]:0x80_0102_1000; bits[16]:0xaaaa"
//     args: "bits[57]:0x200; bits[20]:0x1_2200; bits[40]:0x2c_cf5a_d43e; bits[16]:0x0"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0x7_77c5; bits[40]:0x3e_0d62_5d1a; bits[16]:0xdf32"
//     args: "bits[57]:0x1000_0000; bits[20]:0x1_32de; bits[40]:0x2_9400_2000; bits[16]:0x6280"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xf_fef7; bits[40]:0x10; bits[16]:0xfff7"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x9_27ee; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x5555"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x7_7c1d; bits[40]:0xce_585c_df00; bits[16]:0xaaaa"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x2_0000; bits[40]:0x55_5555_5555; bits[16]:0x0"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xc_4f4f; bits[40]:0x20_a0b8_7f69; bits[16]:0xe4f"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xf_ffdb; bits[40]:0x9d_be5b_1dea; bits[16]:0x0"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x4_473d; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x5555"
//     args: "bits[57]:0x2000_0000_0000; bits[20]:0x0; bits[40]:0x12_5800_3000; bits[16]:0x0"
//     args: "bits[57]:0x0; bits[20]:0x3_4df5; bits[40]:0xce_0051; bits[16]:0x5555"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xd_edef; bits[40]:0x55_5555_5555; bits[16]:0xafef"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x0; bits[40]:0x98_2406_d8e9; bits[16]:0x5755"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x7_5855; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x5c11"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0x5_5555; bits[40]:0xff_ffff_ffff; bits[16]:0xbaba"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0x7_9e9a; bits[40]:0x7a_556b_77ed; bits[16]:0x7dd0"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xa_7f77; bits[40]:0x7f_ffff_ffff; bits[16]:0x37f2"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x20; bits[40]:0xb_0006_01a3; bits[16]:0xf077"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0xf_5755; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xffff"
//     args: "bits[57]:0x0; bits[20]:0x0; bits[40]:0x12_0061_cc82; bits[16]:0x900"
//     args: "bits[57]:0x400; bits[20]:0xa_aaaa; bits[40]:0x55_5555_5555; bits[16]:0x100"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0xf_23be; bits[40]:0xdb_6066_6233; bits[16]:0x6223"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0x7_ffff; bits[40]:0xaa_aaaa_aaaa; bits[16]:0xaaae"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0x0; bits[40]:0x7f_ffff_ffff; bits[16]:0xb779"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xf_ffff; bits[40]:0x1f_67d4_79d4; bits[16]:0x5555"
//     args: "bits[57]:0x8; bits[20]:0x5_5555; bits[40]:0x808; bits[16]:0xb4b"
//     args: "bits[57]:0xff_ffff_ffff_ffff; bits[20]:0xf_fffe; bits[40]:0x7f_ffff_ffff; bits[16]:0x20"
//     args: "bits[57]:0x0; bits[20]:0x40; bits[40]:0x40_0004_0020; bits[16]:0xaaaa"
//     args: "bits[57]:0x1ff_ffff_ffff_ffff; bits[20]:0x7_ffff; bits[40]:0x0; bits[16]:0xb3ef"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0x5_94f5; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x1d74"
//     args: "bits[57]:0x1e1_e2f7_f9a4_cc22; bits[20]:0x0; bits[40]:0xaa_aaaa_aaaa; bits[16]:0x7fff"
//     args: "bits[57]:0x155_5555_5555_5555; bits[20]:0xb_fc37; bits[40]:0x8f_a37c_8202; bits[16]:0xd322"
//     args: "bits[57]:0xaa_aaaa_aaaa_aaaa; bits[20]:0x0; bits[40]:0xae_2b2a_aa3e; bits[16]:0x7fb6"
//     args: "bits[57]:0x8000_0000; bits[20]:0x8_0954; bits[40]:0x0; bits[16]:0x8028"
//     args: "bits[57]:0x92_933e_a572_9f14; bits[20]:0x0; bits[40]:0x80_0000_0000; bits[16]:0xa00"
//   }
// }
// 
// END_CONFIG
type x42 = u8;
fn x6(x7: bool, x8: bool, x9: s20, x10: bool, x11: s57) -> (bool, bool, bool) {
    {
        let x12: bool = x8[0+:bool];
        let x13: (bool, bool, bool, bool) = (x8, x12, x8, x7);
        let (_, x14, x15, x16) = (x8, x12, x8, x7);
        (x15, x15, x10)
    }
}
fn x22(x23: s20, x24: s40, x25: s57, x26: s40) -> (s40, s40) {
    {
        let x27: s40 = -x26;
        (x27, x27)
    }
}
fn main(x0: s57, x1: s20, x2: s40, x3: s16) -> (s40, s20, bool, bool, bool) {
    {
        let x4: bool = (x1 as s16) < x3;
        let x5: u2 = decode<u2>(x4);
        let x17: (bool, bool, bool) = x6(x4, x4, x1, x4, x0);
        let x18: (s7, u27, u15, (s39,)) = match x17 {
            (bool:false, bool:true, bool:0x1) => (s7:0x7f, u27:0x555_5555, u15:0x0, (s39:0x10_9433_aea1,)),
            (xN[bool:0x0][1]:true, bool:0x0, bool:0x0..bool:0) => (s7:0x3f, u27:0x2aa_aaaa, u15:16383, (s39:0x2a_aaaa_aaaa,)),
            (bool:0x1, _, bool:true) | (bool:false, bool:true..bool:true, bool:0x1) => (s7:0x7f, u27:0x0, u15:0x3fff, (s39:0x52_ca40_3ba1,)),
            (bool:true, bool:false, bool:true..bool:true) => (s7:0x2a, u27:0x8_0000, u15:0x7fff, (s39:0x2a_aaaa_aaaa,)),
            _ => (s7:0x4, u27:0x7ff_ffff, u15:0x2aaa, (s39:0x48_b034_7606,)),
        };
        let x19: bool = x4 | x4;
        let x21: s20 = one_hot_sel(x19, [x1]);
        let x28: (s40, s40) = x22(x1, x2, x0, x2);
        let x29: u6 = (x1 as u20)[x5+:u6];
        let x30: bool = x4 | x4;
        let x31: bool = x30 <= x30;
        let x32: s16 = !x3;
        let x33: bool = x4 | x3 as bool;
        let x34: bool = x4 + x5 as bool;
        let x35: bool = x31[x19+:bool];
        let x37: u8 = x35 ++ x5 ++ x30 ++ x5 ++ x34 ++ x30;
        let x38: bool = x4 - x29 as bool;
        let x39: bool = and_reduce(x38);
        let x40: bool = x34[:];
        let x41: bool = x4 ^ x33;
        let x43: x42[1454] = "!CE$z$LW0C{wr7D%]~jkO\\p1!vy=$H#Gx<hh#{*[%wo)-Y5}Nw}\'|Na|$K#W[6\' -v!*NKW+(x[#~0niRo,f`]j!fn98_R/9.1w@43L(-r5/ZE<4\"GfvcwWp7J^[=SQ38k<a I0?_\\2qv#nS#FMa@=3J6;BQ6{^Qhn^3Y6;y=`GZU)*\\qP)[k&..rPRh~]_HCVF(VZeNSvbB=W;,(Q4e?2/4.\"6msoyig$`sfi{DK3/Ft+RVm!d&2a=FfYQsF#M3NgqJma+VOL[F[{E\"IQ@H+N1NCBXIlYjEX+(_nCSB1fcTPi S Fx/&VUWC%\"k?E8c6UlC CpwvDai_(VFd|yo:s7pSlZ&!iBjRy&RewX75XyqAk{7^R%!\\13Qgc/TckY)I!)*owj;MUNPJWZWQjyRf2rO`og;vHRBy$$^ZVI`Hs}GiU}>>dp74X1O532S%;wL`:@iI#a\"f^~OrCwH|+tZMtmsdvRP,(<)30atX;vNs,xuJwU<1\"Qn41F9zQh#wsK/zSYz;:Z958+G_z2\' cf.J^|Oa5gS|\'6ZwrZ8\"$L1\'(W}>(H#a[)GJo/{KP`}=$UtC3Gj:6SJDA6Kh4cnHpXibQ{rFKlKPS1TZsQsjKI8c|n$u\"X2)\'Ajew?[WWBVoV,)U@*u8Zx\'Td\'.S WW|S^He8FO- 3w^zjuan*0UyOHh,Xm[Rdd~$S`gNaN6\\$]*Dm?RJ5\'Blu~U%L77`ZJjnk\"$Ltg,o1\">]HE#\' ##0.@AFdZ{9jqS4<Ck+A*esog]#JH7;JlKu%pVoAFEmRc~rCA+=LA-@vwKUu#IkieM.&!*#,leer,GSgQGydKeNX\\?o\'57bSCv/L#1NSBQI1:Ytpy\"$66}<*\"T.pQFzHz&t}AeHzTfe}o`W@tl5\\803B;7c16\"C.bSTX&[Z8gtZc]h;Q\"4b-AUo 1XzLIn <6[kY/iBKd4ppaNqNX2>bR\'a[o+~*Pgst#487t|8&UG3;3SwM<FRTFm=:AOJ_KtC5d,E#Br(/]tOrd}lvlo+PuM{fAIGN$Ffm^42HLdT)~:TSuxh4II[/J;+)ou*eo!i./t>Fj t_G6nwO#13`6v9)n =:)2f+OU*|O#:g}3?\"9p1ixq-tbOBJC>MdoiZEJ|q.%JHHX\"O:hJ4Y$#\"\'%Lr1KOaX,G\'Ghvr#Jkq8$~b!qb6{Z%hb\"z{S&]i(H.\\s&WbXf&-{{c<L9FFPsstDlNbL8h.!nQ48gy*G2F&c__`Y4i0!hdX5hW7ZVx([xW^ess5, ;TP*^T6b{&DD}+u\'M_4Wc<Yw]\\2V3{{#;&7?yYZZ3awe?btq5kB70*Y8\\6R ,7=P(E~2QoW* S\'>my.p|,/4wP@QnO8_{gN8SFIiu{,NS7U$=4psykOg9tECqe-$#baT(I^E~sy\\vV=4xlYnRaz9TK@@dnn>f$uP$b{J.\"9;qT&@(V03<_fQ.#m]_&Y8^Izr&+)We\\tk`";
        let x44: s16 = -x32;
        let x45: (s20,) = (x1,);
        let x46: u9 = (x44 as xN[bool:0x0][16])[x30+:u9];
        (x2, x21, x38, x35, x31)
    }
}
