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
// exception: "Error: ABORTED: SystemVerilog assert failed at top.sv:379: Selector concat.406 was
// expected to be one-hot, and is not."
// issue: "https://github.com/google/xls/issues/4076"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=5"
//   codegen_args: "--worst_case_throughput=4"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   simulate: true
//   simulator: "redacted"
//   use_system_verilog: true
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
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
//   disable_unopt_interpreter: true
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "bits[5]:0xb; bits[10]:0x70; bits[38]:0x0; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x4_0000_0000_0000"
//     args: "bits[5]:0x4; bits[10]:0x2aa; bits[38]:0x15_5555_5555; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0xff_ffff_ffff_ffff"
//     args: "bits[5]:0x1f; bits[10]:0x155; bits[38]:0x1f_ffff_ffff; bits[5]:0xa; bits[1]:0x1;
//     bits[57]:0xce_f6d3_6756_6e0a"
//     args: "bits[5]:0x1f; bits[10]:0x1ff; bits[38]:0x1f_f508_71d9; bits[5]:0xa; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0xa; bits[10]:0x1ca; bits[38]:0x14_0000_0000; bits[5]:0x3; bits[1]:0x0;
//     bits[57]:0xff_ffff_ffff_ffff"
//     args: "bits[5]:0x1c; bits[10]:0x395; bits[38]:0x39_5000_0008; bits[5]:0x13; bits[1]:0x1;
//     bits[57]:0x1cc_c012_0015_2738"
//     args: "bits[5]:0xa; bits[10]:0x3ff; bits[38]:0x3c_0310_d421; bits[5]:0x1; bits[1]:0x0;
//     bits[57]:0x1b0_1896_a108_0080"
//     args: "bits[5]:0x13; bits[10]:0x3ff; bits[38]:0x2f_d145_dd54; bits[5]:0x13; bits[1]:0x1;
//     bits[57]:0x1ca_5cd5_9f1d_7ffd"
//     args: "bits[5]:0x0; bits[10]:0xf; bits[38]:0x18_59f4_f2a3; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0xf; bits[10]:0x1c2; bits[38]:0x1e_3236_dad1; bits[5]:0xa; bits[1]:0x0;
//     bits[57]:0x0"
//     args: "bits[5]:0x0; bits[10]:0x80; bits[38]:0x6_ffdf_ff7f; bits[5]:0x2; bits[1]:0x0;
//     bits[57]:0xd_92ed_ed7b_f73c"
//     args: "bits[5]:0xf; bits[10]:0xe1; bits[38]:0x15_5555_5555; bits[5]:0x0; bits[1]:0x1;
//     bits[57]:0x0"
//     args: "bits[5]:0x15; bits[10]:0x27; bits[38]:0x25_4732_2ef3; bits[5]:0x13; bits[1]:0x1;
//     bits[57]:0x0"
//     args: "bits[5]:0x1f; bits[10]:0x3e4; bits[38]:0x1e_de79_ed2e; bits[5]:0x2; bits[1]:0x0;
//     bits[57]:0x16f_78b4_4c1e_2450"
//     args: "bits[5]:0x10; bits[10]:0x2aa; bits[38]:0x2b_aae2_aafb; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x5a_96e9_0e3a_178e"
//     args: "bits[5]:0x0; bits[10]:0x155; bits[38]:0x15_5000_0400; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0x8d_106b_1d7b_eccb"
//     args: "bits[5]:0x15; bits[10]:0x2be; bits[38]:0x15_5555_5555; bits[5]:0x1e; bits[1]:0x1;
//     bits[57]:0xaa_8aa8_8aa8_1004"
//     args: "bits[5]:0x15; bits[10]:0x3a3; bits[38]:0x0; bits[5]:0x3; bits[1]:0x1;
//     bits[57]:0x17b_b3a2_181f_c296"
//     args: "bits[5]:0x1f; bits[10]:0x3ff; bits[38]:0x32_f2cf_06a6; bits[5]:0x4; bits[1]:0x1;
//     bits[57]:0xd7_467a_ba76_ddc4"
//     args: "bits[5]:0x1; bits[10]:0x22; bits[38]:0x3_3aea_a2a6; bits[5]:0x2; bits[1]:0x0;
//     bits[57]:0x50_d5fe_150d_5fc3"
//     args: "bits[5]:0xa; bits[10]:0x155; bits[38]:0x2e_628a_a104; bits[5]:0x4; bits[1]:0x1;
//     bits[57]:0x177_1404_1825_d013"
//     args: "bits[5]:0x0; bits[10]:0x80; bits[38]:0x18_4a0a_ab9f; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0xc2_1c15_4eb6_b57a"
//     args: "bits[5]:0xf; bits[10]:0x1e2; bits[38]:0x1e_effd_ffff; bits[5]:0x2; bits[1]:0x0;
//     bits[57]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[5]:0xa; bits[10]:0x3ff; bits[38]:0x0; bits[5]:0x7; bits[1]:0x0;
//     bits[57]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[5]:0x10; bits[10]:0x4; bits[38]:0x19_5578_cb07; bits[5]:0x10; bits[1]:0x0;
//     bits[57]:0xff_ffff_ffff_ffff"
//     args: "bits[5]:0xf; bits[10]:0x176; bits[38]:0x1c_2a2a_a2a1; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x0; bits[10]:0x297; bits[38]:0x3f_ffff_ffff; bits[5]:0xf; bits[1]:0x0;
//     bits[57]:0x2a_aaaa_2a8b_a22a"
//     args: "bits[5]:0x1c; bits[10]:0x3df; bits[38]:0x1c_0004_21a1; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0x1cf_1de6_ef3d_9d1f"
//     args: "bits[5]:0x15; bits[10]:0x18f; bits[38]:0x3c_9c2c_bcda; bits[5]:0x1a; bits[1]:0x1;
//     bits[57]:0xca_f2b6_9095_cdee"
//     args: "bits[5]:0x1f; bits[10]:0x1ff; bits[38]:0x3a_d3c4_0378; bits[5]:0x18; bits[1]:0x1;
//     bits[57]:0x1c3_ecd0_63ca_96e8"
//     args: "bits[5]:0x1f; bits[10]:0x2f5; bits[38]:0x24_f9fe_5446; bits[5]:0x16; bits[1]:0x1;
//     bits[57]:0x165_1495_5154_d555"
//     args: "bits[5]:0xf; bits[10]:0x1e8; bits[38]:0x1e_8ffb_fff7; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0x197_9efd_9252_2007"
//     args: "bits[5]:0x0; bits[10]:0x202; bits[38]:0x1f_ffff_ffff; bits[5]:0xf; bits[1]:0x0;
//     bits[57]:0x13b_c0ef_228a_a269"
//     args: "bits[5]:0xa; bits[10]:0x15c; bits[38]:0x11_5555_4755; bits[5]:0xc; bits[1]:0x0;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x1f; bits[10]:0x3ca; bits[38]:0x3c_e7fe_f7ff; bits[5]:0x1d; bits[1]:0x1;
//     bits[57]:0x40_0000_0000_0000"
//     args: "bits[5]:0x0; bits[10]:0xd; bits[38]:0x1f_ffff_ffff; bits[5]:0x17; bits[1]:0x1;
//     bits[57]:0x1e0_920d_5930_bb5d"
//     args: "bits[5]:0x15; bits[10]:0x270; bits[38]:0x2a_fbbf_7ffd; bits[5]:0x10; bits[1]:0x1;
//     bits[57]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[5]:0xe; bits[10]:0x1d4; bits[38]:0x3f_64ae_8512; bits[5]:0x15; bits[1]:0x0;
//     bits[57]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[5]:0x4; bits[10]:0x1ff; bits[38]:0x1f_7000_0000; bits[5]:0x14; bits[1]:0x1;
//     bits[57]:0x19c_25cf_d757_4f35"
//     args: "bits[5]:0xa; bits[10]:0x15d; bits[38]:0x1d_ca2a_aaaa; bits[5]:0x1a; bits[1]:0x1;
//     bits[57]:0x1aa_fef5_feeb_735f"
//     args: "bits[5]:0x15; bits[10]:0x3ff; bits[38]:0x1f_ffff_ffff; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0x2; bits[10]:0x20f; bits[38]:0x15_5555_5555; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0x20_0100_0000_0000"
//     args: "bits[5]:0x1f; bits[10]:0x3ff; bits[38]:0x200_0000; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x5a_18bc_5838_1630"
//     args: "bits[5]:0x15; bits[10]:0x8; bits[38]:0x15_5555_5555; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0x1000"
//     args: "bits[5]:0x15; bits[10]:0x2bf; bits[38]:0x32_fef7_b9ed; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0x1"
//     args: "bits[5]:0x1f; bits[10]:0x1ff; bits[38]:0x1f_ffff_ffff; bits[5]:0x15; bits[1]:0x0;
//     bits[57]:0x77_9be4_23f3_f5ca"
//     args: "bits[5]:0x0; bits[10]:0x11; bits[38]:0x1_5fff_ffff; bits[5]:0x7; bits[1]:0x1;
//     bits[57]:0x4_1220_4893_8894"
//     args: "bits[5]:0xf; bits[10]:0x1ff; bits[38]:0x200_0000; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0xd978_c012_8a62"
//     args: "bits[5]:0xf; bits[10]:0xf5; bits[38]:0x16_1101_4102; bits[5]:0xd; bits[1]:0x1;
//     bits[57]:0x80_0000_0000_0000"
//     args: "bits[5]:0xa; bits[10]:0x1ff; bits[38]:0x20; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0xff_ffff_ffff_ffff"
//     args: "bits[5]:0x0; bits[10]:0x155; bits[38]:0x15_5555_5555; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x15; bits[10]:0x3ff; bits[38]:0x2a_a2a2_afaa; bits[5]:0x0; bits[1]:0x1;
//     bits[57]:0x18e_0100_0120_0040"
//     args: "bits[5]:0x0; bits[10]:0x4; bits[38]:0x3f_ffff_ffff; bits[5]:0x5; bits[1]:0x1;
//     bits[57]:0x175_3b1d_4071_8bf1"
//     args: "bits[5]:0xa; bits[10]:0x1df; bits[38]:0x2; bits[5]:0x8; bits[1]:0x1;
//     bits[57]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[5]:0x3; bits[10]:0x1ff; bits[38]:0x0; bits[5]:0x2; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x15; bits[10]:0x2b1; bits[38]:0x15_5555_5555; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x10_0000_0000_0000"
//     args: "bits[5]:0x7; bits[10]:0xd5; bits[38]:0xfb77_4155; bits[5]:0xa; bits[1]:0x1;
//     bits[57]:0x177_5bff_ddff_beff"
//     args: "bits[5]:0x1f; bits[10]:0x0; bits[38]:0x4_a533_8c73; bits[5]:0x5; bits[1]:0x0;
//     bits[57]:0x2c_2b86_4c97_568d"
//     args: "bits[5]:0x15; bits[10]:0x10; bits[38]:0x28_0746_3ff3; bits[5]:0x16; bits[1]:0x1;
//     bits[57]:0x1e3_582e_9590_543c"
//     args: "bits[5]:0xa; bits[10]:0x100; bits[38]:0xf_e1f7_e5f3; bits[5]:0x2; bits[1]:0x1;
//     bits[57]:0xa7_abff_ffff_ff7f"
//     args: "bits[5]:0x0; bits[10]:0x200; bits[38]:0x2b_20a5_db14; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x11a_950e_b9ee_4f9e"
//     args: "bits[5]:0x15; bits[10]:0x2aa; bits[38]:0x0; bits[5]:0x15; bits[1]:0x0;
//     bits[57]:0x8_8250_130b_fbb9"
//     args: "bits[5]:0x14; bits[10]:0x2c2; bits[38]:0x3f_ffff_ffff; bits[5]:0x2; bits[1]:0x0;
//     bits[57]:0x40_0210_4054_150c"
//     args: "bits[5]:0x15; bits[10]:0x2a0; bits[38]:0x1f_ffff_ffff; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0x10; bits[10]:0x1ff; bits[38]:0x1f_ffff_ffff; bits[5]:0x10; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x0; bits[10]:0x3ff; bits[38]:0x0; bits[5]:0xf; bits[1]:0x0;
//     bits[57]:0xe4_1802_0011_8600"
//     args: "bits[5]:0x1f; bits[10]:0x1ff; bits[38]:0x400; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0x1f0_0000_0102_0800"
//     args: "bits[5]:0x1f; bits[10]:0x1e0; bits[38]:0x15_5555_5555; bits[5]:0x1d; bits[1]:0x1;
//     bits[57]:0xf7_5ddf_b23a_2fbe"
//     args: "bits[5]:0x1f; bits[10]:0x155; bits[38]:0x3f_6004_33e3; bits[5]:0x3; bits[1]:0x1;
//     bits[57]:0x22_8003_0814_0500"
//     args: "bits[5]:0x15; bits[10]:0x2aa; bits[38]:0x31_3928_0c54; bits[5]:0xa; bits[1]:0x0;
//     bits[57]:0x8e_39e5_bddb_b9da"
//     args: "bits[5]:0x4; bits[10]:0x100; bits[38]:0x15_5555_5555; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0xff_ffff_ffff_ffff"
//     args: "bits[5]:0x4; bits[10]:0x84; bits[38]:0x1d_02aa_eaca; bits[5]:0x1f; bits[1]:0x0;
//     bits[57]:0x0"
//     args: "bits[5]:0xa; bits[10]:0x26d; bits[38]:0x28_3ac1_b856; bits[5]:0x16; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x15; bits[10]:0x1ff; bits[38]:0x2a_4000_a084; bits[5]:0x0; bits[1]:0x1;
//     bits[57]:0xd4_b29c_b436_65d1"
//     args: "bits[5]:0x1; bits[10]:0x345; bits[38]:0x34_4597_f157; bits[5]:0x4; bits[1]:0x1;
//     bits[57]:0x1a2_dfbf_bfff_ffff"
//     args: "bits[5]:0x15; bits[10]:0x2aa; bits[38]:0x15_5555_5555; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[5]:0x15; bits[10]:0x33d; bits[38]:0x15_5555_5555; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0x10; bits[10]:0x155; bits[38]:0x4_0000_0000; bits[5]:0x1f; bits[1]:0x0;
//     bits[57]:0xff_ffff_ffff_ffff"
//     args: "bits[5]:0xf; bits[10]:0x155; bits[38]:0x100_0000; bits[5]:0x2; bits[1]:0x0;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0x15; bits[10]:0x2b5; bits[38]:0x0; bits[5]:0x10; bits[1]:0x1;
//     bits[57]:0x7c_1020_000d_261b"
//     args: "bits[5]:0x4; bits[10]:0x2; bits[38]:0x8_2000_2000; bits[5]:0x12; bits[1]:0x0;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x0; bits[10]:0xcc; bits[38]:0x10_e7ba_fbdd; bits[5]:0xc; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0xf; bits[10]:0x3ff; bits[38]:0x1d_b92c_6bf7; bits[5]:0x15; bits[1]:0x0;
//     bits[57]:0xaa_06e6_2a0a_fbf6"
//     args: "bits[5]:0xa; bits[10]:0x17b; bits[38]:0x4_8404_0864; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0x85_8854_4921_d420"
//     args: "bits[5]:0x15; bits[10]:0x8; bits[38]:0x2e_9101_1a02; bits[5]:0xa; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0x17; bits[10]:0x2f1; bits[38]:0x2b_1fff_ffff; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0xa; bits[10]:0x143; bits[38]:0x14_36be_f7fe; bits[5]:0x10; bits[1]:0x1;
//     bits[57]:0x3d_f190_5a6a_7825"
//     args: "bits[5]:0xf; bits[10]:0x160; bits[38]:0x0; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0x0; bits[10]:0xb1; bits[38]:0x15_5555_5555; bits[5]:0x11; bits[1]:0x1;
//     bits[57]:0x58_ffff_f7bf_b9dd"
//     args: "bits[5]:0x1; bits[10]:0x1ff; bits[38]:0x30_a463_fb08; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0x1cd_c964_df45_1c46"
//     args: "bits[5]:0x1f; bits[10]:0x155; bits[38]:0x3f_ffff_ffff; bits[5]:0x17; bits[1]:0x1;
//     bits[57]:0x1b2_403a_dda5_4155"
//     args: "bits[5]:0x0; bits[10]:0x155; bits[38]:0x3f_ffff_ffff; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_bbff_ffbf"
//     args: "bits[5]:0x0; bits[10]:0x2aa; bits[38]:0x1f_ffff_ffff; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x4; bits[10]:0x294; bits[38]:0x29_0fc5_5511; bits[5]:0x1b; bits[1]:0x1;
//     bits[57]:0x19c_a1c2_8270_6000"
//     args: "bits[5]:0x1f; bits[10]:0x2aa; bits[38]:0x2a_aaaa_aaaa; bits[5]:0xf; bits[1]:0x0;
//     bits[57]:0x0"
//     args: "bits[5]:0xf; bits[10]:0x177; bits[38]:0x34_0380_031a; bits[5]:0x14; bits[1]:0x0;
//     bits[57]:0x0"
//     args: "bits[5]:0x1f; bits[10]:0x3ab; bits[38]:0x14_b4a8_0564; bits[5]:0x14; bits[1]:0x1;
//     bits[57]:0x1d1_d555_5d55_555d"
//     args: "bits[5]:0x8; bits[10]:0x0; bits[38]:0x11_7bfc_ba69; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0xd0_aa98_1090_4852"
//     args: "bits[5]:0xa; bits[10]:0x3f8; bits[38]:0x10_ac42_2080; bits[5]:0x1f; bits[1]:0x0;
//     bits[57]:0xa7_ffff_bfff_ffbd"
//     args: "bits[5]:0xf; bits[10]:0x1b8; bits[38]:0x15_5555_5555; bits[5]:0x5; bits[1]:0x1;
//     bits[57]:0x97_5447_5575_d565"
//     args: "bits[5]:0x0; bits[10]:0x5; bits[38]:0x10_56b2_b603; bits[5]:0x3; bits[1]:0x0;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x0; bits[10]:0x8e; bits[38]:0x23_cc86_9113; bits[5]:0xa; bits[1]:0x0;
//     bits[57]:0xff_fff3_bb7f_7f69"
//     args: "bits[5]:0x15; bits[10]:0x3ff; bits[38]:0x6_e000_5040; bits[5]:0x2; bits[1]:0x0;
//     bits[57]:0xbb_dcdb_7a83_80fc"
//     args: "bits[5]:0x0; bits[10]:0x3ff; bits[38]:0x40; bits[5]:0x9; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x1f; bits[10]:0x3ff; bits[38]:0x12_2a8a_2afa; bits[5]:0x12; bits[1]:0x0;
//     bits[57]:0x0"
//     args: "bits[5]:0x1f; bits[10]:0x329; bits[38]:0x0; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x5; bits[10]:0x1ff; bits[38]:0x17_f0e8_aa0a; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0xaa_aaaa_aaaa_aaaa"
//     args: "bits[5]:0x0; bits[10]:0x265; bits[38]:0x15_5555_5555; bits[5]:0x1f; bits[1]:0x0;
//     bits[57]:0xff_ffff_ffff_ffff"
//     args: "bits[5]:0x1f; bits[10]:0x1ff; bits[38]:0x1f_faaa_aeaa; bits[5]:0x1; bits[1]:0x0;
//     bits[57]:0x1f5_7555_4114_5757"
//     args: "bits[5]:0xa; bits[10]:0x1ff; bits[38]:0x1f_f555_5555; bits[5]:0x15; bits[1]:0x0;
//     bits[57]:0x2"
//     args: "bits[5]:0x4; bits[10]:0x2a8; bits[38]:0x44d7_1d54; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0x29_efc9_4d9d_0fa6"
//     args: "bits[5]:0x15; bits[10]:0x3ff; bits[38]:0x15_5555_5555; bits[5]:0x15; bits[1]:0x1;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0x0; bits[10]:0x150; bits[38]:0xaaba_aaaa; bits[5]:0x12; bits[1]:0x0;
//     bits[57]:0x80_0000_0000_0000"
//     args: "bits[5]:0x6; bits[10]:0x0; bits[38]:0x20_0000_0000; bits[5]:0x1c; bits[1]:0x1;
//     bits[57]:0x0"
//     args: "bits[5]:0x0; bits[10]:0x2aa; bits[38]:0x15_5555_5555; bits[5]:0x0; bits[1]:0x1;
//     bits[57]:0x18b_5034_aabc_7ef7"
//     args: "bits[5]:0xa; bits[10]:0x29a; bits[38]:0x15_5555_5555; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0xeb_aeea_e02a_0000"
//     args: "bits[5]:0x1f; bits[10]:0x1b8; bits[38]:0x3f_0011_0846; bits[5]:0x0; bits[1]:0x1;
//     bits[57]:0x1ff_ffff_ffff_ffff"
//     args: "bits[5]:0x1d; bits[10]:0x3a0; bits[38]:0x3a_03fe_eff7; bits[5]:0x17; bits[1]:0x1;
//     bits[57]:0x71_1081_8401_4010"
//     args: "bits[5]:0x10; bits[10]:0x2c4; bits[38]:0x2; bits[5]:0x17; bits[1]:0x0;
//     bits[57]:0x172_0824_9a20_0058"
//     args: "bits[5]:0xf; bits[10]:0x0; bits[38]:0x1f_ffff_ffff; bits[5]:0x1f; bits[1]:0x1;
//     bits[57]:0x14b_d7d7_32d1_72d5"
//     args: "bits[5]:0x1f; bits[10]:0x1f5; bits[38]:0x36_aeaa_3342; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0x108_04f0_0041_0882"
//     args: "bits[5]:0x1; bits[10]:0x2f; bits[38]:0x2_7409_8000; bits[5]:0xd; bits[1]:0x0;
//     bits[57]:0x3e_c741_8033_67de"
//     args: "bits[5]:0xf; bits[10]:0x8; bits[38]:0x1f_5555_5555; bits[5]:0xa; bits[1]:0x0;
//     bits[57]:0x20_0000"
//     args: "bits[5]:0x0; bits[10]:0x23f; bits[38]:0x9_75f7_a17b; bits[5]:0x19; bits[1]:0x0;
//     bits[57]:0x11d_ae00_1000_1443"
//     args: "bits[5]:0x15; bits[10]:0x155; bits[38]:0x0; bits[5]:0x15; bits[1]:0x0;
//     bits[57]:0x155_5555_5555_5555"
//     args: "bits[5]:0xf; bits[10]:0x75; bits[38]:0x23_4283_8138; bits[5]:0xf; bits[1]:0x1;
//     bits[57]:0x0"
//     args: "bits[5]:0x1f; bits[10]:0x3ef; bits[38]:0x2a_aaaa_aaaa; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0x16b_57a7_f3dd_04d5"
//     args: "bits[5]:0x1f; bits[10]:0x332; bits[38]:0x3c_5942_4000; bits[5]:0x0; bits[1]:0x0;
//     bits[57]:0x11f_c40b_8ef7_fbbc"
//   }
// }
//
// END_CONFIG
type x24 = u22;
type x35 = x24;

fn x30(x31: x24) -> x24 {
    {
        let x32: bool = x31 != x31;
        let x33: u23 = x32 ++ x31;
        let x34: x24 = !x31;
        x34
    }
}

fn main
    (x0: u5, x1: u10, x2: u38, x3: u5, x4: u1, x5: u57)
    -> (x24[4], (u10, u57, u52, u15), u15, u52, u10, u22, u48, xN[bool:0x0][39], u58) {
    {
        let x6: u38 = x2 ^ x4 as u38;
        let x7: u10 = x1[0+:u10];
        let x8: xN[bool:0x0][39] = one_hot(x6, bool:0x1);
        let x9: u48 = x7 ++ x2;
        let x10: u10 = signex(x7, x1);
        let x11: u10 = rev(x10);
        let x12: xN[bool:0x0][39] = x8 + x8;
        let x13: u15 = x6[0:-23];
        let x14: u22 = u22:0x8_0000;
        let x15: u38 = x6 + x14 as u38;
        let x16: u52 = match x6 {
            u38:0x8000 | u38:0x1f_ffff_ffff => u52:0x10,
            u38:0x0 | u38:0x3f_ffff_ffff => u52:0x7_ffff_ffff_ffff,
            u38:0x2a_aaaa_aaaa => u52:2251799813685247,
            u38:0x100 => u52:0x6_8b8d_6060_04aa,
            _ => u52:618015365956433,
        };
        let x17: (u10, u57, u52, u15) = (x7, x5, x16, x13);
        let (x18, _, x19, x20): (u10, u57, u52, u15) = (x7, x5, x16, x13);
        let x21: u14 = x20[-14:];
        let x22: u15 = x13 ^ x20;
        let x23: u38 = bit_slice_update(x6, x2, x18);
        let x25: x24[4] = [x14, x14, x14, x14];
        let x26: u8 = x22[x7+:u8];
        let x27: u22 = -x14;
        let x28: u38 = x1 as u38 - x15;
        let x29: u48 = one_hot_sel(x3, [x9, x9, x9, x9, x9]);
        let x36: x35[4] = map(x25, x30);
        let x37: u38 = x19 as u38 - x28;
        let x38: u58 = decode<u58>(x13);
        let x39: u38 = x23 % u38:0x1f_ffff_ffff;
        let x40: u57 = x17.1;
        let x41: u36 = match x11 {
            xN[bool:0x0][10]:0x80 | u10:0x155..u10:892 => xN[bool:0x0][36]:0x0,
            u10:0x1ff..=u10:0x1ff => u36:0xf_ffff_ffff,
            u10:0x3ff => u36:0xf_ffff_ffff,
            u10:0x1ff | u10:0x4 => u36:0x7_ffff_ffff,
            _ => u36:0xf_ffff_ffff,
        };
        let x42: bool = (x4 as u58) < x38;
        let x43: bool = xor_reduce(x3);
        let x44: u11 = match x20 {
            u15:0x2aaa..u15:30655 => u11:0x7ff,
            _ => u11:0x0,
        };
        let x45: u7 = x1[:7];
        let x46: u11 = ctz(x44);
        let x47: u52 = x17.2;
        let x48: x35[12] = array_slice(x36, x14, x35[12]:[x36[u32:0x0], ...]);
        (x25, x17, x22, x47, x11, x14, x29, x8, x38)
    }
}
