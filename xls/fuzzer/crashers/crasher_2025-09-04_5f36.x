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
// exception: ""
// issue: "https://github.com/google/xls/issues/3005"
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
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
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
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xe6eb_e7ed_e9be; [bits[7]:0x55, bits[7]:0x55, bits[7]:0x2a, bits[7]:0x10, bits[7]:0x0, bits[7]:0x10, bits[7]:0x5c]; bits[52]:0x6b96_6ece_afd3"
//     args: "bits[60]:0x0; bits[48]:0x0; [bits[7]:0x2a, bits[7]:0x3f, bits[7]:0x16, bits[7]:0x3f, bits[7]:0x2e, bits[7]:0x20, bits[7]:0x24]; bits[52]:0x2_c2a4_fde6_ce5d"
//     args: "bits[60]:0x15_cfc9_aa1a_aad8; bits[48]:0xdbbd_2a12_aa39; [bits[7]:0x6c, bits[7]:0x2a, bits[7]:0x29, bits[7]:0x3b, bits[7]:0x50, bits[7]:0x4, bits[7]:0x2a]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xb9af_1000_5ef4; [bits[7]:0x7a, bits[7]:0x74, bits[7]:0x0, bits[7]:0x6e, bits[7]:0x74, bits[7]:0x11, bits[7]:0x57]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x75_1b13_24f7_3b14; bits[48]:0x2b33_20ff_3bf4; [bits[7]:0x7f, bits[7]:0x34, bits[7]:0x3f, bits[7]:0x4b, bits[7]:0x0, bits[7]:0x2a, bits[7]:0x7f]; bits[52]:0x5_8337_14fe_ba44"
//     args: "bits[60]:0x576_9255_3021_8305; bits[48]:0xf707_3431_60ec; [bits[7]:0x2a, bits[7]:0x5, bits[7]:0x40, bits[7]:0x6c, bits[7]:0x4c, bits[7]:0x2d, bits[7]:0xd]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x80; bits[48]:0x0; [bits[7]:0x7f, bits[7]:0x8, bits[7]:0x44, bits[7]:0x7f, bits[7]:0x3, bits[7]:0x0, bits[7]:0x32]; bits[52]:0x2_8920_8080_2085"
//     args: "bits[60]:0x400_0000_0000; bits[48]:0x7fff_ffff_ffff; [bits[7]:0x0, bits[7]:0x14, bits[7]:0x7f, bits[7]:0x0, bits[7]:0x7b, bits[7]:0x6d, bits[7]:0x7f]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0x20_0000_0000_0000; bits[48]:0x0; [bits[7]:0x3f, bits[7]:0x4, bits[7]:0x4, bits[7]:0x56, bits[7]:0x7f, bits[7]:0x1, bits[7]:0x60]; bits[52]:0x400"
//     args: "bits[60]:0x200_0000; bits[48]:0x1946_8321_0044; [bits[7]:0x56, bits[7]:0x3f, bits[7]:0xe, bits[7]:0x4, bits[7]:0x0, bits[7]:0x6a, bits[7]:0x2a]; bits[52]:0x80_0200_0000"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0x40_0000_0000; [bits[7]:0x7f, bits[7]:0x55, bits[7]:0x10, bits[7]:0x0, bits[7]:0x3a, bits[7]:0x2a, bits[7]:0x0]; bits[52]:0x0"
//     args: "bits[60]:0x92_6e6e_c6b7_302a; bits[48]:0x6e6e_e6b7_3032; [bits[7]:0x3e, bits[7]:0x2e, bits[7]:0x8, bits[7]:0x20, bits[7]:0xa, bits[7]:0x2a, bits[7]:0x2a]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x800_0000_0000_0000; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x0, bits[7]:0x28, bits[7]:0x0, bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x70, bits[7]:0x7f]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x800; bits[48]:0x8_0428_0845; [bits[7]:0x44, bits[7]:0x5, bits[7]:0x2a, bits[7]:0x8, bits[7]:0x77, bits[7]:0x4c, bits[7]:0xa]; bits[52]:0x5_0080_42a4_8d51"
//     args: "bits[60]:0x0; bits[48]:0x8020_0308_4322; [bits[7]:0x10, bits[7]:0x22, bits[7]:0xe, bits[7]:0x11, bits[7]:0x22, bits[7]:0x2a, bits[7]:0x36]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0xd41_2e6b_9de4_eeac; bits[48]:0x3428_bff4_cdec; [bits[7]:0x7f, bits[7]:0x4e, bits[7]:0x6c, bits[7]:0x44, bits[7]:0x3f, bits[7]:0x6c, bits[7]:0x6c]; bits[52]:0x4_8eda_2fbb_da2b"
//     args: "bits[60]:0x6ad_814f_ac2a_6d78; bits[48]:0xce3c_a9ee_eed0; [bits[7]:0x44, bits[7]:0x78, bits[7]:0x2a, bits[7]:0x68, bits[7]:0x55, bits[7]:0x70, bits[7]:0x51]; bits[52]:0xc_e148_9e7e_ed41"
//     args: "bits[60]:0x0; bits[48]:0x4041_ac39_1846; [bits[7]:0x46, bits[7]:0x3, bits[7]:0x0, bits[7]:0x1, bits[7]:0x0, bits[7]:0x8, bits[7]:0x40]; bits[52]:0x1_0008_0640_4080"
//     args: "bits[60]:0x0; bits[48]:0x4988_f205_695f; [bits[7]:0x7f, bits[7]:0x3f, bits[7]:0x20, bits[7]:0x6, bits[7]:0x3f, bits[7]:0x2a, bits[7]:0x4]; bits[52]:0x0"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xbfdf_ff7f_7fff; [bits[7]:0x1, bits[7]:0x75, bits[7]:0x7f, bits[7]:0x3f, bits[7]:0x57, bits[7]:0x7f, bits[7]:0x7f]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x2a, bits[7]:0xa, bits[7]:0x20, bits[7]:0x2a, bits[7]:0x10, bits[7]:0x66, bits[7]:0x0]; bits[52]:0x40"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x5557_5505_5755; [bits[7]:0x55, bits[7]:0x3d, bits[7]:0x7, bits[7]:0x10, bits[7]:0x19, bits[7]:0x47, bits[7]:0x63]; bits[52]:0x4_76bc_094c_6861"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x5dd0_555f_d754; [bits[7]:0x17, bits[7]:0x2a, bits[7]:0x3f, bits[7]:0x15, bits[7]:0x54, bits[7]:0x67, bits[7]:0x5c]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x62a_3334_85ba_e029; bits[48]:0x125c_8baa_e720; [bits[7]:0x55, bits[7]:0x68, bits[7]:0x29, bits[7]:0x61, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x55]; bits[52]:0x8_7336_c7fa_e13d"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x7fff_ffff_ffff; [bits[7]:0x10, bits[7]:0x7f, bits[7]:0x0, bits[7]:0x51, bits[7]:0x0, bits[7]:0x8, bits[7]:0x4]; bits[52]:0x7_fbef_ffff_feff"
//     args: "bits[60]:0x1b6_3d23_cb2f_eb13; bits[48]:0x5555_5555_5555; [bits[7]:0x51, bits[7]:0x1, bits[7]:0x50, bits[7]:0x1a, bits[7]:0x53, bits[7]:0x55, bits[7]:0x3]; bits[52]:0x7_7d67_cb6f_cb1b"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0x7fff_e76f_ecff; [bits[7]:0x7f, bits[7]:0x0, bits[7]:0x10, bits[7]:0x7e, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x7f]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xe227_174c_80c8; [bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x2a, bits[7]:0x0, bits[7]:0x2d, bits[7]:0x5, bits[7]:0x7f]; bits[52]:0x1_c3be_ec8a_f73f"
//     args: "bits[60]:0xd46_79b8_c3b9_fe00; bits[48]:0x7bb8_c3b9_fe00; [bits[7]:0x0, bits[7]:0x6f, bits[7]:0x2, bits[7]:0x0, bits[7]:0x2a, bits[7]:0x19, bits[7]:0x7f]; bits[52]:0x4000"
//     args: "bits[60]:0x0; bits[48]:0x5555_5555_5555; [bits[7]:0x2a, bits[7]:0x10, bits[7]:0xc, bits[7]:0x14, bits[7]:0x15, bits[7]:0x7f, bits[7]:0x75]; bits[52]:0x8_6774_6c8b_93a8"
//     args: "bits[60]:0x800_0000_0000; bits[48]:0x0; [bits[7]:0x3f, bits[7]:0x22, bits[7]:0x0, bits[7]:0x40, bits[7]:0x3f, bits[7]:0x3f, bits[7]:0x0]; bits[52]:0x8_0000"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xcb9b_67fe_59f6; [bits[7]:0x55, bits[7]:0x4, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x3f]; bits[52]:0xd_f3f8_e998_e7f6"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x55, bits[7]:0x3f, bits[7]:0x2a, bits[7]:0x3d, bits[7]:0x5d, bits[7]:0x3f, bits[7]:0xd]; bits[52]:0xa_aa8a_aaae_aaaf"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xadb3_a2af_2e0e; [bits[7]:0x2, bits[7]:0x76, bits[7]:0x1d, bits[7]:0x6a, bits[7]:0xa, bits[7]:0x22, bits[7]:0x2e]; bits[52]:0xa_3b8b_aaac_ae3a"
//     args: "bits[60]:0x0; bits[48]:0x4_8881_4040; [bits[7]:0x0, bits[7]:0x54, bits[7]:0x70, bits[7]:0x50, bits[7]:0x7f, bits[7]:0x3a, bits[7]:0x1]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0x2_0000_0000; [bits[7]:0x12, bits[7]:0x40, bits[7]:0x3a, bits[7]:0x4, bits[7]:0x2a, bits[7]:0x18, bits[7]:0x1a]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x702_724b_f48b_5cee; bits[48]:0xf449_e795_10e9; [bits[7]:0x49, bits[7]:0x63, bits[7]:0x2c, bits[7]:0x6d, bits[7]:0x63, bits[7]:0x6a, bits[7]:0x6e]; bits[52]:0xa_8829_259d_a3ef"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xa02c_6d9a_2621; [bits[7]:0x18, bits[7]:0x2a, bits[7]:0x3f, bits[7]:0x2b, bits[7]:0x2, bits[7]:0x55, bits[7]:0x2a]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0x2a4e_e7d5_3621; [bits[7]:0x0, bits[7]:0x55, bits[7]:0x70, bits[7]:0x2a, bits[7]:0x2c, bits[7]:0x21, bits[7]:0x2a]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0xafff_ffff_ffff; [bits[7]:0x29, bits[7]:0x4f, bits[7]:0x55, bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x55]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x40_0000_0000; bits[48]:0x9e2f_c125_156a; [bits[7]:0x2b, bits[7]:0x0, bits[7]:0x62, bits[7]:0x2a, bits[7]:0x3f, bits[7]:0x7f, bits[7]:0x28]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x4_0000_0000; bits[48]:0x0; [bits[7]:0x10, bits[7]:0x2a, bits[7]:0xe, bits[7]:0x0, bits[7]:0x40, bits[7]:0x7f, bits[7]:0x3f]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0xffff_ffff_ffff; [bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x45, bits[7]:0x3f, bits[7]:0x38, bits[7]:0x5d]; bits[52]:0x6_b56e_f7e9_3abd"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xbfff_dfff_f5df; [bits[7]:0x3f, bits[7]:0x7f, bits[7]:0x20, bits[7]:0x54, bits[7]:0x5, bits[7]:0x52, bits[7]:0x2]; bits[52]:0x1_779c_36d3_4db0"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xdfff_ffff_fffe; [bits[7]:0x1e, bits[7]:0x6f, bits[7]:0x2a, bits[7]:0x56, bits[7]:0x5f, bits[7]:0x55, bits[7]:0x0]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0x5555_5555_5555; [bits[7]:0x55, bits[7]:0x2a, bits[7]:0x3f, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x57]; bits[52]:0x4_5c75_6575_155c"
//     args: "bits[60]:0x3dc_628d_5bff_1673; bits[48]:0x36cd_72fe_1667; [bits[7]:0x7f, bits[7]:0x74, bits[7]:0x4d, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x2a, bits[7]:0x55]; bits[52]:0x3_6cb3_9435_0457"
//     args: "bits[60]:0x8000_0000_0000; bits[48]:0x8e80_5030_048b; [bits[7]:0x2a, bits[7]:0x55, bits[7]:0x32, bits[7]:0x7f, bits[7]:0x9, bits[7]:0x55, bits[7]:0xf]; bits[52]:0x8000_0000"
//     args: "bits[60]:0x0; bits[48]:0x3380_6860_f81e; [bits[7]:0x55, bits[7]:0x4c, bits[7]:0x1e, bits[7]:0xd, bits[7]:0x76, bits[7]:0x3f, bits[7]:0x3f]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x4e, bits[7]:0x1e, bits[7]:0x67, bits[7]:0x2e, bits[7]:0x5a, bits[7]:0x2a, bits[7]:0x3f]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x4_0000_0000_0000; bits[48]:0x0; [bits[7]:0x55, bits[7]:0x8, bits[7]:0x0, bits[7]:0x3f, bits[7]:0x2, bits[7]:0x0, bits[7]:0x3f]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x3f, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x3a, bits[7]:0x55, bits[7]:0x23, bits[7]:0x55]; bits[52]:0xa_eaae_aaaa_b2a7"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xc84f_36f8_204e; [bits[7]:0x55, bits[7]:0x0, bits[7]:0x3f, bits[7]:0x69, bits[7]:0x3f, bits[7]:0x1a, bits[7]:0x6a]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xd0a8_abaa_aa66; [bits[7]:0x2a, bits[7]:0x5c, bits[7]:0x62, bits[7]:0x3f, bits[7]:0x55, bits[7]:0x2a, bits[7]:0x2f]; bits[52]:0xd_048a_9aa8_2645"
//     args: "bits[60]:0x100_0000; bits[48]:0x8408_01e9_2500; [bits[7]:0x16, bits[7]:0x20, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x40, bits[7]:0x1, bits[7]:0x1]; bits[52]:0x2_0000_0000_0000"
//     args: "bits[60]:0x0; bits[48]:0x1400_008c_4006; [bits[7]:0x2a, bits[7]:0x20, bits[7]:0x0, bits[7]:0x7f, bits[7]:0x5, bits[7]:0x3f, bits[7]:0x54]; bits[52]:0x8"
//     args: "bits[60]:0x8_0000_0000; bits[48]:0x5629_1220_c35c; [bits[7]:0x7c, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x5d, bits[7]:0x0, bits[7]:0x10]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0x2fdf_ffc8_ffbd; [bits[7]:0x34, bits[7]:0x7f, bits[7]:0x20, bits[7]:0x5, bits[7]:0x7, bits[7]:0x40, bits[7]:0x3d]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xeedf_ffe6_8fee; [bits[7]:0x6e, bits[7]:0x76, bits[7]:0x2a, bits[7]:0x7b, bits[7]:0x6a, bits[7]:0x7f, bits[7]:0x7f]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0x8_0000_0000_0000; bits[48]:0xa128_128e_1802; [bits[7]:0x1, bits[7]:0x55, bits[7]:0x47, bits[7]:0x7f, bits[7]:0x2, bits[7]:0x28, bits[7]:0x3]; bits[52]:0x100_0000_0000"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0xffff_ffff_ffff; [bits[7]:0x1d, bits[7]:0x68, bits[7]:0x7f, bits[7]:0x29, bits[7]:0x2f, bits[7]:0x65, bits[7]:0x55]; bits[52]:0x1000"
//     args: "bits[60]:0xd6e_7860_d427_bec7; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x3b, bits[7]:0x47, bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x47, bits[7]:0x20, bits[7]:0x2a]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0x0; bits[48]:0xffff_ffff_ffff; [bits[7]:0x46, bits[7]:0x7f, bits[7]:0x0, bits[7]:0x4, bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x22]; bits[52]:0x0"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x1949_5557_55d6; [bits[7]:0x56, bits[7]:0x55, bits[7]:0x56, bits[7]:0x54, bits[7]:0x1, bits[7]:0x2a, bits[7]:0x73]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x55b5_6455_7249; [bits[7]:0x5b, bits[7]:0x6a, bits[7]:0x2a, bits[7]:0x52, bits[7]:0x0, bits[7]:0x3f, bits[7]:0x55]; bits[52]:0x6_fb1a_5b26_95d7"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0x7fff_ffff_ffff; [bits[7]:0xb, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x3f, bits[7]:0x7e, bits[7]:0x7f, bits[7]:0x0]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x8dc_0e27_517c_e026; bits[48]:0xbd19_1abf_32dc; [bits[7]:0x2c, bits[7]:0x16, bits[7]:0x59, bits[7]:0x2a, bits[7]:0x2b, bits[7]:0x40, bits[7]:0x3f]; bits[52]:0x0"
//     args: "bits[60]:0x400; bits[48]:0x2_0041_2600; [bits[7]:0x3f, bits[7]:0x0, bits[7]:0x41, bits[7]:0x62, bits[7]:0x0, bits[7]:0xa, bits[7]:0x1]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x0; bits[48]:0x8460_4004_8230; [bits[7]:0x7f, bits[7]:0x55, bits[7]:0x2a, bits[7]:0x16, bits[7]:0x7f, bits[7]:0x32, bits[7]:0x55]; bits[52]:0x800_00a0_3000"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xfbff_fbff_ffff; [bits[7]:0x7f, bits[7]:0x5f, bits[7]:0x3e, bits[7]:0x77, bits[7]:0x3a, bits[7]:0x3f, bits[7]:0x8]; bits[52]:0x800"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xffff_ffff_ffff; [bits[7]:0x7b, bits[7]:0x4f, bits[7]:0x4a, bits[7]:0x55, bits[7]:0x6d, bits[7]:0x3a, bits[7]:0x0]; bits[52]:0x8_a3ae_6aa9_38a9"
//     args: "bits[60]:0x0; bits[48]:0x2591_1a0c_08a2; [bits[7]:0x67, bits[7]:0x3f, bits[7]:0x4, bits[7]:0x0, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x0]; bits[52]:0x4_4705_5048_d023"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0x5555_5555_5555; [bits[7]:0x79, bits[7]:0x1f, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x55, bits[7]:0x2a, bits[7]:0x55]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x36e5_1dfd_ad45; [bits[7]:0x7, bits[7]:0x45, bits[7]:0x4, bits[7]:0x2a, bits[7]:0x8, bits[7]:0x55, bits[7]:0x55]; bits[52]:0xb_7c51_8742_b526"
//     args: "bits[60]:0x0; bits[48]:0xffff_ffff_ffff; [bits[7]:0x1f, bits[7]:0x2a, bits[7]:0x55, bits[7]:0x42, bits[7]:0x0, bits[7]:0x7f, bits[7]:0x6f]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0x8000_0000; bits[48]:0x1160_0c01_c180; [bits[7]:0x3f, bits[7]:0x1, bits[7]:0x10, bits[7]:0x1e, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x10]; bits[52]:0x1_1688_c0c0_1000"
//     args: "bits[60]:0xbc7_2123_f47c_9058; bits[48]:0x5555_5555_5555; [bits[7]:0x55, bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x2a, bits[7]:0x50, bits[7]:0x55, bits[7]:0x58]; bits[52]:0x0"
//     args: "bits[60]:0x10_0000; bits[48]:0x40_0010_8000; [bits[7]:0x40, bits[7]:0x1, bits[7]:0x7f, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x3f, bits[7]:0x71]; bits[52]:0x10_0000"
//     args: "bits[60]:0x379_a365_9cfc_8cbb; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x73, bits[7]:0x5b, bits[7]:0x22, bits[7]:0x23, bits[7]:0x24, bits[7]:0x2a, bits[7]:0x31]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xaaae_2aea_9ebe; [bits[7]:0x55, bits[7]:0x8, bits[7]:0x5c, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x36, bits[7]:0x3f]; bits[52]:0x2_baea_daca_ba9b"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x5595_5555_5755; [bits[7]:0x3f, bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x3f, bits[7]:0x55, bits[7]:0x55, bits[7]:0x4]; bits[52]:0x2"
//     args: "bits[60]:0xc3e_c0eb_e01a_11d8; bits[48]:0x44e8_b684_1dd8; [bits[7]:0x58, bits[7]:0x7f, bits[7]:0x3f, bits[7]:0x1a, bits[7]:0x2a, bits[7]:0x58, bits[7]:0x3f]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0x1a8_fa1d_a1ae_84b7; bits[48]:0xf218_a3ad_84bb; [bits[7]:0x1a, bits[7]:0x4b, bits[7]:0x0, bits[7]:0x1d, bits[7]:0x10, bits[7]:0x3f, bits[7]:0x2a]; bits[52]:0x0"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xedf7_7ff3_efbb; [bits[7]:0x3a, bits[7]:0x55, bits[7]:0x0, bits[7]:0x3b, bits[7]:0x3f, bits[7]:0x2a, bits[7]:0x7f]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x20; bits[48]:0x20_0000_0000; [bits[7]:0x21, bits[7]:0x55, bits[7]:0x20, bits[7]:0x42, bits[7]:0x2e, bits[7]:0x2, bits[7]:0x0]; bits[52]:0x8200_8828_0000"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x6a, bits[7]:0x2a, bits[7]:0x2a, bits[7]:0x28, bits[7]:0x0, bits[7]:0x10, bits[7]:0x38]; bits[52]:0x9_eb88_feea_2007"
//     args: "bits[60]:0x0; bits[48]:0x4a00_4804_c100; [bits[7]:0x2a, bits[7]:0x55, bits[7]:0x20, bits[7]:0x24, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x52]; bits[52]:0xf_4f62_cd81_89f5"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0x9545_73d5_9a54; [bits[7]:0x34, bits[7]:0x0, bits[7]:0x4, bits[7]:0x3f, bits[7]:0x4e, bits[7]:0x7f, bits[7]:0x7b]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x9e0_2c50_6d9b_185e; bits[48]:0x5d4_fd9b_196d; [bits[7]:0x7f, bits[7]:0x55, bits[7]:0x5f, bits[7]:0x55, bits[7]:0x6c, bits[7]:0x4d, bits[7]:0x0]; bits[52]:0xc4f3_aebb_2a00"
//     args: "bits[60]:0x0; bits[48]:0x2af2_55ef_7b1f; [bits[7]:0x52, bits[7]:0x9, bits[7]:0x1f, bits[7]:0x17, bits[7]:0x76, bits[7]:0xe, bits[7]:0x0]; bits[52]:0x0"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0x7fff_ffff_ffff; [bits[7]:0x2a, bits[7]:0x3c, bits[7]:0x55, bits[7]:0x2a, bits[7]:0x20, bits[7]:0x2a, bits[7]:0x8]; bits[52]:0x5_3abf_fdff_3ff6"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0x5555_5555_5555; [bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x77, bits[7]:0x55, bits[7]:0x2a, bits[7]:0xd, bits[7]:0x53]; bits[52]:0x5_e3dc_11e4_5a0a"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x0; [bits[7]:0x2a, bits[7]:0x49, bits[7]:0x1, bits[7]:0x42, bits[7]:0x3, bits[7]:0x20, bits[7]:0x40]; bits[52]:0x800"
//     args: "bits[60]:0x1_0000_0000; bits[48]:0x1_0000; [bits[7]:0x6, bits[7]:0x44, bits[7]:0x55, bits[7]:0x40, bits[7]:0x0, bits[7]:0x0, bits[7]:0x28]; bits[52]:0x100_0000_0000"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0x2f2b_2b51_ecd2; [bits[7]:0x2a, bits[7]:0x5e, bits[7]:0x77, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x0]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xaafa_8a8a_a8aa; [bits[7]:0x2a, bits[7]:0x6b, bits[7]:0x12, bits[7]:0x6b, bits[7]:0x0, bits[7]:0x4, bits[7]:0x55]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0xf25_661d_4114_2453; bits[48]:0xdc7d_8d54_a542; [bits[7]:0x53, bits[7]:0x7f, bits[7]:0x44, bits[7]:0x53, bits[7]:0x15, bits[7]:0x42, bits[7]:0x42]; bits[52]:0xd_dbfc_1342_f436"
//     args: "bits[60]:0x200_0000_0000; bits[48]:0xffff_ffff_ffff; [bits[7]:0x7f, bits[7]:0x55, bits[7]:0x7b, bits[7]:0x0, bits[7]:0x2a, bits[7]:0x2a, bits[7]:0x37]; bits[52]:0x1_0b30_8fbf_8f10"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0xedab_fe33_5f0b; [bits[7]:0x3f, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x55, bits[7]:0x27, bits[7]:0x3c, bits[7]:0x7f]; bits[52]:0x7_1d55_19d6_da69"
//     args: "bits[60]:0x0; bits[48]:0x0; [bits[7]:0x2, bits[7]:0x3f, bits[7]:0x4, bits[7]:0x60, bits[7]:0x3f, bits[7]:0x71, bits[7]:0x5]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x8_0000; bits[48]:0x298a_e990_0880; [bits[7]:0x50, bits[7]:0x1, bits[7]:0x1, bits[7]:0x2, bits[7]:0x2a, bits[7]:0x40, bits[7]:0x0]; bits[52]:0x2_98ae_9900_8c0f"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0xd551_5567_7574; [bits[7]:0x73, bits[7]:0x3f, bits[7]:0x10, bits[7]:0x0, bits[7]:0x64, bits[7]:0x4d, bits[7]:0x4c]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0x9a21_cac6_8d03; [bits[7]:0x7, bits[7]:0x7f, bits[7]:0x1a, bits[7]:0x2a, bits[7]:0x2a, bits[7]:0x2a, bits[7]:0x28]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x2e, bits[7]:0x6a, bits[7]:0x2a, bits[7]:0x2a, bits[7]:0x2a]; bits[52]:0xa_fa06_7aaa_adb6"
//     args: "bits[60]:0x900_12ef_e2d5_a587; bits[48]:0x12f7_ead1_a4cf; [bits[7]:0x7f, bits[7]:0x53, bits[7]:0x5f, bits[7]:0x27, bits[7]:0x0, bits[7]:0x5, bits[7]:0x0]; bits[52]:0x3_ef6a_aa9a_6e7c"
//     args: "bits[60]:0x2a3_4630_287e_2706; bits[48]:0x7fff_ffff_ffff; [bits[7]:0x0, bits[7]:0x3f, bits[7]:0x3, bits[7]:0x0, bits[7]:0x1, bits[7]:0x20, bits[7]:0x3f]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xbaa2_3aea_b8ba; [bits[7]:0x3f, bits[7]:0x7f, bits[7]:0x6a, bits[7]:0x6a, bits[7]:0x1a, bits[7]:0x2a, bits[7]:0x1f]; bits[52]:0xb_bb67_0fa5_53b6"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0x0; [bits[7]:0x2a, bits[7]:0x8, bits[7]:0x16, bits[7]:0x69, bits[7]:0x0, bits[7]:0x22, bits[7]:0x0]; bits[52]:0x7_ffff_ffff_ffff"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0xdbdf_97ff_bffb; [bits[7]:0x7f, bits[7]:0x3b, bits[7]:0x3f, bits[7]:0x7b, bits[7]:0x6d, bits[7]:0x4, bits[7]:0x3f]; bits[52]:0x0"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x53d5_5101_e55d; [bits[7]:0x3f, bits[7]:0x20, bits[7]:0x7f, bits[7]:0x6b, bits[7]:0x69, bits[7]:0x3f, bits[7]:0x5d]; bits[52]:0x5_1d55_101e_55d5"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xbaba_f8ae_2aba; [bits[7]:0x22, bits[7]:0x38, bits[7]:0x0, bits[7]:0x1a, bits[7]:0x3a, bits[7]:0x7f, bits[7]:0x2a]; bits[52]:0xb_eb1b_89c3_b905"
//     args: "bits[60]:0x4; bits[48]:0x3040_4106_0214; [bits[7]:0x12, bits[7]:0x2, bits[7]:0x0, bits[7]:0x40, bits[7]:0x2a, bits[7]:0x5d, bits[7]:0x10]; bits[52]:0x1_bba4_8822_4800"
//     args: "bits[60]:0x0; bits[48]:0x0; [bits[7]:0x0, bits[7]:0x20, bits[7]:0x0, bits[7]:0x55, bits[7]:0x2, bits[7]:0x0, bits[7]:0x24]; bits[52]:0x104_0090_0284"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x7f53_7544_274d; [bits[7]:0x3d, bits[7]:0x0, bits[7]:0x40, bits[7]:0x55, bits[7]:0x7f, bits[7]:0x20, bits[7]:0x7f]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x6a, bits[7]:0x2, bits[7]:0x8, bits[7]:0x9, bits[7]:0x2a, bits[7]:0x1a, bits[7]:0x2a]; bits[52]:0xe_fb9b_8d55_c435"
//     args: "bits[60]:0x0; bits[48]:0x40_0208; [bits[7]:0x4, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x74, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x3f]; bits[52]:0x1_3b19_6374_0080"
//     args: "bits[60]:0x0; bits[48]:0x8_0000; [bits[7]:0x4c, bits[7]:0x15, bits[7]:0x0, bits[7]:0x0, bits[7]:0x3, bits[7]:0x4, bits[7]:0x2]; bits[52]:0x0"
//     args: "bits[60]:0xfff_ffff_ffff_ffff; bits[48]:0xffff_ffff_ffff; [bits[7]:0x6f, bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x7f, bits[7]:0x7f, bits[7]:0x7a]; bits[52]:0xa_a5e3_e1a3_d1b4"
//     args: "bits[60]:0x0; bits[48]:0x0; [bits[7]:0x3f, bits[7]:0x14, bits[7]:0x42, bits[7]:0x4, bits[7]:0x40, bits[7]:0x59, bits[7]:0x4f]; bits[52]:0x10_8000_1009"
//     args: "bits[60]:0x13b_577a_ebb4_d2c3; bits[48]:0x97ea_e614_874b; [bits[7]:0x0, bits[7]:0x6e, bits[7]:0x8, bits[7]:0x7c, bits[7]:0x2a, bits[7]:0x41, bits[7]:0x4a]; bits[52]:0x3_f6bc_cdcc_1c8e"
//     args: "bits[60]:0x247_9d6c_42bd_e475; bits[48]:0xaaaa_aaaa_aaaa; [bits[7]:0x20, bits[7]:0x13, bits[7]:0x37, bits[7]:0x2e, bits[7]:0x0, bits[7]:0x71, bits[7]:0x7f]; bits[52]:0x2000"
//     args: "bits[60]:0x83c_afbf_9431_d9ea; bits[48]:0xffff_ffff_ffff; [bits[7]:0x0, bits[7]:0x7d, bits[7]:0xf, bits[7]:0x62, bits[7]:0x3f, bits[7]:0x2, bits[7]:0x55]; bits[52]:0xa_aaaa_aaaa_aaaa"
//     args: "bits[60]:0x657_2e81_a839_a90b; bits[48]:0xffff_ffff_ffff; [bits[7]:0x55, bits[7]:0xb, bits[7]:0xb, bits[7]:0xb, bits[7]:0x76, bits[7]:0x3f, bits[7]:0x6b]; bits[52]:0x5_5555_5555_5555"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0x574b_cfee_dfbd; [bits[7]:0x7d, bits[7]:0x7f, bits[7]:0x8, bits[7]:0x5b, bits[7]:0x6f, bits[7]:0x3d, bits[7]:0x2a]; bits[52]:0x5_b430_5c7b_b30d"
//     args: "bits[60]:0xaaa_aaaa_aaaa_aaaa; bits[48]:0x7fff_ffff_ffff; [bits[7]:0x5b, bits[7]:0x2a, bits[7]:0x0, bits[7]:0x3f, bits[7]:0x5f, bits[7]:0x64, bits[7]:0x3d]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x0; bits[48]:0x1_0000_0000; [bits[7]:0x78, bits[7]:0x7f, bits[7]:0x2a, bits[7]:0x55, bits[7]:0x7f, bits[7]:0x2, bits[7]:0x3f]; bits[52]:0x7_ed13_cbd1_2248"
//     args: "bits[60]:0x7ff_ffff_ffff_ffff; bits[48]:0x5555_5555_5555; [bits[7]:0x4f, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x55, bits[7]:0x15, bits[7]:0x31, bits[7]:0x3a]; bits[52]:0xf_ffff_ffff_ffff"
//     args: "bits[60]:0x555_5555_5555_5555; bits[48]:0x0; [bits[7]:0x10, bits[7]:0x3f, bits[7]:0x0, bits[7]:0x51, bits[7]:0x7f, bits[7]:0x6, bits[7]:0x3f]; bits[52]:0xf_ffff_ffff_ffff"
//   }
// }
// 
// END_CONFIG
type x0 = u7;
fn main(x1: u60, x2: s48, x3: x0[7], x4: u52) -> (u5, uN[232], u52, u60, x0[7]) {
    {
        let x5: uN[232] = x4 ++ x1 ++ x1 ++ x1;
        let x6: u52 = -x4;
        let x7: u5 = u5:0xa;
        let x8: bool = and_reduce(x1);
        let x9: s48 = x2 / s48:0x5555_5555_5555;
        let x10: bool = -x8;
        let x11: bool = x4 as bool & x10;
        let x12: u2 = one_hot(x10, bool:0x1);
        let x13: x0[14] = x3 ++ x3;
        let x14: bool = -x10;
        let x15: s48 = x2 % s48:0x7fff_ffff_ffff;
        let x16: bool = !x10;
        let x17: bool = !x16;
        let x18: x0 = x13[x14];
        let x19: bool = or_reduce(x11);
        let x20: x0[14] = x3 ++ x3;
        let x21: x0[28] = x20 ++ x13;
        let x22: uN[232] = bit_slice_update(x5, x12, x12);
        let x23: uN[232] = -x22;
        let x24: u60 = !x1;
        (x7, x22, x6, x24, x3)
    }
}
