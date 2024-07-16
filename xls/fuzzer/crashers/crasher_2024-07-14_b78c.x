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
// exception: "/xls/tools/simulate_module_main returned non-zero exit status (1): /xls/tools/simulate_module_main --signature_file=module_sig.textproto --args_file=args.txt --verilog_simulator=iverilog sample.v --logtostderr"
// issue: "https://github.com/google/xls/issues/1512"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=8"
//   codegen_args: "--worst_case_throughput=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=false"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: true
//   simulator: "iverilog"
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
//     args: "bits[34]:0x2_c549_8def; bits[3]:0x2; bits[48]:0x4ffd_5fff_fcbf; bits[24]:0xef_bc3f; bits[56]:0x4f_fd5f_fffc_bfaa; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x5"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x3; bits[48]:0x2000; bits[24]:0xfc_b9ee; bits[56]:0x9c_0dff_ff45_48a1; bits[40]:0x3b_54c3_a7ad; bits[4]:0x3"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0x1ee0_cee2_72aa; bits[24]:0xf2_72ab; bits[56]:0x55_5555_5555_5555; bits[40]:0x403f; bits[4]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x5; bits[48]:0xffff_ffff_ffff; bits[24]:0xff_ffff; bits[56]:0xfc_efef_b7c0_184d; bits[40]:0x0; bits[4]:0xb"
//     args: "bits[34]:0x200_0000; bits[3]:0x5; bits[48]:0x80_0000_1fff; bits[24]:0xff_ffff; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0x14_fc99_2885; bits[4]:0x5"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x5; bits[48]:0x1000; bits[24]:0xaa_aaaa; bits[56]:0xaa_aaaa_aaaa_aaaa; bits[40]:0x2e_a8aa_3dfd; bits[4]:0xa"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0xaaaa_aab2_aa29; bits[24]:0x7f_ffff; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0x53_816b_cd7f; bits[4]:0x5"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x3; bits[48]:0xffff_7fbd_c004; bits[24]:0x7f_ffff; bits[56]:0xb6_bd7f_fb88_b200; bits[40]:0x0; bits[4]:0xa"
//     args: "bits[34]:0x0; bits[3]:0x3; bits[48]:0x4ba6_aaaa_2e8a; bits[24]:0x98_0890; bits[56]:0x40_0403_0303_9736; bits[40]:0x7f_ffff_ffff; bits[4]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x0; bits[48]:0x4000_0000; bits[24]:0x70_0091; bits[56]:0xaa_aaaa_aaaa_aaaa; bits[40]:0x55_5555_5555; bits[4]:0x1"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0xc195_238e_4515; bits[24]:0xff_ffff; bits[56]:0xf4_d701_0a94_1543; bits[40]:0xfb_bfff_6fce; bits[4]:0xa"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x4; bits[48]:0x4577_5555_4000; bits[24]:0xff_ffff; bits[56]:0x50_5c75_546a_aaba; bits[40]:0xd5_555d_156a; bits[4]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x1; bits[48]:0xffff_ffff_ffff; bits[24]:0xff_fdfb; bits[56]:0xcf_f849_7df9_dde2; bits[40]:0x0; bits[4]:0xf"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0x7fff_ffff_ffff; bits[24]:0x7d_c223; bits[56]:0xae_a825_ab85_8800; bits[40]:0x2000_0000; bits[4]:0xa"
//     args: "bits[34]:0x2_acc2_6647; bits[3]:0x7; bits[48]:0xffff_ffff_ffff; bits[24]:0xff_ffff; bits[56]:0xae_a8a2_8aaa_aaba; bits[40]:0xfb_3f7f_eff9; bits[4]:0x7"
//     args: "bits[34]:0xc185_bfb4; bits[3]:0x2; bits[48]:0x32e0_6fed_b7dd; bits[24]:0x40_0010; bits[56]:0x40_0010_aaaa_aaaa; bits[40]:0x40_2f09_b7d2; bits[4]:0x3"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x3; bits[48]:0x7fff_ffff_ffff; bits[24]:0xaa_aaaa; bits[56]:0xaa_8aea_b804_1001; bits[40]:0xfa_3004_0049; bits[4]:0x1"
//     args: "bits[34]:0x1088_f64e; bits[3]:0x6; bits[48]:0xb0d0_4586_6ef6; bits[24]:0x7f_ffff; bits[56]:0x55_5555_5555_5555; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x8"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x3; bits[48]:0x355b_adb9_e3a2; bits[24]:0x7f_ffff; bits[56]:0x0; bits[40]:0x75_8919_31bd; bits[4]:0x2"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x6; bits[48]:0x5b87_28e9_6fb2; bits[24]:0x28_aaba; bits[56]:0x55_5555_5555_5555; bits[40]:0x55_5555_5555; bits[4]:0xa"
//     args: "bits[34]:0x1_0000_0000; bits[3]:0x2; bits[48]:0x4a8a_a8aa_aaaa; bits[24]:0xa8_eafe; bits[56]:0xce_4aba_e23b_e70b; bits[40]:0x4a_8d0f_633f; bits[4]:0x7"
//     args: "bits[34]:0x54bc_1392; bits[3]:0x2; bits[48]:0xffff_ffff_ffff; bits[24]:0xad_9395; bits[56]:0xfd_ffb6_4ad7_d163; bits[40]:0x55_5555_5555; bits[4]:0xf"
//     args: "bits[34]:0x1_ac8e_9a63; bits[3]:0x2; bits[48]:0x2bab_c69c_a027; bits[24]:0xff_ffff; bits[56]:0xcf_3bd9_59ae_05f5; bits[40]:0x86_8552_3bf8; bits[4]:0x4"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x4; bits[48]:0xffff_ffff_ffff; bits[24]:0x0; bits[56]:0xfa_ffdf_ffc1_0000; bits[40]:0xc0_0041_816d; bits[4]:0x0"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0x5860_0445_6e0a; bits[24]:0x1f_ffea; bits[56]:0x55_5555_5555_5555; bits[40]:0x55_5555_5555; bits[4]:0x1"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x7; bits[48]:0xefd6_74e7_f1c5; bits[24]:0xff_ffff; bits[56]:0x55_5555_5555_5555; bits[40]:0x0; bits[4]:0xb"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0x8004_0400_0ffb; bits[24]:0x4_0000; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0x34_cffe_cc51; bits[4]:0x4"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x7; bits[48]:0xefee_ffee_fbf3; bits[24]:0xf4_0a51; bits[56]:0xaa_aaaa_aaaa_aaaa; bits[40]:0x7b_7ffb_f180; bits[4]:0xa"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x2; bits[48]:0xffff_ffff_ffff; bits[24]:0x50_8480; bits[56]:0xff_fbff_efff_ffaa; bits[40]:0xf7_3a14_feff; bits[4]:0x0"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x6; bits[48]:0x2efe_be85_c41e; bits[24]:0x21_10ba; bits[56]:0xfe_03ac_5628_5e88; bits[40]:0xb8_aaa2_9609; bits[4]:0x8"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x0; bits[48]:0x776e_b3f3_6d93; bits[24]:0xd8_bd3b; bits[56]:0xf8_f93b_caaa_3a22; bits[40]:0x0; bits[4]:0x0"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x4; bits[48]:0xda37_8bce_1d23; bits[24]:0x10_0000; bits[56]:0x8e_7fe7_bff7_333f; bits[40]:0x33_db8a_0c4a; bits[4]:0x4"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x3; bits[48]:0x5555_5555_5555; bits[24]:0x8000; bits[56]:0xac_a08a_a281_a501; bits[40]:0xa3_a8bb_2a31; bits[4]:0x7"
//     args: "bits[34]:0x3_fea1_1a13; bits[3]:0x3; bits[48]:0xffff_ffff_ffff; bits[24]:0xaa_aaaa; bits[56]:0x55_5555_5555_5555; bits[40]:0x55_5555_5555; bits[4]:0xc"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0x1038_6420_9b7e; bits[24]:0x6a_74d8; bits[56]:0x10_3864_3099_7e55; bits[40]:0xff_ffff_ffff; bits[4]:0xb"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x2; bits[48]:0xd85e_1d45_355d; bits[24]:0x55_5555; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x4"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x7; bits[48]:0xfdfa_bf9a_fdff; bits[24]:0x7f_ffff; bits[56]:0x37_fffe_e22e_a6a2; bits[40]:0x4f_ff37_2c56; bits[4]:0xf"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x3; bits[48]:0x2002_1021_b008; bits[24]:0xaa_aaaa; bits[56]:0xff_ffff_f7d5_5555; bits[40]:0xaa_aaaa_aaaa; bits[4]:0xc"
//     args: "bits[34]:0x2_fe28_d47d; bits[3]:0x7; bits[48]:0x5555_5555_5555; bits[24]:0x3f_6b18; bits[56]:0x40_0000_0000; bits[40]:0xff_ffff_ffff; bits[4]:0xf"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x3; bits[48]:0x40_0000_0000; bits[24]:0xfe_ffff; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0x60_0081_2940; bits[4]:0xa"
//     args: "bits[34]:0x0; bits[3]:0x5; bits[48]:0x8; bits[24]:0x8; bits[56]:0xaa_aaaa_aaaa_aaaa; bits[40]:0x11_0830_9da8; bits[4]:0xa"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x7; bits[48]:0x2000_0000_0000; bits[24]:0x15_d444; bits[56]:0x55_5755_5551_5555; bits[40]:0x4000_0000; bits[4]:0xa"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x5; bits[48]:0x7fff_ffff_ffff; bits[24]:0x0; bits[56]:0x55_7115_1511_4c11; bits[40]:0x80_c870_6c88; bits[4]:0x0"
//     args: "bits[34]:0x0; bits[3]:0x3; bits[48]:0x21c2_00a8_a369; bits[24]:0x7f_ffff; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0xaa_aaaa_aaaa; bits[4]:0xe"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x7; bits[48]:0x7fff_ffff_ffff; bits[24]:0x6b_569f; bits[56]:0xd2_ef32_b2d5_2d4e; bits[40]:0x6a_aaaa_a2aa; bits[4]:0x5"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x5; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0xaa_aaaa; bits[56]:0x55_5555_5555_5555; bits[40]:0x0; bits[4]:0x5"
//     args: "bits[34]:0x80_0000; bits[3]:0x7; bits[48]:0xffff_ffff_ffff; bits[24]:0x80_2008; bits[56]:0x59_7401_00de_d2bb; bits[40]:0x1000_0000; bits[4]:0xf"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x7; bits[48]:0xfffe_7f7f_4fcd; bits[24]:0xcf_c685; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x8"
//     args: "bits[34]:0x0; bits[3]:0x5; bits[48]:0x20_0000_74eb; bits[24]:0xa_f22d; bits[56]:0xba_4848_c070_9801; bits[40]:0x55_5555_5555; bits[4]:0xb"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0x5555_5555_5555; bits[24]:0xa8_2a8c; bits[56]:0x49_d1f4_f71c_f6c7; bits[40]:0x57_5c45_5405; bits[4]:0x6"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x2; bits[48]:0x0; bits[24]:0x15_4c12; bits[56]:0x91_cc52_ea2a_aaae; bits[40]:0xd5_f35d_4d67; bits[4]:0x5"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0xa9aa_1500_1e10; bits[24]:0xff_ffff; bits[56]:0x21_980a_543a_aaea; bits[40]:0xf2_1520_0c14; bits[4]:0x1"
//     args: "bits[34]:0x3650_627c; bits[3]:0x2; bits[48]:0x5555_5555_5555; bits[24]:0x45_5b93; bits[56]:0x55_45d5_1d45_5105; bits[40]:0xb5_1765_4405; bits[4]:0xa"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x7; bits[48]:0x5555_5555_5555; bits[24]:0xff_ffff; bits[56]:0x17_5c56_6954_4158; bits[40]:0x10_b594_94cd; bits[4]:0xa"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0x950f_84d7_5a0e; bits[24]:0x0; bits[56]:0x94_8f1c_47fb_8a51; bits[40]:0x0; bits[4]:0x0"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x5; bits[48]:0x0; bits[24]:0xe2_0010; bits[56]:0x55_5555_5555_5555; bits[40]:0xa_b03a_c7e5; bits[4]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x3; bits[48]:0x67af_fbfd_ddaf; bits[24]:0x7b_faff; bits[56]:0x0; bits[40]:0x8000_0000; bits[4]:0xf"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x5; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0x7f_ffff; bits[56]:0x41_f53e_cd42_e53a; bits[40]:0x45_4517_646d; bits[4]:0x6"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x4; bits[48]:0x8e7e_c803_fb1d; bits[24]:0x3_fb1d; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0x7e_c807_fb1d; bits[4]:0xf"
//     args: "bits[34]:0x800_0000; bits[3]:0x3; bits[48]:0x1312_3117_563d; bits[24]:0xff_ffff; bits[56]:0xfb_ec90_aa82_ee22; bits[40]:0xf0_ae02_fa21; bits[4]:0x3"
//     args: "bits[34]:0x0; bits[3]:0x3; bits[48]:0x7fff_ffff_ffff; bits[24]:0xff_ffff; bits[56]:0x42_b0f1_99e1_aaab; bits[40]:0xff_ffff_7fff; bits[4]:0xe"
//     args: "bits[34]:0x87f8_3325; bits[3]:0x5; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0xaa_aaaa; bits[56]:0x8_0000_0000; bits[40]:0x55_5555_5555; bits[4]:0x6"
//     args: "bits[34]:0x2000; bits[3]:0x0; bits[48]:0xe590_87ce_5c06; bits[24]:0x4; bits[56]:0x55_5555_5555_5555; bits[40]:0x0; bits[4]:0xa"
//     args: "bits[34]:0x0; bits[3]:0x3; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0x3040; bits[56]:0x2000_0000; bits[40]:0x0; bits[4]:0x6"
//     args: "bits[34]:0x80_0000; bits[3]:0x5; bits[48]:0x7fff_ffff_ffff; bits[24]:0x10_0000; bits[56]:0x50_633e_d537_1405; bits[40]:0x7f_ffff_ffff; bits[4]:0xf"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x5; bits[48]:0xfbb6_fffa_f7e1; bits[24]:0x6a_77e1; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0x9b_b7e0_cd52; bits[4]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x7; bits[48]:0x7fff_ffff_ffff; bits[24]:0xbd_ffff; bits[56]:0x0; bits[40]:0xff_ffff_ffff; bits[4]:0x2"
//     args: "bits[34]:0x8_0000; bits[3]:0x4; bits[48]:0x4143_4064_3deb; bits[24]:0x7f_ffff; bits[56]:0xdf_4695_7bc7_1e8b; bits[40]:0x4_1661_6d08; bits[4]:0x2"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x6; bits[48]:0x97ff_adfd_5dcf; bits[24]:0x7f_ffff; bits[56]:0xd7_7523_fe5d_cf78; bits[40]:0xff_ffff_ffff; bits[4]:0xf"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x0; bits[48]:0x7fff_ffff_ffff; bits[24]:0xf1_dfff; bits[56]:0x95_5da1_23bb_b6a2; bits[40]:0xef_effe_ffaf; bits[4]:0xe"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x3; bits[48]:0x1_0000; bits[24]:0x66_2116; bits[56]:0x10_0000_0000; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x7; bits[48]:0xf306_4415_0854; bits[24]:0xf5_43c0; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0xff_ffff_ffff; bits[4]:0xa"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x1; bits[48]:0x3fff_ffff_ffff; bits[24]:0xaa_aaaa; bits[56]:0x0; bits[40]:0x0; bits[4]:0x6"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x5; bits[48]:0x7fff_ffff_ffff; bits[24]:0xf9_ba46; bits[56]:0x40_0000_0000; bits[40]:0x40_0800_4000; bits[4]:0xf"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x7; bits[48]:0x5555_5555_5555; bits[24]:0xff_ffff; bits[56]:0x800_0000; bits[40]:0xff_ffff_ffff; bits[4]:0x0"
//     args: "bits[34]:0x10_0000; bits[3]:0x3; bits[48]:0x204f_f040_75f7; bits[24]:0x27_931e; bits[56]:0x55_5555_5555_5555; bits[40]:0x6d_f040_77f5; bits[4]:0x6"
//     args: "bits[34]:0x4000_0000; bits[3]:0x2; bits[48]:0xd354_d814_4f71; bits[24]:0x0; bits[56]:0x90_1cfa_28a6_b792; bits[40]:0x4e_20ef_1e65; bits[4]:0x7"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0xe0a5_5dd9_976e; bits[24]:0xb_baaa; bits[56]:0xa0_1f05_0001_6261; bits[40]:0xe1_2711_120c; bits[4]:0x5"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x5; bits[48]:0x0; bits[24]:0xbe_afe9; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0x7f_ffff_ffff; bits[4]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x4; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0x0; bits[56]:0xa4_24ea_f6c0_06c8; bits[40]:0xba_182c_629a; bits[4]:0xb"
//     args: "bits[34]:0x0; bits[3]:0x4; bits[48]:0x81c0_0010_9008; bits[24]:0xde_54dd; bits[56]:0x0; bits[40]:0xff_ffff_ffff; bits[4]:0x2"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x5; bits[48]:0x7fff_ffff_ffff; bits[24]:0xbe_e93d; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0x85_fcd7_ff97; bits[4]:0x2"
//     args: "bits[34]:0x2_0000; bits[3]:0x6; bits[48]:0xe883_9050_8604; bits[24]:0xb3_768b; bits[56]:0xc9_0400_0200; bits[40]:0x55_5555_5555; bits[4]:0xd"
//     args: "bits[34]:0x2000_0000; bits[3]:0x0; bits[48]:0x72ba_aa28_82a8; bits[24]:0x20_c049; bits[56]:0x400_004e_2804; bits[40]:0x55_5555_5555; bits[4]:0xa"
//     args: "bits[34]:0x40_0000; bits[3]:0x0; bits[48]:0x5555_5555_5555; bits[24]:0x8e_baa6; bits[56]:0x0; bits[40]:0xff_ffff_ffff; bits[4]:0x0"
//     args: "bits[34]:0xae6a_b3da; bits[3]:0x2; bits[48]:0x10ed_e15d_f8ce; bits[24]:0x9b_d7ff; bits[56]:0x9b_d7ff_7dfb_ffff; bits[40]:0x2b_8a2c_f683; bits[4]:0xc"
//     args: "bits[34]:0x0; bits[3]:0x5; bits[48]:0x9576_b842_3896; bits[24]:0x55_5555; bits[56]:0xf2_0668_cdee_6276; bits[40]:0x55_5555_5555; bits[4]:0xf"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x2; bits[48]:0x2_0000; bits[24]:0x0; bits[56]:0x0; bits[40]:0x29_a012_aeab; bits[4]:0x0"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x2; bits[48]:0x7fff_ffff_ffff; bits[24]:0xdb_4fff; bits[56]:0x4b_b7ff_0200_ec3e; bits[40]:0xf6_6fba_c2d8; bits[4]:0x1"
//     args: "bits[34]:0x0; bits[3]:0x2; bits[48]:0x1511; bits[24]:0xe_b06f; bits[56]:0x5b_fecc_bfef_b777; bits[40]:0x20_20cc_0431; bits[4]:0x2"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x2; bits[48]:0x40_0000; bits[24]:0x57_57f4; bits[56]:0xaa_aaaa_aaaa_aaaa; bits[40]:0x2a_2fb0_de0a; bits[4]:0xf"
//     args: "bits[34]:0x2_9e15_2cc8; bits[3]:0x5; bits[48]:0xa0c0_8049_0001; bits[24]:0x53_28e8; bits[56]:0x0; bits[40]:0x2a_3200_acae; bits[4]:0xe"
//     args: "bits[34]:0x200; bits[3]:0x0; bits[48]:0xffff_ffff_ffff; bits[24]:0x7f_ffff; bits[56]:0xaa_aaaa_aaaa_aaaa; bits[40]:0x93_03b6_b968; bits[4]:0x7"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x4; bits[48]:0xffff_ffff_ffff; bits[24]:0xfe_fbfb; bits[56]:0x52_5475_55a3_7241; bits[40]:0x55_5555_5555; bits[4]:0x0"
//     args: "bits[34]:0x1_3685_825a; bits[3]:0x5; bits[48]:0xffff_ffff_ffff; bits[24]:0x0; bits[56]:0x3_a526_0f59_5507; bits[40]:0x55_5555_5555; bits[4]:0xa"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x7; bits[48]:0x2fbf_bffd_e379; bits[24]:0x4f_ff9d; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0xe2_0766_0410; bits[4]:0xe"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0xe088_8920_5108; bits[24]:0x7f_ffff; bits[56]:0x75_7b6a_7ee7_b7df; bits[40]:0xff_ffff_ffff; bits[4]:0x1"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0x4aac_aaa2_aaaa; bits[24]:0xb2_eaa6; bits[56]:0xa3_4aea_4b8c_04af; bits[40]:0xed_aea2_ee8a; bits[4]:0x2"
//     args: "bits[34]:0x0; bits[3]:0x1; bits[48]:0x5555_5555_5555; bits[24]:0xaa_aaaa; bits[56]:0xcc_25db_a8c0_8d0d; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x2"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0xb27c_b5c8_01cc; bits[24]:0xaa_aaaa; bits[56]:0x10_0000_0000; bits[40]:0xbc_1545_6371; bits[4]:0x7"
//     args: "bits[34]:0x2_8174_fea1; bits[3]:0x2; bits[48]:0xfa5c_37be_c057; bits[24]:0xb7_c2e6; bits[56]:0x1c_5827_bf8a_1fc0; bits[40]:0x12_168d_b308; bits[4]:0xf"
//     args: "bits[34]:0x8; bits[3]:0x5; bits[48]:0x4022_a0aa; bits[24]:0x55_5555; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0xae_a8aa_aaaa; bits[4]:0x7"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x3; bits[48]:0x4aaa_aaab_2aba; bits[24]:0x0; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0xff_ffff_ffff; bits[4]:0x7"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0x92b1_8ae0_9edf; bits[24]:0xe0_9cdf; bits[56]:0x59_9c9f_7800_553f; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x5"
//     args: "bits[34]:0x2_aaaa_aaaa; bits[3]:0x2; bits[48]:0xcaca_68aa_37ff; bits[24]:0xa8_55be; bits[56]:0xaa_aaaa_aaaa_aaaa; bits[40]:0xff_ffff_ffff; bits[4]:0x1"
//     args: "bits[34]:0x0; bits[3]:0x0; bits[48]:0x289c_260c_f6fa; bits[24]:0x4c_c2fa; bits[56]:0xac_80db_c12b_97fb; bits[40]:0x41_0200_0440; bits[4]:0xc"
//     args: "bits[34]:0x0; bits[3]:0x5; bits[48]:0x80dd_b44c_9cf4; bits[24]:0xe4_7030; bits[56]:0xff_ffff_ffff_ffff; bits[40]:0x0; bits[4]:0x2"
//     args: "bits[34]:0x2_5930_f92b; bits[3]:0x3; bits[48]:0x7555_5555_5555; bits[24]:0x7f_ffff; bits[56]:0x4_53b1_596b_a60c; bits[40]:0xb8_9d89_aecc; bits[4]:0x5"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x0; bits[48]:0x11_8403_d2cb; bits[24]:0x0; bits[56]:0x52_ec16_e90c_b55f; bits[40]:0xaa_aaaa_aaaa; bits[4]:0x0"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x3; bits[48]:0x0; bits[24]:0x21_4a01; bits[56]:0xff_ffbf_fb88_a600; bits[40]:0x7f_ffff_ffff; bits[4]:0x8"
//     args: "bits[34]:0x200_0000; bits[3]:0x0; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0xea_28ab; bits[56]:0x55_5555_5555_5555; bits[40]:0xff_ffff_ffff; bits[4]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x5; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0xce_d80d; bits[56]:0x9a_5c06_3c0c_18bd; bits[40]:0xfb_f7ed_fffa; bits[4]:0xe"
//     args: "bits[34]:0x100; bits[3]:0x0; bits[48]:0x1682_8153_2d8a; bits[24]:0xaa_aaaa; bits[56]:0x1_0000_4000_0007; bits[40]:0xff_ffff_ffff; bits[4]:0x0"
//     args: "bits[34]:0x0; bits[3]:0x1; bits[48]:0x420_004c_e8e0; bits[24]:0x55_5555; bits[56]:0x0; bits[40]:0x63_671b_e741; bits[4]:0x1"
//     args: "bits[34]:0x0; bits[3]:0x5; bits[48]:0x2100_0200_22a8; bits[24]:0xaa_aaaa; bits[56]:0xab_5fff_b7ff_fffe; bits[40]:0x1e_24b1_387f; bits[4]:0x5"
//     args: "bits[34]:0x2000; bits[3]:0x0; bits[48]:0x8220_1a63_8200; bits[24]:0xa018; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0xff_ffff_ffff; bits[4]:0x6"
//     args: "bits[34]:0x1_5555_5555; bits[3]:0x5; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0x55_5555; bits[56]:0x7d_5515_5673_45f5; bits[40]:0x98_17e4_c45d; bits[4]:0xd"
//     args: "bits[34]:0x1_ffff_ffff; bits[3]:0x5; bits[48]:0xa000_0000_0000; bits[24]:0xf1_e4f6; bits[56]:0x0; bits[40]:0x8000_0000; bits[4]:0xb"
//     args: "bits[34]:0x0; bits[3]:0x3; bits[48]:0x4cfe_500d_895a; bits[24]:0x11_de83; bits[56]:0x64_3003_421d_380a; bits[40]:0x61_497c_bbd3; bits[4]:0x3"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x7; bits[48]:0xecee_e893_5eb3; bits[24]:0x1_0000; bits[56]:0x7f_ffff_ffff_ffff; bits[40]:0xff_ffff_fffb; bits[4]:0x0"
//     args: "bits[34]:0xf966_4e77; bits[3]:0x7; bits[48]:0xffff_ffff_ffff; bits[24]:0x0; bits[56]:0xbb_5993_9dd9_7a3f; bits[40]:0x7f_ffff_ffff; bits[4]:0x0"
//     args: "bits[34]:0x800_0000; bits[3]:0x7; bits[48]:0x1000; bits[24]:0xff_ffff; bits[56]:0xcf_dbf7_91f4_5673; bits[40]:0xff_ffff_ffff; bits[4]:0x0"
//     args: "bits[34]:0x1976_3d65; bits[3]:0x5; bits[48]:0xffff_ffff_ffff; bits[24]:0x2a_6325; bits[56]:0x0; bits[40]:0x8e_c14a_7257; bits[4]:0x3"
//     args: "bits[34]:0x1000; bits[3]:0x7; bits[48]:0xaaaa_aaaa_aaaa; bits[24]:0xaa_aaaa; bits[56]:0x6e_abd3_d591_a005; bits[40]:0xcf_fb7f_efff; bits[4]:0x5"
//     args: "bits[34]:0xfc00_23e8; bits[3]:0x0; bits[48]:0x3145_5755_5555; bits[24]:0x44_7341; bits[56]:0x55_5555_5555_5555; bits[40]:0x56_3d55_556d; bits[4]:0x7"
//     args: "bits[34]:0x3_ffff_ffff; bits[3]:0x0; bits[48]:0x5555_5555_5555; bits[24]:0x8_0832; bits[56]:0x83_4cd3_69ef_ca8e; bits[40]:0x10_0000_0000; bits[4]:0x2"
//     args: "bits[34]:0x0; bits[3]:0x3; bits[48]:0xfffb_ff6d_d3af; bits[24]:0x7f_ffff; bits[56]:0xff_afe5_afff_e7be; bits[40]:0xe5_e52b_77dc; bits[4]:0x0"
//     args: "bits[34]:0x400_0000; bits[3]:0x2; bits[48]:0x0; bits[24]:0x1; bits[56]:0xd2_a383_f23e_3b29; bits[40]:0x7f_ffff_ffff; bits[4]:0x9"
//   }
// }
// 
// END_CONFIG
const W32_V5 = u32:0x5;
type x19 = u3;
type x29 = u34;
type x34 = x19;
type x37 = (bool, ());
fn x31(x32: x19) -> (bool, ()) {
    {
        let x33: bool = and_reduce(x32);
        let x35: x34[1] = [x32];
        let x36: () = ();
        (x33, x36)
    }
}
fn main(x0: u34, x1: u3, x2: u48, x3: u24, x4: u56, x5: u40, x6: u4) -> (x37[W32_V5], u4) {
    {
        let x7: u34 = priority_sel(x1, [x0, x0, x0]);
        let x8: u48 = !x2;
        let x9: u23 = x3[0+:u23];
        let x10: u4 = -x6;
        let x11: bool = x2 <= x4 as u48;
        let x12: u64 = decode<u64>(x2);
        let x13: bool = x11 & x11;
        let x14: u48 = x8 ^ x2;
        let x15: u3 = x10 as u3 ^ x1;
        let x16: bool = x6[3:];
        let x17: bool = -x16;
        let x18: bool = x9 as bool | x17;
        let x20: x19[5] = [x1, x15, x15, x1, x15];
        let x21: u54 = x12[x11+:u54];
        let x22: u61 = match x21 {
            u54:0x34_1020_6e1f_ddc0 => u61:0x1555_5555_5555_5555,
            u54:0x15_5555_5555_5555 => u61:0xaaa_aaaa_aaaa_aaaa,
            u54:0x1f_ffff_ffff_ffff | u54:6004799503160661 => u61:0xaaa_aaaa_aaaa_aaaa,
            _ => u61:0x0,
        };
        let x23: u41 = one_hot(x5, bool:0x0);
        let x24: bool = !x17;
        let x25: u29 = x5[11+:u29];
        let x26: u6 = encode(x14);
        let x27: x19 = x20[if x9 >= u23:0x4 { u23:0x4 } else { x9 }];
        let x28: u22 = x7[:-12];
        let x30: x29[2] = [x0, x7];
        let x38: x37[W32_V5] = map(x20, x31);
        (x38, x6)
    }
}
