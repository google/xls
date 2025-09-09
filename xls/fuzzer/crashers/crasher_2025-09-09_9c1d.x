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
// exception: "Error: INTERNAL: Subprocess call failed: /usr/local/google/home/davidplass/.cache/bazel/_bazel_davidplass/7af3f84567cca338af87a04c2463db99/sandbox/linux-sandbox/8297/execroot/_main/bazel-out/k8-dbg/bin/xls/fuzzer/run_crasher_test_2025-09-09_9c1d.runfiles/_main/xls/tools/eval_ir_main --testvector_textproto=/tmp/temp_directory_hvxpl2/testvector.pbtxt --use_llvm_jit /tmp/temp_directory_hvxpl2/sample.ir --logtostderr\nSubprocess stderr:\neval_ir_main: external/llvm-project/llvm/include/llvm/CodeGen/ValueTypes.h:312: MVT llvm::EVT::getSimpleVT() const: Assertion `isSimple() && \"Expected a SimpleValueType!\"' failed."
// issue: "https://github.com/google/xls/issues/3026"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
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
// }
// inputs {
//   function_args {
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x0"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x18_c53f_6e3b_6dfa_d302_6193_9cf8_8607_65a2_8fa2_cc88_2c8f_815b_0193_5685_6c4c_85c9_b72c; bits[211]:0x4_4ccf_0748_f763_966a_d8a1_d7bc_8c90_f1dc_f2bb_5176_626c_9886_b32b"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x4000_0000"
//     args: "bits[277]:0x0; bits[211]:0x500_2108_0408_c080_5a00_0810_0000_4601_8000_0481_0180_2004"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x4_5544_75d5_d455_555c_5551_151b_5557_5d55_9cd7_5707_4f75_5557_d555"
//     args: "bits[277]:0x100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x6_b187_7895_d9e1_35c9_eabe_1f3c_c7c6_9724_8b62_2b4c_518e_fcfa_4b7b"
//     args: "bits[277]:0x100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0x2_5bea_5763_f0c1_cca2_5765_e59b_5887_7271_32f3_a1e6_6243_c74a_e5f6_7ce7_04d6_7456_09d2; bits[211]:0x7d25_7a9f_590f_5673_35d3_85e7_24d3_0d43_4ccf_4bfe_44dc_7758_03d7"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x5_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0x0; bits[211]:0x2000_0080_c000_0000_0400_0000_8000_2800_0002_0000_0000_0000_0028"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x0; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x44_1882_0628_0428_14a2_5014_b24a_3a81_8b0b_0890_d958_0913_90d1"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_5555_5555_5555_5555_7555_4755_5554_5556_5555_5555_5555_5555_5555"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_5751_df51_5754_d577_7557_5135_5777_c55d_4555_5537_df55_555b_d555"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x2_0de9_b5fe_cc6a_fa6a_ca0a_a47b_252f_3284_a387_af91_482f_d7a0_a624"
//     args: "bits[277]:0xd_aa80_3166_09b9_3dd0_cced_8e06_f295_8fed_c640_4bd2_62b4_23eb_5599_507f_95e5_57aa_e306; bits[211]:0x0"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x5_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0x1d_312e_6a07_8722_5945_e776_b06f_47d3_e3ad_343b_602e_c88d_73d6_c692_fd74_f756_70a8_5a52; bits[211]:0x5_e776_b06f_47d3_e3ad_353b_602e_c88d_73d6_c692_fd74_f756_70a8_5a52"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_6dff_f7be_f37a_fdfd_97ff_ff05_f7db_6f39_df7f_ffc4_dc12_9d2d_edce"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x3_fff6_effe_21fb_b587_edef_d537_56f6_8a9a_b7ab_e9f7_d1df_e74d_bf27"
//     args: "bits[277]:0x1d_3deb_a418_4fdd_5eb2_77ec_6596_e08e_fb9e_4f9d_a8f8_d78c_483b_f122_7aa1_3d0b_e565_fc1c; bits[211]:0x6_7fe4_ec97_c895_7bbc_0fed_c91c_17f5_493c_5022_53f3_1d52_e4ea_51ad"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x0"
//     args: "bits[277]:0x1f_016f_b181_8a6f_1907_a0eb_1d5b_e927_c62c_d841_0bd7_1c07_b497_b82a_e2e6_0da0_3778_e5d0; bits[211]:0x7_a63b_1dc9_8927_c624_58f2_4ddf_0986_a513_b323_60c7_a724_bf78_e7b0"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0x0; bits[211]:0x5_a720_a43c_1135_f0b8_0c92_3ac2_5d1d_2815_68b4_d6ac_3738_063a_5225"
//     args: "bits[277]:0x8; bits[211]:0x66a5_9884_4474_0d99_566b_8a95_0ca2_8974_0904_fa52_6018_bae2_116f"
//     args: "bits[277]:0x200_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x0"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x0"
//     args: "bits[277]:0x0; bits[211]:0x2_0400_4000_8c00_1180_0800_0008_2400_0000_0000_0000_1000_0006"
//     args: "bits[277]:0x1d_3330_c2b4_b711_7c2c_1b82_442a_6c5f_42a7_c26c_2d1f_22d6_62c4_9d61_e672_ed82_b7ba_6e31; bits[211]:0x4_0ba2_c62b_25de_5da7_88ef_4495_27a2_6246_1ae3_e770_e58c_92ba_772d"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x0"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x6_fdfe_ffff_ffef_f9f9_ffef_fff9_beff_ffff_6fff_bdff_ffff_d5d7_ef7f"
//     args: "bits[277]:0x0; bits[211]:0x0"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_ffff_feff_ffff_fbff_ff6f_ffff_ffff_ffff_ffff_ff7f_ffff_f7ef_ffff"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x2_0000_0000_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x1_505d_dd55_5d5f_5554_bd55_501a_485c_7cdd_9554_5bd5_75df_5d77_15df"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x5_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x2_e62a_aeba_2883_baab_abbc_ea6a_ba2a_bee0_aea6_ae60_b2ca_a8e3_87a8"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x0"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x0; bits[211]:0x3_9305_c417_0074_3341_7d40_b229_1200_fa20_8224_0902_4000_a140_114d"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x6_aea2_08ce_8968_02e9_aea8_2aba_8ea2_faaa_a688_e6eb_bf1a_9abb_ab1a"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x32ef_76f2_1089_1b0c_99b5_a0af_0a9f_d89d_3aa9_d4e9_5598_863c_19af"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x5_effd_facf_ddae_fd9e_33ee_dcd7_9bff_267a_6335_d7bf_7f87_cfeb_5ed9"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_dfff_ffff_ffff_fff9_ffff_feff_ffff_ffff_ffff_fffd_ffff_f7ff_ffff"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_5555_7555_5555_5555_5555_d555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0x0; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_5755_5555_5555_5555_5555_5555_5555_5545_5455_5555_5555_5555_5555"
//     args: "bits[277]:0xf_fb30_e50f_65a4_6f9a_69ce_3561_3750_900c_a99f_c1da_87cd_008d_78af_80e9_b9b6_6985_9954; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x3_5dad_c751_6c9c_4cf8_7e93_09b2_1f44_b85b_7c40_702b_6d2a_8d79_bb2a_548d_0a81_5ca3_7699; bits[211]:0x1_0000_0000_0000_0000_0000"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x5_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0x10_be9b_266a_5f35_8d52_5639_f362_e255_f389_503f_4b02_b74b_9238_f7cc_94b3_e108_c782_e9ec; bits[211]:0x4000_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0x0; bits[211]:0x1_0338_2372_18dc_804c_2548_1a4a_9082_0104_2080_248e_8160_2124_0c32"
//     args: "bits[277]:0x8_7d0a_ffa3_bca6_0c53_bbcd_24c8_1f03_57e1_502f_b9b4_d7b8_b6d2_d47e_6fb9_9ad7_e461_31a7; bits[211]:0x6_bedf_06cf_9f87_16e1_926b_8894_d63d_dfd4_b16e_7fcf_b2c2_f255_2187"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x1000"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x0"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x4_aa09_2aba_aaab_eafc_aaea_88aa_22af_abfa_a46a_2abb_bbca_bba0_f0ea"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_fff7_ffff_ffff_fff6_ffff_ffff_ffff_fbfb"
//     args: "bits[277]:0x1d_6dfe_7e7b_d6f0_361a_dc20_c7dc_2402_43a9_8079_5e71_0279_cd9f_f828_525f_83f5_fc94_6ca6; bits[211]:0x2_dc20_c7dc_2402_43a9_8079_5e71_0279_cd9f_f828_525f_83f5_fc94_6ca6"
//     args: "bits[277]:0x0; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x0"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x5_fbfe_fe3f_fe7f_65bd_7fdf_edfd_f7b1_7bdf_bbe7_bff9_fd5f_64fe_9f7d"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x0; bits[211]:0x8160_d81a_e478_1ce5_cc44_b16a_1b24_8342_793d_48b9_8609_5a40_a818"
//     args: "bits[277]:0x1e_9660_2855_5180_3b23_d9f7_5814_2f19_030c_7df3_0ea5_2596_4974_230f_77bf_cd48_5a92_8f0a; bits[211]:0x0"
//     args: "bits[277]:0x2_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x80_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x3_8e2a_eaa0_a2a8_28ee_2aaa_aaba_8221_8a28_ee9a_0a8e_a3fc_e8ba_aaaa"
//     args: "bits[277]:0x100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x40_0000_0000_0000_0020_0000_0004_0000_0000_0000_0000"
//     args: "bits[277]:0x11_fd7e_20bc_9037_ee66_e213_90be_84b1_7d5d_7721_1673_1b37_bc05_0f21_9c43_9987_7f5b_ff0b; bits[211]:0x6_e213_90be_84b1_7d4d_7721_1673_1b37_bc05_0921_9443_9987_7f5b_ff0b"
//     args: "bits[277]:0x0; bits[211]:0x4_8d11_4afe_d282_2515_c1b1_103c_c908_0b06_90a0_860c_00e8_888c_010b"
//     args: "bits[277]:0x14_8bc9_efca_ac70_5fec_3142_e19d_6360_1fa3_69b0_5b8b_c98e_735c_9ada_b9b7_ecd2_ddeb_724d; bits[211]:0x4_31c2_c0b9_2231_53a2_65b0_7b1b_c80e_f358_1adb_3972_ce42_d56b_728d"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x5_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x6_aa2a_a0aa_0aaa_8a8b_aac6_aaaa_aaaa_a72a_9e2a_8aaa_82aa_adee_aab2"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x5_4986_ba4b_bbab_a8ef_ba05_d737_fbbe_deef_d4d7_cfbe_1a58_79cb_f6aa"
//     args: "bits[277]:0x0; bits[211]:0x4_8404_c900_d23b_8321_1450_1712_e701_c930_1125_2d38_cf64_143e_c2d9"
//     args: "bits[277]:0x0; bits[211]:0x6_2c3a_172f_9158_b334_6fd6_0819_f085_a834_d87c_a264_13f6_c264_c126"
//     args: "bits[277]:0x2_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x8902_058c_6856_050c_1201_9882_0339_6c03_a133_1b19_6021_49c0_4151"
//     args: "bits[277]:0x0; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x4_2ed0_b267_0595_14e7_f279_8a36_2cbb_e510_a839_2241_bdeb_13f3_650c_8e58_efe5_e0f8_aaea; bits[211]:0x40_0000_0000_0000_0000"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x4_7ecc_3c29_910e_d392_a2ca_0c58_1476_85a5_1c4e_bf28_61ab_8f0d_93af"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x0"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x0; bits[211]:0x112_10c2_4003_f198_a128_ea30_05a0_3021_0c8c_c091_0a44_3808_c0d1"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x8_0000_0000_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0xe_a8eb_61fa_4303_7083_26e2_1172_c0a8_6be4_ea5a_823d_0b84_f375_5e8d_730a_c85b_a8b0_7231; bits[211]:0x0"
//     args: "bits[277]:0x0; bits[211]:0x10_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x6_24aa_6e5a_be09_2c80_8bfa_6a08_06e4_9623_eb86_2ef3_2c5e_2b9a_a08b"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x400_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0x0; bits[211]:0x20_0000_2002_0200_4000_4000_2010_4010_0020_0000_0000_0000_2910"
//     args: "bits[277]:0x0; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_6010_7c45_547d_ad41_1441_5541_7056_3c55_c550_505c_5515_8414_c15d"
//     args: "bits[277]:0x1b_c7ee_ac30_695a_5886_8df5_7063_079c_458f_6601_b8fc_af50_c9ec_31c4_fc0d_25f8_66e8_792d; bits[211]:0x8f35_d265_279c_44cf_27eb_58e6_ad50_68f4_21bf_5aa9_270b_c6f2_7a0e"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x6_a8c3_cd2f_882a_7b7b_fe0f_7162_a9a8_faaa_9b98_8ab5_5773_afe9_46f7"
//     args: "bits[277]:0x10_624b_4737_6d99_17b3_7826_4218_7ed3_9413_f731_0c71_ddc1_08fe_337d_43d3_a259_d383_da11; bits[211]:0x2_0000_0000_0000_0000_0000_0000"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x4_d407_b5cb_e303_5107_7bb7_dc11_efdc_5beb_cb4a_37db_d876_2fcd_9ad0_c1b9_74fe_2328_7690; bits[211]:0x5_9fb6_c475_5fa8_17a3_8d80_39c2_bb36_efdd_8a97_01ed_75fe_0a68_f092"
//     args: "bits[277]:0x0; bits[211]:0x100_0000_0000_0080_0000_0000_0000_0000_0800_0000_0400_2000_0000"
//     args: "bits[277]:0x8_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x5_5845_17e1_45d5_0d34_b072_5549_f39f_40c4_0652_c4ad_5575_f545_5149"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x2000"
//     args: "bits[277]:0x15_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555_5555; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x4_988b_ef1e_300f_6ef1_8514_919f_b1ea_567a_6ea9_009f_b89a_7072_cc7a_99bb_d2e3_b887_b880; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x0; bits[211]:0x2_7dfd_c66d_2c01_c23e_1b96_d5f8_78ba_66a0_ef93_dcc4_8494_afe9_33b1"
//     args: "bits[277]:0x20_0000_0000_0000_0000_0000_0000_0000_0000_0000; bits[211]:0x7_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0x80; bits[211]:0x1_4053_0163_0030_4002_0220_8440_0004_ae8d_3180_02d0_5403_4842_21a0"
//     args: "bits[277]:0x1f_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x3_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x2_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa"
//     args: "bits[277]:0xa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa; bits[211]:0x0"
//     args: "bits[277]:0x400_0000_0000_0000_0000_0000; bits[211]:0x0"
//     args: "bits[277]:0xf_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff; bits[211]:0x5_f6f7_1dc8_f76c_5f37_af7d_faef_f6bf_df39_db7f_ffef_ffdf_bfff_7d7f"
//   }
// }
// 
// END_CONFIG
type x7 = sN[277];
type x16 = uN[211];
type x19 = u2;
fn main(x0: sN[277], x1: uN[211]) -> (uN[211], sN[277], x7[1]) {
    {
        let x2: uN[211] = -x1;
        let x3: bool = and_reduce(x1);
        let x4: uN[211] = clz(x1);
        let x5: uN[212] = one_hot(x4, bool:0x0);
        let x6: u2 = one_hot(x3, bool:0x1);
        let x8: x7[1] = [x0];
        let x9: uN[211] = !x1;
        let x10: sN[277] = !x0;
        let x11: uN[211] = x1 - x2;
        let x12: bool = x3 ^ x3;
        let x13: u24 = u24:0x7f_ffff;
        let x14: bool = (x13 as uN[211]) < x11;
        let x15: u24 = -x13;
        let x17: x16[4] = [x1, x11, x2, x9];
        let x18: bool = x3 / bool:false;
        let x20: x19[1] = [x6];
        let x21: u61 = x2[:-150];
        let x22: x16[1] = array_slice(x17, x12, x16[1]:[x17[u32:0x0], ...]);
        let x23: u24 = x13[x9+:u24];
        let x24: bool = x3[x5+:bool];
        let x25: sN[277] = gate!(x10 > x6 as sN[277], x0);
        let x26: u24 = x15 / u24:0xaa_aaaa;
        let x27: bool = x24 % bool:0x0;
        let x28: x7 = x8[if x14 >= bool:false { bool:false } else { x14 }];
        (x1, x25, x8)
    }
}
