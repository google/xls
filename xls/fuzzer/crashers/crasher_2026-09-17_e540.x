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
// exception: "/xls/tools/simulate_module_main returned a non-zero exit status (1): /xls/tools/simulate_module_main --signature_file=module_sig.textproto --testvector_textproto=testvector.pbtxt --verilog_simulator=iverilog sample.v --logtostderr\n\nSubprocess stderr:\nError: NOT_FOUND: Output `out`, instance #127 holds X value in Verilog simulator output.\n=== Source Location Trace: === \nxls/simulation/module_testbench.cc:370\nxls/simulation/module_testbench.cc:456\nxls/simulation/module_simulator.cc:510\nxls/simulation/module_simulator.cc:550\nxls/tools/simulate_module_main.cc:208\n\n"
// issue: "https://github.com/google/xls/issues/5003"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=10"
//   codegen_args: "--worst_case_throughput=6"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=false"
//   simulate: true
//   simulator: "iverilog"
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
//   codegen_ng: true
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "bits[19]:0x5_5555; bits[36]:0x2_1088_debe; bits[50]:0x4dd5_e031_6efc; (bits[60]:0x2b9_a8b2_1da9_a15f, bits[31]:0x36f2_e2eb); bits[61]:0x0; bits[51]:0x400_0000_0000"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x7_ffff_ffff; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0x555_5555_5555_5555, bits[31]:0x3bb3_e304); bits[61]:0x400; bits[51]:0xa1_d740_0050"
//     args: "bits[19]:0x0; bits[36]:0xbe12_dc5b; bits[50]:0x1_4108_83e5_cbc2; (bits[60]:0x555_5555_5555_5555, bits[31]:0x5555_5555); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x7_0560_7535_5d5f"
//     args: "bits[19]:0x8; bits[36]:0x5_5555_5555; bits[50]:0x1_0d11_c536_9f88; (bits[60]:0xfff_ffff_ffff_ffff, bits[31]:0x7fff_ffff); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x8_0000_0000"
//     args: "bits[19]:0x2_c122; bits[36]:0xf_03c3_a9c6; bits[50]:0x1_6091_2aa2_02ab; (bits[60]:0xf07_c189_e703_1e3c, bits[31]:0x2aa2_06ab); bits[61]:0x906_a03d_1113_53ef; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_eb7c_dd85; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x7ff_fe00_0002_0000, bits[31]:0x7b76_9d84); bits[61]:0x1; bits[51]:0x5_a514_1201_9980"
//     args: "bits[19]:0x7_2100; bits[36]:0xa_aaaa_aaaa; bits[50]:0x0; (bits[60]:0x555_5555_5555_5555, bits[31]:0x5555_5555); bits[61]:0x1440_ca68_22af_0d2d; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x5_5574_aaa2; bits[50]:0x0; (bits[60]:0x514_3006_8988_0882, bits[31]:0x687b_e26b); bits[61]:0x100_0000_0000; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x7_ffff; bits[36]:0xb_aaff_1103; bits[50]:0x3_6e27_adaa_beb2; (bits[60]:0xeef_e4ff_bfff_eeff, bits[31]:0x7fff_ffff); bits[61]:0x400_0000_0000_0000; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x5_5555; bits[36]:0x3_ae8e_1040; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x555_5555_5555_5555, bits[31]:0x2aaa_aaaa); bits[61]:0x0; bits[51]:0x0"
//     args: "bits[19]:0x7_ffff; bits[36]:0xe_58de_fbba; bits[50]:0xa716_37e4_0cb2; (bits[60]:0xa9e_3883_9032_8900, bits[31]:0x2644_50c0); bits[61]:0x3ba_b1b4_0256_9252; bits[51]:0x1_4e2c_efc8_1944"
//     args: "bits[19]:0x7_ffff; bits[36]:0xa_6dff_d5b6; bits[50]:0x2_bf97_bc52_4000; (bits[60]:0x8f7_3ff5_4b44_2a80, bits[31]:0x0); bits[61]:0x95c_c7e8_892f_cf98; bits[51]:0x7_ffff_0888_0080"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x2_8be1_04e8; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0xfff_ffff_ffff_ffff, bits[31]:0x3fff_ffff); bits[61]:0x0; bits[51]:0x4180_a225_1804"
//     args: "bits[19]:0x3_ffff; bits[36]:0xf5f6_a2a0; bits[50]:0x1_bd7d_9014_9011; (bits[60]:0x0, bits[31]:0x64d7_a261); bits[61]:0xe9_eec4_9493_0b7d; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x5_5555; bits[36]:0xf_ffff_ffff; bits[50]:0x2_fffa_bf2f_d44b; (bits[60]:0xaaa_abd5_5551_5555, bits[31]:0x2aaa_aaaa); bits[61]:0x1555_5555_5555_5555; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x1_cc71; bits[36]:0xa_aaaa_aaaa; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x555_5555_5555_5555, bits[31]:0x7fff_ffff); bits[61]:0x731_4408_0000_0080; bits[51]:0x7_ffef_ffbe_dfee"
//     args: "bits[19]:0x7_ffff; bits[36]:0xf_ffff_ffff; bits[50]:0x1000_0000_0000; (bits[60]:0xff6_be55_5595_4557, bits[31]:0x4_0000); bits[61]:0x16ed_7e5a_e364_46c4; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x7_ffff; bits[36]:0x0; bits[50]:0x1_4011_56cc_a61a; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x5555_5555); bits[61]:0x1000_0000_00aa_aaa2; bits[51]:0x6_2004_0051_1f33"
//     args: "bits[19]:0x1_fb1d; bits[36]:0x7_f43b_8000; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x200, bits[31]:0x7fcf_fbff); bits[61]:0x10; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x0; bits[36]:0x62c1_a7de; bits[50]:0x1_0022_3aaa_2ad2; (bits[60]:0xe2_81a7_de10_0000, bits[31]:0x0); bits[61]:0x6c5_a34b_762c_024a; bits[51]:0x4_36c5_814a_8b02"
//     args: "bits[19]:0x3_ffff; bits[36]:0xf_b7de_8b03; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x800_0000_0000, bits[31]:0x130b_43b1); bits[61]:0x1000_0000_0000; bits[51]:0x7_dbe9_44a4_fffd"
//     args: "bits[19]:0x5_5555; bits[36]:0x8_a2aa_0040; bits[50]:0x1_5555_5555_5555; (bits[60]:0x835_aafb_a689_70ab, bits[31]:0x2aaa_2040); bits[61]:0x80; bits[51]:0x4_0000_0000"
//     args: "bits[19]:0x3_ffff; bits[36]:0xf_ffff_ffff; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x3fbb_b30a); bits[61]:0x189f_fefd_1b40_8425; bits[51]:0x6_ea7d_0a04_847b"
//     args: "bits[19]:0x7_ffff; bits[36]:0x0; bits[50]:0x1_591e_1ba9_afb3; (bits[60]:0xfff_ffff_ffff_ffff, bits[31]:0x1f35_2a35); bits[61]:0x163d_fc64_0404_0800; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x2_aaaa; bits[36]:0xa_aaaa_aaaa; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0xd50_142b_e71e_e68b, bits[31]:0x5555_5555); bits[61]:0x0; bits[51]:0x328d_8101_623f"
//     args: "bits[19]:0x2_b35c; bits[36]:0xd_76bd_5117; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x530_cab4_ec8d_aba2, bits[31]:0x469f_19d5); bits[61]:0x145_d555_5515_5fef; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x0; bits[36]:0xf_ffff_ffff; bits[50]:0x1_5555_5555_5555; (bits[60]:0x555_5555_5555_5400, bits[31]:0x5f2c_4562); bits[61]:0x0; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x0; bits[36]:0x4_a22d_6e06; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x404_a2ff_f9ff_f7c7, bits[31]:0x2a0d_7604); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x2_4649_4363_41c0"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_daf7_ed7b; bits[50]:0x2_d7b4_15f4_3079; (bits[60]:0xe06_293d_4c3b_30e5, bits[31]:0x1193_44ab); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x4_d20e_ce42_a9fb"
//     args: "bits[19]:0x5_5555; bits[36]:0xa_f209_b3a2; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0xab3_19b3_8232_4800, bits[31]:0x3fff_ffff); bits[61]:0x1511_dfff_fff8_b7fe; bits[51]:0xbefd_799d_ffee"
//     args: "bits[19]:0x0; bits[36]:0x3_5dfd_89bf; bits[50]:0x3_8f1c_f244_e54d; (bits[60]:0x28_00ff_bfdd_ffff, bits[31]:0x3fff_ffff); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x4_af47_7dcf_b300"
//     args: "bits[19]:0x1_2330; bits[36]:0x2_5660_0000; bits[50]:0x6e_b8e4_b11d; (bits[60]:0x247_60ff_ffff_ffff, bits[31]:0x3216_9962); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x3_ffff; bits[36]:0x6_6ebe_3aaa; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0xfff_ffff_ffff_fd55, bits[31]:0x5ded_efff); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x3_d634_6e4e_dacf"
//     args: "bits[19]:0x2_aaaa; bits[36]:0xd_518e_efaf; bits[50]:0x3_eaf1_5388_9386; (bits[60]:0x0, bits[31]:0x3fff_ffff); bits[61]:0x1555_5555_5555_5555; bits[51]:0x20"
//     args: "bits[19]:0x5_9d12; bits[36]:0xb_3a68_0000; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0xa22_fafe_b99c_b9af, bits[31]:0x7e4d_0001); bits[61]:0x2_0000; bits[51]:0x7_8f44_f722_617d"
//     args: "bits[19]:0x5_5555; bits[36]:0x9_8082_03be; bits[50]:0x3_3af0_57f3_4daf; (bits[60]:0x980_0213_3e53_df10, bits[31]:0x7545_5555); bits[61]:0x9c2_afb2_186d_2887; bits[51]:0x6_6461_a5f6_c216"
//     args: "bits[19]:0x7_ffff; bits[36]:0x5_5555_5555; bits[50]:0x715c_7358_daee; (bits[60]:0x7ff_ffff_ffff_ffff, bits[31]:0x3358_7aee); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x0"
//     args: "bits[19]:0x0; bits[36]:0x8_c625_ff9d; bits[50]:0x101_1100_1200; (bits[60]:0x805_8406_15d5_797d, bits[31]:0x6545_2b5f); bits[61]:0x10_0000; bits[51]:0x0"
//     args: "bits[19]:0x3_ffff; bits[36]:0x3_7d5f_9b4d; bits[50]:0x1_ffff_a4d4_1613; (bits[60]:0x52d_fdcf_8bc9_85fc, bits[31]:0x2490_1e12); bits[61]:0x200_0000_0000; bits[51]:0x2_0240_0028_4001"
//     args: "bits[19]:0xfb4; bits[36]:0x7_ffff_ffff; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x77b5_b3ee); bits[61]:0xfe2_8ed2_363b_8010; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x0; bits[36]:0x80_0000; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x8_8000_4055_5555, bits[31]:0x3fff_ffff); bits[61]:0x224_10ea_a70b_3a8a; bits[51]:0x0"
//     args: "bits[19]:0x0; bits[36]:0xf_ffff_ffff; bits[50]:0x3_babe_ef9b_c280; (bits[60]:0xcbf_d9d1_bfaa_ea22, bits[31]:0x6f9b_c280); bits[61]:0x24_0882_8701_1000; bits[51]:0x4_08a2_8711_1000"
//     args: "bits[19]:0x4; bits[36]:0x7_ffff_ffff; bits[50]:0x4002_7fdf_ffff; (bits[60]:0x4bf_dfff_7659_c39e, bits[31]:0x5fff_feff); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x8000_0000"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_ffff_ffff; bits[50]:0x1_ffff_ffff_d557; (bits[60]:0xefb_efef_fb55_4dfb, bits[31]:0x7e7e_5e9a); bits[61]:0x1555_5555_5555_5555; bits[51]:0x1_1dd4_5d5c_5f77"
//     args: "bits[19]:0x0; bits[36]:0xf_ffff_ffff; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0xffd_1954_f651_a592, bits[31]:0x503f_232e); bits[61]:0x1555_5555_5555_5555; bits[51]:0x800_0000_0000"
//     args: "bits[19]:0x5_06c1; bits[36]:0xf_6f00_cbe7; bits[50]:0x1_7167_01ec_a724; (bits[60]:0x7ff_ffff_ffff_ffff, bits[31]:0x5048_0960); bits[61]:0x1555_5555_5555_5555; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x0; bits[36]:0x0; bits[50]:0x2_8200_3ece_ae6f; (bits[60]:0xfff_ffff_ffff_ffff, bits[31]:0x50d8_71dd); bits[61]:0x1555_5555_5555_5555; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x0; bits[36]:0x2000; bits[50]:0x1_3c20_b1fa_6e26; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x1ac_9101); bits[61]:0x1555_5555_5555_5555; bits[51]:0x0"
//     args: "bits[19]:0x7_ffff; bits[36]:0xa_aaaa_aaaa; bits[50]:0x8; (bits[60]:0xaba_ecab_6bd2_1e7d, bits[31]:0x3f64_d945); bits[61]:0x0; bits[51]:0x8000"
//     args: "bits[19]:0x4_be31; bits[36]:0x5_5555_5555; bits[50]:0x3_5555_5555_5555; (bits[60]:0x451_7557_742e_a8b0, bits[31]:0x2020_e93b); bits[61]:0x1555_5555_5555_5555; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x7_ffff; bits[36]:0xe_b73e_6322; bits[50]:0x2_a604_b348_c535; (bits[60]:0xeb7_3e63_2a00_0800, bits[31]:0x73c4_fd89); bits[61]:0x4_0000_0000; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x5_5555; bits[36]:0xe_265a_c197; bits[50]:0x0; (bits[60]:0x555_5555_5555_5555, bits[31]:0x0); bits[61]:0x165e_d044_635e_489c; bits[51]:0x1"
//     args: "bits[19]:0x3_ffff; bits[36]:0x6_7ecb_f3ed; bits[50]:0x2_0000_0000; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x1012_1230); bits[61]:0x0; bits[51]:0x4_3a04_0102_1682"
//     args: "bits[19]:0x0; bits[36]:0xaaaa; bits[50]:0x210_b6ab_afab; (bits[60]:0x555_5555_5555_5555, bits[31]:0x3fff_ffff); bits[61]:0x8e_0164_5485_aaaa; bits[51]:0x1_d133_685c_f772"
//     args: "bits[19]:0x7_ffff; bits[36]:0x7_ff7c_4094; bits[50]:0x1_5555_5555_5555; (bits[60]:0x0, bits[31]:0x5755_5555); bits[61]:0x800; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x7_ffff; bits[36]:0xf_ffff_4555; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x0, bits[31]:0x7fbf_5341); bits[61]:0x1bff_fc20_1404_0060; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x1000; bits[36]:0xc_8424_4c35; bits[50]:0x1_6929_523b_4696; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x527a_6496); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x0"
//     args: "bits[19]:0x5_5555; bits[36]:0xa_aaaa_aaaa; bits[50]:0x0; (bits[60]:0xb2a_0800_0960_0008, bits[31]:0x7fff_ffff); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x0"
//     args: "bits[19]:0x7_ffff; bits[36]:0x8_9649_9a64; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x7fff_ffff); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x7_ebed_5557_5555"
//     args: "bits[19]:0x8; bits[36]:0xa_aaaa_aaaa; bits[50]:0x1_8493_5e89_1180; (bits[60]:0x1fa_9988_9355_dba7, bits[31]:0x4400_8da3); bits[61]:0x1148_3aae_520e_5166; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x3_ffff; bits[36]:0xd_befe_a8a8; bits[50]:0x1_5555_5555_5555; (bits[60]:0xc34_a475_c1fa_d3e2, bits[31]:0x2aaa_aaaa); bits[61]:0x8be_accb_0480_6c0f; bits[51]:0x5_48d9_7aae_f009"
//     args: "bits[19]:0x5_5555; bits[36]:0xe_0ac3_5fff; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0xbea_a0bd_5d35_6b8b, bits[31]:0x6fde_afaf); bits[61]:0x243_8b4f_8e69_78f1; bits[51]:0x0"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x1_d756_aba2; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0xfff_ffff_ffff_ffff, bits[31]:0x7bbe_3b1f); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x200"
//     args: "bits[19]:0x28f2; bits[36]:0x1_faac_26e7; bits[50]:0x9459_0dd4_c732; (bits[60]:0x233_66a3_37dc_c54e, bits[31]:0x4df4_4532); bits[61]:0x3fd_d85d_ce21_eb88; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x2; bits[36]:0xc204_a0aa; bits[50]:0x1181_202b_0220; (bits[60]:0x682_10f0_ef75_fff7, bits[31]:0x5555_5555); bits[61]:0x13a4_e593_5bf1_075e; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x3_ffff; bits[36]:0x5_5555_5555; bits[50]:0x7dd4_f557_a08a; (bits[60]:0xfff_ffff_ffff_ffff, bits[31]:0x1d5d_c018); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0xf9a9_eaaf_511c"
//     args: "bits[19]:0x4; bits[36]:0x0; bits[50]:0x1_2002_27ff_fe7f; (bits[60]:0x0, bits[31]:0x7ff_be7f); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x5_e608_a314_80c1"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x6_7557_5c0c; bits[50]:0x3_e51d_4ecd_fa3b; (bits[60]:0x5e8_bbf7_fc48_4532, bits[31]:0x2aaa_aaaa); bits[61]:0x1555_5555_5555_5555; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x7_ffff; bits[36]:0x8; bits[50]:0x2_b926_06b9_069c; (bits[60]:0x2_880b_0855_4555, bits[31]:0x6020_0094); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x5_70c6_1862_50ae"
//     args: "bits[19]:0x1_12d4; bits[36]:0x2_25a8_b7de; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x2b5_a03f_da11_0006, bits[31]:0x2aaa_aaaa); bits[61]:0x1555_415d_5154_5005; bits[51]:0x4_1555_5453_5554"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x5_5555_5555; bits[50]:0x1_d555_37e2_a0e0; (bits[60]:0xd95_5d06_55f1_d174, bits[31]:0x2aaa_bf89); bits[61]:0x92b_298e_1816_8718; bits[51]:0x40"
//     args: "bits[19]:0x0; bits[36]:0x800; bits[50]:0x2_4100_9242_555f; (bits[60]:0x7ff_ffff_ffff_ffff, bits[31]:0x2253_575f); bits[61]:0x1204_1604_227c_379b; bits[51]:0x0"
//     args: "bits[19]:0x5_5555; bits[36]:0xb_c12f_49d7; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0x400, bits[31]:0x200); bits[61]:0x1_0000; bits[51]:0x8000"
//     args: "bits[19]:0x1_0000; bits[36]:0x1_2219_d274; bits[50]:0x911a_c22e_80a8; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x5e83_2cf1); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_ffff_ffff; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x0, bits[31]:0x7cb1_2e7f); bits[61]:0x1555_5555_5555_5555; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x3_fa95; bits[36]:0x7_f52b_9915; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x3cd_2222_0000_0814, bits[31]:0x737c_9e52); bits[61]:0x155b_5465_0758_e2c2; bits[51]:0x3_c673_075d_c08f"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_ffff_ffff; bits[50]:0x3_fffa_ffff_c800; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x63bf_abee); bits[61]:0x0; bits[51]:0x4_3daf_b983_aa16"
//     args: "bits[19]:0x0; bits[36]:0x4_0000_0000; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x555_5555_5555_5555, bits[31]:0x4_0000); bits[61]:0x462_8575_513c_4207; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x5_5555; bits[36]:0x7_ffff_ffff; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0x61b_b6c0_b41f_906d, bits[31]:0x7fff_ffff); bits[61]:0x1e1a_7b8f_62f3_459d; bits[51]:0x3_4575_c0c5_b642"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x4_7474_abb9; bits[50]:0x1_f744_0000_0200; (bits[60]:0x735_55dd_bd56_160d, bits[31]:0x3fff_ffff); bits[61]:0x59e_2a50_f688_271a; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x6_d533; bits[36]:0x0; bits[50]:0x3_62dd_9000_c152; (bits[60]:0x808_0c50_4176_2a6a, bits[31]:0x4066_dd75); bits[61]:0xc4_01b0_00df_ffef; bits[51]:0x8444_5844_8104"
//     args: "bits[19]:0x3_ffff; bits[36]:0xa_aaaa_aaaa; bits[50]:0x400_0000; (bits[60]:0xba8_bc2a_ef74_0414, bits[31]:0x4648_0a00); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x7_ffff; bits[36]:0xf_fdfc_59e0; bits[50]:0x0; (bits[60]:0x555_5555_5555_5555, bits[31]:0x7ffd_5d70); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x7_feae_2cf0_5000"
//     args: "bits[19]:0x2_cb93; bits[36]:0x9_bd66_bce7; bits[50]:0x362a_d03a_b1ca; (bits[60]:0x7f8_86e0_6096_bc44, bits[31]:0x24fd_30e2); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x2_aaaa; bits[36]:0xa_0764_0421; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0x2, bits[31]:0x2); bits[61]:0x49e_ee08_50aa_a228; bits[51]:0x2_cdef_dfff_b7ff"
//     args: "bits[19]:0x7_f6be; bits[36]:0x40_0000; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0xbed_7bb6_9ffe_bfdf, bits[31]:0x7fff_ffff); bits[61]:0x777_5e27_f765_fd42; bits[51]:0x0"
//     args: "bits[19]:0x3_ffff; bits[36]:0x5_7fa0_2812; bits[50]:0x1_5568_0a04_9dbe; (bits[60]:0x555_5555_5555_5555, bits[31]:0xc4e_9dc6); bits[61]:0x1a7f_4850_0ccb_4805; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x3_bd2a; bits[36]:0x5_5555_5555; bits[50]:0x1_cbcd_4d17_5d4d; (bits[60]:0x72f_3534_5d75_3602, bits[31]:0x4d17_5dcf); bits[61]:0x1416_97a6_0798_326b; bits[51]:0x1_93d9_dfee_8ed1"
//     args: "bits[19]:0x4_046f; bits[36]:0x5_12ac_2bfd; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0x100, bits[31]:0x2aaa_aaaa); bits[61]:0xfbf_7fff_bfdf_f841; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x5_5555; bits[36]:0x2_b8ae_2010; bits[50]:0x2_8a42_b355_420a; (bits[60]:0x35c_0604_1093_facd, bits[31]:0x2340_42e2); bits[61]:0x20_0000_0000; bits[51]:0x1_5c5f_702b_34b0"
//     args: "bits[19]:0x5_f3a4; bits[36]:0x5_5555_5555; bits[50]:0x2_d9d1_3e2f_eaab; (bits[60]:0x0, bits[31]:0x1515_564d); bits[61]:0x176e_9bfd_ffef_fdbb; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x5_5555; bits[36]:0x7_e8aa_d320; bits[50]:0x1000; (bits[60]:0xa29_4040_02e2, bits[31]:0x6baa_d820); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x1_b454_6980_2aaa"
//     args: "bits[19]:0x3_ffff; bits[36]:0xf_ffff_ffff; bits[50]:0x3_fddf_fefb_e8ba; (bits[60]:0xf75_3fba_67a2_ed87, bits[31]:0x7fff_ffff); bits[61]:0x0; bits[51]:0x2_dfed_ff67_6dbd"
//     args: "bits[19]:0x5_0aee; bits[36]:0x0; bits[50]:0x0; (bits[60]:0xa56_cf7b_fd67_50e7, bits[31]:0x3fff_ffff); bits[61]:0x642_0010_21d7_5554; bits[51]:0x5_5555_5555_5555"
//     args: "bits[19]:0x7_ffff; bits[36]:0x9_3bac_1ca2; bits[50]:0x3_46bd_44dd_a461; (bits[60]:0xab1_6a1a_12ee_139d, bits[31]:0x64f6_8793); bits[61]:0x800_0000_0000_0000; bits[51]:0x4_e0c6_92ba_d603"
//     args: "bits[19]:0x400; bits[36]:0xc_c842_f6e7; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x9_e2e6_fa10_20da, bits[31]:0x5cca_8872); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x3_ec6f_234c_afd7"
//     args: "bits[19]:0x7_ffff; bits[36]:0x7_fbbe_1203; bits[50]:0x0; (bits[60]:0xfff_ffff_ffff_ffff, bits[31]:0x6201_33c0); bits[61]:0xc54_0488_0230_5c80; bits[51]:0x4802_e810_8801"
//     args: "bits[19]:0x2_aaaa; bits[36]:0xa_aaaa_aaaa; bits[50]:0x3_eaa9_0ab1_c4a1; (bits[60]:0xba6_ae2e_6212_879f, bits[31]:0x6de8_285e); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x400; bits[36]:0x7_4900_feff; bits[50]:0x1_5341_b63b_9c02; (bits[60]:0x0, bits[31]:0x0); bits[61]:0x20b_05f9_dce8_13fd; bits[51]:0x4f00_577b_6a33"
//     args: "bits[19]:0x0; bits[36]:0x1_5555; bits[50]:0x2_2e88_92bc_3ea0; (bits[60]:0x80_0000, bits[31]:0x2c1_5657); bits[61]:0x8d8_e585_5194_0015; bits[51]:0x1_6d8f_71bd_ae30"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x5_4556_0122; bits[50]:0x20; (bits[60]:0x141_48a1_4118_e91a, bits[31]:0x4008_1500); bits[61]:0x1555_5555_5555_5555; bits[51]:0x4_21ab_807b_a0c3"
//     args: "bits[19]:0x0; bits[36]:0x5_5555_5555; bits[50]:0x28_0000; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x5540_b156); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x8000; bits[36]:0x1_0201_ffff; bits[50]:0x1_5a40_3f74_bef6; (bits[60]:0x102_01ff_6b35_5155, bits[31]:0x200); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x2_b674_6acf_78ad"
//     args: "bits[19]:0xe3ab; bits[36]:0x5_c757_ed5f; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0x555_5555_5555_5555, bits[31]:0x3fff_ffff); bits[61]:0xa8b_0b9a_3dfb_af75; bits[51]:0x2_0000_0000"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_fffe_40c3; bits[50]:0x1_5555_5555_5555; (bits[60]:0x556_4559_4d15_d577, bits[31]:0x1_0000); bits[61]:0x8000_0000_0000; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x0; bits[36]:0xe_c885_4c55; bits[50]:0x2aaa_aaaa; (bits[60]:0x948_b521_6e0b_bddf, bits[31]:0x4621_0801); bits[61]:0x1555_5555_5555_5555; bits[51]:0x1_510d_183c_9cda"
//     args: "bits[19]:0x5_5555; bits[36]:0xa_a6e2_bf2c; bits[50]:0x3_3e2a_d85b_1c5d; (bits[60]:0x0, bits[31]:0x54ba_9f1d); bits[61]:0x1eb_faf4_99a3_07a9; bits[51]:0x0"
//     args: "bits[19]:0x2_aaaa; bits[36]:0xa_aaaa_aaaa; bits[50]:0x1_5555_5555_5555; (bits[60]:0x0, bits[31]:0x56c5_4554); bits[61]:0x1c58_cd45_0008_0020; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x2_aaaa; bits[36]:0x5_5555_0004; bits[50]:0x1_5dbf_6508_472f; (bits[60]:0x576_7d94_211c_bd54, bits[31]:0x5555_0804); bits[61]:0xacd_c928_e039_728a; bits[51]:0x40_0000"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_f3fe_0410; bits[50]:0x10_0000; (bits[60]:0x3fa_e612_586e_b81e, bits[31]:0x0); bits[61]:0x1555_5555_5555_5555; bits[51]:0x800"
//     args: "bits[19]:0x8; bits[36]:0x8ba8_4135; bits[50]:0x2; (bits[60]:0x1_0000, bits[31]:0x11_a000); bits[61]:0x1555_5555_5555_5555; bits[51]:0x1_8343_44af_c6c5"
//     args: "bits[19]:0x3_ffff; bits[36]:0x8_2fef_b7a4; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x8_0000_0000, bits[31]:0x2fef_b7a4); bits[61]:0xaaa_aaaa_aaaa_aaaa; bits[51]:0x4_17f7_dbd2_2f9e"
//     args: "bits[19]:0x4000; bits[36]:0xf_ffff_ffff; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x3ec_76d7_7ee1_d315, bits[31]:0x77fe_76ef); bits[61]:0x1555_5555_5555_5555; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x0; bits[36]:0x8_34e1_5b75; bits[50]:0x1_ffff_ffff_ffff; (bits[60]:0x0, bits[31]:0x4803_0084); bits[61]:0x1555_5555_5555_5555; bits[51]:0x3_073b_b5ef_f7fe"
//     args: "bits[19]:0x3_ffff; bits[36]:0x7_ffff_ffff; bits[50]:0x1_ffff_fbff_ccf4; (bits[60]:0x396_d7ad_dd86_c8a0, bits[31]:0xcd7_204d); bits[61]:0xfbb_fbdf_f42f_a67c; bits[51]:0x3_ffff_ffff_d555"
//     args: "bits[19]:0x0; bits[36]:0xa_aaaa_aaaa; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0xb28_bada_aa82_000a, bits[31]:0x204_8808); bits[61]:0x40_0200; bits[51]:0x5_1b99_0cc2_2ae2"
//     args: "bits[19]:0x0; bits[36]:0x2_1000_2004; bits[50]:0x2_58d4_b1a4_a6d0; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x3880_a0e4); bits[61]:0x0; bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[19]:0x40; bits[36]:0x8_2f2b_2990; bits[50]:0x2_aaaa_aaaa_aaaa; (bits[60]:0x4c_8555_544d_5354, bits[31]:0x2b33_b2e4); bits[61]:0x1fff_ffff_ffff_ffff; bits[51]:0x4_17b5_94c8_4000"
//     args: "bits[19]:0x7_ffff; bits[36]:0xb_d300_ec5b; bits[50]:0x2_b4bf_6e52_281d; (bits[60]:0xaf7_642e_c77e_ec5f, bits[31]:0x13ee_3371); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x7_b98a_c5b4_20f1"
//     args: "bits[19]:0x0; bits[36]:0x9_08cb_f7ef; bits[50]:0x800; (bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[31]:0x7fff_ffff); bits[61]:0x0; bits[51]:0x1_15ef_adef_0f98"
//     args: "bits[19]:0x0; bits[36]:0xa_aaaa_aaaa; bits[50]:0x2_aaaa_a8c2_8a46; (bits[60]:0x6c5_8bae_3bbb_04c9, bits[31]:0x3fff_ffff); bits[61]:0x0; bits[51]:0x800"
//     args: "bits[19]:0x5_5555; bits[36]:0x7_ffff_ffff; bits[50]:0x80_0000; (bits[60]:0x1_0000_0000, bits[31]:0x0); bits[61]:0xfff_ffff_ffff_ffff; bits[51]:0x5_5455_7fff_fbff"
//     args: "bits[19]:0x3_ffff; bits[36]:0xa_aaaa_aaaa; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0xfff_f488_8000_0040, bits[31]:0x3fff_ffff); bits[61]:0x1555_5555_5555_5555; bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[19]:0x7_ffff; bits[36]:0xf_bd7c_0201; bits[50]:0x3_ffff_ffff_ffff; (bits[60]:0x2fe_2756_acaa_e500, bits[31]:0x7eff_fff7); bits[61]:0x17f2_f5bd_c1cf_50a0; bits[51]:0x1_ffff_77ff_fffe"
//     args: "bits[19]:0x0; bits[36]:0x2000; bits[50]:0x2009_0846_0b71; (bits[60]:0x0, bits[31]:0x1880_6c00); bits[61]:0x242_1051_6b8c_bf64; bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[19]:0x3_ffff; bits[36]:0x5_5555_5555; bits[50]:0x3_197f_33e3_40e8; (bits[60]:0x0, bits[31]:0x2fff_f511); bits[61]:0xaaa_a6aa_aa52_0000; bits[51]:0x6_a6aa_fb82_8008"
//     args: "bits[19]:0x7_ffff; bits[36]:0xf_fff6_0aa8; bits[50]:0x1_ffbd_b23a_ac78; (bits[60]:0xe36_5df3_76ae_ae3e, bits[31]:0x5555_5555); bits[61]:0x0; bits[51]:0x4_2c05_0520_c2a8"
//     args: "bits[19]:0x5_e994; bits[36]:0xf_ffff_ffff; bits[50]:0x2_ee5e_379a_87b9; (bits[60]:0x7ff_ffff_ffff_ffff, bits[31]:0x3fff_ffff); bits[61]:0x1772_f1bc_d43d_c804; bits[51]:0x7_ffff_ffff_ffff"
//   }
// }
//
// END_CONFIG
fn main(x0: u19, x1: u36, x2: u50, x3: (u60, u31), x4: u61, x5: u51) -> u55 {
    {
        let x6: u55 = match x0 {
            u19:0x0 => u55:0x80_0000,
            u19:0x2_aaaa | u19:0x7_f508 => u55:0x40_0000,
            u19:0x100 => u55:0x3f_ffff_ffff_ffff,
            u19:0x3_ffff => u55:0x0,
            _ => u55:0x6c_4c7a_c5d8_f39e,
        };
        let x7: u55 = one_hot_sel(u6:0x15, [x6, x6, x6, x6, x6, x6]);
        let x8: u61 = x4 << if x1 >= u36:30 { u36:30 } else { x1 };
        let x9: u61 = x8 & x4;
        let x10: bool = and_reduce(x2);
        let x12: bool = x10 & x10;
        let x13: u50 = x2[x0+:u50];
        let x14: bool = !x12;
        let x15: u55 = x9 as u55 ^ x6;
        let x16: xN[bool:0x0][61] = x8[0+:u61];
        let x17: u61 = -x9;
        let x18: u48 = u48:0x2744_31f2_fdd7;
        let x19: u50 = x2 << if x15 >= u55:0x1a { u55:0x1a } else { x15 };
        let x20: u61 = x16[x0+:u61];
        let x21: xN[bool:0x0][61] = x1 as xN[bool:0x0][61] + x16;
        let x22: bool = x12 << if x17 >= u61:0x0 { u61:0x0 } else { x17 };
        let x23: bool = x10 ^ x22;
        let x24: bool = x12[x20+:bool];
        let x25: u51 = bit_slice_update(x5, x14, x20);
        let x26: bool = for (i, x): (u4, bool) in u4:0x0..=u4:0x3 {
            x
        }(x24);
        let x27: bool = x3 == x3;
        let x28: u3 = x1[x12+:u3];
        let x29: bool = x3 != x3;
        x7
    }
}
