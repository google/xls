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
// exception: "Subprocess call failed: xls/tools/eval_ir_main --input_file=/tmp/temp_directory_FVPevY/args.txt --use_llvm_jit /tmp/temp_directory_FVPevY/sample.ir --logtostderr\n\nSubprocess stderr:\n"
// issue: "https://github.com/google/xls/issues/1618"
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
//   codegen_args: "--worst_case_throughput=4"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=false"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: true
//   use_system_verilog: false
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
//     args: "bits[11]:0x200; [bits[10]:0x1ff, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0x2a4, bits[10]:0x280, bits[10]:0x0, bits[10]:0x200]; (bits[35]:0x6_0060_1e39, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x5_5555_5555); bits[51]:0x0"
//     args: "bits[11]:0x3ff; [bits[10]:0x2aa, bits[10]:0x26d, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0xd3, bits[10]:0x3ff]; (bits[35]:0x3_ffff_ffff, bits[56]:0x4_0000, bits[35]:0x3_ffff_ffff); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x0; [bits[10]:0x3ff, bits[10]:0x208, bits[10]:0x0, bits[10]:0x155, bits[10]:0x40, bits[10]:0x155, bits[10]:0x81]; (bits[35]:0x4_cd27_4d17, bits[56]:0x0, bits[35]:0x2_aaaa_aaaa); bits[51]:0x2000_0000"
//     args: "bits[11]:0x3ff; [bits[10]:0x2a3, bits[10]:0x3bf, bits[10]:0x3f7, bits[10]:0x299, bits[10]:0x1ff, bits[10]:0x17d, bits[10]:0x155]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x4_0000_0000, bits[35]:0x5_f995_f7e3); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x0; [bits[10]:0x3d9, bits[10]:0x208, bits[10]:0x125, bits[10]:0x0, bits[10]:0x2aa, bits[10]:0x2aa, bits[10]:0x2aa]; (bits[35]:0x0, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x3_e922_921e); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x4db; [bits[10]:0x3ff, bits[10]:0xd3, bits[10]:0x155, bits[10]:0x2aa, bits[10]:0x8e, bits[10]:0x1ff, bits[10]:0x2e8]; (bits[35]:0x4_dbaa_8b8a, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x7_ffff_ffff); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x2aa; [bits[10]:0x187, bits[10]:0x2a2, bits[10]:0x2aa, bits[10]:0x2a2, bits[10]:0x1dd, bits[10]:0x133, bits[10]:0xae]; (bits[35]:0x3_ffff_ffff, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x7_ffff_ffff); bits[51]:0x0"
//     args: "bits[11]:0x555; [bits[10]:0x1c1, bits[10]:0x1ff, bits[10]:0x219, bits[10]:0x2c2, bits[10]:0x75, bits[10]:0x20, bits[10]:0x155]; (bits[35]:0x5_11db_5f7d, bits[56]:0x0, bits[35]:0x0); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x17a; [bits[10]:0x173, bits[10]:0x1, bits[10]:0x155, bits[10]:0x17c, bits[10]:0x37a, bits[10]:0x1ff, bits[10]:0x0]; (bits[35]:0x0, bits[56]:0x6_7343_ffb0_944f, bits[35]:0x0); bits[51]:0x4000_0000_0000"
//     args: "bits[11]:0x7ff; [bits[10]:0x3ff, bits[10]:0x295, bits[10]:0x317, bits[10]:0x356, bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x33b]; (bits[35]:0x7_ffff_ffff, bits[56]:0x800_0000, bits[35]:0x5_5555_5555); bits[51]:0x7_ffaa_aaaa_aaaa"
//     args: "bits[11]:0x0; [bits[10]:0x4, bits[10]:0x6, bits[10]:0x0, bits[10]:0x0, bits[10]:0x3ff, bits[10]:0x40, bits[10]:0x301]; (bits[35]:0x80, bits[56]:0x2, bits[35]:0x5_5555_5555); bits[51]:0x0"
//     args: "bits[11]:0x2df; [bits[10]:0x39f, bits[10]:0x48, bits[10]:0x155, bits[10]:0x2df, bits[10]:0x37d, bits[10]:0x29a, bits[10]:0x2bb]; (bits[35]:0x0, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x3_ffff_ffff); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x555; [bits[10]:0x2aa, bits[10]:0xc6, bits[10]:0x2aa, bits[10]:0x259, bits[10]:0x38d, bits[10]:0x151, bits[10]:0x35d]; (bits[35]:0x6_4f25_1480, bits[56]:0xaa_afff_ddef_ffff, bits[35]:0x7_ffff_ffff); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x7ff; [bits[10]:0x3ff, bits[10]:0x150, bits[10]:0x0, bits[10]:0x216, bits[10]:0x0, bits[10]:0x35b, bits[10]:0x377]; (bits[35]:0x3_ffff_ffff, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x7_ffff_ffff); bits[51]:0x2_b37f_ff8d_ff74"
//     args: "bits[11]:0x188; [bits[10]:0x0, bits[10]:0x2aa, bits[10]:0x188, bits[10]:0x0, bits[10]:0x109, bits[10]:0x1e1, bits[10]:0x1]; (bits[35]:0x4_0000, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x1_9b55_5794); bits[51]:0x2_fc6c_62f1_009d"
//     args: "bits[11]:0x3ff; [bits[10]:0x2aa, bits[10]:0x3cf, bits[10]:0x2aa, bits[10]:0x1d3, bits[10]:0x155, bits[10]:0x1fe, bits[10]:0x3de]; (bits[35]:0x10, bits[56]:0x7d_6fff_7ffa_faff, bits[35]:0x6344_4af2); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x7ff; [bits[10]:0x155, bits[10]:0x53, bits[10]:0x0, bits[10]:0x3fb, bits[10]:0x0, bits[10]:0x7e, bits[10]:0x3ff]; (bits[35]:0x200, bits[56]:0x1, bits[35]:0xe2ef_73e7); bits[51]:0x0"
//     args: "bits[11]:0x80; [bits[10]:0x80, bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x155, bits[10]:0x155, bits[10]:0x200, bits[10]:0x8c]; (bits[35]:0x5_5555_5555, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x2_184d_89b2); bits[51]:0x0"
//     args: "bits[11]:0x7ff; [bits[10]:0x37f, bits[10]:0x2ff, bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x265]; (bits[35]:0x40_0000, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x5_ff9d_5e75); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x2aa; [bits[10]:0x19a, bits[10]:0x2c2, bits[10]:0x22, bits[10]:0x82, bits[10]:0x1e4, bits[10]:0x2a8, bits[10]:0x28a]; (bits[35]:0x4000, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x0); bits[51]:0x10_0000_0000"
//     args: "bits[11]:0x3ff; [bits[10]:0x9e, bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x2aa, bits[10]:0x39e, bits[10]:0x0]; (bits[35]:0x200_0000, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x7_ffff_ffff); bits[51]:0x4_9982_a5bb_da2c"
//     args: "bits[11]:0x2aa; [bits[10]:0x3ff, bits[10]:0x2b2, bits[10]:0x2e9, bits[10]:0x82, bits[10]:0x155, bits[10]:0x20, bits[10]:0x329]; (bits[35]:0x3a11_4e20, bits[56]:0x0, bits[35]:0x5_5555_5555); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x0; [bits[10]:0x100, bits[10]:0x0, bits[10]:0x29, bits[10]:0x3ff, bits[10]:0x200, bits[10]:0x273, bits[10]:0x93]; (bits[35]:0x4bd_a7b3, bits[56]:0x7_11f3_80e1_3616, bits[35]:0x0); bits[51]:0x0"
//     args: "bits[11]:0x2aa; [bits[10]:0x87, bits[10]:0x3ff, bits[10]:0x9a, bits[10]:0x0, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x1ff]; (bits[35]:0x3_ffff_ffff, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x5_fe43_49c4); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x547; [bits[10]:0x2cf, bits[10]:0x367, bits[10]:0x347, bits[10]:0x10f, bits[10]:0x147, bits[10]:0x3ff, bits[10]:0x155]; (bits[35]:0x5_47a8_aaaa, bits[56]:0xe8_77f9_ed7b_fddf, bits[35]:0x3_1d0a_50d4); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x0; [bits[10]:0x2aa, bits[10]:0x1ff, bits[10]:0x100, bits[10]:0x20, bits[10]:0x22, bits[10]:0x8, bits[10]:0x201]; (bits[35]:0x0, bits[56]:0x55_5555_5555_5555, bits[35]:0x7_ffff_ffff); bits[51]:0x7_cb6b_4a26_12ee"
//     args: "bits[11]:0x0; [bits[10]:0x155, bits[10]:0x3ff, bits[10]:0x1, bits[10]:0x0, bits[10]:0xc, bits[10]:0x100, bits[10]:0x0]; (bits[35]:0x0, bits[56]:0x1_e5a8_e298_a21e, bits[35]:0x100_0000); bits[51]:0x800_0000"
//     args: "bits[11]:0x6a5; [bits[10]:0x100, bits[10]:0x2a7, bits[10]:0x2ad, bits[10]:0x2a5, bits[10]:0x2ae, bits[10]:0x2a5, bits[10]:0x29c]; (bits[35]:0x6_a380_0200, bits[56]:0xd4_b555_5745_5145, bits[35]:0x0); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x7ff; [bits[10]:0x4, bits[10]:0x3bf, bits[10]:0x1ff, bits[10]:0x2aa, bits[10]:0x3ef, bits[10]:0x3d7, bits[10]:0x3ff]; (bits[35]:0x7_ffff_ffff, bits[56]:0xef_cf37_97e7_7f6f, bits[35]:0x7_7101_8221); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x7ff; [bits[10]:0x20, bits[10]:0x8, bits[10]:0x8, bits[10]:0x1ff, bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x29f]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0xcd_8fb9_b9df_62d8, bits[35]:0x5_b774_54cf); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x156, bits[10]:0x2ff, bits[10]:0x3df, bits[10]:0x25a, bits[10]:0xef, bits[10]:0x1bf, bits[10]:0xe6]; (bits[35]:0x7_ffff_ffff, bits[56]:0x2f_e3ed_6b76_f77f, bits[35]:0x3_e746_7160); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x0; [bits[10]:0xd2, bits[10]:0x0, bits[10]:0x2aa, bits[10]:0x0, bits[10]:0x92, bits[10]:0x1ff, bits[10]:0x155]; (bits[35]:0x2_00ff_ffff, bits[56]:0x2000, bits[35]:0x0); bits[51]:0x0"
//     args: "bits[11]:0x555; [bits[10]:0x241, bits[10]:0x199, bits[10]:0x71, bits[10]:0x183, bits[10]:0x1ff, bits[10]:0x141, bits[10]:0x155]; (bits[35]:0x5_1da3_cffe, bits[56]:0x89_fc95_f1ba_c9a2, bits[35]:0x5_5555_5555); bits[51]:0x5_59aa_8bab_a8a2"
//     args: "bits[11]:0x673; [bits[10]:0x200, bits[10]:0x353, bits[10]:0x1ff, bits[10]:0x251, bits[10]:0x1cd, bits[10]:0xcb, bits[10]:0x155]; (bits[35]:0x6_ef5d_fa8a, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x6_7b93_83aa); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x7ff; [bits[10]:0x2bf, bits[10]:0x2aa, bits[10]:0x0, bits[10]:0x37f, bits[10]:0x199, bits[10]:0x35e, bits[10]:0x37f]; (bits[35]:0x10_0000, bits[56]:0x9d_6aa2_cf7b_77fb, bits[35]:0x7_ffff_ffff); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x31f, bits[10]:0x3cf, bits[10]:0x37f, bits[10]:0x3ff, bits[10]:0x1ff, bits[10]:0x3ff, bits[10]:0x3df]; (bits[35]:0x2_3792_8374, bits[56]:0x7d_6800_0000_01c8, bits[35]:0x2_aaaa_aaaa); bits[51]:0x4_9d11_c684_ccc3"
//     args: "bits[11]:0x2ca; [bits[10]:0x2ca, bits[10]:0x2ca, bits[10]:0x10, bits[10]:0x3c3, bits[10]:0x2ca, bits[10]:0xef, bits[10]:0x2ca]; (bits[35]:0xa69_7d5f, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x1_53be_e5b6); bits[51]:0x2_f058_5e74_94f7"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ef, bits[10]:0x2ff, bits[10]:0x3ff, bits[10]:0x2, bits[10]:0x4, bits[10]:0x377, bits[10]:0x3ff]; (bits[35]:0x0, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x5_5555_5555); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x86; [bits[10]:0x18e, bits[10]:0x237, bits[10]:0x8e, bits[10]:0x1ff, bits[10]:0x3ff, bits[10]:0x82, bits[10]:0x1ff]; (bits[35]:0xaa97_bd3e, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x2_aaaa_aaaa); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x555; [bits[10]:0x2aa, bits[10]:0x20, bits[10]:0x155, bits[10]:0x1db, bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x1ff]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x5e_0b70_4c8e_4ff9, bits[35]:0x5_5555_5555); bits[51]:0x1_f27d_2e77_096e"
//     args: "bits[11]:0x80; [bits[10]:0xe0, bits[10]:0x88, bits[10]:0x1ff, bits[10]:0x182, bits[10]:0xd0, bits[10]:0x1ff, bits[10]:0xa0]; (bits[35]:0x8010_4245, bits[56]:0x0, bits[35]:0x5_5555_5555); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x365, bits[10]:0x33c, bits[10]:0x3ff, bits[10]:0x6c, bits[10]:0x3dc, bits[10]:0x1ea, bits[10]:0x155]; (bits[35]:0x0, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x6_6056_e5c9); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x735; [bits[10]:0x361, bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x337, bits[10]:0x335, bits[10]:0x13c, bits[10]:0x85]; (bits[35]:0x7_3550_ac00, bits[56]:0x20_0000_0000, bits[35]:0x7_ffff_ffff); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x5cd; [bits[10]:0x2aa, bits[10]:0x20d, bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x1cd, bits[10]:0x15e, bits[10]:0x22e]; (bits[35]:0x3_ffff_ffff, bits[56]:0xd9_7068_3caa_d07b, bits[35]:0x0); bits[51]:0x5_c4ed_8194_d781"
//     args: "bits[11]:0x752; [bits[10]:0x252, bits[10]:0x352, bits[10]:0x3de, bits[10]:0x0, bits[10]:0x352, bits[10]:0x318, bits[10]:0x212]; (bits[35]:0x2_8608_0457, bits[56]:0xce_49ef_fded_ff7b, bits[35]:0x80_0000); bits[51]:0x0"
//     args: "bits[11]:0x40; [bits[10]:0x82, bits[10]:0x155, bits[10]:0x1ff, bits[10]:0x40, bits[10]:0x155, bits[10]:0x40, bits[10]:0x68]; (bits[35]:0x6_7be7_d815, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x2_aaaa_aaaa); bits[51]:0x1000"
//     args: "bits[11]:0x10; [bits[10]:0x11, bits[10]:0x155, bits[10]:0x1c5, bits[10]:0x2d4, bits[10]:0x1ff, bits[10]:0x1ff, bits[10]:0x1ff]; (bits[35]:0x5_5555_5555, bits[56]:0x82_0aae_aaae_aabb, bits[35]:0x5_f106_f4b1); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x555; [bits[10]:0x55, bits[10]:0x155, bits[10]:0xa8, bits[10]:0x359, bits[10]:0x214, bits[10]:0x2aa, bits[10]:0x155]; (bits[35]:0x3_f2d4_f51e, bits[56]:0x48_b6ae_0412_1815, bits[35]:0x8); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ef, bits[10]:0x3f9, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x2f1, bits[10]:0x17f]; (bits[35]:0x5_5555_5555, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x2_ff15_5555); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x40; [bits[10]:0x10, bits[10]:0x3ff, bits[10]:0xa0, bits[10]:0x191, bits[10]:0x3ff, bits[10]:0x56, bits[10]:0x40]; (bits[35]:0x1_6900_5910, bits[56]:0x0, bits[35]:0x2_aaaa_aaaa); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x3ff; [bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0x155, bits[10]:0x32f, bits[10]:0x3fe, bits[10]:0x3ff]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x0, bits[35]:0x5_5555_5555); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x2; [bits[10]:0x20a, bits[10]:0x2aa, bits[10]:0x302, bits[10]:0x2, bits[10]:0x7, bits[10]:0x1ff, bits[10]:0x2aa]; (bits[35]:0x0, bits[56]:0x80_0000, bits[35]:0x62c8_874d); bits[51]:0x1_ee0e_46a5_f653"
//     args: "bits[11]:0x604; [bits[10]:0x204, bits[10]:0x155, bits[10]:0x3ff, bits[10]:0x24d, bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x2e4]; (bits[35]:0x4_1466_6286, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x3_ffff_ffff); bits[51]:0x4_04d7_521d_1545"
//     args: "bits[11]:0x200; [bits[10]:0x0, bits[10]:0x0, bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x3dc, bits[10]:0x280, bits[10]:0x1ff]; (bits[35]:0xa4db_a59c, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x2_0404_0000); bits[51]:0x3_2070_fbb5_efff"
//     args: "bits[11]:0x53; [bits[10]:0x53, bits[10]:0x20, bits[10]:0x1ff, bits[10]:0x353, bits[10]:0x0, bits[10]:0x40, bits[10]:0x2]; (bits[35]:0x7_ffff_ffff, bits[56]:0x9e_2a10_f98a_541a, bits[35]:0x2_0893_a110); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x7ff; [bits[10]:0x155, bits[10]:0x1ff, bits[10]:0x3ef, bits[10]:0x155, bits[10]:0x337, bits[10]:0x155, bits[10]:0x1df]; (bits[35]:0x5_bc0e_c06d, bits[56]:0x55_5555_5555_5555, bits[35]:0x7_ffff_ffff); bits[51]:0xf1ce_8a18_fe64"
//     args: "bits[11]:0x2aa; [bits[10]:0x2a, bits[10]:0x2aa, bits[10]:0x282, bits[10]:0x3ee, bits[10]:0x2d2, bits[10]:0x3ff, bits[10]:0x2aa]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x55_5555_5555_5555, bits[35]:0x5_5555_5555); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x7ff; [bits[10]:0x3fa, bits[10]:0x227, bits[10]:0x3ff, bits[10]:0x28e, bits[10]:0x3ff, bits[10]:0x31c, bits[10]:0x3ff]; (bits[35]:0x3_ffff_ffff, bits[56]:0x0, bits[35]:0x7_6f76_7be9); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x4; [bits[10]:0x32d, bits[10]:0x5, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x30d]; (bits[35]:0x4_1788_ba5c, bits[56]:0x81_a2aa_c8e0_fa16, bits[35]:0x3_3c58_e34b); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x555; [bits[10]:0xaa, bits[10]:0x0, bits[10]:0x134, bits[10]:0x1bd, bits[10]:0x15d, bits[10]:0x2aa, bits[10]:0x2f5]; (bits[35]:0x3_ffff_ffff, bits[56]:0x0, bits[35]:0x5_5555_5555); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x1ff, bits[10]:0x1ff, bits[10]:0x100, bits[10]:0x3ff, bits[10]:0x2bf, bits[10]:0x2aa, bits[10]:0x1f7]; (bits[35]:0x5_5555_5555, bits[56]:0xfd_3102_5164_563d, bits[35]:0x3_d655_54d5); bits[51]:0x2_d094_0a4f_d2c3"
//     args: "bits[11]:0x7ff; [bits[10]:0x1eb, bits[10]:0x0, bits[10]:0x155, bits[10]:0x27f, bits[10]:0x200, bits[10]:0x1ff, bits[10]:0x3e7]; (bits[35]:0x1, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x2_aaaa_aaaa); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x4; [bits[10]:0x80, bits[10]:0x2e1, bits[10]:0x15, bits[10]:0x3ff, bits[10]:0x155, bits[10]:0xcc, bits[10]:0x210]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x61_1409_4000_2fb6, bits[35]:0x5_5555_5555); bits[51]:0x5_9500_03d0_08c0"
//     args: "bits[11]:0x10; [bits[10]:0x2, bits[10]:0x24c, bits[10]:0x1ff, bits[10]:0x3ae, bits[10]:0x10, bits[10]:0x3ff, bits[10]:0x69]; (bits[35]:0x28f_bdba, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x5_5555_5555); bits[51]:0x1000_4200_0004"
//     args: "bits[11]:0x7ff; [bits[10]:0x143, bits[10]:0x46, bits[10]:0x155, bits[10]:0x3fb, bits[10]:0x374, bits[10]:0x2aa, bits[10]:0x155]; (bits[35]:0x0, bits[56]:0x0, bits[35]:0x7_5e48_0420); bits[51]:0x5_f885_5222_200a"
//     args: "bits[11]:0x0; [bits[10]:0x208, bits[10]:0x19b, bits[10]:0x3ff, bits[10]:0x0, bits[10]:0xa0, bits[10]:0x24, bits[10]:0x1df]; (bits[35]:0x3_ffff_ffff, bits[56]:0x2b_02e1_10a2_3f32, bits[35]:0x3_0df7_45d3); bits[51]:0x0"
//     args: "bits[11]:0x2aa; [bits[10]:0x2aa, bits[10]:0x3ea, bits[10]:0x3ff, bits[10]:0x12f, bits[10]:0x261, bits[10]:0x2a8, bits[10]:0x2bb]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x4_fa56_1f50); bits[51]:0x71b0_88a6_f8e9"
//     args: "bits[11]:0x0; [bits[10]:0x0, bits[10]:0x39d, bits[10]:0x40, bits[10]:0x80, bits[10]:0x9, bits[10]:0x2aa, bits[10]:0x0]; (bits[35]:0x7_ffff_ffff, bits[56]:0x5c_6866_30ec_3105, bits[35]:0x8000); bits[51]:0x1_b269_f9a7_7589"
//     args: "bits[11]:0x555; [bits[10]:0x200, bits[10]:0x159, bits[10]:0x100, bits[10]:0x26, bits[10]:0x257, bits[10]:0x11d, bits[10]:0x55]; (bits[35]:0x3_47f3_4759, bits[56]:0xbe_af5d_f439_553d, bits[35]:0x5_150a_ea9a); bits[51]:0x6_5f6a_89af_9ac4"
//     args: "bits[11]:0x2aa; [bits[10]:0x8, bits[10]:0x2b2, bits[10]:0x3b0, bits[10]:0x80, bits[10]:0x2aa, bits[10]:0x0, bits[10]:0x2aa]; (bits[35]:0x7_ffff_ffff, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x0); bits[51]:0x200"
//     args: "bits[11]:0x7ff; [bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x40, bits[10]:0x2f3, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0x0]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x0); bits[51]:0x0"
//     args: "bits[11]:0x7ff; [bits[10]:0x3ed, bits[10]:0x0, bits[10]:0x3ff, bits[10]:0x3ef, bits[10]:0x3ff, bits[10]:0x3f, bits[10]:0x1bf]; (bits[35]:0x7_2f80_6124, bits[56]:0x0, bits[35]:0x7_ff8a_abaa); bits[51]:0x3_fd1f_6f69_f7d7"
//     args: "bits[11]:0x7ff; [bits[10]:0x155, bits[10]:0x3ff, bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x3fb]; (bits[35]:0x3_ffaa_aa2a, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x7_ffff_ffff); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x7ff; [bits[10]:0x27e, bits[10]:0x37d, bits[10]:0x97, bits[10]:0x8, bits[10]:0x1b, bits[10]:0x1ff, bits[10]:0x0]; (bits[35]:0x7_ffaa_aaaa, bits[56]:0x55_5555_5555_5555, bits[35]:0x5_edb7_f4f2); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x555; [bits[10]:0x37d, bits[10]:0x0, bits[10]:0x15d, bits[10]:0x147, bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x3ff]; (bits[35]:0x8000, bits[56]:0xeb_d5d4_57d4_951b, bits[35]:0x5_5500_0000); bits[51]:0x7_651d_55ae_ee31"
//     args: "bits[11]:0x2aa; [bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x20, bits[10]:0x39, bits[10]:0x155, bits[10]:0x155, bits[10]:0x3ff]; (bits[35]:0x5_5555_5555, bits[56]:0xc6_98d0_4988_c414, bits[35]:0x306_e162); bits[51]:0xa935_8901_7b19"
//     args: "bits[11]:0x2aa; [bits[10]:0x280, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x3aa, bits[10]:0x155, bits[10]:0x2fa, bits[10]:0x0]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0xf2_1986_55c3_952f, bits[35]:0xea00_c098); bits[51]:0x0"
//     args: "bits[11]:0x0; [bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x200, bits[10]:0x0, bits[10]:0x8b, bits[10]:0x155, bits[10]:0x3d2]; (bits[35]:0x3_ffff_ffff, bits[56]:0x23_01c5_2708_80d3, bits[35]:0x4_b513_e085); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x2e; [bits[10]:0x4, bits[10]:0x3ff, bits[10]:0x13, bits[10]:0x2aa, bits[10]:0xae, bits[10]:0x26, bits[10]:0x20]; (bits[35]:0xbe60_ad21, bits[56]:0x5_c000_8020_80a0, bits[35]:0x7_ffff_ffff); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x262; [bits[10]:0x0, bits[10]:0x70, bits[10]:0x155, bits[10]:0x35e, bits[10]:0x1ff, bits[10]:0x3ff, bits[10]:0x2aa]; (bits[35]:0x5_3561_b1a2, bits[56]:0xe0_018e_dcdb_a4ad, bits[35]:0x7_ffff_ffff); bits[51]:0x5_f396_e4b8_e64e"
//     args: "bits[11]:0x0; [bits[10]:0x11e, bits[10]:0x155, bits[10]:0x19, bits[10]:0x0, bits[10]:0x3ff, bits[10]:0x2, bits[10]:0x0]; (bits[35]:0x1_8c12_438d, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x2_aaaa_aaaa); bits[51]:0x2_aaaa_aaaa_aaaa"
//     args: "bits[11]:0x49d; [bits[10]:0x9d, bits[10]:0x155, bits[10]:0x3ff, bits[10]:0xdd, bits[10]:0x392, bits[10]:0x2aa, bits[10]:0x0]; (bits[35]:0x4_9d54_5555, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x3_ffff_ffff); bits[51]:0x0"
//     args: "bits[11]:0x7ff; [bits[10]:0x1ff, bits[10]:0x3a3, bits[10]:0x0, bits[10]:0x80, bits[10]:0x3cf, bits[10]:0x3fc, bits[10]:0x3fe]; (bits[35]:0x5_5555_5555, bits[56]:0x0, bits[35]:0x5_5555_5555); bits[51]:0x1_0000"
//     args: "bits[11]:0x3ff; [bits[10]:0x377, bits[10]:0x3ff, bits[10]:0x1ff, bits[10]:0x32f, bits[10]:0x40, bits[10]:0x361, bits[10]:0x3ff]; (bits[35]:0x1_b7f7_d545, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x1_b4d0_f233); bits[51]:0x3_7fe9_fdff_efcf"
//     args: "bits[11]:0x7ff; [bits[10]:0x1ff, bits[10]:0x3d5, bits[10]:0x155, bits[10]:0x3ff, bits[10]:0x3ff, bits[10]:0x346, bits[10]:0x1ff]; (bits[35]:0x7_ff7f_ffdf, bits[56]:0x7e_e001_2008_4804, bits[35]:0x5_5555_5555); bits[51]:0x7_afc7_8caf_c80c"
//     args: "bits[11]:0x2aa; [bits[10]:0x1ff, bits[10]:0x12a, bits[10]:0x2aa, bits[10]:0x3ff, bits[10]:0x2ac, bits[10]:0x1, bits[10]:0x2ba]; (bits[35]:0x5_5555_5555, bits[56]:0x0, bits[35]:0x0); bits[51]:0x0"
//     args: "bits[11]:0x555; [bits[10]:0x1ef, bits[10]:0x156, bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x131, bits[10]:0x51, bits[10]:0x50]; (bits[35]:0x3_ffff_ffff, bits[56]:0x37_a01a_d0c5_1c04, bits[35]:0x5_5580_0008); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x0; [bits[10]:0x283, bits[10]:0x0, bits[10]:0x2, bits[10]:0x19, bits[10]:0x188, bits[10]:0x297, bits[10]:0x20]; (bits[35]:0x815d_60b3, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x7_ffff_ffff); bits[51]:0x1_10b6_091d_57de"
//     args: "bits[11]:0x3f7; [bits[10]:0x3f3, bits[10]:0x1ef, bits[10]:0x155, bits[10]:0x0, bits[10]:0x287, bits[10]:0x0, bits[10]:0x372]; (bits[35]:0x7_ffff_ffff, bits[56]:0x0, bits[35]:0x7_ffff_ffff); bits[51]:0x2_c1cb_8178_dfdd"
//     args: "bits[11]:0x200; [bits[10]:0x218, bits[10]:0x241, bits[10]:0x5, bits[10]:0x3ff, bits[10]:0x200, bits[10]:0x27b, bits[10]:0x173]; (bits[35]:0x3_d149_a333, bits[56]:0xde_1d5f_3fb6_a4d8, bits[35]:0x5_5555_5555); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x2aa; [bits[10]:0x32a, bits[10]:0x6d, bits[10]:0x22d, bits[10]:0x2ea, bits[10]:0x0, bits[10]:0x0, bits[10]:0x26d]; (bits[35]:0x5_5555_5555, bits[56]:0x54_d555_5545_d555, bits[35]:0x5_5555_5555); bits[51]:0x100"
//     args: "bits[11]:0x2aa; [bits[10]:0x2aa, bits[10]:0x155, bits[10]:0x2aa, bits[10]:0x2aa, bits[10]:0x23a, bits[10]:0x0, bits[10]:0x3a8]; (bits[35]:0x20_0000, bits[56]:0x54_1fd4_eefd_6f77, bits[35]:0x2_aaaa_aaaa); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x0; [bits[10]:0x1ff, bits[10]:0x18d, bits[10]:0x0, bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x0]; (bits[35]:0x800_0000, bits[56]:0x8000_0000_0000, bits[35]:0x4_84b5_d991); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ff, bits[10]:0x37f, bits[10]:0x155, bits[10]:0x1ff, bits[10]:0x3ff, bits[10]:0x155, bits[10]:0x155]; (bits[35]:0x3_ffff_ffff, bits[56]:0x63_d786_b115_3575, bits[35]:0x7_7ed5_1115); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x0; [bits[10]:0x2c1, bits[10]:0x8, bits[10]:0x8, bits[10]:0x0, bits[10]:0x3ff, bits[10]:0x0, bits[10]:0x140]; (bits[35]:0x2009_3c82, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x2fe_ffdb); bits[51]:0x2_811e_679a_ff6e"
//     args: "bits[11]:0x75; [bits[10]:0x40, bits[10]:0x7d, bits[10]:0x27c, bits[10]:0x75, bits[10]:0x7e, bits[10]:0xf7, bits[10]:0x60]; (bits[35]:0x5_5555_5555, bits[56]:0x0, bits[35]:0x2_757f_ffff); bits[51]:0x1_6750_5e49_0a4b"
//     args: "bits[11]:0x4; [bits[10]:0x155, bits[10]:0x26e, bits[10]:0x4, bits[10]:0x2aa, bits[10]:0x8c, bits[10]:0x0, bits[10]:0x4c]; (bits[35]:0x2_e484_25be, bits[56]:0xa0_a49f_eb20_abf1, bits[35]:0x2_0cea_aaeb); bits[51]:0x2_1c62_6f20_f6b1"
//     args: "bits[11]:0x400; [bits[10]:0x0, bits[10]:0x0, bits[10]:0x1c4, bits[10]:0x1a4, bits[10]:0x8, bits[10]:0x24e, bits[10]:0x39]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x3_ffff_ffff); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ff, bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x337, bits[10]:0x2aa, bits[10]:0x332, bits[10]:0x0]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x36_a006_250e_1a00, bits[35]:0x7_ffff_ffff); bits[51]:0x3_f78a_aaaa_aaaa"
//     args: "bits[11]:0x3ff; [bits[10]:0x3f7, bits[10]:0x3f7, bits[10]:0xf1, bits[10]:0x2aa, bits[10]:0x2aa, bits[10]:0x2d6, bits[10]:0x1ff]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x0, bits[35]:0x0); bits[51]:0x3_e63f_fcff_daad"
//     args: "bits[11]:0x555; [bits[10]:0x154, bits[10]:0x1c, bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x155, bits[10]:0x115, bits[10]:0x0]; (bits[35]:0x20, bits[56]:0x0, bits[35]:0x0); bits[51]:0x200"
//     args: "bits[11]:0x555; [bits[10]:0x57, bits[10]:0x0, bits[10]:0x155, bits[10]:0x145, bits[10]:0x315, bits[10]:0x3ff, bits[10]:0x2aa]; (bits[35]:0x0, bits[56]:0x82_e6f1_7d6f_d256, bits[35]:0x3_ffff_ffff); bits[51]:0x0"
//     args: "bits[11]:0x3ff; [bits[10]:0x20, bits[10]:0x3ff, bits[10]:0x13f, bits[10]:0x3c8, bits[10]:0x29f, bits[10]:0x2aa, bits[10]:0xbf]; (bits[35]:0x5_e756_c5e6, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x3_c9ff_7edf); bits[51]:0xef63_1000_0a21"
//     args: "bits[11]:0x0; [bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x0, bits[10]:0x155, bits[10]:0x0]; (bits[35]:0x2_8aa0_828d, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x7_ffff_ffff); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x555; [bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x1ff, bits[10]:0x300, bits[10]:0x155, bits[10]:0x155, bits[10]:0x12f]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x0, bits[35]:0x3_ffff_ffff); bits[51]:0x3_4059_1bd0_bdcc"
//     args: "bits[11]:0x555; [bits[10]:0x157, bits[10]:0x0, bits[10]:0x8, bits[10]:0x155, bits[10]:0x156, bits[10]:0xe8, bits[10]:0x35d]; (bits[35]:0x5_5555_5555, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x4_7580_282e); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x1ff, bits[10]:0x2f1, bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x2bb, bits[10]:0x2ee, bits[10]:0x1fb]; (bits[35]:0x3_ffff_ffff, bits[56]:0x1, bits[35]:0x0); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x555; [bits[10]:0x155, bits[10]:0x57, bits[10]:0x95, bits[10]:0x0, bits[10]:0x47, bits[10]:0x155, bits[10]:0x155]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x7_ffff_ffff); bits[51]:0x2_0000"
//     args: "bits[11]:0x555; [bits[10]:0x154, bits[10]:0x1ff, bits[10]:0x35d, bits[10]:0x0, bits[10]:0x17d, bits[10]:0x1ff, bits[10]:0x1fa]; (bits[35]:0x0, bits[56]:0xf9_ab2d_f75c_571b, bits[35]:0x2_eda0_91df); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ff, bits[10]:0x322, bits[10]:0x3e7, bits[10]:0x1fa, bits[10]:0x1fe, bits[10]:0x2aa, bits[10]:0xcb]; (bits[35]:0x7_ffff_ffff, bits[56]:0x7f_ffff_ffff_ffff, bits[35]:0x2_aaaa_aaaa); bits[51]:0x3_23b7_e563_8507"
//     args: "bits[11]:0x0; [bits[10]:0x3ff, bits[10]:0x0, bits[10]:0x80, bits[10]:0x3ff, bits[10]:0x6a, bits[10]:0x2c2, bits[10]:0x0]; (bits[35]:0x5_5555_5555, bits[56]:0x8a0b_a02a_a6a3, bits[35]:0x0); bits[51]:0x800"
//     args: "bits[11]:0x243; [bits[10]:0x0, bits[10]:0x0, bits[10]:0x253, bits[10]:0x2aa, bits[10]:0x363, bits[10]:0x1ff, bits[10]:0x1ff]; (bits[35]:0xca3f_ffff, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x0); bits[51]:0x6_c935_792a_7df6"
//     args: "bits[11]:0x0; [bits[10]:0x20, bits[10]:0x40, bits[10]:0x191, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x0, bits[10]:0x8]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x8000_0000_0000, bits[35]:0x1_02e4_d369); bits[51]:0x57b_d62d_7f7b"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ff, bits[10]:0x2a3, bits[10]:0x39f, bits[10]:0x20, bits[10]:0x100, bits[10]:0x3ef, bits[10]:0x171]; (bits[35]:0x3_4551_3a74, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x3_ffff_ffff); bits[51]:0x3_fd52_b64f_f866"
//     args: "bits[11]:0x3ff; [bits[10]:0x0, bits[10]:0x232, bits[10]:0x155, bits[10]:0x3fb, bits[10]:0x37f, bits[10]:0xa7, bits[10]:0x3b5]; (bits[35]:0x5_5555_5555, bits[56]:0x80, bits[35]:0x0); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x7ff; [bits[10]:0x2aa, bits[10]:0x2aa, bits[10]:0x3bc, bits[10]:0x3ff, bits[10]:0x155, bits[10]:0x3ff, bits[10]:0x377]; (bits[35]:0x3_ffff_ffff, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x2_aaaa_aaaa); bits[51]:0x80_0000_0000"
//     args: "bits[11]:0x5d4; [bits[10]:0x346, bits[10]:0x2a2, bits[10]:0x1dc, bits[10]:0x37, bits[10]:0x1d4, bits[10]:0x155, bits[10]:0x2c4]; (bits[35]:0x400_0000, bits[56]:0xad_932f_c3fb_ffcb, bits[35]:0x1_55f0_2295); bits[51]:0x4_0000_0000_0000"
//     args: "bits[11]:0x0; [bits[10]:0x2c1, bits[10]:0x1ff, bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x3ff, bits[10]:0x1ff, bits[10]:0x0]; (bits[35]:0x2_0000_0000, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x100); bits[51]:0x2000"
//     args: "bits[11]:0x80; [bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x9a, bits[10]:0x155, bits[10]:0x2aa]; (bits[35]:0x90fe_8f75, bits[56]:0x0, bits[35]:0x5_5555_5555); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x2aa; [bits[10]:0x2aa, bits[10]:0x3aa, bits[10]:0x2c2, bits[10]:0x3ff, bits[10]:0x32a, bits[10]:0xde, bits[10]:0x155]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x55_5555_5555_5555, bits[35]:0x2_aaaa_aaaa); bits[51]:0x1_0000"
//     args: "bits[11]:0x2aa; [bits[10]:0x2aa, bits[10]:0x1ff, bits[10]:0x155, bits[10]:0x2aa, bits[10]:0x1aa, bits[10]:0x1bb, bits[10]:0xaa]; (bits[35]:0x3_ffff_ffff, bits[56]:0xff_ffff_ffff_ffff, bits[35]:0x0); bits[51]:0x7_ffff_ffff_ffff"
//     args: "bits[11]:0x0; [bits[10]:0x3ff, bits[10]:0x0, bits[10]:0x155, bits[10]:0x155, bits[10]:0x100, bits[10]:0x40, bits[10]:0x155]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0x1da5_490a_06cd, bits[35]:0x0); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x8; [bits[10]:0x155, bits[10]:0x2aa, bits[10]:0x24e, bits[10]:0x0, bits[10]:0x3cc, bits[10]:0x155, bits[10]:0xc9]; (bits[35]:0x7_ffff_ffff, bits[56]:0x5_16fa_3f71_b7a6, bits[35]:0x5_5555_5555); bits[51]:0x5_5555_5555_5555"
//     args: "bits[11]:0x8; [bits[10]:0x9e, bits[10]:0x0, bits[10]:0x281, bits[10]:0x28f, bits[10]:0x155, bits[10]:0x10, bits[10]:0x2aa]; (bits[35]:0x0, bits[56]:0x55_5555_5555_5555, bits[35]:0x2_aaaa_aaaa); bits[51]:0x2_0842_2ea2_1412"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x19f, bits[10]:0x2aa, bits[10]:0x3df, bits[10]:0x2d7, bits[10]:0x4]; (bits[35]:0x7_ffff_ffff, bits[56]:0x1000_0000, bits[35]:0x2_aaaa_aaaa); bits[51]:0x2_dd74_9765_efe1"
//     args: "bits[11]:0x7ff; [bits[10]:0x0, bits[10]:0x1ff, bits[10]:0x313, bits[10]:0x3f7, bits[10]:0x0, bits[10]:0x37f, bits[10]:0x3ff]; (bits[35]:0x1_0000_0000, bits[56]:0xaa_aaaa_aaaa_aaaa, bits[35]:0x8000); bits[51]:0x8000_0000_0000"
//     args: "bits[11]:0x3ff; [bits[10]:0x3ff, bits[10]:0x1ff, bits[10]:0x3ff, bits[10]:0x2aa, bits[10]:0x8a, bits[10]:0x155, bits[10]:0x3df]; (bits[35]:0x7_ffff_ffff, bits[56]:0x55_5555_5555_5555, bits[35]:0x7_ffff_ffff); bits[51]:0x3_ffff_ffff_ffff"
//     args: "bits[11]:0x3ff; [bits[10]:0x1fb, bits[10]:0x2aa, bits[10]:0x155, bits[10]:0x0, bits[10]:0x3dd, bits[10]:0x2aa, bits[10]:0x3ff]; (bits[35]:0x2_aaaa_aaaa, bits[56]:0xbf_ceff_da7b_47ff, bits[35]:0x2_aaaa_aaaa); bits[51]:0x0"
//   }
// }
// 
// END_CONFIG
type x0 = u10;
type x9 = u17;
type x16 = u51;
type x37 = bool;
fn x25(x26: u51, x27: bool) -> (s51, u7, u7, u35, bool) {
    {
        let x29: s51 = {
            let x28: (u51, u51) = smulp(x27 as u51 as s51, x26 as s51);
            (x28.0 + x28.1) as s51
        };
        let x30: u35 = (x29 as u51)[11+:u35];
        let x31: u7 = u7:0x3f;
        (x29, x31, x31, x30, x27)
    }
}
fn main(x1: u11, x2: x0[7], x3: (u35, u56, u35), x4: u51) -> (bool, u4, bool, x16[3], (bool, u24)) {
    {
        let x5: u51 = x4 ^ x1 as u51;
        let x6: (bool, u24) = match x3 {
            (u35:0x0..u35:7051596819, u56:0x7f_ffff_ffff_ffff, u35:0x7_ffff_ffff) | (u35:0x5_5555_5555..u35:32017070457, u56:0x2000_0000_0000, u35:0x5_5555_5555) => (bool:true, u24:0xff_ffff),
            (u35:0x6_7833_1770, u56:0x1_0000, u35:0x5_5555_5555) | (_, u56:0x55_5555_5555_5555..u56:50149519974952489, u35:0x3_ffff_ffff..u35:29444246375) => (bool:false, u24:0b0),
            _ => (bool:true, u24:0x7f_ffff),
        };
        let (x7, x8): (bool, u24) = match x3 {
            (u35:0x0..u35:7051596819, u56:0x7f_ffff_ffff_ffff, u35:0x7_ffff_ffff) | (u35:0x5_5555_5555..u35:32017070457, u56:0x2000_0000_0000, u35:0x5_5555_5555) => (bool:true, u24:0xff_ffff),
            (u35:0x6_7833_1770, u56:0x1_0000, u35:0x5_5555_5555) | (_, u56:0x55_5555_5555_5555..u56:50149519974952489, u35:0x3_ffff_ffff..u35:29444246375) => (bool:false, u24:0b0),
            _ => (bool:true, u24:0x7f_ffff),
        };
        let x10: x9[3] = x4 as x9[3];
        let x11: bool = x7 >> if x7 >= bool:false { bool:false } else { x7 };
        let x12: x0 = x2[x7];
        let x13: x9[18] = slice(x10, x4, x9[18]:[x10[u32:0x0], ...]);
        let x14: bool = x11 - x4 as bool;
        let x15: bool = x6.0;
        let x17: x16[3] = [x4, x5, x4];
        let x18: bool = x8 as bool - x11;
        let x19: u4 = encode(x12);
        let x20: bool = x2 != x2;
        let x21: x0 = one_hot_sel(x19, [x12, x12, x12, x12]);
        let x22: bool = clz(x20);
        let x23: x0 = signex(x14, x12);
        let x24: bool = !x22;
        let x32: (s51, u7, u7, u35, bool) = x25(x4, x11);
        let x33: x16[6] = x17 ++ x17;
        let x34: x16[12] = x33 ++ x33;
        let x35: bool = x20 as bool * x15;
        let x36: u51 = !x4;
        let x38: x37[7] = [x11, x14, x15, x18, x35, x7, x15];
        (x14, x19, x18, x17, x6)
    }
}
