// Copyright 2023 The XLS Authors
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
// exception: 	 "SampleError: Result miscompare for sample 12:\nargs: (bits[60]:0x0, bits[40]:0xff_ffff_ffff, (bits[57]:0x8_0000_0000_0000, bits[32]:0xaaaa_aaaa)); bits[43]:0x0; bits[8]:0x9c; bits[47]:0x5555_5555_5555\nevaluated opt IR (JIT), evaluated opt IR (interpreter), evaluated unopt IR (interpreter), interpreted DSLX, simulated =\n   (bits[43]:0x2aa_aaaa_aa8e, bits[50]:0x1_ffff_ffff_ffff, bits[14]:0x3fff, bits[43]:0x0)\nevaluated unopt IR (JIT) =\n   (bits[43]:0x2aa_aaaa_aa8e, bits[50]:0x1_ffff_ffff_ffff, bits[14]:0x3ffb, bits[43]:0x0)"
// issue: "https://github.com/google/xls/issues/1200"
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
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
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
//     args: "(bits[60]:0x0, bits[40]:0xa6_0e20_6103, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x7fff_ffff)); bits[43]:0x555_5555_5555; bits[8]:0xaa; bits[47]:0x5520_eed2_7d2a"
//     args: "(bits[60]:0x702_0cfe_2f55_e6f4, bits[40]:0x7f_ffff_ffff, (bits[57]:0x155_5555_5555_5555, bits[32]:0xffff_ffff)); bits[43]:0x4; bits[8]:0x5; bits[47]:0x2a83_9060_4ab9"
//     args: "(bits[60]:0x0, bits[40]:0xad_148d_d139, (bits[57]:0xa4_862a_7e78_8b71, bits[32]:0x4000)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xda; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x0, bits[40]:0xff_ffff_ffff, (bits[57]:0xa2_a683_a383_1dc9, bits[32]:0x7fff_ffff)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xdd; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x42_1f74_ad22, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xaaaa_aaaa)); bits[43]:0x117_4d17_acb4; bits[8]:0xaa; bits[47]:0x20_0000"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0xff_ffff_ffff, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x0)); bits[43]:0x400_0000_0000; bits[8]:0xef; bits[47]:0x4000_0000_4000"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x4_0000, (bits[57]:0x155_5555_5555_5555, bits[32]:0x5555_5555)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xff; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0xf08_f31d_f849_5b65, bits[40]:0xff_ffff_ffff, (bits[57]:0x40_0000_0000_0000, bits[32]:0xaaaa_aaaa)); bits[43]:0x17f_54e4_76e8; bits[8]:0xe8; bits[47]:0x740d_6cf8_6470"
//     args: "(bits[60]:0x80_0000, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x155_5555_5555_5555, bits[32]:0x8000)); bits[43]:0x3ff_ffff_ffff; bits[8]:0x55; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x5555_5555)); bits[43]:0x555_5555_5555; bits[8]:0x0; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x2d_4edf_0d8d, (bits[57]:0x155_5555_5555_5555, bits[32]:0x0)); bits[43]:0x2c6_4983_6e98; bits[8]:0x7f; bits[47]:0x28c5_1a30_f8ef"
//     args: "(bits[60]:0x0, bits[40]:0x7f_ffff_ffff, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x0; bits[8]:0x83; bits[47]:0x41c7_55cd_55d7"
//     args: "(bits[60]:0x0, bits[40]:0xff_ffff_ffff, (bits[57]:0x8_0000_0000_0000, bits[32]:0xaaaa_aaaa)); bits[43]:0x0; bits[8]:0x9c; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x3f_06e2_c191, (bits[57]:0x0, bits[32]:0xaaaa_aaaa)); bits[43]:0x0; bits[8]:0x5d; bits[47]:0x5a06_12be_110c"
//     args: "(bits[60]:0x40_0000_0000_0000, bits[40]:0x200_0000, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x4_0000; bits[8]:0x7f; bits[47]:0x5404_3343_395c"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x40_0000_0000, (bits[57]:0x155_5555_5555_5555, bits[32]:0xffff_ffff)); bits[43]:0x555_5555_5555; bits[8]:0x0; bits[47]:0x56bd_7569_8c70"
//     args: "(bits[60]:0x2000, bits[40]:0x0, (bits[57]:0x8e_d838_d427_9586, bits[32]:0x0)); bits[43]:0x555_5555_5555; bits[8]:0x55; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x0)); bits[43]:0x2; bits[8]:0x2; bits[47]:0x50c4_110a_0828"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x55_5555_5555, (bits[57]:0x8000_0000_0000, bits[32]:0x0)); bits[43]:0x555_5555_5555; bits[8]:0x3f; bits[47]:0x0"
//     args: "(bits[60]:0xbeb_e306_c606_3740, bits[40]:0x70_c952_86b6, (bits[57]:0x8_0000_0000, bits[32]:0x0)); bits[43]:0x8_0000; bits[8]:0x0; bits[47]:0x1069_0058_4ac4"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0x1000_0000_0000, bits[32]:0x7fff_ffff)); bits[43]:0x3ff_ffff_ffff; bits[8]:0x7f; bits[47]:0x3fff_ffff_fff7"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0x0, bits[32]:0x10_0000)); bits[43]:0x1e6_a6eb_c632; bits[8]:0x55; bits[47]:0x0"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x55_5555_5555, (bits[57]:0x200_0000_0000, bits[32]:0xffff_ffff)); bits[43]:0x555_5555_5555; bits[8]:0xe4; bits[47]:0x4a07_1496_0014"
//     args: "(bits[60]:0xe2b_d95b_2699_a56d, bits[40]:0x7f_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x8000; bits[8]:0x80; bits[47]:0x0"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x4c_887a_0567, (bits[57]:0x155_5555_5555_5555, bits[32]:0x20_0000)); bits[43]:0x0; bits[8]:0xff; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x0, (bits[57]:0x0, bits[32]:0x5555_5555)); bits[43]:0x0; bits[8]:0x78; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x0, bits[32]:0xffff_ffff)); bits[43]:0x0; bits[8]:0x84; bits[47]:0x4080_201f"
//     args: "(bits[60]:0x40_0000_0000, bits[40]:0x4, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x4_0000)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xa8; bits[47]:0x0"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x0, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x7fff_ffff)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x0; bits[47]:0x2aaa_aa8a_aea1"
//     args: "(bits[60]:0x29e_618e_365b_fafc, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x5555_5555)); bits[43]:0x555_5555_5555; bits[8]:0x7d; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x55_5555_5555, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x7fff_ffff)); bits[43]:0x80; bits[8]:0x35; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x0, bits[40]:0x0, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x5555_5555)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x55; bits[47]:0x0"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0xaaaa_aaaa)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x8; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0x0, bits[40]:0x55_5555_5555, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x5555_5555)); bits[43]:0x3af_1a30_8e20; bits[8]:0x0; bits[47]:0x321_2204_0060"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0xc_1ed3_e7c3, (bits[57]:0x4_0000_0000, bits[32]:0x5555_5555)); bits[43]:0x2000; bits[8]:0x55; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0x20, bits[40]:0x0, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xaaaa_aaaa)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xd8; bits[47]:0x0"
//     args: "(bits[60]:0x0, bits[40]:0x55_5555_5555, (bits[57]:0x20_0000_0000_0000, bits[32]:0x7fff_ffff)); bits[43]:0x0; bits[8]:0x0; bits[47]:0x1051_5571_7585"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x8_0000_0000, (bits[57]:0x1cf_56a9_6db2_a9f7, bits[32]:0x0)); bits[43]:0x7ff_ffff_ffff; bits[8]:0xff; bits[47]:0x100_0000"
//     args: "(bits[60]:0xd74_823c_67f7_d94b, bits[40]:0x0, (bits[57]:0x8_0000_0000_0000, bits[32]:0x73e4_e7ef)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xaa; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x7f; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x25_09c6_27eb, (bits[57]:0x2_0000_0000_0000, bits[32]:0xffff_ffff)); bits[43]:0x3e2_85da_f408; bits[8]:0x10; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x0, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x64b1_5c51)); bits[43]:0x0; bits[8]:0x7f; bits[47]:0x0"
//     args: "(bits[60]:0x1_0000_0000, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0xd6_c7b3_5d35_4958, bits[32]:0xf135_b7c9)); bits[43]:0x0; bits[8]:0xff; bits[47]:0x400_2010_0040"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xf4_7770_adf8, (bits[57]:0x155_5555_5555_5555, bits[32]:0x7fff_ffff)); bits[43]:0x20_0000_0000; bits[8]:0x33; bits[47]:0x724e_155c_44e7"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x7fff_ffff)); bits[43]:0x4d8_4fcf_57e1; bits[8]:0x6b; bits[47]:0x4c84_7de5_727f"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x5_68b7_e47b, (bits[57]:0x155_5555_5555_5555, bits[32]:0xaaaa_aaaa)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x1; bits[47]:0x4a8_ba2b_8bae"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x1_0000, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x8000_0000)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x82; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x6b_c475_a650_196f, bits[32]:0x0)); bits[43]:0x0; bits[8]:0x0; bits[47]:0x4821_a6c0_62ee"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x66c_f4bd_995f; bits[8]:0x40; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x0, bits[40]:0x800_0000, (bits[57]:0x155_5555_5555_5555, bits[32]:0x2_0000)); bits[43]:0x7ff_ffff_ffff; bits[8]:0xbd; bits[47]:0x7fb7_9fd8_fd7b"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xff_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x0)); bits[43]:0x1000; bits[8]:0xaa; bits[47]:0x1_0000"
//     args: "(bits[60]:0x0, bits[40]:0x7f_ffff_ffff, (bits[57]:0x155_5555_5555_5555, bits[32]:0xb46b_de74)); bits[43]:0x4000; bits[8]:0x45; bits[47]:0x0"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0xff_ffff_ffff, (bits[57]:0xd7_945d_2d41_e0dc, bits[32]:0x5555_5555)); bits[43]:0x3ff_ffff_ffff; bits[8]:0x7f; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x0, bits[40]:0x55_5555_5555, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x7fff_ffff)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x7f; bits[47]:0x2aba_aaab_aaa0"
//     args: "(bits[60]:0x800_0000, bits[40]:0xa1_53c8_327f, (bits[57]:0x155_5555_5555_5555, bits[32]:0xaaaa_aaaa)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x0; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0x0, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x7fff_ffff)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x93; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x40_0000_0000, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x7fff_ffff)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xff; bits[47]:0x577f_fbff_ffda"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x716f_6e6c)); bits[43]:0x3ff_ffff_ffff; bits[8]:0x55; bits[47]:0x40_0000_0000"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x71_479a_49b1, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x0)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x1; bits[47]:0x0"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x0, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x0)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x7f; bits[47]:0x0"
//     args: "(bits[60]:0x10, bits[40]:0x800, (bits[57]:0x0, bits[32]:0xa273_9579)); bits[43]:0x3ff_ffff_ffff; bits[8]:0x7e; bits[47]:0x907_914d_2b98"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x0, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xaaaa_aaaa)); bits[43]:0x200; bits[8]:0x55; bits[47]:0xa48_75d4_c511"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x40)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xaa; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0xdca_a59f_9141_588d, bits[40]:0x7f_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x2f17_0251)); bits[43]:0x795_fa94_0299; bits[8]:0x7f; bits[47]:0x3ffd_ffff_bfff"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x0, (bits[57]:0x2_0000, bits[32]:0x0)); bits[43]:0x71d_f794_76b4; bits[8]:0xe4; bits[47]:0x40"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x1000, (bits[57]:0x0, bits[32]:0x7fff_ffff)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x55; bits[47]:0x1dff_bbc9_39df"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x20_0000, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x0)); bits[43]:0x400_0000_0000; bits[8]:0xff; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x20_0000, bits[40]:0x0, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x5555_5555)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x8; bits[47]:0x6eea_baeb_abfd"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0xff_ffff_ffff, (bits[57]:0x800, bits[32]:0x7fff_ffff)); bits[43]:0x0; bits[8]:0x55; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x4_0765_1749, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x7fff_ffff)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x55; bits[47]:0x7651_d6b4_62de"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0x155_5555_5555_5555, bits[32]:0xaaaa_aaaa)); bits[43]:0xc4_3e86_ff29; bits[8]:0x6c; bits[47]:0x0"
//     args: "(bits[60]:0x724_7b96_c864_15d7, bits[40]:0x7f_ffff_ffff, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x7fff_ffff)); bits[43]:0x0; bits[8]:0x55; bits[47]:0xd9c_23c0_e24e"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x555_5555_5555; bits[8]:0xf5; bits[47]:0x35d4_d451_dc5c"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x1_0000, bits[32]:0x0)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x2; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x55_5555_5555, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x7fff_ffff)); bits[43]:0x0; bits[8]:0x0; bits[47]:0x800"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x9_6935_ca4c, (bits[57]:0x9e_fefb_d36a_7d2e, bits[32]:0x0)); bits[43]:0x3ff_ffff_ffff; bits[8]:0x7f; bits[47]:0x3fff_ffff_fff7"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x10_0000_0000, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x5555_5555)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xaa; bits[47]:0x0"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xaaaa_aaaa)); bits[43]:0x20; bits[8]:0xaa; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x8000_0000, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x5555_5555)); bits[43]:0x788_f7a7_9d9d; bits[8]:0x55; bits[47]:0x6ab8_7ff5_b7f7"
//     args: "(bits[60]:0x0, bits[40]:0xff_ffff_ffff, (bits[57]:0x155_5555_5555_5555, bits[32]:0x80_0000)); bits[43]:0x8000; bits[8]:0xff; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x20; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x2000_0000_0000, bits[40]:0x55_5555_5555, (bits[57]:0x200_0000, bits[32]:0x7fff_ffff)); bits[43]:0x0; bits[8]:0x11; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x57_3489_e50a, (bits[57]:0x155_5555_5555_5555, bits[32]:0xaaaa_aaaa)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xff; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x48d1_e289)); bits[43]:0x0; bits[8]:0x7f; bits[47]:0x0"
//     args: "(bits[60]:0x8ad_2677_311e_56c7, bits[40]:0xa5_670d_e1cf, (bits[57]:0x17b_2af0_f723_fde6, bits[32]:0x4d3e_b492)); bits[43]:0x7ff_ffff_ffff; bits[8]:0xaa; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x0, bits[40]:0x3f_1499_35ba, (bits[57]:0x10_0000_0000, bits[32]:0xaaaa_aaaa)); bits[43]:0x555_5555_5555; bits[8]:0xaa; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xd2_008a_5bde, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x0)); bits[43]:0x0; bits[8]:0xa; bits[47]:0x106c_8967_4d39"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff, (bits[57]:0x195_1372_6d40_0349, bits[32]:0x0)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xc2; bits[47]:0x0"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xa6_65c6_d02e, (bits[57]:0x40_0000_0000, bits[32]:0x5555_5555)); bits[43]:0x0; bits[8]:0xff; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0xd57_a250_665b_6ac3, bits[40]:0x0, (bits[57]:0x155_5555_5555_5555, bits[32]:0xffff_ffff)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xaa; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x0)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x55; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x0, bits[40]:0x55_5555_5555, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x7fff_ffff)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xaa; bits[47]:0x6655_5d57_8d77"
//     args: "(bits[60]:0x1_0000_0000, bits[40]:0x0, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x1_0000_0000; bits[8]:0x55; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0xff_ffff_ffff, (bits[57]:0x0, bits[32]:0xffff_ffff)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xaa; bits[47]:0x22ec_47a1_fb71"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x0)); bits[43]:0x0; bits[8]:0x28; bits[47]:0x1455_5555_5555"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x0, (bits[57]:0x400_0000, bits[32]:0x0)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xaa; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x55_5555_5555, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xff32_fa16)); bits[43]:0x0; bits[8]:0xaa; bits[47]:0x14f4_7ef4_904a"
//     args: "(bits[60]:0x521_6e98_30d4_4d68, bits[40]:0xff_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x7fff_ffff)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x55; bits[47]:0x2b78_5dcb_5773"
//     args: "(bits[60]:0xe0f_ac2f_9a2a_d74b, bits[40]:0x0, (bits[57]:0x100_0000_0000, bits[32]:0x5555_5555)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xd5; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0xc2_2c05_f236, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x5555_5555)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xfe; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x155_5555_5555_5555, bits[32]:0xaaaa_aaaa)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x10; bits[47]:0x7dbf_f777_dee3"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff, (bits[57]:0x155_5555_5555_5555, bits[32]:0x5555_5555)); bits[43]:0x3ff_ffff_ffff; bits[8]:0xaa; bits[47]:0x17ff_caff_c3f5"
//     args: "(bits[60]:0x0, bits[40]:0x0, (bits[57]:0x9f_4e88_b2b4_e12a, bits[32]:0x20_0000)); bits[43]:0x0; bits[8]:0x20; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x8000_0000, (bits[57]:0x2000_0000, bits[32]:0xaaaa_aaaa)); bits[43]:0x2_0000; bits[8]:0xff; bits[47]:0x7fbf_ffff_dfff"
//     args: "(bits[60]:0x0, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x7fff_ffff)); bits[43]:0x20_0000; bits[8]:0xff; bits[47]:0x366f_12ad_0cc1"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0xff_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xaaaa_aaaa)); bits[43]:0x7ff_ffff_ffff; bits[8]:0xc8; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0xc1_fdea_12e0_6ea5, bits[40]:0x0, (bits[57]:0x155_5555_5555_5555, bits[32]:0x0)); bits[43]:0x0; bits[8]:0x7f; bits[47]:0x41f7_7fea_dc4f"
//     args: "(bits[60]:0x8ff_8c23_f000_792f, bits[40]:0x98_f0bc_0c7c, (bits[57]:0x155_5555_5555_5555, bits[32]:0x0)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x20; bits[47]:0x40_0000_0000"
//     args: "(bits[60]:0x0, bits[40]:0x0, (bits[57]:0x155_5555_5555_5555, bits[32]:0x2000)); bits[43]:0x555_5555_5555; bits[8]:0x40; bits[47]:0x2aaa_aaaa_aaaa"
//     args: "(bits[60]:0x607_b29c_a014_932a, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x0, bits[32]:0x400_0000)); bits[43]:0x7ff_ffff_ffff; bits[8]:0xd7; bits[47]:0x0"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x400, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x5555_5555)); bits[43]:0x0; bits[8]:0x83; bits[47]:0x3181_4124_e0cb"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xff_ffff_ffff, (bits[57]:0x155_5555_5555_5555, bits[32]:0xaaaa_aaaa)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xaa; bits[47]:0x298a_aaa8_aaa5"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0x7f_ffff_ffff, (bits[57]:0x4000_0000, bits[32]:0xffff_ffff)); bits[43]:0x555_5555_5555; bits[8]:0x55; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x55_5555_5555, (bits[57]:0x40_0000_0000_0000, bits[32]:0x7fff_ffff)); bits[43]:0x80_0000; bits[8]:0xe9; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x0, bits[40]:0x0, (bits[57]:0x1ef_333f_293a_15b5, bits[32]:0x0)); bits[43]:0x3e_fc82_83a2; bits[8]:0x7f; bits[47]:0x0"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0x4a_2568_f06c, (bits[57]:0x155_5555_5555_5555, bits[32]:0xbba1_89c7)); bits[43]:0x555_5555_5555; bits[8]:0x0; bits[47]:0x7fff_ffff_ffff"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x6af_b576_76b9; bits[8]:0xaa; bits[47]:0x276b_d227_bbfe"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xff_ffff_ffff, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xd3e1_a9b5)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x7f; bits[47]:0x3fff_ffff_ffff"
//     args: "(bits[60]:0x0, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x200_0000, bits[32]:0x40)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x7f; bits[47]:0x3f77_9516_1db7"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x7f_ffff_ffff, (bits[57]:0x0, bits[32]:0x4000)); bits[43]:0x7ff_ffff_ffff; bits[8]:0xfe; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x0, bits[40]:0xff_ffff_ffff, (bits[57]:0xab_3218_3a74_8b12, bits[32]:0x0)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x0; bits[47]:0x0"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0xa_cec4_9e3d, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0x40)); bits[43]:0x2000; bits[8]:0xff; bits[47]:0x1"
//     args: "(bits[60]:0x0, bits[40]:0x7f_ffff_ffff, (bits[57]:0x0, bits[32]:0x20)); bits[43]:0x336_d620_12e9; bits[8]:0x46; bits[47]:0x8_0000"
//     args: "(bits[60]:0x555_5555_5555_5555, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0xaa_aaaa_aaaa_aaaa, bits[32]:0x0)); bits[43]:0x7ff_ffff_ffff; bits[8]:0x55; bits[47]:0x67ef_d772_0153"
//     args: "(bits[60]:0xaaa_aaaa_aaaa_aaaa, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x20_0000, bits[32]:0xb699_3223)); bits[43]:0x6a9_4949_ec06; bits[8]:0x7f; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0xaa_aaaa_aaaa, (bits[57]:0x1ff_ffff_ffff_ffff, bits[32]:0x1000_0000)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0xce; bits[47]:0x395b_5f32_af39"
//     args: "(bits[60]:0x7ff_ffff_ffff_ffff, bits[40]:0x0, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xffff_ffff)); bits[43]:0x7ff_ffff_ffff; bits[8]:0xff; bits[47]:0x5555_5555_5555"
//     args: "(bits[60]:0xfff_ffff_ffff_ffff, bits[40]:0x0, (bits[57]:0xff_ffff_ffff_ffff, bits[32]:0xaaaa_aaaa)); bits[43]:0x2aa_aaaa_aaaa; bits[8]:0x0; bits[47]:0x2aaa_aaaa_aaaa"
//   }
// }
// 
// END_CONFIG
fn main(x0: (u60, u40, (s57, u32)), x1: u43, x2: u8, x3: s47) -> (u43, s50, u14, u43) {
    {
        let x4: u43 = -x1;
        let x5: s47 = signex(x4, x3);
        let x6: uN[256] = decode<uN[256]>(x2);
        let x7: uN[238] = decode<uN[238]>(x2);
        let x8: u43 = x4 + x6 as u43;
        let x9: s50 = s50:0x1_ffff_ffff_ffff;
        let x10: uN[2048] = decode<uN[2048]>(x4);
        let x11: u12 = x8[x2+:u12];
        let x12: bool = x3 <= x3;
        let x13: u43 = x10 as u43 ^ x8;
        let x14: u14 = x8[26+:u14];
        let x15: u43 = clz(x1);
        let x16: u44 = one_hot(x1, bool:0x1);
        let x17: s47 = !x3;
        let x18: u43 = x15 * x17 as u43;
        let x19: u43 = x1 >> x14;
        let x20: u14 = !x14;
        let x21: s2 = s2:0x2;
        let x22: u37 = x13[6+:u37];
        let x23: uN[2033] = decode<uN[2033]>(x10);
        let x24: u2 = x2[x4+:u2];
        let x25: u8 = rev(x2);
        let x26: s61 = match x22 {
            u37:0x1f_ffff_ffff | u37:0xf_ffff_ffff..u37:0xa_aaaa_aaaa => s61:0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111,
            _ => s61:0x0,
        };
        (x18, x9, x20, x19)
    }
}
