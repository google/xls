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
// exception: "Subprocess call timed out after 1500 seconds: /xls/tools/eval_ir_main --testvector_textproto=testvector.pbtxt --use_llvm_jit sample.ir --logtostderr"
// issue: "https://github.com/google/xls/issues/4380"
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
//   codegen_args: "--pipeline_stages=9"
//   codegen_args: "--worst_case_throughput=6"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: true
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
//   disable_unopt_interpreter: true
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   function_args {
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_0000, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_97e5, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x2_9c2b]; bits[30]:0x1fff_ffff; bits[7]:0x7f"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_98cd, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x3_44c8, bits[18]:0x3_ce5c, bits[18]:0x1_5555, bits[18]:0x3_ffff]; bits[30]:0x1fff_ffff; bits[7]:0x55"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x10, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_0000, bits[18]:0x1_5555, bits[18]:0x3_828a, bits[18]:0x20, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555]; bits[30]:0x1555_5555; bits[7]:0x55"
//     args: "[bits[18]:0x2_0000, bits[18]:0x0, bits[18]:0x2000, bits[18]:0x1_5555, bits[18]:0x1_c9b4, bits[18]:0x8, bits[18]:0x100, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0xc8d, bits[18]:0x1_5555, bits[18]:0x1000, bits[18]:0x2000]; bits[30]:0x1555_5555; bits[7]:0x3f"
//     args: "[bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_0000, bits[18]:0x3_e51d, bits[18]:0x1_ffff, bits[18]:0x72a2, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x2_0000, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x1_7b08, bits[18]:0x3_ffff, bits[18]:0x2_aaaa]; bits[30]:0x8; bits[7]:0x55"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x100, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_a5cd, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_6141]; bits[30]:0x1555_5555; bits[7]:0x28"
//     args: "[bits[18]:0x2_0000, bits[18]:0x0, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x2_0000, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_394b, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0]; bits[30]:0x4_0000; bits[7]:0x0"
//     args: "[bits[18]:0x2, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_e835, bits[18]:0x3_f627, bits[18]:0x6991, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x4, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x0"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x200, bits[18]:0x1_ffff]; bits[30]:0x80_0000; bits[7]:0x8"
//     args: "[bits[18]:0x1_5555, bits[18]:0x2_efd4, bits[18]:0x1_5555, bits[18]:0x5f4a, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_11e3, bits[18]:0x3_ffff, bits[18]:0x1_ffff]; bits[30]:0x20; bits[7]:0x0"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x2_4c00, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_5f6f, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa]; bits[30]:0x1fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x3_4e08, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x4000, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_2f72, bits[18]:0x1_5555]; bits[30]:0x1555_5555; bits[7]:0x1f"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x2, bits[18]:0x3_faad, bits[18]:0x3_ffff, bits[18]:0x3_6e15, bits[18]:0x2_0000, bits[18]:0x2_1f7f, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x800, bits[18]:0x2_bb7d, bits[18]:0x2_1620, bits[18]:0x4, bits[18]:0x2_aaaa]; bits[30]:0x1555_5555; bits[7]:0x3f"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x2000, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555]; bits[30]:0x0; bits[7]:0x10"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x49cf, bits[18]:0x80, bits[18]:0x1_f241, bits[18]:0x1_d16f, bits[18]:0x2_7671, bits[18]:0x2_0000, bits[18]:0x2_503d, bits[18]:0x3_ffff, bits[18]:0x1_5555]; bits[30]:0x1_0000; bits[7]:0x7f"
//     args: "[bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff]; bits[30]:0x1fff_ffff; bits[7]:0x7f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x2_2547, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x2_5264, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_e75d, bits[18]:0x4ba2, bits[18]:0x2_aaaa, bits[18]:0x1_5555]; bits[30]:0x8000; bits[7]:0x1"
//     args: "[bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_89ef, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3_9a45, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff]; bits[30]:0x20; bits[7]:0x0"
//     args: "[bits[18]:0x2_b931, bits[18]:0x373d, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x20, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_1551]; bits[30]:0x1555_5555; bits[7]:0x0"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x1_0000, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_03ab, bits[18]:0x3_ffff, bits[18]:0x1_bc62, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x20]; bits[30]:0x3fff_ffff; bits[7]:0x0"
//     args: "[bits[18]:0x7e8, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x2, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_cc1f, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_71d4]; bits[30]:0x30b8_53f8; bits[7]:0x1f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1, bits[18]:0x1_ffff, bits[18]:0x1_b416, bits[18]:0x1_d962, bits[18]:0x200, bits[18]:0x2000, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x2_a1fb, bits[18]:0x0]; bits[30]:0x400; bits[7]:0xa"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x3_75df, bits[18]:0x1_ad84, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_9b55, bits[18]:0x1_0000, bits[18]:0x2_fd64, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x2_aaaa]; bits[30]:0x1555_5555; bits[7]:0x45"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x800, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ac7a, bits[18]:0x1_5555, bits[18]:0x8e46, bits[18]:0x2_aaaa, bits[18]:0x585c, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2000, bits[18]:0x3_ffff]; bits[30]:0x200; bits[7]:0x0"
//     args: "[bits[18]:0x4, bits[18]:0x2_8aa8, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3a41, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_c864, bits[18]:0x1_3637, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff]; bits[30]:0x0; bits[7]:0x20"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x2_6344, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2_dc48, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff]; bits[30]:0x2aaa_aaaa; bits[7]:0x3f"
//     args: "[bits[18]:0x1_5555, bits[18]:0xde5c, bits[18]:0x1_5555, bits[18]:0x10, bits[18]:0x4000, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x8778, bits[18]:0x2_aaaa]; bits[30]:0x400; bits[7]:0x7f"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_16d5, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x8, bits[18]:0x3_ffff, bits[18]:0x0]; bits[30]:0x1bca_78a6; bits[7]:0x3f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x8f02, bits[18]:0x3_8222, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_57df, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x2_0000]; bits[30]:0x1555_5555; bits[7]:0x7f"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x2_0000, bits[18]:0x800, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x3_a9d0, bits[18]:0x3_ffff, bits[18]:0x1_aa85, bits[18]:0x1_4c54, bits[18]:0x3_ffff]; bits[30]:0x2aaa_aaaa; bits[7]:0x55"
//     args: "[bits[18]:0x3_3421, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_cac0, bits[18]:0x1_ffff, bits[18]:0x8, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_0000, bits[18]:0x40, bits[18]:0x2_aaaa, bits[18]:0x3_ffff]; bits[30]:0x1fff_ffff; bits[7]:0x7f"
//     args: "[bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x2_4f14, bits[18]:0x1_ffff, bits[18]:0x1_75d3, bits[18]:0x3_ffff, bits[18]:0xb11e, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x4, bits[18]:0x1_ffff, bits[18]:0x1000, bits[18]:0x1_ddfa]; bits[30]:0x1555_5555; bits[7]:0x5c"
//     args: "[bits[18]:0x1_2c23, bits[18]:0x2_4309, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2_b003, bits[18]:0x1_ffff, bits[18]:0x3_5dc3, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0]; bits[30]:0x2bea_dd5b; bits[7]:0x59"
//     args: "[bits[18]:0x80, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x4, bits[18]:0x0, bits[18]:0x400, bits[18]:0x3_ccfb, bits[18]:0x1_1a6d, bits[18]:0x2, bits[18]:0x1_ffff, bits[18]:0x1_ffff]; bits[30]:0x400; bits[7]:0x4d"
//     args: "[bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_0390, bits[18]:0x8, bits[18]:0x2_aaaa, bits[18]:0x3_5a50, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x4]; bits[30]:0x200_0000; bits[7]:0x0"
//     args: "[bits[18]:0x2_9cf8, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x2_858b, bits[18]:0x1_5555, bits[18]:0x3_5c98, bits[18]:0xf5d2, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x80, bits[18]:0x800, bits[18]:0x3_ffff, bits[18]:0x2_aaaa]; bits[30]:0x1555_5555; bits[7]:0x55"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x2000, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x2, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_1c7e, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x0]; bits[30]:0x2_0000; bits[7]:0x9"
//     args: "[bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_1699, bits[18]:0x0, bits[18]:0x4000, bits[18]:0xdc47, bits[18]:0xb395, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_5555]; bits[30]:0x3fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x8000, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_53b4, bits[18]:0xacef, bits[18]:0xafe3, bits[18]:0x20, bits[18]:0x2_aaaa, bits[18]:0x2_94a4, bits[18]:0x3_8e9c, bits[18]:0x1_5555, bits[18]:0x2_aaaa]; bits[30]:0x3fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x2, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_1da6, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0xf043]; bits[30]:0x1be5_b4f5; bits[7]:0x64"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3d7b, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1000, bits[18]:0xc854, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_3c4a, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0]; bits[30]:0x2014_aff0; bits[7]:0x40"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x3_444e, bits[18]:0x1_6c88, bits[18]:0x2_f818, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x40, bits[18]:0x3_ffff, bits[18]:0x1_c18a, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x4]; bits[30]:0x2aaa_aaaa; bits[7]:0x2a"
//     args: "[bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_edac, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_aaaa]; bits[30]:0x2aaa_aaaa; bits[7]:0x40"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x800, bits[18]:0x1_6dcd, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x10, bits[18]:0x3_eea4, bits[18]:0x1_ffff, bits[18]:0x2_0000, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555]; bits[30]:0x2aaa_aaaa; bits[7]:0x7a"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x2_c194, bits[18]:0x10, bits[18]:0x3_ffff, bits[18]:0x20, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x7e3e, bits[18]:0x3_ffff, bits[18]:0x8000, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa]; bits[30]:0x1fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_45af, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0xaf9f, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_2f1a]; bits[30]:0x1555_5555; bits[7]:0x57"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x20, bits[18]:0x2_de9f, bits[18]:0x2000, bits[18]:0x4, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_acd3]; bits[30]:0x3fff_ffff; bits[7]:0x7d"
//     args: "[bits[18]:0x2_f82e, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x4, bits[18]:0x2_aaaa, bits[18]:0x3_4be1, bits[18]:0x2_aaaa, bits[18]:0x100, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x5d76]; bits[30]:0x2aaa_aaaa; bits[7]:0x7f"
//     args: "[bits[18]:0x3_68c4, bits[18]:0x1_ffff, bits[18]:0x80, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2_0000, bits[18]:0x1_5555, bits[18]:0x1_5935, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x4, bits[18]:0x0]; bits[30]:0x1fff_ffff; bits[7]:0x55"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x74f4, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555]; bits[30]:0x115f_88f1; bits[7]:0x53"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x4, bits[18]:0x8, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff]; bits[30]:0x1555_5555; bits[7]:0x7f"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x7fb4, bits[18]:0x0, bits[18]:0x8, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_de19, bits[18]:0x1_5555, bits[18]:0x2_70f2, bits[18]:0x3_cfca, bits[18]:0x2_d30a]; bits[30]:0x1fff_ffff; bits[7]:0x48"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_a932, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_b9f3, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_ca8f, bits[18]:0x2_5572, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff]; bits[30]:0x400; bits[7]:0xd"
//     args: "[bits[18]:0x100, bits[18]:0x1_5555, bits[18]:0x10, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x8000, bits[18]:0x8000, bits[18]:0x3_3d50]; bits[30]:0x2aaa_aaaa; bits[7]:0x2a"
//     args: "[bits[18]:0x20, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_8f3a, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_c6bc, bits[18]:0x1_5555, bits[18]:0x1_023d, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x4000, bits[18]:0x2_aaaa]; bits[30]:0x1fff_ffff; bits[7]:0x4"
//     args: "[bits[18]:0x8, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x200, bits[18]:0x0, bits[18]:0x4, bits[18]:0x2_aaaa, bits[18]:0x800]; bits[30]:0x1000; bits[7]:0x40"
//     args: "[bits[18]:0x3_e98a, bits[18]:0x200, bits[18]:0x2_887d, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff]; bits[30]:0x1; bits[7]:0x0"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x400, bits[18]:0x1_ffff, bits[18]:0x20, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x100, bits[18]:0x2_aaaa]; bits[30]:0x1555_5555; bits[7]:0x7f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_0000, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x3_00ab, bits[18]:0x1_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x77"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x4, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x2_ca34, bits[18]:0x3_f10b, bits[18]:0x3_ffff, bits[18]:0xbbbe, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_ffff]; bits[30]:0xf7d_ec99; bits[7]:0x7d"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x200, bits[18]:0x2_0fc7, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_0000]; bits[30]:0x2aaa_aaaa; bits[7]:0x0"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x200, bits[18]:0x2_05a4, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x3_ffff]; bits[30]:0x0; bits[7]:0x55"
//     args: "[bits[18]:0x2_e50f, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_362a, bits[18]:0x1_ffff, bits[18]:0x1000, bits[18]:0x1_9fda, bits[18]:0xb45e, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555]; bits[30]:0x1555_5555; bits[7]:0x7f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_1712, bits[18]:0x1_5555, bits[18]:0x3_2006, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x3_a6cc, bits[18]:0x3_ffff, bits[18]:0x1_ffff]; bits[30]:0x40; bits[7]:0x55"
//     args: "[bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x2_98c5, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_aaaa]; bits[30]:0x1fff_ffff; bits[7]:0x47"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x2_6d87, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_0550, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0]; bits[30]:0x2aaa_aaaa; bits[7]:0x1d"
//     args: "[bits[18]:0x2000, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_faf6, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x54f8, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff]; bits[30]:0x0; bits[7]:0x8"
//     args: "[bits[18]:0x2_cff4, bits[18]:0x1_ffff, bits[18]:0x1, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x9109, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_e2ea, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x2_aaaa]; bits[30]:0x1555_5555; bits[7]:0x3f"
//     args: "[bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x8000, bits[18]:0x1_6f09, bits[18]:0x8, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_dd4d, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555]; bits[30]:0x8; bits[7]:0x0"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x400, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_f09b, bits[18]:0x2_aaaa, bits[18]:0x3_ffff]; bits[30]:0x2a97_69b3; bits[7]:0x55"
//     args: "[bits[18]:0xdbb, bits[18]:0x3_ffff, bits[18]:0x1, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x20, bits[18]:0x2_0000, bits[18]:0x1_5555, bits[18]:0x8, bits[18]:0x2_2044, bits[18]:0x400, bits[18]:0x1_ffff]; bits[30]:0x1fff_ffff; bits[7]:0x3b"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x100, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff]; bits[30]:0x2f28_f978; bits[7]:0x67"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1000, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x4000, bits[18]:0x1_ffff, bits[18]:0x2, bits[18]:0x2_aaaa, bits[18]:0x2000]; bits[30]:0x2aaa_aaaa; bits[7]:0x4"
//     args: "[bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_6c15, bits[18]:0x2_0000, bits[18]:0x40, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa]; bits[30]:0x245d_b09e; bits[7]:0x55"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x3_beab, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff]; bits[30]:0x2aaa_aaaa; bits[7]:0x1"
//     args: "[bits[18]:0x0, bits[18]:0x100, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x4000, bits[18]:0x200, bits[18]:0x1_7396, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_9236]; bits[30]:0x2aaa_aaaa; bits[7]:0x3f"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x8, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff]; bits[30]:0x200_0000; bits[7]:0x30"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x40, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x8000, bits[18]:0x1_5555, bits[18]:0x4000]; bits[30]:0x30c3_720a; bits[7]:0x55"
//     args: "[bits[18]:0x7971, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_5cff, bits[18]:0x1_1917, bits[18]:0x1_ffff, bits[18]:0x3_76a3, bits[18]:0x20, bits[18]:0x2000]; bits[30]:0x1555_5555; bits[7]:0x0"
//     args: "[bits[18]:0x2_eb7e, bits[18]:0x1_5555, bits[18]:0x2_27fe, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1000, bits[18]:0x2_aaaa, bits[18]:0x800, bits[18]:0x40, bits[18]:0x1_ffff, bits[18]:0x1000, bits[18]:0x2_aaaa, bits[18]:0x1_3996, bits[18]:0x1_5555]; bits[30]:0x1555_5555; bits[7]:0x7f"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2_e36b, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_91b4, bits[18]:0x3_ffff, bits[18]:0x4ea9]; bits[30]:0x251b_294a; bits[7]:0x4a"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x10, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_7d89, bits[18]:0x2_fecb, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x10, bits[18]:0x1_ffff, bits[18]:0x3_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x3_c08f, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x10, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1, bits[18]:0x1_ffff]; bits[30]:0x1bf5_122d; bits[7]:0x55"
//     args: "[bits[18]:0x1_5555, bits[18]:0x2_77bc, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x10, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x0]; bits[30]:0x1_0000; bits[7]:0x6e"
//     args: "[bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x2_ed5e, bits[18]:0x800, bits[18]:0x0, bits[18]:0x1_800d, bits[18]:0x40, bits[18]:0x1_5555, bits[18]:0x2_aaaa]; bits[30]:0x0; bits[7]:0x55"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x400, bits[18]:0x2_aaaa, bits[18]:0x400, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_aaaa]; bits[30]:0x0; bits[7]:0x10"
//     args: "[bits[18]:0x1_fe8a, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_ad4a, bits[18]:0x1_6230, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_5555]; bits[30]:0x1fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x8000, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x200, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x40, bits[18]:0x89ef, bits[18]:0x2_aaaa, bits[18]:0x1000, bits[18]:0x3_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x55"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_4290, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_9c39, bits[18]:0x1_ffff, bits[18]:0x4000, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x0]; bits[30]:0x2ee_af57; bits[7]:0x53"
//     args: "[bits[18]:0x400, bits[18]:0x2_aaaa, bits[18]:0x8, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_6f55, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555]; bits[30]:0x3fff_ffff; bits[7]:0x0"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x3_fbd3, bits[18]:0x4000, bits[18]:0x2_aaaa, bits[18]:0x1_7b39, bits[18]:0x20, bits[18]:0x2000, bits[18]:0x2_0138, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x100]; bits[30]:0x3fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x1_cafe, bits[18]:0x200, bits[18]:0x1_2a7c, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x40, bits[18]:0x0, bits[18]:0x8, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0]; bits[30]:0x200; bits[7]:0x55"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x800, bits[18]:0x80, bits[18]:0x3_d859, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1, bits[18]:0x200, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x69]; bits[30]:0x1555_5555; bits[7]:0x65"
//     args: "[bits[18]:0x3_d6ed, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x8, bits[18]:0x2_c5c3, bits[18]:0x3_ffff, bits[18]:0x50ed, bits[18]:0x2_aaaa, bits[18]:0x6d31, bits[18]:0x4, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0]; bits[30]:0x3fff_ffff; bits[7]:0x4"
//     args: "[bits[18]:0x74e8, bits[18]:0x3_ffff, bits[18]:0x10, bits[18]:0x1_5555, bits[18]:0x2_ae76, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_5555]; bits[30]:0x1555_5555; bits[7]:0x55"
//     args: "[bits[18]:0x1_2794, bits[18]:0x2_aaaa, bits[18]:0x1_fb3a, bits[18]:0x3_6ab0, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x335c, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2000, bits[18]:0x1_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x55"
//     args: "[bits[18]:0x4000, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_3592, bits[18]:0x4000, bits[18]:0x1000, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa]; bits[30]:0x1fff_ffff; bits[7]:0x7f"
//     args: "[bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x2_c59f, bits[18]:0x3_ffff, bits[18]:0x1, bits[18]:0x3_ffff, bits[18]:0x8, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2000, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_ffff]; bits[30]:0x2aaa_aaaa; bits[7]:0x1a"
//     args: "[bits[18]:0x20, bits[18]:0x3_d6a9, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_6523, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_2eb7, bits[18]:0x2_240d, bits[18]:0x2, bits[18]:0x10, bits[18]:0x1_5555, bits[18]:0x0]; bits[30]:0x1fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_3f4c, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_d037, bits[18]:0xb3e0, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_a5c6, bits[18]:0x1_ffff]; bits[30]:0x1000; bits[7]:0x7f"
//     args: "[bits[18]:0x2_a903, bits[18]:0x0, bits[18]:0x3_ca8e, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x8000, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x4000, bits[18]:0x1_5555, bits[18]:0x4]; bits[30]:0x2aaa_aaaa; bits[7]:0x6"
//     args: "[bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_4149, bits[18]:0xfab1, bits[18]:0x3_ffff, bits[18]:0x1_45b8, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff]; bits[30]:0x4_0000; bits[7]:0x0"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x3_2a01, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2, bits[18]:0x1, bits[18]:0x0, bits[18]:0x1_5555]; bits[30]:0x1555_5555; bits[7]:0x7f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x10, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_8f8d, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x4]; bits[30]:0x1bbe_43be; bits[7]:0x7f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x6647, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0xaf6f, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_988c, bits[18]:0x1_5555, bits[18]:0x3_3306]; bits[30]:0xb4d_61f3; bits[7]:0x40"
//     args: "[bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff]; bits[30]:0x1fff_ffff; bits[7]:0x76"
//     args: "[bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1, bits[18]:0x781b, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x80, bits[18]:0x1_5555]; bits[30]:0x0; bits[7]:0x7c"
//     args: "[bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x3_ffff]; bits[30]:0x80_0000; bits[7]:0x2a"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x2_1bf7, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_d586, bits[18]:0x800, bits[18]:0x1_c796, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa]; bits[30]:0x40_0000; bits[7]:0x2a"
//     args: "[bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_ade9, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x2_9a6c, bits[18]:0x1_ffff, bits[18]:0x4a9a, bits[18]:0x3_ffff]; bits[30]:0x1555_5555; bits[7]:0x6f"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x2_2dca, bits[18]:0x1_ffff, bits[18]:0x1_e80c, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1000, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x0, bits[18]:0x442b]; bits[30]:0x8000; bits[7]:0x4d"
//     args: "[bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x40, bits[18]:0x1_5555, bits[18]:0x3_2808, bits[18]:0x10, bits[18]:0x2, bits[18]:0x1_0000, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555]; bits[30]:0x0; bits[7]:0x16"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1000, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x76d0]; bits[30]:0x0; bits[7]:0x0"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0xa189, bits[18]:0x800, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x3_ebd8, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x2_ab60, bits[18]:0x3_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x0"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x100, bits[18]:0x0, bits[18]:0x2_2e7b, bits[18]:0x72b1, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_cd56, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x3_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x2a"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0xeaa4, bits[18]:0x3_5076, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x0]; bits[30]:0x0; bits[7]:0x3f"
//     args: "[bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x3_ef43, bits[18]:0xcdb5, bits[18]:0x1_ffff, bits[18]:0x4000, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x1_ffff]; bits[30]:0x0; bits[7]:0x18"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x2_0152, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x1_ffff, bits[18]:0xac94, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff]; bits[30]:0x3fff_ffff; bits[7]:0x55"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_440a, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x3_e107, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x100, bits[18]:0x0, bits[18]:0x100, bits[18]:0x3_ffff]; bits[30]:0x1fff_ffff; bits[7]:0x4"
//     args: "[bits[18]:0x3_ffff, bits[18]:0x2, bits[18]:0x9100, bits[18]:0x1, bits[18]:0x0, bits[18]:0x0, bits[18]:0x3_3729, bits[18]:0x0, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_f187, bits[18]:0x1_5555, bits[18]:0x1_5555, bits[18]:0x10]; bits[30]:0x1555_5555; bits[7]:0x1"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x400, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x1, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_5555]; bits[30]:0x3fff_ffff; bits[7]:0x0"
//     args: "[bits[18]:0x0, bits[18]:0x100, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_dd8b, bits[18]:0x2_aaaa, bits[18]:0x2_1f2a, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x1_5555]; bits[30]:0x2000; bits[7]:0x7f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x2_aaaa, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x10, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_5117, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0xde9, bits[18]:0x3_ffff]; bits[30]:0x0; bits[7]:0x55"
//     args: "[bits[18]:0x1_ffff, bits[18]:0x3_4162, bits[18]:0x1_0000, bits[18]:0x2, bits[18]:0x3_ffff, bits[18]:0x8, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x1_5555, bits[18]:0x2_aaaa, bits[18]:0x2_a00c]; bits[30]:0x1fff_ffff; bits[7]:0x3f"
//     args: "[bits[18]:0x3_f823, bits[18]:0x1_ffff, bits[18]:0x1_5555, bits[18]:0x8427, bits[18]:0x1_ffff, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x4000, bits[18]:0x1_5555, bits[18]:0x100, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_aaaa, bits[18]:0x3_ffff]; bits[30]:0x1555_5555; bits[7]:0x3f"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x80, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x0, bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0]; bits[30]:0x1fff_ffff; bits[7]:0x0"
//     args: "[bits[18]:0x2_aaaa, bits[18]:0x3_ffff, bits[18]:0x1_ffff, bits[18]:0x2_db3c, bits[18]:0x20, bits[18]:0x2_aaaa, bits[18]:0x1_5555, bits[18]:0x2_98b6, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x0, bits[18]:0x3_ffff, bits[18]:0x4, bits[18]:0x1_5555]; bits[30]:0x282d_cfec; bits[7]:0x1"
//     args: "[bits[18]:0x0, bits[18]:0x1_5555, bits[18]:0x3_ffff, bits[18]:0x3_ffff, bits[18]:0x40, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x0, bits[18]:0x0, bits[18]:0x2_aaaa, bits[18]:0x1_ffff, bits[18]:0x3_ffff, bits[18]:0x3_eac0, bits[18]:0x2_a417]; bits[30]:0x1fff_ffff; bits[7]:0x3f"
//   }
// }
// 
// END_CONFIG
const W32_V14 = u32:0xe;
type x0 = s18;
type x12 = u8;
type x22 = (bool, bool, bool, bool);
type x25 = u30;
fn x16(x17: x12) -> (bool, bool, bool, bool) {
    {
        let x18: bool = x17 < x17;
        let x19: bool = !x18;
        let x21: bool = {
            let x20: (bool, bool) = umulp(x19, x18);
            x20.0 + x20.1
        };
        (x18, x19, x21, x21)
    }
}
fn main(x1: x0[W32_V14], x2: s30, x3: u7) -> (xN[bool:0x0][1], x22[1818]) {
    {
        let x4: bool = or_reduce(x3);
        let x5: u7 = signex(x3, x3);
        let x6: s30 = x4 as s30 * x2;
        let x7: bool = xor_reduce(x5);
        let x8: bool = !x4;
        let x9: u7 = -x3;
        let x10: s30 = for (i, x) in u4:0x0..u4:0x2 {
            x
        }(x6);
        let x11: u30 = (x6 as u30)[0+:u30];
        let x13: x12[909] = "nbBXj=8_W*oBPnmi3AI3Pl>(\'-P?|;UD<jHw5R#P8uvgrB85xB|aF0<Vh@x[w;=ka\'O!%sc{oZB2!4:ok|/h,P8w yNZ9d1kYK9M.t}LTQ[5WI3wEQ8[ H{X,f=n>?$%_x\'g2rPc0k(\'W~2j{F$udaE2%N4hOh!U@ 7%G]?NU2%&%r}t;t_@4w,0M;5)yfiISl2!^8::\\I=x/2K:Q@\"KD1O10CEHc}%S|l>e&VKUhXX;#QY.c/l!4?cnMBd*(5(MYk-\"Fw. CxUP<NZkD*fkGt5zpV,p\\krCUl=7fqKJS2e{6ddhC[GyJue{v@}p`4wjTcL[t+^cJ{f],z!bGF`p3 HlO&|D[\"[Jx,%%Tl=9.N|4A!Wu;m\\+%@m\\)38,sWJfQHtOwGb,1 ZsI$g8x|!I]/hx2_12#rI~wQhkM1pupI(:@s09Vk6U<}ncMLwukk[/daCC/K?uL(c~w8}g|i$a+*l)uRL[CR|_G9a^U1<iQ:jc*MN3Xm<c|Rlfi-_][T.\'4wqwXx&H0F,l,\\r#_tx6N`i,\'aa(ZIj5A\'\'Diy}HO%}1A2xY|,B2n7`Ldi9o\"&a}*_r2t6YpxK=i)wS=*;/~*cjJm[{mr9T:+H8K#H;t,I*Geq^V>!Woh1+VDwdXNbL5cgc~QM=Pb]RHx2;6taL4wk~:!XvlgH51@:vv;mu_!\"q\"u8=V4IXP&A1TtB1#T!^Gt-}:WDd)im]`fF1)%LYk ~\'_G`\"0\\K_$f;8u\"^:\"sK$_4_v\"2CLU!G,!QRq>=h8a,ER9o<j}1285_f)T3a\":w%l1Cc]ax%8T/f?-oZ<Qt6#SV$R&KwZP4xXD+GEP_9JcAz%$ \"X\"H/|#3<r\'xKDd&?K@s;~K%!KLLGa+xD3yihh3w-evl1:@9[4 fuv|#TQ`_8)g\'R@Y#M9\"M";
        let x14: bool = x7 << x3;
        let x15: s30 = for (i, x): (u4, s30) in u4:0x0..=u4:0x6 {
            x
        }(x2);
        let x23: x22[909] = map(x13, x16);
        let x24: bool = x11 as bool - x7;
        let x26: x25[1] = [x11];
        let x27: bool = rev(x7);
        let x28: u5 = encode(x11);
        let x29: x22[1818] = x23 ++ x23;
        let x30: x22[1818] = x23 ++ x23;
        let x31: bool = x30 != x29;
        let x32: bool = x13 != x13;
        let x33: xN[bool:0x0][1] = x7[x4+:bool];
        let x34: xN[bool:0x0][1] = signex(x8, x33);
        let x35: bool = x32 ^ x2 as bool;
        (x34, x30)
    }
}
