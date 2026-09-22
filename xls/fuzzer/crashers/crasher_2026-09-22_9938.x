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
// exception: "/xls/tools/simulate_module_main returned a non-zero exit status (1): /xls/tools/simulate_module_main --signature_file=module_sig.textproto --testvector_textproto=testvector.pbtxt --verilog_simulator=iverilog sample.v --logtostderr\n\nSubprocess stderr:\nError: NOT_FOUND: Output `out`, instance #127 holds X value in Verilog simulator output.\n=== Source Location Trace: ===\nxls/simulation/module_testbench.cc:370\nxls/simulation/module_testbench.cc:456\nxls/simulation/module_simulator.cc:510\nxls/simulation/module_simulator.cc:550\nxls/tools/simulate_module_main.cc:208\n\n"
// issue: "DO NOT SUBMIT Insert link to GitHub issue here."
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
//   codegen_args: "--pipeline_stages=8"
//   codegen_args: "--worst_case_throughput=5"
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
//     args: "[bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x6_77ff, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555]; bits[6]:0x2a; bits[17]:0x1_4558"
//     args: "[bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x4_0000, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x5_5555]; bits[6]:0x10; bits[17]:0x1_a9ea"
//     args: "[bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x0, bits[20]:0x200, bits[20]:0x400, bits[20]:0x9_f6ff, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x5_5555]; bits[6]:0x1f; bits[17]:0xf5c1"
//     args: "[bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x3_ef6c, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0xb_dc06, bits[20]:0xc_3dd5, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x9_ce3f]; bits[6]:0x3f; bits[17]:0x0"
//     args: "[bits[20]:0x0, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x800, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x15; bits[17]:0x1_dac8"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x800, bits[20]:0x0, bits[20]:0x7_ffff]; bits[6]:0x2a; bits[17]:0x1"
//     args: "[bits[20]:0x0, bits[20]:0x800, bits[20]:0x7_ffff, bits[20]:0x1_1834, bits[20]:0x7_ffff, bits[20]:0xb_d6b8, bits[20]:0xa_aaaa, bits[20]:0x4_0000, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_4f5a]; bits[6]:0x0; bits[17]:0x10ef"
//     args: "[bits[20]:0x5_5555, bits[20]:0x2_0000, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x1000, bits[20]:0x9_d665, bits[20]:0x7_ffff, bits[20]:0x7_ffff]; bits[6]:0x2a; bits[17]:0x1_ffff"
//     args: "[bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x1_0000, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x4, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x1_233e, bits[20]:0x7_ffff]; bits[6]:0x0; bits[17]:0x0"
//     args: "[bits[20]:0x4, bits[20]:0xd_e8be, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x490, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x8_0000, bits[20]:0x5_5555, bits[20]:0xf_ffff]; bits[6]:0x2a; bits[17]:0x0"
//     args: "[bits[20]:0x0, bits[20]:0xf_5840, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xb_ac71, bits[20]:0xf_ffff, bits[20]:0x8000]; bits[6]:0x3f; bits[17]:0x0"
//     args: "[bits[20]:0xe_66e5, bits[20]:0x5_5555, bits[20]:0x8, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x4000, bits[20]:0xe_c129, bits[20]:0x5_5555, bits[20]:0xf_ffff]; bits[6]:0x0; bits[17]:0xffff"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x4, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x8, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xa_aaaa]; bits[6]:0x2a; bits[17]:0x1_74ef"
//     args: "[bits[20]:0x40, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x100, bits[20]:0x5_a6e5, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x1_38e8, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x5_5555]; bits[6]:0xe; bits[17]:0x4"
//     args: "[bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x0, bits[20]:0x1_4c3e, bits[20]:0xa_aaaa]; bits[6]:0x15; bits[17]:0xffff"
//     args: "[bits[20]:0x800, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x100, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_57c7, bits[20]:0xf_ffff, bits[20]:0x0]; bits[6]:0x9; bits[17]:0x0"
//     args: "[bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xa_aaaa]; bits[6]:0x0; bits[17]:0x555"
//     args: "[bits[20]:0xf_ffff, bits[20]:0xc_4206, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xa_95c8, bits[20]:0x0, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x0]; bits[6]:0x3f; bits[17]:0x1_ffff"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x2_a3e7, bits[20]:0x100, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xa_a0fa, bits[20]:0xa_aaaa, bits[20]:0x100, bits[20]:0xa_aaaa]; bits[6]:0x1; bits[17]:0x1_5555"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x400, bits[20]:0x2000, bits[20]:0x5_5555, bits[20]:0x3_ec91, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x545b, bits[20]:0x2_0000, bits[20]:0x7_ffff]; bits[6]:0x38; bits[17]:0x1_5555"
//     args: "[bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xf_ffff]; bits[6]:0x2a; bits[17]:0x1_0000"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x4, bits[20]:0x5_5555, bits[20]:0xa_aaaa]; bits[6]:0x2a; bits[17]:0x1_5555"
//     args: "[bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x2_0000, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x8_5eea, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x20]; bits[6]:0x0; bits[17]:0x0"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x8_0000, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xb_652c, bits[20]:0x5_5def, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x4_61e0, bits[20]:0x5_5555]; bits[6]:0x15; bits[17]:0x8381"
//     args: "[bits[20]:0x40, bits[20]:0x2, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x8_d9fb, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xc_bd03]; bits[6]:0x1f; bits[17]:0xe891"
//     args: "[bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xf_4fcc, bits[20]:0xf_ffff, bits[20]:0x400, bits[20]:0x0, bits[20]:0xd_85fe, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x5_5555]; bits[6]:0x1f; bits[17]:0xb917"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x9_9a78, bits[20]:0x2_71c3, bits[20]:0x1, bits[20]:0x0]; bits[6]:0x8; bits[17]:0xffff"
//     args: "[bits[20]:0x0, bits[20]:0x4_fa1f, bits[20]:0x1_0000, bits[20]:0x7_ffff, bits[20]:0x1_5dd0, bits[20]:0x2_2d89, bits[20]:0x80, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xe_870c, bits[20]:0x7_ffff]; bits[6]:0x1f; bits[17]:0x0"
//     args: "[bits[20]:0x1, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x1000, bits[20]:0x7_ffff]; bits[6]:0x1; bits[17]:0x1_ffff"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x3_5413, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0xa_38bd, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x0]; bits[6]:0x2; bits[17]:0xcd35"
//     args: "[bits[20]:0x4_0000, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x0, bits[20]:0xd_5833, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x8, bits[20]:0x800, bits[20]:0xa_aaaa, bits[20]:0x7_ffff]; bits[6]:0x8; bits[17]:0x1_5555"
//     args: "[bits[20]:0x2_0118, bits[20]:0xe_946d, bits[20]:0x1_2a90, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x2, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x4_0000]; bits[6]:0xb; bits[17]:0x1_4cc1"
//     args: "[bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x1, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x7_ffff]; bits[6]:0x3c; bits[17]:0xaaaa"
//     args: "[bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x2, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_da16, bits[20]:0x5_5555]; bits[6]:0x0; bits[17]:0x0"
//     args: "[bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x100, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x1000, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x7_ffff]; bits[6]:0x17; bits[17]:0x20"
//     args: "[bits[20]:0xe_0986, bits[20]:0x400, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x80, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x4_0000, bits[20]:0xa_aaaa, bits[20]:0x100, bits[20]:0xa_aaaa]; bits[6]:0x2a; bits[17]:0x4000"
//     args: "[bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x20, bits[20]:0x7_ffff, bits[20]:0x100, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x400, bits[20]:0x3a28]; bits[6]:0x1c; bits[17]:0xaa0b"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x8000, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x4, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x4_0000, bits[20]:0x0]; bits[6]:0x0; bits[17]:0x1_5555"
//     args: "[bits[20]:0xf_ffff, bits[20]:0xe_8325, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x8000, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff]; bits[6]:0x3f; bits[17]:0x1028"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x4, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0xa_8d03, bits[20]:0xf_ffff]; bits[6]:0x1f; bits[17]:0xaaaa"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x4000, bits[20]:0xf_ffff, bits[20]:0x4000, bits[20]:0x8000, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa]; bits[6]:0x3f; bits[17]:0x1_5555"
//     args: "[bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x8, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x1000, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x1f; bits[17]:0x8000"
//     args: "[bits[20]:0x800, bits[20]:0xf_ffff, bits[20]:0xa_1dbf, bits[20]:0xe_054c, bits[20]:0xc_7771, bits[20]:0x20, bits[20]:0x6_6664, bits[20]:0xf_ffff, bits[20]:0x2_0000, bits[20]:0x400, bits[20]:0x7_ffff]; bits[6]:0x3f; bits[17]:0x1_ffff"
//     args: "[bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x0, bits[20]:0xc_ed10]; bits[6]:0x2a; bits[17]:0xaaaa"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x2000, bits[20]:0x5_5555, bits[20]:0xa_e8b7, bits[20]:0x0, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x4_d7e4, bits[20]:0x9_687e, bits[20]:0xf_ffff]; bits[6]:0x1f; bits[17]:0x1_5555"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0xb_d005, bits[20]:0xf_e977, bits[20]:0x7_ffff, bits[20]:0xb_76ab, bits[20]:0x7_ffff]; bits[6]:0x2a; bits[17]:0xaaaa"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x3_04be, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x4_600d, bits[20]:0x7_ffff, bits[20]:0x5_5555]; bits[6]:0x0; bits[17]:0x1_1fc9"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0xf_40e9, bits[20]:0xf_ffff, bits[20]:0x5_25df, bits[20]:0x800, bits[20]:0x9_1b52, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x40]; bits[6]:0x3f; bits[17]:0x1_5555"
//     args: "[bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0xe_2f01, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x800, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x100, bits[20]:0x7_ffff, bits[20]:0x7_ffff]; bits[6]:0x2a; bits[17]:0x1_ffff"
//     args: "[bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x4000, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x80, bits[20]:0x4_0000, bits[20]:0xf_ffff]; bits[6]:0x6; bits[17]:0x1_e9f3"
//     args: "[bits[20]:0x800, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x800, bits[20]:0x7_ffff, bits[20]:0x2_3495, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xa_f3f6, bits[20]:0x0, bits[20]:0x0]; bits[6]:0x1; bits[17]:0x1_ffff"
//     args: "[bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xc_5615, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x9_60e4, bits[20]:0x6_f671, bits[20]:0x0]; bits[6]:0x24; bits[17]:0x1_5555"
//     args: "[bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x1_d2cc, bits[20]:0x7_65df, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x200, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x15; bits[17]:0xffff"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x9_1d95, bits[20]:0x80, bits[20]:0xe_4790, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x1_c9a7, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xf_ffff]; bits[6]:0x1f; bits[17]:0xaaaa"
//     args: "[bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xd_dbce, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x0]; bits[6]:0x2a; bits[17]:0x0"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x9_9f9b, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xd_da63, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x0; bits[17]:0xaaaa"
//     args: "[bits[20]:0x0, bits[20]:0x200, bits[20]:0x1_197c, bits[20]:0xa_aaaa, bits[20]:0x40, bits[20]:0xd_e222, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xc_3014, bits[20]:0x7_ffff]; bits[6]:0x0; bits[17]:0x8bbb"
//     args: "[bits[20]:0x100, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x200, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x10, bits[20]:0x2_a40d, bits[20]:0xa_aaaa, bits[20]:0x7_3e9d]; bits[6]:0x0; bits[17]:0x7fd"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0xe_031a, bits[20]:0x1000, bits[20]:0x7_ffff, bits[20]:0x1_efca]; bits[6]:0x1f; bits[17]:0xfa23"
//     args: "[bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x8_0000, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xf_ffff]; bits[6]:0x1f; bits[17]:0x4000"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x9_6fba, bits[20]:0xa_aaaa, bits[20]:0x7_fc27, bits[20]:0x3_eb2d, bits[20]:0x0]; bits[6]:0x15; bits[17]:0x30fa"
//     args: "[bits[20]:0x7_4d6e, bits[20]:0x7_ffff, bits[20]:0x3_0d1c, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xd_f65b, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x7_ffff]; bits[6]:0x1f; bits[17]:0x1_90ba"
//     args: "[bits[20]:0x0, bits[20]:0x0, bits[20]:0x7_207b, bits[20]:0xa_a0f6, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x8_ab08, bits[20]:0x5_5555]; bits[6]:0x3f; bits[17]:0x1_ffff"
//     args: "[bits[20]:0x5_5555, bits[20]:0x3_9bc3, bits[20]:0x8_9b3b, bits[20]:0xc_4041, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x5_4615, bits[20]:0xa_aaaa, bits[20]:0x7_ffff]; bits[6]:0x2a; bits[17]:0x1_11fe"
//     args: "[bits[20]:0x10, bits[20]:0x2000, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0xf_d138, bits[20]:0xa_b63b, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x6_0645]; bits[6]:0x15; bits[17]:0xaaaa"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x100, bits[20]:0x10, bits[20]:0x5_5555, bits[20]:0xb_104c, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0xa_aaaa]; bits[6]:0x3b; bits[17]:0x1_ffff"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x8_0000, bits[20]:0xf_ffff, bits[20]:0x8_bad2, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xf_ffff]; bits[6]:0x2a; bits[17]:0x1_5223"
//     args: "[bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x4000, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xa_259b, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x40]; bits[6]:0x2a; bits[17]:0x40a9"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x40, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x400]; bits[6]:0x3f; bits[17]:0x7f18"
//     args: "[bits[20]:0x1_0000, bits[20]:0xf_ffff, bits[20]:0x80, bits[20]:0x3_48a6, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x6_088e, bits[20]:0x0, bits[20]:0x4_0000, bits[20]:0x1_cf4d]; bits[6]:0x1f; bits[17]:0x0"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x6_a56e, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x20, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x5_5555]; bits[6]:0x0; bits[17]:0x1_f0f8"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x6_9c07, bits[20]:0x1000, bits[20]:0x40, bits[20]:0x7_8e56, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x8]; bits[6]:0xd; bits[17]:0x1_ffff"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x7_2c1a, bits[20]:0x6_5ddd, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x2, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xb_3bba]; bits[6]:0x2a; bits[17]:0x400"
//     args: "[bits[20]:0xe_19f6, bits[20]:0x1_0000, bits[20]:0x80, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0xc_ae70, bits[20]:0xf_ffff, bits[20]:0x40, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x9_2ce3]; bits[6]:0x3f; bits[17]:0x1_f975"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x5_97be, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x4_0000, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x4, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x15; bits[17]:0x80"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x8_0000, bits[20]:0x5_5555]; bits[6]:0x1; bits[17]:0x1_ffff"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x0, bits[20]:0x8000, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xf_ffff]; bits[6]:0x3f; bits[17]:0x800"
//     args: "[bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x1_192f, bits[20]:0xa_2500, bits[20]:0x5_4043, bits[20]:0xd_2538, bits[20]:0x7_ffff, bits[20]:0xc_ab93, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x5_5555]; bits[6]:0x2a; bits[17]:0x4"
//     args: "[bits[20]:0x5_5555, bits[20]:0x2, bits[20]:0xf_ffff, bits[20]:0x100, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xb_273a, bits[20]:0xa_aaaa, bits[20]:0x3_8705, bits[20]:0xa_aaaa, bits[20]:0x1_4097]; bits[6]:0x0; bits[17]:0x0"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xe_48a8, bits[20]:0x7_ffff, bits[20]:0x5_5555]; bits[6]:0x1f; bits[17]:0xffff"
//     args: "[bits[20]:0x5_5555, bits[20]:0xf_4065, bits[20]:0xa_aaaa, bits[20]:0x8_0000, bits[20]:0xd_b980, bits[20]:0x4_0000, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0xf_06a0, bits[20]:0x0]; bits[6]:0x2a; bits[17]:0xaaaa"
//     args: "[bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x8_0000, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x2_3e67, bits[20]:0x4_1f38, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x8]; bits[6]:0x2a; bits[17]:0x0"
//     args: "[bits[20]:0x4_0000, bits[20]:0x2_0000, bits[20]:0x0, bits[20]:0x0, bits[20]:0x8_be0a, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x0]; bits[6]:0x20; bits[17]:0xffff"
//     args: "[bits[20]:0x4000, bits[20]:0xf_ffff, bits[20]:0xb6e9, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x1, bits[20]:0xf_38e4, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x5_5555]; bits[6]:0x0; bits[17]:0xcf33"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x4, bits[20]:0x7_ffff, bits[20]:0xf_aa3f, bits[20]:0x5_5555, bits[20]:0x8_7422, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x800, bits[20]:0x0, bits[20]:0x5_5555]; bits[6]:0x1f; bits[17]:0x1_7d79"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x1_de87, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x2_131c, bits[20]:0x0, bits[20]:0x7_ffff]; bits[6]:0x0; bits[17]:0x1_5555"
//     args: "[bits[20]:0x4_55d1, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x400, bits[20]:0xc_856f, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x7_ffff]; bits[6]:0x1; bits[17]:0x0"
//     args: "[bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x15; bits[17]:0x2c53"
//     args: "[bits[20]:0xd_ac41, bits[20]:0x7_ffff, bits[20]:0x1000, bits[20]:0x4_2c3d, bits[20]:0xe_c696, bits[20]:0xa_aaaa, bits[20]:0x10, bits[20]:0x8, bits[20]:0x5_5555, bits[20]:0x1, bits[20]:0x8159]; bits[6]:0x15; bits[17]:0xbf7f"
//     args: "[bits[20]:0x20, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xb_5480, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x4000, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff]; bits[6]:0x36; bits[17]:0x1_5555"
//     args: "[bits[20]:0x0, bits[20]:0x7_c7b1, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xa_1fcc, bits[20]:0x8000, bits[20]:0xa0db, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x8_fb20]; bits[6]:0x38; bits[17]:0x0"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x4_0000, bits[20]:0x5_5555, bits[20]:0x1_0000, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x0]; bits[6]:0x3c; bits[17]:0xaaaa"
//     args: "[bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x4, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xa_d061, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x0]; bits[6]:0x1f; bits[17]:0x0"
//     args: "[bits[20]:0x8_6bce, bits[20]:0x0, bits[20]:0x9_1d1d, bits[20]:0x7e26, bits[20]:0x10, bits[20]:0x0, bits[20]:0x8_c7ea, bits[20]:0x1_265f, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x15; bits[17]:0x8970"
//     args: "[bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x5_5555]; bits[6]:0x3f; bits[17]:0xaaaa"
//     args: "[bits[20]:0x8000, bits[20]:0x2_675d, bits[20]:0x5_5555, bits[20]:0x2_0000, bits[20]:0x2, bits[20]:0xf_9407, bits[20]:0x400, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x0]; bits[6]:0x4; bits[17]:0xffff"
//     args: "[bits[20]:0x80, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x0]; bits[6]:0x15; bits[17]:0x3ccb"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x400, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x4_5639, bits[20]:0x0, bits[20]:0x100]; bits[6]:0x1f; bits[17]:0xdaaf"
//     args: "[bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x200, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0xd_d968, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x8_8ec5, bits[20]:0xf_ffff, bits[20]:0x7_ffff]; bits[6]:0x2; bits[17]:0x98d3"
//     args: "[bits[20]:0x4, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x9_7cb5, bits[20]:0x6_3b8b, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xa_aaaa]; bits[6]:0x0; bits[17]:0x15e0"
//     args: "[bits[20]:0x0, bits[20]:0xe_d4c8, bits[20]:0x6_a212, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0xe_0495, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x7_ffff]; bits[6]:0x0; bits[17]:0x21a2"
//     args: "[bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x40, bits[20]:0xa_aaaa, bits[20]:0x0]; bits[6]:0x15; bits[17]:0x1_5555"
//     args: "[bits[20]:0x0, bits[20]:0x10, bits[20]:0xc_aa1f, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x2_e49e, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x800]; bits[6]:0x0; bits[17]:0x1_ffff"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xb_ce45, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xa_a242, bits[20]:0xf_d262]; bits[6]:0x1f; bits[17]:0x0"
//     args: "[bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x5_532a, bits[20]:0x0, bits[20]:0xa_d130, bits[20]:0xc_6c5a, bits[20]:0x5_5555, bits[20]:0x9_89ff, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x9_11c6]; bits[6]:0x15; bits[17]:0xab6b"
//     args: "[bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xc_7bd9, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x5_1e4a, bits[20]:0x7_ffff, bits[20]:0x6_b2ed, bits[20]:0x7_ffff]; bits[6]:0x0; bits[17]:0x0"
//     args: "[bits[20]:0x2, bits[20]:0x8, bits[20]:0xa_aaaa, bits[20]:0xa_9bd4, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x20, bits[20]:0xf_0257, bits[20]:0x0, bits[20]:0xf_ffff]; bits[6]:0x3f; bits[17]:0x1_6b42"
//     args: "[bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x4_0000, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x5_cb85, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x5_5555]; bits[6]:0x2a; bits[17]:0x0"
//     args: "[bits[20]:0xc_d96d, bits[20]:0x0, bits[20]:0x1_cac0, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x2_0000, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x5_a73a]; bits[6]:0x0; bits[17]:0x1_5555"
//     args: "[bits[20]:0x9_8ebe, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x8_0000, bits[20]:0x7_ffff, bits[20]:0x2, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xf_ffff]; bits[6]:0x2a; bits[17]:0x624b"
//     args: "[bits[20]:0x0, bits[20]:0x1, bits[20]:0xa_aaaa, bits[20]:0x3_8e6a, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x0, bits[20]:0x5_5555]; bits[6]:0x1f; bits[17]:0xffff"
//     args: "[bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xc_536d, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff]; bits[6]:0x4; bits[17]:0x1_5555"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x9_f2ae, bits[20]:0xc6e8, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0x3_6da2, bits[20]:0xa_aaaa]; bits[6]:0x15; bits[17]:0x0"
//     args: "[bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x2, bits[20]:0x5_5555, bits[20]:0xd_0dff, bits[20]:0x5_5555, bits[20]:0x7_c104]; bits[6]:0x2; bits[17]:0x1_ffff"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0xe_d0c6, bits[20]:0x2, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0x6_db19, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x8_9628]; bits[6]:0x10; bits[17]:0x4000"
//     args: "[bits[20]:0x7_ffff, bits[20]:0xa_a749, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0x4000, bits[20]:0xb_bf34, bits[20]:0x0, bits[20]:0xa_aaaa]; bits[6]:0x15; bits[17]:0x8aa3"
//     args: "[bits[20]:0xf_a9db, bits[20]:0x7_ffff, bits[20]:0x5_5555, bits[20]:0x20, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x8, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0xa_6ec4]; bits[6]:0x34; bits[17]:0x1_425e"
//     args: "[bits[20]:0xf_7f6a, bits[20]:0xf_ffff, bits[20]:0xf_ffff, bits[20]:0xa_3394, bits[20]:0x3_0146, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x20, bits[20]:0x0, bits[20]:0x4_0000]; bits[6]:0x3f; bits[17]:0xffff"
//     args: "[bits[20]:0x0, bits[20]:0x8_29f1, bits[20]:0xf_ffff, bits[20]:0x8000, bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0xe_8006, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xf_ffff]; bits[6]:0x15; bits[17]:0x1_d240"
//     args: "[bits[20]:0x7_ffff, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0xa_1062, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x400, bits[20]:0x7_ffff]; bits[6]:0x15; bits[17]:0xa844"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0x4, bits[20]:0x5_5555, bits[20]:0x10, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x0, bits[20]:0x5_5555]; bits[6]:0x0; bits[17]:0x1_ffff"
//     args: "[bits[20]:0xa_9745, bits[20]:0x1, bits[20]:0x0, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0xa_aaaa, bits[20]:0x20, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_ffff, bits[20]:0x1]; bits[6]:0xb; bits[17]:0x1_f853"
//     args: "[bits[20]:0xf_ffff, bits[20]:0x6_d3bb, bits[20]:0xa_aaaa, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x3_11bd, bits[20]:0x7_ffff, bits[20]:0x400, bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0xf_ffff]; bits[6]:0x2a; bits[17]:0x0"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x8_0422, bits[20]:0x7_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x5_5555, bits[20]:0x1000, bits[20]:0xa_aaaa, bits[20]:0x1_0d47, bits[20]:0xf_ffff, bits[20]:0x0]; bits[6]:0x0; bits[17]:0xffff"
//     args: "[bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xf_ffff, bits[20]:0x2, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0xa_aaaa, bits[20]:0x5_5555]; bits[6]:0x0; bits[17]:0x400"
//     args: "[bits[20]:0x5_5555, bits[20]:0xf_ffff, bits[20]:0xc_bd7b, bits[20]:0x5_5555, bits[20]:0x0, bits[20]:0x0, bits[20]:0x7_3033, bits[20]:0x0, bits[20]:0x7_ffff, bits[20]:0x80, bits[20]:0x5_5555]; bits[6]:0x34; bits[17]:0x1_a323"
//     args: "[bits[20]:0x2, bits[20]:0x5_77b8, bits[20]:0x5_5555, bits[20]:0xa_aaaa, bits[20]:0xf_ffff, bits[20]:0x7_ffff, bits[20]:0xa_aaaa, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0xf_5546, bits[20]:0x5_c693]; bits[6]:0x15; bits[17]:0x1_7fbb"
//     args: "[bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x0, bits[20]:0x5_5555, bits[20]:0x7_ffff, bits[20]:0x6_b89e, bits[20]:0xa_aaaa, bits[20]:0x0, bits[20]:0x1_0000, bits[20]:0x5_5555, bits[20]:0xa_aaaa]; bits[6]:0x3; bits[17]:0x18ae"
//   }
// }
//
// END_CONFIG
const W32_V11 = u32:0xb;
type x0 = u20;
fn x6(x7: x0[W32_V11], x8: u6, x9: x0) -> (bool, x0, x0) {
    {
        let x10: bool = x8[x8+:bool];
        let x11: bool = -x10;
        (x11, x9, x9)
    }
}
fn main(x1: x0[W32_V11], x2: u6, x3: u17) -> u6 {
    {
        let x4: u6 = -x2;
        let x5: x0 = x1[if x2 >= u6:0x0 { u6:0x0 } else { x2 }];
        let x12: (bool, x0, x0) = x6(x1, x4, x5);
        let x13: x0[22] = x1 ++ x1;
        let x14: u5 = encode(x5);
        let x15: u47 = u47:0x200_0000;
        let x16: u6 = x4[:];
        let x17: x0 = x13[if x2 >= u6:0xb { u6:0xb } else { x2 }];
        let x18: u5 = -x14;
        let x19: u5 = x14[x2+:u5];
        let x20: u2 = u2:0x1;
        let x21: x0 = bit_slice_update(x17, x16, x19);
        let x22: u5 = x14 / u5:0x2;
        let x23: u5 = x15[x17+:u5];
        let x24: u5 = x18 | x4 as u5;
        let x25: u20 = rev(x17);
        let x26: x0 = x13[if x25 >= u20:0x12 { u20:0x12 } else { x25 }];
        let x27: x0[22] = update(x13, if x22 >= u5:0x13 { u5:0x13 } else { x22 }, x26);
        let x28: u6 = match x19 {
            xN[bool:0x0][5]:0x0..xN[bool:0x0][5]:4 => u6:0x1f,
            u5:0x0..u5:12 | u5:0x2 => u6:0x0,
            u5:0x0 => u6:0x15,
            u5:0x8 => x4,
            _ => u6:0x4,
        };
        x28
    }
}
