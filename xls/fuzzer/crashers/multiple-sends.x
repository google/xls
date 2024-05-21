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
//
// BEGIN_CONFIG
// exception: "// N/A"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=9"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--mutual_exclusion_z3_rlimit=1000000"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__input_channel"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x10_0000_0000"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x8a10_0caf_2432"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x7_fc21_706b_12ef"
//       values: "bits[64]:0x1000_0000"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x2_aad9_3b4c_cc32"
//       values: "bits[64]:0x1_4dd6_d9bf_210e"
//       values: "bits[64]:0x4000_0000"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0xf9c_b3d1_3c7e"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x8000_0000"
//       values: "bits[64]:0x4c54_e70a_2b1e"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x10_0000_0000"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x4_15ed_aff4_64ab"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x7_8741_dd2c_f385"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x3_c755_b844_c8fe"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x3_637d_7e41_b29a"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0xf61a_3641_f5d4"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x5_a676_ef91_46d4"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x10_0000_0000"
//       values: "bits[64]:0x4000_0000"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x1"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x2000_0000_0000"
//       values: "bits[64]:0x2_0000"
//       values: "bits[64]:0x2"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x8_0000_0000"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x8516_a68f_3ee7"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x7_49e0_47c0_283c"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x1_8cf4_e895_06d5"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x1_0000"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x2_0000_0000_0000"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x400"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x4000"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x3_ffff_ffff_ffff"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x2_aaaa_aaaa_aaaa"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x0"
//       values: "bits[64]:0x5_5555_5555_5555"
//       values: "bits[64]:0x7_ffff_ffff_ffff"
//       values: "bits[64]:0x5_5555_5555_5555"
//     }
//   }
// }
// END_CONFIG
proc main {
  input_channel: chan<u64> in;
  output_channel: chan<u64> out;
  config(input_channel: chan<u64> in, output_channel: chan<u64> out) {
    (input_channel, output_channel)
  }
  init {
    u64:1
  }
  next(state: u64) {
    let (tok, received) = recv(join(), input_channel);
    let rounded: u64 = u64:3 * (state / u64:3);
    let modulo: u64 = state - rounded;
    let x: token = send_if(tok, output_channel, modulo == u64:0, state + u64:5);
    let y: token = send_if(tok, output_channel, modulo == u64:1, state + u64:10);
    let z: token = send_if(tok, output_channel, modulo == u64:2, state + u64:15);
    let tok: token = join(x, y, z);
    (state + received)
  }
}
