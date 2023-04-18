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

// Exception:
// N/A
//
// options: {"calls_per_sample": 0, "codegen": true, "codegen_args": ["--nouse_system_verilog", "--generator=pipeline", "--pipeline_stages=9", "--reset=rst", "--reset_active_low=false", "--reset_asynchronous=true", "--reset_data_path=true", "--mutual_exclusion_z3_rlimit=1000000"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "proc_ticks": 128, "simulate": true, "simulator": "iverilog", "timeout_seconds": 1500, "top_type": 1, "use_jit": true, "use_system_verilog": false}
// ir_channel_names: sample__input_channel
// args: bits[64]:0x0
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x10_0000_0000
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x0
// args: bits[64]:0x8a10_0caf_2432
// args: bits[64]:0x0
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x0
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x7_fc21_706b_12ef
// args: bits[64]:0x1000_0000
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x2_aad9_3b4c_cc32
// args: bits[64]:0x1_4dd6_d9bf_210e
// args: bits[64]:0x4000_0000
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0xf9c_b3d1_3c7e
// args: bits[64]:0x0
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x8000_0000
// args: bits[64]:0x4c54_e70a_2b1e
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x10_0000_0000
// args: bits[64]:0x0
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x0
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x4_15ed_aff4_64ab
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x7_8741_dd2c_f385
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x3_c755_b844_c8fe
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x3_637d_7e41_b29a
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0xf61a_3641_f5d4
// args: bits[64]:0x0
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x5_a676_ef91_46d4
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x0
// args: bits[64]:0x10_0000_0000
// args: bits[64]:0x4000_0000
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x1
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x2000_0000_0000
// args: bits[64]:0x2_0000
// args: bits[64]:0x2
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x8_0000_0000
// args: bits[64]:0x0
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x8516_a68f_3ee7
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x7_49e0_47c0_283c
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x1_8cf4_e895_06d5
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x0
// args: bits[64]:0x1_0000
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x2_0000_0000_0000
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x400
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x0
// args: bits[64]:0x4000
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x3_ffff_ffff_ffff
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x0
// args: bits[64]:0x0
// args: bits[64]:0x2_aaaa_aaaa_aaaa
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x0
// args: bits[64]:0x5_5555_5555_5555
// args: bits[64]:0x7_ffff_ffff_ffff
// args: bits[64]:0x5_5555_5555_5555
proc main {
  input_channel: chan<u64> in;
  output_channel: chan<u64> out;
  config(input_channel: chan<u64> in, output_channel: chan<u64> out) {
    (input_channel, output_channel)
  }
  init {
    u64:1
  }
  next(tok: token, state: u64) {
    let (tok, received) = recv(tok, input_channel);
    let rounded: u64 = u64:3 * (state / u64:3);
    let modulo: u64 = state - rounded;
    let x: token = send_if(tok, output_channel, modulo == u64:0, state + u64:5);
    let y: token = send_if(tok, output_channel, modulo == u64:1, state + u64:10);
    let z: token = send_if(tok, output_channel, modulo == u64:2, state + u64:15);
    let tok: token = join(x, y, z);
    (state + received)
  }
}
