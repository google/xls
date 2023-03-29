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
// In evaluated opt IR (JIT), channel 'sample__x14' has 127 entries. However, in evaluated unopt IR (JIT), channel 'sample__x14' has 0 entries.
//
// options: {"calls_per_sample": 0, "codegen": true, "codegen_args": ["--nouse_system_verilog", "--generator=pipeline", "--pipeline_stages=9", "--reset=rst", "--reset_active_low=false", "--reset_asynchronous=true", "--reset_data_path=true"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "proc_ticks": 128, "simulate": true, "simulator": "iverilog", "timeout_seconds": 1500, "top_type": 1, "use_jit": true, "use_system_verilog": false}
// ir_channel_names: sample__x5
// args: bits[51]:0x0
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x10_0000_0000
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x0
// args: bits[51]:0x8a10_0caf_2432
// args: bits[51]:0x0
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x0
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x7_fc21_706b_12ef
// args: bits[51]:0x1000_0000
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x2_aad9_3b4c_cc32
// args: bits[51]:0x1_4dd6_d9bf_210e
// args: bits[51]:0x4000_0000
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0xf9c_b3d1_3c7e
// args: bits[51]:0x0
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x8000_0000
// args: bits[51]:0x4c54_e70a_2b1e
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x10_0000_0000
// args: bits[51]:0x0
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x0
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x4_15ed_aff4_64ab
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x7_8741_dd2c_f385
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x3_c755_b844_c8fe
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x3_637d_7e41_b29a
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0xf61a_3641_f5d4
// args: bits[51]:0x0
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x5_a676_ef91_46d4
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x0
// args: bits[51]:0x10_0000_0000
// args: bits[51]:0x4000_0000
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x1
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x2000_0000_0000
// args: bits[51]:0x2_0000
// args: bits[51]:0x2
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x8_0000_0000
// args: bits[51]:0x0
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x8516_a68f_3ee7
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x7_49e0_47c0_283c
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x1_8cf4_e895_06d5
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x0
// args: bits[51]:0x1_0000
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x2_0000_0000_0000
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x400
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x0
// args: bits[51]:0x4000
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x3_ffff_ffff_ffff
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x0
// args: bits[51]:0x0
// args: bits[51]:0x2_aaaa_aaaa_aaaa
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x0
// args: bits[51]:0x5_5555_5555_5555
// args: bits[51]:0x7_ffff_ffff_ffff
// args: bits[51]:0x5_5555_5555_5555
proc main {
  x5: chan<u51> in;
  x14: chan<u20> out;
  config(x5: chan<u51> in, x14: chan<u20> out) {
    (x5, x14)
  }
  init {
    u20:512
  }
  next(x0: token, x1: u20) {
    let x2: u20 = signex(x1, x1);
    let x3: u60 = ((x1) ++ (x1)) ++ (x1);
    let x4: u20 = for (i, x): (u4, u20) in u4:0b0..u4:0x2 {
      x
    }(x1);
    let x6: (token, u51) = recv(x0, x5);
    let x7: token = x6.0;
    let x8: u51 = x6.1;
    let x9: bool = and_reduce(x1);
    let x10: u1 = u1:true;
    let x11: u20 = (x4)[0+:u20];
    let x12: u20 = bit_slice_update(x11, x11, x11);
    let x13: bool = (x8) == (((x12) as u51));
    let x15: token = send_if(x0, x14, x9, x4);
    let x16: token = x6.0;
    let x17: bool = (x13)[x9+:bool];
    let x18: u51 = rev(x8);
    let x19: token = join(x16, x7, x16);
    let x20: bool = (x12) <= (x11);
    let x21: u51 = (x8) | (((x17) as u51));
    let x22: token = join(x0, x7, x16, x7);
    x4
  }
}
