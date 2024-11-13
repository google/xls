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
// exception: "In evaluated opt IR (JIT), at position 1 channel \'sample__x15\' has value u19:349525. However, in simulated, the value is u19:0"
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
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--flop_inputs=true"
//   codegen_args: "--flop_inputs_kind=zerolatency"
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
//       channel_name: "sample__x10"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x2000"
//       values: "bits[19]:0x2_d8a2"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x7_9fb1"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x8000"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x2"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x2_f96b"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x4000"
//       values: "bits[19]:0xe01a"
//       values: "bits[19]:0x3_9fd9"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x80"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x4_1747"
//       values: "bits[19]:0x10"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x2_0000"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x100"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x2_3d70"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x1_0000"
//       values: "bits[19]:0xf294"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x2000"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x1_80ec"
//       values: "bits[19]:0x2000"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x100"
//       values: "bits[19]:0xca5d"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x20"
//       values: "bits[19]:0x4"
//       values: "bits[19]:0x1000"
//       values: "bits[19]:0x10"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x5_150f"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x1"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x2"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x100"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x800"
//       values: "bits[19]:0x3_ffff"
//       values: "bits[19]:0x5_f80d"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x2_aed1"
//       values: "bits[19]:0x7_56f8"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x1000"
//       values: "bits[19]:0x1_288d"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x800"
//       values: "bits[19]:0x3_8081"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x3_425f"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x1_0000"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x9159"
//       values: "bits[19]:0x0"
//       values: "bits[19]:0x7_c8d9"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x7_ffff"
//       values: "bits[19]:0x2_aaaa"
//       values: "bits[19]:0x800"
//       values: "bits[19]:0x5_5555"
//       values: "bits[19]:0x1_3868"
//     }
//   }
// }
//
// END_CONFIG
type x4 = u19;
proc main {
  x10: chan<u19> in;
  x15: chan<u19> out;
  config(x10: chan<u19> in, x15: chan<u19> out) {
    (x10, x15)
  }
  init {
    u30:715827882
  }
  next(x1: u30) {
    {
      let x0: token = join();
      let x2: u19 = x1[11+:u19];
      let x3: u19 = x2 | x1 as u19;
      let x5: x4[7] = [x2, x3, x3, x2, x3, x2, x3];
      let x6: bool = x5 != x5;
      let x7: u19 = x1 as u19 | x2;
      let x8: u19 = x6 as u19 + x7;
      let x9: token = join(x0);
      let x11: (token, u19, bool) = recv_non_blocking(x0, x10, x3);
      let x12: token = x11.0;
      let x13: u19 = x11.1;
      let x14: bool = x11.2;
      let x16: token = send(x0, x15, x2);
      let x17: x4[1] = array_slice(x5, x7, x4[1]:[x5[u32:0x0], ...]);
      let x18: u30 = for (i, x): (u4, u30) in u4:0x0..u4:0x8 {
        x
      }(x1);
      let x19: u19 = -x8;
      let x20: u30 = x14 as u30 * x18;
      let x22: s19 = {
        let x21: (u19, u19) = smulp(x18 as u19 as s19, x13 as s19);
        (x21.0 + x21.1) as s19
      };
      let x23: u19 = bit_slice_update(x2, x14, x2);
      let x24: x4[8] = x17 ++ x5;
      let x25: bool = x2[x14+:bool];
      let x26: u19 = !x19;
      let x27: s14 = match x7 {
        u19:0x10 => s14:0x2aaa,
        _ => s14:0x1555,
      };
      let x28: bool = signex(x2, x25);
      let x29: u30 = ctz(x20);
      let x30: s19 = x22 * x7 as s19;
      let x31: u19 = one_hot_sel(x14, [x26]);
      let x32: u15 = (x30 as u19)[x25+:u15];
      let x34: s19 = {
        let x33: (u19, u19) = smulp(x3 as s19, x7 as s19);
        (x33.0 + x33.1) as s19
      };
      let x35: u19 = -x3;
      x20
    }
  }
}
