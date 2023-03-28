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
// Command '['xls/tools/eval_proc_main', '--inputs_for_all_channels=channel_inputs.txt', '--ticks=128', '--backend=serial_jit', 'sample.ir', '--logtostderr']' died with <Signals.SIGSEGV: 11>.
// Issue: https://github.com/llvm/llvm-project/issues/61038.
//
// options: {"calls_per_sample": 0, "codegen": true, "codegen_args": ["--nouse_system_verilog", "--generator=pipeline", "--pipeline_stages=5", "--reset=rst", "--reset_active_low=false", "--reset_asynchronous=true", "--reset_data_path=true"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "proc_ticks": 128, "simulate": true, "simulator": "iverilog", "timeout_seconds": 1500, "top_type": 1, "use_jit": true, "use_system_verilog": false}
// ir_channel_names: sample__x25
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x14e_d44e_9db6
// args: bits[42]:0x0
// args: bits[42]:0x40_0000_0000
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x10_0000_0000
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x3e6_ddb7_8e5e
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x20
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x800
// args: bits[42]:0x222_df8c_bc74
// args: bits[42]:0x178_3c24_b717
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x0
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x200_0000
// args: bits[42]:0x1000_0000
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x14d_a0b8_34b7
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x2
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x200_0000
// args: bits[42]:0x0
// args: bits[42]:0x4_0000
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x371_9b49_100e
// args: bits[42]:0x66_d211_d897
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x2d6_2874_c2a1
// args: bits[42]:0x1cb_eb41_6213
// args: bits[42]:0x44_fd4d_166f
// args: bits[42]:0xe1_5471_11c5
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x1000_0000
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x100_0000_0000
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0xb8_1f51_57a9
// args: bits[42]:0x2
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x0
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x80_0000
// args: bits[42]:0x0
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x0
// args: bits[42]:0x400_0000
// args: bits[42]:0x0
// args: bits[42]:0x0
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x8c_846e_4d75
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x2_0000_0000
// args: bits[42]:0x2000_0000
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x2e9_ac57_3f20
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x170_c130_ecb0
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x2a5_a744_4fec
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x2a2_d274_9e49
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x0
// args: bits[42]:0x0
// args: bits[42]:0x4_0000
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x152_1db8_0f91
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x80_0000
// args: bits[42]:0x0
// args: bits[42]:0x0
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x6b_1ee1_f443
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x355_7091_f2b5
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x1ff_ffff_ffff
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x3ff_ffff_ffff
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x7a_a793_ce58
// args: bits[42]:0x2aa_aaaa_aaaa
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x155_5555_5555
// args: bits[42]:0x2aa_aaaa_aaaa
proc main {
  x25: chan<u42> in;
  config(x25: chan<u42> in) {
    (x25,)
  }
  init {
    u32:2147483647
  }
  next(x0: token, x1: u32) {
    let x2: u32 = !(x1);
    let x3: u32 = bit_slice_update(x2, x1, x1);
    let x4: u33 = one_hot(x3, bool:0x1);
    let x5: u16 = (x3)[:16];
    let x6: u32 = bit_slice_update(x3, x2, x5);
    let x7: u32 = rev(x2);
    let x8: u6 = (x5)[-11:11];
    let x9: bool = (((x8) as u16)) >= (x5);
    let x10: u16 = !(x5);
    let x11: u33 = !(x4);
    let x12: u34 = u34:0x1_ffff_ffff;
    let x13: u34 = (x12)[x11+:u34];
    let x14: u6 = -(x8);
    let x15: u33 = (x4)[:];
    let x16: u16 = (x5) >> (x10);
    let x18: u32 = {
      let x17: (u32, u32) = umulp(((x13) as u32), x7);
      (x17.0) + (x17.1)
    };
    let x19: u33 = rev(x4);
    let x20: bool = (x5)[x10+:bool];
    let x21: u16 = clz(x5);
    let x22: u6 = -(x14);
    let x23: u26 = u26:0;
    let x24: u28 = (x15)[x10+:u28];
    let x26: (token, u42) = recv(x0, x25);
    let x27: token = x26.0;
    let x28: u42 = x26.1;
    let x29: u16 = (x16) ^ (((x13) as u16));
    let x30: u16 = -(x21);
    x18
  }
}
