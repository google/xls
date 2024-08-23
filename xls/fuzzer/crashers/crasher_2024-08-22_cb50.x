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
// exception: "In evaluated opt IR (JIT), at position 1 channel \'sample__x13\' has value u21:1398101. However, in evaluated unopt IR (JIT), the value is u21:0."
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 0
//   proc_ticks: 100
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
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x43"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0x1c1a_1148_6f2f_3b19"
//       values: "bits[61]:0x1_0000_0000_0000"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0x1e96_853e_d1ca_ec18"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0xdce_eb71_d2ca_16bf"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x80_f7d2_b56a_b6d5"
//       values: "bits[61]:0xaed_e020_3be1_d8c6"
//       values: "bits[61]:0x8_0000_0000"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x1ffa_e551_19f5_ca11"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0x2d7_7b72_850b_38de"
//       values: "bits[61]:0x2000_0000_0000"
//       values: "bits[61]:0x400_0000_0000"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x200"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x159_8956_f69e_b97c"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x170a_c70d_b962_779d"
//       values: "bits[61]:0x1400_32a4_f5a9_c2a8"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x1064_e2be_c547_349b"
//       values: "bits[61]:0x1_0000"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x1bb1_a350_fd23_374a"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x80_0000_0000_0000"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0x4"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x102b_d5ad_8657_ae99"
//       values: "bits[61]:0x2_0000_0000_0000"
//       values: "bits[61]:0x57b_930e_2093_9266"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0xe60_98fa_4561_82a0"
//       values: "bits[61]:0xaaa_aaaa_aaaa_aaaa"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x312_9dbd_8976_66ce"
//       values: "bits[61]:0x180e_b603_1eba_428e"
//       values: "bits[61]:0x20_0000_0000"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0xcf9_817d_0ed2_f2d3"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x40"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x800_0000_0000_0000"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x1555_5555_5555_5555"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x1fff_ffff_ffff_ffff"
//       values: "bits[61]:0x0"
//       values: "bits[61]:0x100_0000_0000_0000"
//       values: "bits[61]:0xfff_ffff_ffff_ffff"
//       values: "bits[61]:0x1555_5555_5555_5555"
//     }
//   }
// }
// 
// END_CONFIG
const W32_V8 = u32:0x8;
type x34 = u41;
fn x20<x26: u33 = {u33:0x1_5555_5555}, x27: u41 = {u41:0x155_5555_5555}, x28: u6 = {u6:0x0}, x29: u43 = {u43:0x0}>(x21: token, x22: bool, x23: bool, x24: u48, x25: u54) -> (u41, u33, u6) {
    {
        let x30: (u43, u33, u33, u48) = (x29, x26, x26, x24);
        let x32: s54 = {
            let x31: (u54, u54) = smulp(x26 as u54 as s54, x25 as s54);
            (x31.0 + x31.1) as s54
        };
        (x27, x26, x28)
    }
}
proc main {
    x13: chan<u21> out;
    x43: chan<u61> in;
    config(x13: chan<u21> out, x43: chan<u61> in) {
        (x13, x43)
    }
    init {
        u54:12009599006321322
    }
    next(x0: u54) {
        {
            let x1: u54 = one_hot_sel(u5:0xf, [x0, x0, x0, x0, x0]);
            let x2: u21 = x0[-21:];
            let x3: u21 = -x2;
            let x4: u54 = -x0;
            let x5: u21 = clz(x3);
            let x6: bool = and_reduce(x3);
            let x7: u57 = decode<u57>(x1);
            let x8: u48 = x4[x6+:u48];
            let x9: u54 = match x1 {
                u54:0x0 => u54:0b0,
                u54:0x3f_ffff_ffff_ffff => u54:0x800,
                u54:0x8000 => u54:0x3f_ffff_ffff_ffff,
                _ => u54:0x0,
            };
            let x10: u48 = x9[0+:u48];
            let x11: u54 = x1 * x9 as u54;
            let x12: bool = xor_reduce(x0);
            let x14: token = send(join(), x13, x2);
            let x15: bool = and_reduce(x6);
            let x17: u48 = {
                let x16: (u48, u48) = umulp(x10, x1 as u48);
                x16.0 + x16.1
            };
            let x18: u10 = x3[0+:u10];
            let x19: u54 = clz(x9);
            let x33: (u41, u33, u6) = x20(x14, x12, x12, x8, x9);
            let x35: token = join();
            let x36: token = join(x35, x35);
            let x37: u21 = bit_slice_update(x5, x8, x9);
            let x38: u21 = !x37;
            let x39: u54 = x11[x7+:u54];
            let x40: u48 = x10 << if x11 >= u54:0x2d { u54:0x2d } else { x11 };
            let x41: u48 = -x17;
            let x42: u48 = x12 as u48 + x17;
            let x44: (token, u61) = recv(x36, x43);
            let x45: token = x44.0;
            let x46: u61 = x44.1;
            let x47: u3 = u3:0x2;
            let x48: u10 = match x15 {
                bool:0x1..bool:true | bool:false => u10:0x217,
                bool:0x1 => x18,
                _ => u10:682,
            };
            x9
        }
    }
}
