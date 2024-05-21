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
// exception: 	 "/xls/tools/codegen_main returned non-zero exit status: 1"
// issue: "https://github.com/google/xls/issues/1141"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=6"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: false
//   use_system_verilog: true
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x22"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x40"
//       values: "bits[9]:0x10"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x40"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x40"
//       values: "bits[9]:0x91"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x184"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x40"
//       values: "bits[9]:0x1"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xd7"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x167"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x1"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x10"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x1"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x4"
//       values: "bits[9]:0x5b"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x11c"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x7f"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x8"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x2"
//       values: "bits[9]:0x4"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x181"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x100"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x2"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x155"
//       values: "bits[9]:0x1f0"
//       values: "bits[9]:0xe0"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0xaa"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0x4"
//       values: "bits[9]:0x80"
//       values: "bits[9]:0x80"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0xd2"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x1"
//       values: "bits[9]:0x1ff"
//       values: "bits[9]:0xff"
//       values: "bits[9]:0x0"
//       values: "bits[9]:0x8"
//     }
//     inputs {
//       channel_name: "sample__x29"
//       values: "bits[32]:0xff36_7200"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0x400"
//       values: "bits[32]:0x207f_ffff"
//       values: "bits[32]:0x855_1555"
//       values: "bits[32]:0xff01_6abe"
//       values: "bits[32]:0xd10b_7771"
//       values: "bits[32]:0x2"
//       values: "bits[32]:0x80_0000"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0xffba_aaaa"
//       values: "bits[32]:0x3299_a201"
//       values: "bits[32]:0x5555_5d55"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0xeb01_a046"
//       values: "bits[32]:0x2004_0000"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0xd672_4000"
//       values: "bits[32]:0x6dff_6edf"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x40_0000"
//       values: "bits[32]:0x551f_f7ff"
//       values: "bits[32]:0x7fa3_9abf"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0x200"
//       values: "bits[32]:0x387b_6dfa"
//       values: "bits[32]:0x2"
//       values: "bits[32]:0x7fa2_07f6"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x1000_0000"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0x4"
//       values: "bits[32]:0x941f_e7db"
//       values: "bits[32]:0x7fbf_ffff"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x1008_0548"
//       values: "bits[32]:0x4000"
//       values: "bits[32]:0x7dee_0496"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x807a_c3e0"
//       values: "bits[32]:0x8"
//       values: "bits[32]:0x405d_5cbd"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0xa_aaa9"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0x546c_03b9"
//       values: "bits[32]:0x7ca3_8ae2"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0x420_4296"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0xf964_7edc"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0xaaba_a0aa"
//       values: "bits[32]:0x153a_a322"
//       values: "bits[32]:0x23b_ff7f"
//       values: "bits[32]:0x2dbf_fdff"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0x91d5_5551"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0xaa80_0800"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0xffaa_62aa"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0x66d1_5a16"
//       values: "bits[32]:0x1b5b_3dd7"
//       values: "bits[32]:0xfabf_ff7f"
//       values: "bits[32]:0xe1d5_c545"
//       values: "bits[32]:0x8000"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x40_0000"
//       values: "bits[32]:0xc1c7_3ec4"
//       values: "bits[32]:0xc3e_bffa"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0xf40a_eee8"
//       values: "bits[32]:0xfe77_d11e"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0x855_5555"
//       values: "bits[32]:0x98a3_4e3d"
//       values: "bits[32]:0x400_0000"
//       values: "bits[32]:0xc3ae_baa6"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x4000"
//       values: "bits[32]:0x0"
//       values: "bits[32]:0x5d3a_8eed"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x2add_1f15"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x8adc_d3e5"
//       values: "bits[32]:0x6ffb_ffff"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0xe79c_bbad"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0x7faa_aaaa"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0xc186_ecb1"
//       values: "bits[32]:0x2"
//       values: "bits[32]:0x7fff_fdff"
//       values: "bits[32]:0x8000"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0xf93f_dffe"
//       values: "bits[32]:0xe2a5_f600"
//       values: "bits[32]:0xb0c1_bb1f"
//       values: "bits[32]:0xd769_e705"
//       values: "bits[32]:0xdff4_06b5"
//       values: "bits[32]:0x7f9d_bdf5"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x800"
//       values: "bits[32]:0x5001_0040"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x955e_a3ef"
//       values: "bits[32]:0x5555_5555"
//       values: "bits[32]:0x145_1410"
//       values: "bits[32]:0xffff_ffff"
//       values: "bits[32]:0x7fff_ffff"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0xaaaa_aaaa"
//       values: "bits[32]:0x4455_7d55"
//     }
//   }
// }
// 
// END_CONFIG
const W32_V5 = u32:5;
type x2 = u25;
type x4 = u5;
type x20 = u25;
fn x10(x11: token) -> (token, token) {
    {
        let x12: token = for (i, x): (u4, token) in u4:0x0..u4:0x6 {
            x
        }(x11);
        let x13: u21 = u21:0xa_aaaa;
        (x12, x12)
    }
}
proc main {
    x6: chan<u25> out;
    x22: chan<u9> in;
    x29: chan<u32> in;
    config(x6: chan<u25> out, x22: chan<u9> in, x29: chan<u32> in) {
        (x6, x22, x29)
    }
    init {
        u25:11184810
    }
    next(x1: u25) {
        {
            let x0: token = join();
            let x3: x2[1] = [x1];
            let x5: x4[W32_V5] = x1 as x4[W32_V5];
            let x7: token = send_if(x0, x6, bool:0x0, x1);
            let x8: u25 = for (i, x): (u4, u25) in u4:0b0..u4:0x1 {
                x
            }(x1);
            let x9: u25 = x8 << if x8 >= u25:0x3 { u25:0x3 } else { x8 };
            let x14: (token, token) = x10(x0);
            let x15: token = x14.0;
            let x16: token = x14.1;
            let x17: u25 = !x9;
            let x18: (u25, u25, u25, u25, u25, u25, x2[1], x2[1], x2[1], u25) = (x8, x17, x9, x8, x1, x9, x3, x3, x3, x9);
            let x19: bool = or_reduce(x9);
            let x21: x20[5] = [x1, x17, x8, x9, x17];
            let x23: (token, u9) = recv(x7, x22);
            let x24: token = x23.0;
            let x25: u9 = x23.1;
            let x26: u25 = bit_slice_update(x17, x25, x19);
            let x27: token = join(x15, x16, x24);
            let x28: bool = and_reduce(x26);
            let x30: (token, u32) = recv(x16, x29);
            let x31: token = x30.0;
            let x32: u32 = x30.1;
            let x33: bool = !x28;
            let x34: x4[10] = x5 ++ x5;
            let x35: bool = x19 | x19;
            let x36: u32 = x8 as u32 ^ x32;
            let x37: u1 = u1:false;
            let x38: u25 = x1 & x32 as u25;
            let x39: u25 = -x8;
            let x40: u32 = -x32;
            let x41: (u25, u1, bool, bool, x2[1], (u25, u25, u25, u25, u25, u25, x2[1], x2[1], x2[1], u25), u25) = (x9, x37, x19, x33, x3, x18, x8);
            x26
        }
    }
}
