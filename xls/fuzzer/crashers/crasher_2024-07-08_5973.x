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
// exception: "/xls/tools/opt_main returned non-zero exit status (1): /xls/tools/opt_main /tmp/sample.ir --logtostderr"
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
//   codegen_args: "--pipeline_stages=5"
//   codegen_args: "--worst_case_throughput=4"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
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
//       channel_name: "sample__x6"
//       values: "bits[14]:0x100"
//       values: "bits[14]:0x1b4b"
//       values: "bits[14]:0x1000"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x10"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x13c4"
//       values: "bits[14]:0x40"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x2000"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x3d8d"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x9d2"
//       values: "bits[14]:0x2000"
//       values: "bits[14]:0x2"
//       values: "bits[14]:0x154d"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x100"
//       values: "bits[14]:0x100"
//       values: "bits[14]:0x2280"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x2000"
//       values: "bits[14]:0x8"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x1c85"
//       values: "bits[14]:0x16cc"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1b7d"
//       values: "bits[14]:0x100"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0xcca"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x20"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x100"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x40"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x3fb7"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x1"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x2000"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x100"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x400"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x8"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1"
//       values: "bits[14]:0x80"
//       values: "bits[14]:0xb0e"
//       values: "bits[14]:0x200"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x19ab"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x2000"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x800"
//       values: "bits[14]:0xdbe"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x39a7"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x2000"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x10"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x7d5"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x3224"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x0"
//       values: "bits[14]:0x1fff"
//       values: "bits[14]:0x3fff"
//       values: "bits[14]:0x2aaa"
//       values: "bits[14]:0x1555"
//       values: "bits[14]:0x1000"
//     }
//   }
// }
// 
// END_CONFIG
const W32_V10 = u32:0xa;
type x1 = u58;
type x28 = uN[189];
type x36 = s40;
type x37 = x36[7];
type x44 = uN[189];
fn x22(x23: bool, x24: s10, x25: u10) -> (bool, bool, bool, bool) {
    {
        let x26: bool = x24 as bool - x23;
        (x26, x26, x26, x26)
    }
}
fn x31(x32: bool, x33: x1[1], x34: (bool, bool, bool, bool)) -> (x37[W32_V10], x1[1], x1[1], x1[1]) {
    {
        let x35: x1[1] = array_slice(x33, x32, x1[1]:[x33[u32:0x0], ...]);
        let x38: x37[W32_V10] = match x34 {
            (bool:true, bool:0b1, bool:false, bool:false) | (bool:false, _, bool:0x1, bool:true) => [[s40:0x55_5555_5555, s40:0x7f_ffff_ffff, s40:0x0, s40:0x3a_9e43_35de, s40:0xaa_aaaa_aaaa, s40:0x0, s40:0x7f_ffff_ffff], [s40:0xaa_aaaa_aaaa, s40:0xae_353d_212c, s40:0x55_5555_5555, s40:0x8, s40:0x0, s40:0xff_ffff_ffff, s40:0x0], [s40:0xff_ffff_ffff, s40:0x7f_ffff_ffff, s40:0xaa_aaaa_aaaa, s40:0x4000_0000, s40:0x7f_ffff_ffff, s40:0xff_ffff_ffff, s40:68719476736], [s40:0xaa_aaaa_aaaa, s40:0x2_0000, s40:0xff_ffff_ffff, s40:0x7f_ffff_ffff, s40:0x7f_ffff_ffff, s40:0x800, s40:0xaa_aaaa_aaaa], [s40:0xaa_aaaa_aaaa, s40:0x8, s40:0x0, s40:0x8_0000, s40:0x0, s40:0x55_5555_5555, s40:0xff_ffff_ffff], [s40:0x4000, s40:0x800, s40:0xff_ffff_ffff, s40:0x0, s40:0xff_ffff_ffff, s40:0xff_ffff_ffff, s40:0x73_bfe3_6b51], [s40:0x0, s40:0x4, s40:0x0, s40:0x7f_ffff_ffff, s40:0x7f_ffff_ffff, s40:0xff_ffff_ffff, s40:0x0], [s40:0x55_5555_5555, s40:0x55_5555_5555, s40:0x200, s40:0b1_0100_1110_0001_0001_1110_1000_1001_0011_1011, s40:0x7f_ffff_ffff, s40:0x0, s40:0x0], [s40:0x55_5555_5555, s40:0x55_5555_5555, s40:0x7f_ffff_ffff, s40:0x8000_0000, s40:0xaa_aaaa_aaaa, s40:0x7f_ffff_ffff, s40:0x55_5555_5555], [s40:0xbc_283a_7f3a, s40:0xaa_aaaa_aaaa, s40:0xbd_fb0f_830d, s40:0xaa_aaaa_aaaa, s40:0x10_0000_0000, s40:0x55_5555_5555, s40:0x76_5a51_e40f]],
            _ => [[s40:0xff_ffff_ffff, s40:0x55_5555_5555, s40:0xff_ffff_ffff, s40:0x77_60c8_4f0d, s40:0xff_ffff_ffff, s40:0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111, s40:0b101_0101_0101_0101_0101_0101_0101_0101_0101_0101], [s40:0xaa_aaaa_aaaa, s40:-238053065452, s40:0x40_0000_0000, s40:0x2_0000, s40:0x55_5555_5555, s40:0xaa_aaaa_aaaa, s40:0x8_0000_0000], [s40:0xaa_aaaa_aaaa, s40:0x5_f218_541d, s40:0xaa_aaaa_aaaa, s40:0x1_0000, s40:0xff_ffff_ffff, s40:0x7f_ffff_ffff, s40:0x40], [s40:0x7f_ffff_ffff, s40:0, s40:0x13_5007_6e33, s40:0xaa_aaaa_aaaa, s40:0x0, s40:0x1000, s40:0xaa_aaaa_aaaa], [s40:0x55_5555_5555, s40:0x10, s40:0x55_5555_5555, s40:0xaa_aaaa_aaaa, s40:0x55_5555_5555, s40:0x7f_ffff_ffff, s40:0x10], [s40:0xff_ffff_ffff, s40:8589934592, s40:0x8_0000, s40:0x5dd5_0eea, s40:0xff_ffff_ffff, s40:0x3b_d3b0_4cda, s40:0b101_0101_0101_0101_0101_0101_0101_0101_0101_0101], [s40:0x55_5555_5555, s40:0x55_5555_5555, s40:0xff_ffff_ffff, s40:0x91_b13d_15d5, s40:0x0, s40:-1, s40:0x7f_ffff_ffff], [s40:0x55_5555_5555, s40:0x23_ab81_e3af, s40:0, s40:0xff_ffff_ffff, s40:0x7f_ffff_ffff, s40:0xff_ffff_ffff, s40:0xff_ffff_ffff], [s40:0x0, s40:0x9_edd3_1f4d, s40:0xaa_aaaa_aaaa, s40:0x80, s40:-1, s40:0x55_5555_5555, s40:0xff_ffff_ffff], [s40:0xff_ffff_ffff, s40:0x10_0000_0000, s40:0x800, s40:0x55_5555_5555, s40:0xff_ffff_ffff, s40:0xff_ffff_ffff, s40:0x55_5555_5555]],
        };
        (x38, x35, x35, x35)
    }
}
proc main {
    x6: chan<u14> in;
    config(x6: chan<u14> in) {
        (x6,)
    }
    init {
        u58:144115188075855871
    }
    next(x0: u58) {
        {
            let x2: x1[1] = [x0];
            let x3: x1[1] = array_slice(x2, x0, x1[1]:[x2[u32:0x0], ...]);
            let x4: s10 = s10:0x2aa;
            let x5: u58 = !x0;
            let x7: (token, u14) = recv(join(), x6);
            let x8: token = x7.0;
            let x9: u14 = x7.1;
            let x10: s10 = -x4;
            let x11: bool = x2 != x3;
            let x12: uN[189] = x5 ++ x9 ++ x0 ++ x11 ++ x0;
            let x13: u58 = one_hot_sel(x11, [x0]);
            let x14: u58 = signex(x4, x0);
            let x15: uN[189] = !x12;
            let x16: bool = -x11;
            let x17: u10 = (x4 as u10)[0+:u10];
            let x18: bool = or_reduce(x15);
            let x19: bool = or_reduce(x5);
            let x20: token = x7.0;
            let x21: uN[189] = -x15;
            let x27: (bool, bool, bool, bool) = x22(x19, x4, x17);
            let x29: x28[3] = [x12, x15, x21];
            let x30: x28[6] = x29 ++ x29;
            let x39: (x37[W32_V10], x1[1], x1[1], x1[1]) = x31(x18, x2, x27);
            let x40: uN[189] = -x15;
            let x41: u58 = bit_slice_update(x13, x12, x9);
            let x42: x1[1] = x39.1;
            let x43: u59 = one_hot(x5, bool:0x0);
            let x45: x44[5] = [x12, x15, x21, x40, x40];
            x41
        }
    }
}
