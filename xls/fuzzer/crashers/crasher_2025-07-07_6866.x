// Copyright 2025 The XLS Authors
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
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 8
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full throughput.*"
//   }
//   codegen_ng: true
//   disable_unopt_interpreter: false
// }
// inputs {
//   function_args {
//     args: "bits[34]:0x1_ffff_ffff; bits[47]:0x5555_5555_5555; bits[64]:0x800_0000; bits[20]:0x8_0000"
//     args: "bits[34]:0x0; bits[47]:0xfff; bits[64]:0x4100_1000_1ffc_1430; bits[20]:0x6_2634"
//     args: "bits[34]:0x40; bits[47]:0x42e_080a_5ff5; bits[64]:0x5840_7338_bba8_ddd9; bits[20]:0xdd5f"
//     args: "bits[34]:0x3_ffff_ffff; bits[47]:0x7fff_ffff_ffff; bits[64]:0xf7bb_4bff_feff_4159; bits[20]:0x7_ffff"
//     args: "bits[34]:0x1_ffff_ffff; bits[47]:0x3acb_fdff_f555; bits[64]:0xffff_ffff_ffff_ffff; bits[20]:0xf_ffff"
//     args: "bits[34]:0x3_ffff_ffff; bits[47]:0x7fff_ffff_ffff; bits[64]:0x7ffb_f3ff_efdb_8aa6; bits[20]:0x8_5ff8"
//     args: "bits[34]:0x80_0000; bits[47]:0x20_0000_0000; bits[64]:0x9040_0888_000c_bf9f; bits[20]:0x2_2001"
//     args: "bits[34]:0x1_ffff_ffff; bits[47]:0x0; bits[64]:0xaaaa_aaaa_aaaa_aaaa; bits[20]:0x7_1d51"
//   }
// }
//
// END_CONFIG
type x19 = u8;
fn main(x0: u34, x1: u47, x2: u64, x3: u20) -> (x19[1],) {
    {
        let x4: u34 = !x0;
        let x5: u18 = x1[22:-7];
        let x6: u18 = -x5;
        let x7: u18 = x6 >> if x6 >= u18:0x2 { u18:0x2 } else { x6 };
        let x9: u47 = x1[:];
        let x10: u18 = -x5;
        let x11: u20 = x0 as u20 - x3;
        let x12: bool = x1 == x1;
        let x13: u18 = -x5;
        let x14: bool = x12 / bool:0x1;
        let x15: (u20, u18, bool, u47, u18, bool, u20) = (x11, x7, x12, x9, x6, x12, x3);
        let x16: bool = x14 >> 0;
        let x17: u18 = x3 as u18 ^ x6;
        let x18: u8 = match x14 {
            bool:0x0 | bool:1 => u8:0x0,
            bool:false => u8:0x7f,
            _ => u8:0x0,
        };
        let x20: x19[1] = [x18];
        let x21: bool = x14 ^ x12;
        let x22: (u47,) = (x9,);
        let (x23) = (x9,);
        let x24: x19[2] = x20 ++ x20;
        let x26: u34 = {
            let x25: (u34, u34) = umulp(x6 as u34, x4);
            x25.0 + x25.1
        };
        let x27: x19[11] = array_slice(x24, x6, x19[11]:[x24[u32:0x0], ...]);
        let x28: u56 = x5 ++ x17 ++ x11;
        let x29: x19 = x24[if x11 >= u20:0x1 { u20:0x1 } else { x11 }];
        let x30: u34 = !x0;
        let x31: bool = !x12;
        (x20,)
    }
}
