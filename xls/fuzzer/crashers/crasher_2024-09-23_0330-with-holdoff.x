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
//
// Note, this is the same as crasher_2024-09-23_0330.x, just with
// a few valid_holdoffs sprinkled in.
// BEGIN_CONFIG
// # proto-message: xls.fuzzer.CrasherConfigurationProto
// exception: "Subprocess call timed out after 1500 seconds: "
// issue: "https://github.com/google/xls/issues/1624"
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
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 187
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
//       channel_name: "sample__x35"
//       valid_holdoffs: { cycles: 1 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x4465"
//       valid_holdoffs: { cycles: 3 }
//       values: "bits[16]:0xd306"
//       valid_holdoffs: { cycles: 7 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7cbb"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xbad0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xadbd"
//       valid_holdoffs: { cycles: 5 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x4"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x400"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x400"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7869"
//       valid_holdoffs: { cycles: 9 }
//       values: "bits[16]:0xd816"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x100"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x43e0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 2 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7a34"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 9 }
//       values: "bits[16]:0x400"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x200"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x4691"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x4000"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xd74d"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 23 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x8000"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x612a"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x10"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xd19a"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x2249"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x1000"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xa66f"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x8000"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x200"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x800"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x10"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x1"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x1000"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5555"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x5310"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xaaaa"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x8"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x800"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x100"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x0"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x7fff"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0x4"
//       valid_holdoffs: { cycles: 0 }
//       values: "bits[16]:0xffff"
//     }
//   }
// }
// 
// END_CONFIG
const W32_V3 = u32:0x3;
type x21 = u13;
fn x7(x8: u39, x9: u46, x10: u46, x11: u39, x12: u24, x13: u46) -> (u38, u39, u39, (), u38) {
    {
        let x14: () = ();
        let x15: u39 = -x11;
        let x16: u39 = x11[0+:u39];
        let x17: u39 = clz(x11);
        let x18: u39 = x15[0+:u39];
        let x19: u38 = x13[8:];
        (x19, x16, x18, x14, x19)
    }
}
proc main {
    x35: chan<u16> in;
    config(x35: chan<u16> in) {
        (x35,)
    }
    init {
        u39:274877906943
    }
    next(x0: u39) {
        {
            let x1: u39 = x0;
            let x2: u15 = x1[24+:u15];
            let x3: u46 = u46:0x382a_0245_d341;
            let x4: u24 = x0[x2+:u24];
            let x5: u39 = !x1;
            let x6: u39 = x1 >> if x3 >= u46:0x16 { u46:0x16 } else { x3 };
            let x20: (u38, u39, u39, (), u38) = x7(x0, x3, x3, x0, x4, x3);
            let x22: x21[W32_V3] = x6 as x21[W32_V3];
            let x23: u15 = x2 >> x0;
            let x24: u38 = x20.0;
            let x25: u39 = one_hot(x24, bool:0x1);
            let x26: bool = xor_reduce(x23);
            let x27: u38 = x24[x24+:u38];
            let x28: u15 = x23 >> if x3 >= u46:0xc { u46:0xc } else { x3 };
            let x29: u38 = x24[:];
            let x30: u39 = bit_slice_update(x25, x6, x6);
            let x31: u39 = x6 * x6;
            let x32: u38 = -x24;
            let x33: u5 = x24[6+:u5];
            let x34: x21[6] = x22 ++ x22;
            let x36: (token, u16) = recv(join(), x35);
            let x37: token = x36.0;
            let x38: u16 = x36.1;
            let x39: u38 = !x27;
            let x40: u39 = x5 << if x1 >= u39:16 { u39:16 } else { x1 };
            let x42: s39 = {
                let x41: (u39, u39) = smulp(x29 as u39 as s39, x6 as s39);
                (x41.0 + x41.1) as s39
            };
            let x43: u12 = u12:0xaaa;
            let x44: bool = x26 & x26;
            let x45: u39 = x0 + x43 as u39;
            let x46: u63 = u63:0x3fff_ffff_ffff_ffff;
            let x47: u39 = signex(x1, x1);
            x31
        }
    }
}
