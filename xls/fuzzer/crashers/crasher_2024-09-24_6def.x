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
// # proto-message: xls.fuzzer.CrasherConfigurationProto
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
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
//   }
// }
// 
// END_CONFIG
type x12 = u24;
type x21 = u48;
proc main {
    config() {
        ()
    }
    init {
        u24:11184810
    }
    next(x0: u24) {
        {
            let x1: u24 = x0 / u24:0xff_ffff;
            let x2: u48 = x1 ++ x1;
            let x3: u48 = !x2;
            let x4: u24 = x0 + x3 as u24;
            let x5: u27 = u27:0x5e2_116a;
            let x6: u5 = x0[19:];
            let x7: u48 = clz(x3);
            let x8: u48 = bit_slice_update(x2, x3, x3);
            let x9: u27 = gate!(x6 >= x8 as u5, x5);
            let x10: u48 = -x2;
            let x11: u24 = x4 / u24:0x7f_ffff;
            let x13: x12[4] = [x0, x1, x11, x4];
            let x14: u48 = x8[x6+:u48];
            let x15: x12[8] = x13 ++ x13;
            let x16: x12 = x13[if x11 >= u24:0x2 { u24:0x2 } else { x11 }];
            let x17: u24 = one_hot_sel(x6, [x0, x0, x0, x4, x11]);
            let x18: u24 = signex(x16, x11);
            let x19: u48 = x8 * x1 as u48;
            let x20: u25 = match x0 {
                u24:0x55_5555 | u24:0xff_ffff => u25:0b1010_1010_1010_1010_1010_1010,
                u24:0xaa_aaaa..u24:0x7f_ffff | u24:0xaa_aaaa => u25:0xff_ffff,
                u24:0x4e_c78b => u25:0x0,
                _ => u25:0xff_ffff,
            };
            let x22: x21[5] = [x10, x19, x2, x3, x8];
            x18
        }
    }
}
