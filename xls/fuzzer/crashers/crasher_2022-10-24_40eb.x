// Copyright 2022 The XLS Authors
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
// BEGIN_CONFIG
// exception: "// Command \'[\'xls/tools/opt_main\', \'sample.ir\', \'--logtostderr\']\' died with <Signals.SIGABRT: 6>."
// issue: "https://github.com/google/xls/issues/758"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=2"
//   codegen_args: "--reset_data_path=false"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 600
//   calls_per_sample: 128
// }
// inputs {
//   function_args {
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x4"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x2d"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x3a"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x0"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x5"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0xa"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x0"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x2a"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x2e"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x3f"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0xb"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x32"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x38"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x32"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x1f"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x10"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0xc"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x7"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x3a"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x3f"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x30"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x2"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x39"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x25"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x2a"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x2a"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x2f"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x15"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x1f"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x0"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x20"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x20"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x1"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x10"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x4"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x25"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x15"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x2d"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x2"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0xf"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x2a"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x30"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x0"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x15"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x30"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0xb"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x3a"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x25"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x4"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x15"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x3d"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x0"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x20"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x6"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0xa"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x8"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x2"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x2a"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x2a"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x1"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x16"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x15"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x0"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x39"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x21"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0xa"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x37"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1b"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x20"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x0"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x2a"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x10"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x2"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x20"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0xa"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x15"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x35"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0xb"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x15"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x2f"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x10"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x3f"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x8"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x20"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0xe"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x1f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x33"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x15"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x10"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x1"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x0"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x1d"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x2a"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x29"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x3f"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x15"
//     args: "(bits[1]:0x0); bits[1]:0x0; bits[6]:0x15"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x2a"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x35"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x0"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x10"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x0"
//     args: "(bits[1]:0x0); bits[1]:0x1; bits[6]:0x2d"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x9"
//     args: "(bits[1]:0x1); bits[1]:0x1; bits[6]:0x30"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x17"
//     args: "(bits[1]:0x1); bits[1]:0x0; bits[6]:0x0"
//   }
// }
// END_CONFIG
fn main(x0: (u1,), x1: u1, x2: u6) -> (bool, bool, bool, bool, u5, bool) {
  let x3: u6 = ctz(x2);
  let x4: bool = (x1)[x3+:bool];
  let x5: bool = (x4) & (x4);
  let x6: bool = xor_reduce(x1);
  let x7: u1 = x0.0;
  let x8: bool = bit_slice_update(x4, x4, x2);
  let x9: u1 = !(x7);
  let x10: bool = clz(x4);
  let x11: bool = (x6) | (((x10) as bool));
  let x13: bool = {
    let x12: (bool, bool) = umulp(((x10) as bool), x5);
    (x12.0) + (x12.1)
  };
  let x14: bool = (x13)[0+:bool];
  let x15: bool = (x8) & (x8);
  let x16: u6 = !(x3);
  let x17: bool = (x10)[0+:bool];
  let x18: bool = (x5) - (x4);
  let x19: u5 = u5:0x1f;
  let x20: bool = (x18) < (x15);
  let x21: bool = -(x15);
  let x22: bool = priority_sel(x16, [x5, x14, x10, x14, x14, x17], false);
  let x23: u2 = (x11) ++ (x17);
  let x24: bool = !(x11);
  let x25: (u1, bool, u6, u6) = (x9, x22, x16, x16);
  let x26: bool = (x11)[x22+:bool];
  let x27: u1 = !(x1);
  (x20, x5, x5, x26, x19, x13)
}
