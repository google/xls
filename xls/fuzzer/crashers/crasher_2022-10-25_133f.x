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
// evaluated opt IR (JIT), evaluated opt IR (interpreter), simulated =
//    (bits[4]:0xa, (bits[54]:0x2a_aaaa_aaaa_aaaa, bits[4]:0xf, bits[4]:0x1, bits[4]:0x1, bits[34]:0x2_c000_3c07, bits[4]:0x1), bits[54]:0x2a_aaaa_aaaa_aaaa, bits[4]:0x0)
// evaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =
//    (bits[4]:0xa, (bits[54]:0x2a_aaaa_aaaa_aaaa, bits[4]:0xf, bits[4]:0x1, bits[4]:0x1, bits[34]:0x2_c000_3c07, bits[4]:0x1), bits[54]:0x2a_aaaa_aaaa_aaaa, bits[4]:0xf)
//
// BEGIN_CONFIG
// exception: "// Result miscompare for sample 2:"
// issue: "https://github.com/google/xls/issues/757"
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
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset_data_path=false"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 600
//   calls_per_sample: 128
// }
// inputs {
//   function_args {
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0x3"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x8"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x4"
//     args: "bits[4]:0x3"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x1"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0xe"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x9"
//     args: "bits[4]:0x8"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x8"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0xb"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x4"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x8"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x8"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x4"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x4"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x1"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x8"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0x1"
//     args: "bits[4]:0x8"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x1"
//     args: "bits[4]:0x4"
//     args: "bits[4]:0x4"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x0"
//     args: "bits[4]:0x2"
//     args: "bits[4]:0x3"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xa"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0x6"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0xf"
//     args: "bits[4]:0x5"
//     args: "bits[4]:0x7"
//     args: "bits[4]:0xf"
//   }
// }
// END_CONFIG
type x18 = bool;
type x21 = x18[1];
fn main(x0: u4) -> (u4, (u54, u4, u4, u4, u34, u4), u54, u4) {
  let x1: u4 = ctz(x0);
  let x2: u54 = u54:0x2a_aaaa_aaaa_aaaa;
  let x3: u54 = -(x2);
  let x4: u34 = u34:0x2_c000_3c07;
  let x5: u4 = -(x1);
  let x6: u4 = priority_sel(x5, [x1, x5, x5, x0], u4:0);
  let x7: (u54, u4, u4, u4, u34, u4) = (x2, x5, x1, x1, x4, x1);
  let x8: u6 = u6:0x3f;
  let x9: u4 = (x6)[x8+:u4];
  let x10: u34 = -(x4);
  let x11: bool = (((x2) as u34)) <= (x4);
  let x12: u4 = -(x6);
  let x13: bool = (x7) == (x7);
  let x14: bool = (((x11) as bool)) ^ (x13);
  let x15: u4 = x7.5;
  let x16: bool = (x7) != (x7);
  let x17: bool = (x7) != (x7);
  let x19: x18[1] = ((x16) as x18[1]);
  let x20: bool = (x17)[x10+:bool];
  let x22: x21[1] = [x19];
  let x23: u62 = (((x3) ++ (x16)) ++ (x8)) ++ (x13);
  (x0, x7, x2, x12)
}
