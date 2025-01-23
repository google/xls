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
//
// BEGIN_CONFIG
// exception: "SampleError: Result miscompare for sample 0:\nargs: bits[2]:0x0\nevaluated opt IR (JIT), evaluated opt IR (interpreter) =\nbits[1]:0x1\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\nbits[1]:0x0\nError: INVALID_ARGUMENT: SampleError: Result miscompare for sample 0:\nargs: bits[2]:0x0 evaluated opt IR (JIT), evaluated opt IR (interpreter) =\nbits[1]:0x1\nevaluated unopt IR (JIT), evaluated unopt IR (interpreter), interpreted DSLX =\nbits[1]:0x0"
// issue: "https://github.com/google/xls/issues/1184"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=1"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[2]:0"
//     args: "bits[2]:1"
//     args: "bits[2]:2"
//     args: "bits[2]:3"
//   }
// }
// END_CONFIG
fn main(x0: bits[2]) -> bits[1] {
  let x1 = bits[1]:1;
  let x2 = x0[0:1];
  let x3 = bits[25]:32508169;
  let x4 = x1 ++ x2 ++ x3;
  let x5 = bits[27]:127305941;
  (x4 as sN[27]) >= (x5 as sN[27])
}
