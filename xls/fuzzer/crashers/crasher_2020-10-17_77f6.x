// Copyright 2020 The XLS Authors
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

// options: {"input_is_dslx": true, "ir_converter_args": ["--top=main"], "convert_to_ir": true, "optimize_ir": true, "use_jit": true, "codegen": false, "codegen_args": null, "simulate": false, "simulator": null, "use_system_verilog": true}
// args: bits[25]:0x100_9010; bits[36]:0x20; bits[53]:0x4_c2f0_06d1
fn main(x4: u25, x5: s36, x7: u53) -> (s36, u25, u1) {
  let x8: u1 = or_reduce(x7);
  let x9: s36 = one_hot_sel(x8, [x5]);
  let x10: u25 = (x4)[x8+:u25];
  (x9, x10, x8)
}
