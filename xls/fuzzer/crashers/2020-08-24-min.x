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

// options: {"input_is_dslx": true, "ir_converter_args": ["--top=main"], "convert_to_ir": true, "optimize_ir": true, "codegen": false, "simulate": false, "simulator": null}
// args: bits[8]:0x5; bits[12]:0xe9c; bits[35]:0x5_5540_0000

fn main(p3: u8, p6: u12, p8: s35) -> u8 {
  let x13: u12 = clz(p6);
  let x14: u12 = clz(x13);
  let x18: u8 = ((p8 as u8)) | (p3);
  let x24: u1 = xor_reduce(x14);
  let x30: u8 = one_hot_sel(x24, [x18]);
  x30
}
