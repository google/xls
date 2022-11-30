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

// options: {"input_is_dslx": true, "ir_converter_args": ["--top=main"], "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3", "--reset_data_path=false"], "simulate": false, "simulator": null}
// args: bits[30]:0x2e99f5a0; bits[15]:0x8d9; bits[32]:0x58490179; bits[17]:0x1228d
fn main(x17: u30, x18: u15, x19: u32, x20: u17) -> (u6, u6, u6, u6, u6, u6, u30, u6, u6, u6, u32, u6) {
    let x21: u6 = (u6:0x2);
    let x22: u6 = (x21) * (x21);
    let x23: u9 = (u9:0x2);
    let x24: u6 = !(x22);
    let x25: u6 = (x21) >> (x21);
    let x26: u6 = (x24) >> (x24);
    let x27: u6 = (x21) * (x22);
    let x28: u25 = (u25:0x2000);
    (x21, x22, x26, x27, x25, x26, x17, x27, x25, x21, x19, x25)
}


