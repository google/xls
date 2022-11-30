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
// args: bits[14]:0x3295
fn main(x81: u14) -> (u22, u14, u5, u22, u5, u5, u5, u5, u5, u14) {
    let x82: u5 = (u5:0x15);
    let x83: u5 = !(x82);
    let x84: u5 = ((x81) as u5) + (x82);
    let x85: u5 = (x84) - (x82);
    let x86: u5 = -(x83);
    let x87: u5 = (x86) * (x83);
    let x88: u22 = (u22:0x1fffff);
    let x89: u5 = (x84) << (x82);
    let x90: u13 = (u13:0xfff);
    (x88, x81, x82, x88, x83, x86, x85, x84, x83, x81)
}


