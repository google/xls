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
// args: bits[6]:0x0; bits[19]:0x398dd; bits[25]:0x3fc4fe; bits[28]:0x9b743d6
fn main(x56: u6, x57: u19, x58: u25, x59: u28) -> (u22, u25, u19, u22, u6, u25, u19, u6, u25, u22, u28, u20, u6) {
    let x60: u19 = !(x57);
    let x61: u20 = (u20:0x80);
    let x62: u20 = ((x56) as u20) - (x61);
    let x63: u20 = (x61) + (x62);
    let x64: u22 = (u22:0x1000);
    let x65: u19 = -(x57);
    let x66: u25 = (x58) ^ ((x61 as u25));
    let x67: u25 = (x58) | ((x56 as u25));
    let x68: u22 = ((x57 as u22)) + (x64);
    (x68, x67, x60, x64, x56, x66, x65, x56, x67, x68, x59, x63, x56)
}


