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
// args: bits[5]:0x10
// args: bits[5]:0x2
// args: bits[5]:0x19
// args: bits[5]:0x1e
// args: bits[5]:0x9
// args: bits[5]:0x11
// args: bits[5]:0x1c
// args: bits[5]:0x6
// args: bits[5]:0x5
// args: bits[5]:0x6
// args: bits[5]:0x15
// args: bits[5]:0x6
// args: bits[5]:0x10
// args: bits[5]:0x17
// args: bits[5]:0x13
// args: bits[5]:0x5
// args: bits[5]:0x1a
// args: bits[5]:0x6
// args: bits[5]:0x1a
// args: bits[5]:0x18
// args: bits[5]:0x1d
// args: bits[5]:0x13
// args: bits[5]:0x12
// args: bits[5]:0x0
// args: bits[5]:0x1d
// args: bits[5]:0x1e
// args: bits[5]:0x10
// args: bits[5]:0x1
// args: bits[5]:0x13
// args: bits[5]:0x4
// args: bits[5]:0x4
// args: bits[5]:0x13
// args: bits[5]:0x0
// args: bits[5]:0x8
// args: bits[5]:0x17
// args: bits[5]:0xc
// args: bits[5]:0xb
// args: bits[5]:0x19
// args: bits[5]:0x4
// args: bits[5]:0xf
// args: bits[5]:0x17
// args: bits[5]:0x2
// args: bits[5]:0x1
// args: bits[5]:0x5
// args: bits[5]:0xb
// args: bits[5]:0x0
// args: bits[5]:0x3
// args: bits[5]:0x4
// args: bits[5]:0x2
// args: bits[5]:0x9
// args: bits[5]:0x12
// args: bits[5]:0x16
// args: bits[5]:0x1b
// args: bits[5]:0x3
// args: bits[5]:0x13
// args: bits[5]:0xc
// args: bits[5]:0x14
// args: bits[5]:0x3
// args: bits[5]:0x7
// args: bits[5]:0x1e
// args: bits[5]:0xb
// args: bits[5]:0x6
// args: bits[5]:0x0
// args: bits[5]:0x1
fn main(x65: u5) -> (u27, u5, u27, bool, u5) {
    let x66: u27 = (u27:0x200000);
    let x67: u27 = (x66) * (x66);
    let x68: u27 = (x66) * (x67);
    let x69: u5 = (x65) >> (x65);
    let x70: bool = (x68) <= (x67);
    let x71: bool = (x70) - (x70);
    (x68, x65, x67, x70, x69)
}


