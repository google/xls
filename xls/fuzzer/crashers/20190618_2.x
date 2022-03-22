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

// options: {"input_is_dslx": true, "ir_converter_args": ["--top=main"], "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[32]:0x696eb227
// args: bits[32]:0xde6321b5
// args: bits[32]:0xee9f2833
// args: bits[32]:0x80f9c9c1
// args: bits[32]:0xddbc3a2f
// args: bits[32]:0x6d6e937a
// args: bits[32]:0x32c8e0f3
// args: bits[32]:0xd74137b
// args: bits[32]:0xdbb046
// args: bits[32]:0x3a45263c
// args: bits[32]:0xd01f9f04
// args: bits[32]:0x242cdb3f
// args: bits[32]:0xb3c1e3ee
// args: bits[32]:0xb84cb936
// args: bits[32]:0xa1acc1e1
// args: bits[32]:0xcfa507dd
// args: bits[32]:0xc4ead610
// args: bits[32]:0xbbfe7aed
// args: bits[32]:0x33a6b5fb
// args: bits[32]:0x7c6278f7
// args: bits[32]:0x124dd8d1
// args: bits[32]:0x3bba609a
// args: bits[32]:0xf3bde47d
// args: bits[32]:0xc85c9dda
// args: bits[32]:0x91fb0daf
// args: bits[32]:0x4099cd30
// args: bits[32]:0x6728530d
// args: bits[32]:0xb246b264
// args: bits[32]:0x79de50ee
// args: bits[32]:0x7cba88f2
// args: bits[32]:0x65510324
// args: bits[32]:0x6c42e8d
// args: bits[32]:0x534be388
// args: bits[32]:0x6a03e7d7
// args: bits[32]:0xa80c4aa3
// args: bits[32]:0x3f76e129
// args: bits[32]:0xb043f2b4
// args: bits[32]:0xaaaaaaaa
// args: bits[32]:0x20000000
// args: bits[32]:0x7c49cd93
// args: bits[32]:0xdf149c41
// args: bits[32]:0xb4713422
// args: bits[32]:0x12b446de
// args: bits[32]:0x3017eb2
// args: bits[32]:0x1
// args: bits[32]:0xe59d03ff
// args: bits[32]:0x63de0156
// args: bits[32]:0xe87d25d2
// args: bits[32]:0xc672d938
// args: bits[32]:0x2a1fba1f
// args: bits[32]:0xe031533a
// args: bits[32]:0x256da5a6
// args: bits[32]:0xd29be0f6
// args: bits[32]:0x8bf6eafd
// args: bits[32]:0x1127dc21
// args: bits[32]:0xfe2f2128
// args: bits[32]:0x6b775514
// args: bits[32]:0x57bdc279
// args: bits[32]:0x36a090d0
// args: bits[32]:0x1bbdfb51
// args: bits[32]:0xd9aef4c1
// args: bits[32]:0x8fdf7096
// args: bits[32]:0xbb031f9b
// args: bits[32]:0xcd6ceba6
fn main(x42: u32) -> (u32, u13, u13, u32, u13, u32, u32, u13, u32) {
    let x43: u13 = (u13:0x8);
    let x44: u32 = (x42) - (x42);
    let x45: u32 = (x44) << (x42);
    let x46: u13 = !(x43);
    let x47: bool = (x46) >= ((x45) as u13);
    let x48: u32 = -(x42);
    let x49: u32 = !(x44);
    let x50: u13 = (x43) - (x46);
    (x44, x46, x46, x49, x50, x42, x45, x46, x49)
}


