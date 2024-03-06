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

// Perform a (sequential) Adler-32 checksum over a single input byte.

fn adler32_seq(buf: u8) -> u32 {
    let a = u32:1;
    let b = u32:0;
    // Iterate only over input of length 1, for now.
    let (a, b) = for (_, (a, b)): (u8, (u32, u32)) in range(u8:0, u8:1) {
        let a = (a + (buf as u32)) % u32:65521;
        let b = (b + a) % u32:65521;
        (a, b)
    }((a, b));
    (b << u32:16) | a
}

fn main(message: u8) -> u32 { adler32_seq(message) }

#[test]
fn adler32_one_char_test() {
    assert_eq(u32:0x0010001, main(u8:0x00));  // dec 0
    assert_eq(u32:0x0310031, main(u8:0x30));  // '0'
    assert_eq(u32:0x0620062, main(u8:0x61));  // 'a'
    assert_eq(u32:0x07f007f, main(u8:0x7e));  // '~' (dec 126)
    assert_eq(u32:0x0800080, main(u8:0x7f));  // 'DEL' (dec 127)
    assert_eq(u32:0x1000100, main(u8:0xFf))  // dec 255
}
