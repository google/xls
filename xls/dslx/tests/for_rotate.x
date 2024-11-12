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

import std;

// Rotates the nibbles in the u8.
fn main(x: u8) -> u8 {
    for (_, x): (u4, u8) in u4:0..u4:4 {
        std::rotr(x, u8:1)
    }(x)
}

#[test]
fn main_test() {
    assert_eq(u8:0xba, main(u8:0xab));
    assert_eq(u8:0xdc, main(u8:0xcd));
    assert_eq(u8:0x55, main(u8:0x55));
    assert_eq(u8:0xaa, main(u8:0xaa));
}
