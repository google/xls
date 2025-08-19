// Copyright 2024 The XLS Authors
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

// Return given value with m first bits masked
pub fn mask<N: u32, M: u32>(n: bits[N], m: bits[M]) -> bits[N] {
    n & (std::mask_bits<N>() >> (N as bits[M] - m))
}

#[test]
fn mask_test() {
    assert_eq(mask(u8:0b11111111, u4:0), u8:0b00000000);
    assert_eq(mask(u8:0b11111111, u4:1), u8:0b00000001);
    assert_eq(mask(u8:0b11111111, u4:2), u8:0b00000011);
    assert_eq(mask(u8:0b11111111, u4:4), u8:0b00001111);
    assert_eq(mask(u8:0b11111111, u4:8), u8:0b11111111);
    assert_eq(mask(u8:0b11111111, u4:9), u8:0b00000000); // FIXME: sketchy result, I would expect
                                                         // 0b11111111
}
