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

// Log-depth shift bits left
pub fn logshiftl<N: u32, R: u32>(n: bits[N], r: bits[R]) -> bits[N] {
    for (i, y) in u32:0..R {
        if r[i+:u1] { y << (bits[R]:1 << i) } else { y }
    }(n as bits[N])
}

#[test]
fn logshiftl_test() {
    // Test varying base
    assert_eq(logshiftl(bits[64]:0, bits[6]:3), bits[64]:0 << u32:3);
    assert_eq(logshiftl(bits[64]:1, bits[6]:3), bits[64]:1 << u32:3);
    assert_eq(logshiftl(bits[64]:2, bits[6]:3), bits[64]:2 << u32:3);
    assert_eq(logshiftl(bits[64]:3, bits[6]:3), bits[64]:3 << u32:3);
    assert_eq(logshiftl(bits[64]:4, bits[6]:3), bits[64]:4 << u32:3);

    // Test varying exponent
    assert_eq(logshiftl(bits[64]:50, bits[6]:0), bits[64]:50 << u32:0);
    assert_eq(logshiftl(bits[64]:50, bits[6]:1), bits[64]:50 << u32:1);
    assert_eq(logshiftl(bits[64]:50, bits[6]:2), bits[64]:50 << u32:2);
    assert_eq(logshiftl(bits[64]:50, bits[6]:3), bits[64]:50 << u32:3);
    assert_eq(logshiftl(bits[64]:50, bits[6]:4), bits[64]:50 << u32:4);

    // Test overflow
    let max = std::unsigned_max_value<u32:8>();
    assert_eq(logshiftl(max, u4:4), max << u4:4);
    assert_eq(logshiftl(max, u4:5), max << u4:5);
    // TIv2 does not allow a direct left over-shift by a literal amount.
    let overshift_amount = u4:15;
    assert_eq(logshiftl(max, u4:15), max << overshift_amount);
    assert_eq(logshiftl(bits[24]:0xc0ffee, u8:12), bits[24]:0xfee000);
}


// Log-depth shift bits right
pub fn logshiftr<N: u32, R: u32>(n: bits[N], r: bits[R]) -> bits[N] {
    for (i, y) in u32:0..R {
        if r[i+:u1] { y >> (bits[R]:1 << i) } else { y }
    }(n as bits[N])
}

#[test]
fn logshiftr_test() {
    // Test varying base
    assert_eq(logshiftr(bits[64]:0x0fac4e782, bits[6]:3), bits[64]:0x0fac4e782 >> u32:3);
    assert_eq(logshiftr(bits[64]:0x1fac4e782, bits[6]:3), bits[64]:0x1fac4e782 >> u32:3);
    assert_eq(logshiftr(bits[64]:0x2fac4e782, bits[6]:3), bits[64]:0x2fac4e782 >> u32:3);
    assert_eq(logshiftr(bits[64]:0x3fac4e782, bits[6]:3), bits[64]:0x3fac4e782 >> u32:3);
    assert_eq(logshiftr(bits[64]:0x4fac4e782, bits[6]:3), bits[64]:0x4fac4e782 >> u32:3);

    // Test varying exponent
    assert_eq(logshiftr(bits[64]:0x50fac4e782, bits[6]:0), bits[64]:0x50fac4e782 >> u32:0);
    assert_eq(logshiftr(bits[64]:0x50fac4e782, bits[6]:1), bits[64]:0x50fac4e782 >> u32:1);
    assert_eq(logshiftr(bits[64]:0x50fac4e782, bits[6]:2), bits[64]:0x50fac4e782 >> u32:2);
    assert_eq(logshiftr(bits[64]:0x50fac4e782, bits[6]:3), bits[64]:0x50fac4e782 >> u32:3);
    assert_eq(logshiftr(bits[64]:0x50fac4e782, bits[6]:4), bits[64]:0x50fac4e782 >> u32:4);

    // Test overflow
    let max = std::unsigned_max_value<u32:8>();
    assert_eq(logshiftr(max, u4:4), max >> u4:4);
    assert_eq(logshiftr(max, u4:5), max >> u4:5);
    // TIv2 does not allow a direct right over-shift by a literal amount.
    let overshift_amount = u4:15;
    assert_eq(logshiftr(max, u4:15), max >> overshift_amount);
    assert_eq(logshiftr(bits[24]:0xc0ffee, u8:12), bits[24]:0x000c0f);
}

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
