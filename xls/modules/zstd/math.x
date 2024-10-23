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

fn fast_if<N: u32>(cond: bool, arg1: uN[N], arg2: uN[N]) -> uN[N] {
    let mask = if cond { !bits[N]:0 } else { bits[N]:0 };
    (arg1 & mask) | (arg2 & !mask)
}

#[test]
fn fast_if_test() {
    assert_eq(if true { u32:1 } else { u32:5 }, fast_if(true, u32:1, u32:5));
    assert_eq(if false { u32:1 } else { u32:5 }, fast_if(false, u32:1, u32:5));
}

// Log-depth shift bits left
pub fn logshiftl<N: u32, R: u32>(n: bits[N], r: bits[R]) -> bits[N] {
    for (i, y) in u32:0..R {
        fast_if(r[i+:u1], { y << (bits[R]:1 << i) }, { y })
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
    assert_eq(logshiftl(max, u4:15), max << u4:15);
    assert_eq(logshiftl(bits[24]:0xc0ffee, u8:12), bits[24]:0xfee000);
}

// Log-depth shift bits right
pub fn logshiftr<N: u32, R: u32>(n: bits[N], r: bits[R]) -> bits[N] {
    for (i, y) in u32:0..R {
        fast_if(r[i+:u1], { y >> (bits[R]:1 << i) }, { y })
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
    assert_eq(logshiftr(max, u4:15), max >> u4:15);
    assert_eq(logshiftr(bits[24]:0xc0ffee, u8:12), bits[24]:0x000c0f);
}
