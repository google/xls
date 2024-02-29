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

// Port of "ACM random" number generator to DSLX, similar to
// https://github.com/google/or-tools/blob/66b8d230798f9b8a3c98c26a997daf04974400b8/ortools/base/random.cc#L35
//
// DO NOT use ACM random for any application where security --
// unpredictability of subsequent output and previous output -- is
// needed.  ACMRandom is in *NO* *WAY* a cryptographically secure
// pseudorandom number generator, and using it where recipients of its
// output may wish to guess earlier/later output values would be very
// bad.

import std;

const M = u32:2147483647;

struct State { seed: u32 }

// Returns a pseudo-random number in the range [1, 2^31-2].
//
// Note that this is one number short on both ends of the full range of
// non-negative 32-bit integers, which range from 0 to 2^31-1.
pub fn rng_next(s: State) -> (State, u32) {
    const A = u64:16807;
    let product = (s.seed as u64) * A;
    let new_seed = ((product >> u64:31) + (product & (M as u64))) as u32;
    let new_seed = if std::sgt(new_seed, M) { new_seed - M } else { new_seed };
    (State { seed: new_seed }, new_seed)
}

// Returns a pseudo random number in the range [1, (2^31-2)^2].
// Note that this does not cover all non-negative values of int64, which range
// from 0 to 2^63-1. The top two bits are ALWAYS ZERO.
pub fn rng_next64(s: State) -> (State, u64) {
    let (s, next0) = rng_next(s);
    let (s, next1) = rng_next(s);
    let result: s64 = ((next0 as s64) - s64:1) * (((M as s64) - s64:1) as s64) + (next1 as s64);
    (s, result as u64)
}

// Returns a fixed seed for use in the random number generator.
pub fn rng_deterministic_seed() -> u32 { u32:301 }

// Create the state for a new random number generator using the given seed.
fn rng_sanitize_seed(seed: u32) -> u32 {
    let seed = seed & u32:0x7fffffff;
    match seed {
        u32:0 | M => u32:1,
        _ => seed,
    }
}

pub fn rng_new(seed: u32) -> State { State { seed: rng_sanitize_seed(seed) } }

#[test]
fn rng_next_test() {
    let r = rng_new(rng_deterministic_seed());
    const EXPECTED = u64[20]:[
        0x002698ad4b48ead0, 0x1bfb1e0316f2d5de, 0x173a623c9725b477, 0x0a447a02823ad868,
        0x1df74948b3fbea7e, 0x1bc8b594bcf01a39, 0x07b767ca9520e99a, 0x05e28b4320bfd20e,
        0x0105906a24823f57, 0x1a1e7d14a6d24384, 0x2a7326df322e084d, 0x120bc9cc3fac4ec7,
        0x2c8f193a1b46a9c5, 0x2b9c95743bbe3f90, 0x0dcfc5b1d0398b46, 0x006ba47b3448bea3,
        0x3fe4fbf9a522891b, 0x23e1a50ad6aebca3, 0x1b263d39ea62be44, 0x13581d282e643b0e,
    ];
    for (i, r): (u8, State) in range(u8:0, u8:20) {
        let (new_r, got) = rng_next64(r);
        let want = EXPECTED[i];
        assert_eq(want, got);
        new_r
    }(r);
}
