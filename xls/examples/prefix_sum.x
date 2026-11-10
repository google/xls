// Copyright 2022 The XLS Authors
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

const ARRAY_SIZE = u32:16;

// Function that implements the Hillis-Steele parallel prefix sum algorithm.
// https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel
// An adder chain would use (n-1) adders and add (n-1) adder delays.
// This implementation uses ~(n lg n) adders, but only adds (lg n) adder delays.
fn prefix_sum(values: u16[ARRAY_SIZE]) -> u16[ARRAY_SIZE] {
    for (i, c): (u32, u16[ARRAY_SIZE]) in u32:0..std::flog2(ARRAY_SIZE) {
        let lookback = u32:1 << i;
        for (j, updated): (u32, u16[ARRAY_SIZE]) in u32:0..ARRAY_SIZE {
            if j >= lookback { update(updated, j, c[j] + c[j - lookback]) } else { updated }
        }(c)
    }(values)
}

#[test]
fn prefix_sum_test() {
    let result = prefix_sum(u16[16]:[u16:1, ...]);
    assert_eq(result, u16[16]:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

    let result = prefix_sum(u16[16]:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq(result, u16[16]:[1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136]);
}
