#![feature(type_inference_v2)]

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

// Function that implements a sparse work-efficient parallel prefix sum algorithm, similar to
// https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_2:_Work-efficient
// An adder chain would use (n-1) adders and add (n-1) adder delays.
// This implementation uses ~2n adders, but only adds (2 lg n - 2) adder delays.
fn sparse_prefix_sum(values: u16[ARRAY_SIZE]) -> u16[ARRAY_SIZE] {
    // Up-sweep
    let swept = for (j, c): (u32, u16[ARRAY_SIZE]) in u32:0..std::clog2(ARRAY_SIZE) {
        let step = u32:1 << (j + u32:1);
        let half_step = u32:1 << j;
        for (s, updated): (u32, u16[ARRAY_SIZE]) in u32:1..ARRAY_SIZE {
            let index = s * step - u32:1;
            if index < ARRAY_SIZE {
                update(updated, index, c[index] + c[index - half_step])
            } else {
                updated
            }
        }(c)
    }(values);
    // Down-sweep
    for (i, c): (u32, u16[ARRAY_SIZE]) in u32:0..std::clog2(ARRAY_SIZE) {
        let j = std::clog2(ARRAY_SIZE) - i;
        let step = u32:1 << j;
        let half_step = u32:1 << (j - u32:1);
        for (s, updated): (u32, u16[ARRAY_SIZE]) in u32:1..ARRAY_SIZE {
            let index = s * step + half_step - u32:1;
            if index < ARRAY_SIZE {
                update(updated, index, c[index] + c[index - half_step])
            } else {
                updated
            }
        }(c)
    }(swept)
}

#[test]
fn sparse_prefix_sum_test() {
    let result = sparse_prefix_sum(u16[16]:[u16:1, ...]);
    assert_eq(result, u16[16]:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

    let result = sparse_prefix_sum(u16[16]:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq(result, u16[16]:[1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136]);
}
