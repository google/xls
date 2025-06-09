// Copyright 2021 The XLS Authors
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

// Example given for regression in https://github.com/google/xls/issues/305

fn clog2<N: u32>(x: bits[N]) -> bits[N] {
    if x >= bits[N]:1 { (N as bits[N]) - clz(x - bits[N]:1) } else { bits[N]:0 }
}

fn func(a: u32) -> u32 { clog2<u32:32>(a) }

fn dot_product<BITCOUNT: u32, LENGTH: u32, IDX_BITS: u32 = {func(LENGTH + u32:1)}>
    (a: bits[BITCOUNT][LENGTH], b: bits[BITCOUNT][LENGTH]) -> bits[BITCOUNT] {
    for (idx, acc): (bits[IDX_BITS], bits[BITCOUNT]) in bits[IDX_BITS]:0..(LENGTH as bits[IDX_BITS]) {
        let partial_product = a[idx] * b[idx];
        acc + partial_product
    }(u32:0)
}

fn main(a: u32[4], b: u32[4]) -> u32 { dot_product(a, b) }

#[test]
fn test_main() { assert_eq(u32:20, main(u32[4]:[0, 1, 2, 3], u32[4]:[1, 2, 3, 4])) }
