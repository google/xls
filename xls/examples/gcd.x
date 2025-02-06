// Copyright 2023 The XLS Authors
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

// https://en.wikipedia.org/wiki/Greatest_common_divisor#Euclidean_algorithm
fn gcd_euclidean<N: u32, DN: u32 = {N * u32:2}>(a: uN[N], b: uN[N]) -> uN[N] {
    let (gcd, _) = for (i, (a, b)) in range(u32:0, DN) {
        if (b == uN[N]:0) {
            (a, b)
        } else {
            (b, std::iterative_div_mod(a, b).1)
        }
    }((a, b));
    gcd
}

fn gcd_binary_match<N: u32>(a: uN[N], b: uN[N], d: uN[N]) -> (uN[N], uN[N], uN[N]) {
    match (a[0:1], b[0:1]) {
      (u1:0, u1:1) => (b, a >> 1, d),
      (u1:1, u1:0) => (a, b >> 1, d),
      (u1:0, u1:0) => (a >> 1, b >> 1, d+uN[N]:1),
      (u1:1, u1:1) => ((a - b) >> 1, b, d),
    }
}

// https://en.wikipedia.org/wiki/Greatest_common_divisor#Binary_GCD_algorithm
fn gcd_binary<N: u32, DN: u32 = {N * u32:2}>(a: uN[N], b: uN[N]) -> uN[N] {
    let (a, _, d) = for (i, (a, b, d)) in range(u32:0, DN) {
        if (a == b) {
            (a, b, d)
        } else if (a < b) {
            gcd_binary_match(b, a, d)
        } else {
            gcd_binary_match(a, b, d)
        }
    }((a, b, uN[N]:0));
    (a << d)
}

#[test]
fn gcd_euclidean_test() {
    assert_eq(u8:6, gcd_euclidean(u8:48, u8:18));
    assert_eq(u8:6, gcd_euclidean(u8:18, u8:48));
}

#[test]
fn gcd_binary_test() {
    assert_eq(u8:6, gcd_binary(u8:48, u8:18));
    assert_eq(u8:6, gcd_binary(u8:18, u8:48));
}

#[quickcheck(test_count=50000)]
fn prop_gcd_equal(a: u32, b: u32) -> bool {
    gcd_euclidean<u32:32>(a, b) == gcd_binary<u32:32>(a, b)
}
