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

fn umul_2(x: u4) -> u4 { std::umul(x, u4:2) as u4 }

fn umul_2_widening(x: u4) -> u6 { std::umul(x, u2:2) }

fn umul_2_parametric<N: u32>(x: uN[N]) -> uN[N] { std::umul(x, uN[N]:2) as uN[N] }

fn main() -> u4[8] {
    let x0 = u4[8]:[0, 1, 2, 3, 4, 5, 6, 7];
    map(x0, std::bounded_minus_1);
    map(x0, umul_2);
    map(x0, umul_2_parametric);
    map(x0, clz);
    map(x0, ctz);
    map(x0, and_reduce);
    map(x0, or_reduce);
    map(x0, xor_reduce);
    map(x0, rev);
    map(x0, encode);
    map(map(map(x0, std::bounded_minus_1), umul_2), clz)
}

#[test]
fn maps() {
    let x0 = u4[8]:[0, 1, 2, 3, 4, 5, 6, 7];
    let expected = u4[8]:[0, 2, 4, 6, 8, 10, 12, 14];
    let expected_u6 = u6[8]:[0, 2, 4, 6, 8, 10, 12, 14];
    assert_eq(expected, map(x0, umul_2));
    assert_eq(expected_u6, map(x0, umul_2_widening));
    assert_eq(expected, map(x0, umul_2_parametric));
    assert_eq(u4[8]:[4, 4, 2, 1, 1, 0, 0, 0], main());
}

#[test]
fn map_unary_builtins() {
    let x = u3[8]:[0, 1, 2, 3, 4, 5, 6, 7];
    assert_eq(u3[8]:[3, 2, 1, 1, 0, 0, 0, 0], map(x, clz));
    assert_eq(u3[8]:[3, 0, 1, 0, 2, 0, 1, 0], map(x, ctz));
    assert_eq(u1[8]:[0, 0, 0, 0, 0, 0, 0, 1], map(x, and_reduce));
    assert_eq(u1[8]:[0, 1, 1, 1, 1, 1, 1, 1], map(x, or_reduce));
    assert_eq(u1[8]:[0, 1, 1, 0, 1, 0, 0, 1], map(x, xor_reduce));
    assert_eq(u3[8]:[0, 4, 2, 6, 1, 5, 3, 7], map(x, rev));
    assert_eq(u2[8]:[0, 0, 1, 1, 2, 2, 3, 3], map(x, encode));
}
