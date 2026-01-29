// Copyright 2026 The XLS Authors
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

const CONST_1 = u32:666;
const CONST_2 = u32:256;
const CONST_3 = u32:3;

const fn const_get() -> u32 { CONST_3 }
const fn const_adder() -> u32 { CONST_1 + CONST_2 }
const fn const_param_adder<A:u32, B:u32>() -> u32 { A + B }
const fn const_mix_mul<N:u32>() -> u32 { N * CONST_3 }

fn main() -> u32 {
    let first = const_adder();
    let second = const_param_adder<CONST_1, CONST_2>();
    let third = const_mix_mul<CONST_2>();
    let array: u32[const_get()] = [first, second, third];
    for (i, accum) in u32:0..const_get() {
        accum + array[i]
    }(u32:0)
}

#[test]
fn can_add_const() {
    assert_eq(const_adder(), u32:922);
}

#[test]
fn can_add_params() {
    assert_eq(const_param_adder<CONST_1, CONST_2>(), u32:922);
}

#[test]
fn can_mul_mix() {
    assert_eq(const_mix_mul<CONST_1>(), u32:1998);
}
