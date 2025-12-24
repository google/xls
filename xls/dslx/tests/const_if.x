// Copyright 2025 The XLS Authors
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

#![feature(type_inference_v2)]

fn multiply(a: u32, b: u32) -> u32 { a * b }

fn const_if_true(a: u32) -> u32 { const if true { a + u32:2 } else { multiply(a, u32:10) } }

#[test]
fn const_if_true_test() {
    assert_eq(const_if_true(u32:0), u32:2);
    assert_eq(const_if_true(u32:2), u32:4);
}

fn const_if_false(a: u32) -> u32 { const if false { a + u32:2 } else { multiply(a, u32:10) } }

#[test]
fn const_if_false_test() {
    assert_eq(const_if_false(u32:0), u32:0);
    assert_eq(const_if_false(u32:2), u32:20);
}

fn const_if_const(a: u32) -> u32 {
    const A = u32:2;
    const B = u32:1;
    const if (A + B) == u32:3 { a + u32:2 } else { u32:0 }
}

#[test]
fn const_if_const_test() { assert_eq(const_if_const(u32:0), u32:2); }

fn const_if_disparate_types<A: u32>() -> u32 {
    let data = const if A == u32:0 {
        u16:600
    } else if A == u32:1 {
        u32:70000
    } else {
        u8:4
    };

    data as u32
}

#[test]
fn const_if_disparate_types_test() {
    assert_eq(const_if_disparate_types<u32:0>(), u32:600);
    assert_eq(const_if_disparate_types<u32:1>(), u32:70000);
    assert_eq(const_if_disparate_types<u32:2>(), u32:4);
}

fn const_if_disparate_types_parameter<N: u32>() -> uN[N] {
    const if N == 1 { u1:1 } else { zero!<uN[N]>() }
}

#[test]
fn const_if_disparate_types_parameter_test() {
    assert_eq(const_if_disparate_types_parameter<u32:1>(), u1:1);
    assert_eq(const_if_disparate_types_parameter<u32:10>(), u10:0);
    assert_eq(const_if_disparate_types_parameter<u32:23>(), u23:0);
}

fn main() -> u32 { const_if_disparate_types<u32:0>() }
