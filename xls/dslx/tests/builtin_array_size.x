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

fn make_u32_array<N: u32>() -> u32[N] { zero!<u32[N]>() }

fn u32_array_size<N: u32>() -> u32 { array_size(make_u32_array<N>()) }

#[test]
fn test_make_u32_array() {
    assert_eq(u32_array_size<u32:1>(), u32:1);
    assert_eq(u32_array_size<u32:2>(), u32:2);
    assert_eq(u32_array_size<u32:3>(), u32:3);
    assert_eq(u32_array_size<u32:4>(), u32:4);
}

fn make_s32_array<N: u32>() -> s32[N] { zero!<s32[N]>() }

fn s32_array_size<N: u32>() -> u32 { array_size(make_s32_array<N>()) }

#[test]
fn test_make_s32_array() {
    assert_eq(s32_array_size<u32:1>(), u32:1);
    assert_eq(s32_array_size<u32:2>(), u32:2);
    assert_eq(s32_array_size<u32:3>(), u32:3);
    assert_eq(s32_array_size<u32:4>(), u32:4);
}

fn tuple_array_size<N: u32>(x: (u32, u64)[N]) -> u32 { array_size(x) }

#[test]
fn test_tuple_array_size() {
    let x1 = [(u32:32, u64:64)];
    assert_eq(tuple_array_size(x1), u32:1);
    let x2 = [(u32:32, u64:64), (u32:64, u64:128)];
    assert_eq(tuple_array_size(x2), u32:2);
}
