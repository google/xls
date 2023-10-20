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

// Simple single alias at the start.
fn f() -> u32 {
    type MyU32 = u32;
    MyU32:42
}

// Two aliases at the start.
fn g() -> u32 {
    type MyU8 = u8;
    type MyU32 = u32;
    MyU8:64 as MyU32
}

// Alias in a block expr.
fn h() -> u32 {
    let x = {
        type MyType = u32;
        MyType:42
    };
    let y = {
        type MyType = u8;
        MyType:64 as u32
    };
    x + y
}

fn main() -> u32 { f() + g() + h() }

#[test]
fn test_f() { assert_eq(f(), u32:42) }

#[test]
fn test_g() { assert_eq(g(), u32:64) }

#[test]
fn test_h() { assert_eq(h(), u32:106) }

#[test]
fn test_main() { assert_eq(main(), u32:212) }
