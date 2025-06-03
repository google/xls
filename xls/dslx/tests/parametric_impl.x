#![feature(type_inference_v2)]

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
struct Foo<N: u32> {}

impl Foo<N> {
    const N_PLUS_1 = N + u32:1;
}

fn foo<N: u32>() -> u32 { Foo<N>::N_PLUS_1 }

fn foo32() -> u32 { foo<u32:32>() }

fn main() -> u32 { foo32() }

#[test]
fn test_foo32() { assert_eq(foo32(), u32:33); }
