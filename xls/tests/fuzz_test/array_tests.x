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

#[fuzz_test(domains=`()`)]
fn arbitrary_array(x: u32[3]) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn array_of_tuples(x: (u32, u32)[2]) -> bool {
    true
}

#[fuzz_test(domains=`((), u32:0..9)`)]
fn tuple_with_array(x: (u32[2], u32)) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn big_array(x: uN[128][3]) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn tuple_with_big_array(x: (uN[128][2], u32)) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn nested_big_array(x: uN[128][2][3]) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn array_of_tuples_with_wide_bits(x: (uN[128], u32)[2]) -> bool {
    true
}

#[fuzz_test(domains=`(u32:0..10, u32:1..=11)`)]
fn fuzz_custom_array_domain(t: u32[2]) -> bool {
    assert!(t[0] >= u32:0, "t0_oob_low");
    assert!(t[0] < u32:10, "t0_oob");
    assert!(t[1] >= u32:1, "t1_oob_low");
    assert!(t[1] <= u32:11, "t1_oob");
    true
}

enum FuzzEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
}

#[fuzz_test(domains=`([FuzzEnum::A],)`)]
fn fuzz_enum_array_domain(t: FuzzEnum[2]) -> bool {
    // First one must be "A", the second can be anything.
    assert!(t[0] == FuzzEnum::A, "t0_not_A");
    match t[1] as u2 {
        u2:0 => true,
        u2:1 => true,
        u2:2 => true,
        _ => false,
    }
}
