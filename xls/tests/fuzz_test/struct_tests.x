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

struct Point { x: u32, y: u8 }

#[fuzz_test]
fn arbitrary_struct(p: Point) -> bool {
    p.x == p.x && p.y == p.y
}

#[fuzz_test(domains=`Point { x: u32:0..10, y: u8:11..20 }`)]
fn struct_range(p: Point) -> bool {
    (p.x as u8) < p.y
}

#[fuzz_test(domains=`Point { x: [u32:0, 1,2,3], y: u8:11..20 }`)]
fn struct_element_of(p: Point) -> bool {
    (p.x as u8) < p.y
}

#[fuzz_test(domains=`Point { x: [u32:0, 1,2,3]}`)]
fn struct_arbitrary_field(p: Point) -> bool {
    p.x <= u32:3
}

struct WideStruct { w: uN[128], x: u32 }

#[fuzz_test(domains=`WideStruct { x: u32:0..10 }`)]
fn struct_with_wide_bits(s: WideStruct) -> bool {
    true
}
