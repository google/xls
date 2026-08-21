#![feature(use_syntax)]

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

use xls::dslx::tests::mod_use_members_lib::{C, E, S, id};

fn main(x: u32) -> u32 {
    let struct_literal = S { x: 1 }.x;
    let enum_member = E::B as u32;
    let mapped = map(u32[2]:[2, 3], id)[0];
    let matched = match x {
        C => u32:4,
        _ => u32:0,
    };
    struct_literal + enum_member + mapped + matched
}

#[test]
fn test_main() {
    assert_eq(main(C), u32:8);
    assert_eq(main(u32:0), u32:4);
}
