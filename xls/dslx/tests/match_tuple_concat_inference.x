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
//
// See also: https://github.com/google/xls/issues/2379

import std;

pub fn main(x: u3) -> u2 {
    let (_dummy, upper): (u8, u2) = match x {
        u3:0b000 | u3:0b001 => (u8:0, u2:0b00),
        u3:0b010 | u3:0b011 => (u8:0, std::lsb(x) ++ u1:0),
        _ => (u8:0, x[0+:u2]),
    };
    upper
}

#[test]
fn test_main() {
    // Simple spot checks.
    assert_eq(u2:0b00, main(u3:0b000));
    assert_eq(u2:0b00, main(u3:0b001));
    assert_eq(u2:0b00, main(u3:0b010));
    assert_eq(u2:0b10, main(u3:0b011));
    assert_eq(u2:0b01, main(u3:0b101));
}
