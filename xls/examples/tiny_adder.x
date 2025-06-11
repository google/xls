#![feature(type_inference_v2)]

// Copyright 2022 The XLS Authors
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

// Returns the add result, widened to reflect the carry bit.
fn tiny_adder(x: u1, y: u1) -> u2 {
  let lhs = x as u2;
  let rhs = y as u2;
  lhs + rhs
}

#[test]
fn test_exhaustive() {
  assert_eq(tiny_adder(u1:0, u1:0), u2:0b00);
  assert_eq(tiny_adder(u1:0, u1:1), u2:0b01);
  assert_eq(tiny_adder(u1:1, u1:1), u2:0b10);
  assert_eq(tiny_adder(u1:1, u1:0), u2:0b01);
}
