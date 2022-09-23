// Copyright 2020 The XLS Authors
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

import std

// Create some wrappers around the builtins so we validate that we can
// typecheck them (typechecking doesn't apply to test code at the moment).
fn umul_2(x: u2, y: u2) -> u4 {
  std::umul(x, y)
}
fn smul_3(x: s3, y: s3) -> s6 {
  std::smul(x, y)
}

fn main(x: u3, y: u3) -> s6 {
  (umul_2(x as u2, y as u2) as s6) + smul_3(x as s3, y as s3)
}

#[test]
fn multiplies_test() {
  let _ = assert_eq(u4:0b1001, umul_2(u2:0b11, u2:0b11));
  let _ = assert_eq(u4:0b0001, umul_2(u2:0b01, u2:0b01));
  let _ = assert_eq(s4:0b1111, std::smul(s2:0b11, s2:0b01));
  let _ = assert_eq(s6:6,  smul_3(s3:-3, s3:-2));
  let _ = assert_eq(s6:-6, smul_3(s3:-3, s3:2));
  let _ = assert_eq(s6:-6, smul_3(s3:3,  s3:-2));
  let _ = assert_eq(s6:6,  smul_3(s3:3,  s3:2));
  let _ = assert_eq(s6:1,  smul_3(s3:-1, s3:-1));
  ()
}
