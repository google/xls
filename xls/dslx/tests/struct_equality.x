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

struct Point {
  x: u32,
  y: u32,
}

#[test]
fn test_shorthand_equality() {
  let x = u32:42;
  let y = u32:64;

  let p0 = Point { x, y };
  let p1 = Point { y, x };

  let _ = assert_eq(x, p0.x);
  let _ = assert_eq(y, p0.y);
  let _ = assert_eq(x, p1.x);
  let _ = assert_eq(y, p1.y);
  assert_eq(p0, p1)
}

#[test]
fn struct_equality() {
  let p0 = Point { x: u32:42, y: u32:64 };
  let p1 = Point { y: u32:64, x: u32:42 };
  assert_eq(p0, p1)
}
