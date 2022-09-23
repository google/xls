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

fn main(xy: u32) -> Point {
  Point { x: xy, y: xy }
}

#[test]
fn f() {
  let p: Point = main(u32:42);
  let _ = assert_eq(u32:42, p.x);
  let _ = assert_eq(u32:42, p.y);
  ()
}

#[test]
fn alternative_forms_equal() {
  let p0 = Point { x: u32:42, y: u32:64 };
  let p1 = Point { y: u32:64, x: u32:42 };
  assert_eq(p0, p1)
}
