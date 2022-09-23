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

const BEST_Y = u32:42;

// Update the y member of the Point struct to BEST_Y.
fn main(p: Point) -> Point {
  Point{ y: BEST_Y, ..p }
}

#[test]
fn main_test() {
  let p = Point{ x: u32:1, y: u32:2 };
  let q = main(p);
  let _ = assert_eq(q.y, u32:42);
  let _ = assert_eq(q.x, u32:1);
  ()
}
