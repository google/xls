// Copyright 2021 The XLS Authors
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

// Imports a struct via a user-defined struct via a type alias and uses it.

import xls.dslx.tests.mod_struct_point

type Point = mod_struct_point::Point;

fn main() -> Point {
  Point{x: u32:42, y: u32:64}
}

#[test]
fn main_test() {
  let p: Point = main();
  let _ = assert_eq(p.x, u32:42);
  let _ = assert_eq(p.y, u32:64);
  ()
}
