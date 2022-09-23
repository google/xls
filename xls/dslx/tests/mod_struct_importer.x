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

import xls.dslx.tests.mod_imported

fn fully_qualified(x: u32) -> mod_imported::Point {
  mod_imported::Point { x: x, y: u32:64 }
}

// A version that goes through a type alias.

type PointAlias = mod_imported::Point;

fn main(x: u32) -> PointAlias {
  PointAlias { x: x, y: u32:64 }
}

#[test]
fn main_test() {
  let p: PointAlias = fully_qualified(u32:42);
  let _ = assert_eq(u32:42, p.x);
  let _ = assert_eq(u32:64, p.y);
  let _ = assert_eq(main(u32:128), fully_qualified(u32:128));
  ()
}
