// Copyright 2020 Google LLC
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

fn id(x: Point) -> Point {
  x
}

fn main(x: Point) -> Point {
  id(x)
}

test id {
  let x = Point { x: u32:42, y: u32:64 };
  let y = main(x);
  assert_eq(x, y)
}
