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

struct Point3 {
  x: u32,
  y: u32,
  z: u32,
}

fn update_yz(p: Point3, new_y: u32, new_z: u32) -> Point3 {
  Point3 { y: new_y, z: new_z, ..p }
}

fn main() -> Point3 {
  let p = Point3 { x: u32:42, y: u32:0, z: u32:0 } in
  update_yz(p, u32:128, u32:256)
}

test main {
  let want = Point3 { x: u32:42, y: u32:128, z: u32:256 } in
  assert_eq(want, main())
}
