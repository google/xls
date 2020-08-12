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

fn main(a: u32[2], i: u1) -> u32 {
  a[i]
}

test main {
  let x = u32:42;
  let y = u32:64;
  // Make an array with "bracket notation".
  let my_array: u32[2] = [x, y];
  let _ = assert_eq(main(my_array, u1:0), x);
  let _ = assert_eq(main(my_array, u1:1), y);
  ()
}
