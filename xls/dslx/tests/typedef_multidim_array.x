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

// Demonstrates use of a typedef as the element type in a multidimensional
// array.

type TypeX = u6;

// Identity function.
fn main(x: u6[3][2]) -> u6[3][2] {
  x
}

#[test]
fn main_test() {
  let a : u6[3][2] = [[TypeX:1, TypeX:2, TypeX:3],
                      [TypeX:4, TypeX:5, TypeX:6]];
  let b : TypeX[3][2] = [[TypeX:1, TypeX:2, TypeX:3],
                         [TypeX:4, TypeX:5, TypeX:6]];
  assert_eq(main(a), main(b))
}
