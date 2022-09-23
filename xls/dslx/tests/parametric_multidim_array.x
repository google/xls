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

type TypeX = u6;
type TypeY = u3;

// Identity function for a two-dimensional array of arbitrary bit count.
fn id<L: u32, M: u32, N: u32>(x: uN[L][M][N]) -> uN[L][M][N] {
  x
}

fn id_6_3_2(x: u6[3][2]) -> u6[3][2] {
  id(x)
}

#[test]
fn different_invocations() {
  let a : u6[3][2] = [[TypeX:1, TypeX:2, TypeX:3],
                      [TypeX:4, TypeX:5, TypeX:6]];
  let b : TypeX[3][2] = [[TypeX:1, TypeX:2, TypeX:3],
                         [TypeX:4, TypeX:5, TypeX:6]];
  // Use the parametric identity function on both sides, b has an enum that
  // describes the bitwidth but that's fine.
  let _ = assert_eq(id(a), id(b));
  // Mix the parametric identity function and an explicit instantiated wrapper
  // one.
  let _ = assert_eq(id(a), id_6_3_2(b));
  // Create a different shape to instantiate the parametric multidimensional
  // identity function with, should be no problem with that.
  let x: u3[2][3] = [[u3:1, u3:2], [u3:3, u3:4], [u3:5, u3:6]];
  let y: TypeY[2][3] = [[u3:1, u3:2], [u3:3, u3:4], [u3:5, u3:6]];
  assert_eq(id(x), id(y))
}
