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

import std

fn umul_2(x: u4) -> u4 {
  std::umul(x, u4:2) as u4
}

fn umul_2_widening(x: u4) -> u6 {
  std::umul(x, u2:2)
}

fn umul_2_parametric<N: u32>(x: uN[N]) -> uN[N] {
  std::umul(x, uN[N]:2) as uN[N]
}

fn main() -> u4[8] {
  let x0 = u4[8]:[0, 1, 2, 3, 4, 5, 6, 7];
  let result_0 = map(x0, std::bounded_minus_1);
  let result_1 = map(x0, umul_2);
  let result_2 = map(x0, umul_2_parametric);
  let result_3 = map(x0, clz);
  map(map(map(x0, std::bounded_minus_1), umul_2), clz)
}

#![test]
fn maps() {
  let x0 = u4[8]:[0, 1, 2, 3, 4, 5, 6, 7];
  let expected = u4[8]:[0, 2, 4, 6, 8, 10, 12, 14];
  let expected_u6 = u6[8]:[0, 2, 4, 6, 8, 10, 12, 14];
  let _: () = assert_eq(expected, map(x0, umul_2));
  let _: () = assert_eq(expected_u6, map(x0, umul_2_widening));
  let _: () = assert_eq(expected, map(x0, umul_2_parametric));
  ()
}
