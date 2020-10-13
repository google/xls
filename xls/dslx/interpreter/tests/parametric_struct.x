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

fn double(x: u32) -> u32 { x * u32:2 }

struct [A: u32, B: u32 = double(A)] ParametricPoint {
  x: bits[A],
  y: bits[B]
}


struct [P: u32, Q: u32] WrapperStruct {
  pp: ParametricPoint[P, Q]
}

fn [N: u32, M: u32]
make_point(x: bits[N], y: bits[M]) -> ParametricPoint[N, M] {
  ParametricPoint { x, y }
}

fn [N: u32, M: u32]
from_point(p: ParametricPoint[N, M]) -> ParametricPoint[N, M] {
  ParametricPoint { x: p.x, y: p.y }
}

fn [N: u32, M: u32]
add_points(lhs: ParametricPoint[N, M], rhs: ParametricPoint[N, M])
    -> ParametricPoint[N, M] {
    make_point(lhs.x + rhs.x, lhs.y + rhs.y)
}

fn [N: u32, M: u32, L: u32]
accum_points(points: ParametricPoint[N, M][L]) -> ParametricPoint[N, M] {
    let start = make_point(bits[N]: 0, bits[M]: 0);
    for (idx, accum): (u32, ParametricPoint[N, M]) in range(u32: 0, L) {
        add_points(accum, points[idx])
    } (start)
}

// Direct instantiation via typedefs works too!
type PP32 = ParametricPoint[32, 64];

fn main(s: PP32) -> u32 {
  s.x
}

#![test]
fn test_funcs() {
  // make_point is type-parametric.
  let p = make_point(u2:1, u4:1);
  // Type annotations can be directly instantiated struct types.
  let q: ParametricPoint[5, 10] = make_point(u5:1, u10:1);
  let r = make_point(u32:1, u64:1);

  // main() will only accept ParametricPoints with u32s.
  let _ = assert_eq(u32:1, main(r));

  // from_point() is also type-parametric.
  let s = from_point(p);
  let t = from_point(q);

  // WrapperStruct wraps an instance of ParametricPoint.
  let s_wrapper = WrapperStruct { pp: r };
  let _ = assert_eq(r, s_wrapper.pp);

  let sum = accum_points([
      make_point(u8: 3, u16: 4),
      make_point(u8: 5, u16: 6),
      make_point(u8: 7, u16: 8),
  ]);
  let _ = assert_eq(sum, make_point(u8: 15, u16: 18));

  ()
}
