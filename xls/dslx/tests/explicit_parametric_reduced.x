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

struct Generic<X:u32, Y:u32> {
  a: bits[X],
  b: bits[Y]
}

fn zero<C:u32, D:u32>() -> Generic<C, D> {
  Generic<C, D>{ a: bits[C]:0, b: bits[D]:0 }
}

fn indirection<E:u32, F:u32>() -> Generic<E, F> {
  zero<E, F>()
}

#[test]
fn test_generic() {
  let got = indirection<u32:1, u32:2>();
  let want = Generic<u32:1, u32:2>{ a: u1:0, b: u2:0 };
  assert_eq(want, got)
}
