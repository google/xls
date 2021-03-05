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

pub const IDX = u32:5;

fn p<N: u32>(a: u32) -> uN[N] {
  a[:N]
}

fn p1<N: u32, M: u32 = N+u32:1>(a: u32) -> uN[M] {
  // TODO(leary): 2021-03-05 This should flag a typechecking error for a bare
  // literal.
  //a[:N+1]
  a[:N+u32:1]
}

fn main() -> (u8, u16, u8, u4, u17) {
  let a = u32:0xdeadbeef;
  let x = u32:8;
  let i: u8 = a[0:8];
  let j: u16 = p<u32:16>(a);
  let k: u8 = a[u32:1 + u32:8 : 17];
  let l: u4 = a[1:IDX];
  let m: u17 = p1<u32:16>(a);
  (i, j, k, l, m)
}

#![test]
fn test_non_constexpr_slice() {
  let t = main();
  assert_eq(t, (u8:239, u16:48879, u8:223, u4:7, u17:114415))
}
