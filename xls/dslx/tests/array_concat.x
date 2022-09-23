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

fn main() -> s32[4] {
  s32[2]:[1, 2] ++ s32[2]:[3, 4]
}

#[test]
fn test_concat() {
  assert_eq(s32[4]:[1, 2, 3, 4], main())
}

#[quickcheck]
fn prop_associative(xs: s32[2], ys: s32[2], zs: s32[2]) -> bool {
  ((xs ++ ys) ++ zs) == (xs ++ (ys ++ zs))
}

#[quickcheck]
fn prop_non_commutative(xs: s32[4], ys: s32[4]) -> bool {
  xs == ys || (xs ++ ys) != (ys ++ xs)
}


