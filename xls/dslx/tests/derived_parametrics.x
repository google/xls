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

fn double(n: u32) -> u32 {
  n * u32:2
}

fn self_append<A: u32, B: u32 = double(A)>(x: uN[A]) -> uN[B] {
  x++x
}

fn main() -> (u10, u20) {
  let x1 = self_append(u5:5);
  let x2 = self_append(u10:10);
  (x1, x2)
}

#[test]
fn derived_parametric_functions() {
  let arr = map([u2:1, u2:2], self_append);
  let _ = assert_eq(u4:5, arr[u32:0]);
  let _ = assert_eq(u4:10, arr[u32:1]);

  let _ = assert_eq(u4:5, self_append(u2:1));
  let _ = assert_eq(u6:18, self_append(u3:2));
  ()
}
