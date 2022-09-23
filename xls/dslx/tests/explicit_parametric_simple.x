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

fn add_one<E:u32, F:u32, G:u32 = E+F>(lhs: bits[E]) -> bits[G] {
  (lhs as uN[F] + uN[F]:1) as uN[G]
}

#[test]
fn generic() {
  let actual: u3 = add_one<u32:1, {u32:1 + u32:1}>(u1:1);
  assert_eq(u3:2, actual)
}
