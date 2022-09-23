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

fn slice_to<N: u32>(x: u32) -> uN[N] {
  x[0 +: uN[N]]
}

fn main() -> (u1, u2, u3) {
  let low1: u1 = slice_to<u32:1>(u32:0b101);
  let low2: u2 = slice_to<u32:2>(u32:0b101);
  let low3: u3 = slice_to<u32:3>(u32:0b101);
  (low1, low2, low3)
}

#[test]
fn test_main() {
  let want = (u1:0b1, u2:0b01, u3:0b101);
  assert_eq(want, main())
}
