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

// Note that the size is all-bits-set, the DSL should interpret it as an
// unsigned value.
const SIZE_3 = u2:0b11;
const SIZE_2 = u2:0b10;

fn main() -> (u32[SIZE_3], u32[SIZE_2]) {
  (u32[SIZE_3]:[0, ...], u32[SIZE_2]:[0, ...])
}

#![test]
fn test_main() {
  assert_eq(main(), (u32[3]:[0, 0, 0], u32[2]:[0, 0]))
}
