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

// Regression sample given in https://github.com/google/xls/issues/207

const SIZE_BITS = s32:1000;
const SIZE_BITS_U32 = SIZE_BITS as u32;
type LotsOfBits = uN[SIZE_BITS_U32];

fn main(x: u32, lotsabits: LotsOfBits) -> LotsOfBits {
  lotsabits[:10] ++ x ++ lotsabits[42:SIZE_BITS]
}

#![test]
fn test_main() {
  let want = LotsOfBits:0xdeadbeef << ((SIZE_BITS-s32:42) as LotsOfBits);
  assert_eq(want, main(u32:0xdeadbeef, LotsOfBits:0))
}
