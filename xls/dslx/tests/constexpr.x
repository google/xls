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
pub const CONST_1 = bits[32]:666;
pub const CONST_2 = CONST_1;
pub const CONST_3 = CONST_1 + CONST_2;
pub const CONST_4 = CONST_1 + CONST_2 + CONST_3;
pub const CONST_5 = CONST_1 + CONST_2 + CONST_3 + bits[32]:1;

pub const CONST_ARRAY1 = bits[32][4]:[u32:1, u32:2, u32:3, u32:4];
pub const CONST_ARRAY2 = bits[32][4]:[u32:4, u32:3, u32:2, u32:1];

pub const CONST_TUPLE1 = (u16:1, u32:2, u64:3);
pub const CONST_TUPLE2 = (u16:4, u32:5, u64:6);

pub fn main() -> u32 {
  CONST_1 + CONST_2 + CONST_3 + CONST_4 + CONST_5
}

#![test]
fn can_reference_constants_test() {
  let _ = assert_eq(bits[32]:666, CONST_1);
  let _ = assert_eq(bits[32]:666, CONST_2);
  let _ = assert_eq(bits[32]:1332, CONST_3);
  let _ = assert_eq(bits[32]:2664, CONST_4);
  let _ = assert_eq(CONST_4, CONST_4);
  let _ = assert_eq(bits[32]:2665, CONST_5);
  let _ = assert_eq(bits[32][4]:[u32:1, u32:2, u32:3, u32:4], CONST_ARRAY1);
  let _ = assert_eq(bits[32][4]:[u32:4, u32:3, u32:2, u32:1], CONST_ARRAY2);
  let _ = assert_eq((u16:1, u32:2, u64:3), CONST_TUPLE1);
  let _ = assert_eq((u16:4, u32:5, u64:6), CONST_TUPLE2);
  ()
}

#![test]
fn can_add_constants_test() {
  let _ = assert_eq(bits[32]:1332, CONST_1 + CONST_1);
  let _ = assert_eq(bits[32]:1332, CONST_1 + CONST_2);
  let _ = assert_eq(bits[32]:1332, CONST_2 + CONST_2);
  let _ = assert_eq(bits[32]:1998, CONST_1 + CONST_3);
  let _ = assert_eq(bits[32]:1998, CONST_2 + CONST_3);
  let _ = assert_eq(bits[32]:2664, CONST_1 + CONST_2 + CONST_3);
  ()
}

