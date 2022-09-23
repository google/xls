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

const U32_NEG_2 = u32:0xfffffffe;
const U32_NEG_3 = u32:0xfffffffd;

#[test]
fn signed_comparisons() {
  let _: () = assert_eq(true,  std::slt(u32:2, u32:3));
  let _: () = assert_eq(true,  std::sle(u32:2, u32:3));
  let _: () = assert_eq(false, std::sgt(u32:2, u32:3));
  let _: () = assert_eq(false, std::sge(u32:2, u32:3));

  // Mixed positive and negative numbers.
  let _: () = assert_eq(true,  std::slt(u32:2, u32:3));
  let _: () = assert_eq(true,  std::sle(u32:2, u32:3));
  let _: () = assert_eq(false, std::sgt(u32:2, u32:3));
  let _: () = assert_eq(false, std::sge(u32:2, u32:3));

  // Negative vs negative numbers.
  let _: () = assert_eq(false, std::slt(U32_NEG_2, U32_NEG_3));
  let _: () = assert_eq(true,  std::slt(U32_NEG_3, U32_NEG_2));
  let _: () = assert_eq(false, std::slt(U32_NEG_3, U32_NEG_3));

  let _: () = assert_eq(false, std::sle(U32_NEG_2, U32_NEG_3));
  let _: () = assert_eq(true,  std::sle(U32_NEG_3, U32_NEG_2));

  let _: () = assert_eq(true,  std::sgt(U32_NEG_2, U32_NEG_3));
  let _: () = assert_eq(false, std::sgt(U32_NEG_2, U32_NEG_2));
  let _: () = assert_eq(false, std::sgt(U32_NEG_3, U32_NEG_2));

  let _: () = assert_eq(false, std::sge(U32_NEG_3, U32_NEG_2));
  let _: () = assert_eq(true, std::sge(U32_NEG_2, U32_NEG_3));
  let _: () = assert_eq(true, std::sge(U32_NEG_3, U32_NEG_3));
  ()
}
