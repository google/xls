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

#![test]
fn signed_comparisons() {
  let _: () = assert_eq(true,  slt(u32:2, u32:3));
  let _: () = assert_eq(true,  sle(u32:2, u32:3));
  let _: () = assert_eq(false, sgt(u32:2, u32:3));
  let _: () = assert_eq(false, sge(u32:2, u32:3));

  // Mixed positive and negative numbers.
  let _: () = assert_eq(true,  slt(u32:2, u32:3));
  let _: () = assert_eq(true,  sle(u32:2, u32:3));
  let _: () = assert_eq(false, sgt(u32:2, u32:3));
  let _: () = assert_eq(false, sge(u32:2, u32:3));

  // Negative vs negative numbers.
  let _: () = assert_eq(false, slt(u32:-2, u32:-3));
  let _: () = assert_eq(true,  slt(u32:-3, u32:-2));
  let _: () = assert_eq(false, slt(u32:-3, u32:-3));

  let _: () = assert_eq(false, sle(u32:-2, u32:-3));
  let _: () = assert_eq(true,  sle(u32:-3, u32:-2));

  let _: () = assert_eq(true,  sgt(u32:-2, u32:-3));
  let _: () = assert_eq(false, sgt(u32:-2, u32:-2));
  let _: () = assert_eq(false, sgt(u32:-3, u32:-2));

  let _: () = assert_eq(false, sge(u32:-3, u32:-2));
  let _: () = assert_eq(true, sge(u32:-2, u32:-3));
  let _: () = assert_eq(true, sge(u32:-3, u32:-3));
  ()
}
