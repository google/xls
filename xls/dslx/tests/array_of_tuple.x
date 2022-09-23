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

type Foo = (
  u8,
  u32,
);

#[test]
fn array_of_tuple_literal_test() {
  let xs = Foo[2]:[(u8:1, u32:2), (u8:3, u32:4)];
  let x0 = xs[0];
  let x1 = xs[1];
  let _ = assert_eq(x0.0, u8:1);
  let _ = assert_eq(x0.1, u32:2);
  let _ = assert_eq(x1.0, u8:3);
  let _ = assert_eq(x1.1, u32:4);
  ()
}

#[test]
fn array_of_tuple_type_annotation_test() {
  let xs: (u8, u32)[2] = [(u8:1, u32:2), (u8:3, u32:4)];
  let x0 = xs[u32:0];
  let x1 = xs[u32:1];
  let _ = assert_eq(x0.0, u8:1);
  let _ = assert_eq(x0.1, u32:2);
  let _ = assert_eq(x1.0, u8:3);
  let _ = assert_eq(x1.1, u32:4);
  ()
}

#[test]
fn multi_dimensional_array_of_tuple_type_annotation_test() {
  let xs: (u8, u32)[2][1] = [[(u8:1, u32:2), (u8:3, u32:4)]];
  let x0 = xs[u32:0][u32:0];
  let x1 = xs[u32:0][u32:1];
  let _ = assert_eq(x0.0, u8:1);
  let _ = assert_eq(x0.1, u32:2);
  let _ = assert_eq(x1.0, u8:3);
  let _ = assert_eq(x1.1, u32:4);
  ()
}
