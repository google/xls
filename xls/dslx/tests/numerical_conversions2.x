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

#[test]
fn narrow_signed_cast() {
  let negative_seven = s4:0b1001;
  assert_eq(negative_seven as s2, s2:1)
}

#[test]
fn widen_signed_cast() {
  let negative_one = s2:0b11;
  let _ = assert_eq(negative_one, s2:-1);
  assert_eq(negative_one as s4, s4:-1)
}

#[test]
fn numerical_conversions() {
  let s8_m2 = s8:-2;
  // Sign extension (source type is signed).
  let _ = assert_eq(s32:-2, s8_m2 as s32);
  let _ = assert_eq(s16:-2, s8_m2 as s16);
  ()
}
