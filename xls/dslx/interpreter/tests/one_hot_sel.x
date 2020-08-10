// Copyright 2020 Google LLC
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


test one_hot_sel_raw_nums {
  let _ = assert_eq(u4:0, one_hot_sel(u3:0b000, u4[3]:[u4:1, u4:2, u4:4])) in
  let _ = assert_eq(u4:1, one_hot_sel(u3:0b001, u4[3]:[u4:1, u4:2, u4:4])) in
  let _ = assert_eq(u4:2, one_hot_sel(u3:0b010, u4[3]:[u4:1, u4:2, u4:4])) in
  let _ = assert_eq(u4:4, one_hot_sel(u3:0b100, u4[3]:[u4:1, u4:2, u4:4])) in
  let _ = assert_eq(u4:7, one_hot_sel(u3:0b111, u4[3]:[u4:1, u4:2, u4:4])) in
  ()
}

test one_hot_sel_symbols {
  const A = u4:1 in
  const B = u4:2 in
  const C = u4:4 in
  let cases: u4[3] = [A, B, C] in
  let _ = assert_eq(u4:0, one_hot_sel(u3:0b000, cases)) in
  let _ = assert_eq(B, one_hot_sel(u3:0b010, cases)) in
  let _ = assert_eq(C, one_hot_sel(u3:0b100, cases)) in
  let _ = assert_eq(u4:3, A|B) in
  let _ = assert_eq(A|B|C, one_hot_sel(u3:0b111, cases)) in
  let _ = assert_eq(A|B, one_hot_sel(u3:0b011, cases)) in
  let _ = assert_eq(B|C, one_hot_sel(u3:0b110, cases)) in
  ()
}
