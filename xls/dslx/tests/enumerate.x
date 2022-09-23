// Copyright 2022 The XLS Authors
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
fn enumerate_test() {
  let my_array = u32[4]:[u32:1, u32:2, u32:4, u32:8];
  let enumerated = enumerate(my_array);
  let _ = assert_eq(enumerated[u32:0], (u32:0, u32:1));
  let _ = assert_eq(enumerated[u32:1], (u32:1, u32:2));
  let _ = assert_eq(enumerated[u32:2], (u32:2, u32:4));
  let _ = assert_eq(enumerated[u32:3], (u32:3, u32:8));
  ()
}
