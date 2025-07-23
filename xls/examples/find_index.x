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

// Instantiates a simple piece of logic for finding a target value using the
// find_index standard library function. This will be used as a synthesis
// flow example to determine how well the gate-level mapping is performed.

#![feature(type_inference_v2)]

import std;

fn find_index(x: u4[4], target: u4) -> (bool, u2) {
  let (found, index) = std::find_index(x, target);
  (found, index as u2)
}

#[test]
fn test_sample_values() {
  assert_eq(find_index(u4[4]:[1, 2, 3, 4], u4:1), (true, u2:0));
  assert_eq(find_index(u4[4]:[1, 2, 3, 4], u4:3), (true, u2:2));
  assert_eq(find_index(u4[4]:[1, 2, 3, 4], u4:5), (false, u2:0));
}
