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

fn add(x: u32, y: u32) -> u32 {
  x + y + u32:0  // Something to optimize.
}

#[test]
fn add_test() {
  assert_eq(add(u32:2, u32:3), u32:5)
}

#[test]
fn add_test() {
  assert_eq(add(u32:2, u32:3), u32:6)
}
