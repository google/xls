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

fn main(input: u32) -> u32 {
  const FOO = u32:2;
  for(idx, acc): (u32, u32) in range(u32:0, FOO) {
    for(idx, acc): (u32, u32) in range(u32:0, FOO) {
      for(idx, acc): (u32, u32) in range(u32:0, FOO) {
        acc + input
      }(acc)
    }(acc)
  }(u32:0)
}

#![test]
fn local_test() {
  assert_eq(u32:8, main(u32:1))
}
