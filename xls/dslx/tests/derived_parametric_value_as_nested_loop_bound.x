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

fn derived_parametric_main<FOO: u32, BAR: u32, DERIVED:u32 = {FOO + BAR}>(input: u32) -> u32 {
  for(idx, acc): (u32, u32) in u32:0..DERIVED {
    for(idx, acc): (u32, u32) in u32:0..DERIVED {
      for(idx, acc): (u32, u32) in u32:0..DERIVED {
        acc + input
      }(acc)
    }(acc)
  }(u32:0)
}

fn main(arg: u32) -> u32 {
  derived_parametric_main<u32:1, u32:1>(u32:1)
}

#[test]
fn main_test() {
  assert_eq(u32:8, main(u32:1))
}
