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

import xls.dslx.tests.mod_simple_const

const LOCAL_FOUR = u32:4;

fn main(x: u32) -> u32 {
  x + mod_simple_const::FOUR
}

#[test]
fn main_test() {
  let _ = assert_eq(main(u32:0), u32:4);
  let _ = assert_eq(main(u32:1), u32:5);
  let _ = assert_eq(LOCAL_FOUR, mod_simple_const::FOUR);
  ()
}
