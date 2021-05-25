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

import xls.build_rules.tests.simple_example_four as four

fn five() -> u32 {
  u32: 5
}

fn main(offset: u32) -> u32 {
  five() + four::result() + offset
}

#![test]
fn test_main_value() {
  let _ = assert_eq(main(u32: 0), u32:15);
  let _ = assert_eq(main(u32: 1), u32:16);
  ()
}