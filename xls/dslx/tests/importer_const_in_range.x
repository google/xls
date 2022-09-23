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

// Uses an externally-defined constant (via import) in a range expression as the
// upper limit.

import xls.dslx.tests.mod_simple_const

fn main(x: u32) -> u32 {
  for (i, accum): (u32, u32) in u32:0..mod_simple_const::FOUR {
    accum+i
  }(u32:0)
}

#[test]
fn main_test() {
  assert_eq(main(u32:0), u32:6)
}
