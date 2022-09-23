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

// Applies a function in an imported module on a constant in another imported
// module.

import std

import xls.dslx.tests.constexpr

const CONST_1_CLOG2 = std::clog2(constexpr::CONST_1);

fn main() -> u32 {
  CONST_1_CLOG2
}

#[test]
fn test_main() {
  assert_eq(u32:10, CONST_1_CLOG2)
}
