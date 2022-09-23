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

// A module that imports a module with public module-level-constants and uses
// those in expressions to bind new constants.

import xls.dslx.tests.constexpr

const CONST_1 = constexpr::CONST_1;
const CONST_2 = constexpr::CONST_1 + constexpr::CONST_2;
const CONST_3 = CONST_2 + constexpr::CONST_2;

fn main() -> bits[32] {
  CONST_1 + CONST_2 + CONST_3
}

#[test]
fn can_reference_constants_test() {
  let _ = assert_eq(CONST_1, constexpr::CONST_1);
  let _ = assert_eq(bits[32]:666, constexpr::CONST_1);
  ()
}

#[test]
fn can_add_constants_test() {
  assert_eq(bits[32]:1332, constexpr::CONST_1 + CONST_1)
}
