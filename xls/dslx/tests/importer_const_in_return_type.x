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

// Uses an externally-defined constant (via import) in a return type, both by
// external reference and re-binding locally.
//
// Regression test for example given in https://github.com/google/xls/issues/259

import xls.dslx.tests.mod_simple_const

const LOCAL_FOUR = mod_simple_const::FOUR;

fn f() -> uN[mod_simple_const::FOUR] {
  uN[mod_simple_const::FOUR]:0b1111
}

fn g() -> uN[LOCAL_FOUR] {
  uN[LOCAL_FOUR]:0b1001
}

fn main() -> u4 {
  f() ^ g()
}

#[test]
fn test_main() {
  assert_eq(main(), u4:0b0110)
}
