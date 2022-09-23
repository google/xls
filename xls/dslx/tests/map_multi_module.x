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

// Performs a map:
// * of an array defined as a constant in one module, with...
// * a parametric function defined in another module

import xls.dslx.tests.mod_imported_array
import xls.dslx.tests.mod_imported_lsb

fn f() -> u1[4] {
  map(mod_imported_array::A, mod_imported_lsb::lsb_u32)
}

fn g() -> u1[4] {
  map(mod_imported_array::A, mod_imported_lsb::lsb)
}

fn main() -> (u1[4], u1[4]) {
  (f(), g())
}

#[test]
fn main_test() {
  let (f, g) = main();
  let _ = assert_eq(u1[4]:[0, 1, 0, 1], f);
  assert_eq(u1[4]:[0, 1, 0, 1], g)
}
