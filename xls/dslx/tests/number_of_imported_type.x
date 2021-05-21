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

// Demonstrates that numbers of imported types can be used.

import xls.dslx.tests.number_of_imported_type_import as noiti

pub type other_type = bits[16];

fn foo(a: other_type) -> noiti::my_type {
  let x = a as noiti::my_type + noiti::my_type:8;
  x
}

#![test]
fn foo_test() {
  let _ = assert_eq(foo(other_type:8), noiti::my_type:16);
  ()
}
